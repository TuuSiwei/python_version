"""Download -> slice -> Qwen ASR -> aggregate -> parquet/json -> HF upload (Rust parity)."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from asr_manager import AsrService, TranscribeResponse
from io_utils import link_or_copy_dir_recursive, link_or_copy_file
from progress_store import (
    GROUP_PROGRESS_FILE,
    GroupProgress,
    GroupState,
    load_group_progress,
    save_group_progress,
)
from upload_to_hf import upload_folder

CONVERSATIONS_JSON = "conversations.json"
FINAL_DIR = "final"
FINAL_REPO_DIR = "final/repo"
UPLOAD_QUEUE_DIR = "final/upload_queue"
CACHE_DIR = "cache"
S3_BASE = "https://dl.fbaipublicfiles.com/seamless_interaction"
DOWNLOAD_CONCURRENCY = 6
TRANSCRIPTION_ANOMALIES_FILE = "final/transcription_anomalies.jsonl"
INVALID_WAVS_FILE = "final/invalid_wavs.json"


@dataclass
class AudioConfig:
    sample_rate: int
    bits_per_sample: int


@dataclass
class FileRecord:
    file_id: str
    label: str
    split: str
    conversation_id: str
    participant_id_idx: int


@dataclass
class Word:
    word: str
    start_time: float
    end_time: float


@dataclass
class ConversationUtterance:
    spk: int
    words: list[Word]

    def to_json(self) -> dict[str, Any]:
        return {"spk": self.spk, "words": [w.__dict__ for w in self.words]}


@dataclass
class Conversation:
    conversation_id: str
    utterances: list[ConversationUtterance]
    audio_path: str

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "utterances": [u.to_json() for u in self.utterances],
            "audio_path": self.audio_path,
        }


@dataclass
class AudioSegment:
    segment_path: Path
    offset_seconds: float
    file_id: str
    participant_id_idx: int


@dataclass
class UploadRequest:
    group_idx: int
    conversation_ids: list[str]
    snapshot_dir: Path


@dataclass
class UploadJobResult:
    group_idx: int
    conversation_ids: list[str]
    snapshot_dir: Path
    error: str | None


@dataclass
class PreparedBatch:
    batch_idx: int
    batch_len: int
    start_time: float
    downloaded_sessions: list[tuple[str, list[FileRecord], list[tuple[Path, Path]]]]
    all_segments: dict[str, list[AudioSegment]]
    original_wav_paths: dict[str, list[Path]]


@dataclass
class ConversationTranscriptionStats:
    conversation_id: str
    total_words: int
    anomalous_words: int
    anomaly_ratio: float


@dataclass
class BatchProgress:
    group_idx: int
    completed_count: int
    total_conversations: int
    timestamp: str


@dataclass
class BatchResultEntry:
    conversation: Conversation
    stats: ConversationTranscriptionStats


def ensure_dirs() -> None:
    Path(FINAL_DIR).mkdir(parents=True, exist_ok=True)
    Path(FINAL_REPO_DIR).mkdir(parents=True, exist_ok=True)
    Path(UPLOAD_QUEUE_DIR).mkdir(parents=True, exist_ok=True)
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)


def group_snapshot_dir(group_idx: int) -> Path:
    return Path(UPLOAD_QUEUE_DIR) / f"group_{group_idx}"


def ensure_group_dirs(group_idx: int) -> None:
    Path(FINAL_REPO_DIR, f"audio_{group_idx}").mkdir(parents=True, exist_ok=True)


def clean_group(group_idx: int, conversation_ids: list[str]) -> None:
    print(f"\n=== 清理 Group {group_idx} 的本地数据 ===")
    for conv_id in conversation_ids:
        cache_dir = Path(CACHE_DIR) / conv_id
        if cache_dir.exists():
            import shutil

            shutil.rmtree(cache_dir)
            print(f"✓ 已清理: {cache_dir}")
    audio_dir = Path(FINAL_REPO_DIR) / f"audio_{group_idx}"
    if audio_dir.exists():
        import shutil

        shutil.rmtree(audio_dir)
        print(f"✓ 已清理: {audio_dir}")
    parquet_file = Path(FINAL_REPO_DIR) / f"data_{group_idx}.parquet"
    if parquet_file.exists():
        parquet_file.unlink()
        print(f"✓ 已清理: {parquet_file}")
    json_file = Path(FINAL_REPO_DIR) / f"data_{group_idx}.json"
    if json_file.exists():
        json_file.unlink()
        print(f"✓ 已清理: {json_file}")
    snap = group_snapshot_dir(group_idx)
    if snap.exists():
        import shutil

        shutil.rmtree(snap)
        print(f"✓ 已清理: {snap}")


def prepare_group_upload_snapshot(group_idx: int) -> Path:
    snapshot_dir = group_snapshot_dir(group_idx)
    if snapshot_dir.exists():
        import shutil

        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    parquet_name = f"data_{group_idx}.parquet"
    json_name = f"data_{group_idx}.json"
    audio_name = f"audio_{group_idx}"

    parquet_src = Path(FINAL_REPO_DIR) / parquet_name
    json_src = Path(FINAL_REPO_DIR) / json_name
    audio_src = Path(FINAL_REPO_DIR) / audio_name

    if not parquet_src.exists():
        raise FileNotFoundError(f"缺少待上传文件: {parquet_src}")
    if not json_src.exists():
        raise FileNotFoundError(f"缺少待上传文件: {json_src}")
    if not audio_src.exists():
        raise FileNotFoundError(f"缺少待上传目录: {audio_src}")

    link_or_copy_file(parquet_src, snapshot_dir / parquet_name)
    link_or_copy_file(json_src, snapshot_dir / json_name)
    link_or_copy_dir_recursive(audio_src, snapshot_dir / audio_name)
    return snapshot_dir


def append_anomalies_to_file(anomalies: list[dict[str, Any]]) -> None:
    if not anomalies:
        return
    Path(TRANSCRIPTION_ANOMALIES_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(TRANSCRIPTION_ANOMALIES_FILE, "a", encoding="utf-8") as w:
        for rec in anomalies:
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_conversation_files(
    conversations: list[dict[str, Any]],
) -> dict[str, list[FileRecord]]:
    print("开始构建文件列表...")
    m: dict[str, list[FileRecord]] = {}
    total_files = 0
    for conversation in conversations:
        cid = conversation["conversation_id"]
        for file in conversation["files"]:
            rec = FileRecord(
                file_id=file["file_id"],
                label=file["label"],
                split=file["split"],
                conversation_id=cid,
                participant_id_idx=int(file["participant_id_idx"]),
            )
            m.setdefault(cid, []).append(rec)
            total_files += 1
    print(f"  → 共 {len(m)} 个 conversations，{total_files} 个文件\n")
    return dict(sorted(m.items()))


async def download_to_path(client: httpx.AsyncClient, url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix
    tmp = path.with_suffix(f"{ext}.tmp" if ext else ".tmp")
    max_retries = 3
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            async with client.stream("GET", url, follow_redirects=True) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as f:
                    async for chunk in resp.aiter_bytes():
                        f.write(chunk)
            os.replace(tmp, path)
            return
        except Exception as e:
            last_err = e
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            print(f"  ⚠ 下载失败 (第{attempt}/{max_retries}次): {url} - {e}")
            if attempt < max_retries:
                await asyncio.sleep(2**attempt)
    raise RuntimeError(f"下载失败: {url}") from last_err


async def download_file_assets(
    client: httpx.AsyncClient,
    record: FileRecord,
    sem: asyncio.Semaphore,
) -> tuple[Path, Path]:
    async with sem:
        base_dir = (
            Path(CACHE_DIR)
            / record.conversation_id
            / record.label
            / record.split
        )
        wav_url = f"{S3_BASE}/{record.label}/{record.split}/audio/{record.file_id}.wav"
        jsonl_url = f"{S3_BASE}/{record.label}/{record.split}/metadata/transcript/{record.file_id}.jsonl"
        wav_path = base_dir / f"{record.file_id}.wav"
        jsonl_path = base_dir / f"{record.file_id}.jsonl"
        await download_to_path(client, wav_url, wav_path)
        await download_to_path(client, jsonl_url, jsonl_path)
        return wav_path, jsonl_path


async def download_session_files(
    conversation_id: str,
    files: list[FileRecord],
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
) -> list[tuple[Path, Path]]:
    print(f"  → [{conversation_id}] 开始下载 {len(files)} 个文件...")
    tasks = [download_file_assets(client, r, sem) for r in files]
    return list(await asyncio.gather(*tasks))


def resample_linear(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate:
        return samples.astype(np.float32, copy=False)
    ratio = from_rate / to_rate
    new_len = int(np.ceil(len(samples) / ratio))
    out = np.empty(new_len, dtype=np.float32)
    src = samples.astype(np.float32, copy=False)
    for i in range(new_len):
        src_idx = i * ratio
        idx0 = int(np.floor(src_idx))
        frac = float(src_idx - idx0)
        s0 = float(src[idx0]) if idx0 < len(src) else 0.0
        s1 = float(src[idx0 + 1]) if idx0 + 1 < len(src) else s0
        out[i] = s0 + frac * (s1 - s0)
    return out


def _read_wav_interleaved_float(path: Path) -> tuple[np.ndarray, int, int]:
    """Interleaved float32 samples (frames*channels), sample_rate, channels."""
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    ch = int(data.shape[1])
    interleaved = np.ascontiguousarray(data.reshape(-1))
    return interleaved, int(sr), ch


def mix_audio_files(
    input_paths: list[Path],
    output_path: Path,
    audio_config: AudioConfig | None,
) -> None:
    if not input_paths:
        raise ValueError("没有输入音频文件")
    _s0, sr0, ch0 = _read_wav_interleaved_float(input_paths[0])
    print(f"  → 音频格式: float32 interleaved, {ch0} ch, {sr0} Hz")

    all_samples: list[np.ndarray] = []
    max_len = 0
    for idx, path in enumerate(input_paths):
        s, sr, ch = _read_wav_interleaved_float(path)
        if sr != sr0:
            raise ValueError(
                f"音频 {idx} 的采样率 ({sr}) 与第一个文件 ({sr0}) 不一致"
            )
        if ch != ch0:
            raise ValueError(
                f"音频 {idx} 的声道数 ({ch}) 与第一个文件 ({ch0}) 不一致"
            )
        max_len = max(max_len, len(s))
        all_samples.append(s)

    mixed = np.zeros(max_len, dtype=np.float32)
    for s in all_samples:
        mixed[: len(s)] += s

    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    if peak > 1.0:
        mixed = mixed / peak

    if audio_config is not None:
        frames = len(mixed) // ch0
        if frames * ch0 != len(mixed):
            raise ValueError("混音样本长度不是声道数整数倍，无法重采样")
        de = mixed.reshape(frames, ch0)
        out_chans = [
            resample_linear(de[:, c], sr0, audio_config.sample_rate)
            for c in range(ch0)
        ]
        new_frames = len(out_chans[0])
        mixed = np.stack(out_chans, axis=1).reshape(-1)
        out_sr = audio_config.sample_rate
        bps = audio_config.bits_per_sample
        print(f"  → 音频转换: {sr0}Hz -> {out_sr}Hz/{bps}bit")
    else:
        out_sr = sr0
        bps = 16

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if bps != 16:
        raise ValueError("当前仅支持输出 16-bit PCM（与 Rust 默认配置一致）")
    frames_out = len(mixed) // ch0
    pcm = np.clip(
        mixed.reshape(frames_out, ch0) * 32767.0, -32768, 32767
    ).astype(np.int16)
    sf.write(
        str(output_path),
        pcm,
        out_sr,
        subtype="PCM_16",
        format="WAV",
    )


def convert_wav_format(path: Path, config: AudioConfig) -> None:
    inter, sr, ch = _read_wav_interleaved_float(path)
    info = sf.info(str(path))
    if sr == config.sample_rate and str(info.subtype).upper().startswith("PCM_16"):
        return
    frames = len(inter) // ch
    if frames * ch != len(inter):
        raise ValueError("WAV 样本长度不是声道数整数倍")
    de = inter.reshape(frames, ch)
    out_chans = [resample_linear(de[:, c], sr, config.sample_rate) for c in range(ch)]
    new_f = len(out_chans[0])
    out_inter = np.stack(out_chans, axis=1).reshape(-1)
    pcm = np.clip(
        out_inter.reshape(new_f, ch) * 32767.0, -32768, 32767
    ).astype(np.int16)
    sf.write(
        str(path),
        pcm,
        config.sample_rate,
        subtype="PCM_16",
        format="WAV",
    )


def convert_wav_batch(paths: list[Path], config: AudioConfig) -> None:
    for p in paths:
        convert_wav_format(p, config)


def parse_transcript_jsonl(jsonl_path: Path) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_str in f:
            line_str = line_str.strip()
            if not line_str:
                continue
            try:
                tl = json.loads(line_str)
            except json.JSONDecodeError as e:
                print(f"JSONL 解析失败，跳过该行: {e}")
                continue
            if tl.get("start") is None or tl.get("end") is None:
                raise ValueError(
                    f"JSONL {jsonl_path} start or end is None: {line_str}"
                )
            lines.append(tl)
    return lines


def merge_transcript_lines(lines: list[dict[str, Any]]) -> list[tuple[float, float]]:
    if not lines:
        return []
    segments: list[tuple[float, float]] = []
    current_start = float(lines[0]["start"])
    current_end = float(lines[0]["end"])
    for line in lines[1:]:
        line_start = float(line["start"])
        line_end = float(line["end"])
        if line_start - current_end < 0.5:
            current_end = line_end
        else:
            segments.append((current_start, current_end))
            current_start = line_start
            current_end = line_end
    segments.append((current_start, current_end))
    return segments


def split_wav_by_segments(
    wav_path: Path,
    time_segments: list[tuple[float, float]],
    file_id: str,
    participant_id_idx: int,
) -> list[AudioSegment]:
    data, sr = sf.read(str(wav_path), always_2d=True)
    ch = int(data.shape[1])
    total_frames = int(data.shape[0])
    out_dir = wav_path.parent
    segments: list[AudioSegment] = []

    for idx, (start, end) in enumerate(time_segments):
        start_frame = int(start * sr)
        end_frame = min(int(end * sr), total_frames)
        if start_frame >= end_frame or start_frame >= total_frames:
            continue
        segment_path = out_dir / f"{file_id}_seg{idx}.wav"
        if not segment_path.exists():
            chunk = data[start_frame:end_frame, :]
            if np.issubdtype(chunk.dtype, np.floating):
                f32 = chunk.astype(np.float32)
                pcm = np.clip(f32 * 32767.0, -32768, 32767).astype(np.int16)
                sf.write(str(segment_path), pcm, sr, subtype="PCM_16", format="WAV")
            else:
                sf.write(
                    str(segment_path),
                    chunk.astype(np.int16, copy=False),
                    sr,
                    subtype="PCM_16",
                    format="WAV",
                )
        segments.append(
            AudioSegment(
                segment_path=segment_path,
                offset_seconds=start,
                file_id=file_id,
                participant_id_idx=participant_id_idx,
            )
        )
    return segments


def batch_progress_path(group_idx: int) -> Path:
    return Path(FINAL_DIR) / f"batch_progress_{group_idx}.json"


def batch_results_path(group_idx: int) -> Path:
    return Path(FINAL_DIR) / f"batch_results_{group_idx}.jsonl"


def load_batch_progress(group_idx: int) -> BatchProgress | None:
    p = batch_progress_path(group_idx)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return BatchProgress(
        group_idx=d["group_idx"],
        completed_count=d["completed_count"],
        total_conversations=d["total_conversations"],
        timestamp=d["timestamp"],
    )


def save_batch_progress(progress: BatchProgress) -> None:
    p = batch_progress_path(progress.group_idx)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress.__dict__, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def append_batch_results(group_idx: int, entries: list[BatchResultEntry]) -> None:
    path = batch_results_path(group_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as w:
        for e in entries:
            w.write(
                json.dumps(
                    {
                        "conversation": e.conversation.to_json_obj(),
                        "stats": e.stats.__dict__,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_batch_results(group_idx: int) -> list[BatchResultEntry]:
    path = batch_results_path(group_idx)
    if not path.exists():
        return []
    out: list[BatchResultEntry] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cj = d["conversation"]
            utts: list[ConversationUtterance] = []
            for u in cj["utterances"]:
                words = [
                    Word(word=w["word"], start_time=w["start_time"], end_time=w["end_time"])
                    for w in u["words"]
                ]
                utts.append(ConversationUtterance(spk=u["spk"], words=words))
            conv = Conversation(
                conversation_id=cj["conversation_id"],
                utterances=utts,
                audio_path=cj["audio_path"],
            )
            st = d["stats"]
            stats = ConversationTranscriptionStats(
                conversation_id=st["conversation_id"],
                total_words=st["total_words"],
                anomalous_words=st["anomalous_words"],
                anomaly_ratio=st["anomaly_ratio"],
            )
            out.append(BatchResultEntry(conversation=conv, stats=stats))
    return out


def clean_batch_progress_files(group_idx: int) -> None:
    for p in (batch_progress_path(group_idx), batch_results_path(group_idx)):
        if p.exists():
            p.unlink()


def build_conversation(
    conversation_id: str,
    wav_paths: list[Path],
    segments: list[AudioSegment],
    transcriptions: dict[str, TranscribeResponse],
    group_idx: int,
    audio_config: AudioConfig | None,
) -> tuple[Conversation, list[dict[str, Any]], ConversationTranscriptionStats]:
    all_words: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []

    for segment in segments:
        seg_path_str = str(segment.segment_path)
        asr_result = transcriptions.get(seg_path_str)
        if asr_result and asr_result.timestamps:
            for word in asr_result.timestamps:
                all_words.append(
                    {
                        "word": word["text"],
                        "start_time": word["start_time"] + segment.offset_seconds,
                        "end_time": word["end_time"] + segment.offset_seconds,
                        "raw_start_time": word["start_time"],
                        "raw_end_time": word["end_time"],
                        "spk": segment.participant_id_idx,
                        "file_id": segment.file_id,
                        "segment_file": segment.segment_path.name,
                    }
                )
        elif asr_result:
            print(f"[{conversation_id}] ASR 返回的 timestamps 为空: {seg_path_str}")
            anomalies.append(
                {
                    "conversation_id": conversation_id,
                    "kind": "timestamps_empty",
                    "file_id": segment.file_id,
                    "segment_file": segment.segment_path.name,
                    "time_delta": None,
                    "word": None,
                    "start_time": None,
                    "end_time": None,
                }
            )
        else:
            print(f"[{conversation_id}] 未找到转录结果: {seg_path_str}")
            anomalies.append(
                {
                    "conversation_id": conversation_id,
                    "kind": "transcription_missing",
                    "file_id": segment.file_id,
                    "segment_file": segment.segment_path.name,
                    "time_delta": None,
                    "word": None,
                    "start_time": None,
                    "end_time": None,
                }
            )

    all_words.sort(key=lambda x: x["start_time"])

    utterances: list[ConversationUtterance] = []
    current_spk: int | None = None
    current_words: list[Word] = []
    total_word_count = 0
    anomalous_word_count = 0

    for wsp in all_words:
        spk = wsp["spk"]
        if current_spk is None or current_spk != spk:
            if current_words:
                utterances.append(
                    ConversationUtterance(spk=current_spk, words=list(current_words))
                )
                current_words = []
            current_spk = spk

        time_delta = wsp["raw_end_time"] - wsp["raw_start_time"]
        total_word_count += 1
        if time_delta >= 1.5:
            anomalous_word_count += 1
            anomalies.append(
                {
                    "conversation_id": conversation_id,
                    "kind": "word_time_delta_too_large",
                    "time_delta": time_delta,
                    "word": wsp["word"],
                    "file_id": wsp["file_id"],
                    "segment_file": wsp["segment_file"],
                    "start_time": wsp["raw_start_time"],
                    "end_time": wsp["raw_end_time"],
                }
            )

        current_words.append(
            Word(
                word=wsp["word"],
                start_time=wsp["start_time"],
                end_time=wsp["end_time"],
            )
        )

    if current_words and current_spk is not None:
        utterances.append(ConversationUtterance(spk=current_spk, words=current_words))

    mixed_audio_path = Path(FINAL_REPO_DIR) / f"audio_{group_idx}" / f"{conversation_id}.wav"
    mix_audio_files(wav_paths, mixed_audio_path, audio_config)
    audio_rel_path = f"audio_{group_idx}/{conversation_id}.wav"
    conversation = Conversation(
        conversation_id=conversation_id,
        utterances=utterances,
        audio_path=audio_rel_path,
    )
    stats = ConversationTranscriptionStats(
        conversation_id=conversation_id,
        total_words=total_word_count,
        anomalous_words=anomalous_word_count,
        anomaly_ratio=(
            anomalous_word_count / total_word_count if total_word_count else 0.0
        ),
    )
    return conversation, anomalies, stats


def to_parquet(conversations: list[Conversation], group_idx: int) -> None:
    parquet_path = Path(FINAL_REPO_DIR) / f"data_{group_idx}.parquet"
    total_utts = sum(len(c.utterances) for c in conversations)
    print(
        f"写入最终 parquet: {parquet_path} ({len(conversations)} conversations, {total_utts} utterances)"
    )
    ids = [c.conversation_id for c in conversations]
    ujson = [json.dumps([u.to_json() for u in c.utterances], ensure_ascii=False) for c in conversations]
    apaths = [c.audio_path for c in conversations]
    table = pa.table(
        {
            "conversation_id": ids,
            "utterances_json": ujson,
            "audio_path": apaths,
        }
    )
    pq.write_table(table, parquet_path)
    print(f"✓ parquet 已生成: {parquet_path}")


def to_json(conversations: list[Conversation], group_idx: int) -> None:
    json_path = Path(FINAL_REPO_DIR) / f"data_{group_idx}.json"
    total_utts = sum(len(c.utterances) for c in conversations)
    print(
        f"写入最终 json: {json_path} ({len(conversations)} conversations, {total_utts} utterances)"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            [c.to_json_obj() for c in conversations],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"✓ json 已生成: {json_path}")


def apply_upload_job_result(
    job: UploadJobResult,
    group_progress: GroupProgress,
    clean: bool,
) -> None:
    if job.error:
        print(f"✗ Group {job.group_idx} 上传失败: {job.error}")
        print(f"  快照目录已保留: {job.snapshot_dir}")
        print("  本地数据已保留，可重新运行继续上传")
        return
    print(f"✓ Group {job.group_idx} 上传成功")
    group_state = next(
        (g for g in group_progress.groups if g.group_idx == job.group_idx),
        GroupState(
            group_idx=job.group_idx,
            conversation_ids=list(job.conversation_ids),
            conversation_count=len(job.conversation_ids),
            utterance_count=0,
            generated=True,
            uploaded=False,
            cleaned=False,
            timestamp=datetime.now().astimezone().isoformat(),
        ),
    )
    group_state.uploaded = True
    group_state.timestamp = datetime.now().astimezone().isoformat()
    group_progress.update_or_insert(group_state)
    save_group_progress(group_progress)
    if clean and not group_state.cleaned:
        clean_group(job.group_idx, job.conversation_ids)
        group_state.cleaned = True
        group_state.timestamp = datetime.now().astimezone().isoformat()
        group_progress.update_or_insert(group_state)
        save_group_progress(group_progress)


@dataclass
class DownloadAndTransformParams:
    start_from_group: int | None
    upload: bool
    clean: bool
    clean_cache_each_batch: bool
    audio_config: AudioConfig | None
    stop_after_group_count: int | None
    group_count: int
    batch_size: int


async def process_group(
    group_idx: int,
    group_conversations: list[tuple[str, list[FileRecord]]],
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    batch_size: int,
    audio_config: AudioConfig | None,
    asr_service: AsrService,
    clean_cache_each_batch: bool,
) -> int:
    total_conversations = len(group_conversations)
    if batch_size == 0:
        raise ValueError("batch_size 不能为 0")

    print(f"\n=== Group {group_idx} 开始处理（包含 {total_conversations} 个 conversations）===")

    total_utterance_count = 0
    all_conversations: list[Conversation] = []
    all_conv_stats: list[ConversationTranscriptionStats] = []

    skip_count = 0
    bp = load_batch_progress(group_idx)
    if bp and bp.completed_count > 0:
        saved = load_batch_results(group_idx)
        for entry in saved:
            total_utterance_count += len(entry.conversation.utterances)
            all_conversations.append(entry.conversation)
            all_conv_stats.append(entry.stats)
        print(
            f"  恢复进度：跳过前 {bp.completed_count} 个 conversations（已恢复 {len(all_conversations)} 条结果）"
        )
        skip_count = bp.completed_count

    remaining = group_conversations[skip_count:]
    total_batches = (len(remaining) + batch_size - 1) // batch_size
    completed_count = skip_count

    queue: asyncio.Queue[PreparedBatch | None] = asyncio.Queue(maxsize=1)
    producer_error: list[BaseException | None] = [None]

    async def producer() -> None:
        try:
            for batch_idx, batch_start in enumerate(range(0, len(remaining), batch_size)):
                batch = remaining[batch_start : batch_start + batch_size]
                current_batch = batch_idx + 1
                batch_start_t = time.perf_counter()
                print(
                    f"\n[Group {group_idx}, Batch {current_batch}/{total_batches}] "
                    f"开始处理（包含 {len(batch)} 个 conversations）"
                )
                print(f"\n[Group {group_idx}, Batch {current_batch}] Phase 1: 并发下载...")
                download_tasks = [
                    download_session_files(cid, files, client, sem)
                    for cid, files in batch
                    if files
                ]
                conv_ids_batch = [cid for cid, files in batch if files]
                downloaded: list[tuple[str, list[FileRecord], list[tuple[Path, Path]]]] = []
                results = await asyncio.gather(*download_tasks, return_exceptions=True)
                for cid, files, res in zip(conv_ids_batch, [f for _, f in batch if f], results):
                    if isinstance(res, BaseException):
                        raise RuntimeError(f"  ✗ {cid} 下载失败: {res}") from res
                    downloaded.append((cid, files, res))

                if not downloaded:
                    await queue.put(
                        PreparedBatch(
                            batch_idx=batch_idx,
                            batch_len=len(batch),
                            start_time=batch_start_t,
                            downloaded_sessions=[],
                            all_segments={},
                            original_wav_paths={},
                        )
                    )
                    continue

                if audio_config is not None:
                    print(
                        f"\n[Group {group_idx}, Batch {current_batch}] Phase 1.5: "
                        f"转码为 {audio_config.sample_rate}Hz/{audio_config.bits_per_sample}bit..."
                    )
                    all_wav = [wav for _, _, pairs in downloaded for wav, _ in pairs]
                    await asyncio.to_thread(convert_wav_batch, all_wav, audio_config)
                    print(f"  → {len(all_wav)} 个文件转码完成")

                print(f"\n[Group {group_idx}, Batch {current_batch}] Phase 1.7: 切割音频...")
                all_segments: dict[str, list[AudioSegment]] = {}
                original_wav_paths: dict[str, list[Path]] = {}
                invalid_wavs: list[dict[str, Any]] = []

                for conversation_id, files, paths in downloaded:
                    conv_segments: list[AudioSegment] = []
                    conv_wav_paths: list[Path] = []
                    fallback = False
                    for file, (wav_path, jsonl_path) in zip(files, paths):
                        conv_wav_paths.append(wav_path)
                        try:
                            info = sf.info(str(wav_path))
                            if info.frames == 0:
                                raise ValueError("samples为0")
                        except Exception as e:
                            print(f"  ⚠ [{conversation_id}] {file.file_id} 源WAV无效({e})，跳过")
                            invalid_wavs.append(
                                {
                                    "path": str(wav_path),
                                    "reason": str(e),
                                    "file_id": file.file_id,
                                    "conversation_id": conversation_id,
                                }
                            )
                            continue

                        try:

                            def _split() -> list[AudioSegment]:
                                tl = parse_transcript_jsonl(jsonl_path)
                                if not tl:
                                    raise ValueError("JSONL 为空或所有行无法解析")
                                ts = merge_transcript_lines(tl)
                                if not ts:
                                    raise ValueError("合并后无有效时间段")
                                return split_wav_by_segments(
                                    wav_path, ts, file.file_id, file.participant_id_idx
                                )

                            segs = await asyncio.to_thread(_split)
                            conv_segments.extend(segs)
                        except Exception as e:
                            print(
                                f"  ⚠ [{conversation_id}] {file.file_id} 切割失败，回退到完整音频: {e}"
                            )
                            conv_segments.append(
                                AudioSegment(
                                    segment_path=wav_path,
                                    offset_seconds=0.0,
                                    file_id=file.file_id,
                                    participant_id_idx=file.participant_id_idx,
                                )
                            )
                            fallback = True
                    if fallback:
                        print(f"  ⚠ [{conversation_id}] 部分文件回退到完整音频模式")
                    all_segments[conversation_id] = conv_segments
                    original_wav_paths[conversation_id] = conv_wav_paths

                total_segs = sum(len(v) for v in all_segments.values())
                print(f"  → 共切割为 {total_segs} 个音频片段")

                if invalid_wavs:
                    p = Path(INVALID_WAVS_FILE)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    existing: list[Any] = []
                    if p.exists():
                        try:
                            existing = json.loads(p.read_text(encoding="utf-8"))
                        except Exception:
                            existing = []
                    existing.extend(invalid_wavs)
                    p.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(
                        f"  ⚠ 本批次过滤掉异常源WAV，已记录到 {INVALID_WAVS_FILE}"
                    )

                await queue.put(
                    PreparedBatch(
                        batch_idx=batch_idx,
                        batch_len=len(batch),
                        start_time=batch_start_t,
                        downloaded_sessions=downloaded,
                        all_segments=all_segments,
                        original_wav_paths=original_wav_paths,
                    )
                )
        except BaseException as e:
            producer_error[0] = e
        finally:
            await queue.put(None)

    async def consumer() -> None:
        nonlocal total_utterance_count, completed_count, all_conversations, all_conv_stats

        while True:
            prepared = await queue.get()
            if prepared is None:
                break
            current_batch = prepared.batch_idx + 1
            if not prepared.downloaded_sessions:
                completed_count += prepared.batch_len
                save_batch_progress(
                    BatchProgress(
                        group_idx=group_idx,
                        completed_count=completed_count,
                        total_conversations=total_conversations,
                        timestamp=datetime.now().astimezone().isoformat(),
                    )
                )
                continue

            all_seg = prepared.all_segments
            orig_wav = prepared.original_wav_paths

            print(f"\n[Group {group_idx}, Batch {current_batch}] Phase 2: 批量转录...")
            all_paths: list[str] = []
            for segs in all_seg.values():
                for s in segs:
                    all_paths.append(str(s.segment_path))
            print(f"  → 收集到 {len(all_paths)} 个音频片段，开始批量转录...")
            transcriptions = await asr_service.transcribe_batch_cached(all_paths)
            print(f"  → 批量转录完成，共 {len(transcriptions)} 个结果")

            print(f"\n[Group {group_idx}, Batch {current_batch}] Phase 3: 并发后处理...")
            batch_conv_ids = [cid for cid, _, _ in prepared.downloaded_sessions]

            async def _one(
                conversation_id: str,
                files: list[FileRecord],
                _paths: list[tuple[Path, Path]],
            ) -> tuple[str, Conversation, list[dict], ConversationTranscriptionStats]:
                segments = all_seg.get(conversation_id, [])
                wav_paths = orig_wav.get(conversation_id, [])
                conv, anom, stats = await asyncio.to_thread(
                    build_conversation,
                    conversation_id,
                    wav_paths,
                    segments,
                    transcriptions,
                    group_idx,
                    audio_config,
                )
                return conversation_id, conv, anom, stats

            tasks = [
                _one(cid, files, paths)
                for cid, files, paths in prepared.downloaded_sessions
            ]
            batch_results_raw = await asyncio.gather(*tasks)
            batch_conversations: list[Conversation] = []
            batch_anomalies: list[dict] = []
            batch_stats: list[ConversationTranscriptionStats] = []
            for _cid, conv, anom, stats in batch_results_raw:
                print(
                    f"  → {_cid} 后处理完成，{len(conv.utterances)} 个 utterances "
                    f"（异常词 {stats.anomalous_words}/{stats.total_words}，{stats.anomaly_ratio * 100:.2f}%）"
                )
                batch_conversations.append(conv)
                batch_anomalies.extend(anom)
                batch_stats.append(stats)

            append_anomalies_to_file(batch_anomalies)
            batch_utt = sum(len(c.utterances) for c in batch_conversations)
            elapsed = time.perf_counter() - prepared.start_time
            print(
                f"\n✓ [Group {group_idx}, Batch {current_batch}] 完成，生成 {len(batch_conversations)} "
                f"个 conversations，共 {batch_utt} 个 utterances，耗时 {elapsed:.1f}s"
            )
            total_utterance_count += batch_utt

            entries = [
                BatchResultEntry(conversation=c, stats=s)
                for c, s in zip(batch_conversations, batch_stats)
            ]
            append_batch_results(group_idx, entries)

            if clean_cache_each_batch:
                for conv_id in batch_conv_ids:
                    cdir = Path(CACHE_DIR) / conv_id
                    if cdir.exists():
                        import shutil

                        shutil.rmtree(cdir)
                print(
                    f"  🧹 Batch {current_batch} 缓存已清理（{len(batch_conv_ids)} 个 conversations）"
                )

            all_conversations.extend(batch_conversations)
            all_conv_stats.extend(batch_stats)
            completed_count += prepared.batch_len
            save_batch_progress(
                BatchProgress(
                    group_idx=group_idx,
                    completed_count=completed_count,
                    total_conversations=total_conversations,
                    timestamp=datetime.now().astimezone().isoformat(),
                )
            )

    prod_task = asyncio.create_task(producer())
    cons_task = asyncio.create_task(consumer())
    await asyncio.gather(prod_task, cons_task)
    if producer_error[0] is not None:
        raise producer_error[0]

    print(
        f"\n✓ Group {group_idx} 所有 conversations 处理完成，共生成 {total_utterance_count} 个 utterances"
    )
    to_parquet(all_conversations, group_idx)
    to_json(all_conversations, group_idx)

    global_total = sum(s.total_words for s in all_conv_stats)
    global_anom = sum(s.anomalous_words for s in all_conv_stats)
    stats_obj = {
        "total_conversations": len(all_conv_stats),
        "total_words": global_total,
        "total_anomalous_words": global_anom,
        "overall_anomaly_ratio": (
            global_anom / global_total if global_total else 0.0
        ),
        "conversations": [s.__dict__ for s in all_conv_stats],
    }
    stats_path = Path(FINAL_REPO_DIR) / f"transcription_stats_{group_idx}.json"
    stats_path.write_text(json.dumps(stats_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"✓ 转录统计已生成: {stats_path}（异常词 {global_anom}/{global_total}，"
        f"{stats_obj['overall_anomaly_ratio'] * 100:.2f}%）"
    )
    return total_utterance_count


async def download_and_transform(
    asr_service: AsrService,
    params: DownloadAndTransformParams,
) -> None:
    ensure_dirs()
    with open(CONVERSATIONS_JSON, encoding="utf-8") as f:
        conversations = json.load(f)
    conversation_files = build_conversation_files(conversations)

    sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)
    if params.audio_config:
        print(
            f"音频输出配置: {params.audio_config.sample_rate}Hz, {params.audio_config.bits_per_sample}-bit"
        )
    else:
        print("音频输出配置: 保持原始格式（不转换）")

    conv_list = list(conversation_files.items())
    total_conversations = len(conv_list)
    total_groups = (total_conversations + params.group_count - 1) // params.group_count
    print("\n=== 数据处理配置 ===")
    print(f"总会话数: {total_conversations}")
    print(f"每组会话数: {params.group_count}")
    print(f"总组数: {total_groups}")
    print(f"组内并发数: {params.batch_size}")
    if params.start_from_group is not None:
        print(f"从第 {params.start_from_group} 组开始")
    if params.stop_after_group_count is not None:
        print(f"处理 {params.stop_after_group_count} 组后停止")

    initial_progress = load_group_progress()
    if initial_progress.groups and initial_progress.group_count != params.group_count:
        raise RuntimeError(
            f"group_count 已从 {initial_progress.group_count} 变更为 {params.group_count}，与已有进度不兼容。"
            f"如需使用新的 group_count，请先删除 {GROUP_PROGRESS_FILE} 或确保所有组已完成。"
        )
    initial_progress.group_count = params.group_count
    save_group_progress(initial_progress)

    progress_lock = asyncio.Lock()

    async def get_progress() -> GroupProgress:
        async with progress_lock:
            return load_group_progress()

    async def save_state(state: GroupState) -> None:
        async with progress_lock:
            gp = load_group_progress()
            gp.update_or_insert(state)
            save_group_progress(gp)

    processed_groups = 0
    upload_q: asyncio.Queue[UploadRequest | None] = asyncio.Queue()

    async def upload_worker() -> None:
        while True:
            req = await upload_q.get()
            if req is None:
                break
            err: str | None = None
            try:
                await asyncio.to_thread(upload_folder, str(req.snapshot_dir), ".")
            except Exception as e:
                err = str(e)
            async with progress_lock:
                gp = load_group_progress()
                apply_upload_job_result(
                    UploadJobResult(
                        group_idx=req.group_idx,
                        conversation_ids=req.conversation_ids,
                        snapshot_dir=req.snapshot_dir,
                        error=err,
                    ),
                    gp,
                    params.clean,
                )

    upload_task = asyncio.create_task(upload_worker())

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(600.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        follow_redirects=True,
    ) as client:
        for group_idx in range(total_groups):
            if (
                params.start_from_group is not None
                and group_idx < params.start_from_group
            ):
                print(f"\n⏭️  跳过 Group {group_idx} (start_from_group={params.start_from_group})")
                continue

            start = group_idx * params.group_count
            end = min((group_idx + 1) * params.group_count, total_conversations)
            group_conversations = conv_list[start:end]

            gp = await get_progress()
            st = gp.find_group(group_idx)

            if st and st.uploaded and st.cleaned:
                print(f"\n⏭️  跳过已完成的 Group {group_idx}")
                processed_groups += 1
                continue
            if st and st.uploaded and not st.cleaned:
                print(f"\n🧹 Group {group_idx} 已上传，执行清理")
                if params.clean:
                    clean_group(group_idx, st.conversation_ids)
                    ns = GroupState(
                        group_idx=st.group_idx,
                        conversation_ids=st.conversation_ids,
                        conversation_count=st.conversation_count,
                        utterance_count=st.utterance_count,
                        generated=st.generated,
                        uploaded=True,
                        cleaned=True,
                        timestamp=datetime.now().astimezone().isoformat(),
                    )
                    await save_state(ns)
                processed_groups += 1
                continue
            if st and st.generated and not st.uploaded:
                print(f"\n📤 Group {group_idx} 已生成，仅上传")
                if params.upload:
                    snap = prepare_group_upload_snapshot(group_idx)
                    print(f"📦 Group {group_idx} 上传快照已准备: {snap}")
                    await upload_q.put(
                        UploadRequest(
                            group_idx=group_idx,
                            conversation_ids=list(st.conversation_ids),
                            snapshot_dir=snap,
                        )
                    )
                processed_groups += 1
                continue

            print(f"\n🔄 完整处理 Group {group_idx}")
            ensure_group_dirs(group_idx)
            conversation_ids = [cid for cid, _ in group_conversations]

            t0 = time.perf_counter()
            utterance_count = await process_group(
                group_idx,
                group_conversations,
                client,
                sem,
                params.batch_size,
                params.audio_config,
                asr_service,
                params.clean_cache_each_batch,
            )
            dt = time.perf_counter() - t0
            print(f"  → 处理组 {group_idx} 完成: 耗时 {dt}s")

            await save_state(
                GroupState(
                    group_idx=group_idx,
                    conversation_ids=conversation_ids,
                    conversation_count=len(group_conversations),
                    utterance_count=utterance_count,
                    generated=True,
                    uploaded=False,
                    cleaned=False,
                    timestamp=datetime.now().astimezone().isoformat(),
                )
            )
            clean_batch_progress_files(group_idx)

            if params.upload:
                snap = prepare_group_upload_snapshot(group_idx)
                print(f"📦 Group {group_idx} 上传快照已准备: {snap}")
                await upload_q.put(
                    UploadRequest(
                        group_idx=group_idx,
                        conversation_ids=conversation_ids,
                        snapshot_dir=snap,
                    )
                )

            processed_groups += 1
            if (
                params.stop_after_group_count is not None
                and processed_groups >= params.stop_after_group_count
            ):
                print(f"\n已完成 {processed_groups} 个组，停止执行。")
                break

    await upload_q.put(None)
    await upload_task
    print(f"\n✓ 所有组处理完成，共处理 {processed_groups} 个组")
