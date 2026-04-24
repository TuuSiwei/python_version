"""Build SI conversation-level WAV + RTTM configs for Hugging Face Datasets."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from io_utils import atomic_write_json

CONVERSATIONS_JSON = "conversations.json"
FINAL_DIR = Path("final")
STAGING_ROOT = FINAL_DIR / "si_rttm_staging"
LOG_DIR = FINAL_DIR / "si_rttm_logs"
FAILED_LOG = LOG_DIR / "failed.jsonl"
SKIPPED_LOG = LOG_DIR / "skipped.jsonl"
PROGRESS_FILE = FINAL_DIR / "si_rttm_progress.json"
S3_BASE = "https://dl.fbaipublicfiles.com/seamless_interaction"
EXPECTED_SAMPLE_RATE = 48_000


@dataclass
class ConversationFile:
    file_id: str
    label: str
    split: str
    participant_id: str
    participant_id_idx: int


@dataclass
class ConversationJob:
    conversation_id: str
    files: list[ConversationFile]
    global_index: int


@dataclass
class SiRttmParams:
    repo_id: str
    config_prefix: str
    chunk_size: int
    upload: bool
    download_concurrency: int
    process_concurrency: int
    max_shards: int
    start_from_conversation: int | None
    stop_after_conversation_count: int | None
    window_sec: float
    shift_sec: float
    max_spks: int
    feat_per_sec: int


@dataclass
class ConversationResult:
    conversation_id: str
    global_index: int
    status: str
    audio_path: str | None = None
    rttm_path: str | None = None
    duration: float | None = None
    rttm_lines: int = 0
    error: str | None = None


@dataclass
class ChunkProgress:
    chunk_index: int
    config_name: str | None
    start_index: int
    end_index: int
    input_count: int
    processed_count: int
    failed_count: int
    skipped_count: int
    dataset_rows: int
    uploaded: bool
    cleaned: bool
    timestamp: str


@dataclass
class DatasetBuildResult:
    dataset: Any
    rows: int
    skipped_short: int
    skipped_no_valid: int
    skipped_records: list[dict[str, Any]]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def ensure_dirs() -> None:
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_progress() -> dict[str, Any]:
    if not PROGRESS_FILE.exists():
        return {"chunks": [], "last_updated": "", "schema_version": 1}
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("chunks", [])
    data.setdefault("schema_version", 1)
    return data


def save_progress(data: dict[str, Any]) -> None:
    data["last_updated"] = now_iso()
    atomic_write_json(PROGRESS_FILE, data, indent=2)


def progress_chunks(progress: dict[str, Any]) -> list[dict[str, Any]]:
    return list(progress.get("chunks") or [])


def find_completed_chunk(
    progress: dict[str, Any],
    start_index: int,
    end_index: int,
) -> dict[str, Any] | None:
    for chunk in progress_chunks(progress):
        if (
            int(chunk.get("start_index", -1)) == start_index
            and int(chunk.get("end_index", -1)) == end_index
            and bool(chunk.get("uploaded"))
            and bool(chunk.get("cleaned"))
        ):
            return chunk
    return None


def upsert_chunk_progress(progress: dict[str, Any], state: ChunkProgress) -> None:
    rows = progress.setdefault("chunks", [])
    state_obj = asdict(state)
    for idx, existing in enumerate(rows):
        if (
            int(existing.get("start_index", -1)) == state.start_index
            and int(existing.get("end_index", -1)) == state.end_index
        ):
            rows[idx] = state_obj
            return
    rows.append(state_obj)


def load_conversation_jobs() -> list[ConversationJob]:
    with open(CONVERSATIONS_JSON, encoding="utf-8") as f:
        conversations = json.load(f)

    jobs: list[ConversationJob] = []
    for global_index, conversation in enumerate(conversations):
        files: list[ConversationFile] = []
        for file_obj in conversation.get("files") or []:
            files.append(
                ConversationFile(
                    file_id=str(file_obj["file_id"]),
                    label=str(file_obj["label"]),
                    split=str(file_obj["split"]),
                    participant_id=str(file_obj["participant_id"]),
                    participant_id_idx=int(file_obj["participant_id_idx"]),
                )
            )
        jobs.append(
            ConversationJob(
                conversation_id=str(conversation["conversation_id"]),
                files=files,
                global_index=global_index,
            )
        )
    return jobs


async def download_to_path(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    url: str,
    path: Path,
    max_retries: int = 3,
) -> None:
    if path.exists() and path.stat().st_size > 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    last_err: Exception | None = None

    async with sem:
        for attempt in range(1, max_retries + 1):
            try:
                if tmp.exists():
                    tmp.unlink()
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with open(tmp, "wb") as f:
                        async for chunk in resp.aiter_bytes():
                            f.write(chunk)
                os.replace(tmp, path)
                return
            except Exception as exc:
                last_err = exc
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
                if attempt < max_retries:
                    await asyncio.sleep(2**attempt)

    raise RuntimeError(f"download failed after {max_retries} attempts: {url}") from last_err


async def download_conversation_assets(
    client: httpx.AsyncClient,
    download_sem: asyncio.Semaphore,
    job: ConversationJob,
    raw_dir: Path,
) -> None:
    tasks: list[asyncio.Task[None]] = []
    for file_obj in job.files:
        base_dir = raw_dir / file_obj.label / file_obj.split
        wav_url = f"{S3_BASE}/{file_obj.label}/{file_obj.split}/audio/{file_obj.file_id}.wav"
        jsonl_url = (
            f"{S3_BASE}/{file_obj.label}/{file_obj.split}/metadata/transcript/"
            f"{file_obj.file_id}.jsonl"
        )
        wav_path = base_dir / f"{file_obj.file_id}.wav"
        jsonl_path = base_dir / f"{file_obj.file_id}.jsonl"
        tasks.append(asyncio.create_task(download_to_path(client, download_sem, wav_url, wav_path)))
        tasks.append(asyncio.create_task(download_to_path(client, download_sem, jsonl_url, jsonl_path)))
    await asyncio.gather(*tasks)


def raw_asset_paths(raw_dir: Path, file_obj: ConversationFile) -> tuple[Path, Path]:
    base_dir = raw_dir / file_obj.label / file_obj.split
    return base_dir / f"{file_obj.file_id}.wav", base_dir / f"{file_obj.file_id}.jsonl"


def read_jsonl_rttm_rows(
    job: ConversationJob,
    raw_dir: Path,
) -> list[tuple[float, str, str]]:
    rows: list[tuple[float, str, str]] = []
    for file_obj in job.files:
        _wav_path, jsonl_path = raw_asset_paths(raw_dir, file_obj)
        with open(jsonl_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    start = float(item["start"])
                    end = float(item["end"])
                except Exception as exc:
                    raise ValueError(
                        f"{jsonl_path}:{line_no} missing valid start/end"
                    ) from exc
                duration = end - start
                if duration <= 0:
                    raise ValueError(
                        f"{jsonl_path}:{line_no} has non-positive duration: {duration}"
                    )
                rttm = (
                    f"SPEAKER {job.conversation_id} 1 {start:.3f} {duration:.3f} "
                    f"<NA> <NA> {file_obj.participant_id} <NA> <NA>"
                )
                rows.append((start, file_obj.participant_id, rttm))
    rows.sort(key=lambda row: (row[0], row[1]))
    return rows


def build_conversation_outputs(
    job: ConversationJob,
    raw_dir: Path,
    audio_out: Path,
    rttm_out: Path,
) -> ConversationResult:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install soundfile or run pip install -r requirements.txt") from exc

    if audio_out.exists() and audio_out.stat().st_size > 0 and rttm_out.exists():
        info = sf.info(str(audio_out))
        return ConversationResult(
            conversation_id=job.conversation_id,
            global_index=job.global_index,
            status="ok",
            audio_path=str(audio_out),
            rttm_path=str(rttm_out),
            duration=float(info.duration),
            rttm_lines=sum(1 for _ in open(rttm_out, encoding="utf-8")),
        )

    if not job.files:
        raise ValueError("conversation has no files")

    arrays: list[np.ndarray] = []
    sample_rate: int | None = None
    channels: int | None = None
    max_frames = 0

    for file_obj in job.files:
        wav_path, _jsonl_path = raw_asset_paths(raw_dir, file_obj)
        data, sr = sf.read(str(wav_path), always_2d=True, dtype="float32")
        if int(sr) != EXPECTED_SAMPLE_RATE:
            raise ValueError(f"{wav_path} sample_rate={sr}, expected {EXPECTED_SAMPLE_RATE}")
        if sample_rate is None:
            sample_rate = int(sr)
            channels = int(data.shape[1])
        elif int(sr) != sample_rate:
            raise ValueError(f"{wav_path} sample_rate={sr}, expected {sample_rate}")
        if int(data.shape[1]) != channels:
            raise ValueError(f"{wav_path} channels={data.shape[1]}, expected {channels}")
        arrays.append(np.asarray(data, dtype=np.float32))
        max_frames = max(max_frames, int(data.shape[0]))

    if sample_rate is None or channels is None or max_frames <= 0:
        raise ValueError("no valid audio samples")

    mixed = np.zeros((max_frames, channels), dtype=np.float32)
    for data in arrays:
        mixed[: data.shape[0], :] += data

    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        mixed /= peak

    rttm_rows = read_jsonl_rttm_rows(job, raw_dir)

    audio_out.parent.mkdir(parents=True, exist_ok=True)
    rttm_out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_out), mixed, sample_rate, subtype="FLOAT", format="WAV")
    with open(rttm_out, "w", encoding="utf-8") as f:
        for _start, _speaker, rttm in rttm_rows:
            f.write(rttm + "\n")

    return ConversationResult(
        conversation_id=job.conversation_id,
        global_index=job.global_index,
        status="ok",
        audio_path=str(audio_out),
        rttm_path=str(rttm_out),
        duration=max_frames / sample_rate,
        rttm_lines=len(rttm_rows),
    )


async def process_conversation(
    client: httpx.AsyncClient,
    download_sem: asyncio.Semaphore,
    process_sem: asyncio.Semaphore,
    chunk_dir: Path,
    job: ConversationJob,
) -> ConversationResult:
    raw_dir = chunk_dir / "raw" / job.conversation_id
    audio_out = chunk_dir / "audio" / f"{job.conversation_id}.wav"
    rttm_out = chunk_dir / "rttm" / f"{job.conversation_id}.rttm"

    try:
        if not (audio_out.exists() and audio_out.stat().st_size > 0 and rttm_out.exists()):
            await download_conversation_assets(client, download_sem, job, raw_dir)
        async with process_sem:
            result = await asyncio.to_thread(
                build_conversation_outputs,
                job,
                raw_dir,
                audio_out,
                rttm_out,
            )
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        return result
    except Exception as exc:
        if raw_dir.exists():
            shutil.rmtree(raw_dir, ignore_errors=True)
        return ConversationResult(
            conversation_id=job.conversation_id,
            global_index=job.global_index,
            status="failed",
            error=str(exc),
        )


async def process_chunk(
    jobs: list[ConversationJob],
    chunk_dir: Path,
    params: SiRttmParams,
) -> list[ConversationResult]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    (chunk_dir / "audio").mkdir(parents=True, exist_ok=True)
    (chunk_dir / "rttm").mkdir(parents=True, exist_ok=True)

    download_sem = asyncio.Semaphore(max(1, params.download_concurrency))
    process_sem = asyncio.Semaphore(max(1, params.process_concurrency))
    queue: asyncio.Queue[ConversationJob | None] = asyncio.Queue()
    results: list[ConversationResult] = []

    for job in jobs:
        queue.put_nowait(job)

    worker_count = min(max(1, params.download_concurrency), max(1, len(jobs)))
    timeout = httpx.Timeout(600.0)
    limits = httpx.Limits(
        max_connections=max(10, params.download_concurrency * 2),
        max_keepalive_connections=max(10, params.download_concurrency),
    )

    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
        async def worker(worker_id: int) -> None:
            while True:
                job = await queue.get()
                try:
                    if job is None:
                        return
                    started = time.perf_counter()
                    result = await process_conversation(
                        client,
                        download_sem,
                        process_sem,
                        chunk_dir,
                        job,
                    )
                    elapsed = time.perf_counter() - started
                    if result.status == "ok":
                        print(
                            f"[worker {worker_id}] {job.conversation_id} ok "
                            f"({result.duration:.1f}s, rttm={result.rttm_lines}, {elapsed:.1f}s)"
                        )
                    else:
                        print(f"[worker {worker_id}] {job.conversation_id} failed: {result.error}")
                    results.append(result)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker(i)) for i in range(worker_count)]
        await queue.join()
        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers)

    raw_root = chunk_dir / "raw"
    if raw_root.exists() and not any(raw_root.iterdir()):
        raw_root.rmdir()

    results.sort(key=lambda item: item.global_index)
    return results


Segments = list[tuple[float, float, str]]

_RTTM_RE = re.compile(
    r"^SPEAKER\s+\S+\s+\d+\s+"
    r"(?P<start>[\d.]+)\s+"
    r"(?P<dur>[\d.]+)\s+"
    r"\S+\s+\S+\s+"
    r"(?P<spk>\S+)"
)


def parse_rttm(path: str) -> Segments:
    segments: Segments = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            match = _RTTM_RE.match(line.strip())
            if match is None:
                continue
            start = float(match.group("start"))
            dur = float(match.group("dur"))
            spk = match.group("spk")
            if dur > 0:
                segments.append((start, start + dur, spk))
    return segments


def segments_to_full_targets(
    segments: Segments,
    duration: float,
    feat_per_sec: int,
) -> tuple[np.ndarray, list[str]]:
    spk_map: dict[str, int] = {}
    for _start, _end, spk in segments:
        if spk not in spk_map:
            spk_map[spk] = len(spk_map)

    n_spks = len(spk_map) if spk_map else 1
    total_frames = int(duration * feat_per_sec)
    targets = np.zeros((total_frames, n_spks), dtype=np.int8)

    for start, end, spk in segments:
        col = spk_map[spk]
        stt_fr = max(0, int(start * feat_per_sec))
        end_fr = min(total_frames, int(end * feat_per_sec))
        targets[stt_fr:end_fr, col] = 1

    speaker_ids = [""] * n_spks
    for spk, idx in spk_map.items():
        speaker_ids[idx] = spk
    return targets, speaker_ids


def compute_valid_offsets(
    targets: np.ndarray,
    duration: float,
    window_sec: float,
    shift_sec: float,
    max_spks: int,
    feat_per_sec: int,
) -> list[float]:
    window_frames = int(window_sec * feat_per_sec)
    valid: list[float] = []
    offset_sec = 0.0
    while offset_sec + window_sec <= duration + 1e-6:
        fr_s = int(offset_sec * feat_per_sec)
        fr_e = min(fr_s + window_frames, targets.shape[0])
        active_count = int(targets[fr_s:fr_e, :].any(axis=0).sum())
        if active_count <= max_spks:
            valid.append(round(offset_sec, 6))
        offset_sec += shift_sec
    return valid


def scan_audio_rttm_pairs(audio_dir: Path, rttm_dir: Path) -> list[tuple[Path, Path]]:
    rttm_by_stem = {
        path.stem.lower(): path
        for path in rttm_dir.rglob("*.rttm")
        if path.is_file()
    }
    pairs: list[tuple[Path, Path]] = []
    for audio_path in sorted(audio_dir.rglob("*.wav")):
        rttm_path = rttm_by_stem.get(audio_path.stem.lower())
        if rttm_path is not None:
            pairs.append((audio_path, rttm_path))
    return pairs


def build_dataset_for_chunk(
    chunk_dir: Path,
    config_name: str,
    params: SiRttmParams,
) -> DatasetBuildResult:
    try:
        import soundfile as sf
        from datasets import Audio, Dataset, Features, Sequence, Value
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: install datasets and soundfile with pip install -r requirements.txt"
        ) from exc

    audio_dir = chunk_dir / "audio"
    rttm_dir = chunk_dir / "rttm"
    pairs = scan_audio_rttm_pairs(audio_dir, rttm_dir)
    if not pairs:
        raise RuntimeError(f"No audio/RTTM pairs found in {audio_dir} / {rttm_dir}")

    rows: dict[str, list[Any]] = {
        "session_id": [],
        "audio": [],
        "targets": [],
        "speaker_ids": [],
        "duration": [],
        "num_speakers": [],
        "valid_offsets": [],
    }
    skipped_records: list[dict[str, Any]] = []
    skipped_short = 0
    skipped_no_valid = 0

    for audio_path, rttm_path in pairs:
        info = sf.info(str(audio_path))
        duration = float(info.duration)
        stem = audio_path.stem
        if duration < params.window_sec:
            skipped_short += 1
            skipped_records.append(
                {
                    "conversation_id": stem,
                    "config_name": config_name,
                    "kind": "short",
                    "duration": duration,
                    "window_sec": params.window_sec,
                    "timestamp": now_iso(),
                }
            )
            continue

        segments = parse_rttm(str(rttm_path))
        targets, speaker_ids = segments_to_full_targets(
            segments,
            duration=duration,
            feat_per_sec=params.feat_per_sec,
        )
        valid_offsets = compute_valid_offsets(
            targets,
            duration=duration,
            window_sec=params.window_sec,
            shift_sec=params.shift_sec,
            max_spks=params.max_spks,
            feat_per_sec=params.feat_per_sec,
        )
        if not valid_offsets:
            skipped_no_valid += 1
            skipped_records.append(
                {
                    "conversation_id": stem,
                    "config_name": config_name,
                    "kind": "no_valid_offsets",
                    "duration": duration,
                    "num_speakers": len(speaker_ids),
                    "timestamp": now_iso(),
                }
            )
            continue

        rows["session_id"].append(f"{config_name}/{stem}")
        rows["audio"].append(str(audio_path))
        rows["targets"].append(targets.tolist())
        rows["speaker_ids"].append(speaker_ids)
        rows["duration"].append(round(duration, 6))
        rows["num_speakers"].append(len(speaker_ids))
        rows["valid_offsets"].append(valid_offsets)

    features = Features(
        {
            "session_id": Value("string"),
            "audio": Audio(),
            "targets": Sequence(Sequence(Value("int8"))),
            "speaker_ids": Sequence(Value("string")),
            "duration": Value("float64"),
            "num_speakers": Value("int32"),
            "valid_offsets": Sequence(Value("float64")),
        }
    )
    dataset = Dataset.from_dict(rows, features=features)
    return DatasetBuildResult(
        dataset=dataset,
        rows=len(dataset),
        skipped_short=skipped_short,
        skipped_no_valid=skipped_no_valid,
        skipped_records=skipped_records,
    )


def remote_config_names(repo_id: str) -> set[str]:
    token = os.getenv("HF_TOKEN") or None
    names: set[str] = set()
    errors: list[str] = []
    discovered = False

    try:
        from datasets import get_dataset_config_names

        try:
            names.update(get_dataset_config_names(repo_id, token=token))
        except TypeError:
            names.update(get_dataset_config_names(repo_id))
        discovered = True
    except Exception as exc:
        errors.append(f"datasets.get_dataset_config_names: {exc}")

    try:
        from huggingface_hub import HfApi

        files = HfApi(token=token).list_repo_files(repo_id=repo_id, repo_type="dataset")
        discovered = True
        for path in files:
            for match in re.finditer(r"(?:^|/)([A-Za-z][A-Za-z0-9_]*_\d+)(?:[-/])", path):
                names.add(match.group(1))
    except Exception as exc:
        errors.append(f"huggingface_hub.list_repo_files: {exc}")

    if not discovered:
        raise RuntimeError("Could not discover remote configs safely: " + " | ".join(errors))
    return names


def used_config_indices(progress: dict[str, Any], prefix: str, remote_names: set[str]) -> set[int]:
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    used: set[int] = set()
    for name in remote_names:
        match = pat.match(name)
        if match:
            used.add(int(match.group(1)))
    for chunk in progress_chunks(progress):
        if not bool(chunk.get("uploaded")):
            continue
        name = chunk.get("config_name")
        if isinstance(name, str):
            match = pat.match(name)
            if match:
                used.add(int(match.group(1)))
    return used


def next_config_name(prefix: str, used_indices: set[int]) -> str:
    idx = 1
    while idx in used_indices:
        idx += 1
    used_indices.add(idx)
    return f"{prefix}_{idx}"


def push_dataset(dataset: Any, params: SiRttmParams, config_name: str) -> None:
    if len(dataset) == 0:
        return
    token = os.getenv("HF_TOKEN") or None
    num_shards = min(params.max_shards, max(1, len(dataset)))
    print(
        f"Pushing config {config_name} to {params.repo_id}: "
        f"rows={len(dataset)}, num_shards={num_shards}"
    )
    try:
        dataset.push_to_hub(
            params.repo_id,
            config_name=config_name,
            num_shards=num_shards,
            token=token,
        )
    except TypeError:
        dataset.push_to_hub(
            params.repo_id,
            config_name=config_name,
            num_shards=num_shards,
        )


async def run_si_rttm_pipeline(params: SiRttmParams) -> None:
    ensure_dirs()
    all_jobs = load_conversation_jobs()
    total = len(all_jobs)
    start = params.start_from_conversation or 0
    if start < 0 or start >= total:
        raise ValueError(f"start_from_conversation out of range: {start}, total={total}")
    end = total
    if params.stop_after_conversation_count is not None:
        if params.stop_after_conversation_count <= 0:
            raise ValueError("stop_after_conversation_count must be positive")
        end = min(total, start + params.stop_after_conversation_count)

    selected_jobs = all_jobs[start:end]
    print("=== SI RTTM dataset pipeline ===")
    print(f"conversations total={total}, selected={len(selected_jobs)}, range=[{start}, {end})")
    print(
        f"chunk_size={params.chunk_size}, download_concurrency={params.download_concurrency}, "
        f"process_concurrency={params.process_concurrency}"
    )
    print(f"repo_id={params.repo_id}, config_prefix={params.config_prefix}, upload={params.upload}")

    progress = load_progress()
    remote_names = remote_config_names(params.repo_id) if params.upload else set()
    used_indices = used_config_indices(progress, params.config_prefix, remote_names)
    processed_chunks = 0

    for chunk_start in range(start, end, params.chunk_size):
        chunk_end = min(chunk_start + params.chunk_size, end)
        chunk_jobs = all_jobs[chunk_start:chunk_end]
        chunk_index = chunk_start // params.chunk_size

        progress = load_progress()
        completed = find_completed_chunk(progress, chunk_start, chunk_end)
        if completed:
            print(
                f"Skip completed chunk {chunk_index}: "
                f"[{chunk_start}, {chunk_end}) config={completed.get('config_name')}"
            )
            continue

        config_name = next_config_name(params.config_prefix, used_indices)
        chunk_dir = STAGING_ROOT / (
            f"chunk_{chunk_index:05d}_{chunk_start:08d}_{chunk_end:08d}_{config_name}"
        )

        print(
            f"\n=== Chunk {chunk_index}: [{chunk_start}, {chunk_end}) "
            f"config={config_name}, conversations={len(chunk_jobs)} ==="
        )
        results = await process_chunk(chunk_jobs, chunk_dir, params)

        failed = [r for r in results if r.status != "ok"]
        append_jsonl(
            FAILED_LOG,
            [
                {
                    "conversation_id": r.conversation_id,
                    "global_index": r.global_index,
                    "chunk_index": chunk_index,
                    "config_name": config_name,
                    "error": r.error,
                    "timestamp": now_iso(),
                }
                for r in failed
            ],
        )

        ok_count = len(results) - len(failed)
        print(f"Chunk {chunk_index} processed: ok={ok_count}, failed={len(failed)}")
        if ok_count == 0:
            raise RuntimeError(f"Chunk {chunk_index} produced no valid conversation outputs")

        build = build_dataset_for_chunk(chunk_dir, config_name, params)
        append_jsonl(SKIPPED_LOG, build.skipped_records)
        print(
            f"Chunk {chunk_index} dataset: rows={build.rows}, "
            f"skipped_short={build.skipped_short}, skipped_no_valid={build.skipped_no_valid}"
        )

        if params.upload and build.rows > 0:
            push_dataset(build.dataset, params, config_name)
            uploaded = True
        elif params.upload and build.rows == 0:
            print(f"Chunk {chunk_index} has no dataset rows; no config will be pushed.")
            uploaded = True
        else:
            print(f"--no-upload set; keeping staging directory: {chunk_dir}")
            uploaded = False

        cleaned = False
        if uploaded:
            shutil.rmtree(chunk_dir, ignore_errors=True)
            cleaned = True
            print(f"Cleaned staging directory: {chunk_dir}")

        if uploaded:
            progress = load_progress()
            upsert_chunk_progress(
                progress,
                ChunkProgress(
                    chunk_index=chunk_index,
                    config_name=config_name if build.rows > 0 else None,
                    start_index=chunk_start,
                    end_index=chunk_end,
                    input_count=len(chunk_jobs),
                    processed_count=ok_count,
                    failed_count=len(failed),
                    skipped_count=build.skipped_short + build.skipped_no_valid,
                    dataset_rows=build.rows,
                    uploaded=uploaded,
                    cleaned=cleaned,
                    timestamp=now_iso(),
                ),
            )
            save_progress(progress)
        processed_chunks += 1

    print(f"\nDone. processed_chunks={processed_chunks}")
