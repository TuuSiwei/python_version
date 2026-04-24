"""HF parquet download, enrich, regroup (Rust spk_convert parity)."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

CONVERSATIONS_JSON = "conversations.json"
FINAL_REPO_DIR = "final/repo"
DOWNLOAD_CACHE_DIR = "final/spk_convert/source"
HF_DATASET_BASE_URL = "https://huggingface.co/datasets/humanify/si/resolve/main"
INPUT_FILE_START = 0
INPUT_FILE_END = 26
SHARD_SIZE = 2400


@dataclass
class ConversationMetadata:
    label: str
    split: str
    speaker_id_json: str


@dataclass
class SourceRecord:
    conversation_id: str
    utterances_json: str
    audio_path: str


@dataclass
class EnrichedRecord:
    conversation_id: str
    utterances_json: str
    audio_path: str
    label: str
    split: str
    speaker_id: str


def load_conversation_metadata() -> dict[str, ConversationMetadata]:
    with open(CONVERSATIONS_JSON, encoding="utf-8") as f:
        conversations: list[dict] = json.load(f)
    metadata_map: dict[str, ConversationMetadata] = {}
    for conversation in conversations:
        files = conversation.get("files") or []
        if not files:
            continue
        first = files[0]
        label = first["label"]
        split = first["split"]
        speaker_map: dict[int, str] = {}
        for file in files:
            if file["label"] != label or file["split"] != split:
                label = first["label"]
                split = first["split"]
            idx = int(file["participant_id_idx"])
            if idx not in speaker_map:
                speaker_map[idx] = file["participant_id"]
        speaker_id_json = json.dumps(speaker_map, ensure_ascii=False)
        metadata_map[conversation["conversation_id"]] = ConversationMetadata(
            label=label,
            split=split,
            speaker_id_json=speaker_id_json,
        )
    return metadata_map


async def download_source_parquet(client: httpx.AsyncClient, idx: int) -> Path:
    local_path = Path(DOWNLOAD_CACHE_DIR) / f"data_{idx}.parquet"
    if local_path.exists() and local_path.stat().st_size > 0:
        print(f"复用已下载文件: {local_path}")
        return local_path
    tmp_path = Path(DOWNLOAD_CACHE_DIR) / f"data_{idx}.parquet.part"
    if tmp_path.exists():
        tmp_path.unlink()
    url = f"{HF_DATASET_BASE_URL}/data_{idx}.parquet"
    print(f"下载 {url}")
    r = await client.get(url)
    r.raise_for_status()
    tmp_path.write_bytes(r.content)
    tmp_path.rename(local_path)
    return local_path


def read_source_parquet(path: Path) -> list[SourceRecord]:
    table = pq.read_table(path)
    rows: list[SourceRecord] = []
    # columns: conversation_id, utterances_json, audio_path
    col0 = table.column(0).to_pylist()
    col1 = table.column(1).to_pylist()
    col2 = table.column(2).to_pylist()
    for a, b, c in zip(col0, col1, col2):
        rows.append(
            SourceRecord(
                conversation_id=str(a),
                utterances_json=str(b) if not isinstance(b, str) else b,
                audio_path=str(c),
            )
        )
    return rows


def sanitize_segment(value: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in value)


def write_enriched_parquet(output_path: Path, rows: list[EnrichedRecord]) -> None:
    conversation_ids = [r.conversation_id for r in rows]
    utterances_json = [r.utterances_json for r in rows]
    audio_paths = [r.audio_path for r in rows]
    labels = [r.label for r in rows]
    splits = [r.split for r in rows]
    speaker_ids = [r.speaker_id for r in rows]
    table = pa.table(
        {
            "conversation_id": conversation_ids,
            "utterances_json": utterances_json,
            "audio_path": audio_paths,
            "label": labels,
            "split": splits,
            "speaker_id": speaker_ids,
        }
    )
    pq.write_table(table, output_path)
    print(f"写入 {output_path}")


def write_grouped_outputs(
    grouped: OrderedDict[tuple[str, str], list[EnrichedRecord]],
) -> tuple[int, int]:
    written_files = 0
    written_rows = 0
    for (label, split), records in grouped.items():
        for idx, chunk_start in enumerate(range(0, len(records), SHARD_SIZE)):
            chunk = records[chunk_start : chunk_start + SHARD_SIZE]
            file_name = f"{sanitize_segment(label)}_{sanitize_segment(split)}_{idx}.parquet"
            output_path = Path(FINAL_REPO_DIR) / file_name
            write_enriched_parquet(output_path, chunk)
            written_files += 1
            written_rows += len(chunk)
    return written_files, written_rows


async def spk_convert() -> None:
    started = time.perf_counter()
    Path(FINAL_REPO_DIR).mkdir(parents=True, exist_ok=True)
    Path(DOWNLOAD_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    metadata_map = load_conversation_metadata()
    print(f"已加载 conversation 元数据: {len(metadata_map)} 条")

    grouped: OrderedDict[tuple[str, str], list[EnrichedRecord]] = OrderedDict()
    total_input_rows = 0
    missing_metadata_rows = 0
    total_grouped_rows = 0

    async with httpx.AsyncClient(follow_redirects=True, timeout=600.0) as client:
        for idx in range(INPUT_FILE_START, INPUT_FILE_END + 1):
            source_path = await download_source_parquet(client, idx)
            rows = read_source_parquet(source_path)
            total_input_rows += len(rows)
            for row in rows:
                meta = metadata_map.get(row.conversation_id)
                if meta is None:
                    missing_metadata_rows += 1
                    continue
                key = (meta.label, meta.split)
                grouped.setdefault(key, []).append(
                    EnrichedRecord(
                        conversation_id=row.conversation_id,
                        utterances_json=row.utterances_json,
                        audio_path=row.audio_path,
                        label=meta.label,
                        split=meta.split,
                        speaker_id=meta.speaker_id_json,
                    )
                )
                total_grouped_rows += 1

    written_files, written_rows = write_grouped_outputs(grouped)
    print("spk_convert 完成")
    print(f"输入 parquet 行数: {total_input_rows}")
    print(f"缺失元数据行数(已跳过): {missing_metadata_rows}")
    print(f"分组后待写出行数: {total_grouped_rows}")
    print(f"实际写出 parquet 行数: {written_rows}")
    print(f"实际写出 parquet 文件数: {written_files}")
    print(f"耗时: {time.perf_counter() - started:.2f}s")
    if written_rows != total_grouped_rows:
        raise RuntimeError(
            f"写出行数不一致: grouped={total_grouped_rows} written={written_rows}"
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(spk_convert())
