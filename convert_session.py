"""Convert filelist.csv to conversations.json (Rust convert_session parity)."""

from __future__ import annotations

import csv
import json
import resource
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

INPUT_CSV = "filelist.csv"
OUTPUT_JSON = "conversations.json"


@dataclass
class FileListRecord:
    file_id: str
    label: str
    split: str
    batch_idx: int
    archive_idx: int


@dataclass
class FileEntry:
    file_id: str
    label: str
    split: str
    batch_idx: int
    archive_idx: int
    participant_id: str
    participant_id_idx: int = 0


@dataclass
class Conversation:
    conversation_id: str
    files: list[FileEntry] = field(default_factory=list)

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "files": [
                {
                    "file_id": f.file_id,
                    "label": f.label,
                    "split": f.split,
                    "batch_idx": f.batch_idx,
                    "archive_idx": f.archive_idx,
                    "participant_id": f.participant_id,
                    "participant_id_idx": f.participant_id_idx,
                }
                for f in self.files
            ],
        }


def extract_version_id(file_id: str) -> str | None:
    for part in file_id.split("_"):
        if part.startswith("V") and len(part) > 1:
            return part
    return None


def extract_session_id(file_id: str) -> str | None:
    for part in file_id.split("_"):
        if part.startswith("S") and len(part) > 1:
            return part
    return None


def extract_interaction_id(file_id: str) -> str | None:
    for part in file_id.split("_"):
        if part.startswith("I") and len(part) > 1:
            return part
    return None


def extract_participant(file_id: str) -> str | None:
    for part in file_id.split("_"):
        if part.startswith("P") and len(part) > 1:
            return part[1:]
    return None


def current_rss_bytes() -> int | None:
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # Linux: ru_maxrss in KB; macOS: bytes
        if sys.platform == "darwin":
            return int(ru.ru_maxrss)
        return int(ru.ru_maxrss) * 1024
    except Exception:
        return None


def validate_sessions_file(
    path: str,
    expected_total_files: int,
    expected_counts: OrderedDict[str, int],
) -> bool:
    with open(path, encoding="utf-8") as f:
        conversations: list[dict] = json.load(f)

    actual_counts: OrderedDict[str, int] = OrderedDict()
    for c in conversations:
        cid = c["conversation_id"]
        actual_counts[cid] = len(c["files"])

    actual_total_files = sum(actual_counts.values())

    extra: list[str] = []
    missing: list[str] = []
    mismatches: list[tuple[str, int, int]] = []

    for cid, ac in actual_counts.items():
        if cid not in expected_counts:
            extra.append(cid)
        elif expected_counts[cid] != ac:
            mismatches.append((cid, expected_counts[cid], ac))

    for cid in expected_counts:
        if cid not in actual_counts:
            missing.append(cid)

    ok = True
    if actual_total_files != expected_total_files:
        ok = False
        print(
            f"验证失败: files 总数不一致，期望 {expected_total_files}，实际 {actual_total_files}",
            file=sys.stderr,
        )
    if len(actual_counts) != len(expected_counts):
        ok = False
        print(
            f"验证失败: session 数量不一致，期望 {len(expected_counts)}，实际 {len(actual_counts)}",
            file=sys.stderr,
        )
    if extra:
        ok = False
        print(
            f"验证失败: 发现多余 session（展示前 5 个）: {extra[:5]}",
            file=sys.stderr,
        )
    if missing:
        ok = False
        print(
            f"验证失败: 发现缺失 session（展示前 5 个）: {missing[:5]}",
            file=sys.stderr,
        )
    if mismatches:
        ok = False
        print("验证失败: session 文件数不一致（展示前 5 个）:", file=sys.stderr)
        for cid, exp, act in mismatches[:5]:
            print(f"  {cid}: 期望 {exp}，实际 {act}", file=sys.stderr)

    if ok:
        print("验证通过: conversations.json 与 CSV 统计一致")
    return ok


def convert_session() -> None:
    start_total = time.perf_counter()
    grouped: OrderedDict[str, list[FileEntry]] = OrderedDict()
    total_files = 0

    with open(INPUT_CSV, newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            record = FileListRecord(
                file_id=row["file_id"],
                label=row["label"],
                split=row["split"],
                batch_idx=int(row["batch_idx"]),
                archive_idx=int(row["archive_idx"]),
            )
            vid = extract_version_id(record.file_id)
            sid = extract_session_id(record.file_id)
            iid = extract_interaction_id(record.file_id)
            part = extract_participant(record.file_id)
            if vid is None:
                print(f"无法从 file_id 解析 version_id: {record.file_id}", file=sys.stderr)
                continue
            if sid is None:
                print(f"无法从 file_id 解析 session_id: {record.file_id}", file=sys.stderr)
                continue
            if iid is None:
                print(f"无法从 file_id 解析 interaction_id: {record.file_id}", file=sys.stderr)
                continue
            if part is None:
                print(f"无法从 file_id 解析 participant: {record.file_id}", file=sys.stderr)
                continue

            entry = FileEntry(
                file_id=record.file_id,
                label=record.label,
                split=record.split,
                batch_idx=record.batch_idx,
                archive_idx=record.archive_idx,
                participant_id=part,
            )
            cid = f"{vid}_{sid}_{iid}"
            grouped.setdefault(cid, []).append(entry)
            total_files += 1

    expected_counts: OrderedDict[str, int] = OrderedDict(
        (k, len(v)) for k, v in grouped.items()
    )
    read_elapsed = time.perf_counter() - start_total
    rss_after_read = current_rss_bytes()

    conversations: list[Conversation] = []
    for conversation_id, files in grouped.items():
        unique_participants = sorted({f.participant_id for f in files})
        participant_to_idx = {pid: idx for idx, pid in enumerate(unique_participants)}
        for f in files:
            f.participant_id_idx = participant_to_idx[f.participant_id]
        conversations.append(Conversation(conversation_id=conversation_id, files=files))

    with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
        json.dump(
            [c.to_json_obj() for c in conversations],
            out,
            ensure_ascii=False,
            indent=2,
        )

    total_elapsed = time.perf_counter() - start_total
    rss_after_write = current_rss_bytes()

    print(f"转换完成: conversations={len(conversations)} files={total_files}")
    print(f"读取并分组耗时: {read_elapsed:.3f}s")
    print(f"总耗时: {total_elapsed:.3f}s")
    if rss_after_read is not None:
        print(f"读取后RSS: {rss_after_read / (1024 * 1024):.2f} MB")
    if rss_after_write is not None:
        print(f"写出后RSS: {rss_after_write / (1024 * 1024):.2f} MB")

    validate_sessions_file(OUTPUT_JSON, total_files, expected_counts)


if __name__ == "__main__":
    convert_session()
