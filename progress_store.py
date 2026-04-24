"""Group-level progress JSON (final/group_progress.json)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from io_utils import atomic_write_json

GROUP_PROGRESS_FILE = Path("final/group_progress.json")


@dataclass
class GroupState:
    group_idx: int
    conversation_ids: list[str]
    conversation_count: int
    utterance_count: int
    generated: bool
    uploaded: bool
    cleaned: bool
    timestamp: str


@dataclass
class GroupProgress:
    groups: list[GroupState] = field(default_factory=list)
    last_updated: str = ""
    group_count: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GroupProgress:
        groups = [
            GroupState(
                group_idx=g["group_idx"],
                conversation_ids=list(g["conversation_ids"]),
                conversation_count=int(g["conversation_count"]),
                utterance_count=int(g["utterance_count"]),
                generated=bool(g["generated"]),
                uploaded=bool(g["uploaded"]),
                cleaned=bool(g["cleaned"]),
                timestamp=str(g["timestamp"]),
            )
            for g in d.get("groups", [])
        ]
        return cls(
            groups=groups,
            last_updated=str(d.get("last_updated", "")),
            group_count=int(d.get("group_count", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "groups": [asdict(g) for g in self.groups],
            "last_updated": self.last_updated,
            "group_count": self.group_count,
        }

    def find_group(self, group_idx: int) -> GroupState | None:
        for g in self.groups:
            if g.group_idx == group_idx:
                return g
        return None

    def update_or_insert(self, state: GroupState) -> None:
        for i, g in enumerate(self.groups):
            if g.group_idx == state.group_idx:
                self.groups[i] = state
                return
        self.groups.append(state)


def load_group_progress() -> GroupProgress:
    path = GROUP_PROGRESS_FILE
    if not path.exists():
        return GroupProgress()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    gp = GroupProgress.from_dict(data)
    print(f"✓ 加载组进度: {len(gp.groups)} 个组已完成")
    return gp


def save_group_progress(progress: GroupProgress) -> None:
    GROUP_PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    progress.last_updated = datetime.now().astimezone().isoformat()
    atomic_write_json(GROUP_PROGRESS_FILE, progress.to_dict(), indent=2)
