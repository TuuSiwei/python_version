"""Atomic I/O helpers and file tree copy (hardlink with copy fallback)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


def atomic_write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def link_or_copy_dir_recursive(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"目录不存在，无法构建快照: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for entry in os.scandir(src):
        s = Path(entry.path)
        d = dst / entry.name
        if entry.is_dir(follow_symlinks=False):
            link_or_copy_dir_recursive(s, d)
        elif entry.is_file(follow_symlinks=False):
            link_or_copy_file(s, d)
