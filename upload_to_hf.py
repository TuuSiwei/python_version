#!/usr/bin/env python3
"""Upload files/folders to Hugging Face Hub (parity with py-scripts/upload_to_hf.py)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO_ID = "xhmm/si"
REPO_TYPE = "dataset"


def _repo_id() -> str:
    return os.environ.get("HF_REPO_ID", DEFAULT_REPO_ID)


def upload_file(file_path: str, path_in_repo: str) -> None:
    print(f"上传文件: {file_path} -> {path_in_repo}")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "未找到 HF_TOKEN 环境变量。请在 .env 中设置或运行: huggingface-cli login"
        )
    api = HfApi(token=token)
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=path_in_repo,
            repo_id=_repo_id(),
            repo_type=REPO_TYPE,
            commit_message=f"Upload {Path(file_path).name}",
        )
        print(f"✓ 上传成功: {path_in_repo}")
    except Exception as e:
        raise RuntimeError(f"上传失败: {e}") from e


def upload_folder(folder_path: str, path_in_repo: str) -> None:
    print(f"上传文件夹: {folder_path} -> {path_in_repo}")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "未找到 HF_TOKEN。请在 .env 中设置或运行: huggingface-cli login"
        )
    if path_in_repo not in ("", "."):
        raise RuntimeError(
            "upload_large_folder 不支持自定义 path_in_repo；远程路径请使用 '.'"
        )
    api = HfApi(token=token)
    try:
        print(f"\n开始上传到 {_repo_id()}...")
        api.upload_large_folder(
            folder_path=folder_path,
            repo_id=_repo_id(),
            repo_type=REPO_TYPE,
        )
        print("\n✓ 文件夹上传成功")
    except Exception as e:
        raise RuntimeError(f"上传失败: {e}") from e


def main() -> None:
    if len(sys.argv) < 3:
        print("用法:")
        print("  上传文件:   python upload_to_hf.py file <本地文件路径> <远程路径>")
        print("  上传文件夹: python upload_to_hf.py folder <本地文件夹路径> .")
        sys.exit(1)
    mode = sys.argv[1]
    local_path = sys.argv[2]
    remote_path = sys.argv[3] if len(sys.argv) > 3 else Path(local_path).name
    try:
        if mode == "file":
            upload_file(local_path, remote_path)
        elif mode == "folder":
            upload_folder(local_path, remote_path)
        else:
            print(f"错误: 未知模式 '{mode}'，请使用 'file' 或 'folder'", file=sys.stderr)
            sys.exit(1)
    except RuntimeError as e:
        print(f"✗ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
