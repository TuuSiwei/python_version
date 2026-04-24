#!/usr/bin/env python3
"""Unified CLI for SI conversation WAV + RTTM dataset processing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure sibling modules resolve when run as `python main.py` from this directory
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env from cwd or repo root
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _cmd_convert_session(_args: argparse.Namespace) -> int:
    from convert_session import convert_session

    convert_session()
    return 0


def _cmd_download_transform(args: argparse.Namespace) -> int:
    import asyncio

    from si_rttm_pipeline import SiRttmParams, run_si_rttm_pipeline

    params = SiRttmParams(
        repo_id=args.repo_id,
        config_prefix=args.config_prefix,
        chunk_size=args.chunk_size,
        upload=not args.no_upload,
        download_concurrency=args.download_concurrency,
        process_concurrency=args.process_concurrency,
        max_shards=args.max_shards,
        start_from_conversation=args.start_from_conversation,
        stop_after_conversation_count=args.stop_after_conversation_count,
        window_sec=args.window_sec,
        shift_sec=args.shift_sec,
        max_spks=args.max_spks,
        feat_per_sec=args.feat_per_sec,
    )
    asyncio.run(run_si_rttm_pipeline(params))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SI conversation WAV + RTTM dataset pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cs = sub.add_parser("convert-session", help="filelist.csv -> conversations.json")
    p_cs.set_defaults(func=_cmd_convert_session)

    p_dt = sub.add_parser(
        "download-transform",
        help="Download SI conversations -> mixed 48 kHz WAV + RTTM -> HF Dataset configs",
    )
    p_dt.add_argument("--repo-id", default="humanify/real_dia_dataset")
    p_dt.add_argument("--config-prefix", default="si")
    p_dt.add_argument("--chunk-size", type=int, default=5000)
    p_dt.add_argument("--download-concurrency", type=int, default=64)
    p_dt.add_argument("--process-concurrency", type=int, default=16)
    p_dt.add_argument("--max-shards", type=int, default=128)
    p_dt.add_argument("--start-from-conversation", type=int, default=None)
    p_dt.add_argument("--stop-after-conversation-count", type=int, default=None)
    p_dt.add_argument("--no-upload", action="store_true", help="Build locally without pushing to HF")
    p_dt.add_argument("--window-sec", type=float, default=90.0)
    p_dt.add_argument("--shift-sec", type=float, default=8.0)
    p_dt.add_argument("--max-spks", type=int, default=4)
    p_dt.add_argument("--feat-per-sec", type=int, default=25)
    p_dt.set_defaults(func=_cmd_download_transform)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    # Match Rust: exit on unhandled thread errors in strict environments
    sys.exit(main())
