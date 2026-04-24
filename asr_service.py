"""Qwen3 ASR Flask service (parity with py-scripts/asr.py)."""

from __future__ import annotations

import gc
import os
import signal
import sys
import traceback

import torch
from flask import Flask, jsonify, request
from qwen_asr import Qwen3ASRModel


def _setup_signal_handlers() -> None:
    def _handle_signal(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"收到信号 {sig_name}，正在退出...", file=sys.stderr, flush=True)
        os._exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)


LANGUAGE = "English"
DEVICE = "cuda:0"


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("must be positive")
        return parsed
    except ValueError:
        print(
            f"环境变量 {name}={value} 非法，回退默认值 {default}",
            file=sys.stderr,
            flush=True,
        )
        return default


MAX_INFERENCE_BATCH_SIZE = _get_env_int("ASR_MAX_INFERENCE_BATCH_SIZE", 5)


def transcribe_with_retry(model, audio_paths, language, min_batch=1):
    """带 OOM 自适应重试的转录，对半拆分直到成功或单文件 OOM"""
    try:
        return model.transcribe(
            audio=audio_paths, language=language, return_time_stamps=True
        )
    except torch.OutOfMemoryError as e:
        e.__traceback__ = None
        del e
        gc.collect()
        torch.cuda.empty_cache()
        if len(audio_paths) <= min_batch:
            raise
        mid = len(audio_paths) // 2
        freed = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(
            f"CUDA OOM，拆分 batch: {len(audio_paths)} -> {mid} + {len(audio_paths) - mid}"
            f"（已用 {freed:.1f}G / 总共 {total:.1f}G）",
            file=sys.stderr,
            flush=True,
        )
        left = transcribe_with_retry(model, audio_paths[:mid], language, min_batch)
        right = transcribe_with_retry(model, audio_paths[mid:], language, min_batch)
        return left + right


def resolve_device(preferred_device: str) -> str:
    if preferred_device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA 不可用，自动回退到 CPU（速度会显著变慢）。", file=sys.stderr)
            return "cpu"
        device_idx = preferred_device.split(":")[-1] if ":" in preferred_device else "0"
        try:
            device_idx_int = int(device_idx)
            if device_idx_int >= torch.cuda.device_count():
                print(
                    f"指定的 GPU {preferred_device} 不存在 "
                    f"(共 {torch.cuda.device_count()} 块)，回退到 cuda:0。",
                    file=sys.stderr,
                )
                return "cuda:0"
        except ValueError:
            print(f"无法解析设备索引 '{device_idx}'，回退到 cuda:0。", file=sys.stderr)
            return "cuda:0"
    return preferred_device


def load_model(device: str):
    print("加载模型...", flush=True)
    try:
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        if device.startswith("cuda"):
            try:
                import flash_attn  # noqa: F401

                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "eager"
        else:
            attn_impl = "eager"

        print("attn_impl: ", attn_impl, flush=True)
        print("max_inference_batch_size: ", MAX_INFERENCE_BATCH_SIZE, flush=True)
        model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-1.7B",
            dtype=dtype,
            device_map=device,
            attn_implementation=attn_impl,
            max_inference_batch_size=MAX_INFERENCE_BATCH_SIZE,
            max_new_tokens=1024,
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
            forced_aligner_kwargs=dict(
                dtype=dtype,
                device_map=device,
                attn_implementation=attn_impl,
            ),
        )
        print("模型加载成功。", flush=True)
        return model
    except torch.OutOfMemoryError:
        print(
            "GPU 显存不足，无法加载模型。请尝试使用更小的模型或释放 GPU 内存。",
            file=sys.stderr,
        )
        raise
    except OSError as e:
        print(f"模型文件加载失败（可能未下载或路径错误）: {e}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"模型加载时发生未知错误: {e}\n{traceback.format_exc()}", file=sys.stderr)
        raise


model = None
app = Flask(__name__)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        data = request.json
        if not data or "audio_paths" not in data:
            return jsonify({"error": "missing audio_paths"}), 400

        audio_paths = data["audio_paths"]
        if not isinstance(audio_paths, list) or len(audio_paths) == 0:
            return jsonify({"error": "audio_paths must be a non-empty list"}), 400

        print(f"批量转录 {len(audio_paths)} 个文件...", flush=True)
        results = transcribe_with_retry(model, audio_paths, LANGUAGE)

        response_results = []
        for audio_path, result in zip(audio_paths, results):
            item = {
                "audio_path": audio_path,
                "language": result.language,
                "text": result.text,
            }
            if hasattr(result, "time_stamps") and result.time_stamps:
                ts = result.time_stamps
                items = ts.items if hasattr(ts, "items") else ts
                item["timestamps"] = [
                    {
                        "text": it.text,
                        "start_time": it.start_time,
                        "end_time": it.end_time,
                    }
                    for it in items
                ]
            response_results.append(item)

        print(f"批量转录完成，共 {len(response_results)} 个结果", flush=True)
        return jsonify({"results": response_results})

    except Exception as e:
        print(f"转录失败: {e}\n{traceback.format_exc()}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def run_server(port: int) -> None:
    """Entry point for `python main.py asr-server <port>`."""
    global model
    _setup_signal_handlers()
    if not (1024 <= port <= 65535):
        print(f"错误: 端口号必须在 1024-65535 之间: {port}", file=sys.stderr)
        sys.exit(1)
    device = resolve_device(DEVICE)
    print(f"使用设备: {device}", flush=True)
    try:
        model = load_model(device)
        model.forced_aligner.aligner_processor.is_kept_char = lambda x: True
    except Exception:
        print("模型加载失败，程序终止。", file=sys.stderr)
        sys.exit(1)
    print(f"启动 Flask 服务，端口: {port}", file=sys.stderr, flush=True)
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


def main_cli() -> None:
    _setup_signal_handlers()
    try:
        print("asr_service.py start running")
        if len(sys.argv) < 2:
            print("错误: 缺少端口号参数", file=sys.stderr)
            print("用法: python asr_service.py <port>", file=sys.stderr)
            sys.exit(1)
        run_server(int(sys.argv[1]))
    except KeyboardInterrupt:
        print("用户强制中断 (KeyboardInterrupt)。", file=sys.stderr)
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        print(
            f"程序发生未捕获的致命错误: {e}\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
