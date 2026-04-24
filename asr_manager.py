"""Manage multi-GPU Qwen ASR worker processes and HTTP batch transcribe (Rust asr_service parity)."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


def _parse_env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _parse_env_u16(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


@dataclass
class TranscribeResponse:
    language: str | None
    text: str | None
    timestamps: list[dict[str, Any]] | None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "language": self.language,
            "text": self.text,
            "timestamps": self.timestamps,
        }
        if self.error is not None:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TranscribeResponse:
        return cls(
            language=d.get("language"),
            text=d.get("text"),
            timestamps=d.get("timestamps"),
            error=d.get("error"),
        )


def _log_pipe(name: str, pipe: Any) -> None:
    try:
        for line in iter(pipe.readline, b""):
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            print(f"[{name}] {text}", flush=True)
    finally:
        pipe.close()


@dataclass
class AsrWorker:
    id: int
    gpu_id: int
    base_url: str
    process: subprocess.Popen
    client: httpx.AsyncClient
    transcribe_lock: asyncio.Lock


class AsrService:
    def __init__(self, workers: list[AsrWorker], request_chunk_size: int) -> None:
        self.workers = workers
        self.request_chunk_size = request_chunk_size

    @classmethod
    async def start(cls) -> AsrService:
        gpu_count = max(1, _parse_env_int("ASR_GPU_COUNT", 1))
        gpu_start_index = _parse_env_int("ASR_GPU_START_INDEX", 0)
        base_port = _parse_env_u16("ASR_BASE_PORT", 8654)
        request_chunk_size = max(1, _parse_env_int("ASR_REQUEST_CHUNK_SIZE", 200))
        max_infer_batch = os.getenv("ASR_MAX_INFERENCE_BATCH_SIZE", "80")

        print(
            f"启动 ASR 多实例: gpu_count={gpu_count}, gpu_start_index={gpu_start_index}, "
            f"base_port={base_port}, request_chunk_size={request_chunk_size}, "
            f"max_inference_batch_size={max_infer_batch}"
        )

        script_path = Path(__file__).resolve().parent / "asr_service.py"
        workers: list[AsrWorker] = []

        for worker_id in range(gpu_count):
            gpu_id = gpu_start_index + worker_id
            worker_port = base_port + worker_id
            if worker_port > 65535:
                raise ValueError("ASR_BASE_PORT + ASR_GPU_COUNT 超出端口范围")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["ASR_MAX_INFERENCE_BATCH_SIZE"] = str(max_infer_batch)

            proc = subprocess.Popen(
                [sys.executable, str(script_path), str(worker_port)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            if proc.stdout:
                threading.Thread(
                    target=_log_pipe,
                    args=(f"asr_worker_{worker_id}_stdout", proc.stdout),
                    daemon=True,
                ).start()
            if proc.stderr:
                threading.Thread(
                    target=_log_pipe,
                    args=(f"asr_worker_{worker_id}_stderr", proc.stderr),
                    daemon=True,
                ).start()

            base_url = f"http://127.0.0.1:{worker_port}"
            client = httpx.AsyncClient(timeout=httpx.Timeout(3600.0))
            workers.append(
                AsrWorker(
                    id=worker_id,
                    gpu_id=gpu_id,
                    base_url=base_url,
                    process=proc,
                    client=client,
                    transcribe_lock=asyncio.Lock(),
                )
            )

        for w in workers:
            await _wait_for_health(w.client, w.base_url, w.id)
            print(f"ASR worker {w.id} 已就绪: gpu={w.gpu_id}, url={w.base_url}")

        return cls(workers, request_chunk_size)

    async def kill(self) -> None:
        for w in self.workers:
            if w.process.poll() is None:
                try:
                    os.killpg(w.process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    w.process.terminate()
            try:
                await asyncio.to_thread(lambda: w.process.wait(timeout=120))
            except Exception:
                pass
            await w.client.aclose()
            print(f"ASR worker {w.id} 已关闭 (gpu={w.gpu_id})")

    @staticmethod
    async def _transcribe_batch_on_worker(
        worker: AsrWorker,
        audio_paths: list[str],
    ) -> list[tuple[str, TranscribeResponse]]:
        t0 = time.perf_counter()
        async with worker.transcribe_lock:
            url = f"{worker.base_url}/transcribe"
            r = await worker.client.post(url, json={"audio_paths": audio_paths})
            if r.status_code < 200 or r.status_code >= 300:
                raise RuntimeError(f"批量转录请求失败，状态码: {r.status_code}")
            batch = r.json()
            if batch.get("error"):
                raise RuntimeError(f"批量转录失败: {batch['error']}")
            items = batch.get("results")
            if items is None:
                raise RuntimeError("批量转录响应缺少 results 字段")
            out: list[tuple[str, TranscribeResponse]] = []
            for item in items:
                resp = TranscribeResponse(
                    language=item.get("language"),
                    text=item.get("text"),
                    timestamps=item.get("timestamps"),
                    error=None,
                )
                out.append((item["audio_path"], resp))
        dt = time.perf_counter() - t0
        print(
            f"ASR 批量转录完成: url={worker.base_url}, 文件数={len(audio_paths)}, 耗时: {dt:.1f}s"
        )
        return out

    async def transcribe_batch_cached(self, audio_paths: list[str]) -> dict[str, TranscribeResponse]:
        results: dict[str, TranscribeResponse] = {}
        uncached: list[str] = []
        for path in audio_paths:
            cache_path = path.replace(".wav", ".asr.json")
            if os.path.isfile(cache_path):
                try:
                    import json

                    with open(cache_path, encoding="utf-8") as f:
                        data = json.load(f)
                    results[path] = TranscribeResponse.from_dict(data)
                    continue
                except Exception:
                    pass
            uncached.append(path)

        if not uncached:
            return results

        print(
            f"ASR 缓存未命中 {len(uncached)} 个文件，开始并行转录（worker={len(self.workers)}, "
            f"chunk={self.request_chunk_size}）..."
        )
        if not self.workers:
            raise RuntimeError("没有可用的 ASR worker")

        buckets: list[list[str]] = [[] for _ in self.workers]
        for i, p in enumerate(uncached):
            buckets[i % len(self.workers)].append(p)

        tasks: list[asyncio.Task] = []
        for worker_idx, bucket in enumerate(buckets):
            if not bucket:
                continue
            worker = self.workers[worker_idx]
            for i in range(0, len(bucket), self.request_chunk_size):
                chunk = bucket[i : i + self.request_chunk_size]
                tasks.append(
                    asyncio.create_task(self._transcribe_batch_on_worker(worker, chunk))
                )

        import json

        for t in asyncio.as_completed(tasks):
            batch_results = await t
            for audio_path, resp in batch_results:
                cache_path = audio_path.replace(".wav", ".asr.json")
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(resp.to_json(), f, ensure_ascii=False)
                results[audio_path] = resp

        return results


async def _wait_for_health(client: httpx.AsyncClient, base_url: str, worker_id: int) -> None:
    health_url = f"{base_url}/health"
    for _ in range(120):
        print(f"检测 ASR worker {worker_id} 就绪中...")
        try:
            r = await client.get(health_url)
            if 200 <= r.status_code < 300:
                return
        except Exception:
            pass
        await asyncio.sleep(1.0)
    raise RuntimeError(f"ASR worker {worker_id} 启动超时: {base_url}")
