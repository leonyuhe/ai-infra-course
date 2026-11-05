"""测量工具：环境快照、计时、显存与带宽估算。

所有章节共用同一套测量口径，否则跨章数字不可比：
- 计时前 ``sync(device)``（MPS/CUDA 都是异步执行，不同步会测出假快）
- warmup ≥ 3 次，取 5 次重复的中位数与 p90
- 结果统一用毫秒
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass

import psutil
import torch


# --------------------------------------------------------------------------- #
# 设备
# --------------------------------------------------------------------------- #
def pick_device(prefer: str | None = None) -> torch.device:
    """优先用显式指定；否则 mps > cuda > cpu。"""
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync(device: str | torch.device | None = None) -> None:
    """等待设备队列清空。计时前必须调用。"""
    dev = pick_device() if device is None else torch.device(device)
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# 计时
# --------------------------------------------------------------------------- #
@dataclass
class BenchResult:
    name: str
    median_ms: float
    p90_ms: float
    min_ms: float
    mean_ms: float
    reps: int

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:  # 方便直接在 Notebook 里 print
        return (
            f"{self.name}: median={self.median_ms:.2f}ms  p90={self.p90_ms:.2f}ms  "
            f"min={self.min_ms:.2f}ms  (reps={self.reps})"
        )


def bench(
    fn,
    *args,
    name: str = "op",
    warmup: int = 3,
    reps: int = 5,
    device: str | torch.device | None = None,
    **kwargs,
) -> BenchResult:
    """跑 fn(*args, **kwargs) 并返回毫秒统计。

    每个 repetition 内部只调用一次 fn。想测"平均每步"，请让 fn 内部循环。
    """
    for _ in range(warmup):
        fn(*args, **kwargs)
    sync(device)
    samples = []
    for _ in range(reps):
        sync(device)
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        sync(device)
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    p90 = samples[min(len(samples) - 1, int(len(samples) * 0.9))]
    return BenchResult(
        name=name,
        median_ms=statistics.median(samples),
        p90_ms=p90,
        min_ms=samples[0],
        mean_ms=statistics.fmean(samples),
        reps=reps,
    )


# --------------------------------------------------------------------------- #
# 环境快照
# --------------------------------------------------------------------------- #
def _sysctl(key: str, default: str = "unknown") -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except Exception:
        return default


def env_report() -> dict:
    """返回可直接显示成表格的环境信息。"""
    dev = pick_device()
    total_gb = psutil.virtual_memory().total / 1024**3
    info = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "chip": _sysctl("machdep.cpu.brand_string") if platform.system() == "Darwin" else platform.processor(),
        "memory_gb": round(total_gb, 1),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(dev),
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
        "cpu_count": psutil.cpu_count(logical=False),
    }
    if dev.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    return info


def print_env() -> dict:
    import pandas as pd

    info = env_report()
    for k, v in info.items():
        print(f"{k:10} {v}")
    return info


def mem_report() -> dict:
    """进程 RSS 与当前设备已分配显存（尽可能）。"""
    rss_gb = psutil.Process().memory_info().rss / 1024**3
    out = {"rss_gb": round(rss_gb, 3)}
    if torch.backends.mps.is_available():
        out["mps_alloc_gb"] = round(torch.mps.current_allocated_memory() / 1024**3, 3)
        out["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / 1024**3, 3)
    if torch.cuda.is_available():
        out["cuda_alloc_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
    return out


# --------------------------------------------------------------------------- #
# 估算公式：本课程反复使用
# --------------------------------------------------------------------------- #
def kv_cache_bytes(
    n_layer: int,
    n_kv_head: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> int:
    """KV cache 字节数：2 (K+V) × layers × heads × head_dim × seq × batch × dtype。"""
    return 2 * n_layer * n_kv_head * head_dim * seq_len * batch_size * dtype_bytes


def decode_tokens_per_sec(bandwidth_gbps: float, bytes_per_token: float) -> float:
    """decode 阶段的理论上限：总带宽 ÷ 每 token 需读取的字节。

    decode 时 batch=1、算术强度 ≈ 1 FLOP/byte，因此完全受显存带宽约束。
    """
    return bandwidth_gbps * 1e9 / bytes_per_token


def model_bytes_per_token(n_params: int, dtype_bytes: int = 2) -> int:
    """每生成 1 个 token，至少要读一遍全部权重。"""
    return n_params * dtype_bytes


def json_roundtrip(obj):
    """把结果落成 json 便于跨章节对比（outputs/ 下）。"""
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def human_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"
