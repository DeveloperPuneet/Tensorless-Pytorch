"""Hardware auto-detection.

Preference order is TPU -> GPU -> CPU, but selection is "intelligent"
rather than blind: we verify each backend is actually usable (not just
importable) before choosing it, and we fall back gracefully -- including
at *runtime*, if a chosen device turns out to error out mid-training, the
trainer (see `training/trainer.py`) will catch that and fall back too.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch


def _tpu_available() -> bool:
    try:
        import torch_xla.core.xla_model as xm  # noqa: F401

        return True
    except Exception:
        return False


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        return torch.backends.mps.is_available()
    except Exception:
        return False


def _cuda_supports_bf16() -> bool:
    """`torch.cuda.is_bf16_supported()` returns True as soon as bf16 ops
    can be *executed* at all, which is true even on GPUs (like the T4)
    that have no native bf16 tensor-core support and just emulate it --
    training still runs, but slower, and torch.compile's inductor backend
    will skip bf16 codegen and print a warning for every kernel. Real
    accelerated bf16 needs Ampere or newer (compute capability >= 8.0),
    so check that directly instead.
    """
    try:
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 8
    except Exception:
        return False


def auto_select_device(user_device: Optional[str], user_precision: Optional[str]) -> Tuple[str, str]:
    """Resolve the device and precision to use.

    `user_device` / `user_precision` are honored if given (with a
    graceful downgrade if the requested device isn't actually available).
    Otherwise we pick automatically: tpu > cuda > mps > cpu.
    """
    if user_device is not None:
        device = user_device
        if device == "tpu" and not _tpu_available():
            device = "cuda" if _cuda_available() else "cpu"
        elif device == "cuda" and not _cuda_available():
            device = "cpu"
        elif device == "mps" and not _mps_available():
            device = "cpu"
    else:
        if _tpu_available():
            device = "tpu"
        elif _cuda_available():
            device = "cuda"
        elif _mps_available():
            device = "mps"
        else:
            device = "cpu"

    if user_precision is not None:
        precision = user_precision
    else:
        if device == "cuda" and _cuda_supports_bf16():
            precision = "bf16"
        elif device == "cuda":
            precision = "fp16"
        elif device == "tpu":
            precision = "bf16"
        else:
            # CPU and MPS: stick to fp32 for correctness/stability by default.
            precision = "fp32"

    return device, precision


def tpu_core_count() -> int:
    """Number of TPU cores visible to this process (1 if not on TPU)."""
    try:
        import torch_xla.core.xla_model as xm

        return xm.xrt_world_size()
    except Exception:
        try:
            import torch_xla.runtime as xr

            return xr.world_size()
        except Exception:
            return 1


def mark_step(device: torch.device) -> None:
    """XLA/TPU tensors are lazily evaluated -- without periodically calling
    `xm.mark_step()` the compute graph never actually executes and a
    'TPU-enabled' training run will just silently build up an ever-larger
    graph (or hang) instead of doing real work. This is a no-op on every
    other device, so it's safe to call unconditionally from the trainer.
    """
    if device.type == "xla":
        try:
            import torch_xla.core.xla_model as xm

            xm.mark_step()
        except Exception:
            pass


def get_device_info(device: torch.device) -> Dict[str, Any]:
    """Human-readable + programmatic info about the active compute device,
    used both for the startup log line and for memory-aware auto-sizing.
    """
    info: Dict[str, Any] = {"type": device.type, "name": device.type, "memory_gb": None, "num_devices": 1}
    try:
        if device.type == "cuda":
            idx = device.index if device.index is not None else 0
            props = torch.cuda.get_device_properties(idx)
            info["name"] = props.name
            info["memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
            info["num_devices"] = torch.cuda.device_count()
            info["bf16"] = _cuda_supports_bf16()
        elif device.type == "xla":
            info["name"] = "TPU"
            info["num_devices"] = tpu_core_count()
        elif device.type == "mps":
            info["name"] = "Apple Silicon (MPS)"
        else:
            info["name"] = "CPU"
            info["num_devices"] = os.cpu_count() or 1
    except Exception:
        pass
    return info


def recommend_max_params(device: torch.device) -> int:
    """Rough ceiling on trainable parameter count so an auto-configured
    model has a reasonable chance of fitting in memory alongside AdamW's
    optimizer state (~2x params) and activations. Deliberately
    conservative -- users can always override architecture args by hand.
    """
    info = get_device_info(device)
    if device.type == "cuda" and info.get("memory_gb"):
        # Rule of thumb for AdamW + fp16/bf16 mixed precision + activations:
        # budget roughly 1M params per 15MB of GPU memory, capped well
        # inside the "upper-mid" tier regardless of how much memory exists.
        budget = int(info["memory_gb"] * 1024 * (1_000_000 / 15))
        return max(2_000_000, min(budget, 350_000_000))
    if device.type == "xla":
        # TPU v3/v4 cores typically expose >=16GB HBM each.
        return 350_000_000
    if device.type == "mps":
        return 60_000_000
    # CPU: keep it trainable in a reasonable amount of wall-clock time.
    return 40_000_000


def get_torch_device(device: str) -> torch.device:
    """Convert our string device name into a torch.device, with a
    runtime fallback to CPU if the requested backend is unavailable.
    """
    try:
        if device == "tpu":
            import torch_xla.core.xla_model as xm

            return xm.xla_device()
        if device == "cuda":
            if not _cuda_available():
                return torch.device("cpu")
            return torch.device("cuda")
        if device == "mps":
            if not _mps_available():
                return torch.device("cpu")
            return torch.device("mps")
        return torch.device("cpu")
    except Exception:
        return torch.device("cpu")
