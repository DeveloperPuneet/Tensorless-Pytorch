"""Automatic configuration.

Turns a `Dataset` + a user-supplied `TrainConfig` (with mostly-None fields)
into a fully-resolved `ResolvedConfig`. Every automatic choice is driven by
simple, explainable heuristics based on dataset size and available
hardware -- this is not meant to be state-of-the-art NAS, it's meant to
produce a *sane, working* default so `tl.train("./data")` just works,
scaling all the way from tiny CPU-only experiments up to "upper-mid"
scale models (roughly 100M-350M parameters for transformers) when the
data and hardware support it.

Model-size tiers (approximate parameter counts, transformer LM):
    small       < 10M
    lower-mid   10M  - 50M
    upper-mid   50M  - 350M   <- Tensorless auto-config tops out here
    large       350M - 3B     (never auto-selected; pass explicit
                                d_model=/layers=/etc. yourself)
    extreme     3B+           (out of scope for this project)
"""

from __future__ import annotations

from typing import Tuple

from ..config import ResolvedConfig, TrainConfig
from ..data.loader import Dataset
from ..devices.device import auto_select_device, get_device_info, get_torch_device, recommend_max_params
from .detector import detect_task

# ---------------------------------------------------------------------------
# Architecture sizing
# ---------------------------------------------------------------------------

def _text_model_tier(n_examples: int) -> Tuple[int, int, int, int]:
    """Return (d_model, layers, heads, ff_mult), scaled up to the
    "upper-mid" ceiling as effective dataset size grows. `ff_mult` is
    interpreted by the v2 (SwiGLU) architecture as a multiplier on the
    gated hidden width, same as v1's plain MLP hidden width.
    """
    if n_examples < 200:
        return 64, 2, 2, 2          # tiny / smoke-test scale
    elif n_examples < 5_000:
        return 128, 4, 4, 3         # small
    elif n_examples < 50_000:
        return 256, 6, 8, 4         # small / lower-mid boundary
    elif n_examples < 200_000:
        return 384, 8, 8, 4         # lower-mid
    elif n_examples < 1_000_000:
        return 640, 12, 10, 4       # lower-mid / upper-mid boundary (~80M params)
    else:
        return 1024, 16, 16, 4      # upper-mid ceiling (~250-300M params)


def _tabular_model_tier(n_examples: int) -> Tuple[int, int, int, int]:
    if n_examples < 500:
        return 32, 2, 1, 2
    elif n_examples < 20_000:
        return 64, 3, 1, 2
    elif n_examples < 200_000:
        return 128, 5, 1, 2
    else:
        return 256, 8, 1, 2         # upper-mid ceiling for tabular data


def _estimate_transformer_params(vocab_size: int, d_model: int, layers: int, ff_mult: int) -> int:
    swiglu_hidden = max(int(d_model * ff_mult * (2 / 3)), d_model)
    per_layer = 4 * d_model * d_model + 3 * d_model * swiglu_hidden  # attn proj + swiglu gate/up/down
    return vocab_size * d_model + layers * per_layer


def _fit_within_param_budget(
    vocab_size: int, d_model: int, layers: int, heads: int, ff_mult: int, max_params: int
) -> Tuple[int, int, int, int]:
    """If the tier-selected architecture would exceed what the detected
    hardware can comfortably hold, scale it back down -- first by
    trimming depth, then width -- until it fits. Keeps `d_model`
    divisible by `heads` throughout.
    """
    while _estimate_transformer_params(vocab_size, d_model, layers, ff_mult) > max_params and layers > 2:
        layers -= 1
    while _estimate_transformer_params(vocab_size, d_model, layers, ff_mult) > max_params and d_model > 64:
        d_model = max(64, d_model - 64)
        while d_model % heads != 0 and heads > 1:
            heads -= 1
    return d_model, layers, heads, ff_mult


def _auto_batch_size(n_examples: int, max_seq_len: int, device: str, num_devices: int = 1) -> int:
    # GPUs/TPUs benefit from a larger token budget per step than CPU.
    token_budget = 32768 if device in ("cuda", "tpu") else 8192
    sequence_batch = max(1, token_budget // max_seq_len)
    if n_examples < 200:
        base = min(8, sequence_batch)
    elif n_examples < 2000:
        base = min(16, sequence_batch)
    elif n_examples < 20000:
        base = min(32, sequence_batch)
    elif n_examples < 200000:
        base = min(64, sequence_batch)
    else:
        base = min(128, sequence_batch)
    # When training will be split across multiple GPUs (DataParallel or
    # DDP), scale the global batch up so each GPU still gets a
    # reasonably-sized per-device batch instead of shrinking as GPUs are
    # added.
    return base * max(1, num_devices)


def _auto_epochs(n_examples: int) -> int:
    if n_examples < 200:
        return 40
    elif n_examples < 2000:
        return 20
    elif n_examples < 20000:
        return 10
    else:
        return 5


def _effective_text_size(ds: Dataset) -> int:
    """Estimate useful training examples for raw corpora."""
    if ds.kind in ("text", "text_labeled"):
        return max(len(ds), sum(len(text) for text in ds.texts) // 200)
    return len(ds)


def _auto_vocab_size(ds: Dataset, n_examples: int) -> int:
    if ds.kind not in ("text", "text_labeled"):
        return 1000
    unique_chars = len(set("".join(ds.texts)))
    # Bigger corpora justify (and benefit from) a bigger BPE vocabulary --
    # fewer tokens per sequence, more of max_seq_len spent on real context.
    if n_examples < 5_000:
        ceiling = 2048
    elif n_examples < 200_000:
        ceiling = 8192
    else:
        ceiling = 32000
    return min(ceiling, max(64, unique_chars * 8))


def _auto_num_workers(device: str) -> int:
    import os

    cpu_count = os.cpu_count() or 1
    if device in ("cuda", "tpu"):
        # Leave headroom for the main process / accelerator driver threads.
        return max(0, min(4, cpu_count - 1))
    return 0  # CPU training: extra worker processes just add overhead


def resolve_config(ds: Dataset, user: TrainConfig) -> ResolvedConfig:
    n = _effective_text_size(ds)
    task = user.task or detect_task(ds)
    model_type = user.model_type or (
        "transformer" if task in ("text-generation", "text-classification") else "mlp"
    )
    tokenizer = user.tokenizer or "bpe"
    if tokenizer not in ("char", "bpe"):
        raise ValueError("tokenizer must be 'char' or 'bpe'")

    architecture = user.architecture or "v2"
    if architecture not in ("v1", "v2"):
        raise ValueError("architecture must be 'v1' or 'v2'")

    device, precision = auto_select_device(user.device, user.precision)
    torch_device = get_torch_device(device)
    device_info = get_device_info(torch_device)
    num_gpus = device_info.get("num_devices", 1) if device == "cuda" else 1

    multi_gpu = user.multi_gpu
    if multi_gpu is None:
        multi_gpu = device == "cuda" and num_gpus > 1

    if model_type == "transformer":
        d_model, layers, heads, ff_mult = _text_model_tier(n)
    else:
        d_model, layers, heads, ff_mult = _tabular_model_tier(n)

    max_seq_len = user.max_seq_len or (256 if ds.kind in ("text", "text_labeled") else 1)
    bpe_vocab_size = (
        user.bpe_vocab_size if user.bpe_vocab_size is not None else _auto_vocab_size(ds, n)
    )

    # Memory-aware cap: only clamp automatically-chosen sizes, never a
    # user's explicit d_model=/layers=/heads=/ff_mult= override.
    user_set_arch = any(
        v is not None for v in (user.d_model, user.layers, user.heads, user.ff_mult)
    )
    if model_type == "transformer" and not user_set_arch:
        max_params = recommend_max_params(torch_device)
        d_model, layers, heads, ff_mult = _fit_within_param_budget(
            bpe_vocab_size if tokenizer == "bpe" else 256, d_model, layers, heads, ff_mult, max_params
        )

    gradient_checkpointing = user.gradient_checkpointing
    if gradient_checkpointing is None:
        # Auto-enable on GPU/TPU once the model is large enough that
        # activation memory (not parameter memory) becomes the bottleneck.
        approx_params = _estimate_transformer_params(
            bpe_vocab_size if tokenizer == "bpe" else 256, d_model, layers, ff_mult
        ) if model_type == "transformer" else 0
        gradient_checkpointing = device in ("cuda", "tpu") and approx_params > 60_000_000

    compile_model = user.compile
    if compile_model is None:
        compile_model = device == "cuda"

    num_workers = user.num_workers if user.num_workers is not None else _auto_num_workers(device)

    out = user.out or "model.tl"
    checkpoint_dir = user.checkpoint_dir or (out + ".ckpt")

    resolved = ResolvedConfig(
        out=out,
        force=bool(user.force),
        resume=user.resume,
        ask_on_data_change=bool(user.ask_on_data_change),
        task=task,
        model_type=model_type,
        architecture=architecture,
        d_model=user.d_model or d_model,
        layers=user.layers or layers,
        heads=user.heads or heads,
        ff_mult=user.ff_mult or ff_mult,
        dropout=user.dropout if user.dropout is not None else 0.1,
        max_seq_len=max_seq_len,
        tokenizer=tokenizer,
        bpe_vocab_size=bpe_vocab_size,
        optimizer=user.optimizer or "adamw",
        learning_rate=user.learning_rate or (3e-4 if model_type == "transformer" else 1e-3),
        weight_decay=user.weight_decay if user.weight_decay is not None else 0.01,
        batch_size=user.batch_size if user.batch_size is not None else _auto_batch_size(
            n, max_seq_len, device, num_gpus if multi_gpu else 1
        ),
        epochs=user.epochs or _auto_epochs(n),
        max_steps=user.max_steps,
        gradient_accumulation_steps=user.gradient_accumulation_steps or 1,
        grad_clip=user.grad_clip if user.grad_clip is not None else 1.0,
        warmup_steps=user.warmup_steps if user.warmup_steps is not None else min(100, max(1, n // 10)),
        val_split=user.val_split if user.val_split is not None else (0.1 if n >= 50 else 0.0),
        patience=user.patience if user.patience is not None else 3,
        min_delta=user.min_delta if user.min_delta is not None else 1e-4,
        device=device,
        precision=precision,
        checkpoint_every=user.checkpoint_every or 50,
        checkpoint_dir=checkpoint_dir,
        gradient_checkpointing=gradient_checkpointing,
        compile=compile_model,
        num_workers=num_workers,
        multi_gpu=multi_gpu,
        seed=user.seed,
        verbose=user.verbose,
    )
    return resolved
