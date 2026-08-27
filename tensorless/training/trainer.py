"""The training loop.

`run_training` is the single entry point that: prepares data, builds (or
resumes) the model + optimizer, trains with early stopping, checkpoints
periodically so interrupted runs can resume, and returns everything
needed to write the final `.tl` file.
"""

from __future__ import annotations

import time
import math
import os
from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ..data.loader import Dataset
from ..devices.device import get_torch_device, get_device_info, mark_step
from ..checkpoint.manager import CheckpointManager
from ..models.registry import build_model
from ..tokenization import tokenizer_from_state_dict
from ..data.tabular import TabularPreprocessor
from ..errors import ModelError
from .early_stopping import EarlyStopping
from . import data_prep as dp


def _build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name = cfg["optimizer"].lower()
    lr = cfg["learning_rate"]
    wd = cfg["weight_decay"]
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer '{name}'")


def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.1, 1.0 - progress)


def _compute_loss(task: str, model_type: str, model: nn.Module, batch, device, pad_id: int) -> torch.Tensor:
    non_blocking = device.type == "cuda"
    if model_type == "transformer" and task == "text-generation":
        x, y = batch
        x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad_id
        )
        return loss
    elif model_type == "transformer" and task == "text-classification":
        input_ids, attn_mask, labels = batch
        input_ids = input_ids.to(device, non_blocking=non_blocking)
        attn_mask = attn_mask.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        logits = model(input_ids, attention_mask=attn_mask)
        return F.cross_entropy(logits, labels)
    elif model_type == "mlp" and task == "classification":
        numeric, categorical, target = batch
        numeric = numeric.to(device, non_blocking=non_blocking)
        categorical = categorical.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)
        logits = model(numeric, categorical)
        return F.cross_entropy(logits, target)
    elif model_type == "mlp" and task == "regression":
        numeric, categorical, target = batch
        numeric = numeric.to(device, non_blocking=non_blocking)
        categorical = categorical.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)
        pred = model(numeric, categorical)
        return F.mse_loss(pred, target)
    else:
        raise ValueError(f"Unsupported task/model_type combination: {task}/{model_type}")


def _amp_context(device: torch.device, precision: str):
    """Return an autocast context when the resolved precision supports it.
    TPU (XLA) doesn't use CUDA-style autocast -- bf16 there is handled by
    casting the model itself (see `_maybe_cast_for_tpu`), so this only
    ever activates for CUDA.
    """
    if device.type != "cuda" or precision not in ("fp16", "bf16"):
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _maybe_cast_for_tpu(model: nn.Module, device: torch.device, precision: str) -> nn.Module:
    """TPUs get their mixed-precision speedup from running the whole model
    in bfloat16 (XLA handles the numerics well since bf16 has the same
    exponent range as fp32), rather than an autocast context manager.
    """
    if device.type == "xla" and precision == "bf16":
        return model.to(torch.bfloat16)
    return model


def _maybe_compile(model: nn.Module, device: torch.device, cfg: Dict[str, Any], log_fn) -> nn.Module:
    if device.type != "cuda" or not cfg.get("compile"):
        return model
    try:
        compiled = torch.compile(model)
        return compiled
    except Exception as e:
        if cfg.get("verbose"):
            log_fn(f"[tensorless] torch.compile unavailable ({e}); continuing uncompiled")
        return model


def _build_grad_scaler(enabled: bool):
    """Build the available PyTorch scaler API across supported versions."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _load_pretrained_weights(model: nn.Module, pretrained_sd: Dict[str, Any], log_fn, verbose: bool) -> None:
    """Initialize `model` from a pretrained state dict, skipping any keys
    whose shape doesn't match (typically the task head -- e.g. going from
    text-generation to text-classification, or a different n_classes).
    Everything else (embeddings, attention/MLP blocks, norms) transfers.
    """
    model_sd = model.state_dict()
    compatible: Dict[str, Any] = {}
    skipped = []
    for key, value in pretrained_sd.items():
        if key in model_sd and model_sd[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    model.load_state_dict(compatible, strict=False)
    if verbose:
        log_fn(
            f"[tensorless] loaded {len(compatible)}/{len(pretrained_sd)} pretrained weight "
            f"tensors" + (f"; reinitialized {len(skipped)} mismatched (e.g. task head): {skipped[:4]}"
                           f"{'...' if len(skipped) > 4 else ''}" if skipped else "")
        )


def run_training(
    ds: Dataset,
    cfg: Dict[str, Any],
    checkpoint_mgr: CheckpointManager,
    dataset_fingerprint: str,
    resume_state: Optional[Dict[str, Any]] = None,
    init_from: Optional[Dict[str, Any]] = None,
    log_fn=print,
) -> Dict[str, Any]:
    task = cfg["task"]
    model_type = cfg["model_type"]
    torch.manual_seed(cfg["seed"])

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    device = get_torch_device(cfg["device"])
    if distributed and device.type == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(device)
    cfg = dict(cfg)
    cfg["verbose"] = cfg["verbose"] and rank == 0
    amp_enabled = device.type == "cuda" and cfg["precision"] in ("fp16", "bf16")
    scaler = _build_grad_scaler(amp_enabled and cfg["precision"] == "fp16")
    if cfg["verbose"]:
        actual_precision = cfg["precision"] if (amp_enabled or device.type == "xla") else "fp32"
        info = get_device_info(device)
        mem = f", {info['memory_gb']}GB" if info.get("memory_gb") else ""
        ndev = f" x{info['num_devices']}" if info.get("num_devices", 1) > 1 else ""
        log_fn(
            f"[tensorless] task={task} model={model_type} architecture={cfg.get('architecture', 'v1')} "
            f"device={device} ({info['name']}{ndev}{mem}) precision={actual_precision}"
        )

    # ---- data prep (resume tokenizer/preprocessor if available) ----
    tokenizer = None
    preprocessor = None
    if resume_state is not None:
        if resume_state.get("tokenizer_state") is not None:
            tokenizer = tokenizer_from_state_dict(resume_state["tokenizer_state"])
        if resume_state.get("preprocessor_state") is not None:
            preprocessor = TabularPreprocessor.from_state_dict(resume_state["preprocessor_state"])
    elif init_from is not None:
        # Fine-tuning: reuse the pretrained model's exact tokenizer/
        # preprocessor rather than fitting a new one on this dataset --
        # token ids (and numeric scaling/categorical vocabs) must line up
        # with what the pretrained weights were trained on.
        if init_from.get("tokenizer_state") is not None:
            tokenizer = tokenizer_from_state_dict(init_from["tokenizer_state"])
        if init_from.get("preprocessor_state") is not None:
            preprocessor = TabularPreprocessor.from_state_dict(init_from["preprocessor_state"])

    if task == "text-generation":
        prepared = dp.prepare_text_generation(ds, cfg, tokenizer=tokenizer)
    elif task == "text-classification":
        classes = resume_state["meta"]["classes"] if resume_state else None
        prepared = dp.prepare_text_classification(ds, cfg, tokenizer=tokenizer, classes=classes)
    elif task in ("classification", "regression"):
        prepared = dp.prepare_tabular(ds, cfg, task=task, preprocessor=preprocessor)
    else:
        raise ValueError(f"Unsupported task '{task}'")

    # ---- model / optimizer / scheduler ----
    model = build_model(task, model_type, cfg, prepared.meta).to(device)
    model = _maybe_cast_for_tpu(model, device, cfg["precision"])
    if init_from is not None and resume_state is None:
        _load_pretrained_weights(model, init_from["model_state_dict"], log_fn, cfg["verbose"])
    multi_gpu_dp = False
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
        )
    elif device.type == "cuda" and cfg.get("multi_gpu") and torch.cuda.device_count() > 1:
        # Single-process multi-GPU (e.g. a Kaggle/Colab notebook with 2+
        # GPUs): without this, only cuda:0 is ever touched and any other
        # visible GPU sits idle at 0% utilization the whole run.
        gpu_ids = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=gpu_ids)
        multi_gpu_dp = True
        if cfg["verbose"]:
            log_fn(f"[tensorless] using {len(gpu_ids)} GPUs via DataParallel (gpu_ids={gpu_ids})")
    uses_module_wrapper = distributed or multi_gpu_dp
    # Keep a stable reference to the model *before* torch.compile wraps it.
    # torch.compile's OptimizedModule.state_dict() prefixes every key with
    # "_orig_mod." (an implementation detail of how it traces the module),
    # so saving/loading straight from the compiled object silently produces
    # a checkpoint whose keys don't match the real model architecture at
    # all -- always save/load/resume through this pre-compile reference
    # instead, since its parameters are the same tensors the compiled
    # function actually trains (compile wraps, it doesn't copy).
    ckpt_model = model
    if multi_gpu_dp and cfg.get("compile"):
        if cfg["verbose"]:
            log_fn("[tensorless] skipping torch.compile with DataParallel (not well supported together)")
    else:
        model = _maybe_compile(model, device, cfg, log_fn)
    optimizer = _build_optimizer(model, cfg)

    accumulation_steps = cfg.get("gradient_accumulation_steps", 1)
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    steps_per_epoch = max(1, math.ceil(len(prepared.train_loader) / accumulation_steps))
    total_steps = cfg.get("max_steps") or steps_per_epoch * cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _lr_lambda(s, cfg["warmup_steps"], total_steps)
    )

    early_stopper = EarlyStopping(patience=cfg["patience"], min_delta=cfg["min_delta"])
    start_epoch = 0
    global_step = 0

    def _ckpt_module() -> nn.Module:
        return ckpt_model.module if uses_module_wrapper else ckpt_model

    def _state_dict_cpu() -> Dict[str, Any]:
        return {k: v.detach().cpu().clone() for k, v in _ckpt_module().state_dict().items()}

    # Tracks the single best checkpoint seen so far (lowest val_loss when a
    # validation split exists, else lowest train_loss), independent of the
    # live/current weights. This is what actually gets saved at the end --
    # protects against a run that looks fine for most of training and then
    # diverges near the end (the loss log tapering off into NaN, which
    # would otherwise silently produce a broken final checkpoint even
    # though a perfectly good one existed a few epochs earlier).
    best_metric = float("inf")
    best_metric_name: Optional[str] = None
    best_epoch: Optional[int] = None
    best_state_dict: Optional[Dict[str, Any]] = None

    def _maybe_update_best(value: Optional[float], name: str, epoch_num: int) -> None:
        nonlocal best_metric, best_metric_name, best_epoch, best_state_dict
        if value is not None and math.isfinite(value) and value < best_metric:
            best_metric, best_metric_name, best_epoch = value, name, epoch_num
            best_state_dict = _state_dict_cpu()

    if resume_state is not None:
        (ckpt_model.module if uses_module_wrapper else ckpt_model).load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        if resume_state.get("scaler_state_dict"):
            scaler.load_state_dict(resume_state["scaler_state_dict"])
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        early_stopper.best = resume_state.get("early_stopping_best", float("inf"))
        early_stopper.num_bad_checks = resume_state.get("early_stopping_bad_checks", 0)
        best_metric = resume_state.get("best_metric", float("inf"))
        best_metric_name = resume_state.get("best_metric_name")
        best_epoch = resume_state.get("best_epoch")
        best_state_dict = resume_state.get("best_model_state_dict")
        if cfg["verbose"]:
            log_fn(f"[tensorless] resuming from checkpoint: epoch={start_epoch}, step={global_step}")

    pad_id = prepared.meta.get("pad_id", 0)

    def _checkpoint(epoch: int, training_complete: bool) -> None:
        if distributed and rank != 0:
            return
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": (ckpt_model.module if uses_module_wrapper else ckpt_model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "early_stopping_best": early_stopper.best,
            "early_stopping_bad_checks": early_stopper.num_bad_checks,
            "best_metric": best_metric,
            "best_metric_name": best_metric_name,
            "best_epoch": best_epoch,
            "best_model_state_dict": best_state_dict,
            "config": cfg,
            "meta": prepared.meta,
            "tokenizer_state": prepared.tokenizer.state_dict() if prepared.tokenizer else None,
            "preprocessor_state": prepared.preprocessor.state_dict() if prepared.preprocessor else None,
            "dataset_fingerprint": dataset_fingerprint,
            "training_complete": training_complete,
        }
        checkpoint_mgr.save(state)

    # ---- training loop ----
    model.train()
    optimizer.zero_grad()
    stop = False
    t0 = time.time()
    last_val_loss = None
    last_train_loss = None

    for epoch in range(start_epoch, cfg["epochs"]):
        for loader in (prepared.train_loader, prepared.val_loader):
            if loader is not None and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(prepared.train_loader):
            with _amp_context(device, cfg["precision"]):
                loss = _compute_loss(task, model_type, model, batch, device, pad_id)
            if not torch.isfinite(loss):
                # A non-finite loss (can happen from fp16 overflow on a
                # bad batch/step) must never be backpropagated -- doing so
                # poisons every parameter with NaN permanently. Skip it
                # and move on instead of corrupting the whole model.
                if cfg["verbose"]:
                    log_fn(f"[tensorless] warning: non-finite loss at step {global_step}, skipping batch")
                optimizer.zero_grad()
                continue
            last_train_loss = loss.item()
            group_start = batch_index - (batch_index % accumulation_steps)
            group_size = min(accumulation_steps, len(prepared.train_loader) - group_start)
            loss = loss / group_size
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            is_last_batch = batch_index + 1 == len(prepared.train_loader)
            if (batch_index + 1) % accumulation_steps == 0 or is_last_batch:
                if scaler.is_enabled():
                    if cfg["grad_clip"]:
                        # unscale_ must run before clipping, or clipping
                        # operates on the fp16 loss-scaled gradients (i.e.
                        # gradients inflated by up to 2^16x) instead of
                        # their true magnitude -- which silently makes the
                        # clip threshold meaningless and lets huge updates
                        # through, exactly the kind that blow up a model's
                        # weights into inf/NaN under sustained fp16 training.
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if cfg["grad_clip"]:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                mark_step(device)
                global_step += 1

                if global_step % cfg["checkpoint_every"] == 0:
                    _checkpoint(epoch, training_complete=False)

                if cfg.get("max_steps") and global_step >= cfg["max_steps"]:
                    stop = True
                    break
        if stop:
            break

        # ---- validation / early stopping ----
        if prepared.val_loader is not None:
            model.eval()
            losses = []
            with torch.no_grad():
                for batch in prepared.val_loader:
                    with _amp_context(device, cfg["precision"]):
                        losses.append(_compute_loss(task, model_type, model, batch, device, pad_id).item())
                    mark_step(device)
            val_loss = sum(losses) / max(1, len(losses))
            if distributed:
                val_tensor = torch.tensor([val_loss], device=device)
                dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
                val_loss = (val_tensor / world_size).item()
            last_val_loss = val_loss
            model.train()
            is_best = early_stopper.step(val_loss, state=None)
            if cfg["verbose"]:
                log_fn(
                    f"[tensorless] epoch {epoch + 1}/{cfg['epochs']} "
                    f"train_loss={last_train_loss:.4f} val_loss={val_loss:.4f}"
                    f"{' (best)' if is_best else ''}"
                )
            _maybe_update_best(val_loss, "val_loss", epoch + 1)
            if early_stopper.should_stop:
                if cfg["verbose"]:
                    log_fn(f"[tensorless] early stopping at epoch {epoch + 1} (no improvement)")
                _checkpoint(epoch, training_complete=True)
                break
        else:
            if cfg["verbose"]:
                log_fn(f"[tensorless] epoch {epoch + 1}/{cfg['epochs']} train_loss={last_train_loss:.4f}")
            _maybe_update_best(last_train_loss, "train_loss", epoch + 1)

        _checkpoint(epoch + 1, training_complete=(epoch + 1 >= cfg["epochs"]))

    elapsed = time.time() - t0
    if cfg["verbose"]:
        log_fn(f"[tensorless] training finished in {elapsed:.1f}s ({global_step} steps)")

    metrics = {
        "final_train_loss": last_train_loss,
        "final_val_loss": last_val_loss,
        "best_metric": best_metric if best_state_dict is not None else None,
        "best_metric_name": best_metric_name,
        "best_epoch": best_epoch,
        "global_step": global_step,
        "elapsed_seconds": elapsed,
    }

    live_state = _ckpt_module().state_dict()
    live_is_finite = all(
        torch.isfinite(v).all() for v in live_state.values() if torch.is_floating_point(v)
    )
    if best_state_dict is not None:
        final_state_dict = best_state_dict
        if cfg["verbose"]:
            log_fn(
                f"[tensorless] using best checkpoint (epoch {best_epoch}, "
                f"{best_metric_name}={best_metric:.4f}) as the saved model"
            )
    elif live_is_finite:
        final_state_dict = live_state
    else:
        raise ModelError(
            "Training never produced a usable (finite-loss) checkpoint -- every "
            "recorded epoch had a non-finite loss. Try a lower learning_rate=, a "
            "different precision (e.g. precision='bf16' on an Ampere+ GPU, or "
            "precision='fp32'), or check the dataset for problematic examples."
        )

    return {
        "model": model,
        "model_state_dict": final_state_dict,
        "meta": prepared.meta,
        "tokenizer": prepared.tokenizer,
        "preprocessor": prepared.preprocessor,
        "metrics": metrics,
        "device": device,
    }
