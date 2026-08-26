"""Model registry.

Central place that knows how to build a model given a task + resolved
config + the dataset-derived metadata (vocab size, n_classes, etc). Kept
separate from `auto/config.py` so new model types/backends can be added
here without touching auto-configuration logic, per the "extensible
architecture" requirement.

Also owns the v1/v2 architecture switch: `cfg.get("architecture", "v1")`
decides which concrete class gets built. Checkpoints saved before the v2
upgrade have no "architecture" key in their stored config at all, so
`.get(..., "v1")` transparently resolves them to the original
implementation -- old `.tl` files keep loading and running correctly.
New training runs get "v2" (better quality, KV-cache generation, scales
to "upper-mid" sizes) unless the user explicitly pins `architecture="v1"`.
"""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from .transformer import TinyTransformerV1, TinyTransformerV2
from .mlp import TabularMLPV1, TabularMLPV2
from ..errors import ModelError


def build_model(task: str, model_type: str, cfg: Dict[str, Any], meta: Dict[str, Any]) -> nn.Module:
    """Build a fresh, randomly-initialized model.

    `cfg` is the resolved training config (dict). `meta` carries
    task-specific sizing info produced during data prep, e.g.:
      - text tasks: {"vocab_size": int, "pad_id": int, "n_classes": int}
      - tabular tasks: {"n_numeric": int, "categorical_vocab_sizes": [...], "n_classes": int}
    """
    architecture = cfg.get("architecture", "v1")

    if model_type == "transformer":
        common = dict(
            vocab_size=meta["vocab_size"],
            d_model=cfg["d_model"],
            layers=cfg["layers"],
            heads=cfg["heads"],
            ff_mult=cfg["ff_mult"],
            dropout=cfg["dropout"],
            max_seq_len=cfg["max_seq_len"],
            task=task,
            n_classes=meta.get("n_classes", 0),
            pad_id=meta.get("pad_id", 0),
        )
        if architecture == "v2":
            return TinyTransformerV2(
                gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
                **common,
            )
        return TinyTransformerV1(**common)

    elif model_type == "mlp":
        common = dict(
            n_numeric=meta["n_numeric"],
            categorical_vocab_sizes=meta["categorical_vocab_sizes"],
            d_model=cfg["d_model"],
            layers=cfg["layers"],
            dropout=cfg["dropout"],
            task=task,
            n_classes=meta.get("n_classes", 0),
        )
        if architecture == "v2":
            return TabularMLPV2(**common)
        return TabularMLPV1(**common)

    else:
        raise ModelError(f"Unknown model_type '{model_type}'.")
