"""Checkpoint management.

Handles all the state needed to resume training safely and transparently:
model weights, optimizer state, scheduler state, epoch/step counters, the
resolved training config, tokenizer/preprocessor state, the dataset
fingerprint used for training, and the best-metric-so-far for early
stopping.

Users never touch this directly -- `tl.train()` decides automatically
whether to create, update, or resume from a checkpoint (see
`training/trainer.py` and the "Smart Auto Check" logic in `api.py`).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Dict, Optional

import torch

from ..errors import CheckpointError

CHECKPOINT_FILENAME = "checkpoint.pt"


class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.path = os.path.join(checkpoint_dir, CHECKPOINT_FILENAME)

    def exists(self) -> bool:
        return os.path.isfile(self.path)

    def save(self, state: Dict[str, Any]) -> None:
        """Atomically write `state` to the checkpoint file.

        Writes to a temp file first and renames it into place, so a crash
        or interruption mid-write never leaves a corrupt checkpoint that
        would block resumption.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=self.checkpoint_dir, suffix=".tmp")
        os.close(fd)
        try:
            torch.save(state, tmp_path)
            shutil.move(tmp_path, self.path)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise CheckpointError(f"Failed to save checkpoint to '{self.path}': {e}") from e

    def load(self, map_location: Optional[str] = "cpu") -> Dict[str, Any]:
        """Load the checkpoint.

        Defaults to `map_location="cpu"` (rather than the original
        device the tensors were saved from) so a checkpoint written on a
        CUDA machine can still be loaded and resumed on a CPU-only
        machine, or one with a different number/kind of GPUs -- resuming
        training then moves things back to the actually-resolved device.
        Without this, `torch.load` tries to deserialize storages onto
        their *original* device and raises if that device isn't
        available on the current machine.

        `weights_only=True` restricts unpickling to a safe, well-known
        set of types (tensors, dicts, lists, primitives, etc.), so
        loading a checkpoint can't be used to execute arbitrary code via
        a crafted pickle -- relevant since checkpoints can live on
        shared/networked storage in multi-machine or resumed-elsewhere
        setups.
        """
        if not self.exists():
            raise CheckpointError(f"No checkpoint found at '{self.path}'.")
        try:
            return torch.load(self.path, map_location=map_location, weights_only=True)
        except Exception as e:
            raise CheckpointError(
                f"Checkpoint at '{self.path}' is corrupt or incompatible: {e}"
            ) from e

    def clear(self) -> None:
        if os.path.isdir(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir, ignore_errors=True)
