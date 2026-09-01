"""Task-appropriate evaluation metrics.

Loss alone doesn't tell a human much about a model's real-world quality --
"val_loss=1.8" is far less legible than "accuracy=62%" or "perplexity=6.2"
or "MAE=3.4, R2=0.81". This module computes those from small running sums
accumulated batch-by-batch, so a full pass over the data only needs to
happen once.

It's shared by the training loop's per-epoch validation reporting (see
`training/trainer.py`) and the public `LoadedModel.evaluate()` API, so
"accuracy" (or any other metric) means exactly the same thing whether it's
being reported during training or computed after the fact on new data.
"""

from __future__ import annotations

import math
from typing import Dict, Optional


def perplexity_from_loss(mean_loss: Optional[float]) -> Optional[float]:
    """Perplexity = exp(mean cross-entropy loss), the standard language-model
    metric. Returns None if the loss isn't a finite number (e.g. no
    validation batches were run), and +inf rather than raising if the loss
    is so large that exp() would overflow.
    """
    if mean_loss is None or not math.isfinite(mean_loss):
        return None
    try:
        return math.exp(mean_loss)
    except OverflowError:
        return float("inf")


class ClassificationAccumulator:
    """Accumulates correct/total counts across batches to compute accuracy
    without holding every prediction in memory at once."""

    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, correct: int, total: int) -> None:
        self.correct += correct
        self.total += total

    def compute(self) -> Dict[str, Optional[float]]:
        return {"accuracy": (self.correct / self.total) if self.total else None}


class RegressionAccumulator:
    """Accumulates the running sums needed for MAE / RMSE / R^2 across
    batches, without needing every prediction/target held in memory (R^2's
    total-sum-of-squares term is derived algebraically from
    sum(y), sum(y^2), and n rather than requiring a second pass with the
    mean known up front).
    """

    def __init__(self) -> None:
        self.sq_err_sum = 0.0
        self.abs_err_sum = 0.0
        self.target_sum = 0.0
        self.target_sq_sum = 0.0
        self.n = 0

    def update(self, sq_err_sum: float, abs_err_sum: float, target_sum: float,
               target_sq_sum: float, n: int) -> None:
        self.sq_err_sum += sq_err_sum
        self.abs_err_sum += abs_err_sum
        self.target_sum += target_sum
        self.target_sq_sum += target_sq_sum
        self.n += n

    def compute(self) -> Dict[str, Optional[float]]:
        if self.n == 0:
            return {"mae": None, "rmse": None, "r2": None}
        mse = self.sq_err_sum / self.n
        mae = self.abs_err_sum / self.n
        rmse = math.sqrt(mse)
        mean_target = self.target_sum / self.n
        # sum((y - mean)^2) == sum(y^2) - n * mean^2
        ss_tot = self.target_sq_sum - self.n * mean_target * mean_target
        r2 = (1.0 - (self.sq_err_sum / ss_tot)) if ss_tot > 1e-12 else None
        return {"mae": mae, "rmse": rmse, "r2": r2}
