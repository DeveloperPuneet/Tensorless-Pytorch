"""MLP model for tabular data (classification / regression).

Numeric columns feed directly into the network; each categorical column
gets its own small embedding table, and all features are concatenated
before the hidden layers.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _embedding_dim(vocab_size: int) -> int:
    # Common rule of thumb, capped to keep tiny-data models tiny.
    return max(2, min(32, round(1.6 * (vocab_size ** 0.56))))


class TabularMLPV1(nn.Module):
    def __init__(
        self,
        n_numeric: int,
        categorical_vocab_sizes: List[int],
        d_model: int,
        layers: int,
        dropout: float,
        task: str,
        n_classes: int = 0,
    ):
        super().__init__()
        self.task = task
        self.n_numeric = n_numeric
        self.categorical_vocab_sizes = categorical_vocab_sizes

        self.embeddings = nn.ModuleList(
            [nn.Embedding(v, _embedding_dim(v)) for v in categorical_vocab_sizes]
        )
        cat_dim = sum(_embedding_dim(v) for v in categorical_vocab_sizes)
        in_dim = n_numeric + cat_dim

        dims = [in_dim] + [d_model] * layers
        modules = []
        for i in range(len(dims) - 1):
            modules += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        self.backbone = nn.Sequential(*modules)

        out_dim = n_classes if task == "classification" else 1
        self.head = nn.Linear(dims[-1], out_dim)

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.n_numeric > 0:
            parts.append(numeric)
        for i, emb in enumerate(self.embeddings):
            parts.append(emb(categorical[:, i]))
        x = torch.cat(parts, dim=1) if parts else numeric
        x = self.backbone(x)
        out = self.head(x)
        if self.task == "regression":
            return out.squeeze(-1)
        return out


class _ResidualBlock(nn.Module):
    """Pre-norm residual MLP block (LayerNorm -> Linear -> GELU -> Linear
    -> residual add). Deeper `TabularMLPV2` stacks of these train more
    stably than plain stacked Linear+ReLU at larger widths/depths.
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        return x + self.drop(h)


class TabularMLPV2(nn.Module):
    """Residual, pre-norm tabular MLP -- same public interface as
    `TabularMLPV1`, but scales more gracefully to the wider/deeper configs
    Tensorless now auto-selects for larger tabular datasets.
    """

    def __init__(
        self,
        n_numeric: int,
        categorical_vocab_sizes: List[int],
        d_model: int,
        layers: int,
        dropout: float,
        task: str,
        n_classes: int = 0,
    ):
        super().__init__()
        self.task = task
        self.n_numeric = n_numeric
        self.categorical_vocab_sizes = categorical_vocab_sizes

        self.embeddings = nn.ModuleList(
            [nn.Embedding(v, _embedding_dim(v)) for v in categorical_vocab_sizes]
        )
        cat_dim = sum(_embedding_dim(v) for v in categorical_vocab_sizes)
        in_dim = n_numeric + cat_dim

        self.in_proj = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([_ResidualBlock(d_model, dropout) for _ in range(layers)])
        self.norm_f = nn.LayerNorm(d_model)

        out_dim = n_classes if task == "classification" else 1
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.n_numeric > 0:
            parts.append(numeric)
        for i, emb in enumerate(self.embeddings):
            parts.append(emb(categorical[:, i]))
        x = torch.cat(parts, dim=1) if parts else numeric
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        out = self.head(x)
        if self.task == "regression":
            return out.squeeze(-1)
        return out


# New training runs default to the residual v2 architecture; v1 stays
# available (and is what old checkpoints without an "architecture" key
# in their config resolve to) for backward compatibility.
TabularMLP = TabularMLPV2
