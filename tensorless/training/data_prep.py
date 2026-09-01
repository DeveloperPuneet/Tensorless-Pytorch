"""Turns a loaded `Dataset` + resolved config into PyTorch-ready tensors,
train/val splits, and the tokenizer/preprocessor that produced them.

Kept separate from `trainer.py` so the "how do I turn data into tensors"
logic can be tested and extended independently of the training loop.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler

from ..auto.detector import target_column
from ..data.loader import Dataset
from ..data.tabular import TabularPreprocessor
from ..errors import DataError
from ..tokenization.bpe_tokenizer import BPETokenizer
from ..tokenization.char_tokenizer import CharTokenizer


@dataclass
class PreparedData:
    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    meta: Dict[str, Any]
    tokenizer: Optional[CharTokenizer] = None
    preprocessor: Optional[TabularPreprocessor] = None


class _StreamingLMChunkDataset(IterableDataset):
    """Tokenize text once up front, then yield fixed-size language-model
    chunks lazily. Tokenizing eagerly (rather than inside `__iter__`)
    matters once vocab/corpus sizes get big: re-running BPE encoding on
    every text on every epoch is the single biggest avoidable cost in the
    training loop at "upper-mid" scale.
    """

    def __init__(self, texts: List[str], tokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self._token_seqs: List[List[int]] = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=True)
            if len(ids) <= block_size:
                ids = ids + [tokenizer.pad_id] * (block_size + 1 - len(ids))
            self._token_seqs.append(ids)
        self._length = sum(max(1, len(ids) - block_size) for ids in self._token_seqs)

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        # Combine DDP rank sharding with DataLoader worker sharding so
        # that using num_workers > 0 never duplicates data across workers.
        shard_id = rank * num_workers + worker_id
        shard_count = world_size * num_workers
        chunk_index = 0
        for ids in self._token_seqs:
            for start in range(0, len(ids) - self.block_size):
                chunk = ids[start: start + self.block_size + 1]
                if len(chunk) != self.block_size + 1:
                    continue
                if chunk_index % shard_count != shard_id:
                    chunk_index += 1
                    continue
                chunk_index += 1
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )


class _ClsTextDataset(TorchDataset):
    def __init__(self, input_ids: List[List[int]], attn_masks: List[List[int]], labels: List[int]):
        self.input_ids = input_ids
        self.attn_masks = attn_masks
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.attn_masks[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def _collate_text_classification(batch, pad_id: int):
    input_ids, attention_masks, labels = zip(*batch)
    return (
        pad_sequence(input_ids, batch_first=True, padding_value=pad_id),
        pad_sequence(attention_masks, batch_first=True, padding_value=0),
        torch.stack(labels),
    )


class _TabularDataset(TorchDataset):
    def __init__(self, numeric: torch.Tensor, categorical: torch.Tensor, target: torch.Tensor):
        self.numeric = numeric
        self.categorical = categorical
        self.target = target

    def __len__(self) -> int:
        return self.numeric.shape[0]

    def __getitem__(self, idx: int):
        return self.numeric[idx], self.categorical[idx], self.target[idx]


def _split_indices(n: int, val_split: float, seed: int) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = int(n * val_split)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if not train_idx:
        train_idx = idx
        val_idx = []
    return train_idx, val_idx


def _loader_options(dataset, shuffle: bool):
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        return {"sampler": DistributedSampler(dataset, shuffle=shuffle), "shuffle": False}
    return {"shuffle": shuffle}


def _loader_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Shared DataLoader performance knobs: worker processes + pinned
    memory for fast host->GPU transfer. A no-op on CPU/TPU where
    `num_workers` auto-resolves to 0 (see `auto/config.py`).
    """
    num_workers = cfg.get("num_workers", 0)
    kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": cfg.get("device") == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return kwargs


def _build_tokenizer(ds: Dataset, cfg: Dict[str, Any]):
    if cfg.get("tokenizer", "bpe") == "bpe":
        return BPETokenizer.build(ds.texts, vocab_size=cfg.get("bpe_vocab_size", 1000))
    return CharTokenizer.build(ds.texts)


def prepare_text_generation(
    ds: Dataset, cfg: Dict[str, Any], tokenizer=None
) -> PreparedData:
    tokenizer = tokenizer or _build_tokenizer(ds, cfg)
    block_size = cfg["max_seq_len"]
    n_val_texts = int(len(ds.texts) * cfg["val_split"])
    if n_val_texts and n_val_texts < len(ds.texts):
        train_texts = ds.texts[:-n_val_texts]
        val_texts = ds.texts[-n_val_texts:]
    elif cfg["val_split"] > 0 and len(ds.texts) == 1:
        text = ds.texts[0]
        split_point = int(len(text) * (1.0 - cfg["val_split"]))
        if split_point >= block_size:
            train_texts = [text[:split_point]]
            val_texts = [text[split_point:]]
        else:
            train_texts, val_texts = ds.texts, None
    else:
        train_texts, val_texts = ds.texts, None

    train_ds = _StreamingLMChunkDataset(train_texts, tokenizer, block_size)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=False, drop_last=False, **_loader_kwargs(cfg)
    )

    val_loader = None
    if val_texts is not None:
        val_ds = _StreamingLMChunkDataset(val_texts, tokenizer, block_size)
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, **_loader_kwargs(cfg))

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.pad_id,
        "n_classes": 0,
    }
    return PreparedData(train_loader=train_loader, val_loader=val_loader, meta=meta, tokenizer=tokenizer)


def prepare_text_classification(
    ds: Dataset,
    cfg: Dict[str, Any],
    tokenizer=None,
    classes: Optional[List[str]] = None,
) -> PreparedData:
    tokenizer = tokenizer or _build_tokenizer(ds, cfg)
    classes = classes or sorted(set(ds.labels))
    label2id = {c: i for i, c in enumerate(classes)}

    block_size = cfg["max_seq_len"]
    input_ids, attn_masks, labels = [], [], []
    for text, label in zip(ds.texts, ds.labels):
        ids = tokenizer.encode(text, add_special_tokens=True)[:block_size]
        mask = [1] * len(ids)
        input_ids.append(ids)
        attn_masks.append(mask)
        labels.append(label2id.get(label, 0))

    n = len(labels)
    train_idx, val_idx = _split_indices(n, cfg["val_split"], cfg["seed"])

    def subset(indices):
        return _ClsTextDataset(
            [input_ids[i] for i in indices],
            [attn_masks[i] for i in indices],
            [labels[i] for i in indices],
        )

    train_dataset = subset(train_idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        **_loader_options(train_dataset, True),
        **_loader_kwargs(cfg),
        collate_fn=lambda batch: _collate_text_classification(batch, tokenizer.pad_id),
    )
    val_loader = (
        DataLoader(
            (val_dataset := subset(val_idx)),
            batch_size=cfg["batch_size"],
            **_loader_options(val_dataset, False),
            **_loader_kwargs(cfg),
            collate_fn=lambda batch: _collate_text_classification(batch, tokenizer.pad_id),
        )
        if val_idx
        else None
    )

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.pad_id,
        "n_classes": len(classes),
        "classes": classes,
    }
    return PreparedData(train_loader=train_loader, val_loader=val_loader, meta=meta, tokenizer=tokenizer)


def prepare_tabular(
    ds: Dataset,
    cfg: Dict[str, Any],
    task: str,
    preprocessor: Optional[TabularPreprocessor] = None,
) -> PreparedData:
    target_col = target_column(ds)
    if target_col is None:
        raise DataError("Could not determine a target column for tabular training.")

    n = len(ds.records)
    train_idx, val_idx = _split_indices(n, cfg["val_split"], cfg["seed"])

    # Fit preprocessing statistics (numeric median/MAD, categorical
    # vocabularies, target mean/std or class list) on the *training*
    # split only. Fitting on the full dataset (including what becomes the
    # validation split) would leak validation-set statistics into the
    # features/targets used to train the model, making the validation
    # loss an overly-optimistic estimate of generalization.
    if preprocessor is not None:
        prep = preprocessor
    else:
        train_records = [ds.records[i] for i in train_idx]
        prep = TabularPreprocessor().fit(train_records, ds.columns, target_col, task)

    transformed = prep.transform(ds.records, with_target=True)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long)

    train_ds = _TabularDataset(
        transformed["numeric"][train_idx_t],
        transformed["categorical"][train_idx_t],
        transformed["target"][train_idx_t],
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], **_loader_options(train_ds, True),
        pin_memory=cfg.get("device") == "cuda",
    )

    val_loader = None
    if val_idx:
        val_idx_t = torch.tensor(val_idx, dtype=torch.long)
        val_ds = _TabularDataset(
            transformed["numeric"][val_idx_t],
            transformed["categorical"][val_idx_t],
            transformed["target"][val_idx_t],
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg["batch_size"], **_loader_options(val_ds, False),
            pin_memory=cfg.get("device") == "cuda",
        )

    meta = {
        "n_numeric": len(prep.numeric_columns),
        "categorical_vocab_sizes": prep.categorical_vocab_sizes(),
        "n_classes": len(prep.classes) if task == "classification" else 0,
    }
    return PreparedData(train_loader=train_loader, val_loader=val_loader, meta=meta, preprocessor=prep)
