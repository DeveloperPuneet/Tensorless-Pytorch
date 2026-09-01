"""Dataset loading.

Turns a path (file or directory) into a normalized `Dataset` object that
the rest of Tensorless PyTorch can reason about, regardless of whether the data
started life as .txt, .json, .jsonl, .csv, or a directory of any of those
(optionally organized into class subfolders).

Design goal: never silently drop or mutate user data. If something looks
wrong (empty dataset, unreadable file, inconsistent columns) we raise a
`DataError` with an actionable message rather than guessing.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..errors import DataError
from .format_detect import (
    flatten_records_to_text,
    looks_textual,
    normalize_records_to_text,
)

TEXT_EXTENSIONS = {".txt", ".md"}
JSON_EXTENSIONS = {".json"}
JSONL_EXTENSIONS = {".jsonl", ".ndjson"}
CSV_EXTENSIONS = {".csv", ".tsv"}
YAML_EXTENSIONS = {".yaml", ".yml"}

# Common field names we look for when a JSON/JSONL record represents a
# single piece of free text (e.g. for language modeling).
_TEXT_FIELD_CANDIDATES = ("text", "content", "body", "document", "sentence")
_LABEL_FIELD_CANDIDATES = ("label", "target", "class", "category", "y")


@dataclass
class Dataset:
    """Normalized in-memory representation of a loaded dataset.

    kind is one of:
      - "text"        : a corpus of raw text (language modeling)
      - "text_labeled": (text, label) pairs (text classification)
      - "tabular"      : rows of named columns (classification/regression)
    """

    kind: str
    source: str
    texts: List[str] = field(default_factory=list)
    labels: List[Any] = field(default_factory=list)
    records: List[Dict[str, Any]] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    n_files: int = 0

    def __len__(self) -> int:
        if self.kind in ("text", "text_labeled"):
            return len(self.texts)
        return len(self.records)


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            return f.read()
    except UnicodeDecodeError as e:
        raise DataError(
            f"File '{path}' is not valid UTF-8 text. Tensorless PyTorch expects "
            f"text datasets to be UTF-8 encoded."
        ) from e


def _read_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Either a single record, or {"data": [...]}
        for key in ("data", "records", "items", "examples"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise DataError(f"Unsupported JSON structure in '{path}': expected a list of records.")
    return data


def _read_jsonl_records(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise DataError(f"Malformed JSON on line {i + 1} of '{path}': {e}") from e
    return records


def _read_yaml_records(path: str) -> List[Dict[str, Any]]:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise DataError(
            f"'{path}' is a YAML file, but the optional 'pyyaml' package "
            f"isn't installed. Run `pip install pyyaml` to load YAML "
            f"datasets, or convert the file to .json/.jsonl."
        ) from e
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict):
        for key in ("data", "records", "items", "examples"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise DataError(f"Unsupported YAML structure in '{path}': expected a list of records.")
    if not all(isinstance(r, dict) for r in data):
        raise DataError(f"Unsupported YAML structure in '{path}': expected a list of mappings.")
    return data


def _read_csv_records(path: str) -> List[Dict[str, Any]]:
    delimiter = "\t" if path.lower().endswith(".tsv") else ","
    records = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise DataError(f"'{path}' has no header row / is empty.")
        for row in reader:
            records.append(dict(row))
    return records


def _extract_text_field(record: Dict[str, Any]) -> Optional[str]:
    for key in _TEXT_FIELD_CANDIDATES:
        if key in record and isinstance(record[key], str):
            return record[key]
    return None


def _extract_label_field(record: Dict[str, Any]) -> Optional[Any]:
    for key in _LABEL_FIELD_CANDIDATES:
        if key in record:
            return record[key]
    return None


def _records_to_dataset(records: List[Dict[str, Any]], source: str) -> Dataset:
    if not records:
        raise DataError(f"No records found in '{source}'.")

    text_field_present = all(_extract_text_field(r) is not None for r in records[:50])
    if text_field_present:
        texts = [_extract_text_field(r) or "" for r in records]
        labels = [_extract_label_field(r) for r in records]
        if any(label is not None for label in labels):
            return Dataset(kind="text_labeled", source=source, texts=texts, labels=labels)
        return Dataset(kind="text", source=source, texts=texts)

    # Not a plain {"text": ...} shape -- try recognizing it as chat /
    # instruction-tuning data instead: turn lists ("messages": [...],
    # "conversations": [...]) or flat conversational pairs (user/bot,
    # instruction/output, prompt/completion, etc.), under any of their
    # common field-name spellings.
    normalized = normalize_records_to_text(records)
    if normalized is not None:
        return Dataset(kind="text", source=source, texts=normalized)

    # Otherwise treat as generic tabular data.
    # Preserve column order as it appears in the data (important: the
    # "use the last column as the target" heuristic in auto/detector.py
    # depends on this being the *original* column order, not alphabetical).
    columns: List[str] = []
    seen = set()
    for r in records:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                columns.append(k)

    # Last resort: the record shape didn't match any known text/chat
    # convention, but it also doesn't look like a structured table (its
    # string fields read like free-form prose, not short categorical/
    # numeric values). Rather than forcing it through the tabular
    # pipeline -- or raising -- flatten each record into readable
    # "Key: value" text and train on that. This is what makes truly
    # arbitrary/unrecognized formats "just work" instead of erroring out.
    if looks_textual(records):
        print(
            f"[tensorless] note: records in '{source}' didn't match a known "
            f"text/tabular/chat format -- auto-normalizing each record into "
            f"plain text for language-model training. Pass a recognized "
            f"format (e.g. a 'text' field, or 'messages'/'user'+'bot' style "
            f"records) if this isn't what you want."
        )
        return Dataset(kind="text", source=source, texts=flatten_records_to_text(records))

    return Dataset(kind="tabular", source=source, records=records, columns=columns)


def _load_directory(path: str) -> Dataset:
    entries = [e for e in sorted(os.listdir(path)) if not e.startswith(".")]
    if not entries:
        raise DataError(f"Directory '{path}' is empty.")

    subdirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(path, e))]

    # Case 1: class subfolders of text files -> text classification.
    if subdirs and not files:
        texts, labels = [], []
        n_files = 0
        empty_labels = []
        for label in subdirs:
            sub = os.path.join(path, label)
            n_before = len(texts)
            for fname in sorted(os.listdir(sub)):
                fpath = os.path.join(sub, fname)
                if not os.path.isfile(fpath):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext in TEXT_EXTENSIONS:
                    texts.append(_read_text_file(fpath))
                    labels.append(label)
                    n_files += 1
            if len(texts) == n_before:
                empty_labels.append(label)
        if not texts:
            raise DataError(
                f"Directory '{path}' contains subfolders but no readable .txt files inside them."
            )
        # Strict validation: a class subfolder with no readable text files
        # would otherwise silently vanish as a class rather than raising --
        # producing a dataset with fewer labels than the directory
        # structure implies, with no indication anything was dropped.
        if empty_labels:
            names = ", ".join(f"'{label}'" for label in empty_labels)
            raise DataError(
                f"Class subfolder(s) {names} in '{path}' contain no readable "
                f".txt files -- every class subfolder must have at least one "
                f"text file. Remove the empty subfolder(s) or add data to them."
            )
        ds = Dataset(kind="text_labeled", source=path, texts=texts, labels=labels, n_files=n_files)
        return ds

    # Case 2: flat directory of files -> merge by type.
    all_records: List[Dict[str, Any]] = []
    all_texts: List[str] = []
    n_files = 0
    for fname in files:
        fpath = os.path.join(path, fname)
        ext = os.path.splitext(fname)[1].lower()
        if ext in TEXT_EXTENSIONS:
            all_texts.append(_read_text_file(fpath))
            n_files += 1
        elif ext in JSON_EXTENSIONS:
            all_records.extend(_read_json_records(fpath))
            n_files += 1
        elif ext in JSONL_EXTENSIONS:
            all_records.extend(_read_jsonl_records(fpath))
            n_files += 1
        elif ext in CSV_EXTENSIONS:
            all_records.extend(_read_csv_records(fpath))
            n_files += 1
        elif ext in YAML_EXTENSIONS:
            all_records.extend(_read_yaml_records(fpath))
            n_files += 1
        # silently skip unknown extensions (e.g. README, .gitkeep) -- but
        # never silently skip *data*-looking files; this is only for
        # incidental non-data files.

    if all_records and all_texts:
        raise DataError(
            f"Directory '{path}' mixes plain text files with structured "
            f"(json/csv) files. Please keep one data format per directory."
        )
    if all_records:
        ds = _records_to_dataset(all_records, path)
        ds.n_files = n_files
        return ds
    if all_texts:
        return Dataset(kind="text", source=path, texts=all_texts, n_files=n_files)

    raise DataError(
        f"No supported data files found in '{path}'. Supported: "
        f".txt, .md, .json, .jsonl, .csv, .tsv, .yaml, .yml"
    )


def load_dataset(path: str) -> Dataset:
    """Load a dataset from `path` (file or directory) into a `Dataset`."""
    if not os.path.exists(path):
        raise DataError(f"Path '{path}' does not exist.")

    if os.path.isdir(path):
        return _load_directory(path)

    ext = os.path.splitext(path)[1].lower()
    if ext in TEXT_EXTENSIONS:
        text = _read_text_file(path)
        if not text.strip():
            raise DataError(f"'{path}' is empty.")
        return Dataset(kind="text", source=path, texts=[text], n_files=1)
    if ext in JSON_EXTENSIONS:
        records = _read_json_records(path)
        ds = _records_to_dataset(records, path)
        ds.n_files = 1
        return ds
    if ext in JSONL_EXTENSIONS:
        records = _read_jsonl_records(path)
        ds = _records_to_dataset(records, path)
        ds.n_files = 1
        return ds
    if ext in CSV_EXTENSIONS:
        records = _read_csv_records(path)
        ds = _records_to_dataset(records, path)
        ds.n_files = 1
        return ds
    if ext in YAML_EXTENSIONS:
        records = _read_yaml_records(path)
        ds = _records_to_dataset(records, path)
        ds.n_files = 1
        return ds

    raise DataError(
        f"Unsupported file type '{ext}' for '{path}'. Supported: "
        f".txt, .md, .json, .jsonl, .csv, .tsv, .yaml, .yml, or a directory containing these."
    )
