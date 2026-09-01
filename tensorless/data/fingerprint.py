"""Dataset fingerprinting.

We hash the *content* of every data file under a path (plus filenames and
sizes) so that Tensorless PyTorch can cheaply detect whether a dataset has changed
between runs -- this is what powers the "don't retrain if nothing changed"
and "resume if interrupted" behaviour.

The fingerprint is intentionally content-based (not mtime-based) so that
copying a dataset to a new machine, or touching a file without changing it,
does not trigger an unnecessary retrain.
"""

from __future__ import annotations

import hashlib
import os
from typing import List

from .loader import (
    CSV_EXTENSIONS,
    JSON_EXTENSIONS,
    JSONL_EXTENSIONS,
    TEXT_EXTENSIONS,
    YAML_EXTENSIONS,
)

# Cap how many bytes of large files we hash, to keep fingerprinting fast on
# huge datasets. We hash the size + a content sample (head/tail) rather than
# the whole file when it exceeds this threshold.
_MAX_FULL_HASH_BYTES = 25 * 1024 * 1024  # 25 MB
_SAMPLE_BYTES = 1 * 1024 * 1024  # 1 MB head + 1 MB tail for big files

# Only fingerprint files `load_dataset` would actually read. Mirroring the
# loader's supported extensions here (rather than hashing every file in the
# directory) matters for two reasons:
#  - correctness: `os.walk` previously also descended into hidden
#    directories like `.git`, so switching branches or editing unrelated
#    dotfiles could change the fingerprint and trigger a spurious retrain,
#    even though none of the *data* changed.
#  - relevance: a stray non-data file (a README, a `.DS_Store`, a model
#    checkpoint someone dropped next to the dataset) would also perturb the
#    fingerprint despite the loader silently ignoring it -- the fingerprint
#    should reflect exactly what training will actually see.
SUPPORTED_EXTENSIONS = (
    TEXT_EXTENSIONS | JSON_EXTENSIONS | JSONL_EXTENSIONS | CSV_EXTENSIONS | YAML_EXTENSIONS
)


def _iter_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    files = []
    for root, dirs, names in os.walk(path):
        dirs.sort()
        # Don't descend into hidden directories (.git, .venv, .ipynb_checkpoints, ...).
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in sorted(names):
            if name.startswith("."):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            files.append(os.path.join(root, name))
    return files


def _hash_file(path: str, hasher: "hashlib._Hash") -> None:
    size = os.path.getsize(path)
    hasher.update(str(size).encode("utf-8"))
    with open(path, "rb") as f:
        if size <= _MAX_FULL_HASH_BYTES:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        else:
            hasher.update(f.read(_SAMPLE_BYTES))
            f.seek(max(0, size - _SAMPLE_BYTES))
            hasher.update(f.read(_SAMPLE_BYTES))


def fingerprint_path(path: str) -> str:
    """Return a stable hex digest fingerprinting the dataset at `path`.

    The fingerprint changes if any file's content, size, name, or the set
    of files itself changes.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path '{path}' does not exist.")
    path = os.path.abspath(path)
    hasher = hashlib.sha256()
    files = _iter_files(path)
    for f in files:
        rel = os.path.relpath(f, path if os.path.isdir(path) else os.path.dirname(path))
        hasher.update(rel.encode("utf-8"))
        _hash_file(f, hasher)
    hasher.update(str(len(files)).encode("utf-8"))
    return hasher.hexdigest()
