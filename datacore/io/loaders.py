# datacore/io/loaders.py
"""
Dataset loaders shared across tools.

- `load_local_dataset(path)` reads `.json` or `.jsonl` and unwraps common
  container shapes (`{"data": [...]}`, `{"entries": [...]}`, etc.).
- `load_huggingface_dataset(name, subset, token)` wraps `datasets.load_dataset`
  and returns a plain `list[dict]` so callers don't have to deal with the
  `Dataset` object.
"""

from __future__ import annotations

import json
import os
from typing import Optional


def load_local_dataset(path: str) -> list[dict]:
    """Load a local JSON or JSONL file into a list of dict entries.

    Accepts a top-level list, a single dict, or a wrapper dict whose payload is
    one of the common container keys (``data``, ``entries``, ``rows``, ``items``).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Local dataset file not found: {path}")

    if path.lower().endswith(".jsonl"):
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("data", "entries", "rows", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    return data if isinstance(data, list) else []


def load_huggingface_dataset(
    name: str,
    subset: Optional[str] = None,
    token: Optional[str] = None,
    split: str = "train",
) -> list[dict]:
    """Load a HuggingFace dataset as a plain list of dict entries."""
    from datasets import load_dataset

    if subset:
        ds = load_dataset(name, subset, split=split, token=token)
    else:
        ds = load_dataset(name, split=split, token=token)
    return [dict(x) for x in ds]
