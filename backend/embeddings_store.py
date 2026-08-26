"""
Computes and caches Ollama embeddings for the movie catalog.

Embeddings are cached to disk keyed by a hash of (csv path, row count,
embedding model, text-builder version) so switching datasets or models
never silently serves stale vectors -- it just re-embeds.
"""

from __future__ import annotations
import hashlib
import json
import os
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ollama_client import get_embedding, OllamaError

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
TEXT_BUILDER_VERSION = "v1"


def _cache_key(csv_path: str, n_rows: int, model: str) -> str:
    raw = f"{os.path.abspath(csv_path)}|{n_rows}|{model}|{TEXT_BUILDER_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _paths(key: str) -> tuple[str, str]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return (
        os.path.join(CACHE_DIR, f"emb_{key}.npy"),
        os.path.join(CACHE_DIR, f"emb_{key}.meta.json"),
    )


def load_cached(csv_path: str, n_rows: int, model: str) -> Optional[np.ndarray]:
    key = _cache_key(csv_path, n_rows, model)
    npy_path, meta_path = _paths(key)
    if os.path.exists(npy_path) and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("complete") and meta.get("n_rows") == n_rows:
            return np.load(npy_path)
    return None


def build_embeddings(
    df: pd.DataFrame,
    text_fn: Callable,
    model: str,
    host: str,
    csv_path: str,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Compute embeddings for every row in df, with resumable disk checkpointing.

    progress_cb(done, total) is called after each embedding so a UI can show
    a progress bar.
    """
    n_rows = len(df)
    key = _cache_key(csv_path, n_rows, model)
    npy_path, meta_path = _paths(key)

    vectors: list[Optional[np.ndarray]] = [None] * n_rows
    start_idx = 0

    # Resume from a partial run if one exists for this exact key.
    if os.path.exists(npy_path) and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("n_rows") == n_rows and not meta.get("complete"):
            partial = np.load(npy_path)
            start_idx = meta.get("done", 0)
            for i in range(start_idx):
                vectors[i] = partial[i]

    dim = None
    for i in range(start_idx, n_rows):
        text = text_fn(df.iloc[i])
        vec = get_embedding(text, model=model, host=host)
        arr = np.array(vec, dtype=np.float32)
        if dim is None:
            dim = arr.shape[0]
        vectors[i] = arr

        # Checkpoint every 50 movies so a crash/interrupt doesn't lose progress.
        if (i + 1) % 50 == 0 or i == n_rows - 1:
            _checkpoint(vectors, dim, i + 1, n_rows, npy_path, meta_path, complete=(i + 1 == n_rows))

        if progress_cb:
            progress_cb(i + 1, n_rows)

    return np.stack(vectors).astype(np.float32)


def _checkpoint(vectors, dim, done, n_rows, npy_path, meta_path, complete):
    filled = np.zeros((n_rows, dim), dtype=np.float32)
    for i in range(done):
        filled[i] = vectors[i]
    np.save(npy_path, filled)
    with open(meta_path, "w") as f:
        json.dump({"done": done, "n_rows": n_rows, "complete": complete}, f)


def get_or_build_embeddings(
    df: pd.DataFrame,
    text_fn: Callable,
    model: str,
    host: str,
    csv_path: str,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    cached = load_cached(csv_path, len(df), model)
    if cached is not None:
        return cached
    return build_embeddings(df, text_fn, model, host, csv_path, progress_cb)
