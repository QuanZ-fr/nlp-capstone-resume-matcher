"""Sentence embeddings with a cache on disk (encoding 12k resumes takes a few minutes on a laptop)."""
import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer

from . import DATA_DIR

MODEL_NAME = "all-MiniLM-L6-v2"     # 384 dimensions, max 256 tokens (longer texts are truncated)
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode(texts, name=None, batch_size=64):
    """Normalized embeddings (cosine similarity = dot product). Cached in data/<name>.npy when `name` is given."""
    texts = list(texts)
    if name:
        digest = hashlib.md5("\n".join(texts).encode()).hexdigest()[:8]     # cache invalidated if the texts change
        path = DATA_DIR / f"emb_{name}_{digest}.npy"
        if path.exists():
            return np.load(path)
    emb = get_model().encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=bool(name))
    if name:
        np.save(path, emb)
    return emb
