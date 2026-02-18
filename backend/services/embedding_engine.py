from __future__ import annotations

import threading
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    """Local-only sentence embedding engine with singleton model cache."""

    _lock = threading.Lock()

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = self._load_model(model_name)

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_model(model_name: str) -> SentenceTransformer:
        with EmbeddingEngine._lock:
            return SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @property
    def embedding_dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def centroid(self, texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
        text_list = [t for t in texts if t and t.strip()]
        if not text_list:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        embs = self.encode(text_list, batch_size=batch_size)
        centroid = embs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            return centroid
        return (centroid / norm).astype(np.float32)


__all__ = ["EmbeddingEngine"]
