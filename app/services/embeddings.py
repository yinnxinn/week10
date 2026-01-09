from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings


class EmbeddingService:
    """Encapsulates the embedding model lifecycle."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: Iterable[str]) -> np.ndarray:
        """Generate embeddings for a batch of documents."""
        embeddings = self.model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Generate an embedding for a single query."""
        embedding = self.embed_documents([query])
        return embedding.reshape(1, -1)


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
