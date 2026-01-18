from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import os

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_ALIASES = {
    "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "clip": "sentence-transformers/clip-ViT-B-32",
    "clip-vit-b-32": "sentence-transformers/clip-ViT-B-32",
    "sentence-transformers/clip-vit-b-32": "sentence-transformers/clip-ViT-B-32",
}


class EmbeddingService:
    def __init__(self, model_name: str | None = None):
        self.model_name = self._resolve_model_name(model_name)
        self._model: SentenceTransformer | None = None

    @staticmethod
    def _resolve_model_name(name: str | None) -> str:
        base = name or settings.model_name
        key = base.lower()
        if key in MODEL_ALIASES:
            return MODEL_ALIASES[key]
        return base

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: Iterable[str]) -> np.ndarray:
        embeddings = self.model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.embed_documents([query])
        return embedding.reshape(1, -1)


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


if __name__ == "__main__":
    service = EmbeddingService("clip")
    vector = service.embed_query("machine learning is a data driven application")
    print(vector.shape)


