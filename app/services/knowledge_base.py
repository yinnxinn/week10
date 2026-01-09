from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import faiss
import numpy as np

from app.models.document import Document, QueryResult


@dataclass
class KnowledgeBase:
    """Manages FAISS index storage and retrieval."""

    index_path: Path
    metadata_path: Path
    _index: faiss.Index | None = field(init=False, default=None)
    _metadata: list[dict] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.load()

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    def load(self) -> None:
        """Load index and metadata from disk if available."""
        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
        if self.metadata_path.exists():
            self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def save(self) -> None:
        """Persist the index and metadata to disk."""
        if self._index is None:
            return
        faiss.write_index(self._index, str(self.index_path))
        self.metadata_path.write_text(json.dumps(self._metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def _initialise_index(self, dim: int) -> None:
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)

    def add_documents(self, embeddings: np.ndarray, documents: List[Document]) -> list[str]:
        """Add documents and embeddings to the FAISS index."""
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")
        if embeddings.shape[0] != len(documents):
            raise ValueError("Mismatch between embeddings and documents.")

        self._initialise_index(dim=embeddings.shape[1])
        if self._index is None:
            raise RuntimeError("Failed to initialise FAISS index.")

        self._index.add(embeddings)
        new_metadata = [doc.model_dump() for doc in documents]
        self._metadata.extend(new_metadata)
        self.save()
        return [doc.id for doc in documents]

    def query(self, embedding: np.ndarray, top_k: int = 5) -> list[QueryResult]:
        """Query the knowledge base for the most similar documents."""
        if not self.is_ready:
            return []

        scores, indices = self._index.search(embedding, top_k)
        results: list[QueryResult] = []
        for row_scores, row_indices in zip(scores, indices):
            for score, idx in zip(row_scores, row_indices):
                if idx == -1 or idx >= len(self._metadata):
                    continue
                metadata = self._metadata[idx]
                results.append(
                    QueryResult(score=float(score), document=Document.model_validate(metadata))
                )
        return results
