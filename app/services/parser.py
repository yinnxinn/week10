from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from app.models.document import Document


def chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 50) -> Iterable[str]:
    """Split raw text into overlapping chunks based on word count."""
    words = text.split()
    if not words:
        return

    stride = max(chunk_size - chunk_overlap, 1)
    for start in range(0, len(words), stride):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            continue
        yield " ".join(chunk_words)


def parse_text_file(path: Path, chunk_size: int = 300, chunk_overlap: int = 50) -> List[Document]:
    """Read a text file and explode it into chunk documents."""
    text = path.read_text(encoding="utf-8").strip()
    documents: List[Document] = []
    for idx, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)):
        documents.append(
            Document(
                id=f"{path.stem}-{idx}",
                text=chunk,
                metadata={"source_path": str(path), "chunk_index": str(idx)},
            )
        )
    return documents


def parse_directory(
    directory: Path, suffix: str = ".txt", chunk_size: int = 300, chunk_overlap: int = 50
) -> List[Document]:
    """Parse all text files in a directory into document chunks."""
    documents: List[Document] = []
    for file_path in directory.glob(f"*{suffix}"):
        documents.extend(parse_text_file(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return documents
