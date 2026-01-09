from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import settings
from app.services.embeddings import get_embedding_service
from app.services.knowledge_base import KnowledgeBase
from app.services.parser import parse_directory


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest text files into the knowledge base.")
    parser.add_argument(
        "--source",
        type=Path,
        default=settings.raw_data_dir,
        help="Directory containing *.txt files.",
    )
    parser.add_argument("--chunk-size", type=int, default=300, help="Number of words per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap in words between chunks.")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    settings.ensure_directories()
    source_dir: Path = args.source
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")

    documents = parse_directory(
        directory=source_dir, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    if not documents:
        print("No documents found for ingestion.")
        return

    embedding_service = get_embedding_service()
    embeddings = embedding_service.embed_documents([doc.text for doc in documents])

    knowledge_base = KnowledgeBase(index_path=settings.index_path, metadata_path=settings.metadata_path)

    ids = knowledge_base.add_documents(embeddings, documents)
    print(f"Ingested {len(ids)} document chunks into the knowledge base.")


if __name__ == "__main__":
    main()
