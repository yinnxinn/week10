from __future__ import annotations

from typing import List
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from app.models.document import (
    Document,
    IngestFromTextRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.embeddings import EmbeddingService, get_embedding_service
from app.services.knowledge_base import KnowledgeBase


def get_knowledge_base() -> KnowledgeBase:
    from app.core.config import settings

    return KnowledgeBase(index_path=settings.index_path, metadata_path=settings.metadata_path)


router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/ingest", response_model=IngestResponse)
def ingest_text(
    payload: IngestFromTextRequest,
    embeddings: EmbeddingService = Depends(get_embedding_service),
    knowledge_base: KnowledgeBase = Depends(get_knowledge_base),
) -> IngestResponse:
    document_id = payload.document_id or str(uuid4())
    document = Document(id=document_id, text=payload.text, metadata=payload.metadata)
    vector = embeddings.embed_documents([document.text])
    ids = knowledge_base.add_documents(vector, [document])
    return IngestResponse(document_ids=ids)


@router.post("/query", response_model=QueryResponse)
def query_knowledge_base(
    payload: QueryRequest,
    embeddings: EmbeddingService = Depends(get_embedding_service),
    knowledge_base: KnowledgeBase = Depends(get_knowledge_base),
) -> QueryResponse:
    if payload.top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1.")
    query_vector = embeddings.embed_query(payload.query)
    results = knowledge_base.query(query_vector, top_k=payload.top_k)
    return QueryResponse(results=results)
