from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents a knowledge base chunk with optional metadata."""

    id: str = Field(description="Unique identifier for the document chunk.")
    text: str = Field(description="Raw text content of the document chunk.")
    metadata: Dict[str, str] = Field(default_factory=dict)


class QueryResult(BaseModel):
    """A single retrieval result from the knowledge base."""

    score: float
    document: Document


class QueryResponse(BaseModel):
    """API response schema for knowledge base queries."""

    results: list[QueryResult]


class IngestResponse(BaseModel):
    """API response schema for ingestion actions."""

    document_ids: list[str]


class IngestFromTextRequest(BaseModel):
    """Request payload for ingesting ad-hoc text via the API."""

    text: str
    document_id: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Request payload for semantic search queries."""

    query: str
    top_k: int = 5
