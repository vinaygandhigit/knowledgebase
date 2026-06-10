from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ReindexResponse(BaseModel):
    job_id: str
    status: str


class FileIngestionRequest(BaseModel):
    file_path: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    total_files: int
    processed_files: int
    failed_files: int
    total_chunks: int
    message: str


class HealthResponse(BaseModel):
    status: str
    app: str
    db: str


class RetrievalMetadata(BaseModel):
    chunk_id: str
    document_id: str
    file_name: str
    chunk_text: str = ""
    score: float
    search_type: str
    source_location: dict[str, Any] = Field(default_factory=dict)
    visual_refs: list[dict[str, str]] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str
    hybrid_alpha: float = 0.5


class QueryResponse(BaseModel):
    query: str
    response: str
    retrieved_chunks: list[RetrievalMetadata]
    execution_time_ms: float
