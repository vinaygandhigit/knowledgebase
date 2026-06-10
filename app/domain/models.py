from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class FileDescriptor:
    document_id: str
    file_name: str
    file_path: Path
    file_extension: str
    file_size: int
    created_timestamp: datetime
    modified_timestamp: datetime
    checksum_sha256: str


@dataclass(slots=True)
class ParsedSection:
    text: str
    page_number: int | None = None
    slide_number: int | None = None
    section_title: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedDocument:
    document_id: str
    sections: list[ParsedSection] = field(default_factory=list)


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    chunk_index: int
    chunk_text: str
    metadata_json: str


@dataclass(slots=True)
class JobStatus:
    job_id: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    message: str = ""
