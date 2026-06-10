from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone

from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker

from app.core.config import Settings
from app.domain.models import ChunkRecord, FileDescriptor, ParsedDocument
from app.services.embeddings.base import EmbeddingProvider


class LangChainEmbeddingAdapter(Embeddings):
    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.provider.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.provider.embed_query(text)


class SemanticChunkingService:
    """Semantic chunking with size guard-rails.

    SemanticChunker alone can emit chunks that are either too large (exceeding
    the embedding model's window, so the tail is silently truncated before
    embedding) or too small (single sentences that add retrieval noise). This
    service runs the semantic splitter first, then normalises chunk sizes:
    oversized chunks are split on natural boundaries with overlap, and
    undersized chunks are merged with neighbours within the same section.
    """

    def __init__(self, settings: Settings, embedding_provider: EmbeddingProvider) -> None:
        self.settings = settings
        self.max_chars = max(settings.chunk_max_chars, 1)
        self.min_chars = max(settings.chunk_min_chars, 0)
        self.overlap_chars = max(settings.chunk_overlap_chars, 0)
        self.prepend_section_title = settings.chunk_prepend_section_title
        self.strategy = settings.chunk_strategy.lower().strip()

        # The semantic chunker embeds every sentence to find breakpoints — a large
        # cost during ingestion. Only build it when the semantic strategy is in use;
        # the "recursive" strategy avoids chunk-time embeddings entirely.
        self._chunker: SemanticChunker | None = None
        if self.strategy == "semantic":
            self._chunker = SemanticChunker(
                embeddings=LangChainEmbeddingAdapter(embedding_provider),
                breakpoint_threshold_type=settings.chunk_breakpoint_threshold_type,
                breakpoint_threshold_amount=settings.chunk_breakpoint_threshold_amount,
            )

    @staticmethod
    def _split_oversized(text: str, max_chars: int, overlap: int) -> list[str]:
        """Split text larger than max_chars on the most natural boundary available
        (paragraph -> sentence -> hard cut), carrying a small overlap forward."""
        if len(text) <= max_chars:
            return [text]

        # Prefer paragraph boundaries, then sentence boundaries.
        units = re.split(r"\n\s*\n", text)
        if len(units) == 1:
            units = re.split(r"(?<=[.!?])\s+", text)

        pieces: list[str] = []
        buffer = ""
        for unit in units:
            unit = unit.strip()
            if not unit:
                continue
            if not buffer:
                buffer = unit
            elif len(buffer) + len(unit) + 1 <= max_chars:
                buffer = f"{buffer} {unit}"
            else:
                pieces.append(buffer)
                tail = buffer[-overlap:] if overlap else ""
                buffer = f"{tail} {unit}".strip() if tail else unit
        if buffer:
            pieces.append(buffer)

        # A single unit may still exceed max_chars; hard-cut it as a last resort.
        result: list[str] = []
        for piece in pieces:
            if len(piece) <= max_chars:
                result.append(piece)
                continue
            step = max(max_chars - overlap, 1)
            for start in range(0, len(piece), step):
                result.append(piece[start : start + max_chars])
        return result

    def _normalize_sizes(self, pieces: list[str]) -> list[str]:
        expanded: list[str] = []
        for piece in pieces:
            expanded.extend(self._split_oversized(piece, self.max_chars, self.overlap_chars))

        merged: list[str] = []
        for piece in expanded:
            if (
                merged
                and len(merged[-1]) < self.min_chars
                and len(merged[-1]) + len(piece) + 1 <= self.max_chars
            ):
                merged[-1] = f"{merged[-1]}\n{piece}"
            else:
                merged.append(piece)
        return merged

    def _split_table(self, text: str) -> list[str]:
        """Tables must stay intact for the UI to re-render them, so they bypass the
        semantic splitter. Oversized tables are split on row boundaries with the
        ``[TABLE]`` header repeated on each part."""
        if len(text) <= self.max_chars:
            return [text]

        lines = text.split("\n")
        header = lines[0] if lines and lines[0].lstrip().startswith("[TABLE]") else "[TABLE]"
        rows = lines[1:] if lines and lines[0].lstrip().startswith("[TABLE]") else lines

        parts: list[str] = []
        current: list[str] = []
        current_len = len(header)
        for row in rows:
            if current and current_len + len(row) + 1 > self.max_chars:
                parts.append(header + "\n" + "\n".join(current))
                current = []
                current_len = len(header)
            current.append(row)
            current_len += len(row) + 1
        if current:
            parts.append(header + "\n" + "\n".join(current))
        return parts

    @staticmethod
    def _is_table(text: str) -> bool:
        return text.lstrip().startswith("[TABLE]")

    def _pieces_for_section(self, section) -> list[str]:
        text = section.text.strip()
        if self._is_table(text):
            return self._split_table(text)

        if self._chunker is None:
            # Recursive strategy: boundary-based splitting, no embeddings.
            return self._normalize_sizes([text])

        documents = self._chunker.create_documents(texts=[text])
        raw_pieces = [doc.page_content.strip() for doc in documents if doc.page_content.strip()]
        return self._normalize_sizes(raw_pieces)

    def chunk_document(self, file_descriptor: FileDescriptor, parsed: ParsedDocument) -> list[ChunkRecord]:
        chunks_data: list[dict[str, object]] = []
        now = datetime.now(timezone.utc).isoformat()

        for section in parsed.sections:
            text = section.text.strip()
            if not text:
                continue

            section_title = section.section_title or ""
            visual_refs = section.metadata.get("visual_refs", [])
            source_location = {
                "page_number": section.page_number,
                "slide_number": section.slide_number,
                "section_title": section_title,
            }

            for piece in self._pieces_for_section(section):
                chunk_text = piece
                if (
                    self.prepend_section_title
                    and section_title
                    and section_title not in piece
                ):
                    chunk_text = f"{section_title}\n{piece}"

                chunks_data.append(
                    {
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_text": chunk_text,
                        "source_location": source_location,
                        "visual_refs": visual_refs,
                        "created_at": now,
                    }
                )

        records: list[ChunkRecord] = []
        for index, chunk_data in enumerate(chunks_data):
            prev_chunk_id = chunks_data[index - 1]["chunk_id"] if index > 0 else None
            next_chunk_id = chunks_data[index + 1]["chunk_id"] if index < len(chunks_data) - 1 else None

            source_location = chunk_data["source_location"]
            metadata = {
                "document_id": file_descriptor.document_id,
                "file_name": file_descriptor.file_name,
                "file_path": str(file_descriptor.file_path),
                "file_type": file_descriptor.file_extension,
                "checksum": file_descriptor.checksum_sha256,
                "chunk_id": chunk_data["chunk_id"],
                "chunk_index": index,
                "page_number": source_location.get("page_number"),
                "slide_number": source_location.get("slide_number"),
                "section_title": source_location.get("section_title", ""),
                "previous_chunk_id": prev_chunk_id,
                "next_chunk_id": next_chunk_id,
                "source_location": source_location,
                "visual_refs": chunk_data.get("visual_refs", []),
                "created_at": chunk_data["created_at"],
                "updated_at": chunk_data["created_at"],
            }

            records.append(
                ChunkRecord(
                    chunk_id=chunk_data["chunk_id"],
                    document_id=file_descriptor.document_id,
                    chunk_index=index,
                    chunk_text=chunk_data["chunk_text"],
                    metadata_json=json.dumps(metadata, ensure_ascii=True),
                )
            )

        return records
