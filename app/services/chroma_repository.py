from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import Settings
from app.domain.models import ChunkRecord, FileDescriptor


class ChromaVectorRepository:
    """Vector store backed by a local persistent ChromaDB collection.

    Oracle modelled three tables (documents / chunks / embeddings). ChromaDB
    stores a vector, its source text, and arbitrary metadata together, so the
    three are collapsed into a single collection: one entry per chunk whose
    ``id`` is the chunk id, ``document`` is the chunk text, ``embedding`` is the
    vector, and ``metadata`` carries the denormalised document fields
    (``document_id``, ``file_name``, ``status``) plus the chunk metadata JSON.

    Document-level rows were only ever read back for the ``file_name`` and
    ``status`` columns during retrieval, so denormalising them onto each chunk
    preserves behaviour without a separate documents table. An in-memory
    registry tracks document metadata between the ``upsert_document`` calls and
    ``store_chunks_and_embeddings`` within a single ingestion run.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.collection_name = settings.chroma_collection
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self._get_or_create_collection()
        self._documents: dict[str, dict] = {}

    def _get_or_create_collection(self):
        # Cosine space mirrors Oracle's VECTOR_DISTANCE(..., COSINE); Chroma then
        # returns cosine distance (1 - similarity) which we convert back to a score.
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ping(self) -> bool:
        # heartbeat() raises if the persistent client is unreachable.
        self.client.heartbeat()
        return True

    def truncate_for_rebuild(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            # Collection may not exist yet on a first-time rebuild.
            pass
        self.collection = self._get_or_create_collection()
        self._documents.clear()

    def upsert_document(self, file_descriptor: FileDescriptor, status: str) -> None:
        document_id = file_descriptor.document_id
        self._documents[document_id] = {
            "file_name": file_descriptor.file_name,
            "file_path": str(file_descriptor.file_path),
            "file_type": file_descriptor.file_extension,
            "checksum": file_descriptor.checksum_sha256,
            "file_size": file_descriptor.file_size,
            "status": status,
        }

        # If chunks for this document already exist (e.g. the status flips to
        # INGESTED after storage), sync the denormalised fields onto them so
        # retrieval's status filter and file_name stay correct.
        existing = self.collection.get(where={"document_id": document_id})
        ids = existing.get("ids") or []
        if ids:
            metadatas = existing.get("metadatas") or []
            for metadata in metadatas:
                metadata["status"] = status
                metadata["file_name"] = file_descriptor.file_name
            self.collection.update(ids=ids, metadatas=metadatas)

    def replace_chunks_and_embeddings(
        self, document_id: str, chunks: list[ChunkRecord], embeddings: list[list[float]]
    ) -> None:
        """Replace a single document's chunks/embeddings (used by targeted ingestion)."""
        self.store_chunks_and_embeddings(document_id, chunks, embeddings, delete_existing=True)

    def store_chunks_and_embeddings(
        self,
        document_id: str,
        chunks: list[ChunkRecord],
        embeddings: list[list[float]],
        delete_existing: bool = True,
    ) -> None:
        """Persist chunks and embeddings.

        ``delete_existing`` removes any prior entries for the document first.
        During a full reindex the collection is already truncated, so callers
        pass ``False`` to skip a redundant delete per file.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings count mismatch")

        if delete_existing:
            self.collection.delete(where={"document_id": document_id})

        if not chunks:
            return

        doc_meta = self._documents.get(document_id, {})
        file_name = doc_meta.get("file_name", "")
        # Chunks are written while the document is still PROCESSING; the
        # subsequent upsert_document(status="INGESTED") flips them so they
        # become visible to retrieval's status filter.
        status = doc_meta.get("status", "INGESTED")

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.chunk_text for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "file_name": file_name,
                "status": status,
                "metadata_json": chunk.metadata_json,
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def fetch_all_chunks(self) -> list[tuple[str, str, str]]:
        """Return ``(chunk_id, chunk_text, metadata_json)`` for every stored chunk.

        Used to build the BM25 keyword index.
        """
        result = self.collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        rows: list[tuple[str, str, str]] = []
        for chunk_id, chunk_text, metadata in zip(ids, documents, metadatas):
            metadata = metadata or {}
            rows.append((chunk_id, chunk_text or "", metadata.get("metadata_json", "")))
        # Stable ordering keeps BM25 index positions deterministic across runs.
        rows.sort(key=lambda row: row[0])
        return rows

    def vector_search(
        self, query_embedding: list[float], limit: int
    ) -> list[tuple[str, str, str, str, str, float]]:
        """Cosine similarity search over INGESTED chunks.

        Returns ``(chunk_id, chunk_text, document_id, metadata_json, file_name,
        distance)`` tuples ordered by ascending distance, mirroring the columns
        the previous Oracle query produced.
        """
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where={"status": "INGESTED"},
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        rows: list[tuple[str, str, str, str, str, float]] = []
        for chunk_id, chunk_text, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            metadata = metadata or {}
            rows.append(
                (
                    chunk_id,
                    chunk_text or "",
                    metadata.get("document_id", ""),
                    metadata.get("metadata_json", ""),
                    metadata.get("file_name", ""),
                    float(distance),
                )
            )
        return rows

    def close(self) -> None:
        # PersistentClient flushes to disk on each write; nothing to close.
        pass
