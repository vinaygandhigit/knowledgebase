from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

from app.core.config import settings
from app.core.logging import configure_logging
from app.services.chunking import SemanticChunkingService
from app.services.embeddings.factory import EmbeddingFactory
from app.services.ingestion_service import IngestionService, JobStore
from app.services.chroma_repository import ChromaVectorRepository
from app.services.parsers.factory import ParserFactory
from app.services.scanner import FileScanner


def build_ingestion_service() -> tuple[IngestionService, JobStore, ChromaVectorRepository]:
    embedding_provider = EmbeddingFactory.create(settings)
    chunking_service = SemanticChunkingService(settings, embedding_provider)
    scanner = FileScanner(settings)
    parser_factory = ParserFactory()
    repository = ChromaVectorRepository(settings)
    job_store = JobStore()

    ingestion_service = IngestionService(
        settings=settings,
        scanner=scanner,
        parser_factory=parser_factory,
        chunking_service=chunking_service,
        embedding_provider=embedding_provider,
        repository=repository,
        job_store=job_store,
    )
    return ingestion_service, job_store, repository


async def run_ingestion(root_path: str | None, file_path: str | None) -> int:
    configure_logging(settings.log_level)

    original_root_path = settings.root_path
    if root_path:
        settings.root_path = Path(root_path)

    ingestion_service, job_store, repository = build_ingestion_service()
    job_id = str(uuid.uuid4())

    try:
        await job_store.create(job_id)
        if file_path:
            await ingestion_service.run_targeted_ingestion(
                job_id,
                correlation_id=f"console-{job_id}",
                file_paths=[Path(file_path)],
            )
        else:
            await ingestion_service.run_full_reindex(job_id, correlation_id=f"console-{job_id}")
        job = await job_store.to_dict(job_id)

        if not job:
            print("Ingestion finished, but job status was unavailable.")
            return 1

        print(f"job_id: {job['job_id']}")
        print(f"status: {job['status']}")
        print(f"processed_files: {job['processed_files']}/{job['total_files']}")
        print(f"failed_files: {job['failed_files']}")
        print(f"total_chunks: {job['total_chunks']}")
        print(f"message: {job['message']}")
        return 0 if job["status"] == "COMPLETED" else 1
    finally:
        settings.root_path = original_root_path
        repository.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG ingestion from the console.")
    parser.add_argument(
        "--root-path",
        dest="root_path",
        help="Optional override for ROOT_PATH from the environment.",
    )
    parser.add_argument(
        "--file-path",
        dest="file_path",
        help="Optional file path to ingest only that file without full reindex.",
    )
    args = parser.parse_args()
    return asyncio.run(run_ingestion(args.root_path, args.file_path))


if __name__ == "__main__":
    sys.exit(main())
