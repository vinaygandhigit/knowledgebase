from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import structlog

from app.core.config import Settings
from app.core.logging import get_logger
from app.domain.models import JobStatus
from app.services.chunking import SemanticChunkingService
from app.services.embeddings.base import EmbeddingProvider
from app.services.chroma_repository import ChromaVectorRepository
from app.services.parsers.factory import ParserFactory
from app.services.scanner import FileScanner

logger = get_logger(component="ingestion_service")


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}
        self._lock = asyncio.Lock()

    async def create(self, job_id: str) -> JobStatus:
        async with self._lock:
            job = JobStatus(job_id=job_id, status="RUNNING", started_at=datetime.now(timezone.utc))
            self._jobs[job_id] = job
            return job

    async def get(self, job_id: str) -> JobStatus | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def update(self, job_id: str, **kwargs) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in kwargs.items():
                setattr(job, key, value)

    async def to_dict(self, job_id: str) -> dict | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return asdict(job)


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        scanner: FileScanner,
        parser_factory: ParserFactory,
        chunking_service: SemanticChunkingService,
        embedding_provider: EmbeddingProvider,
        repository: ChromaVectorRepository,
        job_store: JobStore,
    ) -> None:
        self.settings = settings
        self.scanner = scanner
        self.parser_factory = parser_factory
        self.chunking_service = chunking_service
        self.embedding_provider = embedding_provider
        self.repository = repository
        self.job_store = job_store

    def _process_file_sync(self, file_descriptor, delete_existing: bool = True) -> tuple[str, int, str | None]:
        file_start = time.perf_counter()
        try:
            parser = self.parser_factory.get_parser(file_descriptor.file_extension)
            parsed_document = parser.parse(file_descriptor)

            self.repository.upsert_document(file_descriptor, status="PROCESSING")

            chunks = self.chunking_service.chunk_document(file_descriptor, parsed_document)
            # Single call — the embedding provider batches internally.
            embeddings = self.embedding_provider.embed_documents([chunk.chunk_text for chunk in chunks])

            self.repository.store_chunks_and_embeddings(
                document_id=file_descriptor.document_id,
                chunks=chunks,
                embeddings=embeddings,
                delete_existing=delete_existing,
            )
            self.repository.upsert_document(file_descriptor, status="INGESTED")

            logger.info(
                "File ingested",
                file_path=str(file_descriptor.file_path),
                chunk_count=len(chunks),
                execution_time_ms=round((time.perf_counter() - file_start) * 1000, 2),
            )
            return file_descriptor.file_name, len(chunks), None
        except Exception as error:
            try:
                self.repository.upsert_document(file_descriptor, status="FAILED")
            except Exception:
                logger.exception(
                    "Failed to update document status to FAILED",
                    file_path=str(file_descriptor.file_path),
                )

            logger.exception(
                "File ingestion failed",
                file_path=str(file_descriptor.file_path),
                error=str(error),
            )
            return file_descriptor.file_name, 0, str(error)

    async def _process_file(
        self, semaphore: asyncio.Semaphore, file_descriptor, delete_existing: bool = True
    ) -> tuple[str, int, str | None]:
        async with semaphore:
            return await asyncio.to_thread(self._process_file_sync, file_descriptor, delete_existing)

    async def run_full_reindex(self, job_id: str, correlation_id: str) -> None:
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id, job_id=job_id)
        start_time = time.perf_counter()

        try:
            await self.job_store.update(job_id, status="RUNNING", message="Starting reindex")
            self.repository.truncate_for_rebuild()

            files = self.scanner.scan()
            await self.job_store.update(job_id, total_files=len(files), message="Files discovered")

            if files:
                worker_count = max(1, self.settings.worker_count)
                semaphore = asyncio.Semaphore(worker_count)
                # Tables were just truncated — skip per-file deletes.
                tasks = [
                    asyncio.create_task(
                        self._process_file(semaphore, file_descriptor, delete_existing=False)
                    )
                    for file_descriptor in files
                ]

                processed_files = 0
                failed_files = 0
                total_chunks = 0

                for task in asyncio.as_completed(tasks):
                    file_name, chunk_count, error_message = await task
                    processed_files += 1
                    total_chunks += chunk_count
                    if error_message:
                        failed_files += 1
                        status_message = f"Failed on file {file_name}: {error_message}"
                    else:
                        status_message = "Ingestion in progress"

                    await self.job_store.update(
                        job_id,
                        processed_files=processed_files,
                        failed_files=failed_files,
                        total_chunks=total_chunks,
                        message=status_message,
                    )

            await self.job_store.update(
                job_id,
                status="COMPLETED",
                finished_at=datetime.now(timezone.utc),
                message="Reindex completed",
            )
            logger.info(
                "Reindex job completed",
                execution_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )
        except Exception as error:
            await self.job_store.update(
                job_id,
                status="FAILED",
                finished_at=datetime.now(timezone.utc),
                message=str(error),
            )
            logger.exception("Reindex job failed", error=str(error))
        finally:
            structlog.contextvars.clear_contextvars()

    async def run_targeted_ingestion(
        self,
        job_id: str,
        correlation_id: str,
        file_paths: list[Path],
    ) -> None:
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id, job_id=job_id)
        start_time = time.perf_counter()

        try:
            await self.job_store.update(job_id, status="RUNNING", message="Starting targeted ingestion")

            descriptors = [self.scanner.scan_single(path) for path in file_paths]
            await self.job_store.update(job_id, total_files=len(descriptors), message="Files discovered")

            worker_count = max(1, self.settings.worker_count)
            semaphore = asyncio.Semaphore(worker_count)
            tasks = [
                asyncio.create_task(self._process_file(semaphore, file_descriptor))
                for file_descriptor in descriptors
            ]

            processed_files = 0
            failed_files = 0
            total_chunks = 0

            for task in asyncio.as_completed(tasks):
                file_name, chunk_count, error_message = await task
                processed_files += 1
                total_chunks += chunk_count
                if error_message:
                    failed_files += 1
                    status_message = f"Failed on file {file_name}: {error_message}"
                else:
                    status_message = "Ingestion in progress"

                await self.job_store.update(
                    job_id,
                    processed_files=processed_files,
                    failed_files=failed_files,
                    total_chunks=total_chunks,
                    message=status_message,
                )

            await self.job_store.update(
                job_id,
                status="COMPLETED",
                finished_at=datetime.now(timezone.utc),
                message="Targeted ingestion completed",
            )
            logger.info(
                "Targeted ingestion job completed",
                execution_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
                file_count=len(descriptors),
            )
        except Exception as error:
            await self.job_store.update(
                job_id,
                status="FAILED",
                finished_at=datetime.now(timezone.utc),
                message=str(error),
            )
            logger.exception("Targeted ingestion job failed", error=str(error))
        finally:
            structlog.contextvars.clear_contextvars()
