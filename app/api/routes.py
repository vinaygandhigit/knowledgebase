from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import FileResponse

from app.api.schemas import (
    FileIngestionRequest,
    HealthResponse,
    JobStatusResponse,
    QueryRequest,
    QueryResponse,
    ReindexResponse,
    RetrievalMetadata,
)
from app.core.logging import get_logger
from app.core.container import AppContainer
from app.core.security import is_within_root

router = APIRouter()
TESTER_PAGE = Path(__file__).resolve().parent.parent / "static" / "rag_tester.html"
logger = get_logger(component="api_routes")


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


async def _run_reindex_and_refresh_index(
    container: AppContainer,
    job_id: str,
    correlation_id: str,
) -> None:
    await container.ingestion_service.run_full_reindex(job_id, correlation_id)
    try:
        container.initialize_retriever()
    except Exception as error:
        # Keep job completion intact even if keyword index refresh fails.
        logger.warning("BM25 refresh failed after reindex", job_id=job_id, error=str(error))


async def _run_targeted_ingestion_and_refresh_index(
    container: AppContainer,
    job_id: str,
    correlation_id: str,
    file_path: str,
) -> None:
    await container.ingestion_service.run_targeted_ingestion(
        job_id=job_id,
        correlation_id=correlation_id,
        file_paths=[Path(file_path)],
    )
    try:
        container.initialize_retriever()
    except Exception as error:
        # Keep job completion intact even if keyword index refresh fails.
        logger.warning(
            "BM25 refresh failed after targeted ingestion",
            job_id=job_id,
            file_path=file_path,
            error=str(error),
        )


def _format_context(results) -> str:
    """Build a citation-friendly context block from retrieved chunks so the
    model can attribute answers to a specific file / page / slide / section."""
    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        location = result.source_location or {}
        location_bits: list[str] = []
        if location.get("page_number"):
            location_bits.append(f"page {location['page_number']}")
        if location.get("slide_number"):
            location_bits.append(f"slide {location['slide_number']}")
        if location.get("section_title"):
            location_bits.append(str(location["section_title"]))
        suffix = f" ({', '.join(location_bits)})" if location_bits else ""
        blocks.append(f"[Source {index}: {result.file_name}{suffix}]\n{result.chunk_text}")
    return "\n\n".join(blocks)


@router.get("/rag-tester", include_in_schema=False)
async def rag_tester() -> FileResponse:
    return FileResponse(TESTER_PAGE)


@router.post(
    "/api/v1/ingestion/reindex",
    response_model=ReindexResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def reindex(request: Request) -> ReindexResponse:
    container = get_container(request)
    correlation_id = request.state.correlation_id
    job_id = str(uuid.uuid4())
    await container.job_store.create(job_id)

    asyncio.create_task(_run_reindex_and_refresh_index(container, job_id, correlation_id))

    return ReindexResponse(job_id=job_id, status="RUNNING")


@router.post(
    "/api/v1/ingestion/file",
    response_model=ReindexResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_single_file(payload: FileIngestionRequest, request: Request) -> ReindexResponse:
    container = get_container(request)
    correlation_id = request.state.correlation_id
    job_id = str(uuid.uuid4())
    await container.job_store.create(job_id)

    asyncio.create_task(
        _run_targeted_ingestion_and_refresh_index(
            container=container,
            job_id=job_id,
            correlation_id=correlation_id,
            file_path=payload.file_path,
        )
    )

    return ReindexResponse(job_id=job_id, status="RUNNING")


@router.get("/api/v1/ingestion/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, request: Request) -> JobStatusResponse:
    container = get_container(request)
    job = await container.job_store.to_dict(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return JobStatusResponse(**job)


@router.get("/actuator/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    container = get_container(request)
    try:
        db_ok = container.repository.ping()
    except Exception:
        db_ok = False

    return HealthResponse(
        status="UP" if db_ok else "DEGRADED",
        app=container.settings.app_name,
        db="UP" if db_ok else "DOWN",
    )


@router.get("/api/v1/assets")
async def get_asset(path: str = Query(..., description="Absolute path to an extracted visual asset"), request: Request = None):
    if request is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Request context is required")

    container = get_container(request)
    root = container.settings.root_path.resolve()
    candidate = Path(path).resolve()

    if not is_within_root(root, candidate):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Asset path is outside ROOT_PATH")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")

    return FileResponse(candidate)


@router.post("/api/v1/retrieval/query", response_model=QueryResponse)
async def query(query_req: QueryRequest, request: Request) -> QueryResponse:
    container = get_container(request)
    correlation_id = request.state.correlation_id
    start_time = time.perf_counter()

    try:
        retrieved = container.retriever.hybrid_search(
            query=query_req.query,
            alpha=query_req.hybrid_alpha,
        )

        logger.info(
            "Retrieved chunks for query",
            chunk_count=len(retrieved),
            query=query_req.query,
            correlation_id=correlation_id,
        )

        if not retrieved:
            return QueryResponse(
                query=query_req.query,
                response="No relevant documents found for your query.",
                retrieved_chunks=[],
                execution_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )

        context = _format_context(retrieved)
        response_text = container.llm_provider.generate_response(
            context=context,
            query=query_req.query,
        )

        metadata = [
            RetrievalMetadata(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                file_name=result.file_name,
                chunk_text=result.chunk_text,
                score=result.score,
                search_type=result.search_type,
                source_location=result.source_location,
                visual_refs=result.visual_refs,
            )
            for result in retrieved
        ]

        execution_time = round((time.perf_counter() - start_time) * 1000, 2)

        return QueryResponse(
            query=query_req.query,
            response=response_text,
            retrieved_chunks=metadata,
            execution_time_ms=execution_time,
        )
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(error)}",
        ) from error
