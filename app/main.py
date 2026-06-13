from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request

from app.api.routes import router
from app.core.config import settings
from app.core.container import AppContainer
from app.core.logging import configure_logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    container = AppContainer()
    app.state.container = container
    try:
        container.initialize_retriever()
    except Exception as error:
        structlog.get_logger().warning(
            "Failed to initialize retriever on startup",
            error=str(error),
        )
    yield
    container.shutdown()


def create_app() -> FastAPI:
    configure_logging(settings.log_level)

    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def correlation_middleware(request: Request, call_next):
        request_id = request.headers.get("x-correlation-id", str(uuid.uuid4()))
        request.state.correlation_id = request_id

        structlog.contextvars.bind_contextvars(correlation_id=request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            structlog.get_logger().info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                execution_time_ms=elapsed,
            )
            structlog.contextvars.clear_contextvars()

    app.include_router(router)
    return app


app = create_app()
