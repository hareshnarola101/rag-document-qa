import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api import documents, health, qa
from app.container import Services, build_services
from app.core.config import Settings, get_settings
from app.ingest.parsers import DocumentParseError, UnsupportedFileTypeError
from app.ingest.pipeline import EmptyDocumentError
from app.providers.base import ProviderError

logger = logging.getLogger(__name__)

ExceptionHandler = Callable[[Request, Exception], Awaitable[JSONResponse]]


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    """Build the app. Passing `services` lets tests inject their own (and own their lifecycle)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        app.state.services = services or build_services(settings or get_settings())
        try:
            yield
        finally:
            if services is None:
                await app.state.services.aclose()

    app = FastAPI(
        title="rag-document-qa",
        version="0.1.0",
        description="Question answering over your own documents, with cited sources.",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(qa.router)
    _register_error_handlers(app)
    return app


def _error_handler(status_code: int, *, log: bool = False) -> ExceptionHandler:
    async def handle(request: Request, exc: Exception) -> JSONResponse:
        if log:
            logger.error("%s %s failed: %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    return handle


def _register_error_handlers(app: FastAPI) -> None:
    unprocessable = _error_handler(status.HTTP_422_UNPROCESSABLE_CONTENT)
    app.add_exception_handler(
        UnsupportedFileTypeError, _error_handler(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
    )
    app.add_exception_handler(DocumentParseError, unprocessable)
    app.add_exception_handler(EmptyDocumentError, unprocessable)
    app.add_exception_handler(ProviderError, _error_handler(status.HTTP_502_BAD_GATEWAY, log=True))


app = create_app()
