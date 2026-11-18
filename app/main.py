"""Application factory. `create_app()` accepts injected components so tests need no infra."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import data_quality, health, ports, predict, routes, shipments
from app.config import Settings
from app.errors import register_error_handlers
from app.middleware import RequestContextMiddleware
from app.repository import DuckDBRepository, ShipmentRepository
from app.services.predictor import ModelPredictor, Predictor

log = logging.getLogger("app")


def _load_components(app: FastAPI) -> None:
    settings: Settings = app.state.settings
    if app.state.repo is None:
        try:
            app.state.repo = DuckDBRepository(settings.db_path)
            log.info("data_store_opened", extra={"db_path": str(settings.db_path)})
        except Exception as exc:
            # Start anyway so /health can report *why* we are not ready.
            app.state.repo_error = str(exc)
            log.error("data_store_unavailable", extra={"error": str(exc)})

    if app.state.predictor is None and app.state.repo is not None:
        # Load the trained artefact once. No silent fallback: a missing/incompatible model
        # makes /health report degraded and /predict-delay return 503.
        try:
            app.state.predictor = ModelPredictor(settings.model_path, app.state.repo.connection)
            log.info(
                "model_loaded",
                extra={
                    "model_version": app.state.predictor.version,
                    "model_path": str(settings.model_path),
                },
            )
        except Exception as exc:
            app.state.model_error = f"{type(exc).__name__}: {exc}"
            log.error("model_unavailable", extra={"error": app.state.model_error})


def create_app(
    settings: Settings | None = None,
    repo: ShipmentRepository | None = None,
    predictor: Predictor | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _load_components(app)
        log.info(
            "startup_complete", extra={"version": settings.version, "git_sha": settings.git_sha}
        )
        yield
        close = getattr(app.state.repo, "close", None)
        if close:
            close()

    app = FastAPI(
        title="Supply Chain Intelligence API",
        version=settings.version,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.repo = repo
    app.state.predictor = predictor

    register_error_handlers(app)
    app.add_middleware(RequestContextMiddleware)
    for module in (health, shipments, routes, ports, predict, data_quality):
        app.include_router(module.router)
    return app
