"""FastAPI dependencies resolving shared components from app.state."""

from __future__ import annotations

from fastapi import Request

from app.config import Settings
from app.errors import ServiceUnavailableError
from app.repository import ShipmentRepository
from app.services.predictor import Predictor


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_repo(request: Request) -> ShipmentRepository:
    repo = getattr(request.app.state, "repo", None)
    if repo is None:
        raise ServiceUnavailableError("Data store is not available", code="DATA_STORE_UNAVAILABLE")
    return repo


def get_predictor(request: Request) -> Predictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise ServiceUnavailableError("Prediction model is not loaded", code="MODEL_UNAVAILABLE")
    return predictor
