from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas import HealthResponse

router = APIRouter(tags=["ops"])
log = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> JSONResponse:
    """Liveness + readiness. 200 when the data store and model are usable, else 503 (degraded).

    The DQ report is informational: its absence only affects /data-quality/report.
    """
    state = request.app.state
    settings = state.settings
    checks: dict[str, dict[str, object]] = {}

    repo = getattr(state, "repo", None)
    if repo is None:
        checks["data_store"] = {"ok": False, "error": getattr(state, "repo_error", "not loaded")}
    else:
        try:
            checks["data_store"] = {"ok": True, **repo.ping()}
        except Exception as exc:  # report, never crash the health check
            log.warning("health_db_check_failed", extra={"error": str(exc)})
            checks["data_store"] = {"ok": False, "error": type(exc).__name__}

    predictor = getattr(state, "predictor", None)
    checks["model"] = (
        {"ok": True, "version": predictor.version}
        if predictor is not None
        else {"ok": False, "error": getattr(state, "model_error", "not loaded")}
    )
    checks["dq_report"] = {"ok": settings.dq_report_path.exists(), "critical": False}

    healthy = checks["data_store"]["ok"] and checks["model"]["ok"]
    body = HealthResponse(
        status="ok" if healthy else "degraded",
        version=settings.version,
        git_sha=settings.git_sha,
        checks=checks,
    )
    return JSONResponse(status_code=200 if healthy else 503, content=body.model_dump(mode="json"))
