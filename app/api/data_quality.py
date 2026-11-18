from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends

from app.config import Settings
from app.deps import get_settings
from app.errors import ServiceUnavailableError

router = APIRouter(tags=["data-quality"])
log = logging.getLogger(__name__)


@router.get("/data-quality/report")
def data_quality_report(settings: Settings = Depends(get_settings)) -> dict:
    """Serves the JSON written by the standalone DQ step (`python -m dq.dq_check`).

    Read on each request so a re-run of the DQ step is visible without restarting the API.
    """
    path = settings.dq_report_path
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise ServiceUnavailableError(
            "Data quality report has not been generated yet; run `python -m dq.dq_check`",
            code="DQ_REPORT_UNAVAILABLE",
        ) from None
    except json.JSONDecodeError as exc:
        log.error("dq_report_corrupt", extra={"path": str(path), "error": str(exc)})
        raise ServiceUnavailableError(
            "Data quality report is unreadable", code="DQ_REPORT_CORRUPT"
        ) from exc
