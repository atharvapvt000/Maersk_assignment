"""Runtime configuration, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app import __version__


@dataclass(frozen=True)
class Settings:
    db_path: Path
    dq_report_path: Path
    model_path: Path
    version: str
    git_sha: str
    log_level: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            db_path=Path(os.getenv("DB_PATH", "data/warehouse/supply_chain.duckdb")),
            dq_report_path=Path(os.getenv("DQ_REPORT_PATH", "data/reports/dq_report.json")),
            model_path=Path(os.getenv("MODEL_PATH", "models/delay_model.joblib")),
            version=os.getenv("APP_VERSION", __version__),
            git_sha=os.getenv("GIT_SHA", "unknown"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
