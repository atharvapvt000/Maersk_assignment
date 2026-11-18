"""Delay-risk predictor: the API depends only on the `Predictor` protocol.

`ModelPredictor` loads the artefact written by `ml.train` ONCE at startup and computes features
with the exact function used in training (`ml.features.build_features`), against the same
serving tables, so training and serving cannot drift apart.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

import duckdb
import joblib
import pandas as pd
import sklearn

log = logging.getLogger(__name__)

RiskBand = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class BookingFeatures:
    """Everything known at booking time. Deliberately has no actual_* fields or status."""

    origin_port: str
    destination_port: str
    planned_departure: datetime
    planned_arrival: datetime
    booking_date: datetime
    cargo_type: str | None = None
    container_count: int | None = None
    weight_tons: float | None = None
    vessel_id: str | None = None

    @property
    def route_key(self) -> str:
        return f"{self.origin_port} → {self.destination_port}"


class Predictor(Protocol):
    version: str
    threshold: float

    def predict_proba(self, features: BookingFeatures) -> float: ...


def risk_band(probability: float) -> RiskBand:
    if probability < 0.3:
        return "low"
    if probability < 0.6:
        return "medium"
    return "high"


class ModelPredictor:
    def __init__(self, artefact_path: Path, con: duckdb.DuckDBPyConnection) -> None:
        from ml.features import FEATURES  # local import: keeps app importable without ml deps

        bundle = joblib.load(artefact_path)
        if bundle["features"] != FEATURES:
            raise ValueError("Model artefact feature list does not match ml.features.FEATURES")
        if bundle.get("sklearn_version") != sklearn.__version__:
            log.warning(
                "sklearn_version_mismatch",
                extra={
                    "trained_with": bundle.get("sklearn_version"),
                    "running": sklearn.__version__,
                },
            )
        self._pipeline = bundle["pipeline"]
        self._features = FEATURES
        self._con = con
        self.version: str = bundle["model_version"]
        self.threshold: float = float(bundle["threshold"])

    def predict_proba(self, features: BookingFeatures) -> float:
        from ml.features import build_features

        request = pd.DataFrame([{"req_id": "request", **asdict(features)}])
        X = build_features(self._con, request)
        return float(self._pipeline.predict_proba(X[self._features])[0, 1])
