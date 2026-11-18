"""ML: leakage guards, point-in-time correctness of features, metric helpers, committed artefact."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from app.services.predictor import BookingFeatures, ModelPredictor
from ml import train
from ml.features import FEATURES, FORBIDDEN, build_features

ROOT = Path(__file__).resolve().parents[2]
ARTEFACT = ROOT / "models" / "delay_model.joblib"
CARD = ROOT / "models" / "model_card.json"


@pytest.fixture
def con(built_fixture_db):
    c = duckdb.connect(str(built_fixture_db), read_only=True)
    yield c
    c.close()


def _request(**kw) -> pd.DataFrame:
    row = {
        "req_id": "r1",
        "origin_port": "CNSHA",
        "destination_port": "NLRTM",
        "cargo_type": "Electronics",
        "container_count": 10,
        "weight_tons": 100.0,
        "vessel_id": "V1",
        "booking_date": datetime(2024, 3, 1),
        "planned_departure": datetime(2024, 3, 5),
        "planned_arrival": datetime(2024, 4, 1),
    }
    row.update(kw)
    return pd.DataFrame([row])


# ----------------------------------------------------------------------------- leakage guards


def test_no_forbidden_feature():
    assert not set(FEATURES) & FORBIDDEN


def test_booking_features_have_no_post_booking_fields():
    fields = set(BookingFeatures.__dataclass_fields__)
    assert not fields & FORBIDDEN


# ----------------------------------------------------------------------------- point in time


def test_route_history_only_counts_shipments_arrived_before_booking(con):
    # CNSHA → NLRTM arrivals in the fixture: SHP-00001 (2024-02-02 00:00, on time),
    # SHP-00002 (2024-02-03 00:00, 48h late).
    early = build_features(con, _request(booking_date=datetime(2024, 1, 15)))
    assert early.loc[0, "route_hist_n"] == 0
    mid = build_features(con, _request(booking_date=datetime(2024, 2, 2, 12)))
    assert mid.loc[0, "route_hist_n"] == 1
    late = build_features(con, _request(booking_date=datetime(2024, 3, 1)))
    assert late.loc[0, "route_hist_n"] == 2
    # smoothed: (1 late + 10 * global 1/3) / (2 + 10); global = 1 late of 3 arrivals by then
    assert late.loc[0, "route_hist_late_rate"] == pytest.approx((1 + 10 / 3) / 12)


def test_port_activity_excludes_the_booking_day_and_later(con):
    # NLRTM events: 2024-01-01 (30 min), 2024-02-01 (15 min).
    same_day = build_features(con, _request(booking_date=datetime(2024, 2, 1, 23)))
    assert same_day.loc[0, "destination_delay_minutes_30d"] == 0  # 02-01 not yet "past"
    next_day = build_features(con, _request(booking_date=datetime(2024, 2, 2)))
    assert next_day.loc[0, "destination_delay_minutes_30d"] == 15


def test_booking_after_snapshot_uses_latest_history(con):
    future = build_features(con, _request(booking_date=datetime(2030, 1, 1)))
    asof = build_features(con, _request(booking_date=datetime(2024, 3, 1)))
    assert future.loc[0, "route_hist_n"] == asof.loc[0, "route_hist_n"] == 2


def test_missing_categoricals_become_explicit_category(con):
    out = build_features(con, _request(cargo_type=None, vessel_id=None))
    assert out.loc[0, "cargo_type"] == "MISSING" and out.loc[0, "vessel_id"] == "MISSING"
    assert list(out.columns) == ["req_id", *FEATURES]


def test_build_features_rejects_incomplete_requests(con):
    with pytest.raises(ValueError):
        build_features(con, _request().drop(columns=["booking_date"]))


# ----------------------------------------------------------------------------- training helpers


def test_split_is_time_based():
    df = pd.DataFrame(
        {"booking_date": pd.to_datetime(["2025-03-31", "2025-04-01", "2025-06-30", "2025-07-01"])}
    )
    tr, va, te = train.split(df)
    assert (len(tr), len(va), len(te)) == (1, 2, 1)


def test_prior_and_column_score_baselines():
    X = pd.DataFrame({"route_hist_late_rate": [0.2, 0.9]})
    assert train.Prior().fit(X, [1, 0, 0, 0]).predict_proba(X)[:, 1].tolist() == [0.25, 0.25]
    assert train.ColumnScore().fit(X, [0, 1]).predict_proba(X)[:, 1].tolist() == [0.2, 0.9]


def test_metrics_and_threshold():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert train.best_f1_threshold(y, p) == pytest.approx(0.35)
    m = train.classification_metrics(y, p, 0.5)
    assert (m["precision"], m["recall"], m["roc_auc"]) == (1.0, 0.5, 0.75)


def test_permutation_p_value_detects_signal_and_noise(monkeypatch):
    monkeypatch.setattr(train, "N_PERMUTATIONS", 200)  # same logic, fewer shuffles
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    assert train.permutation_p_value(y, y + rng.normal(0, 0.5, 400), rng) < 0.01
    assert train.permutation_p_value(y, rng.normal(size=400), rng) > 0.01


# ----------------------------------------------------------------------------- committed artefact


@pytest.mark.skipif(not ARTEFACT.exists(), reason="model artefact not trained")
def test_committed_artefact_loads_and_predicts(con):
    predictor = ModelPredictor(ARTEFACT, con)
    p = predictor.predict_proba(
        BookingFeatures(
            origin_port="CNSHA",
            destination_port="NLRTM",
            booking_date=datetime(2024, 3, 1),
            planned_departure=datetime(2024, 3, 5),
            planned_arrival=datetime(2024, 4, 1),
            cargo_type="Electronics",
            container_count=10,
            weight_tons=100.0,
            vessel_id="V1",
        )
    )
    assert 0.0 < p < 1.0
    assert predictor.version.startswith("delay-")


@pytest.mark.skipif(not CARD.exists(), reason="model card not generated")
def test_model_card_is_honest_and_complete():
    card = json.loads(CARD.read_text())
    chosen = card["results"][card["selected_model"]]
    for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "brier"):
        assert key in chosen["test"]
    assert "test_auc_95ci" in chosen and "test_auc_permutation_p" in chosen
    # The harness must be able to see leakage when it exists.
    assert card["leakage_canary"]["test_roc_auc"] > 0.9
    assert set(card["excluded_as_leakage"]) == FORBIDDEN
