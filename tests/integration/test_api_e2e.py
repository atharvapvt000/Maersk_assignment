"""End-to-end tests over real HTTP against the running stack (built from the real CSVs).

    docker compose up -d --build && uv run pytest -m integration
    BASE_URL=http://staging:8000 uv run pytest -m integration   # any deployed instance

The session fixture polls /health for up to 90 s, so container start-up time is not a source of
flakiness. Assertions use stable facts about the shipped dataset, not timing.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import httpx
import pytest

pytestmark = pytest.mark.integration

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
KNOWN_ID = "SHP-00421"


@pytest.fixture(scope="session")
def api() -> Iterator[httpx.Client]:
    with httpx.Client(base_url=BASE_URL, timeout=10) as client:
        deadline = time.monotonic() + float(os.getenv("STARTUP_TIMEOUT_S", "90"))
        last_error: str = ""
        while time.monotonic() < deadline:
            try:
                if client.get("/health").status_code == 200:
                    break
                last_error = "health not ok yet"
            except httpx.TransportError as exc:
                last_error = repr(exc)
            time.sleep(1)
        else:
            pytest.fail(f"Service at {BASE_URL} never became healthy: {last_error}")
        yield client


def test_health(api):
    body = api.get("/health").json()
    assert body["status"] == "ok"
    assert body["checks"]["data_store"]["shipments"] == 4906
    assert body["checks"]["data_store"]["data_as_of"] == "2025-10-14T23:55:00"
    assert body["version"]


def test_get_shipment_success(api):
    resp = api.get(f"/shipments/{KNOWN_ID}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["shipment_id"] == KNOWN_ID
    assert body["route_key"] == f"{body['origin_port']} → {body['destination_port']}"
    assert resp.headers["x-request-id"]


def test_get_shipment_unknown_id_404(api):
    resp = api.get("/shipments/SHP-99999")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "SHIPMENT_NOT_FOUND"


def test_status_corrupted_rows_served_as_unknown(api):
    # e.g. SHP-00219 has raw status 'Complete' but no actual timestamps -> UNKNOWN, flagged.
    body = api.get("/shipments", params={"status": "UNKNOWN", "limit": 200}).json()
    assert body["total"] == 144
    assert all("missing_actuals" in s["dq_flags"] for s in body["items"])


def test_duplicate_ids_are_served_once(api):
    # SHP-00010 appears twice in the raw CSV; the curated layer keeps exactly one version.
    assert api.get("/shipments/SHP-00010").status_code == 200
    page = api.get("/shipments", params={"limit": 200}).json()
    ids = [s["shipment_id"] for s in page["items"]]
    assert len(ids) == len(set(ids))


def test_list_filters_and_pagination(api):
    params = {"origin": "CNSHA", "destination": "NLRTM", "limit": 5}
    first = api.get("/shipments", params=params).json()
    assert first["total"] >= len(first["items"]) > 0
    assert all(s["origin_port"] == "CNSHA" for s in first["items"])
    second = api.get("/shipments", params={**params, "offset": 5}).json()
    assert {s["shipment_id"] for s in first["items"]}.isdisjoint(
        s["shipment_id"] for s in second["items"]
    )


def test_list_bad_status_422(api):
    resp = api.get("/shipments", params={"status": "LOST"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


def test_route_stats(api):
    body = api.get("/routes/CNSHA/NLRTM/stats").json()
    assert body["shipment_count"] >= body["arrived_count"] > 0
    assert 0 <= body["on_time_rate"] <= 1


def test_route_stats_unknown_port_404(api):
    assert api.get("/routes/CNSHA/XXTST/stats").status_code == 404


def test_data_quality_report(api):
    resp = api.get("/data-quality/report")
    assert resp.status_code == 200
    report = resp.json()
    names = {c["check"] for c in report["checks"]}
    assert {"shipments.superseded_version", "shipments.unknown_port"} <= names
    for check in report["checks"]:
        assert {
            "rows_affected",
            "pct_affected",
            "severity",
            "status",
            "action",
            "detection",
            "reasoning",
        } <= set(check)
    assert report["inputs"]["shipments"]["rows"] == 5028
    assert report["summary"]["fail"] == 0


def test_predict_delay_by_id(api):
    resp = api.post("/predict-delay", json={"shipment_id": KNOWN_ID})
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["delay_probability"] <= 1
    assert body["risk_band"] in {"low", "medium", "high"}
    assert body["model_version"].startswith("delay-")  # the trained artefact, not a stub


def test_predict_delay_by_features(api):
    resp = api.post(
        "/predict-delay",
        json={
            "origin_port": "CNSHA",
            "destination_port": "NLRTM",
            "cargo_type": "Electronics",
            "container_count": 12,
            "booking_date": "2025-02-20T00:00:00",
            "planned_departure": "2025-03-01T00:00:00",
            "planned_arrival": "2025-03-30T00:00:00",
        },
    )
    assert resp.status_code == 200


def test_predict_delay_invalid_body_422(api):
    resp = api.post("/predict-delay", json={"origin_port": "CNSHA"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


def test_route_rankings(api):
    body = api.get("/routes/rankings", params={"quarter": "2025-Q3", "limit": 5}).json()
    assert len(body["routes"]) == 5
    delays = [r["avg_delay_hours"] for r in body["routes"]]
    assert delays == sorted(delays, reverse=True)
    assert all(r["arrived_count"] >= 5 for r in body["routes"])


def test_ports(api):
    assert len(api.get("/ports").json()) == 25


def test_health_reports_trained_model(api):
    assert api.get("/health").json()["checks"]["model"]["version"].startswith("delay-")
