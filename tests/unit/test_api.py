"""Router-level tests with the repository and model faked out (no DB, no files)."""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


def assert_error(resp, status: int, code: str) -> dict:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == code
    assert body["error"]["request_id"] == resp.headers["x-request-id"]
    return body["error"]


# ----------------------------------------------------------------------------- health


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == "9.9.9"
    assert body["git_sha"] == "abc123"
    assert body["checks"]["data_store"]["ok"] is True
    assert body["checks"]["model"] == {"ok": True, "version": "fake-v1"}


def test_health_degraded_when_store_missing(settings):
    # No repo injected and settings.db_path does not exist -> app still starts, reports 503.
    with TestClient(create_app(settings)) as c:
        resp = c.get("/health")
        assert resp.status_code == 503
        assert resp.json()["status"] == "degraded"
        assert resp.json()["checks"]["data_store"]["ok"] is False
        # Data endpoints fail clearly rather than with a generic 500.
        assert_error(c.get("/shipments/SHP-00001"), 503, "DATA_STORE_UNAVAILABLE")


# ----------------------------------------------------------------------------- shipments


def test_get_shipment_found(client):
    resp = client.get("/shipments/SHP-00421")
    assert resp.status_code == 200
    body = resp.json()
    assert body["shipment_id"] == "SHP-00421"
    assert body["route_key"] == "CNSHA → NLRTM"
    assert body["actual_delay_hours"] == 30.0
    assert body["on_time_flag"] is False


def test_get_shipment_not_found(client):
    err = assert_error(client.get("/shipments/SHP-99999"), 404, "SHIPMENT_NOT_FOUND")
    assert "SHP-99999" in err["message"]


@pytest.mark.parametrize("bad_id", ["123", "SHP-1", "shp-00001", "SHP-0000A"])
def test_get_shipment_malformed_id_is_422(client, bad_id):
    assert_error(client.get(f"/shipments/{bad_id}"), 422, "VALIDATION_ERROR")


def test_list_shipments_passes_normalised_filters(client, fake_repo):
    resp = client.get(
        "/shipments",
        params={
            "origin": "cnsha",
            "destination": "NLRTM",
            "status": "DELAYED",
            "departure_from": "2025-01-01",
            "departure_to": "2025-01-31",
            "limit": 10,
            "offset": 0,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["total"] == 1
    f = fake_repo.last_filters
    assert (f.origin, f.destination, f.status) == ("CNSHA", "NLRTM", "DELAYED")
    assert (f.departure_from, f.departure_to) == (date(2025, 1, 1), date(2025, 1, 31))
    assert fake_repo.last_page == (10, 0)


def test_list_shipments_pagination_envelope(client):
    body = client.get("/shipments", params={"limit": 5, "offset": 0}).json()
    assert set(body) == {"items", "total", "limit", "offset"}
    assert (body["limit"], body["offset"]) == (5, 0)


@pytest.mark.parametrize(
    "params",
    [{"limit": 0}, {"limit": 201}, {"offset": -1}, {"status": "LOST"}, {"origin": "TOOLONG"}],
)
def test_list_shipments_invalid_params_422(client, params):
    assert_error(client.get("/shipments", params=params), 422, "VALIDATION_ERROR")


def test_list_shipments_inverted_date_range_400(client):
    resp = client.get(
        "/shipments", params={"departure_from": "2025-02-01", "departure_to": "2025-01-01"}
    )
    assert_error(resp, 400, "INVALID_DATE_RANGE")


# ----------------------------------------------------------------------------- routes


def test_route_stats(client):
    resp = client.get("/routes/cnsha/nlrtm/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["route_key"] == "CNSHA → NLRTM"
    assert body["shipment_count"] == 4
    assert body["on_time_rate"] == pytest.approx(0.6667)


def test_route_stats_valid_route_without_data_is_200(client):
    body = client.get("/routes/SGSIN/NLRTM/stats").json()
    assert body["shipment_count"] == 0
    assert body["avg_delay_hours"] is None


def test_route_stats_unknown_port_404(client):
    err = assert_error(client.get("/routes/CNSHA/XXTST/stats"), 404, "PORT_NOT_FOUND")
    assert err["details"] == {"unknown_ports": ["XXTST"]}


def test_route_stats_same_port_400(client):
    assert_error(client.get("/routes/CNSHA/CNSHA/stats"), 400, "INVALID_ROUTE")


# ----------------------------------------------------------------------------- predict


def test_predict_by_shipment_id_uses_only_booking_features(client, fake_predictor):
    resp = client.post("/predict-delay", json={"shipment_id": "SHP-00421"})
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "shipment_id": "SHP-00421",
        "delay_probability": 0.72,
        "predicted_late": True,
        "risk_band": "high",
        "threshold": 0.5,
        "model_version": "fake-v1",
    }
    features = fake_predictor.calls[-1]
    assert not any(name.startswith("actual") for name in vars(features))
    assert not hasattr(features, "status")


def test_predict_by_features(client, fake_predictor):
    resp = client.post(
        "/predict-delay",
        json={
            "origin_port": "cnsha",
            "destination_port": "NLRTM",
            "cargo_type": "Electronics",
            "booking_date": "2025-02-20T00:00:00",
            "planned_departure": "2025-03-01T00:00:00",
            "planned_arrival": "2025-03-30T00:00:00",
        },
    )
    assert resp.status_code == 200
    assert fake_predictor.calls[-1].origin_port == "CNSHA"


def test_predict_unknown_shipment_404(client):
    assert_error(
        client.post("/predict-delay", json={"shipment_id": "SHP-99999"}), 404, "SHIPMENT_NOT_FOUND"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},  # neither mode
        {"shipment_id": "SHP-00421", "origin_port": "CNSHA"},  # both modes
        {"shipment_id": "SHP-00421", "actual_arrival": "2025-01-01T00:00:00"},  # leakage field
        {
            "origin_port": "CNSHA",
            "destination_port": "NLRTM",
            "planned_departure": "2025-03-30T00:00:00",
            "planned_arrival": "2025-03-01T00:00:00",
        },
        {
            "origin_port": "CNSHA",
            "destination_port": "NLRTM",
            "container_count": 0,
            "planned_departure": "2025-03-01T00:00:00",
            "planned_arrival": "2025-03-30T00:00:00",
        },
    ],
)
def test_predict_invalid_body_422(client, payload):
    assert_error(client.post("/predict-delay", json=payload), 422, "VALIDATION_ERROR")


def test_predict_unknown_port_422(client):
    resp = client.post(
        "/predict-delay",
        json={
            "origin_port": "XXTST",
            "destination_port": "NLRTM",
            "booking_date": "2025-02-20T00:00:00",
            "planned_departure": "2025-03-01T00:00:00",
            "planned_arrival": "2025-03-30T00:00:00",
        },
    )
    assert_error(resp, 422, "UNKNOWN_PORT")


def test_predict_without_model_503(settings, fake_repo):
    # No `with` block -> lifespan does not run -> no predictor is ever loaded.
    c = TestClient(create_app(settings, repo=fake_repo))
    resp = c.post("/predict-delay", json={"shipment_id": "SHP-00421"})
    assert_error(resp, 503, "MODEL_UNAVAILABLE")


# ----------------------------------------------------------------------------- data quality


def test_dq_report_served(client, dq_report):
    resp = client.get("/data-quality/report")
    assert resp.status_code == 200
    assert resp.json() == dq_report


def test_dq_report_missing_503(client):
    assert_error(client.get("/data-quality/report"), 503, "DQ_REPORT_UNAVAILABLE")


def test_dq_report_corrupt_503(client, settings):
    settings.dq_report_path.write_text("{not json")
    assert_error(client.get("/data-quality/report"), 503, "DQ_REPORT_CORRUPT")


# ----------------------------------------------------------------------------- framework errors


def test_unknown_path_uses_error_envelope(client):
    assert_error(client.get("/does-not-exist"), 404, "NOT_FOUND")


def test_wrong_method_405(client):
    assert_error(client.delete("/health"), 405, "METHOD_NOT_ALLOWED")


# ----------------------------------------------------------------------------- rankings / ports


def test_route_rankings(client, fake_repo):
    resp = client.get("/routes/rankings", params={"quarter": "2025-Q3", "limit": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["routes"][0]["rank"] == 1 and body["available_quarters"] == ["2025-Q2", "2025-Q3"]
    assert body["data_as_of"] is None or isinstance(body["data_as_of"], str)
    assert fake_repo.last_ranking == ("2025-Q3", "avg_delay_hours", False, 5, 3)


def test_route_rankings_unknown_quarter_404(client):
    err = assert_error(
        client.get("/routes/rankings", params={"quarter": "2019-Q1"}), 404, "QUARTER_NOT_FOUND"
    )
    assert err["details"]["available_quarters"] == ["2025-Q2", "2025-Q3"]


@pytest.mark.parametrize(
    "params", [{"quarter": "2025Q3"}, {"metric": "p90"}, {"order": "up"}, {"limit": 0}]
)
def test_route_rankings_invalid_params(client, params):
    assert_error(client.get("/routes/rankings", params=params), 422, "VALIDATION_ERROR")


def test_ports(client):
    body = client.get("/ports").json()
    assert [p["port_code"] for p in body] == ["CNSHA", "NLRTM"]


def test_predict_by_features_requires_booking_date(client):
    resp = client.post(
        "/predict-delay",
        json={
            "origin_port": "CNSHA",
            "destination_port": "NLRTM",
            "planned_departure": "2025-03-01T00:00:00",
            "planned_arrival": "2025-03-30T00:00:00",
        },
    )
    assert_error(resp, 422, "VALIDATION_ERROR")
