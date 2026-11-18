"""Request id propagation, structured access logs, and the last-resort 500 handler."""

from __future__ import annotations

import json
import logging

from fastapi.testclient import TestClient

from app.logging_setup import JsonFormatter, request_id_var
from app.main import create_app


def _access_records(caplog):
    return [r for r in caplog.records if r.name == "app.access"]


def test_access_log_has_required_fields(client, caplog):
    caplog.set_level(logging.INFO, logger="app.access")
    client.get("/shipments/SHP-00421")
    rec = _access_records(caplog)[-1]
    assert rec.method == "GET"
    assert rec.path == "/shipments/SHP-00421"
    assert rec.route == "/shipments/{shipment_id}"
    assert rec.status == 200
    assert isinstance(rec.latency_ms, float)


def test_request_id_echoed_when_supplied(client):
    resp = client.get("/health", headers={"X-Request-ID": "trace-123"})
    assert resp.headers["x-request-id"] == "trace-123"


def test_request_id_generated_when_absent(client):
    a = client.get("/shipments/SHP-99999")
    b = client.get("/shipments/SHP-99999")
    assert a.headers["x-request-id"] != b.headers["x-request-id"]
    assert a.json()["error"]["request_id"] == a.headers["x-request-id"]


def test_unhandled_exception_returns_json_500_and_logs(settings, fake_repo, fake_predictor, caplog):
    def boom(_shipment_id):
        raise RuntimeError("db exploded")

    fake_repo.get_shipment = boom
    caplog.set_level(logging.INFO)
    with TestClient(create_app(settings, repo=fake_repo, predictor=fake_predictor)) as c:
        resp = c.get("/shipments/SHP-00421")
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "db exploded" not in resp.text  # internals never leak to clients
    assert resp.headers["x-request-id"]
    assert _access_records(caplog)[-1].status == 500
    assert any(r.name == "app.error" and r.exc_info for r in caplog.records)


def test_json_formatter_includes_extras_and_request_id():
    token = request_id_var.set("rid-1")
    try:
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello", (), None)
        record.status = 201
        out = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert out["msg"] == "hello"
    assert out["level"] == "INFO"
    assert out["status"] == 201
    assert out["request_id"] == "rid-1"
