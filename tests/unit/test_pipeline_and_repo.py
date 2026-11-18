"""Pipeline derivations, rule application, idempotency, post-load assertions, repository SQL.

Runs the real pipeline on a 9-row fixture with embedded DuckDB (~100 ms, no services)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest

from app.repository import DuckDBRepository, ShipmentFilters
from pipeline.assertions import PipelineAssertionError, run_assertions
from pipeline.build import build


@pytest.fixture
def db(built_fixture_db: Path) -> Path:
    """Shared read-only build (see conftest)."""
    return built_fixture_db


@pytest.fixture
def fresh_db(raw_dir: Path, tmp_path: Path) -> Path:
    """Private build for tests that mutate or rebuild the warehouse."""
    path = tmp_path / "wh.duckdb"
    build(raw_dir, path)
    return path


@pytest.fixture
def con(db: Path):
    c = duckdb.connect(str(db), read_only=True)
    yield c
    c.close()


@pytest.fixture
def repo(db: Path):
    r = DuckDBRepository(db)
    yield r
    r.close()


# ----------------------------------------------------------------------------- derivations


def test_derived_fields(repo):
    s = repo.get_shipment("SHP-00001")
    # last occurrence wins: actual_arrival 2024-02-02 00:00 vs planned 2024-02-01 00:00
    assert s["actual_delay_hours"] == 24.0
    assert s["on_time_flag"] is True  # exactly 24h is on time (<= 24)
    assert s["route_key"] == "CNSHA → NLRTM"
    assert s["transit_days_planned"] == 27.0
    assert s["transit_days_actual"] == pytest.approx((27 * 24 - 2 + 24) / 24)
    assert s["status"] == "DELIVERED"  # 'delivered' normalised


def test_late_shipment(repo):
    s = repo.get_shipment("SHP-00002")
    assert s["actual_delay_hours"] == 48.0
    assert s["on_time_flag"] is False
    assert s["cargo_type"] == "Electronics"


def test_extra_derived_fields(con):
    lead, dep_delay = con.execute(
        "SELECT booking_lead_days, departure_delay_hours FROM serving.shipments "
        "WHERE shipment_id = 'SHP-00002'"
    ).fetchone()
    assert lead == 4.0
    assert dep_delay == 24.0


def test_null_safe_delay_and_flags(repo):
    s = repo.get_shipment("SHP-00003")
    assert s["actual_delay_hours"] is None
    assert s["on_time_flag"] is None
    assert s["transit_days_actual"] is None
    assert s["status"] == "UNKNOWN"
    assert s["weight_tons"] is None and s["container_count"] is None
    assert s["cargo_type"] == "Chemicals"  # leading whitespace trimmed
    assert set(s["dq_flags"]) == {
        "container_count_non_positive",
        "weight_non_positive",
        "status_missing",
        "missing_actuals",
    }


def test_cancelled_without_actuals_not_flagged_missing(repo):
    s = repo.get_shipment("SHP-00004")
    assert s["status"] == "CANCELLED"
    assert set(s["dq_flags"]) == {"weight_non_numeric"}


# ----------------------------------------------------------------------------- Part 2 rules


def test_departure_before_booking_nulls_departure_only(repo):
    s = repo.get_shipment("SHP-00006")
    assert s["actual_departure"] is None
    assert s["transit_days_actual"] is None
    assert s["actual_delay_hours"] == 1.0  # arrival metric kept
    assert s["status"] == "DELIVERED"  # 'Complete' WITH actuals is a legitimate delivery
    assert s["container_count"] is None
    assert {"actual_departure_inconsistent", "container_count_outlier"} <= set(s["dq_flags"])


def test_status_contradicting_actuals_becomes_unknown(repo):
    s = repo.get_shipment("SHP-00007")
    assert s["status"] == "UNKNOWN"
    assert set(s["dq_flags"]) == {"status_contradicts_actuals", "missing_actuals"}


def test_rejected_rows_have_reasons_and_raw_record(con):
    rows = con.execute(
        "SELECT _source_row, record_id, reasons, disposition, raw_record->>'status' "
        "FROM curated.rejected_rows WHERE dataset = 'shipments' ORDER BY _source_row"
    ).fetchall()
    assert rows == [
        (1, "SHP-00001", ["shipments.superseded_version"], "quarantined", "DELIVERED"),
        (2, "SHP-00002", ["shipments.exact_duplicate_row"], "dropped", "DELAYED"),
        (5, "SHP-00005", ["shipments.unknown_port"], "quarantined", "Complete"),
    ]


def test_colliding_event_ids_are_all_kept(con):
    rows = con.execute(
        "SELECT event_key, event_id, port_code, list_contains(dq_flags, 'event_id_collision') "
        "FROM curated.port_events WHERE event_id = 'EVT-1' ORDER BY event_key"
    ).fetchall()
    assert rows == [("EVT-1", "EVT-1", "CNSHA", True), ("EVT-1#2", "EVT-1", "NLRTM", True)]


def test_event_flags(con):
    flags = dict(con.execute("SELECT event_key, dq_flags FROM curated.port_events").fetchall())
    assert flags["EVT-2"] == ["missing_event_type"]
    assert set(flags["EVT-3"]) == {"delayed_event_without_delay", "unknown_vessel"}


def test_port_flags(con):
    assert con.execute("SELECT dq_flags FROM curated.ports WHERE port_code = 'BEANR'").fetchone()[
        0
    ] == ["missing_country"]


def test_marts(con):
    q = con.execute(
        "SELECT quarter, shipment_count, arrived_count, avg_delay_hours, on_time_rate "
        "FROM serving.route_performance WHERE route_key = 'CNSHA → NLRTM'"
    ).fetchall()
    assert q == [("2024-Q1", 4, 2, 36.0, 0.5)]
    events = con.execute(
        "SELECT sum(events), sum(delayed_events) FROM serving.port_daily_activity"
    ).fetchone()
    assert events == (4, 1)


def test_build_info(con):
    as_of, counts = con.execute("SELECT data_as_of, row_counts FROM serving.build_info").fetchone()
    assert str(as_of) == "2024-03-01 00:00:00"
    assert '"curated.shipments": 6' in counts


# ----------------------------------------------------------------------------- idempotency / safety


def test_pipeline_is_idempotent(raw_dir, fresh_db):
    db = fresh_db

    def snapshot():
        c = duckdb.connect(str(db), read_only=True)
        out = [
            c.execute(f"SELECT * FROM {t} ORDER BY ALL").fetchall()
            for t in (
                "serving.shipments",
                "curated.port_events",
                "curated.rejected_rows",
                "serving.route_performance",
            )
        ]
        c.close()
        return out

    first = snapshot()
    build(raw_dir, db)
    build(raw_dir, db)
    assert snapshot() == first


def test_assertions_pass_on_good_build(db):
    c = duckdb.connect(str(db), read_only=True)
    assert run_assertions(c) == {}
    c.close()


def test_assertions_catch_broken_output(fresh_db):
    c = duckdb.connect(str(fresh_db))
    c.execute(
        "UPDATE curated.shipments SET on_time_flag = NOT on_time_flag "
        "WHERE shipment_id = 'SHP-00001'"
    )
    c.execute("DELETE FROM curated.port_events WHERE event_key = 'EVT-2'")
    failures = run_assertions(c)
    c.close()
    assert set(failures) >= {"on_time_flag_consistent", "port_events_rows_conserved"}


def test_failed_build_keeps_previous_warehouse(raw_dir, fresh_db, monkeypatch):
    db = fresh_db
    before = db.stat().st_mtime_ns
    monkeypatch.setattr("pipeline.build.run_assertions", lambda _con: {"injected_failure": 1})
    with pytest.raises(PipelineAssertionError):
        build(raw_dir, db)
    assert db.stat().st_mtime_ns == before
    assert not db.with_suffix(".duckdb.tmp").exists()


def test_missing_input_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        build(tmp_path, tmp_path / "x.duckdb")


# ----------------------------------------------------------------------------- repository


def test_list_filters_and_pagination(repo):
    items, total = repo.list_shipments(ShipmentFilters(origin="CNSHA"), limit=2, offset=0)
    assert total == 5 and len(items) == 2
    _, total = repo.list_shipments(ShipmentFilters(status="CANCELLED"), limit=10, offset=0)
    assert total == 1
    _, total = repo.list_shipments(
        ShipmentFilters(departure_from=date(2024, 1, 6), departure_to=date(2024, 1, 6)), 10, 0
    )
    assert total == 2  # departure_to is inclusive of the whole day
    page2, _ = repo.list_shipments(ShipmentFilters(), limit=2, offset=2)
    page1, _ = repo.list_shipments(ShipmentFilters(), limit=2, offset=0)
    assert {r["shipment_id"] for r in page1}.isdisjoint(r["shipment_id"] for r in page2)


def test_route_stats(repo):
    stats = repo.route_stats("CNSHA", "NLRTM", None, None)
    assert stats == {
        "shipment_count": 4,
        "arrived_count": 2,
        "cancelled_count": 0,
        "avg_delay_hours": 36.0,
        "on_time_rate": 0.5,
    }
    empty = repo.route_stats("BEANR", "NLRTM", None, None)
    assert empty["shipment_count"] == 0 and empty["on_time_rate"] is None


def test_port_exists(repo):
    assert repo.port_exists("BEANR")
    assert not repo.port_exists("XXTST")


def test_missing_warehouse_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DuckDBRepository(tmp_path / "nope.duckdb")
