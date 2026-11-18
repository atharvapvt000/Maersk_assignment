"""Shared fixtures. Nothing here needs Docker, a network, or the real data files."""

from __future__ import annotations

import json
import textwrap
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.repository import ShipmentFilters
from app.services.predictor import BookingFeatures

# ---------------------------------------------------------------------------------------------
# Tiny raw dataset exercising each known data problem once. Used by pipeline + DQ unit tests.
# ---------------------------------------------------------------------------------------------

PORTS_CSV = """\
port_code,port_name,country,region,timezone,avg_congestion_score
CNSHA,Shanghai,China,APAC,UTC+8,0.85
NLRTM,Rotterdam,Netherlands,EUR,UTC+1,0.8
BEANR,Antwerp,,EUR,UTC+1,0.7
"""

_HDR = (
    "shipment_id,customer_id,origin_port,destination_port,vessel_id,booking_date,"
    "planned_departure,actual_departure,planned_arrival,actual_arrival,container_count,"
    "cargo_type,weight_tons,status"
)
SHIPMENTS_CSV = textwrap.dedent(f"""\
{_HDR}
SHP-00001,C1,CNSHA,NLRTM,V1,2024-01-01 00:00:00,2024-01-05 00:00:00,2024-01-05 02:00:00,2024-02-01 00:00:00,2024-02-01 12:00:00,10,Electronics,100.5,DELIVERED
SHP-00002,C1,CNSHA,NLRTM,V1,2024-01-01 00:00:00,2024-01-05 00:00:00,2024-01-06 00:00:00,2024-02-01 00:00:00,2024-02-03 00:00:00,5,electronics,50,DELAYED
SHP-00003,C2,CNSHA,NLRTM,V2,2024-01-02 00:00:00,2024-01-06 00:00:00,,2024-02-02 00:00:00,,0, Chemicals,-3,N/A
SHP-00004,C2,NLRTM,CNSHA,V2,2024-01-02 00:00:00,2024-01-06 00:00:00,,2024-02-02 00:00:00,,3,FURNITURE,abc,CANCELLED
SHP-00005,C3,XXTST,NLRTM,V3,2024-01-03 00:00:00,2024-01-07 00:00:00,2024-01-07 00:00:00,2024-02-03 00:00:00,2024-02-03 00:00:00,1,Machinery,10,Complete
SHP-00001,C1,CNSHA,NLRTM,V1,2024-01-01 00:00:00,2024-01-05 00:00:00,2024-01-05 02:00:00,2024-02-01 00:00:00,2024-02-02 00:00:00,10,Electronics,100.5,delivered
SHP-00002,C1,CNSHA,NLRTM,V1,2024-01-01 00:00:00,2024-01-05 00:00:00,2024-01-06 00:00:00,2024-02-01 00:00:00,2024-02-03 00:00:00,5,electronics,50,DELAYED
SHP-00006,C4,CNSHA,BEANR,V1,2024-01-10 00:00:00,2024-01-15 00:00:00,2024-01-08 00:00:00,2024-02-10 00:00:00,2024-02-10 01:00:00,500,Machinery,20,Complete
SHP-00007,C4,CNSHA,NLRTM,V2,2024-01-03 00:00:00,2024-01-08 00:00:00,,2024-02-05 00:00:00,,4,Apparel,30,COMPLETED
""")

# Row numbers (1-based, as in the DQ report / _source_row):
#   1 superseded by 6 | 2 exact copy of 7 | 3 N/A status, no actuals, 0 containers, weight -3
#   4 cancelled, weight 'abc' | 5 placeholder port | 8 departed before booking + 500 containers
#   9 'COMPLETED' but no actuals (status contradicts actuals)

EVENTS_CSV = """\
event_id,event_timestamp,port_code,vessel_id,event_type,delay_minutes,notes
EVT-1,2024-01-01 00:00:00,CNSHA,V1,ARRIVAL,0,Normal operations
EVT-1,2024-02-01 09:00:00,NLRTM,V2,DEPARTURE,15,Berth congestion
EVT-2,2024-01-01 01:00:00,NLRTM,V2,,30,
EVT-3,2024-03-01 00:00:00,BEANR,VX,DELAYED,0,Weather delay
"""


def write_raw(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "ports.csv").write_text(PORTS_CSV)
    (d / "shipments.csv").write_text(SHIPMENTS_CSV)
    (d / "port_events.csv").write_text(EVENTS_CSV)
    return d


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    return write_raw(tmp_path / "raw")


@pytest.fixture(scope="session")
def built_fixture_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fixture warehouse built ONCE per session. Treat as read-only."""
    from pipeline.build import build

    root = tmp_path_factory.mktemp("fixture_wh")
    db = root / "wh.duckdb"
    build(write_raw(root / "raw"), db)
    return db


# ---------------------------------------------------------------------------------------------
# Fakes for API unit tests
# ---------------------------------------------------------------------------------------------


def make_shipment(shipment_id: str = "SHP-00421", **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "shipment_id": shipment_id,
        "customer_id": "CUST-1",
        "origin_port": "CNSHA",
        "origin_port_name": "Shanghai",
        "destination_port": "NLRTM",
        "destination_port_name": "Rotterdam",
        "route_key": "CNSHA → NLRTM",
        "vessel_id": "VSL-1",
        "booking_date": datetime(2025, 1, 1),
        "planned_departure": datetime(2025, 1, 5),
        "actual_departure": datetime(2025, 1, 5, 3),
        "planned_arrival": datetime(2025, 2, 1),
        "actual_arrival": datetime(2025, 2, 2, 6),
        "container_count": 10,
        "cargo_type": "Electronics",
        "weight_tons": 120.0,
        "status": "DELAYED",
        "actual_delay_hours": 30.0,
        "on_time_flag": False,
        "transit_days_planned": 27.0,
        "transit_days_actual": 28.125,
        "dq_flags": [],
    }
    row.update(overrides)
    return row


class FakeRepo:
    def __init__(self, shipments: list[dict[str, Any]] | None = None) -> None:
        self.shipments = {s["shipment_id"]: s for s in (shipments or [make_shipment()])}
        self.ports = {"CNSHA", "NLRTM", "SGSIN"}
        self.last_filters: ShipmentFilters | None = None
        self.last_page: tuple[int, int] | None = None

    def ping(self) -> dict[str, Any]:
        return {"shipments": len(self.shipments), "built_at": "2025-01-01T00:00:00Z"}

    def get_shipment(self, shipment_id: str) -> dict[str, Any] | None:
        return self.shipments.get(shipment_id)

    def list_shipments(self, filters, limit, offset):
        self.last_filters, self.last_page = filters, (limit, offset)
        items = list(self.shipments.values())
        return items[offset : offset + limit], len(items)

    def port_exists(self, port_code: str) -> bool:
        return port_code in self.ports

    def route_stats(self, origin, destination, departure_from, departure_to):
        if (origin, destination) == ("CNSHA", "NLRTM"):
            return {
                "shipment_count": 4,
                "arrived_count": 3,
                "cancelled_count": 1,
                "avg_delay_hours": 20.5,
                "on_time_rate": 0.6667,
            }
        return {
            "shipment_count": 0,
            "arrived_count": 0,
            "cancelled_count": 0,
            "avg_delay_hours": None,
            "on_time_rate": None,
        }

    def list_ports(self):
        return [
            {
                "port_code": "CNSHA",
                "port_name": "Shanghai",
                "country": "China",
                "region": "APAC",
                "timezone": "UTC+8",
                "avg_congestion_score": 0.85,
            },
            {
                "port_code": "NLRTM",
                "port_name": "Rotterdam",
                "country": "Netherlands",
                "region": "EUR",
                "timezone": "UTC+1",
                "avg_congestion_score": 0.8,
            },
        ]

    def quarters(self):
        return ["2025-Q2", "2025-Q3"]

    def route_rankings(self, quarter, metric, ascending, min_arrived, limit):
        self.last_ranking = (quarter, metric, ascending, min_arrived, limit)
        return [
            {
                "route_key": "CNSHA → NLRTM",
                "origin_port": "CNSHA",
                "destination_port": "NLRTM",
                "shipment_count": 12,
                "arrived_count": 10,
                "late_count": 6,
                "avg_delay_hours": 40.5,
                "on_time_rate": 0.4,
            },
        ][:limit]


class FakePredictor:
    version = "fake-v1"
    threshold = 0.5

    def __init__(self, probability: float = 0.72) -> None:
        self.probability = probability
        self.calls: list[BookingFeatures] = []

    def predict_proba(self, features: BookingFeatures) -> float:
        self.calls.append(features)
        return self.probability


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        db_path=tmp_path / "missing.duckdb",
        dq_report_path=tmp_path / "dq_report.json",
        model_path=tmp_path / "model.joblib",
        version="9.9.9",
        git_sha="abc123",
        log_level="INFO",
    )


@pytest.fixture
def fake_repo() -> FakeRepo:
    return FakeRepo()


@pytest.fixture
def fake_predictor() -> FakePredictor:
    return FakePredictor()


@pytest.fixture
def client(settings, fake_repo, fake_predictor) -> Iterator[TestClient]:
    app = create_app(settings, repo=fake_repo, predictor=fake_predictor)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def dq_report(settings: Settings) -> dict:
    report = {"report_version": "1.0", "summary": {"checks_run": 1}, "checks": []}
    settings.dq_report_path.write_text(json.dumps(report))
    return report
