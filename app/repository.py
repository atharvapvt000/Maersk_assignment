"""Read-only data access over the serving layer. All SQL used by the API lives here."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Protocol

import duckdb

RANKING_METRICS = ("avg_delay_hours", "on_time_rate", "late_count", "shipment_count")

SHIPMENT_COLUMNS = (
    "shipment_id, customer_id, origin_port, origin_port_name, destination_port, "
    "destination_port_name, route_key, vessel_id, booking_date, planned_departure, "
    "actual_departure, planned_arrival, actual_arrival, container_count, cargo_type, "
    "weight_tons, status, actual_delay_hours, on_time_flag, transit_days_planned, "
    "transit_days_actual, dq_flags"
)


@dataclass(frozen=True)
class ShipmentFilters:
    origin: str | None = None
    destination: str | None = None
    status: str | None = None
    departure_from: date | None = None
    departure_to: date | None = None  # inclusive (whole day)


class ShipmentRepository(Protocol):
    def ping(self) -> dict[str, Any]: ...
    def get_shipment(self, shipment_id: str) -> dict[str, Any] | None: ...
    def list_shipments(
        self, filters: ShipmentFilters, limit: int, offset: int
    ) -> tuple[list[dict[str, Any]], int]: ...
    def port_exists(self, port_code: str) -> bool: ...
    def route_stats(
        self, origin: str, destination: str, departure_from: date | None, departure_to: date | None
    ) -> dict[str, Any]: ...
    def quarters(self) -> list[str]: ...
    def list_ports(self) -> list[dict[str, Any]]: ...
    def route_rankings(
        self, quarter: str | None, metric: str, ascending: bool, min_arrived: int, limit: int
    ) -> list[dict[str, Any]]: ...


def _departure_window(
    departure_from: date | None, departure_to: date | None
) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if departure_from:
        clauses.append("planned_departure >= ?")
        params.append(departure_from)
    if departure_to:
        clauses.append("planned_departure < ?")
        params.append(departure_to + timedelta(days=1))
    return clauses, params


class DuckDBRepository:
    """One read-only connection; each call uses its own cursor (DuckDB cursors are thread-safe
    duplicates of the parent connection, so this works with FastAPI's threadpool)."""

    def __init__(self, db_path: Path) -> None:
        if not db_path.exists():
            raise FileNotFoundError(f"Warehouse not found at {db_path}; run the pipeline first")
        self._con = duckdb.connect(str(db_path), read_only=True)

    def close(self) -> None:
        self._con.close()

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Shared read-only connection (callers must use .cursor() per thread)."""
        return self._con

    def _query(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        cur = self._con.cursor()
        try:
            rel = cur.execute(sql, params or [])
            cols = [d[0] for d in rel.description]
            return [dict(zip(cols, row, strict=True)) for row in rel.fetchall()]
        finally:
            cur.close()

    def ping(self) -> dict[str, Any]:
        row = self._query(
            "SELECT (SELECT count(*) FROM serving.shipments) AS shipments, "
            "(SELECT strftime(max(built_at), '%Y-%m-%dT%H:%M:%SZ') FROM serving.build_info) "
            "AS built_at, "
            "(SELECT strftime(max(data_as_of), '%Y-%m-%dT%H:%M:%S') FROM serving.build_info) "
            "AS data_as_of"
        )[0]
        return dict(row)

    def get_shipment(self, shipment_id: str) -> dict[str, Any] | None:
        rows = self._query(
            f"SELECT {SHIPMENT_COLUMNS} FROM serving.shipments WHERE shipment_id = ?",
            [shipment_id],
        )
        return rows[0] if rows else None

    def list_shipments(
        self, filters: ShipmentFilters, limit: int, offset: int
    ) -> tuple[list[dict[str, Any]], int]:
        clauses, params = _departure_window(filters.departure_from, filters.departure_to)
        for column, value in (
            ("origin_port", filters.origin),
            ("destination_port", filters.destination),
            ("status", filters.status),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        total = self._query(f"SELECT count(*) AS n FROM serving.shipments {where}", params)[0]["n"]
        items = self._query(
            f"SELECT {SHIPMENT_COLUMNS} FROM serving.shipments {where} "
            "ORDER BY planned_departure, shipment_id LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        return items, total

    def port_exists(self, port_code: str) -> bool:
        return bool(self._query("SELECT 1 FROM serving.ports WHERE port_code = ?", [port_code]))

    def route_stats(
        self, origin: str, destination: str, departure_from: date | None, departure_to: date | None
    ) -> dict[str, Any]:
        clauses, params = _departure_window(departure_from, departure_to)
        clauses = ["origin_port = ?", "destination_port = ?", *clauses]
        params = [origin, destination, *params]
        return self._query(
            f"""
            SELECT
                count(*)                                              AS shipment_count,
                count(actual_delay_hours)                             AS arrived_count,
                count(*) FILTER (WHERE status = 'CANCELLED')          AS cancelled_count,
                round(avg(actual_delay_hours), 2)                     AS avg_delay_hours,
                round(avg(on_time_flag::INTEGER), 4)                  AS on_time_rate
            FROM serving.shipments
            WHERE {" AND ".join(clauses)}
            """,
            params,
        )[0]

    def quarters(self) -> list[str]:
        rows = self._query("SELECT DISTINCT quarter FROM serving.route_performance ORDER BY 1")
        return [r["quarter"] for r in rows]

    def route_rankings(
        self, quarter: str | None, metric: str, ascending: bool, min_arrived: int, limit: int
    ) -> list[dict[str, Any]]:
        """Rank routes from the quarterly mart. Averages are re-weighted by arrived_count, so
        an all-quarters ranking is exact rather than an average of averages."""
        if metric not in RANKING_METRICS:  # never interpolate unvalidated text into SQL
            raise ValueError(f"unsupported metric {metric}")
        where = "WHERE quarter = ?" if quarter else ""
        params: list[Any] = [quarter] if quarter else []
        order = "ASC" if ascending else "DESC"
        return self._query(
            f"""
            SELECT route_key, origin_port, destination_port,
                   sum(shipment_count)::INTEGER                       AS shipment_count,
                   sum(arrived_count)::INTEGER                        AS arrived_count,
                   sum(late_count)::INTEGER                           AS late_count,
                   round(sum(avg_delay_hours * arrived_count)
                         / nullif(sum(arrived_count), 0), 2)          AS avg_delay_hours,
                   round(sum(on_time_rate * arrived_count)
                         / nullif(sum(arrived_count), 0), 4)          AS on_time_rate
            FROM serving.route_performance
            {where}
            GROUP BY ALL
            HAVING sum(arrived_count) >= ?
            ORDER BY {metric} {order} NULLS LAST, route_key
            LIMIT ?
            """,
            [*params, min_arrived, limit],
        )

    def list_ports(self) -> list[dict[str, Any]]:
        return self._query(
            "SELECT port_code, port_name, country, region, timezone, avg_congestion_score "
            "FROM serving.ports ORDER BY port_code"
        )
