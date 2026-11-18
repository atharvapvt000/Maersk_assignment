"""Post-load assertions: invariants of the built warehouse, checked BEFORE it replaces the live one.

The DQ step validates the *input*; these validate the *pipeline's own output* (keys, references,
derived-field consistency, row conservation). Each query returns the number of violating rows;
any non-zero result aborts the build and the previous warehouse stays live.
"""

from __future__ import annotations

import duckdb

from app.domain import ON_TIME_THRESHOLD_HOURS
from dq.rules import MAX_CONTAINERS_PER_SHIPMENT


class PipelineAssertionError(RuntimeError):
    def __init__(self, failures: dict[str, int]) -> None:
        super().__init__(f"Post-load assertions failed: {failures}")
        self.failures = failures


ASSERTIONS: dict[str, str] = {
    # ---- keys
    "shipments_pk_unique_not_null": """
        SELECT count(*) - count(DISTINCT shipment_id) + count(*) FILTER (WHERE shipment_id IS NULL)
        FROM serving.shipments""",
    "port_events_key_unique_not_null": """
        SELECT count(*) - count(DISTINCT event_key) + count(*) FILTER (WHERE event_key IS NULL)
        FROM curated.port_events""",
    "ports_pk_unique": "SELECT count(*) - count(DISTINCT port_code) FROM curated.ports",
    # ---- referential integrity
    "shipments_ports_fk": """
        SELECT count(*) FROM curated.shipments s
        WHERE s.origin_port NOT IN (SELECT port_code FROM curated.ports)
           OR s.destination_port NOT IN (SELECT port_code FROM curated.ports)""",
    "port_events_ports_fk": """
        SELECT count(*) FROM curated.port_events
        WHERE port_code NOT IN (SELECT port_code FROM curated.ports)""",
    # ---- row conservation: every raw row is either curated or rejected, exactly once
    "shipments_rows_conserved": """
        SELECT abs((SELECT count(*) FROM raw.shipments)
                 - (SELECT count(*) FROM curated.shipments)
                 - (SELECT count(*) FROM curated.rejected_rows WHERE dataset = 'shipments'))""",
    "port_events_rows_conserved": """
        SELECT abs((SELECT count(*) FROM raw.port_events)
                 - (SELECT count(*) FROM curated.port_events)
                 - (SELECT count(*) FROM curated.rejected_rows WHERE dataset = 'port_events'))""",
    "no_rejected_row_curated": """
        SELECT count(*) FROM curated.shipments s
        JOIN curated.rejected_rows r
          ON r.dataset = 'shipments' AND r._source_row = s._source_row""",
    # ---- derived-field consistency
    "on_time_flag_consistent": f"""
        SELECT count(*) FROM curated.shipments
        WHERE (on_time_flag IS NULL) <> (actual_delay_hours IS NULL)
           OR on_time_flag <> (actual_delay_hours <= {ON_TIME_THRESHOLD_HOURS})""",
    "route_key_consistent": """
        SELECT count(*) FROM curated.shipments
        WHERE route_key <> origin_port || ' → ' || destination_port""",
    "transit_times_non_negative": """
        SELECT count(*) FROM curated.shipments
        WHERE transit_days_planned <= 0 OR transit_days_actual <= 0""",
    "attribute_bounds": f"""
        SELECT count(*) FROM curated.shipments
        WHERE weight_tons <= 0
           OR container_count NOT BETWEEN 1 AND {MAX_CONTAINERS_PER_SHIPMENT}""",
    "status_domain": """
        SELECT count(*) FROM curated.shipments
        WHERE status NOT IN ('DELIVERED', 'DELAYED', 'CANCELLED', 'UNKNOWN')""",
    "delivered_has_arrival": """
        SELECT count(*) FROM curated.shipments
        WHERE status IN ('DELIVERED', 'DELAYED') AND actual_arrival IS NULL""",
    # ---- marts reconcile with their source
    "route_performance_reconciles": """
        SELECT abs((SELECT sum(shipment_count) FROM serving.route_performance)
                 - (SELECT count(*) FROM serving.shipments))""",
    "port_daily_activity_reconciles": """
        SELECT abs((SELECT sum(events) FROM serving.port_daily_activity)
                 - (SELECT count(*) FROM serving.port_events))""",
}


def run_assertions(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Return {assertion name: violating row count} for every failing assertion."""
    failures: dict[str, int] = {}
    for name, sql in ASSERTIONS.items():
        violations = int(con.execute(sql).fetchone()[0] or 0)  # type: ignore[index]
        if violations:
            failures[name] = violations
    return failures
