"""Layered warehouse build: raw CSV -> raw.* -> curated.* -> serving.* (one DuckDB file).

Layers
  raw      Faithful, untyped (VARCHAR) copy of each CSV + lineage (_row_num, _source_file,
           _source_sha256). Never modified.
  curated  Typed, cleaned, deduplicated. Every DQ rule in dq.rules is applied here exactly as the
           DQ report states: rejected rows go to curated.rejected_rows (with reasons and the raw
           record), fixed/flagged rows carry dq_flags naming the rule. Derived fields live here.
  serving  What consumers read: the API's shipments table, reference ports, events, and two
           marts (route_performance by quarter, port_daily_activity) for analytics, the GenAI
           assistant and ML features.

Idempotency: every run rebuilds into `<db>.tmp` from the raw CSVs, runs post-load assertions,
and only then atomically swaps it into place (os.replace). Re-running never duplicates data; a
failed run leaves the previous warehouse untouched.

    python -m pipeline.build --input data/raw --db data/warehouse/supply_chain.duckdb
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from app.domain import CANONICAL_CARGO_TYPES, ON_TIME_THRESHOLD_HOURS, STATUS_ALIASES
from dq.rules import MAX_CONTAINERS_PER_SHIPMENT, RULES
from pipeline.assertions import PipelineAssertionError, run_assertions

log = logging.getLogger("pipeline")

RAW_FILES = {"ports": "ports.csv", "shipments": "shipments.csv", "port_events": "port_events.csv"}
TS_REGEX = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"  # same strict format the DQ step enforces


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ts(col: str) -> str:
    """Strictly parse 'YYYY-MM-DD HH:MM:SS' (matching dq_check.parse_ts), else NULL."""
    return (
        f"CASE WHEN regexp_full_match(trim({col}), '{TS_REGEX}') "
        f"THEN TRY_CAST(trim({col}) AS TIMESTAMP) END"
    )


def _flags(*cases: tuple[str, str]) -> str:
    """SQL list of rule names whose condition is true, e.g. ['weight_non_numeric']."""
    parts = ", ".join(f"CASE WHEN {cond} THEN '{rule}' END" for rule, cond in cases)
    return f"list_filter([{parts}], x -> x IS NOT NULL)"


# ============================================================================================
# raw
# ============================================================================================


def _load_raw(con: duckdb.DuckDBPyConnection, raw_dir: Path, sources: dict[str, str]) -> None:
    con.execute("CREATE SCHEMA raw")
    for table, filename in RAW_FILES.items():
        path = raw_dir / filename
        # preserve_insertion_order (DuckDB default) keeps rowid == file order, which makes
        # _row_num identical to the DQ step's 1-based row numbers (verified by reconciliation).
        con.execute(
            "CREATE TEMP TABLE _load AS "
            "SELECT * FROM read_csv(?, header = true, all_varchar = true, quote = '\"')",
            [str(path)],
        )
        con.execute(
            f"""
            CREATE TABLE raw.{table} AS
            SELECT rowid + 1 AS _row_num, ? AS _source_file, ? AS _source_sha256, *
            FROM _load ORDER BY rowid
            """,
            [filename, sources[table]],
        )
        con.execute("DROP TABLE _load")


def _build_ref(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA ref")
    con.execute("CREATE TABLE ref.status_alias (raw_upper VARCHAR PRIMARY KEY, status VARCHAR)")
    con.executemany(
        "INSERT INTO ref.status_alias VALUES (?, ?)",
        [(k, v.value) for k, v in STATUS_ALIASES.items()],
    )
    con.execute("CREATE TABLE ref.cargo_type (cargo_lower VARCHAR PRIMARY KEY, cargo_type VARCHAR)")
    con.executemany(
        "INSERT INTO ref.cargo_type VALUES (?, ?)", [(c.lower(), c) for c in CANONICAL_CARGO_TYPES]
    )
    con.execute(
        "CREATE TABLE ref.dq_rules (name VARCHAR PRIMARY KEY, dataset VARCHAR, rule VARCHAR, "
        "severity VARCHAR, action VARCHAR, detection VARCHAR, reasoning VARCHAR)"
    )
    con.executemany(
        "INSERT INTO ref.dq_rules VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (r.name, r.dataset, r.rule, r.severity, r.action, r.detection, r.reasoning)
            for r in RULES
        ],
    )


# ============================================================================================
# curated
# ============================================================================================


def _curate_ports(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TEMP TABLE t_ports AS
        SELECT
            _row_num,
            upper(trim(port_code))                                   AS port_code,
            trim(port_name)                                          AS port_name,
            nullif(trim(country), '')                                AS country,
            trim(region)                                             AS region,
            trim(timezone)                                           AS timezone,
            TRY_CAST(trim(avg_congestion_score) AS DOUBLE)           AS score_raw,
            row_number() OVER (PARTITION BY upper(trim(port_code)) ORDER BY _row_num) AS occ,
            to_json(p)                                               AS raw_record
        FROM raw.ports p
        """
    )
    con.execute(
        """
        INSERT INTO curated.rejected_rows
        SELECT 'ports', _row_num, port_code, ['ports.duplicate_code'], 'dropped', raw_record
        FROM t_ports WHERE occ > 1
        """
    )
    con.execute(
        f"""
        CREATE TABLE curated.ports AS
        SELECT
            port_code, port_name, country, region, timezone,
            CASE WHEN score_raw BETWEEN 0 AND 1 THEN score_raw END   AS avg_congestion_score,
            {
            _flags(
                ("missing_country", "country IS NULL"),
                ("invalid_congestion_score", "NOT coalesce(score_raw BETWEEN 0 AND 1, false)"),
            )
        }
                                                                     AS dq_flags,
            _row_num                                                 AS _source_row
        FROM t_ports WHERE occ = 1
        """
    )


def _curate_shipments(con: duckdb.DuckDBPyConnection) -> None:
    # 1. Type everything, keep raw text where a rule needs it.
    con.execute(
        f"""
        CREATE TEMP TABLE t_ship AS
        SELECT
            s._row_num,
            trim(s.shipment_id)                                       AS shipment_id,
            nullif(trim(s.customer_id), '')                           AS customer_id,
            upper(trim(s.origin_port))                                AS origin_port,
            upper(trim(s.destination_port))                           AS destination_port,
            nullif(trim(s.vessel_id), '')                             AS vessel_id,
            {_ts("s.booking_date")}                                   AS booking_date,
            {_ts("s.planned_departure")}                              AS planned_departure,
            {_ts("s.actual_departure")}                               AS actual_departure_raw,
            {_ts("s.planned_arrival")}                                AS planned_arrival,
            {_ts("s.actual_arrival")}                                 AS actual_arrival,
            trim(s.actual_departure) <> '' AND {_ts("s.actual_departure")} IS NULL
              OR trim(s.actual_arrival) <> '' AND {_ts("s.actual_arrival")} IS NULL
                                                                      AS actual_unparseable,
            TRY_CAST(trim(s.container_count) AS INTEGER)              AS container_count_raw,
            TRY_CAST(trim(s.weight_tons) AS DOUBLE)                   AS weight_tons_raw,
            c.cargo_type                                              AS cargo_type,
            a.status                                                  AS status_norm,
            -- list_value keeps NULL positions, so (a, NULL, b) and (a, b, NULL) differ
            md5(to_json(list_value(s.shipment_id, s.customer_id, s.origin_port,
                s.destination_port, s.vessel_id, s.booking_date, s.planned_departure,
                s.actual_departure, s.planned_arrival, s.actual_arrival, s.container_count,
                s.cargo_type, s.weight_tons, s.status))::VARCHAR)     AS row_hash,
            to_json(s)                                                AS raw_record
        FROM raw.shipments s
        LEFT JOIN ref.status_alias a ON a.raw_upper = upper(trim(s.status))
        LEFT JOIN ref.cargo_type   c ON c.cargo_lower = lower(trim(s.cargo_type))
        """
    )

    # 2. Rejection rules (drop / quarantine). Dedup: the LAST occurrence of a shipment_id wins.
    con.execute(
        """
        CREATE TEMP TABLE t_ship_rules AS
        WITH ranked AS (
            SELECT
                t.*,
                max(_row_num) OVER (PARTITION BY shipment_id)                       AS kept_row,
                first_value(row_hash) OVER (PARTITION BY shipment_id ORDER BY _row_num DESC)
                                                                                    AS kept_hash,
                o.port_code IS NOT NULL AND d.port_code IS NOT NULL                 AS ports_ok
            FROM t_ship t
            LEFT JOIN curated.ports o ON o.port_code = t.origin_port
            LEFT JOIN curated.ports d ON d.port_code = t.destination_port
        )
        SELECT *,
            list_filter([
                CASE WHEN NOT regexp_full_match(coalesce(shipment_id, ''), '^SHP-[0-9]{5,}$')
                     THEN 'shipments.invalid_id_format' END,
                CASE WHEN _row_num < kept_row AND row_hash = kept_hash
                     THEN 'shipments.exact_duplicate_row' END,
                CASE WHEN _row_num < kept_row AND row_hash <> kept_hash
                     THEN 'shipments.superseded_version' END,
                CASE WHEN NOT ports_ok THEN 'shipments.unknown_port' END,
                CASE WHEN origin_port = destination_port
                     THEN 'shipments.same_origin_destination' END,
                CASE WHEN booking_date IS NULL OR planned_departure IS NULL
                          OR planned_arrival IS NULL
                     THEN 'shipments.required_timestamp_invalid' END,
                CASE WHEN booking_date > planned_departure OR planned_arrival <= planned_departure
                     THEN 'shipments.planned_chronology_violation' END
            ], x -> x IS NOT NULL) AS reject_reasons
        FROM ranked
        """
    )
    con.execute(
        """
        INSERT INTO curated.rejected_rows
        SELECT 'shipments', _row_num, shipment_id, reject_reasons,
               CASE WHEN reject_reasons = ['shipments.exact_duplicate_row'] THEN 'dropped'
                    ELSE 'quarantined' END,
               raw_record
        FROM t_ship_rules WHERE len(reject_reasons) > 0
        """
    )

    # 3. Survivors: apply fixes, flags and derivations.
    late = ON_TIME_THRESHOLD_HOURS
    con.execute(
        f"""
        CREATE TABLE curated.shipments AS
        WITH x AS (
            SELECT *,
                actual_departure_raw IS NOT NULL AND (
                    actual_departure_raw < booking_date
                    OR actual_departure_raw >= actual_arrival)          AS dep_inconsistent,
                actual_departure_raw > (SELECT snapshot FROM t_snapshot)
                    OR actual_arrival > (SELECT snapshot FROM t_snapshot) AS after_snapshot
            FROM t_ship_rules WHERE len(reject_reasons) = 0
        ), y AS (
            SELECT *,
                CASE WHEN dep_inconsistent THEN NULL ELSE actual_departure_raw END
                                                                        AS actual_departure,
                CASE WHEN status_norm IS NULL THEN 'UNKNOWN'
                     WHEN status_norm IN ('DELIVERED', 'DELAYED') AND actual_arrival IS NULL
                         THEN 'UNKNOWN'
                     ELSE status_norm END                               AS status
            FROM x
        )
        SELECT
            shipment_id, customer_id, origin_port, destination_port,
            origin_port || ' → ' || destination_port                    AS route_key,
            vessel_id, booking_date, planned_departure, actual_departure,
            planned_arrival, actual_arrival,
            CASE WHEN container_count_raw BETWEEN 1 AND {MAX_CONTAINERS_PER_SHIPMENT}
                 THEN container_count_raw END                           AS container_count,
            CASE WHEN weight_tons_raw > 0 THEN weight_tons_raw END      AS weight_tons,
            cargo_type,
            status,
            -- derived ------------------------------------------------------------------
            date_diff('minute', planned_arrival, actual_arrival) / 60.0 AS actual_delay_hours,
            CASE WHEN actual_arrival IS NULL THEN NULL
                 ELSE date_diff('minute', planned_arrival, actual_arrival) / 60.0 <= {late}
            END                                                         AS on_time_flag,
            date_diff('minute', planned_departure, planned_arrival) / 1440.0
                                                                        AS transit_days_planned,
            date_diff('minute', actual_departure, actual_arrival) / 1440.0
                                                                        AS transit_days_actual,
            date_diff('minute', planned_departure, actual_departure) / 60.0
                                                                        AS departure_delay_hours,
            date_diff('minute', booking_date, planned_departure) / 1440.0
                                                                        AS booking_lead_days,
            {
            _flags(
                ("actual_timestamp_unparseable", "actual_unparseable"),
                ("actual_departure_inconsistent", "dep_inconsistent"),
                ("actual_after_snapshot", "coalesce(after_snapshot, false)"),
                ("status_missing", "status_norm IS NULL"),
                (
                    "status_contradicts_actuals",
                    "status_norm IN ('DELIVERED', 'DELAYED') AND actual_arrival IS NULL",
                ),
                (
                    "missing_actuals",
                    "coalesce(status_norm, '') <> 'CANCELLED' AND actual_arrival IS NULL",
                ),
                (
                    "cancelled_with_actuals",
                    "status_norm = 'CANCELLED' AND actual_arrival IS NOT NULL",
                ),
                ("cargo_type_missing", "cargo_type IS NULL"),
                ("weight_non_numeric", "weight_tons_raw IS NULL"),
                ("weight_non_positive", "weight_tons_raw <= 0"),
                (
                    "container_count_non_positive",
                    "container_count_raw IS NULL OR container_count_raw <= 0",
                ),
                ("container_count_outlier", f"container_count_raw > {MAX_CONTAINERS_PER_SHIPMENT}"),
            )
        }                                                          AS dq_flags,
            _row_num                                                    AS _source_row
        FROM y
        ORDER BY shipment_id
        """
    )


def _curate_events(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        f"""
        CREATE TEMP TABLE t_events AS
        SELECT
            e._row_num,
            trim(e.event_id)                                              AS event_id,
            row_number() OVER (PARTITION BY trim(e.event_id) ORDER BY e._row_num) AS occ,
            count(*) OVER (PARTITION BY trim(e.event_id))                 AS id_count,
            {_ts("e.event_timestamp")}                                    AS event_timestamp,
            upper(trim(e.port_code))                                      AS port_code,
            nullif(trim(e.vessel_id), '')                                 AS vessel_id,
            nullif(upper(trim(e.event_type)), '')                         AS event_type,
            TRY_CAST(trim(e.delay_minutes) AS INTEGER)                    AS delay_raw,
            nullif(trim(e.notes), '')                                     AS notes,
            p.port_code IS NOT NULL                                       AS port_ok,
            to_json(e)                                                    AS raw_record
        FROM raw.port_events e
        LEFT JOIN curated.ports p ON p.port_code = upper(trim(e.port_code))
        """
    )
    con.execute(
        """
        INSERT INTO curated.rejected_rows
        SELECT 'port_events', _row_num, event_id,
               list_filter([
                   CASE WHEN event_timestamp IS NULL THEN 'port_events.invalid_timestamp' END,
                   CASE WHEN NOT port_ok THEN 'port_events.unknown_port' END
               ], x -> x IS NOT NULL),
               'quarantined', raw_record
        FROM t_events WHERE event_timestamp IS NULL OR NOT port_ok
        """
    )
    con.execute(
        f"""
        CREATE TABLE curated.port_events AS
        SELECT
            CASE WHEN occ = 1 THEN event_id ELSE event_id || '#' || occ END AS event_key,
            event_id, event_timestamp, port_code, vessel_id, event_type,
            CASE WHEN delay_raw >= 0 THEN delay_raw END                     AS delay_minutes,
            notes,
            {
            _flags(
                ("event_id_collision", "id_count > 1"),
                ("missing_event_type", "event_type IS NULL"),
                ("invalid_delay_minutes", "delay_raw IS NULL OR delay_raw < 0"),
                ("delayed_event_without_delay", "event_type = 'DELAYED' AND delay_raw = 0"),
                (
                    "unknown_vessel",
                    "vessel_id IS NOT NULL AND vessel_id NOT IN "
                    "(SELECT vessel_id FROM t_ship WHERE vessel_id IS NOT NULL)",
                ),
            )
        }                                                              AS dq_flags,
            _row_num                                                        AS _source_row
        FROM t_events
        WHERE event_timestamp IS NOT NULL AND port_ok
        ORDER BY event_timestamp, event_key
        """
    )


def _build_curated(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA curated")
    con.execute(
        """
        CREATE TABLE curated.rejected_rows (
            dataset VARCHAR, _source_row BIGINT, record_id VARCHAR, reasons VARCHAR[],
            disposition VARCHAR, raw_record JSON)
        """
    )
    _curate_ports(con)
    # Data snapshot ("as of"): latest booking or event in the feed. Same definition as DQ.
    con.execute(
        f"""
        CREATE TEMP TABLE t_snapshot AS
        SELECT greatest(
            (SELECT max({_ts("booking_date")}) FROM raw.shipments),
            (SELECT max({_ts("event_timestamp")}) FROM raw.port_events)) AS snapshot
        """
    )
    _curate_shipments(con)
    _curate_events(con)


# ============================================================================================
# serving
# ============================================================================================


def _build_serving(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA serving")
    con.execute(
        """
        CREATE TABLE serving.shipments AS
        SELECT
            s.shipment_id, s.customer_id,
            s.origin_port, o.port_name AS origin_port_name,
            s.destination_port, d.port_name AS destination_port_name,
            s.route_key, s.vessel_id,
            s.booking_date, s.planned_departure, s.actual_departure,
            s.planned_arrival, s.actual_arrival,
            s.container_count, s.cargo_type, s.weight_tons, s.status,
            s.actual_delay_hours, s.on_time_flag,
            s.transit_days_planned, s.transit_days_actual,
            s.departure_delay_hours, s.booking_lead_days, s.dq_flags
        FROM curated.shipments s
        JOIN curated.ports o ON o.port_code = s.origin_port
        JOIN curated.ports d ON d.port_code = s.destination_port
        ORDER BY s.shipment_id
        """
    )
    con.execute("CREATE UNIQUE INDEX serving_shipments_pk ON serving.shipments (shipment_id)")
    con.execute("CREATE TABLE serving.ports AS SELECT * EXCLUDE (_source_row) FROM curated.ports")
    con.execute(
        "CREATE TABLE serving.port_events AS SELECT * EXCLUDE (_source_row) "
        "FROM curated.port_events"
    )

    # Route performance per departure quarter. Delay metrics use arrived shipments only.
    con.execute(
        f"""
        CREATE TABLE serving.route_performance AS
        SELECT
            route_key, origin_port, destination_port,
            date_trunc('quarter', planned_departure)::DATE                   AS quarter_start,
            strftime(planned_departure, '%Y') || '-Q' || quarter(planned_departure) AS quarter,
            count(*)                                                         AS shipment_count,
            count(actual_delay_hours)                                        AS arrived_count,
            count(*) FILTER (WHERE status = 'CANCELLED')                     AS cancelled_count,
            count(*) FILTER (WHERE actual_delay_hours > {ON_TIME_THRESHOLD_HOURS}) AS late_count,
            round(avg(actual_delay_hours), 2)                                AS avg_delay_hours,
            round(median(actual_delay_hours), 2)                             AS median_delay_hours,
            round(quantile_cont(actual_delay_hours, 0.9), 2)                 AS p90_delay_hours,
            round(avg(on_time_flag::INTEGER), 4)                             AS on_time_rate
        FROM serving.shipments
        GROUP BY ALL
        ORDER BY quarter_start, route_key
        """
    )

    # Daily port activity from the event feed: congestion signal for analytics / ML features.
    con.execute(
        """
        CREATE TABLE serving.port_daily_activity AS
        SELECT
            port_code,
            event_timestamp::DATE                                         AS activity_date,
            count(*)                                                      AS events,
            count(*) FILTER (WHERE event_type = 'ARRIVAL')                AS arrivals,
            count(*) FILTER (WHERE event_type = 'DEPARTURE')              AS departures,
            count(*) FILTER (WHERE event_type = 'DELAYED')                AS delayed_events,
            count(*) FILTER (WHERE delay_minutes > 0)                     AS events_with_delay,
            coalesce(sum(delay_minutes), 0)                               AS delay_minutes_total,
            round(avg(delay_minutes), 2)                                  AS avg_delay_minutes
        FROM serving.port_events
        GROUP BY ALL
        ORDER BY activity_date, port_code
        """
    )


def _counts(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    tables = [
        "raw.ports",
        "raw.shipments",
        "raw.port_events",
        "curated.ports",
        "curated.shipments",
        "curated.port_events",
        "serving.shipments",
        "serving.route_performance",
        "serving.port_daily_activity",
    ]
    counts: dict[str, Any] = {
        t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        for t in tables  # type: ignore[index]
    }
    counts["rejected"] = {
        f"{ds}.{disp}": n
        for ds, disp, n in con.execute(
            "SELECT dataset, disposition, count(*) FROM curated.rejected_rows "
            "GROUP BY ALL ORDER BY ALL"
        ).fetchall()
    }
    return counts


def build(raw_dir: Path, db_path: Path) -> dict[str, Any]:
    """Rebuild the warehouse from scratch; atomically replace `db_path` only if all post-load
    assertions pass. Returns row counts."""
    missing = [f for f in RAW_FILES.values() if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw input(s) in {raw_dir}: {', '.join(missing)}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(db_path.suffix + ".tmp")
    tmp_path.unlink(missing_ok=True)
    Path(str(tmp_path) + ".wal").unlink(missing_ok=True)

    sources = {name: _sha256(raw_dir / f) for name, f in RAW_FILES.items()}
    con = duckdb.connect(str(tmp_path))
    try:
        _load_raw(con, raw_dir, sources)
        _build_ref(con)
        _build_curated(con)
        _build_serving(con)
        failures = run_assertions(con)
        if failures:
            raise PipelineAssertionError(failures)
        counts = _counts(con)
        snapshot = con.execute("SELECT snapshot FROM t_snapshot").fetchone()[0]  # type: ignore[index]
        con.execute(
            "CREATE TABLE serving.build_info (built_at TIMESTAMP, data_as_of TIMESTAMP, "
            "source_sha256 JSON, row_counts JSON)"
        )
        con.execute(
            "INSERT INTO serving.build_info VALUES (?, ?, ?, ?)",
            [
                datetime.now(UTC).replace(tzinfo=None),
                snapshot,
                json.dumps(sources),
                json.dumps(counts),
            ],
        )
        con.execute("CHECKPOINT")
    except BaseException:
        con.close()
        tmp_path.unlink(missing_ok=True)
        raise
    con.close()
    os.replace(tmp_path, db_path)
    return counts


def main(argv: list[str] | None = None) -> int:
    from app.logging_setup import configure_logging

    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--db", type=Path, default=Path("data/warehouse/supply_chain.duckdb"))
    args = parser.parse_args(argv)

    start = time.perf_counter()
    try:
        counts = build(args.input, args.db)
    except PipelineAssertionError as exc:
        log.error("pipeline_assertions_failed", extra={"failures": exc.failures})
        return 1
    except Exception:
        log.exception("pipeline_failed")
        return 1
    log.info(
        "pipeline_complete",
        extra={
            "db": str(args.db),
            "row_counts": counts,
            "duration_s": round(time.perf_counter() - start, 2),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
