"""Standalone data-quality checks over the RAW CSV files: no database, no ingestion, stdlib only.

    python -m dq.dq_check --input data/raw --output data/reports/dq_report.json \
        [--markdown dq.md] [--fail-on fail|warn|never]

Every rule in `dq.rules.RULES` is evaluated and reported with: rows affected, % of total,
severity, pass/warn/fail status against its thresholds, how it was detected, the action the
pipeline takes, the reasoning, and up to 5 sample ids. The same JSON is served by
GET /data-quality/report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.domain import CANONICAL_CARGO_TYPES, SHIPMENT_ID_PATTERN, STATUS_ALIASES, ShipmentStatus
from dq.rules import MAX_CONTAINERS_PER_SHIPMENT, RULES, Rule

REPORT_VERSION = "2.0"
TS_FORMAT = "%Y-%m-%d %H:%M:%S"
SAMPLE_SIZE = 5
INPUT_FILES = {"ports": "ports.csv", "shipments": "shipments.csv", "port_events": "port_events.csv"}
ID_FIELD = {"ports": "port_code", "shipments": "shipment_id", "port_events": "event_id"}

Row = dict[str, str]

# --------------------------------------------------------------------------------------------
# Parsing helpers (pure; mirror the pipeline's TRY_CAST semantics)
# --------------------------------------------------------------------------------------------


def parse_ts(value: str | None) -> datetime | None:
    try:
        return datetime.strptime((value or "").strip(), TS_FORMAT)
    except ValueError:
        return None


def parse_float(value: str | None) -> float | None:
    try:
        return float((value or "").strip())
    except ValueError:
        return None


def parse_int(value: str | None) -> int | None:
    try:
        return int((value or "").strip())
    except ValueError:
        return None


CARGO_BY_LOWER = {c.lower(): c for c in CANONICAL_CARGO_TYPES}
CANONICAL_STATUS = {s.value for s in ShipmentStatus} - {ShipmentStatus.UNKNOWN.value}


def normalise_status(raw: str) -> str | None:
    """Canonical status for a raw value, or None if it is missing / unrecognised."""
    status = STATUS_ALIASES.get(raw.strip().upper())
    return status.value if status else None


def read_csv(path: Path) -> list[Row]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------------------------
# Cross-row context (computed once per run)
# --------------------------------------------------------------------------------------------


@dataclass
class Context:
    port_codes: set[str]
    shipment_vessels: set[str]
    snapshot: datetime | None
    dup_port_rows: set[int] = field(default_factory=set)
    exact_dup_rows: set[int] = field(default_factory=set)
    superseded_rows: set[int] = field(default_factory=set)
    event_collision_rows: set[int] = field(default_factory=set)


def _build_context(data: dict[str, list[Row]]) -> Context:
    ports, shipments, events = data["ports"], data["shipments"], data["port_events"]

    stamps = [parse_ts(r["booking_date"]) for r in shipments]
    stamps += [parse_ts(r["event_timestamp"]) for r in events]
    valid = [s for s in stamps if s]

    ctx = Context(
        port_codes={r["port_code"].strip().upper() for r in ports},
        shipment_vessels={r["vessel_id"].strip() for r in shipments if r["vessel_id"].strip()},
        snapshot=max(valid) if valid else None,
    )

    seen: set[str] = set()
    for i, r in enumerate(ports, start=1):
        code = r["port_code"].strip().upper()
        if code in seen:
            ctx.dup_port_rows.add(i)
        seen.add(code)

    # Duplicate shipment_ids: the LAST occurrence is kept; earlier ones are either exact copies
    # of it (drop) or superseded versions (quarantine). Mirrors pipeline dedup exactly.
    by_id: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(shipments, start=1):
        by_id[r["shipment_id"].strip()].append(i)
    for rows in by_id.values():
        if len(rows) < 2:
            continue
        kept = tuple(shipments[rows[-1] - 1].values())
        for i in rows[:-1]:
            target = (
                ctx.exact_dup_rows
                if tuple(shipments[i - 1].values()) == kept
                else (ctx.superseded_rows)
            )
            target.add(i)

    ev_ids = Counter(r["event_id"].strip() for r in events)
    ctx.event_collision_rows = {
        i for i, r in enumerate(events, start=1) if ev_ids[r["event_id"].strip()] > 1
    }
    return ctx


# --------------------------------------------------------------------------------------------
# Rule predicates: (row, 1-based row number, context) -> affected?
# --------------------------------------------------------------------------------------------

Predicate = Callable[[Row, int, Context], bool]


def _ship_plan(r: Row) -> tuple[datetime | None, datetime | None, datetime | None]:
    return (
        parse_ts(r["booking_date"]),
        parse_ts(r["planned_departure"]),
        parse_ts(r["planned_arrival"]),
    )


def _actual_departure_inconsistent(r: Row, _i: int, _c: Context) -> bool:
    ad, aa, b = (
        parse_ts(r["actual_departure"]),
        parse_ts(r["actual_arrival"]),
        parse_ts(r["booking_date"]),
    )
    return ad is not None and ((b is not None and ad < b) or (aa is not None and ad >= aa))


def _planned_chronology(r: Row, _i: int, _c: Context) -> bool:
    b, pd, pa = _ship_plan(r)
    return bool((b and pd and b > pd) or (pd and pa and pa <= pd))


def _after_snapshot(r: Row, _i: int, c: Context) -> bool:
    if c.snapshot is None:
        return False
    return any(
        (t := parse_ts(r[col])) is not None and t > c.snapshot
        for col in ("actual_departure", "actual_arrival")
    )


def _not_cancelled(r: Row) -> bool:
    return normalise_status(r["status"]) != ShipmentStatus.CANCELLED.value


PREDICATES: dict[str, Predicate] = {
    # ports
    "ports.duplicate_code": lambda r, i, c: i in c.dup_port_rows,
    "ports.missing_country": lambda r, i, c: not r["country"].strip(),
    "ports.invalid_congestion_score": lambda r, i, c: (
        (s := parse_float(r["avg_congestion_score"])) is None or not 0 <= s <= 1
    ),
    # shipments: identity
    "shipments.invalid_id_format": lambda r, i, c: (
        not re.fullmatch(SHIPMENT_ID_PATTERN, r["shipment_id"].strip())
    ),
    "shipments.exact_duplicate_row": lambda r, i, c: i in c.exact_dup_rows,
    "shipments.superseded_version": lambda r, i, c: i in c.superseded_rows,
    # shipments: references
    "shipments.unknown_port": lambda r, i, c: (
        r["origin_port"].strip().upper() not in c.port_codes
        or r["destination_port"].strip().upper() not in c.port_codes
    ),
    "shipments.same_origin_destination": lambda r, i, c: (
        r["origin_port"].strip().upper() == r["destination_port"].strip().upper()
    ),
    # shipments: time
    "shipments.required_timestamp_invalid": lambda r, i, c: any(x is None for x in _ship_plan(r)),
    "shipments.planned_chronology_violation": _planned_chronology,
    "shipments.actual_timestamp_unparseable": lambda r, i, c: any(
        r[col].strip() and parse_ts(r[col]) is None
        for col in ("actual_departure", "actual_arrival")
    ),
    "shipments.actual_departure_inconsistent": _actual_departure_inconsistent,
    "shipments.actual_after_snapshot": _after_snapshot,
    # shipments: status
    "shipments.status_nonstandard_spelling": lambda r, i, c: (
        r["status"] not in CANONICAL_STATUS and normalise_status(r["status"]) is not None
    ),
    "shipments.status_missing": lambda r, i, c: normalise_status(r["status"]) is None,
    "shipments.status_contradicts_actuals": lambda r, i, c: (
        normalise_status(r["status"])
        in {ShipmentStatus.DELIVERED.value, ShipmentStatus.DELAYED.value}
        and parse_ts(r["actual_arrival"]) is None
    ),
    "shipments.missing_actuals": lambda r, i, c: (
        _not_cancelled(r) and parse_ts(r["actual_arrival"]) is None
    ),
    "shipments.cancelled_with_actuals": lambda r, i, c: (
        not _not_cancelled(r) and parse_ts(r["actual_arrival"]) is not None
    ),
    # shipments: attributes
    "shipments.cargo_type_nonstandard_format": lambda r, i, c: (
        r["cargo_type"] not in CANONICAL_CARGO_TYPES
        and r["cargo_type"].strip().lower() in CARGO_BY_LOWER
    ),
    "shipments.cargo_type_missing": lambda r, i, c: (
        r["cargo_type"].strip().lower() not in CARGO_BY_LOWER
    ),
    "shipments.weight_non_numeric": lambda r, i, c: parse_float(r["weight_tons"]) is None,
    "shipments.weight_non_positive": lambda r, i, c: (
        (w := parse_float(r["weight_tons"])) is not None and w <= 0
    ),
    "shipments.container_count_non_positive": lambda r, i, c: (
        (k := parse_int(r["container_count"])) is None or k <= 0
    ),
    "shipments.container_count_outlier": lambda r, i, c: (
        (k := parse_int(r["container_count"])) is not None and k > MAX_CONTAINERS_PER_SHIPMENT
    ),
    # port events
    "port_events.invalid_timestamp": lambda r, i, c: parse_ts(r["event_timestamp"]) is None,
    "port_events.unknown_port": lambda r, i, c: r["port_code"].strip().upper() not in c.port_codes,
    "port_events.event_id_collision": lambda r, i, c: i in c.event_collision_rows,
    "port_events.missing_event_type": lambda r, i, c: not r["event_type"].strip(),
    "port_events.invalid_delay_minutes": lambda r, i, c: (
        (d := parse_int(r["delay_minutes"])) is None or d < 0
    ),
    "port_events.delayed_event_without_delay": lambda r, i, c: (
        r["event_type"].strip().upper() == "DELAYED" and parse_int(r["delay_minutes"]) == 0
    ),
    "port_events.unknown_vessel": lambda r, i, c: (
        bool(r["vessel_id"].strip()) and r["vessel_id"].strip() not in c.shipment_vessels
    ),
    "port_events.missing_notes": lambda r, i, c: not r["notes"].strip(),
}


# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------


def status_for(rule: Rule, pct: float) -> str:
    if pct > 0 and pct >= rule.fail_pct:
        return "fail"
    if pct > 0 and pct >= rule.warn_pct:
        return "warn"
    return "pass"


def evaluate(input_dir: Path) -> tuple[dict[str, Any], dict[str, set[int]]]:
    """Run all rules. Returns (JSON-able report, {rule name: set of 1-based raw row numbers})."""
    missing = [f for f in INPUT_FILES.values() if not (input_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s) in {input_dir}: {', '.join(missing)}")
    data = {name: read_csv(input_dir / f) for name, f in INPUT_FILES.items()}
    ctx = _build_context(data)

    if set(PREDICATES) != {r.name for r in RULES}:  # registry and code must agree
        raise RuntimeError("dq.rules.RULES and PREDICATES are out of sync")

    hits: dict[str, set[int]] = {}
    checks: list[dict[str, Any]] = []
    for rule in RULES:
        rows = data[rule.dataset]
        pred = PREDICATES[rule.name]
        affected = [(i, r) for i, r in enumerate(rows, start=1) if pred(r, i, ctx)]
        hits[rule.name] = {i for i, _ in affected}
        pct = round(100 * len(affected) / len(rows), 2) if rows else 0.0
        samples = list(dict.fromkeys(r[ID_FIELD[rule.dataset]] for _, r in affected))
        checks.append(
            {
                "check": rule.name,
                "dataset": rule.dataset,
                "severity": rule.severity,
                "status": status_for(rule, pct),
                "rows_affected": len(affected),
                "total_rows": len(rows),
                "pct_affected": pct,
                "thresholds": {"warn_pct": rule.warn_pct, "fail_pct": rule.fail_pct},
                "action": rule.action,
                "detection": rule.detection,
                "reasoning": rule.reasoning,
                "sample_ids": samples[:SAMPLE_SIZE],
            }
        )

    statuses = Counter(c["status"] for c in checks)
    report = {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_snapshot_inferred": ctx.snapshot.isoformat() if ctx.snapshot else None,
        "inputs": {
            name: {
                "file": f,
                "rows": len(data[name]),
                "sha256": hashlib.sha256((input_dir / f).read_bytes()).hexdigest(),
            }
            for name, f in INPUT_FILES.items()
        },
        "summary": {
            "checks_run": len(checks),
            "pass": statuses.get("pass", 0),
            "warn": statuses.get("warn", 0),
            "fail": statuses.get("fail", 0),
            "issues_found": sum(1 for c in checks if c["rows_affected"]),
            "issues_by_severity": dict(
                Counter(c["severity"] for c in checks if c["rows_affected"])
            ),
            "failing_checks": [c["check"] for c in checks if c["status"] == "fail"],
        },
        "checks": checks,
    }
    return report, hits


def run_checks(input_dir: Path) -> dict[str, Any]:
    return evaluate(input_dir)[0]


def to_markdown(report: dict[str, Any]) -> str:
    """README table: every rule that found something, ordered by severity then size."""
    order = {"critical": 0, "warning": 1, "info": 2}
    rows = sorted(
        (c for c in report["checks"] if c["rows_affected"]),
        key=lambda c: (order[c["severity"]], -c["rows_affected"]),
    )
    lines = [
        "| Check | Rows affected | Severity | Status | Decision | How detected | Reasoning |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in rows:
        cells = [
            f"`{c['check']}`",
            f"{c['rows_affected']:,} / {c['total_rows']:,} ({c['pct_affected']}%)",
            c["severity"],
            c["status"],
            f"**{c['action']}**",
            c["detection"].replace("|", "\\|"),
            c["reasoning"].replace("|", "\\|"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    clean = [c["check"] for c in report["checks"] if not c["rows_affected"]]
    lines.append("")
    lines.append(f"Checks that found nothing ({len(clean)}): " + ", ".join(f"`{c}`" for c in clean))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run data-quality checks on raw CSVs.")
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/dq_report.json"))
    parser.add_argument("--markdown", type=Path, help="Also write a Markdown summary table here.")
    parser.add_argument(
        "--fail-on",
        choices=["fail", "warn", "never"],
        default="never",
        help="Exit 2 if any check's status is at/above this level (use 'fail' to gate a pipeline).",
    )
    args = parser.parse_args(argv)

    report, _ = evaluate(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(args.output)  # atomic: the API never serves a half-written report
    if args.markdown:
        args.markdown.write_text(to_markdown(report))

    print(json.dumps({"msg": "dq_check_complete", "output": str(args.output), **report["summary"]}))
    blocking = {"fail": {"fail"}, "warn": {"warn", "fail"}, "never": set()}[args.fail_on]
    return 2 if any(c["status"] in blocking for c in report["checks"]) else 0


if __name__ == "__main__":
    sys.exit(main())
