"""The DQ report and the warehouse must tell the same story, row by row.

For every rule in dq.rules, the set of raw rows the standalone DQ step reports must equal the set
of rows the pipeline acted on:
  drop / quarantine -> rows in curated.rejected_rows with that rule as a reason
  flagged rules     -> surviving curated rows carrying the rule in dq_flags

Run on both the synthetic fixture and the REAL shipped CSVs (embedded DuckDB, ~1-2 s total).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from dq import dq_check
from dq.rules import RULES, Rule
from pipeline.build import build

REAL_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
CURATED = {
    "ports": "curated.ports",
    "shipments": "curated.shipments",
    "port_events": "curated.port_events",
}


def _reconcile(raw_dir: Path, tmp_path: Path) -> dict[str, tuple[set[int], set[int]]]:
    _, hits = dq_check.evaluate(raw_dir)
    db = tmp_path / "wh.duckdb"
    build(raw_dir, db)
    con = duckdb.connect(str(db), read_only=True)
    mismatches: dict[str, tuple[set[int], set[int]]] = {}
    try:
        for rule in RULES:
            if rule.action in ("drop", "quarantine"):
                acted = _rows(
                    con,
                    "SELECT _source_row FROM curated.rejected_rows "
                    "WHERE dataset = ? AND list_contains(reasons, ?)",
                    [rule.dataset, rule.name],
                )
                expected = hits[rule.name]
            elif rule.flagged:
                acted = _rows(
                    con,
                    f"SELECT _source_row FROM {CURATED[rule.dataset]} "
                    "WHERE list_contains(dq_flags, ?)",
                    [rule.rule],
                )
                survivors = _rows(con, f"SELECT _source_row FROM {CURATED[rule.dataset]}", [])
                expected = hits[rule.name] & survivors
            else:
                continue
            if acted != expected:
                mismatches[rule.name] = (expected - acted, acted - expected)
    finally:
        con.close()
    return mismatches


def _rows(con: duckdb.DuckDBPyConnection, sql: str, params: list) -> set[int]:
    return {r[0] for r in con.execute(sql, params).fetchall()}


def test_every_rule_is_reconcilable():
    # A rule must either reject rows, flag rows, or be an explicitly unflagged fix/none.
    unflagged = {r.name for r in RULES if r.action in ("fix", "flag") and not r.flagged}
    assert unflagged == {
        "shipments.status_nonstandard_spelling",  # outcome checked: status domain assertion
        "shipments.cargo_type_nonstandard_format",  # outcome checked below
    }
    assert all(isinstance(r, Rule) for r in RULES)


def test_reconciles_on_fixture(raw_dir, tmp_path):
    assert _reconcile(raw_dir, tmp_path) == {}


@pytest.mark.slow
@pytest.mark.skipif(not REAL_RAW.exists(), reason="real data not present")
def test_reconciles_on_real_data(tmp_path):
    mismatches = _reconcile(REAL_RAW, tmp_path)
    assert mismatches == {}, {k: (sorted(a)[:5], sorted(b)[:5]) for k, (a, b) in mismatches.items()}


@pytest.mark.slow
@pytest.mark.skipif(not REAL_RAW.exists(), reason="real data not present")
def test_real_data_normalisation_outcomes(tmp_path):
    db = tmp_path / "wh.duckdb"
    build(REAL_RAW, db)  # separate build on purpose: asserts on outcomes, not on the report
    con = duckdb.connect(str(db), read_only=True)
    cargo = {
        r[0] for r in con.execute("SELECT DISTINCT cargo_type FROM curated.shipments").fetchall()
    }
    statuses = dict(
        con.execute("SELECT status, count(*) FROM curated.shipments GROUP BY 1").fetchall()
    )
    events = con.execute("SELECT count(*) FROM curated.port_events").fetchone()[0]
    con.close()
    from app.domain import CANONICAL_CARGO_TYPES

    assert cargo <= {*CANONICAL_CARGO_TYPES, None}
    assert set(statuses) <= {"DELIVERED", "DELAYED", "CANCELLED", "UNKNOWN"}
    assert events == 25_000  # colliding event ids are kept, not deduplicated
