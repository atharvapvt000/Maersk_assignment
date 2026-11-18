from __future__ import annotations

import json
from datetime import datetime

import pytest

from app.services.predictor import risk_band
from dq import dq_check
from dq.rules import RULES, Rule

# ----------------------------------------------------------------------------- DQ


@pytest.fixture
def evaluated(raw_dir):
    return dq_check.evaluate(raw_dir)


@pytest.mark.parametrize(
    ("name", "rows"),
    [
        ("ports.missing_country", {3}),
        ("shipments.superseded_version", {1}),
        ("shipments.exact_duplicate_row", {2}),
        ("shipments.unknown_port", {5}),
        ("shipments.status_nonstandard_spelling", {5, 6, 8, 9}),
        ("shipments.status_missing", {3}),
        ("shipments.status_contradicts_actuals", {9}),  # row 5/8 'Complete' HAVE actuals
        ("shipments.missing_actuals", {3, 9}),  # row 4 is cancelled
        ("shipments.cargo_type_nonstandard_format", {2, 3, 4, 7}),
        ("shipments.weight_non_numeric", {4}),
        ("shipments.weight_non_positive", {3}),
        ("shipments.container_count_non_positive", {3}),
        ("shipments.container_count_outlier", {8}),
        ("shipments.actual_departure_inconsistent", {8}),
        ("shipments.actual_after_snapshot", set()),
        ("shipments.planned_chronology_violation", set()),
        ("port_events.event_id_collision", {1, 2}),
        ("port_events.missing_event_type", {3}),
        ("port_events.delayed_event_without_delay", {4}),
        ("port_events.unknown_vessel", {4}),
        ("port_events.missing_notes", {3}),
    ],
)
def test_rule_hits(evaluated, name, rows):
    _, hits = evaluated
    assert hits[name] == rows


def test_check_result_shape(evaluated):
    report, _ = evaluated
    c = next(c for c in report["checks"] if c["check"] == "shipments.unknown_port")
    assert c["severity"] == "critical"
    assert c["action"] == "quarantine"
    assert c["total_rows"] == 9
    assert c["pct_affected"] == pytest.approx(11.11)
    assert c["status"] == "fail"  # 11% >= fail threshold of 5%
    assert c["sample_ids"] == ["SHP-00005"]
    assert c["detection"] and c["reasoning"]
    assert report["data_snapshot_inferred"] == "2024-03-01T00:00:00"
    assert report["summary"]["checks_run"] == len(RULES)


@pytest.mark.parametrize(
    ("pct", "expected"), [(0.0, "pass"), (0.5, "pass"), (1.0, "warn"), (4.99, "warn"), (5, "fail")]
)
def test_status_thresholds(pct, expected):
    rule = Rule("shipments", "x", "warning", "flag", "d", "r", warn_pct=1, fail_pct=5)
    assert dq_check.status_for(rule, pct) == expected


def test_every_rule_has_a_predicate_and_docs():
    assert set(dq_check.PREDICATES) == {r.name for r in RULES}
    for r in RULES:
        assert r.detection and r.reasoning, r.name
        assert r.warn_pct <= r.fail_pct, r.name


def test_cli_writes_report_markdown_and_gates(raw_dir, tmp_path):
    out, md = tmp_path / "r.json", tmp_path / "r.md"
    assert (
        dq_check.main(["--input", str(raw_dir), "--output", str(out), "--markdown", str(md)]) == 0
    )
    assert json.loads(out.read_text())["inputs"]["shipments"]["rows"] == 9
    assert "`shipments.unknown_port`" in md.read_text()
    # The tiny fixture breaches fail thresholds on purpose -> gate trips.
    assert dq_check.main(["--input", str(raw_dir), "--output", str(out), "--fail-on", "fail"]) == 2


def test_missing_input_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        dq_check.evaluate(tmp_path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2024-01-01 10:00:00", datetime(2024, 1, 1, 10)),
        ("", None),
        ("01/02/2024", None),
        (None, None),
    ],
)
def test_parse_ts(raw, expected):
    assert dq_check.parse_ts(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("DELIVERED", "DELIVERED"),
        (" complete ", "DELIVERED"),
        ("COMPLETED", "DELIVERED"),
        ("delivered", "DELIVERED"),
        ("N/A", None),
        ("", None),
    ],
)
def test_normalise_status(raw, expected):
    assert dq_check.normalise_status(raw) == expected


# ----------------------------------------------------------------------------- risk bands


@pytest.mark.parametrize(
    ("p", "band"),
    [(0.0, "low"), (0.29, "low"), (0.3, "medium"), (0.59, "medium"), (0.6, "high"), (1.0, "high")],
)
def test_risk_band(p, band):
    assert risk_band(p) == band
