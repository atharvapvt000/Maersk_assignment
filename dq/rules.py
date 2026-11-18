"""Registry of data-quality rules: the single source of truth for *what* each rule means and
*what the pipeline does about it*.

- `dq.dq_check` evaluates every rule against the raw CSVs and reports on it.
- `pipeline.build` implements each rule's `action` and uses `rule.flag` (row-level flag) /
  `rule.name` (quarantine reason) so outcomes are traceable back to a rule.
- `tests/unit/test_reconciliation.py` proves, row by row on the real data, that the report and the
  warehouse agree.

Severity = how bad ONE affected row is.  Thresholds = when the *feed* is considered broken:
`pct_affected >= warn_pct` -> status "warn", `>= fail_pct` -> "fail" (gates the pipeline when
run with `--fail-on fail`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Severity = Literal["critical", "warning", "info"]
Action = Literal["drop", "quarantine", "fix", "flag", "none"]
Dataset = Literal["ports", "shipments", "port_events"]

MAX_CONTAINERS_PER_SHIPMENT = 100
"""Domain sanity bound. Observed p99 is 40; the offending values are 510-996 (likely a unit or
keying error). Fixed in advance rather than fitted to the data so it cannot drift with it."""


@dataclass(frozen=True)
class Rule:
    dataset: Dataset
    rule: str
    severity: Severity
    action: Action
    detection: str
    reasoning: str
    warn_pct: float = 0.0
    fail_pct: float = 100.0
    flagged: bool = False  # True -> pipeline adds `rule` to the row's dq_flags

    @property
    def name(self) -> str:
        return f"{self.dataset}.{self.rule}"


RULES: tuple[Rule, ...] = (
    # ------------------------------------------------------------------ ports (reference data)
    Rule(
        "ports",
        "duplicate_code",
        "critical",
        "drop",
        "port_code (trimmed, upper-cased) occurs more than once.",
        "Reference data must be unique or joins fan out. Keep the first row.",
        warn_pct=0,
        fail_pct=0.01,
    ),
    Rule(
        "ports",
        "missing_country",
        "warning",
        "flag",
        "country is empty after trimming.",
        "Port code, region and timezone are valid, so the port is still usable for joins. "
        "Country should be backfilled upstream (UN/LOCODE prefix 'BE' -> Belgium).",
        warn_pct=0.01,
        fail_pct=20,
        flagged=True,
    ),
    Rule(
        "ports",
        "invalid_congestion_score",
        "warning",
        "fix",
        "avg_congestion_score missing, non-numeric, or outside [0, 1].",
        "Set to NULL rather than guess; it is a descriptive attribute only.",
        warn_pct=0.01,
        fail_pct=20,
        flagged=True,
    ),
    # ------------------------------------------------------------------ shipments: identity
    Rule(
        "shipments",
        "invalid_id_format",
        "critical",
        "quarantine",
        "shipment_id does not match ^SHP-\\d{5,}$.",
        "Row cannot be addressed via the API or joined reliably.",
        warn_pct=0.01,
        fail_pct=1,
    ),
    Rule(
        "shipments",
        "exact_duplicate_row",
        "warning",
        "drop",
        "Row is identical in all 14 fields to the LAST row with the same shipment_id.",
        "Pure redundancy (e.g. a double export). Dropping loses no information.",
        warn_pct=0.01,
        fail_pct=5,
    ),
    Rule(
        "shipments",
        "superseded_version",
        "warning",
        "quarantine",
        "shipment_id repeats with differing field values; every occurrence except the last "
        "in file order.",
        "There is no updated_at column, so file order is the only recency signal: keep the "
        "last version, keep older versions in quarantine for audit instead of silently "
        "deleting them.",
        warn_pct=0.01,
        fail_pct=5,
    ),
    # ------------------------------------------------------------------ shipments: references
    Rule(
        "shipments",
        "unknown_port",
        "critical",
        "quarantine",
        "origin_port or destination_port not in ports reference (seen: XXTST, ZZZZZ, UNKNW).",
        "Placeholder/test codes cannot be attributed to a real route; including them would "
        "invent routes in route stats and the ML model.",
        warn_pct=0.5,
        fail_pct=5,
    ),
    Rule(
        "shipments",
        "same_origin_destination",
        "critical",
        "quarantine",
        "origin_port equals destination_port.",
        "Not a valid route.",
        warn_pct=0.01,
        fail_pct=1,
    ),
    # ------------------------------------------------------------------ shipments: time
    Rule(
        "shipments",
        "required_timestamp_invalid",
        "critical",
        "quarantine",
        "booking_date, planned_departure or planned_arrival missing or not 'YYYY-MM-DD HH:MM:SS'.",
        "Delay, transit time and every ML feature depend on the plan dates.",
        warn_pct=0.01,
        fail_pct=1,
    ),
    Rule(
        "shipments",
        "planned_chronology_violation",
        "critical",
        "quarantine",
        "booking_date > planned_departure, or planned_arrival <= planned_departure.",
        "The plan itself is impossible, so derived delays would be meaningless.",
        warn_pct=0.01,
        fail_pct=1,
    ),
    Rule(
        "shipments",
        "actual_timestamp_unparseable",
        "warning",
        "fix",
        "actual_departure/actual_arrival present but not a valid timestamp.",
        "Treat as missing (NULL) and flag.",
        warn_pct=0.01,
        fail_pct=5,
        flagged=True,
    ),
    Rule(
        "shipments",
        "actual_departure_inconsistent",
        "warning",
        "fix",
        "actual_departure < booking_date (left before it was booked) or "
        "actual_departure >= actual_arrival.",
        "The departure timestamp is wrong, but actual_arrival is independently plausible, so "
        "the headline metric (arrival delay) is kept. Only actual_departure is set to NULL, "
        "which nulls transit_days_actual.",
        warn_pct=0.1,
        fail_pct=5,
        flagged=True,
    ),
    Rule(
        "shipments",
        "actual_after_snapshot",
        "info",
        "flag",
        "actual_departure/actual_arrival later than the data snapshot, inferred as "
        "max(booking_date, event_timestamp).",
        "Outcomes recorded after the extract date are suspicious in a real feed. Here they "
        "look like a simulation artefact, so they are kept and flagged, not dropped. For ML "
        "they are labels, not features.",
        warn_pct=10,
        fail_pct=50,
        flagged=True,
    ),
    # ------------------------------------------------------------------ shipments: status
    Rule(
        "shipments",
        "status_nonstandard_spelling",
        "info",
        "fix",
        "status is a known synonym but not canonical: Complete, COMPLETED, delivered.",
        "Normalise case and synonyms: COMPLETE/COMPLETED -> DELIVERED.",
        warn_pct=1,
        fail_pct=20,
    ),
    Rule(
        "shipments",
        "status_missing",
        "warning",
        "fix",
        "status empty or a placeholder such as 'N/A'.",
        "Set status to UNKNOWN. Metrics derive from timestamps, never from status.",
        warn_pct=0.5,
        fail_pct=10,
        flagged=True,
    ),
    Rule(
        "shipments",
        "status_contradicts_actuals",
        "warning",
        "fix",
        "status (after normalisation) is DELIVERED or DELAYED, but actual_arrival is empty.",
        "A delivered shipment without an arrival time is self-contradictory. In this data it "
        "is the same rows as the non-standard spellings: one corrupted batch. Status is "
        "untrustworthy, so set it to UNKNOWN.",
        warn_pct=0.5,
        fail_pct=10,
        flagged=True,
    ),
    Rule(
        "shipments",
        "missing_actuals",
        "warning",
        "flag",
        "Not cancelled and no actual_arrival. In this data all 146 rows are the status-corrupted "
        "batch (see status_missing, status_contradicts_actuals). None is plausibly in "
        "transit: only 4 have planned_arrival after the snapshot.",
        "Keep the row (booking fields are fine). Exclude it from delay metrics and from ML "
        "training labels, because the outcome is unknown.",
        warn_pct=1,
        fail_pct=15,
        flagged=True,
    ),
    Rule(
        "shipments",
        "cancelled_with_actuals",
        "warning",
        "flag",
        "status CANCELLED but actual_arrival present.",
        "Status and timestamps contradict each other. Keep the timestamps and flag.",
        warn_pct=0.01,
        fail_pct=5,
        flagged=True,
    ),
    # ------------------------------------------------------------------ shipments: attributes
    Rule(
        "shipments",
        "cargo_type_nonstandard_format",
        "info",
        "fix",
        "cargo_type matches a canonical category only after trim + case-folding "
        "(e.g. ' Chemicals', 'FURNITURE', 'machinery ').",
        "Normalise to the canonical spelling so group-bys and ML categories are not split.",
        warn_pct=1,
        fail_pct=30,
    ),
    Rule(
        "shipments",
        "cargo_type_missing",
        "warning",
        "fix",
        "cargo_type empty or not one of the 10 known categories.",
        "Set to NULL and flag. The ML pipeline treats NULL as its own category.",
        warn_pct=0.1,
        fail_pct=10,
        flagged=True,
    ),
    Rule(
        "shipments",
        "weight_non_numeric",
        "warning",
        "fix",
        "weight_tons empty or not a number.",
        "Set to NULL and flag. Impute only inside the ML feature pipeline, never in the data "
        "product.",
        warn_pct=0.5,
        fail_pct=10,
        flagged=True,
    ),
    Rule(
        "shipments",
        "weight_non_positive",
        "warning",
        "fix",
        "weight_tons <= 0.",
        "Physically impossible, most likely a sign error. Set to NULL and flag rather than "
        "taking abs().",
        warn_pct=0.5,
        fail_pct=10,
        flagged=True,
    ),
    Rule(
        "shipments",
        "container_count_non_positive",
        "warning",
        "fix",
        "container_count empty, not an integer, or <= 0.",
        "A shipment needs at least one container. Set to NULL and flag.",
        warn_pct=0.1,
        fail_pct=10,
        flagged=True,
    ),
    Rule(
        "shipments",
        "container_count_outlier",
        "warning",
        "fix",
        f"container_count > {MAX_CONTAINERS_PER_SHIPMENT} (domain bound; observed p99 = 40, "
        "offending values 510-996).",
        "Probably a keying or unit error (x10 / x25). Set to NULL and flag. Delay data on the "
        "row is still valid, so the row is kept.",
        warn_pct=0.1,
        fail_pct=5,
        flagged=True,
    ),
    # ------------------------------------------------------------------ port events
    Rule(
        "port_events",
        "invalid_timestamp",
        "critical",
        "quarantine",
        "event_timestamp missing or unparseable.",
        "Events cannot be ordered or windowed without a time.",
        warn_pct=0.01,
        fail_pct=1,
    ),
    Rule(
        "port_events",
        "unknown_port",
        "critical",
        "quarantine",
        "port_code not in ports reference.",
        "Cannot be attributed to a port.",
        warn_pct=0.01,
        fail_pct=1,
    ),
    Rule(
        "port_events",
        "event_id_collision",
        "warning",
        "fix",
        "event_id shared by rows whose other fields ALL differ (timestamp, port, vessel, type). "
        "These are distinct events, not replays. All rows in a colliding group are counted.",
        "Dropping 'duplicates' would delete real events. Keep every row and give it a stable "
        "surrogate event_key: the first occurrence keeps event_id, later ones get "
        "event_id#2, #3, and so on. Flag the rows and report the ID generator upstream.",
        warn_pct=0.1,
        fail_pct=5,
        flagged=True,
    ),
    Rule(
        "port_events",
        "missing_event_type",
        "warning",
        "fix",
        "event_type empty.",
        "Keep the row (it still carries delay and port activity) with event_type NULL and "
        "flag it. Type-specific aggregates ignore it.",
        warn_pct=0.1,
        fail_pct=5,
        flagged=True,
    ),
    Rule(
        "port_events",
        "invalid_delay_minutes",
        "warning",
        "fix",
        "delay_minutes empty, not an integer, or negative.",
        "Set to NULL and flag.",
        warn_pct=0.01,
        fail_pct=5,
        flagged=True,
    ),
    Rule(
        "port_events",
        "delayed_event_without_delay",
        "info",
        "flag",
        "event_type = DELAYED but delay_minutes = 0. Every other DELAYED event has a delay "
        "greater than 0.",
        "Type and value disagree, and it is unknown which one is right. Keep and flag. "
        "Delay aggregates use delay_minutes as recorded.",
        warn_pct=1,
        fail_pct=10,
        flagged=True,
    ),
    Rule(
        "port_events",
        "unknown_vessel",
        "info",
        "flag",
        "vessel_id never appears in shipments.csv.",
        "May be vessels outside this customer's bookings. Informational only.",
        warn_pct=5,
        fail_pct=50,
        flagged=True,
    ),
    Rule(
        "port_events",
        "missing_notes",
        "info",
        "none",
        "notes empty.",
        "Free-text notes are optional. No action.",
        warn_pct=50,
        fail_pct=100,
    ),
)

RULES_BY_NAME: dict[str, Rule] = {r.name: r for r in RULES}
