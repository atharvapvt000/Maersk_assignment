#!/usr/bin/env sh
# One-shot data preparation: DQ checks on raw CSVs, then raw -> curated -> serving build.
# Used by the `pipeline` compose service and `make data`. Safe to re-run (idempotent).
set -eu

RAW_DIR="${RAW_DIR:-data/raw}"
DB_PATH="${DB_PATH:-data/warehouse/supply_chain.duckdb}"
DQ_REPORT_PATH="${DQ_REPORT_PATH:-data/reports/dq_report.json}"
PYTHON="${PYTHON:-python}"

"$PYTHON" -m dq.dq_check --input "$RAW_DIR" --output "$DQ_REPORT_PATH" --fail-on "${DQ_FAIL_ON:-never}"
"$PYTHON" -m pipeline.build --input "$RAW_DIR" --db "$DB_PATH"
