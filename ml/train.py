"""Train and honestly evaluate the delay-risk model; write the artefact + model card.

    python -m ml.train --db data/warehouse/supply_chain.duckdb --out models/

Protocol
  1. Time-based split on booking_date (the model always predicts the future in production):
       train < 2025-04-01 <= validation < 2025-07-01 <= test
  2. Candidates: base-rate prior, route-history baseline, logistic regression, gradient boosting.
     Hyper-parameters and the decision threshold are chosen on VALIDATION only.
  3. Every candidate is refit on train+validation and scored ONCE on test.
  4. Uncertainty: bootstrap 95% CI for test AUC, and a label-permutation test
     ("could this AUC arise from a model with no signal?").
  5. Leakage canary: the same pipeline given one post-departure field must score far higher,
     proving the harness would expose leakage if it existed.
  6. The selected model (lowest validation log-loss) is shipped exactly as evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ml.features import CATEGORICAL, FEATURES, FORBIDDEN, NUMERIC, training_frame

log = logging.getLogger("ml.train")

TRAIN_END = pd.Timestamp("2025-04-01")
VAL_END = pd.Timestamp("2025-07-01")
SEED = 42
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 1000

assert not set(FEATURES) & FORBIDDEN, "leakage: forbidden column in FEATURES"


# ============================================================================================
# Candidates
# ============================================================================================


class ColumnScore(BaseEstimator, ClassifierMixin):
    """Non-learned baseline: uses one feature column directly as the probability."""

    def __init__(self, column: str = "route_hist_late_rate") -> None:
        self.column = column

    def fit(self, X: pd.DataFrame, y: Any) -> ColumnScore:
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X[self.column].to_numpy(dtype=float).clip(0, 1)
        return np.column_stack([1 - p, p])


class Prior(BaseEstimator, ClassifierMixin):
    """Predicts the training base rate for everyone (the 'no model' reference)."""

    def fit(self, X: pd.DataFrame, y: Any) -> Prior:
        self.classes_ = np.array([0, 1])
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])


def make_logreg(C: float, numeric: list[str] = NUMERIC) -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20),
                CATEGORICAL,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=C, max_iter=5000))])


def make_hgb(learning_rate: float, max_depth: int) -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL,
            ),
            ("num", "passthrough", NUMERIC),
        ]
    )
    clf = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_iter=200,
        l2_regularization=1.0,
        categorical_features=list(range(len(CATEGORICAL))),
        random_state=SEED,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


@dataclass
class Candidate:
    name: str
    grid: list[dict[str, Any]]
    factory: Any


CANDIDATES = [
    Candidate("prior", [{}], lambda: Prior()),
    Candidate("route_history_baseline", [{}], lambda: ColumnScore("route_hist_late_rate")),
    Candidate("logistic_regression", [{"C": c} for c in (0.001, 0.01, 0.1, 1.0)], make_logreg),
    Candidate(
        "gradient_boosting",
        [{"learning_rate": lr, "max_depth": d} for lr in (0.03, 0.1) for d in (2, 4)],
        make_hgb,
    ),
]


# ============================================================================================
# Metrics
# ============================================================================================


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.unique(np.round(p, 4))
    if len(grid) == 1:
        return float(grid[0])
    scores = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = p >= threshold
    single_class = len(np.unique(p)) == 1
    return {
        "n": len(y),
        "positive_rate": round(float(np.mean(y)), 4),
        "threshold": round(threshold, 4),
        "predicted_positive_rate": round(float(np.mean(pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": None if single_class else round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4),
    }


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, rng: np.random.Generator) -> list[float]:
    n, aucs = len(y), []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)]


def permutation_p_value(y: np.ndarray, p: np.ndarray, rng: np.random.Generator) -> float:
    observed = roc_auc_score(y, p)
    null = np.array([roc_auc_score(rng.permutation(y), p) for _ in range(N_PERMUTATIONS)])
    return round(float((np.sum(null >= observed) + 1) / (N_PERMUTATIONS + 1)), 4)


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 5) -> list[dict[str, Any]]:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & ((p <= hi) if i == bins - 1 else (p < hi))
        if mask.any():
            out.append(
                {
                    "bin": f"{lo:.3f}-{hi:.3f}",
                    "n": int(mask.sum()),
                    "mean_predicted": round(float(p[mask].mean()), 4),
                    "observed_rate": round(float(y[mask].mean()), 4),
                }
            )
    return out


# ============================================================================================
# Training run
# ============================================================================================


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    b = pd.to_datetime(df["booking_date"])
    return df[b < TRAIN_END], df[(b >= TRAIN_END) & (b < VAL_END)], df[b >= VAL_END]


def _proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def run(db_path: Path, out_dir: Path) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    con = duckdb.connect(str(db_path), read_only=True)
    df = training_frame(con)
    dep_delay = con.execute(
        "SELECT shipment_id AS req_id, departure_delay_hours FROM serving.shipments"
    ).df()
    data_as_of = str(con.execute("SELECT max(data_as_of) FROM serving.build_info").fetchone()[0])
    con.close()

    train, val, test = split(df)
    trainval = pd.concat([train, val])
    y_tr, y_va, y_te = (d["late"].to_numpy() for d in (train, val, test))
    y_tv = trainval["late"].to_numpy()
    log.info("split", extra={"train": len(train), "validation": len(val), "test": len(test)})

    # ---- 1. model selection on validation -----------------------------------------------
    selection: dict[str, Any] = {}
    for cand in CANDIDATES:
        best: dict[str, Any] | None = None
        for params in cand.grid:
            model = cand.factory(**params).fit(train[FEATURES], y_tr)
            p = _proba(model, val[FEATURES])
            score = log_loss(y_va, np.clip(p, 1e-6, 1 - 1e-6))
            if best is None or score < best["val_log_loss"]:
                best = {
                    "params": params,
                    "val_log_loss": round(float(score), 4),
                    "val_auc": round(float(roc_auc_score(y_va, p)), 4)
                    if len(np.unique(p)) > 1
                    else None,
                    "threshold": best_f1_threshold(y_va, p),
                }
        selection[cand.name] = best

    # ---- 2. refit on train+val, score ONCE on test ---------------------------------------
    results: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    for cand in CANDIDATES:
        sel = selection[cand.name]
        model = cand.factory(**sel["params"]).fit(trainval[FEATURES], y_tv)
        p = _proba(model, test[FEATURES])
        fitted[cand.name] = model
        res = {
            "params": sel["params"],
            "validation": {"log_loss": sel["val_log_loss"], "roc_auc": sel["val_auc"]},
            "test": classification_metrics(y_te, p, sel["threshold"]),
            "test_at_0_5": classification_metrics(y_te, p, 0.5),
        }
        if len(np.unique(p)) > 1:
            res["test_auc_95ci"] = bootstrap_auc_ci(y_te, p, rng)
            res["test_auc_permutation_p"] = permutation_p_value(y_te, p, rng)
        results[cand.name] = res

    always_late = classification_metrics(y_te, np.ones(len(y_te)), 0.5)

    learned = [c for c in ("logistic_regression", "gradient_boosting")]
    chosen = min(learned, key=lambda n: selection[n]["val_log_loss"])
    model = fitted[chosen]
    p_te = _proba(model, test[FEATURES])

    # ---- 3. diagnostics ----------------------------------------------------------------------
    imp = permutation_importance(
        model, test[FEATURES], y_te, scoring="roc_auc", n_repeats=10, random_state=SEED
    )
    importance = sorted(
        (
            {"feature": f, "auc_drop_mean": round(float(m), 4), "auc_drop_std": round(float(s), 4)}
            for f, m, s in zip(FEATURES, imp.importances_mean, imp.importances_std, strict=True)
        ),
        key=lambda d: -d["auc_drop_mean"],
    )

    # Leakage canary: same logistic pipeline + ONE post-departure field.
    tv_c = trainval.merge(dep_delay, on="req_id")
    te_c = test.merge(dep_delay, on="req_id")
    canary = make_logreg(1.0, [*NUMERIC, "departure_delay_hours"]).fit(
        tv_c[[*FEATURES, "departure_delay_hours"]], tv_c["late"]
    )
    canary_auc = roc_auc_score(
        te_c["late"], canary.predict_proba(te_c[[*FEATURES, "departure_delay_hours"]])[:, 1]
    )

    # ---- 4. artefact + card ----------------------------------------------------------------------
    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(df[["req_id", *FEATURES, "late"]], index=False).values.tobytes()
    ).hexdigest()
    version = f"delay-{chosen.replace('_', '-')}-v1-{data_hash[:8]}"
    threshold = selection[chosen]["threshold"]

    card = {
        "model_version": version,
        "selected_model": chosen,
        "selection_rule": "lowest validation log-loss among learned models",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_as_of": data_as_of,
        "training_data_sha256": data_hash,
        "sklearn_version": sklearn.__version__,
        "label": "actual_arrival - planned_arrival > 24h (arrived shipments only)",
        "features": {"categorical": CATEGORICAL, "numeric": NUMERIC},
        "excluded_as_leakage": sorted(FORBIDDEN),
        "split": {
            "rule": "time-based on booking_date",
            "train": f"< {TRAIN_END.date()} (n={len(train)})",
            "validation": f"[{TRAIN_END.date()}, {VAL_END.date()}) (n={len(val)})",
            "test": f">= {VAL_END.date()} (n={len(test)})",
            "shipped_model_trained_on": "train + validation",
        },
        "decision_threshold": threshold,
        "results": results,
        "always_predict_late_reference": always_late,
        "test_calibration": reliability(y_te, p_te),
        "test_permutation_importance": importance,
        "leakage_canary": {
            "description": "same logistic pipeline + departure_delay_hours (post-departure)",
            "test_roc_auc": round(float(canary_auc), 4),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": model,
            "model_version": version,
            "threshold": threshold,
            "features": FEATURES,
            "sklearn_version": sklearn.__version__,
        },
        out_dir / "delay_model.joblib",
        compress=3,
    )
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    return card


def main(argv: list[str] | None = None) -> int:
    from app.logging_setup import configure_logging

    configure_logging("INFO")
    parser = argparse.ArgumentParser(description="Train the delay-risk model.")
    parser.add_argument("--db", type=Path, default=Path("data/warehouse/supply_chain.duckdb"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args(argv)
    card = run(args.db, args.out)
    chosen = card["results"][card["selected_model"]]
    log.info(
        "training_complete",
        extra={
            "model_version": card["model_version"],
            "test": chosen["test"],
            "test_auc_95ci": chosen.get("test_auc_95ci"),
            "permutation_p": chosen.get("test_auc_permutation_p"),
            "leakage_canary_auc": card["leakage_canary"]["test_roc_auc"],
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
