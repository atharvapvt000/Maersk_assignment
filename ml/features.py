"""Booking-time feature engineering, shared verbatim by training and serving (no train/serve skew).

Every feature is computed AS OF the booking:
  - static booking fields (route, cargo, size, vessel, planned dates)
  - port reference data (congestion score)
  - point-in-time history: route / vessel late rates using only shipments that had ARRIVED
    before the booking, and port activity from events strictly BEFORE the booking date.

`as_of = least(booking_date, data_as_of)`: a booking dated after the data snapshot sees the
latest history we have, never an empty window.

Nothing here reads actual_*, status, or any field only known after departure (see FORBIDDEN).
"""

from __future__ import annotations

import duckdb
import pandas as pd

from app.domain import ON_TIME_THRESHOLD_HOURS

LATE = ON_TIME_THRESHOLD_HOURS
PRIOR_STRENGTH = 10.0  # pseudo-observations pulling sparse history towards the global rate

REQUEST_COLUMNS = [
    "req_id",
    "origin_port",
    "destination_port",
    "cargo_type",
    "container_count",
    "weight_tons",
    "vessel_id",
    "booking_date",
    "planned_departure",
    "planned_arrival",
]

CATEGORICAL = ["origin_port", "destination_port", "cargo_type", "vessel_id", "dep_month"]
NUMERIC = [
    "container_count",
    "weight_tons",
    "booking_lead_days",
    "transit_days_planned",
    "origin_congestion",
    "destination_congestion",
    "route_hist_n",
    "route_hist_late_rate",
    "vessel_hist_n",
    "vessel_hist_late_rate",
    "origin_events_7d",
    "origin_delayed_events_7d",
    "origin_delay_minutes_30d",
    "destination_events_7d",
    "destination_delayed_events_7d",
    "destination_delay_minutes_30d",
]
FEATURES = CATEGORICAL + NUMERIC

# Known only after the booking: using any of these is leakage and disqualifies the model.
FORBIDDEN = {
    "actual_departure",
    "actual_arrival",
    "actual_delay_hours",
    "on_time_flag",
    "status",
    "transit_days_actual",
    "departure_delay_hours",
    "dq_flags",
}

_FEATURE_SQL = f"""
WITH snap AS (SELECT max(data_as_of) AS d FROM serving.build_info),
r AS (
    SELECT q.*,
           origin_port || ' → ' || destination_port               AS route_key,
           least(booking_date, (SELECT d FROM snap))               AS as_of
    FROM req q
),
global_h AS (
    SELECT r.req_id,
           count(h.shipment_id)                                    AS n,
           count(h.shipment_id) FILTER (WHERE h.actual_delay_hours > {LATE}) AS late
    FROM r LEFT JOIN serving.shipments h
      ON h.actual_arrival < r.as_of AND h.actual_delay_hours IS NOT NULL
    GROUP BY r.req_id
),
route_h AS (
    SELECT r.req_id,
           count(h.shipment_id)                                    AS n,
           count(h.shipment_id) FILTER (WHERE h.actual_delay_hours > {LATE}) AS late
    FROM r LEFT JOIN serving.shipments h
      ON h.route_key = r.route_key AND h.actual_arrival < r.as_of
     AND h.actual_delay_hours IS NOT NULL
    GROUP BY r.req_id
),
vessel_h AS (
    SELECT r.req_id,
           count(h.shipment_id)                                    AS n,
           count(h.shipment_id) FILTER (WHERE h.actual_delay_hours > {LATE}) AS late
    FROM r LEFT JOIN serving.shipments h
      ON h.vessel_id = r.vessel_id AND h.actual_arrival < r.as_of
     AND h.actual_delay_hours IS NOT NULL
    GROUP BY r.req_id
),
port_act AS (
    SELECT r.req_id, side,
           coalesce(sum(a.events) FILTER (
               WHERE a.activity_date >= r.as_of::DATE - 7), 0)          AS events_7d,
           coalesce(sum(a.delayed_events) FILTER (
               WHERE a.activity_date >= r.as_of::DATE - 7), 0)          AS delayed_7d,
           coalesce(sum(a.delay_minutes_total), 0)                      AS delay_min_30d
    FROM r
    CROSS JOIN (VALUES ('origin'), ('destination')) s(side)
    LEFT JOIN serving.port_daily_activity a
      ON a.port_code = CASE side WHEN 'origin' THEN r.origin_port ELSE r.destination_port END
     AND a.activity_date <  r.as_of::DATE          -- strictly before the booking day
     AND a.activity_date >= r.as_of::DATE - 30
    GROUP BY r.req_id, side
)
SELECT
    r.req_id,
    r.origin_port,
    r.destination_port,
    coalesce(r.cargo_type, 'MISSING')                                   AS cargo_type,
    coalesce(r.vessel_id, 'MISSING')                                    AS vessel_id,
    CAST(month(r.planned_departure) AS VARCHAR)                         AS dep_month,
    CAST(r.container_count AS DOUBLE)                                   AS container_count,
    CAST(r.weight_tons AS DOUBLE)                                       AS weight_tons,
    date_diff('minute', r.booking_date, r.planned_departure) / 1440.0   AS booking_lead_days,
    date_diff('minute', r.planned_departure, r.planned_arrival) / 1440.0 AS transit_days_planned,
    po.avg_congestion_score                                             AS origin_congestion,
    pd.avg_congestion_score                                             AS destination_congestion,
    CAST(rh.n AS DOUBLE)                                                AS route_hist_n,
    (rh.late + {PRIOR_STRENGTH} * coalesce(g.late / nullif(g.n, 0), 0.5))
        / (rh.n + {PRIOR_STRENGTH})                                     AS route_hist_late_rate,
    CAST(vh.n AS DOUBLE)                                                AS vessel_hist_n,
    (vh.late + {PRIOR_STRENGTH} * coalesce(g.late / nullif(g.n, 0), 0.5))
        / (vh.n + {PRIOR_STRENGTH})                                     AS vessel_hist_late_rate,
    CAST(ao.events_7d AS DOUBLE)                                        AS origin_events_7d,
    CAST(ao.delayed_7d AS DOUBLE)                                       AS origin_delayed_events_7d,
    CAST(ao.delay_min_30d AS DOUBLE)                                    AS origin_delay_minutes_30d,
    CAST(ad.events_7d AS DOUBLE)                                        AS destination_events_7d,
    CAST(ad.delayed_7d AS DOUBLE)                AS destination_delayed_events_7d,
    CAST(ad.delay_min_30d AS DOUBLE)             AS destination_delay_minutes_30d
FROM r
JOIN global_h g  ON g.req_id = r.req_id
JOIN route_h rh  ON rh.req_id = r.req_id
JOIN vessel_h vh ON vh.req_id = r.req_id
JOIN port_act ao ON ao.req_id = r.req_id AND ao.side = 'origin'
JOIN port_act ad ON ad.req_id = r.req_id AND ad.side = 'destination'
LEFT JOIN serving.ports po ON po.port_code = r.origin_port
LEFT JOIN serving.ports pd ON pd.port_code = r.destination_port
ORDER BY r.req_id
"""


def build_features(con: duckdb.DuckDBPyConnection, requests: pd.DataFrame) -> pd.DataFrame:
    """Feature rows for booking requests (columns: REQUEST_COLUMNS). Returns req_id + FEATURES."""
    missing = set(REQUEST_COLUMNS) - set(requests.columns)
    if missing:
        raise ValueError(f"requests missing columns: {sorted(missing)}")
    req = requests[REQUEST_COLUMNS].copy()
    for col in ("booking_date", "planned_departure", "planned_arrival"):
        req[col] = pd.to_datetime(req[col])
    cur = con.cursor()  # thread-safe: the API calls this from FastAPI's threadpool
    try:
        cur.register("req", req)
        out = cur.execute(_FEATURE_SQL).df()
    finally:
        cur.close()
    return out[["req_id", *FEATURES]]


def training_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Labelled rows: every shipment with a known outcome, booking-time features + label."""
    base = con.execute(
        f"""
        SELECT shipment_id AS req_id, origin_port, destination_port, cargo_type,
               container_count, weight_tons, vessel_id, booking_date, planned_departure,
               planned_arrival, (actual_delay_hours > {LATE})::INTEGER AS late
        FROM serving.shipments
        WHERE actual_delay_hours IS NOT NULL
        ORDER BY shipment_id
        """
    ).df()
    feats = build_features(con, base)
    return feats.merge(base[["req_id", "booking_date", "late"]], on="req_id", validate="1:1")
