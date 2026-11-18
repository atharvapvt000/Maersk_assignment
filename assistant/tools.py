"""Tools the assistant may call. Every tool is a thin, read-only client of the HTTP API, so the
assistant gets exactly the validation, data and error semantics any other consumer gets.

Each declaration is a JSON Schema the model sees; `ToolBox.call()` validates the model's arguments
again before any request is made (never trust model-produced arguments).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

PORT = {"type": "string", "pattern": "^[A-Z]{5}$", "description": "UN/LOCODE, e.g. CNSHA"}
DATE = {"type": "string", "format": "date", "description": "YYYY-MM-DD (planned departure)"}

DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "query_shipments",
        "description": (
            "List shipments from the curated data product, filtered by route, status and planned "
            "departure date range. Returns the total match count and up to `limit` shipments."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "origin": PORT,
                "destination": PORT,
                "status": {
                    "type": "string",
                    "enum": ["DELIVERED", "DELAYED", "CANCELLED", "UNKNOWN"],
                },
                "departure_from": DATE,
                "departure_to": DATE,
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_shipment",
        "description": "Full curated record for one shipment, including delay and dq_flags.",
        "parameters": {
            "type": "object",
            "properties": {"shipment_id": {"type": "string", "pattern": "^SHP-\\d{5,}$"}},
            "required": ["shipment_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_route_stats",
        "description": (
            "Aggregated delay metrics for one origin->destination route: shipment count, "
            "average arrival delay (hours), on-time rate (delay <= 24h). Optional departure window."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "origin": PORT,
                "destination": PORT,
                "departure_from": DATE,
                "departure_to": DATE,
            },
            "required": ["origin", "destination"],
            "additionalProperties": False,
        },
    },
    {
        "name": "rank_routes",
        "description": (
            "Rank routes by a delay metric, optionally within one planned-departure quarter "
            "(e.g. '2025-Q3'). Use for 'which routes are worst/best' questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "quarter": {"type": "string", "pattern": "^\\d{4}-Q[1-4]$"},
                "metric": {
                    "type": "string",
                    "enum": ["avg_delay_hours", "on_time_rate", "late_count", "shipment_count"],
                },
                "order": {"type": "string", "enum": ["asc", "desc"]},
                "min_arrived": {"type": "integer", "minimum": 1, "maximum": 100},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["metric", "order"],
            "additionalProperties": False,
        },
    },
    {
        "name": "predict_delay",
        "description": (
            "Delay-risk prediction from the ML model for an existing shipment: probability of "
            "arriving more than 24h late, risk band and model version."
        ),
        "parameters": {
            "type": "object",
            "properties": {"shipment_id": {"type": "string", "pattern": "^SHP-\\d{5,}$"}},
            "required": ["shipment_id"],
            "additionalProperties": False,
        },
    },
]

_BY_NAME = {d["name"]: d for d in DECLARATIONS}

# Fields returned per shipment in list results: enough to answer questions, cheap in tokens.
_LIST_FIELDS = (
    "shipment_id",
    "route_key",
    "status",
    "cargo_type",
    "planned_departure",
    "planned_arrival",
    "actual_arrival",
    "actual_delay_hours",
    "on_time_flag",
)


class ToolArgumentError(ValueError):
    pass


def validate_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Minimal JSON-schema check for our flat schemas (types, enums, patterns, bounds)."""
    decl = _BY_NAME.get(name)
    if decl is None:
        raise ToolArgumentError(f"unknown tool {name!r}")
    schema = decl["parameters"]
    props = schema["properties"]
    unknown = set(args) - set(props)
    if unknown:
        raise ToolArgumentError(f"unexpected argument(s): {sorted(unknown)}")
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        raise ToolArgumentError(f"missing required argument(s): {missing}")
    clean: dict[str, Any] = {}
    for key, value in args.items():
        spec = props[key]
        if spec["type"] == "integer":
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ToolArgumentError(f"{key} must be an integer")
            if not spec.get("minimum", value) <= value <= spec.get("maximum", value):
                raise ToolArgumentError(f"{key} out of range")
        else:
            if not isinstance(value, str):
                raise ToolArgumentError(f"{key} must be a string")
            value = value.strip()
            if key in ("origin", "destination", "shipment_id"):
                value = value.upper()
            if "enum" in spec and value not in spec["enum"]:
                raise ToolArgumentError(f"{key} must be one of {spec['enum']}")
            if "pattern" in spec and not re.fullmatch(spec["pattern"], value):
                raise ToolArgumentError(f"{key} has an invalid format")
            if spec.get("format") == "date" and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ToolArgumentError(f"{key} must be YYYY-MM-DD")
        clean[key] = value
    return clean


@dataclass
class ToolResult:
    tool_result_id: str
    name: str
    args: dict[str, Any]
    ok: bool
    data: dict[str, Any]
    latency_ms: float
    http_status: int | None = None

    def for_model(self) -> dict[str, Any]:
        key = "data" if self.ok else "error"
        return {"tool_result_id": self.tool_result_id, key: self.data}


@dataclass
class ToolBox:
    http: httpx.Client
    _counter: int = field(default=0, init=False)

    def call(self, name: str, raw_args: dict[str, Any]) -> ToolResult:
        self._counter += 1
        rid = f"T{self._counter}"
        start = time.perf_counter()
        try:
            args = validate_args(name, dict(raw_args or {}))
        except ToolArgumentError as exc:
            return ToolResult(
                rid,
                name,
                dict(raw_args or {}),
                False,
                {"code": "INVALID_ARGUMENTS", "message": str(exc)},
                0.0,
            )
        try:
            status, data = getattr(self, f"_{name}")(args)
        except httpx.HTTPError as exc:
            status, data = None, {"code": "API_UNREACHABLE", "message": type(exc).__name__}
        ok = status is not None and 200 <= status < 300
        if not ok and "error" in data:
            data = data["error"]  # unwrap the API's error envelope
        latency = round((time.perf_counter() - start) * 1000, 1)
        return ToolResult(rid, name, args, ok, data, latency, status)

    # ---- tool implementations (one HTTP call each) ------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        r = self.http.get(path, params=params)
        return r.status_code, r.json()

    def _query_shipments(self, a: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        params = {"limit": a.pop("limit", 20), **a}
        status, body = self._get("/shipments", params)
        if status == 200:
            body = {
                "total": body["total"],
                "returned": len(body["items"]),
                "items": [{k: s.get(k) for k in _LIST_FIELDS} for s in body["items"]],
            }
        return status, body

    def _get_shipment(self, a: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return self._get(f"/shipments/{a['shipment_id']}")

    def _get_route_stats(self, a: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        o, d = a.pop("origin"), a.pop("destination")
        return self._get(f"/routes/{o}/{d}/stats", a)

    def _rank_routes(self, a: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return self._get("/routes/rankings", a)

    def _predict_delay(self, a: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        r = self.http.post("/predict-delay", json={"shipment_id": a["shipment_id"]})
        return r.status_code, r.json()
