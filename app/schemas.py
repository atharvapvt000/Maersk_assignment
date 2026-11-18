"""Pydantic request/response models — the API's public contract."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain import PORT_CODE_PATTERN, SHIPMENT_ID_PATTERN, ShipmentStatus


class Shipment(BaseModel):
    shipment_id: str
    customer_id: str | None
    origin_port: str
    origin_port_name: str | None
    destination_port: str
    destination_port_name: str | None
    route_key: str
    vessel_id: str | None
    booking_date: datetime | None
    planned_departure: datetime | None
    actual_departure: datetime | None
    planned_arrival: datetime | None
    actual_arrival: datetime | None
    container_count: int | None
    cargo_type: str | None
    weight_tons: float | None
    status: ShipmentStatus
    actual_delay_hours: float | None = Field(
        description="actual_arrival - planned_arrival in hours; null until the shipment arrives"
    )
    on_time_flag: bool | None = Field(description="actual_delay_hours <= 24; null if not arrived")
    transit_days_planned: float | None
    transit_days_actual: float | None
    dq_flags: list[str] = Field(default_factory=list, description="Row-level data-quality flags")


class ShipmentPage(BaseModel):
    items: list[Shipment]
    total: int
    limit: int
    offset: int


class RouteStats(BaseModel):
    origin: str
    destination: str
    route_key: str
    shipment_count: int = Field(description="All shipments on the route in the window")
    arrived_count: int = Field(description="Shipments with an actual arrival (used for metrics)")
    cancelled_count: int
    avg_delay_hours: float | None
    on_time_rate: float | None = Field(description="Share of arrived shipments with delay <= 24h")
    departure_from: date | None = None
    departure_to: date | None = None


class Port(BaseModel):
    port_code: str
    port_name: str
    country: str | None
    region: str | None
    timezone: str | None
    avg_congestion_score: float | None


class RankedRoute(BaseModel):
    rank: int
    route_key: str
    origin_port: str
    destination_port: str
    shipment_count: int
    arrived_count: int
    late_count: int
    avg_delay_hours: float | None
    on_time_rate: float | None


class RouteRanking(BaseModel):
    quarter: str | None = Field(description="Planned-departure quarter, e.g. 2025-Q3; null = all")
    metric: str
    order: Literal["asc", "desc"]
    min_arrived: int
    data_as_of: str | None = Field(description="Snapshot date of the data ('this quarter' anchor)")
    available_quarters: list[str]
    routes: list[RankedRoute]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    git_sha: str
    checks: dict[str, dict[str, object]]


class PredictDelayRequest(BaseModel):
    """Either `shipment_id` (features looked up from the curated store) or booking-time fields.

    Only information known at booking time is accepted — no actual departure/arrival.
    """

    model_config = ConfigDict(extra="forbid")

    shipment_id: str | None = Field(default=None, pattern=SHIPMENT_ID_PATTERN)
    origin_port: str | None = Field(default=None, pattern=PORT_CODE_PATTERN)
    destination_port: str | None = Field(default=None, pattern=PORT_CODE_PATTERN)
    cargo_type: str | None = None
    container_count: int | None = Field(default=None, ge=1, le=10_000)
    weight_tons: float | None = Field(default=None, gt=0, le=1_000_000)
    vessel_id: str | None = None
    booking_date: datetime | None = None
    planned_departure: datetime | None = None
    planned_arrival: datetime | None = None

    @model_validator(mode="after")
    def _one_mode(self) -> Self:
        feature_fields = (
            "origin_port",
            "destination_port",
            "cargo_type",
            "container_count",
            "weight_tons",
            "vessel_id",
            "booking_date",
            "planned_departure",
            "planned_arrival",
        )
        given = [f for f in feature_fields if getattr(self, f) is not None]
        if self.shipment_id is not None:
            if given:
                raise ValueError("Provide either shipment_id or booking features, not both")
            return self
        required = (
            "origin_port",
            "destination_port",
            "booking_date",
            "planned_departure",
            "planned_arrival",
        )
        missing = [f for f in required if getattr(self, f) is None]
        if missing:
            raise ValueError(f"Missing required booking fields: {', '.join(missing)}")
        self.origin_port = self.origin_port.upper()  # type: ignore[union-attr]
        self.destination_port = self.destination_port.upper()  # type: ignore[union-attr]
        if self.origin_port == self.destination_port:
            raise ValueError("origin_port and destination_port must differ")
        if self.planned_arrival <= self.planned_departure:  # type: ignore[operator]
            raise ValueError("planned_arrival must be after planned_departure")
        if self.booking_date and self.booking_date > self.planned_departure:  # type: ignore[operator]
            raise ValueError("booking_date must not be after planned_departure")
        return self


class DelayPrediction(BaseModel):
    shipment_id: str | None
    delay_probability: float = Field(ge=0, le=1, description="P(arrival > 24h late)")
    predicted_late: bool
    risk_band: Literal["low", "medium", "high"]
    threshold: float
    model_version: str
