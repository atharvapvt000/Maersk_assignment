"""Domain constants shared by the pipeline, API and ML code (single source of truth)."""

from enum import StrEnum

ON_TIME_THRESHOLD_HOURS = 24.0
"""A shipment is on time when actual_arrival - planned_arrival <= this many hours."""

SHIPMENT_ID_PATTERN = r"^SHP-\d{5,}$"
PORT_CODE_PATTERN = r"^[A-Za-z]{5}$"


class ShipmentStatus(StrEnum):
    DELIVERED = "DELIVERED"
    DELAYED = "DELAYED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


# Raw status spellings seen in shipments.csv -> canonical status.
STATUS_ALIASES: dict[str, ShipmentStatus] = {
    "DELIVERED": ShipmentStatus.DELIVERED,
    "COMPLETE": ShipmentStatus.DELIVERED,
    "COMPLETED": ShipmentStatus.DELIVERED,
    "DELAYED": ShipmentStatus.DELAYED,
    "CANCELLED": ShipmentStatus.CANCELLED,
    "CANCELED": ShipmentStatus.CANCELLED,
}

CANONICAL_CARGO_TYPES: tuple[str, ...] = (
    "Apparel",
    "Automotive Parts",
    "Chemicals",
    "Consumer Goods",
    "Electronics",
    "Food & Beverage",
    "Furniture",
    "Machinery",
    "Pharmaceuticals",
    "Raw Materials",
)
