from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from app.deps import get_repo
from app.domain import PORT_CODE_PATTERN, SHIPMENT_ID_PATTERN, ShipmentStatus
from app.errors import BadRequestError, NotFoundError
from app.repository import ShipmentFilters, ShipmentRepository
from app.schemas import Shipment, ShipmentPage

router = APIRouter(prefix="/shipments", tags=["shipments"])

MAX_PAGE_SIZE = 200


@router.get("/{shipment_id}", response_model=Shipment)
def get_shipment(
    shipment_id: Annotated[str, Path(pattern=SHIPMENT_ID_PATTERN, examples=["SHP-00421"])],
    repo: ShipmentRepository = Depends(get_repo),
) -> dict:
    row = repo.get_shipment(shipment_id)
    if row is None:
        raise NotFoundError(f"Shipment {shipment_id} not found", code="SHIPMENT_NOT_FOUND")
    return row


@router.get("", response_model=ShipmentPage)
def list_shipments(
    origin: Annotated[str | None, Query(pattern=PORT_CODE_PATTERN)] = None,
    destination: Annotated[str | None, Query(pattern=PORT_CODE_PATTERN)] = None,
    status: ShipmentStatus | None = None,
    departure_from: Annotated[date | None, Query(description="planned_departure >= (date)")] = None,
    departure_to: Annotated[
        date | None, Query(description="planned_departure <= (inclusive)")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    repo: ShipmentRepository = Depends(get_repo),
) -> ShipmentPage:
    if departure_from and departure_to and departure_from > departure_to:
        raise BadRequestError(
            "departure_from must be on or before departure_to", code="INVALID_DATE_RANGE"
        )
    filters = ShipmentFilters(
        origin=origin.upper() if origin else None,
        destination=destination.upper() if destination else None,
        status=status.value if status else None,
        departure_from=departure_from,
        departure_to=departure_to,
    )
    items, total = repo.list_shipments(filters, limit, offset)
    return ShipmentPage(
        items=[Shipment.model_validate(i) for i in items], total=total, limit=limit, offset=offset
    )
