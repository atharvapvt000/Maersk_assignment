from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Path, Query

from app.deps import get_repo
from app.domain import PORT_CODE_PATTERN
from app.errors import BadRequestError, NotFoundError
from app.repository import ShipmentRepository
from app.schemas import RankedRoute, RouteRanking, RouteStats

router = APIRouter(prefix="/routes", tags=["routes"])

PortCode = Annotated[str, Path(pattern=PORT_CODE_PATTERN, examples=["CNSHA"])]


QUARTER_PATTERN = r"^\d{4}-Q[1-4]$"


@router.get("/rankings", response_model=RouteRanking)
def route_rankings(
    quarter: Annotated[
        str | None,
        Query(pattern=QUARTER_PATTERN, description="Planned-departure quarter, e.g. 2025-Q3"),
    ] = None,
    metric: Literal["avg_delay_hours", "on_time_rate", "late_count", "shipment_count"] = (
        "avg_delay_hours"
    ),
    order: Literal["asc", "desc"] = "desc",
    min_arrived: Annotated[int, Query(ge=1, le=1000)] = 5,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    repo: ShipmentRepository = Depends(get_repo),
) -> RouteRanking:
    """Rank routes by a delay metric, optionally within one departure quarter.

    `min_arrived` guards against tiny samples topping the list (one late shipment = 100% late).
    """
    quarters = repo.quarters()
    if quarter and quarter not in quarters:
        raise NotFoundError(
            f"No data for quarter {quarter}",
            code="QUARTER_NOT_FOUND",
            details={"available_quarters": quarters},
        )
    rows = repo.route_rankings(quarter, metric, order == "asc", min_arrived, limit)
    return RouteRanking(
        quarter=quarter,
        metric=metric,
        order=order,
        min_arrived=min_arrived,
        data_as_of=repo.ping().get("data_as_of"),
        available_quarters=quarters,
        routes=[RankedRoute(rank=i, **r) for i, r in enumerate(rows, start=1)],
    )


@router.get("/{origin}/{destination}/stats", response_model=RouteStats)
def route_stats(
    origin: PortCode,
    destination: PortCode,
    departure_from: date | None = None,
    departure_to: date | None = None,
    repo: ShipmentRepository = Depends(get_repo),
) -> RouteStats:
    """Aggregates over shipments on the route. Delay metrics use arrived shipments only.

    Unknown port codes -> 404. A valid route with no shipments -> 200 with zero counts and
    null metrics ("no data" is an answer, not an error).
    """
    origin, destination = origin.upper(), destination.upper()
    if origin == destination:
        raise BadRequestError("origin and destination must differ", code="INVALID_ROUTE")
    if departure_from and departure_to and departure_from > departure_to:
        raise BadRequestError(
            "departure_from must be on or before departure_to", code="INVALID_DATE_RANGE"
        )
    unknown = [p for p in (origin, destination) if not repo.port_exists(p)]
    if unknown:
        raise NotFoundError(
            f"Unknown port code(s): {', '.join(unknown)}",
            code="PORT_NOT_FOUND",
            details={"unknown_ports": unknown},
        )
    stats = repo.route_stats(origin, destination, departure_from, departure_to)
    return RouteStats(
        origin=origin,
        destination=destination,
        route_key=f"{origin} → {destination}",
        departure_from=departure_from,
        departure_to=departure_to,
        **stats,
    )
