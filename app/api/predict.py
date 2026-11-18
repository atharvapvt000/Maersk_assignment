from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.deps import get_predictor, get_repo
from app.errors import NotFoundError, UnprocessableError
from app.repository import ShipmentRepository
from app.schemas import DelayPrediction, PredictDelayRequest
from app.services.predictor import BookingFeatures, Predictor, risk_band

router = APIRouter(tags=["ml"])
log = logging.getLogger(__name__)


def _features_from_shipment(row: dict) -> BookingFeatures:
    # Only booking-time columns are copied; actual_* and status are never passed to the model.
    if row["planned_departure"] is None or row["planned_arrival"] is None:
        raise UnprocessableError(
            f"Shipment {row['shipment_id']} lacks planned dates needed for prediction",
            code="INSUFFICIENT_FEATURES",
        )
    return BookingFeatures(
        origin_port=row["origin_port"],
        destination_port=row["destination_port"],
        planned_departure=row["planned_departure"],
        planned_arrival=row["planned_arrival"],
        booking_date=row["booking_date"],
        cargo_type=row["cargo_type"],
        container_count=row["container_count"],
        weight_tons=row["weight_tons"],
        vessel_id=row["vessel_id"],
    )


@router.post("/predict-delay", response_model=DelayPrediction)
def predict_delay(
    body: PredictDelayRequest,
    repo: ShipmentRepository = Depends(get_repo),
    predictor: Predictor = Depends(get_predictor),
) -> DelayPrediction:
    if body.shipment_id is not None:
        row = repo.get_shipment(body.shipment_id)
        if row is None:
            raise NotFoundError(f"Shipment {body.shipment_id} not found", code="SHIPMENT_NOT_FOUND")
        features = _features_from_shipment(row)
    else:
        unknown = [
            p
            for p in (body.origin_port, body.destination_port)
            if not repo.port_exists(p)  # type: ignore[arg-type]
        ]
        if unknown:
            raise UnprocessableError(
                f"Unknown port code(s): {', '.join(unknown)}",
                code="UNKNOWN_PORT",
                details={"unknown_ports": unknown},
            )
        features = BookingFeatures(**body.model_dump(exclude={"shipment_id"}))

    probability = float(predictor.predict_proba(features))
    result = DelayPrediction(
        shipment_id=body.shipment_id,
        delay_probability=round(probability, 4),
        predicted_late=probability >= predictor.threshold,
        risk_band=risk_band(probability),
        threshold=predictor.threshold,
        model_version=predictor.version,
    )
    log.info(
        "prediction",
        extra={
            "shipment_id": body.shipment_id,
            "route_key": features.route_key,
            "delay_probability": result.delay_probability,
            "model_version": predictor.version,
        },
    )
    return result
