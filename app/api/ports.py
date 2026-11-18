from __future__ import annotations

from fastapi import APIRouter, Depends

from app.deps import get_repo
from app.repository import ShipmentRepository
from app.schemas import Port

router = APIRouter(tags=["reference"])


@router.get("/ports", response_model=list[Port])
def list_ports(repo: ShipmentRepository = Depends(get_repo)) -> list[dict]:
    """Reference list of ports (code, name, country, region, timezone, congestion score)."""
    return repo.list_ports()
