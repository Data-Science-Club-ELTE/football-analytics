from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.dependencies import db_dependency
from api.schemas import CompetitionOut, Envelope
from database.models import Competition

router = APIRouter(prefix="/api/v1/competitions", tags=["competitions"])


@router.get("", response_model=Envelope)
def list_competitions(db: Session = Depends(db_dependency)):
    rows = db.execute(select(Competition)).scalars().all()
    return Envelope(data=[CompetitionOut.model_validate(r) for r in rows])
