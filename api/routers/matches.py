from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.dependencies import db_dependency
from api.schemas import Envelope, EventCountOut, MatchDetailOut, MatchOut
from database.models import Event, EventType, Match

router = APIRouter(prefix="/api/v1/matches", tags=["matches"])


@router.get("", response_model=Envelope)
def list_matches(
    competition_id: int | None = Query(None),
    db: Session = Depends(db_dependency),
):
    stmt = select(Match)
    if competition_id is not None:
        stmt = stmt.where(Match.competition_id == competition_id)
    stmt = stmt.order_by(Match.match_date)

    rows = db.execute(stmt).scalars().all()
    return Envelope(data=[MatchOut.model_validate(r) for r in rows])


@router.get("/{match_id}", response_model=Envelope)
def get_match(match_id: int, db: Session = Depends(db_dependency)):
    match = db.get(Match, match_id)
    if not match:
        raise HTTPException(404, detail="Match not found")

    counts = db.execute(
        select(
            func.count().label("total"),
            func.count().filter(Event.event_type == EventType.PASS).label("passes"),
            func.count().filter(Event.event_type == EventType.CARRY).label("carries"),
            func.count().filter(Event.event_type == EventType.SHOT).label("shots"),
            func.count().filter(Event.is_goal.is_(True)).label("goals"),
        ).where(Event.match_id == match_id)
    ).one()

    data = MatchOut.model_validate(match).model_dump()
    data["event_count"] = counts.total
    data["events"] = EventCountOut(
        passes=counts.passes,
        carries=counts.carries,
        shots=counts.shots,
        goals=counts.goals,
        total=counts.total,
    )

    return Envelope(data=data)
