from datetime import date, datetime
from typing import Any

from pydantic import BaseModel


# ── Envelope ────────────────────────────────────────────────────


class Envelope(BaseModel):
    success: bool = True
    data: Any


# ── Competition ─────────────────────────────────────────────────


class CompetitionOut(BaseModel):
    id: int
    competition_name: str
    season_name: str
    statsbomb_competition_id: int
    statsbomb_season_id: int
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Match ───────────────────────────────────────────────────────


class MatchOut(BaseModel):
    id: int
    competition_id: int
    home_team: str
    away_team: str
    match_date: date
    statsbomb_match_id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class MatchDetailOut(MatchOut):
    event_count: int


# ── Event counts ────────────────────────────────────────────────


class EventCountOut(BaseModel):
    passes: int
    carries: int
    shots: int
    goals: int
    total: int


# ── xT ──────────────────────────────────────────────────────────


class ZoneXTOut(BaseModel):
    zone_name: str
    xt_value: float
    direct_goal_pct: float
    loss_pct: float
    event_count: int
    team: str | None = None

    model_config = {"from_attributes": True}


class XTResultOut(BaseModel):
    scope: str
    zones: list[ZoneXTOut]


# ── Zone definitions ────────────────────────────────────────────


class ZoneDefOut(BaseModel):
    code: str
    x: float
    y: float
    width: float
    height: float
