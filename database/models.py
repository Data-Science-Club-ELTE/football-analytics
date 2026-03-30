import enum
from datetime import date, datetime

from sqlalchemy import (
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Boolean,
    Date,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ── Enums ──────────────────────────────────────────────────────


class EventType(str, enum.Enum):
    PASS = "PASS"
    CARRY = "CARRY"
    SHOT = "SHOT"


class Scope(str, enum.Enum):
    MATCH = "MATCH"
    SEASON = "SEASON"


# ── Tables ─────────────────────────────────────────────────────


class Competition(Base):
    __tablename__ = "competitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    competition_name: Mapped[str] = mapped_column(String(120))
    season_name: Mapped[str] = mapped_column(String(60))
    statsbomb_competition_id: Mapped[int] = mapped_column(Integer)
    statsbomb_season_id: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    matches: Mapped[list["Match"]] = relationship(back_populates="competition")
    xt_results: Mapped[list["XTResult"]] = relationship(back_populates="competition")


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"))
    home_team: Mapped[str] = mapped_column(String(120))
    away_team: Mapped[str] = mapped_column(String(120))
    match_date: Mapped[date] = mapped_column(Date)
    statsbomb_match_id: Mapped[int] = mapped_column(Integer, unique=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    competition: Mapped["Competition"] = relationship(back_populates="matches")
    events: Mapped[list["Event"]] = relationship(back_populates="match")
    xt_results: Mapped[list["XTResult"]] = relationship(back_populates="match")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    event_type: Mapped[EventType] = mapped_column(Enum(EventType))
    team: Mapped[str] = mapped_column(String(120))
    start_zone: Mapped[str | None] = mapped_column(String(10))
    end_zone: Mapped[str | None] = mapped_column(String(10))
    is_goal: Mapped[bool] = mapped_column(Boolean, default=False)
    is_loss: Mapped[bool] = mapped_column(Boolean, default=False)
    start_x: Mapped[float | None] = mapped_column(Float)
    start_y: Mapped[float | None] = mapped_column(Float)
    end_x: Mapped[float | None] = mapped_column(Float)
    end_y: Mapped[float | None] = mapped_column(Float)

    match: Mapped["Match"] = relationship(back_populates="events")


class XTResult(Base):
    __tablename__ = "xt_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"))
    match_id: Mapped[int | None] = mapped_column(ForeignKey("matches.id"), nullable=True)
    team: Mapped[str] = mapped_column(String(120))
    zone_name: Mapped[str] = mapped_column(String(10))
    xt_value: Mapped[float] = mapped_column(Float)
    direct_goal_pct: Mapped[float] = mapped_column(Float)
    loss_pct: Mapped[float] = mapped_column(Float)
    event_count: Mapped[int] = mapped_column(Integer)
    scope: Mapped[Scope] = mapped_column(Enum(Scope))

    competition: Mapped["Competition"] = relationship(back_populates="xt_results")
    match: Mapped["Match"] = relationship(back_populates="xt_results")
