"""
Shared fixtures for the football-analytics test suite.
All tests use an in-memory SQLite database — no network calls.
"""

import os
import sys
from datetime import date, datetime
from pathlib import Path

# Force SQLite BEFORE any database module is imported, so the
# module-level create_engine() in database/database.py never
# tries to load psycopg2.
os.environ["DATABASE_URL"] = "sqlite://"

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Make markov_xT importable
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "expected-threat-markov")
)

from database.models import (
    Base,
    Competition,
    Event,
    EventType,
    Match,
    Scope,
    XTResult,
)


# ── In-memory SQLite DB ───────────────────────────────────────


@pytest.fixture()
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def db_session(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


# ── FastAPI TestClient with overridden DB ─────────────────────


@pytest.fixture()
def client(engine, db_session):
    from fastapi.testclient import TestClient

    from api.dependencies import db_dependency
    from api.main import app

    def _override_db():
        yield db_session

    app.dependency_overrides[db_dependency] = _override_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ── Sample processed-event DataFrame ─────────────────────────
# Mimics the output of _process_events(): columns used by
# build_transition_matrix are type, team, start_zone, end_zone,
# is_goal, is_loss  (x/y columns included for completeness).


@pytest.fixture()
def sample_events_df() -> pd.DataFrame:
    rows = [
        # Defensive -> Midfield transitions
        {"type": "Pass", "team": "TeamA", "start_zone": "DC", "end_zone": "MC", "is_goal": False, "is_loss": False, "x": 20, "y": 40, "end_x": 50, "end_y": 40},
        {"type": "Pass", "team": "TeamA", "start_zone": "DL", "end_zone": "ML", "is_goal": False, "is_loss": False, "x": 20, "y": 70, "end_x": 50, "end_y": 70},
        {"type": "Pass", "team": "TeamA", "start_zone": "DR", "end_zone": "MR", "is_goal": False, "is_loss": False, "x": 20, "y": 10, "end_x": 50, "end_y": 10},
        {"type": "Carry", "team": "TeamA", "start_zone": "DC", "end_zone": "MC", "is_goal": False, "is_loss": False, "x": 30, "y": 40, "end_x": 60, "end_y": 40},
        {"type": "Pass", "team": "TeamA", "start_zone": "DC", "end_zone": "ML", "is_goal": False, "is_loss": False, "x": 25, "y": 35, "end_x": 55, "end_y": 65},
        # Midfield -> Attack transitions
        {"type": "Pass", "team": "TeamA", "start_zone": "MC", "end_zone": "AC", "is_goal": False, "is_loss": False, "x": 60, "y": 40, "end_x": 90, "end_y": 40},
        {"type": "Pass", "team": "TeamA", "start_zone": "ML", "end_zone": "AL", "is_goal": False, "is_loss": False, "x": 60, "y": 70, "end_x": 90, "end_y": 70},
        {"type": "Pass", "team": "TeamA", "start_zone": "MR", "end_zone": "AR", "is_goal": False, "is_loss": False, "x": 60, "y": 10, "end_x": 90, "end_y": 10},
        {"type": "Carry", "team": "TeamA", "start_zone": "MC", "end_zone": "AC", "is_goal": False, "is_loss": False, "x": 70, "y": 40, "end_x": 95, "end_y": 40},
        {"type": "Pass", "team": "TeamA", "start_zone": "MC", "end_zone": "MR", "is_goal": False, "is_loss": False, "x": 55, "y": 40, "end_x": 55, "end_y": 10},
        # Attack -> Box transitions
        {"type": "Pass", "team": "TeamA", "start_zone": "AC", "end_zone": "Box_C", "is_goal": False, "is_loss": False, "x": 90, "y": 40, "end_x": 110, "end_y": 40},
        {"type": "Pass", "team": "TeamA", "start_zone": "AL", "end_zone": "Box_L", "is_goal": False, "is_loss": False, "x": 90, "y": 70, "end_x": 110, "end_y": 70},
        {"type": "Carry", "team": "TeamA", "start_zone": "AC", "end_zone": "Box_C", "is_goal": False, "is_loss": False, "x": 95, "y": 40, "end_x": 115, "end_y": 40},
        {"type": "Pass", "team": "TeamA", "start_zone": "AR", "end_zone": "Box_R", "is_goal": False, "is_loss": False, "x": 90, "y": 10, "end_x": 110, "end_y": 10},
        # Losses from midfield / attack
        {"type": "Pass", "team": "TeamA", "start_zone": "MC", "end_zone": None, "is_goal": False, "is_loss": True, "x": 60, "y": 40, "end_x": None, "end_y": None},
        {"type": "Pass", "team": "TeamA", "start_zone": "AL", "end_zone": None, "is_goal": False, "is_loss": True, "x": 90, "y": 70, "end_x": None, "end_y": None},
        {"type": "Pass", "team": "TeamA", "start_zone": "AC", "end_zone": None, "is_goal": False, "is_loss": True, "x": 95, "y": 40, "end_x": None, "end_y": None},
        {"type": "Pass", "team": "TeamA", "start_zone": "DL", "end_zone": None, "is_goal": False, "is_loss": True, "x": 20, "y": 70, "end_x": None, "end_y": None},
        # Goals from box
        {"type": "Shot", "team": "TeamA", "start_zone": "Box_C", "end_zone": None, "is_goal": True, "is_loss": False, "x": 110, "y": 40, "end_x": 120, "end_y": 40},
        {"type": "Shot", "team": "TeamA", "start_zone": "Box_C", "end_zone": None, "is_goal": True, "is_loss": False, "x": 112, "y": 38, "end_x": 120, "end_y": 36},
        {"type": "Shot", "team": "TeamA", "start_zone": "Box_L", "end_zone": None, "is_goal": True, "is_loss": False, "x": 108, "y": 60, "end_x": 120, "end_y": 38},
        # Shots that are not goals (losses)
        {"type": "Shot", "team": "TeamA", "start_zone": "Box_C", "end_zone": None, "is_goal": False, "is_loss": True, "x": 110, "y": 40, "end_x": 120, "end_y": 50},
        {"type": "Shot", "team": "TeamA", "start_zone": "Box_R", "end_zone": None, "is_goal": False, "is_loss": True, "x": 108, "y": 10, "end_x": 120, "end_y": 20},
        # More midfield transitions to fill the matrix
        {"type": "Pass", "team": "TeamA", "start_zone": "MC", "end_zone": "MC", "is_goal": False, "is_loss": False, "x": 50, "y": 40, "end_x": 70, "end_y": 40},
        {"type": "Pass", "team": "TeamA", "start_zone": "ML", "end_zone": "MC", "is_goal": False, "is_loss": False, "x": 50, "y": 70, "end_x": 60, "end_y": 40},
        {"type": "Carry", "team": "TeamA", "start_zone": "MR", "end_zone": "MC", "is_goal": False, "is_loss": False, "x": 50, "y": 10, "end_x": 60, "end_y": 40},
        {"type": "Pass", "team": "TeamA", "start_zone": "DC", "end_zone": "DC", "is_goal": False, "is_loss": False, "x": 20, "y": 40, "end_x": 30, "end_y": 40},
        # Back-passes
        {"type": "Pass", "team": "TeamA", "start_zone": "MC", "end_zone": "DC", "is_goal": False, "is_loss": False, "x": 50, "y": 40, "end_x": 30, "end_y": 40},
        {"type": "Pass", "team": "TeamA", "start_zone": "AC", "end_zone": "MC", "is_goal": False, "is_loss": False, "x": 90, "y": 40, "end_x": 60, "end_y": 40},
        # Lateral passes
        {"type": "Pass", "team": "TeamA", "start_zone": "Box_C", "end_zone": "Box_L", "is_goal": False, "is_loss": False, "x": 110, "y": 40, "end_x": 110, "end_y": 60},
    ]
    return pd.DataFrame(rows)


# ── DB seeding fixture ────────────────────────────────────────


@pytest.fixture()
def seeded_db(db_session, sample_events_df):
    """Seed a competition, match, and events into the test DB."""
    comp = Competition(
        id=1,
        competition_name="La Liga",
        season_name="2017/2018",
        statsbomb_competition_id=11,
        statsbomb_season_id=4,
        created_at=datetime(2024, 1, 1),
    )
    db_session.add(comp)

    match = Match(
        id=1,
        competition_id=1,
        home_team="Barcelona",
        away_team="Real Madrid",
        match_date=date(2017, 12, 23),
        statsbomb_match_id=9924,
        created_at=datetime(2024, 1, 1),
    )
    db_session.add(match)
    db_session.flush()

    for _, row in sample_events_df.iterrows():
        evt = Event(
            match_id=1,
            event_type=EventType(row["type"].upper()),
            team=row["team"],
            start_zone=row["start_zone"],
            end_zone=row["end_zone"] if pd.notna(row["end_zone"]) else None,
            is_goal=bool(row["is_goal"]),
            is_loss=bool(row["is_loss"]),
            start_x=row["x"],
            start_y=row["y"],
            end_x=row["end_x"] if pd.notna(row.get("end_x", None)) else None,
            end_y=row["end_y"] if pd.notna(row.get("end_y", None)) else None,
        )
        db_session.add(evt)

    db_session.commit()
    return db_session
