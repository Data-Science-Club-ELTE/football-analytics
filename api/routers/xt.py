import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

# Make markov_xT importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "expected-threat-markov"))

from markov_xT import (
    ZONES,
    ZONE_RECTS,
    build_transition_matrix,
    plot_xT,
    solve_xT,
)

from api.dependencies import db_dependency
from api.schemas import Envelope, XTResultOut, ZoneDefOut, ZoneXTOut
from database.models import Event, Match, Scope, XTResult

router = APIRouter(prefix="/api/v1/xt", tags=["xT"])


# ── helpers ─────────────────────────────────────────────────────


def _events_to_dataframe(events: list[Event]) -> pd.DataFrame:
    """Convert DB Event rows into the DataFrame format that
    build_transition_matrix() expects."""
    rows = []
    for e in events:
        rows.append({
            "type": e.event_type.value.capitalize(),  # PASS -> Pass
            "start_zone": e.start_zone,
            "end_zone": e.end_zone,
            "is_goal": e.is_goal,
            "is_loss": e.is_loss,
        })
    return pd.DataFrame(rows)


def _compute_and_store(
    db: Session,
    events: list[Event],
    competition_id: int,
    match_id: int | None,
    team: str,
    scope: Scope,
) -> list[ZoneXTOut]:
    """Run the Markov solve on events and persist to xt_results."""
    df = _events_to_dataframe(events)
    if df.empty:
        raise HTTPException(400, detail="No events found for this selection")

    probs = build_transition_matrix(df)
    xt = solve_xT(probs)
    row_totals = probs.attrs.get("row_totals")

    results = []
    for zone in ZONES:
        row = XTResult(
            competition_id=competition_id,
            match_id=match_id,
            team=team,
            zone_name=zone,
            xt_value=xt[zone],
            direct_goal_pct=float(probs.loc[zone, "Goal"]),
            loss_pct=float(probs.loc[zone, "Loss"]),
            event_count=int(row_totals[zone]) if row_totals is not None else 0,
            scope=scope,
        )
        db.add(row)
        results.append(ZoneXTOut.model_validate(row))

    db.commit()
    return results


def _get_match_or_404(db: Session, statsbomb_match_id: int) -> Match:
    match = db.execute(
        select(Match).where(Match.statsbomb_match_id == statsbomb_match_id)
    ).scalar_one_or_none()
    if not match:
        raise HTTPException(404, detail=f"Match with statsbomb_match_id={statsbomb_match_id} not found")
    return match


# ── GET /api/v1/xt/teams/{competition_id} ───────────────────────


@router.get("/teams/{competition_id}", response_model=Envelope)
def get_teams(competition_id: int, db: Session = Depends(db_dependency)):
    """Return list of teams that have seeded xT results for a competition."""
    rows = db.execute(
        select(XTResult.team).where(
            XTResult.competition_id == competition_id,
            XTResult.scope == Scope.SEASON,
        ).distinct()
    ).scalars().all()
    return Envelope(data=sorted(rows))


# ── GET /api/v1/xt/zones ────────────────────────────────────────


@router.get("/zones", response_model=Envelope)
def get_zones():
    """Return the 12 zone definitions with their pitch coordinates."""
    zones = [
        ZoneDefOut(code=code, x=rect[0], y=rect[1], width=rect[2], height=rect[3])
        for code, rect in ZONE_RECTS.items()
    ]
    return Envelope(data=zones)


# ── GET /api/v1/xt/season/{competition_id} ──────────────────────


@router.get("/season/{competition_id}", response_model=Envelope)
def get_season_xt(
    competition_id: int,
    team: str | None = Query(None, description="Filter by team name"),
    db: Session = Depends(db_dependency),
):
    """Return season-wide xT values from the xt_results table."""
    stmt = select(XTResult).where(
        XTResult.competition_id == competition_id,
        XTResult.scope == Scope.SEASON,
        XTResult.match_id.is_(None),
    )
    if team:
        stmt = stmt.where(XTResult.team == team)
    else:
        # No team specified — pick the team with the most events
        all_rows = db.execute(stmt).scalars().all()
        if not all_rows:
            raise HTTPException(404, detail="No season xT results found for this competition")

        # Group by team, pick the one with most total events
        by_team: dict[str, list] = {}
        for r in all_rows:
            by_team.setdefault(r.team, []).append(r)
        best_team = max(by_team, key=lambda t: sum(r.event_count for r in by_team[t]))
        rows = by_team[best_team]
        zones = [ZoneXTOut.model_validate(r) for r in rows]
        return Envelope(data=XTResultOut(scope="SEASON", zones=zones))

    rows = db.execute(stmt).scalars().all()
    if not rows:
        raise HTTPException(404, detail="No season xT results found for this competition")

    zones = [ZoneXTOut.model_validate(r) for r in rows]
    return Envelope(data=XTResultOut(scope="SEASON", zones=zones))


# ── GET /api/v1/xt/match/{statsbomb_match_id} ───────────────────


@router.get("/match/{statsbomb_match_id}", response_model=Envelope)
def get_match_xt(
    statsbomb_match_id: int,
    team: str | None = Query(None, description="Filter events by team name"),
    db: Session = Depends(db_dependency),
):
    """Return match-level xT. Computes on the fly from stored events,
    caches the result in xt_results for subsequent requests."""
    match = _get_match_or_404(db, statsbomb_match_id)

    # Check cache
    cache_stmt = select(XTResult).where(
        XTResult.match_id == match.id,
        XTResult.scope == Scope.MATCH,
    )
    if team:
        cache_stmt = cache_stmt.where(XTResult.team == team)
    cached = db.execute(cache_stmt).scalars().all()

    if cached:
        zones = [ZoneXTOut.model_validate(r) for r in cached]
        return Envelope(data=XTResultOut(scope="MATCH", zones=zones))

    # Compute from stored events
    stmt = select(Event).where(Event.match_id == match.id)
    if team:
        stmt = stmt.where(Event.team == team)
    events = db.execute(stmt).scalars().all()

    # Determine team name for storage
    team_name = team or (events[0].team if events else "Unknown")

    zones = _compute_and_store(
        db, events, match.competition_id, match.id, team_name, Scope.MATCH,
    )
    return Envelope(data=XTResultOut(scope="MATCH", zones=zones))


# ── GET /api/v1/xt/match/{statsbomb_match_id}/heatmap ───────────


@router.get("/match/{statsbomb_match_id}/heatmap")
def get_match_heatmap(
    statsbomb_match_id: int,
    team: str | None = Query(None),
    db: Session = Depends(db_dependency),
):
    """Generate and return a PNG heatmap for a match's xT values."""
    match = _get_match_or_404(db, statsbomb_match_id)

    # Get or compute xT values
    cache_stmt = select(XTResult).where(
        XTResult.match_id == match.id,
        XTResult.scope == Scope.MATCH,
    )
    if team:
        cache_stmt = cache_stmt.where(XTResult.team == team)
    cached = db.execute(cache_stmt).scalars().all()

    if cached:
        xt_dict = {r.zone_name: r.xt_value for r in cached}
    else:
        stmt = select(Event).where(Event.match_id == match.id)
        if team:
            stmt = stmt.where(Event.team == team)
        events = db.execute(stmt).scalars().all()

        df = _events_to_dataframe(events)
        if df.empty:
            raise HTTPException(400, detail="No events found")

        probs = build_transition_matrix(df)
        xt_dict = solve_xT(probs)

    title = f"xT — {match.home_team} vs {match.away_team}"
    if team:
        title += f" ({team})"
    fig, _ = plot_xT(xt_dict, title=title, show=False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    buf.seek(0)
    plt.close(fig)

    return StreamingResponse(buf, media_type="image/png")
