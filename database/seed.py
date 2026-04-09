"""
seed.py - CLI tool for seeding the football analytics database.

Usage:
    python -m database.seed --list-competitions
    python -m database.seed --list-seasons --competition-id 16
    python -m database.seed --list-teams --competition-id 11 --season-id 1
    python -m database.seed --list-matches --competition-id 11 --season-id 1 --team "Barcelona"

    python -m database.seed --competition-id 11 --season-id 1 --team "Barcelona"
    python -m database.seed --competition-id 11 --season-id 1   (interactive team select)

    python -m database.seed --match-id 303596
    python -m database.seed --match-id 303596 --force
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, select
from statsbombpy import sb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "expected-threat-markov"))

from markov_xT import (
    _process_events,
    build_transition_matrix,
    solve_xT,
    ZONES,
)

from database.database import engine, SessionLocal
from database.models import (
    Base,
    Competition,
    Event,
    EventType,
    Match,
    Scope,
    XTResult,
)

# ── ANSI colors ─────────────────────────────────────────────────

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def ok(msg):
    print(f"{GREEN}  [OK] {msg}{RESET}")


def skip(msg):
    print(f"{YELLOW}  [SKIP] {msg}{RESET}")


def err(msg):
    print(f"{RED}  [ERR] {msg}{RESET}")


def head(msg):
    print(f"\n{BOLD}{msg}{RESET}")


EVENT_TYPE_MAP = {
    "Pass": EventType.PASS,
    "Carry": EventType.CARRY,
    "Shot": EventType.SHOT,
}


# ── Validation helpers ──────────────────────────────────────────


def fetch_competitions():
    return sb.competitions()


def validate_competition(comps, competition_id):
    rows = comps[comps["competition_id"] == competition_id]
    if rows.empty:
        err(f"Competition ID {competition_id} not found in StatsBomb data.")
        print(f"    Run {DIM}python -m database.seed --list-competitions{RESET} to see options.")
        return None
    return rows


def validate_season(comps, competition_id, season_id):
    row = comps[
        (comps["competition_id"] == competition_id) &
        (comps["season_id"] == season_id)
    ]
    if row.empty:
        comp_name = comps[comps["competition_id"] == competition_id].iloc[0]["competition_name"]
        err(f"Season ID {season_id} not found for {comp_name}.")
        print(f"    Run {DIM}python -m database.seed --list-seasons "
              f"--competition-id {competition_id}{RESET} to see options.")
        return None
    return row.iloc[0]


def validate_team(matches_df, team, comp_name, season_name):
    team_matches = matches_df[
        (matches_df["home_team"] == team) | (matches_df["away_team"] == team)
    ]
    if team_matches.empty:
        available = sorted(set(matches_df["home_team"]) | set(matches_df["away_team"]))
        err(f"Team '{team}' not found in {comp_name} {season_name}.")
        print(f"    Available teams: {', '.join(available)}")
        return None
    return team_matches


# ── DB helpers ──────────────────────────────────────────────────


def get_or_create_competition(session, competition_id, season_id, comp_name, season_name):
    competition = session.execute(
        select(Competition).where(
            Competition.statsbomb_competition_id == competition_id,
            Competition.statsbomb_season_id == season_id,
        )
    ).scalar_one_or_none()

    if competition:
        ok(f"Using existing competition: {comp_name} {season_name} (id={competition.id})")
    else:
        competition = Competition(
            competition_name=comp_name,
            season_name=season_name,
            statsbomb_competition_id=competition_id,
            statsbomb_season_id=season_id,
        )
        session.add(competition)
        session.flush()
        ok(f"Created competition: {comp_name} {season_name} (id={competition.id})")

    return competition


def is_team_season_seeded(session, competition_id, team):
    return session.execute(
        select(XTResult.id).where(
            XTResult.competition_id == competition_id,
            XTResult.team == team,
            XTResult.scope == Scope.SEASON,
        ).limit(1)
    ).first() is not None


def is_match_team_seeded(session, match_db_id, team):
    return session.execute(
        select(Event.id).where(
            Event.match_id == match_db_id,
            Event.team == team,
        ).limit(1)
    ).first() is not None


def delete_team_season_data(session, competition_id, team):
    """Remove all events and xT results for a team in a competition."""
    # Find match IDs for this competition
    match_ids = [
        mid for (mid,) in session.execute(
            select(Match.id).where(Match.competition_id == competition_id)
        ).all()
    ]
    if match_ids:
        session.execute(
            delete(Event).where(Event.match_id.in_(match_ids), Event.team == team)
        )
    session.execute(
        delete(XTResult).where(
            XTResult.competition_id == competition_id,
            XTResult.team == team,
        )
    )
    session.flush()


def delete_match_data(session, match_db_id):
    """Remove all events and xT results for a match."""
    session.execute(delete(Event).where(Event.match_id == match_db_id))
    session.execute(delete(XTResult).where(XTResult.match_id == match_db_id))
    session.flush()


def get_or_create_match(session, competition_id, sb_match_id, home, away, match_date_val):
    db_match = session.execute(
        select(Match).where(Match.statsbomb_match_id == sb_match_id)
    ).scalar_one_or_none()

    if not db_match:
        db_match = Match(
            competition_id=competition_id,
            home_team=home,
            away_team=away,
            match_date=match_date_val,
            statsbomb_match_id=sb_match_id,
        )
        session.add(db_match)
        session.flush()

    return db_match


def insert_team_events(session, db_match_id, team, team_events_df):
    db_events = []
    for _, evt in team_events_df.iterrows():
        db_events.append(Event(
            match_id=db_match_id,
            event_type=EVENT_TYPE_MAP[evt["type"]],
            team=team,
            start_zone=evt.get("start_zone"),
            end_zone=evt.get("end_zone"),
            is_goal=bool(evt["is_goal"]),
            is_loss=bool(evt["is_loss"]),
            start_x=evt.get("x"),
            start_y=evt.get("y"),
            end_x=evt.get("end_x"),
            end_y=evt.get("end_y"),
        ))
    session.add_all(db_events)
    return len(db_events)


def store_xt_results(session, competition_id, match_id, team, scope, events_df):
    probs = build_transition_matrix(events_df)
    xt = solve_xT(probs)
    row_totals = probs.attrs.get("row_totals")

    for zone in ZONES:
        session.add(XTResult(
            competition_id=competition_id,
            match_id=match_id,
            team=team,
            zone_name=zone,
            xt_value=xt[zone],
            direct_goal_pct=float(probs.loc[zone, "Goal"]),
            loss_pct=float(probs.loc[zone, "Loss"]),
            event_count=int(row_totals[zone]) if row_totals is not None else 0,
            scope=scope,
        ))
    return xt


# ── List commands ───────────────────────────────────────────────


def cmd_list_competitions():
    head("Available StatsBomb Competitions")
    comps = fetch_competitions().sort_values(["competition_name", "season_name"])
    print(f"\n  {'ID':>4}  {'Season ID':>9}  {'Competition':<30}  {'Season':<15}  Country")
    print(f"  {'-' * 83}")
    for _, row in comps.iterrows():
        print(f"  {row['competition_id']:>4}  {row['season_id']:>9}  "
              f"{row['competition_name']:<30}  {row['season_name']:<15}  "
              f"{row.get('country_name', '')}")
    print(f"\n  {len(comps)} competitions available.")


def cmd_list_seasons(competition_id):
    comps = fetch_competitions()
    rows = validate_competition(comps, competition_id)
    if rows is None:
        return

    comp_name = rows.iloc[0]["competition_name"]
    head(f"Available Seasons for {comp_name} (ID: {competition_id})")
    rows = rows.sort_values("season_name")
    print(f"\n  {'Season ID':>9}  Season")
    print(f"  {'-' * 30}")
    for _, row in rows.iterrows():
        print(f"  {row['season_id']:>9}  {row['season_name']}")
    print(f"\n  {len(rows)} seasons available.")


def cmd_list_teams(competition_id, season_id):
    comps = fetch_competitions()
    comp_info = validate_season(comps, competition_id, season_id)
    if comp_info is None:
        return

    comp_name = comp_info["competition_name"]
    season_name = comp_info["season_name"]
    matches_df = sb.matches(competition_id=competition_id, season_id=season_id)
    teams = sorted(set(matches_df["home_team"]) | set(matches_df["away_team"]))

    head(f"Teams in {comp_name} {season_name}")
    for i, t in enumerate(teams, 1):
        count = len(matches_df[(matches_df["home_team"] == t) | (matches_df["away_team"] == t)])
        print(f"  {i:>3}. {t} ({count} matches)")
    print(f"\n  {len(teams)} teams.")


def cmd_list_matches(competition_id, season_id, team):
    comps = fetch_competitions()
    comp_info = validate_season(comps, competition_id, season_id)
    if comp_info is None:
        return

    comp_name = comp_info["competition_name"]
    season_name = comp_info["season_name"]
    matches_df = sb.matches(competition_id=competition_id, season_id=season_id)
    team_matches = validate_team(matches_df, team, comp_name, season_name)
    if team_matches is None:
        return

    head(f"{team} matches in {comp_name} {season_name}")
    for _, row in team_matches.sort_values("match_date").iterrows():
        score = f"{row['home_score']}-{row['away_score']}"
        print(f"  {row['match_id']:>8}  {row['match_date']}  "
              f"{row['home_team']} {score} {row['away_team']}")
    print(f"\n  {len(team_matches)} matches.")


# ── Seed commands ───────────────────────────────────────────────


def cmd_seed_season(competition_id, season_id, team, force):
    comps = fetch_competitions()
    comp_info = validate_season(comps, competition_id, season_id)
    if comp_info is None:
        return

    comp_name = comp_info["competition_name"]
    season_name = comp_info["season_name"]
    matches_df = sb.matches(competition_id=competition_id, season_id=season_id)

    # Interactive team selection if no team specified
    if team is None:
        teams = sorted(set(matches_df["home_team"]) | set(matches_df["away_team"]))
        head(f"Available teams in {comp_name} {season_name}:")
        for i, t in enumerate(teams, 1):
            print(f"  {i:>3}. {t}")
        print()
        try:
            choice = input("  Select team (number): ").strip()
            idx = int(choice) - 1
            if not (0 <= idx < len(teams)):
                err("Invalid selection.")
                return
            team = teams[idx]
        except (ValueError, EOFError, KeyboardInterrupt):
            print()
            err("Cancelled.")
            return

    team_matches = validate_team(matches_df, team, comp_name, season_name)
    if team_matches is None:
        return

    head(f"Seeding {team} - {comp_name} {season_name}")

    Base.metadata.create_all(engine)
    session = SessionLocal()

    try:
        competition = get_or_create_competition(
            session, competition_id, season_id, comp_name, season_name
        )

        if is_team_season_seeded(session, competition.id, team):
            if force:
                skip(f"Force mode - deleting existing {team} data...")
                delete_team_season_data(session, competition.id, team)
            else:
                skip(f"Already seeded: {team} in {comp_name} {season_name}.")
                return

        all_processed = []
        total = len(team_matches)

        for i, (_, row) in enumerate(team_matches.iterrows()):
            sb_match_id = int(row["match_id"])
            home = row["home_team"]
            away = row["away_team"]
            match_date_val = row["match_date"]
            if isinstance(match_date_val, str):
                match_date_val = date.fromisoformat(match_date_val)

            print(f"  [{i+1:>{len(str(total))}}/{total}] {home} vs {away} "
                  f"{DIM}({sb_match_id}){RESET}")

            db_match = get_or_create_match(
                session, competition.id, sb_match_id, home, away, match_date_val
            )

            raw_events = sb.events(match_id=sb_match_id)
            processed = _process_events(raw_events)
            team_events = processed[processed["team"] == team]
            all_processed.append(team_events)

            count = insert_team_events(session, db_match.id, team, team_events)
            print(f"         {DIM}{count} events{RESET}")

        season_events = pd.concat(all_processed, ignore_index=True)
        print()
        store_xt_results(
            session, competition.id, None, team, Scope.SEASON, season_events
        )

        session.commit()
        ok(f"Done - {total} matches, {len(season_events)} events, "
           f"12 xT zones stored for {team}.")

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def cmd_seed_match(sb_match_id, force):
    head(f"Seeding match {sb_match_id}")

    # Validate match exists in StatsBomb
    try:
        raw_events = sb.events(match_id=sb_match_id)
    except Exception:
        err(f"Match ID {sb_match_id} not found in StatsBomb data.")
        return

    if raw_events.empty:
        err(f"No events returned for match {sb_match_id}.")
        return

    teams = sorted(raw_events["team"].dropna().unique())
    ok(f"Found match with teams: {', '.join(teams)}")

    processed = _process_events(raw_events)

    # Figure out competition from StatsBomb matches data
    comps = fetch_competitions()
    # Try to find the competition by scanning all of them (StatsBomb doesn't
    # provide a direct match→competition lookup)
    comp_id_found = None
    season_id_found = None
    match_meta = None
    for _, comp_row in comps.iterrows():
        try:
            matches_df = sb.matches(
                competition_id=int(comp_row["competition_id"]),
                season_id=int(comp_row["season_id"]),
            )
            match_row = matches_df[matches_df["match_id"] == sb_match_id]
            if not match_row.empty:
                comp_id_found = int(comp_row["competition_id"])
                season_id_found = int(comp_row["season_id"])
                match_meta = match_row.iloc[0]
                break
        except Exception:
            continue

    if comp_id_found is None:
        err("Could not determine competition for this match.")
        return

    comp_info = comps[
        (comps["competition_id"] == comp_id_found) &
        (comps["season_id"] == season_id_found)
    ].iloc[0]
    comp_name = comp_info["competition_name"]
    season_name = comp_info["season_name"]
    home = match_meta["home_team"]
    away = match_meta["away_team"]
    match_date_val = match_meta["match_date"]
    if isinstance(match_date_val, str):
        match_date_val = date.fromisoformat(match_date_val)

    ok(f"{home} vs {away} - {comp_name} {season_name}")

    Base.metadata.create_all(engine)
    session = SessionLocal()

    try:
        competition = get_or_create_competition(
            session, comp_id_found, season_id_found, comp_name, season_name
        )

        db_match = get_or_create_match(
            session, competition.id, sb_match_id, home, away, match_date_val
        )

        for team in teams:
            if is_match_team_seeded(session, db_match.id, team):
                if force:
                    skip(f"Force mode - re-seeding {team}...")
                    delete_match_data(session, db_match.id)
                else:
                    skip(f"{team} already seeded for this match.")
                    continue

            team_events = processed[processed["team"] == team]
            count = insert_team_events(session, db_match.id, team, team_events)

            xt = store_xt_results(
                session, competition.id, db_match.id, team, Scope.MATCH, team_events
            )
            top_zone = max(xt, key=xt.get)
            ok(f"{team}: {count} events, top zone {top_zone} "
               f"(xT={xt[top_zone]:.4f})")

        session.commit()
        ok("Match seeding complete.")

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── CLI ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Seed the football analytics database with StatsBomb data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python -m database.seed --list-competitions
  python -m database.seed --list-teams --competition-id 11 --season-id 1
  python -m database.seed --competition-id 11 --season-id 1 --team "Barcelona"
  python -m database.seed --competition-id 11 --season-id 1         (interactive)
  python -m database.seed --match-id 303596
  python -m database.seed --match-id 303596 --force
        """,
    )

    # List modes
    list_group = parser.add_argument_group("list modes")
    list_group.add_argument(
        "--list-competitions", action="store_true",
        help="List all StatsBomb competitions",
    )
    list_group.add_argument(
        "--list-seasons", action="store_true",
        help="List seasons for a competition (requires --competition-id)",
    )
    list_group.add_argument(
        "--list-teams", action="store_true",
        help="List teams in a competition/season (requires --competition-id --season-id)",
    )
    list_group.add_argument(
        "--list-matches", action="store_true",
        help="List matches for a team (requires --competition-id --season-id --team)",
    )

    # Seed parameters
    seed_group = parser.add_argument_group("seed parameters")
    seed_group.add_argument("--competition-id", type=int, help="StatsBomb competition ID")
    seed_group.add_argument("--season-id", type=int, help="StatsBomb season ID")
    seed_group.add_argument("--team", type=str, help="Team name")
    seed_group.add_argument("--match-id", type=int, help="StatsBomb match ID (single match mode)")
    seed_group.add_argument(
        "--force", action="store_true",
        help="Re-seed even if data already exists (deletes old data first)",
    )

    args = parser.parse_args()

    # ── Dispatch ────────────────────────────────────────────────

    if args.list_competitions:
        cmd_list_competitions()

    elif args.list_seasons:
        if not args.competition_id:
            err("--list-seasons requires --competition-id")
            return
        cmd_list_seasons(args.competition_id)

    elif args.list_teams:
        if not args.competition_id or not args.season_id:
            err("--list-teams requires --competition-id and --season-id")
            return
        cmd_list_teams(args.competition_id, args.season_id)

    elif args.list_matches:
        if not args.competition_id or not args.season_id or not args.team:
            err("--list-matches requires --competition-id, --season-id, and --team")
            return
        cmd_list_matches(args.competition_id, args.season_id, args.team)

    elif args.match_id:
        cmd_seed_match(args.match_id, args.force)

    elif args.competition_id and args.season_id:
        cmd_seed_season(args.competition_id, args.season_id, args.team, args.force)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
