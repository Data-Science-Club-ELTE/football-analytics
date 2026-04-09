"""Tests for the SQLAlchemy database models."""

from datetime import date, datetime

from database.models import (
    Competition,
    Event,
    EventType,
    Match,
    Scope,
    XTResult,
)


class TestCompetitionModel:
    def test_create_competition(self, db_session):
        comp = Competition(
            competition_name="Premier League",
            season_name="2023/2024",
            statsbomb_competition_id=2,
            statsbomb_season_id=90,
            created_at=datetime(2024, 6, 1),
        )
        db_session.add(comp)
        db_session.commit()

        fetched = db_session.get(Competition, comp.id)
        assert fetched is not None
        assert fetched.competition_name == "Premier League"
        assert fetched.season_name == "2023/2024"
        assert fetched.statsbomb_competition_id == 2


class TestMatchModel:
    def test_create_match(self, db_session):
        comp = Competition(
            competition_name="La Liga",
            season_name="2017/2018",
            statsbomb_competition_id=11,
            statsbomb_season_id=4,
        )
        db_session.add(comp)
        db_session.flush()

        match = Match(
            competition_id=comp.id,
            home_team="Barcelona",
            away_team="Real Madrid",
            match_date=date(2017, 12, 23),
            statsbomb_match_id=9924,
        )
        db_session.add(match)
        db_session.commit()

        fetched = db_session.get(Match, match.id)
        assert fetched is not None
        assert fetched.home_team == "Barcelona"
        assert fetched.competition_id == comp.id


class TestEventModel:
    def test_create_events(self, db_session):
        comp = Competition(
            competition_name="La Liga",
            season_name="2017/2018",
            statsbomb_competition_id=11,
            statsbomb_season_id=4,
        )
        db_session.add(comp)
        db_session.flush()

        match = Match(
            competition_id=comp.id,
            home_team="Barcelona",
            away_team="Real Madrid",
            match_date=date(2017, 12, 23),
            statsbomb_match_id=9924,
        )
        db_session.add(match)
        db_session.flush()

        evt = Event(
            match_id=match.id,
            event_type=EventType.PASS,
            team="Barcelona",
            start_zone="MC",
            end_zone="AC",
            is_goal=False,
            is_loss=False,
            start_x=50.0,
            start_y=40.0,
            end_x=90.0,
            end_y=40.0,
        )
        db_session.add(evt)
        db_session.commit()

        fetched = db_session.get(Event, evt.id)
        assert fetched is not None
        assert fetched.event_type == EventType.PASS
        assert fetched.start_zone == "MC"
        assert fetched.is_goal is False


class TestXTResultModel:
    def _make_comp_and_match(self, db_session):
        comp = Competition(
            competition_name="La Liga",
            season_name="2017/2018",
            statsbomb_competition_id=11,
            statsbomb_season_id=4,
        )
        db_session.add(comp)
        db_session.flush()

        match = Match(
            competition_id=comp.id,
            home_team="Barcelona",
            away_team="Real Madrid",
            match_date=date(2017, 12, 23),
            statsbomb_match_id=9924,
        )
        db_session.add(match)
        db_session.flush()
        return comp, match

    def test_xt_result_season_scope(self, db_session):
        comp, _ = self._make_comp_and_match(db_session)

        result = XTResult(
            competition_id=comp.id,
            match_id=None,
            team="Barcelona",
            zone_name="Box_C",
            xt_value=0.15,
            direct_goal_pct=0.08,
            loss_pct=0.25,
            event_count=120,
            scope=Scope.SEASON,
        )
        db_session.add(result)
        db_session.commit()

        fetched = db_session.get(XTResult, result.id)
        assert fetched.scope == Scope.SEASON
        assert fetched.match_id is None
        assert fetched.xt_value == 0.15

    def test_xt_result_match_scope(self, db_session):
        comp, match = self._make_comp_and_match(db_session)

        result = XTResult(
            competition_id=comp.id,
            match_id=match.id,
            team="Barcelona",
            zone_name="MC",
            xt_value=0.04,
            direct_goal_pct=0.0,
            loss_pct=0.15,
            event_count=45,
            scope=Scope.MATCH,
        )
        db_session.add(result)
        db_session.commit()

        fetched = db_session.get(XTResult, result.id)
        assert fetched.scope == Scope.MATCH
        assert fetched.match_id == match.id

    def test_competition_cascade(self, db_session):
        comp, match = self._make_comp_and_match(db_session)
        db_session.commit()

        # Access relationships
        fetched_comp = db_session.get(Competition, comp.id)
        assert len(fetched_comp.matches) == 1
        assert fetched_comp.matches[0].home_team == "Barcelona"

        fetched_match = db_session.get(Match, match.id)
        assert fetched_match.competition.competition_name == "La Liga"
