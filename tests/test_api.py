"""Tests for the FastAPI endpoints using TestClient."""

import pytest


class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestCompetitions:
    def test_list_competitions(self, client, seeded_db):
        resp = client.get("/api/v1/competitions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert len(body["data"]) == 1
        assert body["data"][0]["competition_name"] == "La Liga"

    def test_list_competitions_empty(self, client):
        resp = client.get("/api/v1/competitions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"] == []


class TestMatches:
    def test_list_matches(self, client, seeded_db):
        resp = client.get("/api/v1/matches", params={"competition_id": 1})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert len(body["data"]) == 1
        assert body["data"][0]["home_team"] == "Barcelona"


class TestXTZones:
    def test_xt_zones(self, client):
        resp = client.get("/api/v1/xt/zones")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert len(body["data"]) == 12
        codes = {z["code"] for z in body["data"]}
        assert "Box_C" in codes
        assert "MC" in codes


class TestXTSeason:
    def test_xt_season_not_found(self, client):
        resp = client.get("/api/v1/xt/season/999")
        assert resp.status_code == 404

    def test_xt_season(self, client, seeded_db):
        # First seed season-scoped xT results into the DB
        from database.models import Scope, XTResult

        for zone in [
            "DL", "DC", "DR", "ML", "MC", "MR",
            "AL", "AC", "AR", "Box_L", "Box_C", "Box_R",
        ]:
            seeded_db.add(XTResult(
                competition_id=1,
                match_id=None,
                team="TeamA",
                zone_name=zone,
                xt_value=0.05,
                direct_goal_pct=0.01,
                loss_pct=0.1,
                event_count=10,
                scope=Scope.SEASON,
            ))
        seeded_db.commit()

        resp = client.get("/api/v1/xt/season/1", params={"team": "TeamA"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["scope"] == "SEASON"
        assert len(body["data"]["zones"]) == 12


class TestXTMatch:
    def test_xt_match(self, client, seeded_db):
        resp = client.get("/api/v1/xt/match/9924")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["scope"] == "MATCH"
        assert len(body["data"]["zones"]) == 12
        # All xT values should be between 0 and 1
        for z in body["data"]["zones"]:
            assert 0.0 <= z["xt_value"] <= 1.0
