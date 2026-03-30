"""Tests for the core Markov chain xT logic."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "expected-threat-markov")
)

from markov_xT import (
    ZONES,
    build_transition_matrix,
    solve_xT,
    statsbomb_to_zone,
)


# ── Zone mapping ─────────────────────────────────────────────


class TestZoneMapping:
    def test_zone_mapping(self):
        assert statsbomb_to_zone(110, 40) == "Box_C"
        assert statsbomb_to_zone(50, 40) == "MC"
        assert statsbomb_to_zone(20, 40) == "DC"
        assert statsbomb_to_zone(90, 70) == "AL"
        assert statsbomb_to_zone(50, 10) == "MR"
        assert statsbomb_to_zone(115, 10) == "Box_R"
        assert statsbomb_to_zone(115, 70) == "Box_L"

    def test_zone_mapping_boundary(self):
        # x boundaries: 40, 80, 102
        assert statsbomb_to_zone(40, 40) == "DC"   # x=40 -> D
        assert statsbomb_to_zone(40.01, 40) == "MC"  # just over -> M
        assert statsbomb_to_zone(80, 40) == "MC"   # x=80 -> M
        assert statsbomb_to_zone(80.01, 40) == "AC"  # just over -> A
        assert statsbomb_to_zone(102, 40) == "AC"  # x=102 -> A
        assert statsbomb_to_zone(102.01, 40) == "Box_C"  # just over -> Box

        # y boundaries: 26.67, 53.33
        assert statsbomb_to_zone(50, 26.67) == "MR"  # y=26.67 -> R
        assert statsbomb_to_zone(50, 26.68) == "MC"  # just over -> C
        assert statsbomb_to_zone(50, 53.33) == "MC"  # y=53.33 -> C
        assert statsbomb_to_zone(50, 53.34) == "ML"  # just over -> L

    def test_zone_mapping_nan(self):
        assert statsbomb_to_zone(float("nan"), 40) is None
        assert statsbomb_to_zone(50, float("nan")) is None
        assert statsbomb_to_zone(float("nan"), float("nan")) is None
        assert statsbomb_to_zone(None, 40) is None
        assert statsbomb_to_zone(50, None) is None


# ── Transition matrix ────────────────────────────────────────


class TestTransitionMatrix:
    def test_transition_matrix_shape(self, sample_events_df):
        probs = build_transition_matrix(sample_events_df)
        assert probs.shape == (12, 14)  # 12 zones x (12 zones + Goal + Loss)

    def test_transition_matrix_rows_sum_to_one(self, sample_events_df):
        probs = build_transition_matrix(sample_events_df)
        row_sums = probs.sum(axis=1)
        for zone in ZONES:
            total = row_sums[zone]
            # Rows with events should sum to ~1; rows with no events sum to 0
            if total > 0:
                assert math.isclose(total, 1.0, abs_tol=1e-9), (
                    f"Row {zone} sums to {total}"
                )


# ── solve_xT ─────────────────────────────────────────────────


class TestSolveXT:
    def test_solve_xt_values_between_zero_and_one(self, sample_events_df):
        probs = build_transition_matrix(sample_events_df)
        xt = solve_xT(probs)
        for zone, val in xt.items():
            assert 0.0 <= val <= 1.0, f"{zone} xT={val} out of [0,1]"

    def test_solve_xt_box_highest(self, sample_events_df):
        probs = build_transition_matrix(sample_events_df)
        xt = solve_xT(probs)
        top_zones = sorted(xt, key=xt.get, reverse=True)[:3]
        assert "Box_C" in top_zones, (
            f"Box_C should be among top 3 but top 3 are {top_zones}"
        )

    def test_all_zones_present(self, sample_events_df):
        probs = build_transition_matrix(sample_events_df)
        xt = solve_xT(probs)
        assert set(xt.keys()) == set(ZONES)
        assert len(xt) == 12

    def test_singular_matrix_fallback(self):
        """Sparse data that can make (I-Q) singular; iterative fallback
        should still produce valid results."""
        rows = [
            {"type": "Pass", "team": "A", "start_zone": "MC", "end_zone": "MC",
             "is_goal": False, "is_loss": False, "x": 50, "y": 40,
             "end_x": 60, "end_y": 40},
            {"type": "Shot", "team": "A", "start_zone": "Box_C", "end_zone": None,
             "is_goal": True, "is_loss": False, "x": 110, "y": 40,
             "end_x": 120, "end_y": 40},
        ]
        df = pd.DataFrame(rows)
        probs = build_transition_matrix(df)
        xt = solve_xT(probs)

        # Should still return 12 zones with valid values
        assert len(xt) == 12
        for val in xt.values():
            assert 0.0 <= val <= 1.0 or math.isclose(val, 0.0, abs_tol=1e-9)

    def test_empty_events(self):
        df = pd.DataFrame(
            columns=["type", "team", "start_zone", "end_zone",
                      "is_goal", "is_loss", "x", "y", "end_x", "end_y"]
        )
        probs = build_transition_matrix(df)
        xt = solve_xT(probs)
        assert len(xt) == 12
        for val in xt.values():
            assert val == 0.0
