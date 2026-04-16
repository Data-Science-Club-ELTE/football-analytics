"""
run_season_xT.py — Calculate xT for Barcelona's full La Liga season
===================================================================
Uses the markov_xT module to load all Barcelona matches from StatsBomb,
build a season-wide transition matrix, and produce xT values.

Usage:
    python run_season_xT.py
"""

import os
from markov_xT import (
    load_match_events,
    load_season_events,
    build_transition_matrix,
    solve_xT,
    plot_xT,
    export_xT,
)

# Create output folder if it doesn't exist
os.makedirs("output", exist_ok=True)

# ============================================================
# 1. Full season xT for Barcelona
# ============================================================

print("=" * 60)
print("Loading ALL Barcelona matches — La Liga 2017/18")
print("=" * 60)

# competition_id=11 → La Liga
# season_id → check sb.competitions() for 2017/18
# Change season_id if needed based on your earlier exploration
COMPETITION_ID = 11
SEASON_ID = 1  # adjust this to match your 2017/18 season_id

season_events = load_season_events(
    competition_id=COMPETITION_ID,
    season_id=SEASON_ID,
    team="Barcelona"
)

print("\n" + "=" * 60)
print("Building transition matrix from full season")
print("=" * 60)

season_probs = build_transition_matrix(season_events)

print("\n" + "=" * 60)
print("Solving for Expected Threat")
print("=" * 60)

season_xT = solve_xT(season_probs)

# ============================================================
# 2. Visualize and export
# ============================================================

plot_xT(
    season_xT,
    title="Expected Threat (xT) — Barcelona Full Season La Liga 2017/18",
    save_path="output/xT_season_barcelona.png"
)

export_xT(
    season_xT,
    season_probs,
    save_path="output/xT_season_data.csv"
)

# ============================================================
# 3. Compare with single El Clasico match
# ============================================================

print("\n" + "=" * 60)
print("Comparing: Full Season vs El Clasico only")
print("=" * 60)

clasico_events = load_match_events(match_id=9924)
clasico_events = clasico_events[clasico_events["team"] == "Barcelona"]
clasico_probs = build_transition_matrix(clasico_events)
clasico_xT = solve_xT(clasico_probs)

# Side by side comparison
print(f"\n{'Zone':<8} {'Season xT':>12} {'El Clasico xT':>14} {'Difference':>12}")
print("-" * 50)
for zone in sorted(season_xT.keys(), key=lambda z: -season_xT[z]):
    s_val = season_xT[zone]
    c_val = clasico_xT[zone]
    diff = s_val - c_val
    print(f"{zone:<8} {s_val:>12.4f} {c_val:>14.4f} {diff:>+12.4f}")

print("\nDone! Check the output/ folder for plots and CSV.")