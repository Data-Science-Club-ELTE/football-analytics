"""
markov_xT.py — Expected Threat (xT) via Markov Chains
======================================================
Reusable module for the Data Science Club Football Analytics project.

Usage:
    from markov_xT import load_match_events, build_transition_matrix, solve_xT, plot_xT

    # Single match
    events = load_match_events(match_id=9924)
    probs = build_transition_matrix(events)
    xT = solve_xT(probs)
    plot_xT(xT, title="El Clasico xT", save_path="output/xT.png")

    # Full season
    events = load_season_events(competition_id=11, season_id=4, team="Barcelona")
    probs = build_transition_matrix(events)
    xT = solve_xT(probs)
    plot_xT(xT, title="Barcelona 2017/18 Season xT", save_path="output/xT_season.png")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Pitch
from statsbombpy import sb

# ============================================================
# CONFIGURATION — 12-zone system from Excel template
# ============================================================

ZONES = ["DL", "DC", "DR", "ML", "MC", "MR",
         "AL", "AC", "AR", "Box_L", "Box_C", "Box_R"]

ABSORBING = ["Goal", "Loss"]
ALL_STATES = ZONES + ABSORBING
VALID_ZONES = set(ZONES)
N_ZONES = len(ZONES)

# Zone rectangles for plotting (StatsBomb coordinates)
ZONE_RECTS = {
    "DL":    (0,   53.33, 40, 26.67),
    "DC":    (0,   26.67, 40, 26.67),
    "DR":    (0,   0,     40, 26.67),
    "ML":    (40,  53.33, 40, 26.67),
    "MC":    (40,  26.67, 40, 26.67),
    "MR":    (40,  0,     40, 26.67),
    "AL":    (80,  53.33, 22, 26.67),
    "AC":    (80,  26.67, 22, 26.67),
    "AR":    (80,  0,     22, 26.67),
    "Box_L": (102, 53.33, 18, 26.67),
    "Box_C": (102, 26.67, 18, 26.67),
    "Box_R": (102, 0,     18, 26.67),
}


# ============================================================
# FUNCTION 1: Map StatsBomb (x, y) to zone code
# ============================================================

def statsbomb_to_zone(x, y):
    """
    Convert StatsBomb coordinates to our 12-zone code.

    Parameters:
        x (float): 0-120, own goal line to opponent goal line
        y (float): 0-80, right touchline to left touchline

    Returns:
        str or None: zone code like 'MC', 'Box_C', etc.
    """
    if pd.isna(x) or pd.isna(y):
        return None

    if x <= 40:
        third = "D"
    elif x <= 80:
        third = "M"
    elif x <= 102:
        third = "A"
    else:
        third = "Box"

    if y <= 26.67:
        lane = "R"
    elif y <= 53.33:
        lane = "C"
    else:
        lane = "L"

    if third == "Box":
        return f"Box_{lane}"
    else:
        return f"{third}{lane}"
    
    # ============================================================
# FUNCTION 2: Process raw events into zone transitions
# ============================================================

def _process_events(events):
    """
    Internal function. Takes raw StatsBomb events, filters ball-moving
    actions, assigns zones, and classifies as transition/goal/loss.

    Parameters:
        events (DataFrame): raw StatsBomb events from sb.events()

    Returns:
        DataFrame: processed events with start_zone, end_zone, is_goal, is_loss
    """
    # Keep only ball-moving events
    move = events[events["type"].isin(["Pass", "Carry", "Shot"])].copy()

    # Extract start coordinates
    move["x"] = move["location"].apply(
        lambda loc: loc[0] if isinstance(loc, list) else None
    )
    move["y"] = move["location"].apply(
        lambda loc: loc[1] if isinstance(loc, list) else None
    )

    # Extract end coordinates — each event type stores them differently
    def get_end_xy(row):
        if row["type"] == "Pass" and isinstance(row.get("pass_end_location"), list):
            return row["pass_end_location"][:2]
        elif row["type"] == "Carry" and isinstance(row.get("carry_end_location"), list):
            return row["carry_end_location"][:2]
        elif row["type"] == "Shot" and isinstance(row.get("shot_end_location"), list):
            return row["shot_end_location"][:2]
        return [None, None]

    end_coords = move.apply(get_end_xy, axis=1)
    move["end_x"] = end_coords.apply(lambda loc: loc[0])
    move["end_y"] = end_coords.apply(lambda loc: loc[1])

    # Map coordinates to zones
    move["start_zone"] = move.apply(
        lambda r: statsbomb_to_zone(r["x"], r["y"]), axis=1
    )
    move["end_zone"] = move.apply(
        lambda r: statsbomb_to_zone(r["end_x"], r["end_y"]), axis=1
    )

    # Classify: goal, loss, or transition
    move["is_goal"] = (
        (move["type"] == "Shot") &
        (move["shot_outcome"] == "Goal")
    )
    move["is_loss"] = (
        ((move["type"] == "Pass") &
         (move["pass_outcome"].isin(["Incomplete", "Out", "Unknown", "Pass Offside"])))
        |
        ((move["type"] == "Shot") &
         (move["shot_outcome"] != "Goal"))
    )

    return move


# ============================================================
# FUNCTION 3: Load and process a single match
# ============================================================

def load_match_events(match_id):
    """
    Load and process events from a single StatsBomb match.

    Parameters:
        match_id (int): StatsBomb match ID (e.g. 9924 for El Clasico)

    Returns:
        DataFrame: processed events with zones and classifications
    """
    print(f"Loading match {match_id}...")
    events = sb.events(match_id=match_id)
    processed = _process_events(events)
    print(f"  {len(processed)} ball-moving events processed")
    return processed


# ============================================================
# FUNCTION 4: Load and process all matches for a team in a season
# ============================================================

def load_season_events(competition_id, season_id, team=None):
    """
    Load and process events from ALL matches in a season.
    Optionally filter by team.

    Parameters:
        competition_id (int): e.g. 11 for La Liga
        season_id (int): e.g. 4 for 2018/19, check sb.competitions()
        team (str or None): e.g. "Barcelona" — if None, loads all teams

    Returns:
        DataFrame: all processed events across the season
    """
    matches = sb.matches(competition_id=competition_id, season_id=season_id)

    if team:
        matches = matches[
            (matches["home_team"] == team) | (matches["away_team"] == team)
        ]
        print(f"Found {len(matches)} matches for {team}")
    else:
        print(f"Found {len(matches)} matches in season")

    all_events = []
    for i, (_, match) in enumerate(matches.iterrows()):
        match_id = match["match_id"]
        home = match["home_team"]
        away = match["away_team"]
        score = f"{match['home_score']}-{match['away_score']}"
        print(f"  [{i+1}/{len(matches)}] {home} vs {away} ({score})...")

        try:
            events = sb.events(match_id=match_id)
            processed = _process_events(events)

            # If filtering by team, keep only that team's events
            if team:
                processed = processed[processed["team"] == team]

            processed["match_id"] = match_id
            all_events.append(processed)
        except Exception as e:
            print(f"    Error loading match {match_id}: {e}")
            continue

    result = pd.concat(all_events, ignore_index=True)
    print(f"\nTotal: {len(result)} events across {len(all_events)} matches")
    return result
# ============================================================
# FUNCTION 5: Build the transition probability matrix
# ============================================================

def build_transition_matrix(events):
    """
    Count zone transitions and normalize to probabilities.

    Parameters:
        events (DataFrame): processed events from load_match_events()
                            or load_season_events()

    Returns:
        DataFrame: transition probability matrix (12 rows x 14 columns)
    """
    # Initialize counts with zeros
    counts = pd.DataFrame(0, index=ZONES, columns=ALL_STATES)

    for _, row in events.iterrows():
        sz = row["start_zone"]
        ez = row["end_zone"]

        if sz not in VALID_ZONES:
            continue

        if row["is_goal"]:
            counts.loc[sz, "Goal"] += 1
        elif row["is_loss"]:
            counts.loc[sz, "Loss"] += 1
        elif ez in VALID_ZONES:
            counts.loc[sz, ez] += 1

    # Normalize rows to probabilities
    row_totals = counts.sum(axis=1)
    probs = counts.div(row_totals, axis=0).fillna(0)

    # Store counts and totals as attributes for later use
    probs.attrs["counts"] = counts
    probs.attrs["row_totals"] = row_totals

    n_goals = int(counts["Goal"].sum())
    n_events = int(counts.values.sum())
    print(f"\nTransition matrix built:")
    print(f"  {n_events} total transitions")
    print(f"  {n_goals} goals")

    return probs


# ============================================================
# FUNCTION 6: Solve for Expected Threat
# ============================================================

def solve_xT(transition_probs):
    """
    Solve xT = (I - Q)^{-1} * g using the Markov chain fundamental matrix.

    Parameters:
        transition_probs (DataFrame): from build_transition_matrix()

    Returns:
        dict: {zone_code: xT_value} e.g. {"Box_C": 0.15, "MC": 0.04, ...}
    """
    Q = transition_probs[ZONES].values
    g = transition_probs["Goal"].values.reshape(-1, 1)

    I = np.eye(N_ZONES)

    try:
        N = np.linalg.inv(I - Q)
        xT = N @ g
    except np.linalg.LinAlgError:
        # Fallback: iterative approach
        print("Matrix singular — using iterative method")
        xT = np.zeros((N_ZONES, 1))
        for _ in range(200):
            xT = Q @ xT + g

    # Build result dictionary
    xT_dict = {zone: float(val) for zone, val in zip(ZONES, xT.flatten())}

    # Print summary sorted by xT
    print("\nExpected Threat per zone:")
    print(f"  {'Zone':<8} {'xT':>8}")
    print(f"  {'-'*18}")
    for zone, val in sorted(xT_dict.items(), key=lambda x: -x[1]):
        print(f"  {zone:<8} {val:>8.4f}")

    return xT_dict


# ============================================================
# FUNCTION 7: Plot xT on a football pitch
# ============================================================

def plot_xT(xT_dict, title="Expected Threat (xT)", save_path=None, show=True):
    """
    Draw xT values as a heatmap on a football pitch.

    Parameters:
        xT_dict (dict): {zone_code: xT_value} from solve_xT()
        title (str): plot title
        save_path (str or None): if provided, saves the plot to this path
        show (bool): if True, displays the plot

    Returns:
        fig, ax: matplotlib figure and axes objects
    """
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a2e',
                  line_color='#ffffff', line_zorder=2, linewidth=1)
    fig, ax = pitch.draw(figsize=(14, 9))

    cmap = LinearSegmentedColormap.from_list(
        "xT", ["#1a1a2e", "#c0392b", "#f39c12", "#f1c40f"]
    )

    max_xT = max(xT_dict.values()) if max(xT_dict.values()) > 0 else 1

    for zone, (x, y, w, h) in ZONE_RECTS.items():
        val = xT_dict.get(zone, 0)
        normalized = val / max_xT if max_xT > 0 else 0
        color = cmap(normalized)

        rect = plt.Rectangle((x, y), w, h, facecolor=color,
                              edgecolor="white", linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)

        text_color = "white" if normalized < 0.55 else "black"
        ax.text(x + w/2, y + h/2, f"{zone}\n{val:.4f}",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=text_color, zorder=3)

    ax.set_title(title, fontsize=15, fontweight="bold", color="white", pad=20)
    fig.patch.set_facecolor("#1a1a2e")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        print(f"\nPlot saved: {save_path}")

    if show:
        plt.show()

    return fig, ax


# ============================================================
# FUNCTION 8: Export xT results to CSV
# ============================================================

def export_xT(xT_dict, transition_probs, save_path="output/xT_results.csv"):
    """
    Save xT values and transition data to a CSV file.

    Parameters:
        xT_dict (dict): from solve_xT()
        transition_probs (DataFrame): from build_transition_matrix()
        save_path (str): where to save the CSV
    """
    row_totals = transition_probs.attrs.get("row_totals", pd.Series(0, index=ZONES))

    df = pd.DataFrame({
        "zone": ZONES,
        "xT": [xT_dict[z] for z in ZONES],
        "direct_goal_prob": transition_probs["Goal"].values,
        "loss_prob": transition_probs["Loss"].values,
        "events_from_zone": row_totals.values
    }).sort_values("xT", ascending=False)

    df.to_csv(save_path, index=False)
    print(f"Results exported: {save_path}")
    return df