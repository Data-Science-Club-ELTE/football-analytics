"""
Football Analytics - Expected Threat Dashboard
================================================
Streamlit frontend that consumes the FastAPI backend.

Usage:
    1. Start the API:   uvicorn api.main:app --reload
    2. Start this app:  streamlit run dashboard/app.py
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st

# -- Config --

API_BASE = st.sidebar.text_input("API URL", value="http://localhost:8000")

# Zone rectangles (same as markov_xT.ZONE_RECTS) for client-side heatmap
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


# -- API helpers --


def api_get(path: str, params: dict | None = None):
    """Call the API and return the envelope data, or None on error."""
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
        r.raise_for_status()
        body = r.json()
        return body.get("data", body)
    except requests.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE}. Is the server running?")
        return None
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.status_code} - {e.response.text}")
        return None


def api_get_bytes(path: str, params: dict | None = None) -> bytes | None:
    """Fetch raw bytes from the API (for PNG heatmaps)."""
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.content
    except requests.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE}. Is the server running?")
        return None
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.status_code} - {e.response.text}")
        return None


# -- Drawing --


def draw_heatmap(zones: list[dict], title: str):
    """Draw a pitch heatmap from zone xT data, matching the style in
    markov_xT.plot_xT() - dark background, warm colour ramp."""
    from mplsoccer import Pitch

    xt_dict = {z["zone_name"]: z["xt_value"] for z in zones}

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e",
                  line_color="#ffffff", line_zorder=2, linewidth=1)
    fig, ax = pitch.draw(figsize=(14, 9))

    cmap = LinearSegmentedColormap.from_list(
        "xT", ["#1a1a2e", "#c0392b", "#f39c12", "#f1c40f"]
    )
    max_xt = max(xt_dict.values()) if xt_dict and max(xt_dict.values()) > 0 else 1

    for zone, (x, y, w, h) in ZONE_RECTS.items():
        val = xt_dict.get(zone, 0)
        norm = val / max_xt if max_xt > 0 else 0
        color = cmap(norm)

        rect = plt.Rectangle((x, y), w, h, facecolor=color,
                              edgecolor="white", linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)

        text_color = "white" if norm < 0.55 else "black"
        ax.text(x + w / 2, y + h / 2, f"{zone}\n{val:.4f}",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=text_color, zorder=3)

    ax.set_title(title, fontsize=15, fontweight="bold", color="white", pad=20)
    fig.patch.set_facecolor("#1a1a2e")
    plt.tight_layout()
    return fig


def draw_bar_chart(zones: list[dict]):
    """Horizontal bar chart ranking zones by xT value."""
    df = pd.DataFrame(zones).sort_values("xt_value")
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    cmap = LinearSegmentedColormap.from_list(
        "xT", ["#c0392b", "#f39c12", "#f1c40f"]
    )
    max_val = df["xt_value"].max() if df["xt_value"].max() > 0 else 1
    colors = [cmap(v / max_val) for v in df["xt_value"]]

    ax.barh(df["zone_name"], df["xt_value"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("xT Value", color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#333")
    plt.tight_layout()
    return fig


# -- Page config --

st.set_page_config(
    page_title="Football Analytics - xT",
    page_icon="",
    layout="wide",
)

st.markdown(
    "<h1 style='text-align:center; color:#f1c40f;'>"
    "Football Analytics - Expected Threat"
    "</h1>",
    unsafe_allow_html=True,
)

# -- Sidebar --

st.sidebar.header("Navigation")

competitions = api_get("/api/v1/competitions")
if not competitions:
    st.stop()

comp_labels = {
    c["id"]: f"{c['competition_name']} {c['season_name']}" for c in competitions
}
selected_comp_id = st.sidebar.selectbox(
    "Competition",
    options=list(comp_labels.keys()),
    format_func=lambda cid: comp_labels[cid],
)

# Team selector - fetch teams that have xT data for this competition
teams = api_get(f"/api/v1/xt/teams/{selected_comp_id}")
if not teams:
    st.warning("No teams seeded for this competition yet. Run the seed script first.")
    st.stop()

selected_team = st.sidebar.selectbox("Team", options=teams)

view = st.sidebar.radio("View", ["Season View", "Match View"])

# -- Season View --

if view == "Season View":
    data = api_get(
        f"/api/v1/xt/season/{selected_comp_id}",
        params={"team": selected_team},
    )
    if not data:
        st.warning(f"No season xT results for {selected_team}. Seed this team first.")
        st.stop()

    zones = data["zones"]
    df = pd.DataFrame(zones).sort_values("xt_value", ascending=False)

    # Summary cards
    total_events = df["event_count"].sum()
    total_goals = int((df["direct_goal_pct"] * df["event_count"]).sum())
    top_zone = df.iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Events", f"{total_events:,}")
    c2.metric("Est. Goals", f"{total_goals}")
    c3.metric("Highest Threat Zone", f"{top_zone['zone_name']}  ({top_zone['xt_value']:.4f})")

    # Heatmap and bar chart side by side
    col_heat, col_bar = st.columns([3, 2])
    with col_heat:
        st.subheader("xT Heatmap")
        title = f"{selected_team} - {comp_labels[selected_comp_id]}"
        fig = draw_heatmap(zones, title)
        st.pyplot(fig)
        plt.close(fig)

    with col_bar:
        st.subheader("Zone Ranking")
        fig = draw_bar_chart(zones)
        st.pyplot(fig)
        plt.close(fig)

    # Detailed table
    st.subheader("Zone Details")
    display_df = df.rename(columns={
        "zone_name": "Zone",
        "xt_value": "xT",
        "direct_goal_pct": "Direct Goal %",
        "loss_pct": "Loss %",
        "event_count": "Events",
    })
    display_df["Direct Goal %"] = (display_df["Direct Goal %"] * 100).round(2)
    display_df["Loss %"] = (display_df["Loss %"] * 100).round(2)
    display_df["xT"] = display_df["xT"].round(4)
    # Drop the team column from display - it's already shown in the sidebar
    if "team" in display_df.columns:
        display_df = display_df.drop(columns=["team"])
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

# -- Match View --

elif view == "Match View":
    matches = api_get("/api/v1/matches", params={"competition_id": selected_comp_id})
    if not matches:
        st.warning("No matches found for this competition.")
        st.stop()

    match_labels = {
        m["statsbomb_match_id"]: f"{m['home_team']} vs {m['away_team']}  ({m['match_date']})"
        for m in matches
    }
    selected_sb_id = st.sidebar.selectbox(
        "Match",
        options=list(match_labels.keys()),
        format_func=lambda mid: match_labels[mid],
    )

    # Find the internal match id for the detail endpoint
    selected_match = next(m for m in matches if m["statsbomb_match_id"] == selected_sb_id)

    # Match details
    detail = api_get(f"/api/v1/matches/{selected_match['id']}")
    if detail:
        st.subheader(f"{detail['home_team']} vs {detail['away_team']}")
        st.caption(f"Date: {detail['match_date']}  |  Team: {selected_team}")

        if "events" in detail:
            ev = detail["events"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Passes", ev["passes"])
            c2.metric("Carries", ev["carries"])
            c3.metric("Shots", ev["shots"])
            c4.metric("Goals", ev["goals"])

    # Match xT heatmap (PNG from API, filtered by team)
    st.subheader("Match xT Heatmap")
    heatmap_bytes = api_get_bytes(
        f"/api/v1/xt/match/{selected_sb_id}/heatmap",
        params={"team": selected_team},
    )
    if heatmap_bytes:
        st.image(heatmap_bytes, use_container_width=True)

    # Match xT table
    match_xt = api_get(
        f"/api/v1/xt/match/{selected_sb_id}",
        params={"team": selected_team},
    )
    if match_xt:
        match_zones = match_xt["zones"]
        match_df = pd.DataFrame(match_zones).sort_values("xt_value", ascending=False)

        # Comparison with season
        season_data = api_get(
            f"/api/v1/xt/season/{selected_comp_id}",
            params={"team": selected_team},
        )
        if season_data:
            season_zones = {z["zone_name"]: z["xt_value"] for z in season_data["zones"]}
            match_df["season_xt"] = match_df["zone_name"].map(season_zones)
            match_df["diff"] = match_df["xt_value"] - match_df["season_xt"]

        st.subheader(f"Zone xT Values - {selected_team} Match vs Season")
        display_df = match_df.rename(columns={
            "zone_name": "Zone",
            "xt_value": "Match xT",
            "direct_goal_pct": "Direct Goal %",
            "loss_pct": "Loss %",
            "event_count": "Events",
            "season_xt": "Season xT",
            "diff": "Diff",
        })
        for col in ["Direct Goal %", "Loss %"]:
            if col in display_df.columns:
                display_df[col] = (display_df[col] * 100).round(2)
        for col in ["Match xT", "Season xT", "Diff"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(4)
        if "team" in display_df.columns:
            display_df = display_df.drop(columns=["team"])
        st.dataframe(display_df.reset_index(drop=True), use_container_width=True)
