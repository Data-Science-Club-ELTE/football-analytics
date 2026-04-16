import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('expected-threat-markov')
from markov_xT import ZONE_RECTS, ZONES, plot_xT, load_match_events, build_transition_matrix, solve_xT, export_xT, load_season_events

st.header("⚽ Expected Threat (xT) Analysis")

# Options
analysis_type = st.radio("Choose Analysis Type", ["Load from CSV", "Compute from Data"])

if analysis_type == "Load from CSV":
    # Load xT data
    try:
        xt_data = pd.read_csv("expected-threat-markov/output/xT_season_data.csv")
        st.write("xT Data Preview:")
        st.dataframe(xt_data.head())

        # Create xT dict
        xT_dict = dict(zip(xt_data['zone'], xt_data['xT']))

        # Use the plot_xT function
        fig, ax = plot_xT(xT_dict, title="Expected Threat (xT) Heatmap", save_path=None, show=False)
        st.pyplot(fig)

        # Additional stats
        st.subheader("Zone Statistics")
        st.dataframe(xt_data[['zone', 'xT', 'direct_goal_prob', 'loss_prob', 'events_from_zone']])

    except FileNotFoundError:
        st.error("xT data file not found. Please ensure the expected-threat-markov/output/xT_season_data.csv exists.")
    except Exception as e:
        st.error(f"Error loading xT data: {e}")

else:
    # Compute from Data
    data_type = st.radio("Data Type", ["Single Match", "Full Season"])

    if data_type == "Single Match":
        match_id = st.number_input("Match ID", value=9924, step=1)
        if st.button("Compute xT for Match"):
            with st.spinner("Loading match events..."):
                events = load_match_events(match_id=match_id)
            with st.spinner("Building transition matrix..."):
                probs = build_transition_matrix(events)
            with st.spinner("Solving xT..."):
                xT = solve_xT(probs)
            fig, ax = plot_xT(xT, title=f"xT for Match {match_id}", save_path=None, show=False)
            st.pyplot(fig)
            st.write("xT Values:", xT)

    else:  # Full Season
        team = st.text_input("Team Name", "Barcelona")
        competition_id = st.number_input("Competition ID", value=11, step=1)
        season_id = st.number_input("Season ID", value=1, step=1)
        if st.button("Compute xT for Season"):
            with st.spinner("Loading season events..."):
                events = load_season_events(competition_id=competition_id, season_id=season_id, team=team)
            with st.spinner("Building transition matrix..."):
                probs = build_transition_matrix(events)
            with st.spinner("Solving xT..."):
                xT = solve_xT(probs)
            fig, ax = plot_xT(xT, title=f"xT for {team} Season", save_path=None, show=False)
            st.pyplot(fig)
            st.write("xT Values:", xT)
            # Export
            export_xT(xT, probs, save_path="expected-threat-markov/output/xT_season_data.csv")
            st.success("xT data exported to CSV.")