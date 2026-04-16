import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import sys
sys.path.append('.')
from xgfun import calculate_xg_df

st.header("🎯 Expected Goals (xG) Analysis")

# Load xG data
try:
    xg_data = pd.read_csv("xg_model.csv")
    st.write("xG Data Preview:")
    st.dataframe(xg_data.head())

    # Calculate xG using the function
    xg_data_with_xg = calculate_xg_df(xg_data, x_col='x', y_col='y', normalize=True)

    st.write("Data with xG calculated:")
    st.dataframe(xg_data_with_xg[['x', 'y', 'is_goal', 'xg']].head())

    # Visualize shot locations with xG
    st.subheader("Shot Locations Colored by xG")
    fig, ax = plt.subplots(figsize=(10, 6))
    pitch = Pitch(pitch_type='opta', pitch_color='#22312b', line_color='#c7d5cc')  # Opta is 100x100
    pitch.draw(ax=ax)

    # Plot shots colored by xG
    scatter = ax.scatter(xg_data_with_xg['x'], xg_data_with_xg['y'], 
                        c=xg_data_with_xg['xg'], cmap='viridis', alpha=0.7, s=50, edgecolors='w')
    plt.colorbar(scatter, ax=ax, label='xG Value')
    ax.set_title("Shot Locations Colored by Expected Goals (xG)")
    st.pyplot(fig)

    # Statistics
    st.subheader("xG Statistics")
    total_shots = len(xg_data_with_xg)
    total_goals = xg_data_with_xg['is_goal'].sum()
    avg_xg = xg_data_with_xg['xg'].mean()
    total_xg = xg_data_with_xg['xg'].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Shots", total_shots)
    with col2:
        st.metric("Total Goals", int(total_goals))
    with col3:
        st.metric("Average xG", f"{avg_xg:.3f}")
    with col4:
        st.metric("Total xG", f"{total_xg:.1f}")

    # Shot types
    st.subheader("Shot Types")
    shot_types = ['LeftFoot', 'RightFoot', 'Head', 'OtherBodyPart']
    shot_counts = xg_data_with_xg[shot_types].sum()
    fig, ax = plt.subplots()
    shot_counts.plot(kind='bar', ax=ax)
    ax.set_title("Shots by Body Part")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # xG distribution
    st.subheader("xG Distribution")
    fig, ax = plt.subplots()
    ax.hist(xg_data_with_xg['xg'], bins=20, alpha=0.7, color='blue')
    ax.set_xlabel('xG')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of xG Values')
    st.pyplot(fig)

except FileNotFoundError:
    st.error("xG data file not found. Please ensure xg_model.csv exists in the parent directory.")
except Exception as e:
    st.error(f"Error loading xG data: {e}")