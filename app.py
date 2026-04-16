import streamlit as st

# Set page config
st.set_page_config(
    page_title="Football Analytics Dashboard",
    page_icon="⚽",
    layout="wide"
)

# Define pages
expected_threat_page = st.Page("pages/expected_threat.py", title="Expected Threat (xT)", icon="⚽")
statistical_analysis_page = st.Page("pages/statistical_analysis.py", title="Statistical Analysis", icon="📊")
expected_goals_page = st.Page("pages/expected_goals.py", title="Expected Goals (xG)", icon="🎯")

# Navigation
pg = st.navigation([expected_threat_page, statistical_analysis_page, expected_goals_page])

# Run the selected page
pg.run()