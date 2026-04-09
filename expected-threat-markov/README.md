# Expected Threat (xT) via Markov Chains

This folder contains a **reusable Markov chain model** that calculates **Expected Threat (xT)** — the probability of eventually scoring a goal from any zone on the pitch. It was built as part of the Data Science Club Football Analytics project, which aims to bring modern performance analytics to Hungarian football clubs.

---

## What is Expected Threat?

Expected Threat (xT), first introduced by Sarah Rudd (2011) and popularized by Karun Singh, answers the question: **"If the ball is in zone X, what is the probability it eventually becomes a goal?"**

Unlike xG (expected goals), which only evaluates shots, xT values **every ball action** — passes, carries, and shots — by measuring how much each action moves the ball closer to (or further from) a goal.

---

## How the Markov Chain Model Works

### 1. Divide the pitch into 12 zones

```
Attacking direction -->

 ┌──────────┬──────────┬──────────┬──────────┐
 │    DL    │    ML    │    AL    │  Box_L   │
 ├──────────┼──────────┼──────────┼──────────┤
 │    DC    │    MC    │    AC    │  Box_C   │
 ├──────────┼──────────┼──────────┼──────────┤
 │    DR    │    MR    │    AR    │  Box_R   │
 └──────────┴──────────┴──────────┴──────────┘

D = Defensive third (x: 0-40)
M = Middle third    (x: 40-80)
A = Attacking third (x: 80-102)
Box = Penalty box   (x: 102-120)

L = Left lane   (y: 53.33-80)
C = Center lane (y: 26.67-53.33)
R = Right lane  (y: 0-26.67)
```

### 2. Track every ball-moving action as a transition

Every **pass**, **carry**, and **shot** is a transition from one zone to another. Two states are **absorbing** (the chain ends there):
- **Goal** — a shot that goes in
- **Loss** — failed pass, shot saved/blocked/missed, ball out of play

### 3. Build a transition probability matrix

Count how many times the ball moved from each zone to every other zone (including Goal and Loss), then normalize each row to get probabilities. This produces a **12x14 matrix** (12 starting zones x 12 destination zones + Goal + Loss).

### 4. Solve for xT

Split the probability matrix into:
- **Q** (12x12) — zone-to-zone transition probabilities
- **g** (12x1) — direct goal probability from each zone

Then solve:

```
xT = (I - Q)^{-1} * g
```

where `(I - Q)^{-1}` is the **fundamental matrix** of the Markov chain. It captures all possible future paths the ball can take before reaching an absorbing state. The result is one number per zone: the probability of eventually scoring from there.

---

## Results: Barcelona 2017/18 Full La Liga Season

We ran the model on **all 36 Barcelona matches** from the La Liga 2017/18 season using StatsBomb open data.

**Dataset summary:**
- 36 matches, 44,643 ball-moving events, 90 goals
- Source: StatsBomb free data (`statsbombpy` API)

### xT Results Table

| Zone   | Name               |     xT | Direct Goal % | Loss % | Events |
| :----- | :----------------- | -----: | ------------: | -----: | -----: |
| Box_C  | Penalty Box Center | 0.1747 |         11.2% |  43.8% |    698 |
| Box_R  | Penalty Box Right  | 0.0483 |          0.1% |  23.0% |  1,080 |
| Box_L  | Penalty Box Left   | 0.0472 |          0.0% |  23.7% |  1,095 |
| AC     | Attacking Center   | 0.0466 |          0.3% |  16.6% |  2,611 |
| AL     | Attacking Left     | 0.0347 |          0.0% |   7.9% |  3,469 |
| AR     | Attacking Right    | 0.0343 |          0.1% |   8.5% |  3,283 |
| MC     | Middle Center      | 0.0258 |          0.0% |   5.2% |  7,076 |
| ML     | Middle Left        | 0.0250 |          0.0% |   4.8% |  7,783 |
| MR     | Middle Right       | 0.0241 |          0.0% |   5.3% |  7,552 |
| DC     | Defensive Center   | 0.0181 |          0.0% |   7.5% |  4,007 |
| DR     | Defensive Right    | 0.0179 |          0.0% |   7.3% |  3,110 |
| DL     | Defensive Left     | 0.0179 |          0.0% |   8.2% |  2,879 |

**Key insights:**
- **Box_C is king** — 17.5% chance of eventually scoring, driven by an 11.2% direct goal rate (but also 43.8% loss rate — high risk, high reward)
- **Attacking zones (AL, AC, AR)** carry 3-5% xT, acting as the main supply line into the box
- **Midfield is symmetric** — ML, MC, MR all hover around 2.4-2.6% xT, reflecting Barcelona's balanced buildup
- **Defense is flat** — all three defensive zones sit near 1.8% xT, meaning it barely matters *where* you start in your own third

### Heatmap Visualizations

The `output/` folder contains pitch heatmaps with xT values overlaid on each zone:

- `output/xT_season_barcelona.png` — Full season (36 matches)
- `output/xT_el_clasico.png` — Single match: Barcelona vs Real Madrid 2-2
- `output/xT_season_data.csv` — Raw CSV with all zone values

---

## Files in This Folder

| File | Description |
| :--- | :---------- |
| `markov_xT.py` | Reusable Python module — all core functions for loading data, building the matrix, solving xT, plotting, and exporting |
| `run_season_xT.py` | Script that runs the full Barcelona 2017/18 season analysis and saves results to `output/` |
| `markov-chain-demo.ipynb` | Step-by-step Jupyter notebook that walks through the entire model on a single El Clasico match, with explanations and visualizations |
| `output/` | Generated plots (PNG) and data (CSV) |

---

## How to Import and Use `markov_xT.py`

The module is designed to be imported into any Python script or Streamlit app. Here are the main functions:

### Quick Start — Single Match

```python
from markov_xT import load_match_events, build_transition_matrix, solve_xT, plot_xT

# Load El Clasico (Barcelona vs Real Madrid 2-2)
events = load_match_events(match_id=9924)

# Filter to one team
events = events[events["team"] == "Barcelona"]

# Build transition matrix and solve
probs = build_transition_matrix(events)
xT = solve_xT(probs)
# xT is a dict: {"Box_C": 0.1785, "MC": 0.0285, ...}

# Plot on a pitch
plot_xT(xT, title="Barcelona xT — El Clasico", save_path="output/xT.png")
```

### Full Season Analysis

```python
from markov_xT import load_season_events, build_transition_matrix, solve_xT, plot_xT, export_xT

# Load all Barcelona matches from La Liga 2017/18
events = load_season_events(competition_id=11, season_id=1, team="Barcelona")

probs = build_transition_matrix(events)
xT = solve_xT(probs)

plot_xT(xT, title="Barcelona 2017/18 Season xT", save_path="output/xT_season.png")
export_xT(xT, probs, save_path="output/xT_season_data.csv")
```

### Using in a Streamlit App

```python
import streamlit as st
from markov_xT import load_match_events, build_transition_matrix, solve_xT, plot_xT

st.title("Expected Threat (xT) Explorer")

match_id = st.number_input("StatsBomb Match ID", value=9924)

if st.button("Calculate xT"):
    events = load_match_events(match_id=match_id)
    teams = events["team"].unique().tolist()
    team = st.selectbox("Select team", teams)
    events = events[events["team"] == team]

    probs = build_transition_matrix(events)
    xT = solve_xT(probs)

    fig, ax = plot_xT(xT, title=f"{team} xT", show=False)
    st.pyplot(fig)

    # Show as a table
    st.dataframe(
        [{"Zone": k, "xT": round(v, 4)} for k, v in sorted(xT.items(), key=lambda x: -x[1])]
    )
```

### Function Reference

| Function | Input | Output |
| :------- | :---- | :----- |
| `load_match_events(match_id)` | StatsBomb match ID | DataFrame with zones and classifications |
| `load_season_events(competition_id, season_id, team=None)` | Competition/season IDs, optional team filter | DataFrame of all events across the season |
| `build_transition_matrix(events)` | Processed events DataFrame | 12x14 transition probability matrix |
| `solve_xT(transition_probs)` | Probability matrix | `dict` mapping zone code to xT value |
| `plot_xT(xT_dict, title, save_path, show)` | xT dict + display options | `(fig, ax)` matplotlib objects |
| `export_xT(xT_dict, transition_probs, save_path)` | xT dict + probs matrix | Saves CSV, returns DataFrame |
| `statsbomb_to_zone(x, y)` | StatsBomb coordinates | Zone code string (e.g. `"MC"`, `"Box_C"`) |

---

## What's Needed Before Running on FTC vs MTK

The FTC vs MTK (3-1) match from the Hungarian league is **manually collected data**, not from StatsBomb. Before we can run the Markov model on it, we need:

1. **Event-level data in the right format** — Each row needs:
   - `type`: one of `"Pass"`, `"Carry"`, or `"Shot"`
   - `team`: team name string
   - `start_zone`: one of our 12 zone codes (`DL`, `DC`, `DR`, `ML`, `MC`, `MR`, `AL`, `AC`, `AR`, `Box_L`, `Box_C`, `Box_R`)
   - `end_zone`: destination zone code
   - `is_goal`: `True`/`False`
   - `is_loss`: `True`/`False`

2. **If zones are already coded** (from the Excel template) — You can skip `load_match_events()` entirely and feed the DataFrame straight into `build_transition_matrix()`:

   ```python
   import pandas as pd
   from markov_xT import build_transition_matrix, solve_xT, plot_xT

   # Load your manually collected data
   df = pd.read_csv("path/to/ftc_vs_mtk_events.csv")

   # Make sure it has: start_zone, end_zone, is_goal, is_loss columns
   probs = build_transition_matrix(df)
   xT = solve_xT(probs)
   plot_xT(xT, title="FTC vs MTK (3-1) xT")
   ```

3. **If you have raw (x, y) coordinates instead** — Use `statsbomb_to_zone(x, y)` to map them first, but note the coordinate system must match StatsBomb's scale (x: 0-120, y: 0-80). If your coordinates use a different scale, you'll need to normalize them first.

4. **Minimum data volume** — A single match (~1,000-2,000 events) will produce noisy xT values (many zones will have few observations). The model works best with multiple matches. For reliable results, aim for 10+ matches.

---

## Dependencies

```
pip install numpy pandas matplotlib mplsoccer statsbombpy
```

`statsbombpy` is only needed for loading StatsBomb data. If you're working with manually collected data (like FTC vs MTK), you only need `numpy`, `pandas`, `matplotlib`, and `mplsoccer`.

---

## Database Layer (PostgreSQL)

The project includes a PostgreSQL database for persisting events and xT results. See the root `database/` directory.

### Setup

```bash
# 1. Activate the virtual environment
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Create a .env file from the example
cp .env.example .env
# Edit .env with your PostgreSQL connection string

# 4. Create the database (in psql)
# CREATE DATABASE football_analytics;

# 5. Run migrations
alembic upgrade head

# 6. Seed with Barcelona 2017/18 data
python -m database.seed
```

### Tables

| Table | Description |
|-------|-------------|
| `competitions` | League + season metadata (links to StatsBomb IDs) |
| `matches` | Individual match records with teams and date |
| `events` | Every pass, carry, and shot with zones and coordinates |
| `xt_results` | Computed xT values per zone, scoped to match or season |
