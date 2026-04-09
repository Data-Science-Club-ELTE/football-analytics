# Streamlit Dashboard

Interactive frontend for the Expected Threat (xT) analytics platform. Consumes the FastAPI backend — does **not** connect to the database directly.

## Prerequisites

- PostgreSQL database seeded with data (`python -m database.seed`)
- FastAPI server running

## How to Run

```bash
# Terminal 1 — start the API
source venv/bin/activate
uvicorn api.main:app --reload

# Terminal 2 — start the dashboard
source venv/bin/activate
streamlit run dashboard/app.py
```

The dashboard opens at `http://localhost:8501` by default.

## Features

- **Season View** — 12-zone xT heatmap, bar chart ranking, summary stats, detailed table
- **Match View** — match details with event breakdown, heatmap (PNG from API), zone table with season comparison
- Sidebar selectors for competition and match
- All data fetched from the API (`/api/v1/...`)
