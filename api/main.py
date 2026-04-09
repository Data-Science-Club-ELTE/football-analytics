from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import competitions, matches, xt

app = FastAPI(
    title="Football Analytics API",
    description="Expected Threat (xT) and match data for the Data Science Club project",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(competitions.router)
app.include_router(matches.router)
app.include_router(xt.router)


@app.get("/health")
def health():
    return {"status": "ok"}
