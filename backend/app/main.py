from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from app.routers import ingest, query, synthesise
from app.services.vector_store import get_client
from app.services.paper_parser import paper_registry
from app.models import HealthResponse
from app.config import get_settings
import mlflow

s = get_settings()

# ── MLflow setup ──
# Tracking URI comes from MLFLOW_TRACKING_URI env var (set in docker-compose
# to point at the mlflow service). Don't open the sqlite file directly here —
# only the mlflow service should touch it, to avoid version/schema conflicts.
mlflow.set_experiment("synthesis-ai")

# ── App ──
app = FastAPI(
    title="SynthesisAI API",
    description="Multi-paper literature review engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

# ── Register routers ──
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(synthesise.router)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    qdrant_ok = False
    try:
        client = get_client()
        qdrant_ok = client.get_collections() is not None
    except Exception:
        pass

    return HealthResponse(
        qdrant=qdrant_ok,
        papers_loaded=len(paper_registry),
    )