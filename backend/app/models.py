from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Ingest ────────────────────────────────────────
class IngestResponse(BaseModel):
    paper_id: str
    title: str
    filename: str
    chunks_indexed: int
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


# ── Query ─────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    paper_id: Optional[str] = None   # None = search all papers
    top_k: int = Field(default=8, ge=1, le=20)


class SourceChunk(BaseModel):
    text: str
    paper_id: Optional[str] = None
    paper_title: Optional[str] = None
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    model: str
    latency_ms: float


# ── Per-Paper Metadata ────────────────────────────
class PaperMeta(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    year: Optional[int] = None
    abstract: str
    methods: list[str]
    datasets: list[str]
    metrics: dict[str, float]


# ── Cross-Paper: Method Comparison ────────────────
class MethodComparison(BaseModel):
    method_name: str
    papers_using_it: list[str]
    best_result: Optional[str] = None
    notes: str


# ── Cross-Paper: Contradiction ────────────────────
class Contradiction(BaseModel):
    claim_a: str
    paper_a: str
    claim_b: str
    paper_b: str
    explanation: str
    severity: str   # "minor" | "major" | "critical"


# ── Cross-Paper: Research Gap ─────────────────────
class ResearchGap(BaseModel):
    gap_description: str
    evidence: str
    potential_approach: str


# ── Cross-Paper: Graph Edge ───────────────────────
class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str   # "uses_method" | "uses_dataset" | "shares_methods"
    weight: float = 1.0


# ── Full Synthesis Response ───────────────────────
class SynthesisResult(BaseModel):
    corpus_size: int
    method_matrix: list[MethodComparison]
    contradictions: list[Contradiction]
    research_gaps: list[ResearchGap]
    graph_edges: list[GraphEdge]
    lit_review_draft: str
    results_leaderboard: list[dict]


# ── Health ────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "ok"
    qdrant: bool
    papers_loaded: int
    version: str = "1.0.0"