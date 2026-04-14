import asyncio
import time
from fastapi import APIRouter, HTTPException, Depends
import mlflow

from app.middleware.auth import require_api_key
from app.services.paper_parser import paper_registry
from app.services.synthesiser import (
    compare_methods, find_contradictions, find_gaps,
    aggregate_results, generate_lit_review, build_graph_edges,
)
from app.models import SynthesisResult

router = APIRouter(prefix="/synthesise", tags=["synthesis"])


@router.post("/", response_model=SynthesisResult)
async def synthesise_corpus(
    _: str = Depends(require_api_key),
):
    """Run all 6 cross-paper analysis features on the entire corpus."""
    papers = list(paper_registry.values())

    if len(papers) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 2 papers for cross-paper synthesis. "
                   f"Currently loaded: {len(papers)}. Upload more papers first."
        )

    start = time.time()

    with mlflow.start_run(run_name="synthesis"):
        mlflow.log_param("corpus_size", len(papers))
        mlflow.log_param("paper_ids", [p.paper_id for p in papers])

        # Run 3 LLM-based extractors in parallel
        methods, contradictions, gaps = await asyncio.gather(
            compare_methods(papers),
            find_contradictions(papers),
            find_gaps(papers),
        )

        # CPU-only tasks — no LLM needed, instant
        leaderboard = aggregate_results(papers)
        graph_edges = build_graph_edges(papers)

        # Lit review depends on methods + gaps, so runs after
        lit_review = await generate_lit_review(papers, methods, gaps)

        latency = time.time() - start

        mlflow.log_metrics({
            "latency_seconds": round(latency, 2),
            "methods_found": len(methods),
            "contradictions_found": len(contradictions),
            "gaps_found": len(gaps),
            "leaderboard_rows": len(leaderboard),
            "graph_edges": len(graph_edges),
        })

    return SynthesisResult(
        corpus_size=len(papers),
        method_matrix=methods,
        contradictions=contradictions,
        research_gaps=gaps,
        graph_edges=graph_edges,
        lit_review_draft=lit_review,
        results_leaderboard=leaderboard,
    )


@router.get("/papers")
async def list_papers(_: str = Depends(require_api_key)):
    """List all papers currently in the corpus."""
    return [
        {
            "paper_id": p.paper_id,
            "title": p.title,
            "year": p.year,
            "methods_count": len(p.methods),
            "datasets_count": len(p.datasets),
        }
        for p in paper_registry.values()
    ]