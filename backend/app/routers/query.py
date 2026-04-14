import time
from fastapi import APIRouter, Depends
from app.middleware.auth import require_api_key
from app.services.vector_store import search
from app.services.llm import get_llm
from app.models import QueryRequest, QueryResponse, SourceChunk
from app.config import get_settings

router = APIRouter(prefix="/query", tags=["query"])


QUERY_PROMPT = """You are a research assistant with access to a corpus of academic papers.
Answer the question based on the provided context. Be detailed and thorough.
If the context contains relevant information, synthesize it into a clear answer.
Always mention which paper(s) the information comes from.

CONTEXT FROM CORPUS:
{context}

QUESTION: {question}

Provide a detailed answer:"""


@router.post("/", response_model=QueryResponse)
async def query_corpus(
    body: QueryRequest,
    _: str = Depends(require_api_key),
):
    """RAG query — optionally filtered to a single paper or entire corpus."""
    s = get_settings()
    t0 = time.perf_counter()

    # Retrieve relevant chunks (corpus-wide or per-paper)
    results = search(
        query=body.question,
        top_k=body.top_k,
        paper_id=body.paper_id,
    )

    if not results:
        return QueryResponse(
            answer="No relevant information found in the corpus.",
            sources=[],
            model=s.llm_model,
            latency_ms=0,
        )

    # Build context with paper attribution
    context = "\n\n---\n\n".join(
        f"[From: {r['paper_title']} ({r['paper_id']})]\n{r['text']}"
        for r in results
    )

    # Ask LLM
    llm = get_llm()
    response = await llm.ainvoke(
        QUERY_PROMPT.format(context=context, question=body.question)
    )

    latency = (time.perf_counter() - t0) * 1000

    return QueryResponse(
        answer=response.content,
        sources=[
            SourceChunk(
                text=r["text"][:200],
                paper_id=r.get("paper_id"),
                paper_title=r.get("paper_title"),
                score=r.get("score", 0),
            )
            for r in results
        ],
        model=s.llm_model,
        latency_ms=round(latency, 2),
    )