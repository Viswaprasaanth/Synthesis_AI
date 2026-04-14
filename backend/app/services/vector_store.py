from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from app.services.embedder import embed_texts, embed_query
from app.config import get_settings
import uuid

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        s = get_settings()
        _client = QdrantClient(
            url=s.qdrant_url,
            api_key=s.qdrant_api_key,
            timeout=60,
        )
    return _client


def ensure_collection():
    """Create the collection if it doesn't exist."""
    s = get_settings()
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]
    if s.collection_name not in collections:
        client.create_collection(
            collection_name=s.collection_name,
            vectors_config=VectorParams(
                size=384,  # all-MiniLM-L6-v2 dimension
                distance=Distance.COSINE,
            ),
        )


def upsert_chunks(chunks: list[dict]):
    """
    Store chunks with paper metadata as Qdrant payload.
    Each chunk's payload includes paper_id, paper_title, year, chunk_index.
    """
    s = get_settings()
    client = get_client()
    ensure_collection()

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "text": chunk["text"],
                **chunk["metadata"],
            },
        )
        for vec, chunk in zip(vectors, chunks)
    ]

    # Batch the upsert so each request stays small and fast — a single
    # request with hundreds of points can time out against Qdrant Cloud.
    BATCH_SIZE = 64
    for i in range(0, len(points), BATCH_SIZE):
        client.upsert(
            collection_name=s.collection_name,
            points=points[i : i + BATCH_SIZE],
            wait=True,
        )


def search(
    query: str,
    top_k: int = 10,
    paper_id: str | None = None,
) -> list[dict]:
    """
    Semantic search with optional paper_id filter.
    - paper_id=None  → search across ALL papers (corpus-wide)
    - paper_id="abc" → search only within that one paper
    """
    s = get_settings()
    client = get_client()
    query_vec = embed_query(query)

    qfilter = None
    if paper_id:
        qfilter = Filter(must=[
            FieldCondition(
                key="paper_id",
                match=MatchValue(value=paper_id),
            )
        ])

    results = client.query_points(
        collection_name=s.collection_name,
        query=query_vec,
        limit=top_k,
        query_filter=qfilter,
    )

    return [
        {
            "text": r.payload["text"],
            "paper_id": r.payload.get("paper_id"),
            "paper_title": r.payload.get("paper_title"),
            "year": r.payload.get("year"),
            "score": r.score,
        }
        for r in results.points
    ]


def delete_paper(paper_id: str):
    """Delete all chunks belonging to a specific paper."""
    s = get_settings()
    client = get_client()
    client.delete(
        collection_name=s.collection_name,
        points_selector=Filter(must=[
            FieldCondition(
                key="paper_id",
                match=MatchValue(value=paper_id),
            )
        ]),
    )