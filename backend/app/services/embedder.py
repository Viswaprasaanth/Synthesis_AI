from sentence_transformers import SentenceTransformer
from app.config import get_settings

_model = None


def _get_model():
    global _model
    if _model is None:
        s = get_settings()
        _model = SentenceTransformer(s.embedding_model)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of text strings into vectors."""
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]