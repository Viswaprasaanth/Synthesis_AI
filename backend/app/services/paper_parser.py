import json, re, hashlib
from app.services.llm import get_llm
from app.models import PaperMeta

# ── In-memory registry of all parsed papers ──
# Key: paper_id, Value: PaperMeta
paper_registry: dict[str, PaperMeta] = {}


PARSE_PROMPT = """You are a research paper metadata extractor.
Given the FULL TEXT of an academic paper, extract EXACTLY this JSON:
{{
  "title": "...",
  "authors": ["Author 1", "Author 2"],
  "year": 2024,
  "abstract": "first 2-3 sentences of abstract",
  "methods": ["ResNet50", "YOLO v5", "transfer learning", ...],
  "datasets": ["ImageNet", "COCO", "custom IR thermal dataset", ...],
  "metrics": {{"accuracy": 0.97, "f1_score": 0.95, "precision": 0.96}}
}}

RULES:
- "methods" = every ML model, algorithm, or technique mentioned as USED
  (not just cited). Include architectures, training strategies, ensembles.
- "datasets" = every dataset used for training or evaluation
- "metrics" = only the BEST reported result for each metric (float values)
- If a field is not found, use null for scalars, [] for lists, {{}} for dicts
- Return ONLY valid JSON, no markdown fences, no explanation

PAPER TEXT:
{paper_text}"""


async def parse_paper(paper_text: str) -> PaperMeta:
    """Extract structured metadata from a single paper via LLM."""
    llm = get_llm()

    # Truncate to ~6000 tokens to stay within context window
    truncated = paper_text[:20000]

    response = await llm.ainvoke(
        PARSE_PROMPT.format(paper_text=truncated)
    )

    # Parse LLM JSON response
    raw = response.content.strip()
    raw = re.sub(r"```json\s*|```", "", raw).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: create minimal metadata
        data = {
            "title": "Untitled Paper",
            "authors": [],
            "year": None,
            "abstract": paper_text[:300],
            "methods": [],
            "datasets": [],
            "metrics": {},
        }

    # Generate deterministic paper_id from title
    title = data.get("title", "unknown")
    paper_id = hashlib.md5(title.lower().encode()).hexdigest()[:12]

    # Ensure metrics values are floats
    metrics = {}
    for k, v in data.get("metrics", {}).items():
        try:
            metrics[k] = float(v)
        except (ValueError, TypeError):
            pass

    paper = PaperMeta(
        paper_id=paper_id,
        title=title,
        authors=data.get("authors", []),
        year=data.get("year"),
        abstract=data.get("abstract", ""),
        methods=data.get("methods", []),
        datasets=data.get("datasets", []),
        metrics=metrics,
    )

    # Register in global registry
    paper_registry[paper_id] = paper

    return paper