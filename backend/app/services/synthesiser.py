import json
import re
from app.services.llm import get_llm
from app.models import (
    PaperMeta, MethodComparison, Contradiction, ResearchGap,
)


def _corpus_summary(papers: list[PaperMeta]) -> str:
    """Build a structured text summary of all papers for LLM context."""
    lines = []
    for p in papers:
        metrics_str = ", ".join(
            f"{k}: {v}" for k, v in p.metrics.items()
        ) or "none reported"
        lines.append(
            f"[{p.paper_id}] \"{p.title}\" ({p.year or 'year unknown'})\n"
            f"  Authors: {', '.join(p.authors) or 'unknown'}\n"
            f"  Methods: {', '.join(p.methods) or 'none listed'}\n"
            f"  Datasets: {', '.join(p.datasets) or 'none listed'}\n"
            f"  Metrics: {metrics_str}"
        )
    return "\n\n".join(lines)


def _parse_json(text: str):
    """Safely parse JSON from LLM response, stripping markdown fences."""
    cleaned = re.sub(r"```json\s*|```", "", text.strip())
    return json.loads(cleaned)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. METHOD COMPARATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def compare_methods(papers: list[PaperMeta]) -> list[MethodComparison]:
    """Build method-vs-paper comparison matrix."""
    prompt = f"""You are a systematic review assistant.
Given these papers from a research corpus:

{_corpus_summary(papers)}

Create a METHOD COMPARISON MATRIX. For each unique method/model/technique
used across the corpus, list:
1. Which papers use it (by paper_id)
2. The best reported result for that method across all papers
3. Any notes about how usage differs between papers

Return ONLY a JSON array, no other text:
[{{"method_name": "ResNet50", "papers_using_it": ["id1","id2"],
   "best_result": "Paper id1: 98.2% accuracy", "notes": "id1 uses frozen transfer learning, id2 uses full fine-tuning"}}]
"""
    llm = get_llm()
    resp = await llm.ainvoke(prompt)
    try:
        data = _parse_json(resp.content)
        return [MethodComparison(**item) for item in data]
    except (json.JSONDecodeError, Exception):
        return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. CONTRADICTION DETECTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def find_contradictions(papers: list[PaperMeta]) -> list[Contradiction]:
    """Find papers that report conflicting results."""
    prompt = f"""You are a critical research analyst.
Given these papers from a research corpus:

{_corpus_summary(papers)}

Find CONTRADICTIONS — cases where two papers:
- Report conflicting results for the same method on similar data
- Make opposing claims about the same technique's effectiveness
- Reach different conclusions on similar experiments

For each contradiction found:
- Quote the specific claims from each paper
- Rate severity: "minor" (slightly different numbers),
  "major" (opposite conclusions), "critical" (fundamentally incompatible)

Return ONLY a JSON array:
[{{"claim_a": "ResNet50 achieves 97% accuracy on thermal images",
   "paper_a": "id1",
   "claim_b": "ResNet50 only reaches 82% on thermal fault detection",
   "paper_b": "id2",
   "explanation": "Different dataset sizes may explain the discrepancy",
   "severity": "major"}}]

If no contradictions found, return an empty array: []
"""
    llm = get_llm()
    resp = await llm.ainvoke(prompt)
    try:
        data = _parse_json(resp.content)
        return [Contradiction(**item) for item in data]
    except (json.JSONDecodeError, Exception):
        return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. RESEARCH GAP FINDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def find_gaps(papers: list[PaperMeta]) -> list[ResearchGap]:
    """Identify research gaps from the corpus."""
    prompt = f"""You are a research advisor analyzing a corpus of {len(papers)} papers.

{_corpus_summary(papers)}

Identify RESEARCH GAPS — methods, datasets, combinations, or questions
that the corpus does NOT address but logically SHOULD. Focus on:
1. Methods used in some papers but never tested with datasets from others
2. Evaluation metrics used inconsistently across the corpus
3. Missing baselines or comparisons that would strengthen the field
4. Unexplored variations of existing approaches
5. Combinations of techniques no paper has tried

Return ONLY a JSON array:
[{{"gap_description": "No paper has applied Vision Transformers to IR thermal fault detection",
   "evidence": "All 5 papers use CNN-based architectures; ViT has shown strong results in similar domains",
   "potential_approach": "Fine-tune ViT-Base on the COCO-annotated thermal dataset from paper id1"}}]
"""
    llm = get_llm()
    resp = await llm.ainvoke(prompt)
    try:
        data = _parse_json(resp.content)
        return [ResearchGap(**item) for item in data]
    except (json.JSONDecodeError, Exception):
        return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. RESULTS AGGREGATOR / LEADERBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def aggregate_results(papers: list[PaperMeta]) -> list[dict]:
    """Build a sortable leaderboard from all papers' metrics. No LLM needed."""
    rows = []
    for p in papers:
        if p.metrics:
            rows.append({
                "paper_id": p.paper_id,
                "title": p.title,
                "year": p.year,
                "methods": ", ".join(p.methods),
                **p.metrics,
            })

    # Sort by accuracy descending (or first available metric)
    sort_key = None
    for preferred in ["accuracy", "f1_score", "f1", "precision"]:
        if any(preferred in r for r in rows):
            sort_key = preferred
            break

    if sort_key:
        rows.sort(key=lambda r: r.get(sort_key, 0), reverse=True)
    return rows


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. LIT REVIEW DRAFT GENERATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def generate_lit_review(
    papers: list[PaperMeta],
    methods: list[MethodComparison],
    gaps: list[ResearchGap],
) -> str:
    """Generate a structured literature review draft."""
    methods_text = "\n".join(
        f"- {m.method_name}: used by {len(m.papers_using_it)} papers, "
        f"best: {m.best_result or 'N/A'}"
        for m in methods
    ) or "No method data available"

    gaps_text = "\n".join(
        f"- {g.gap_description}" for g in gaps
    ) or "No gaps identified"

    prompt = f"""Write a structured LITERATURE REVIEW (800-1200 words) based on
this corpus of {len(papers)} research papers.

CORPUS:
{_corpus_summary(papers)}

METHOD LANDSCAPE:
{methods_text}

IDENTIFIED GAPS:
{gaps_text}

STRUCTURE YOUR REVIEW AS:
1. **Introduction** — scope, motivation, number of papers reviewed
2. **Methodological Approaches** — group papers by technique families,
   compare approaches, note trends
3. **Results and Comparative Analysis** — which methods perform best,
   on which datasets, under what conditions
4. **Research Gaps and Future Directions** — based on the gaps identified
5. **Conclusion** — synthesize key findings

RULES:
- Reference papers by [paper_id] inline, e.g. "Smith et al. [abc123] proposed..."
- Use academic tone throughout
- Highlight both agreements AND disagreements between papers
- End with concrete, actionable future work suggestions
- Do NOT invent papers or results not in the corpus
"""
    llm = get_llm()
    resp = await llm.ainvoke(prompt)
    return resp.content


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. KNOWLEDGE GRAPH BUILDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_graph_edges(papers: list[PaperMeta]) -> list[dict]:
    """Build edges for the knowledge graph from structured metadata.
    No LLM needed — pure Python logic on parsed metadata."""
    edges = []

    # Paper → Method edges
    for p in papers:
        for method in p.methods:
            edges.append({
                "source": p.paper_id,
                "target": method,
                "relation": "uses_method",
                "weight": 1.0,
            })

    # Paper → Dataset edges
    for p in papers:
        for ds in p.datasets:
            edges.append({
                "source": p.paper_id,
                "target": ds,
                "relation": "uses_dataset",
                "weight": 1.0,
            })

    # Paper ↔ Paper edges (shared methods = research similarity)
    for i, p1 in enumerate(papers):
        for p2 in papers[i + 1:]:
            shared_methods = set(p1.methods) & set(p2.methods)
            if shared_methods:
                edges.append({
                    "source": p1.paper_id,
                    "target": p2.paper_id,
                    "relation": "shares_methods",
                    "weight": float(len(shared_methods)),
                })

            shared_datasets = set(p1.datasets) & set(p2.datasets)
            if shared_datasets:
                edges.append({
                    "source": p1.paper_id,
                    "target": p2.paper_id,
                    "relation": "shares_datasets",
                    "weight": float(len(shared_datasets)),
                })

    return edges