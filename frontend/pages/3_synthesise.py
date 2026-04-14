import streamlit as st
import httpx, os, re
import plotly.express as px
import pandas as pd

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "default-key")
HEADERS = {"X-API-Key": API_KEY}

st.header("🧪 Cross-Paper Synthesis")

if st.button("🔬 Run Full Synthesis", type="primary"):
    with st.spinner("Analysing corpus — running 3 LLM calls in parallel + graph builder..."):
        try:
            resp = httpx.post(
                f"{API_URL}/synthesise/",
                headers=HEADERS,
                timeout=180,
            )
            if resp.status_code == 200:
                st.session_state["synthesis"] = resp.json()
                try:
                    papers_resp = httpx.get(
                        f"{API_URL}/synthesise/papers",
                        headers=HEADERS,
                        timeout=30,
                    )
                    if papers_resp.status_code == 200:
                        st.session_state["paper_names"] = {
                            p["paper_id"]: p["title"] for p in papers_resp.json()
                        }
                    else:
                        st.session_state["paper_names"] = {}
                except Exception:
                    st.session_state["paper_names"] = {}
                st.success("Synthesis complete!")
            else:
                st.error(f"Synthesis failed: {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")

if "synthesis" not in st.session_state:
    st.info("Upload papers first, then click 'Run Full Synthesis'.")
    st.stop()

data = st.session_state["synthesis"]
paper_names: dict[str, str] = st.session_state.get("paper_names", {})

def name_of(pid: str) -> str:
    return paper_names.get(pid, pid)

def replace_ids(text: str) -> str:
    if not text or not paper_names:
        return text
    for pid, title in paper_names.items():
        text = text.replace(pid, title)
    return text

st.caption(f"Corpus size: {data['corpus_size']} papers")

t1, t2, t3, t4, t5, t6 = st.tabs([
    "🔀 Method Matrix",
    "⚡ Contradictions",
    "🕳️ Research Gaps",
    "📊 Leaderboard",
    "📝 Lit Review Draft",
    "🕸️ Knowledge Graph",
])

# ── Tab 1: Method Matrix ──
with t1:
    st.subheader("Method Comparison Matrix")
    if not data["method_matrix"]:
        st.info("No methods detected across the corpus.")
    for m in data["method_matrix"]:
        with st.expander(
            f"**{m['method_name']}** — "
            f"used by {len(m['papers_using_it'])} paper(s)"
        ):
            st.write(f"**Best result:** {replace_ids(m['best_result']) if m['best_result'] else 'N/A'}")
            st.write(f"**Notes:** {replace_ids(m['notes'])}")
            st.write(f"**Papers:** {', '.join(name_of(pid) for pid in m['papers_using_it'])}")

# ── Tab 2: Contradictions ──
with t2:
    st.subheader("Contradictions Found")
    if not data["contradictions"]:
        st.success("No contradictions found — the corpus is consistent.")
    for c in data["contradictions"]:
        severity_icon = {
            "minor": "🟡", "major": "🟠", "critical": "🔴"
        }
        st.markdown(
            f"{severity_icon.get(c['severity'], '⚪')} "
            f"**{c['severity'].upper()}**"
        )
        col1, col2 = st.columns(2)
        col1.warning(f"**{name_of(c['paper_a'])}:** {c['claim_a']}")
        col2.error(f"**{name_of(c['paper_b'])}:** {c['claim_b']}")
        st.caption(replace_ids(c["explanation"]))
        st.divider()

# ── Tab 3: Research Gaps ──
with t3:
    st.subheader("Research Gaps Identified")
    if not data["research_gaps"]:
        st.info("No obvious gaps found.")
    for i, g in enumerate(data["research_gaps"], 1):
        with st.expander(f"Gap {i}: {replace_ids(g['gap_description'])}"):
            st.write(f"**Evidence:** {replace_ids(g['evidence'])}")
            st.info(f"💡 **Suggested approach:** {replace_ids(g['potential_approach'])}")

# ── Tab 4: Leaderboard ──
with t4:
    st.subheader("Results Leaderboard")
    if not data["results_leaderboard"]:
        st.info("No quantitative results extracted from the corpus.")
    else:
        df = pd.DataFrame(data["results_leaderboard"])
        st.dataframe(df, use_container_width=True)

        # Chart: accuracy by paper (if available)
        if "accuracy" in df.columns:
            fig = px.bar(
                df, x="title", y="accuracy",
                color="methods",
                title="Accuracy by Paper",
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 5: Lit Review ──
with t5:
    st.subheader("Auto-Generated Literature Review")
    lit_review = replace_ids(data["lit_review_draft"])
    st.markdown(lit_review)
    st.download_button(
        "📥 Download as Markdown",
        data=lit_review,
        file_name="literature_review.md",
        mime="text/markdown",
    )

# ── Tab 6: Knowledge Graph (summary) ──
with t6:
    st.subheader("Knowledge Graph Summary")
    edges = data["graph_edges"]
    st.metric("Total Edges", len(edges))

    # Group edges by relation type
    relation_counts = {}
    for e in edges:
        rel = e["relation"]
        relation_counts[rel] = relation_counts.get(rel, 0) + 1

    for rel, count in relation_counts.items():
        st.write(f"**{rel}:** {count} edges")

    st.info("👈 See the **Knowledge Graph** page for the interactive visualisation.")