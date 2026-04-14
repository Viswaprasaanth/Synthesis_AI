import streamlit as st
import httpx, os

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "default-key")
HEADERS = {"X-API-Key": API_KEY}

st.header("📚 Corpus Overview")

if st.button("🔄 Refresh"):
    try:
        resp = httpx.get(
            f"{API_URL}/synthesise/papers",
            headers=HEADERS,
            timeout=30,
        )
        if resp.status_code == 200:
            papers = resp.json()
            st.session_state["corpus_papers"] = papers
    except Exception as e:
        st.error(f"Could not connect to backend: {e}")

papers = st.session_state.get("corpus_papers", [])

if not papers:
    st.warning("No papers in corpus yet. Go to Upload page first.")
    st.stop()

st.metric("Papers in Corpus", len(papers))

for p in papers:
    with st.expander(f"📄 {p['title']} ({p.get('year', '?')})"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Paper ID", p["paper_id"])
        col2.metric("Methods", p["methods_count"])
        col3.metric("Datasets", p["datasets_count"])