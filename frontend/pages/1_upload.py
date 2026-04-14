import streamlit as st
import httpx, os

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "default-key")
HEADERS = {"X-API-Key": API_KEY}

st.header("📄 Upload Research Papers")
st.caption("Upload 2–50 PDFs to build your research corpus.")

files = st.file_uploader(
    "Drop your PDFs here",
    type=["pdf"],
    accept_multiple_files=True,
)

if files and st.button("🚀 Ingest All Papers", type="primary"):
    progress = st.progress(0)
    status = st.empty()

    papers_ingested = []

    for i, f in enumerate(files):
        status.text(f"Processing {f.name} ({i+1}/{len(files)})...")
        try:
            resp = httpx.post(
                f"{API_URL}/ingest/",
                headers=HEADERS,
                files={"file": (f.name, f.read(), "application/pdf")},
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                st.success(
                    f"✓ {f.name} → \"{data['title']}\" "
                    f"({data['chunks_indexed']} chunks)"
                )
                papers_ingested.append(data)
            else:
                st.error(f"✕ {f.name} — {resp.text}")
        except Exception as e:
            st.error(f"✕ {f.name} — {str(e)}")

        progress.progress((i + 1) / len(files))

    if papers_ingested:
        st.session_state["papers"] = papers_ingested
        st.balloons()
        st.success(f"🎉 Corpus ready: {len(papers_ingested)} papers ingested.")
        st.info("Go to **Corpus Overview** or **Synthesise** in the sidebar.")