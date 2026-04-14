import streamlit as st

st.set_page_config(
    page_title="SynthesisAI — Literature Review Engine",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 SynthesisAI")
st.markdown("### Systematic Literature Review Engine")
st.markdown(
    "Upload a corpus of research papers (5–50 PDFs) → get cross-paper "
    "method comparison, contradiction detection, research gap analysis, "
    "an interactive knowledge graph, and an auto-generated literature "
    "review draft with cross-citations."
)
st.info("👈 Start by uploading papers in the **Upload** page.")