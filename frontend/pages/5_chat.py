import streamlit as st
import httpx, os

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "default-key")
HEADERS = {"X-API-Key": API_KEY}

st.header("💬 Chat with Your Corpus")
st.caption(
    "Ask questions across all uploaded papers. "
    "The AI will reference specific papers in its answers."
)

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if question := st.chat_input("Ask anything about your research corpus..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Get response from backend
    with st.chat_message("assistant"):
        with st.spinner("Searching corpus..."):
            try:
                resp = httpx.post(
                    f"{API_URL}/query/",
                    headers=HEADERS,
                    json={
                        "question": question,
                        "top_k": 8,
                    },
                    timeout=60,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])

                    st.write(answer)

                    # Show source papers
                    if sources:
                        with st.expander(f"📄 Sources ({len(sources)} chunks)"):
                            for s in sources:
                                st.caption(
                                    f"**{s.get('paper_title', 'Unknown')}** "
                                    f"[{s.get('paper_id', '?')}] — "
                                    f"Score: {s.get('score', 0):.2f}"
                                )
                                st.write(s["text"])
                                st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                    })
                else:
                    st.error("Query failed. Is the backend running?")
            except Exception as e:
                st.error(f"Connection error: {e}")