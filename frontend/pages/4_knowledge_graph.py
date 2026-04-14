import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

st.header("🕸️ Interactive Knowledge Graph")

if "synthesis" not in st.session_state:
    st.warning("Run synthesis first on the Synthesise page.")
    st.stop()

edges_data = st.session_state["synthesis"]["graph_edges"]

if not edges_data:
    st.info("No graph edges generated. Upload more papers.")
    st.stop()

# ── Collect unique node IDs ──
node_ids = set()
for e in edges_data:
    node_ids.add(e["source"])
    node_ids.add(e["target"])

# ── Determine node types by checking relation types ──
paper_ids = set()
method_ids = set()
dataset_ids = set()

for e in edges_data:
    if e["relation"] == "uses_method":
        paper_ids.add(e["source"])
        method_ids.add(e["target"])
    elif e["relation"] == "uses_dataset":
        paper_ids.add(e["source"])
        dataset_ids.add(e["target"])

# ── Build nodes with color-coding ──
nodes = []
for nid in node_ids:
    if nid in paper_ids:
        nodes.append(Node(id=nid, label=nid[:8] + "…",
                          color="#60a5fa", size=28))
    elif nid in method_ids:
        nodes.append(Node(id=nid, label=nid,
                          color="#5eead4", size=20))
    elif nid in dataset_ids:
        nodes.append(Node(id=nid, label=nid,
                          color="#fbbf24", size=20))
    else:
        nodes.append(Node(id=nid, label=nid,
                          color="#a78bfa", size=18))

# ── Build edges ──
edge_colors = {
    "uses_method": "#2dd4bf",
    "uses_dataset": "#fbbf24",
    "shares_methods": "#60a5fa",
    "shares_datasets": "#a78bfa",
}

edges = [
    Edge(
        source=e["source"],
        target=e["target"],
        label=e["relation"].replace("_", " "),
        color=edge_colors.get(e["relation"], "#3b3d52"),
    )
    for e in edges_data
]

# ── Render graph ──
config = Config(
    width=900,
    height=600,
    directed=False,
    physics=True,
    hierarchical=False,
    nodeHighlightBehavior=True,
    highlightColor="#e85d26",
)

st.caption("🔵 Papers  ·  🟢 Methods  ·  🟡 Datasets  ·  🟣 Shared links")
agraph(nodes=nodes, edges=edges, config=config)