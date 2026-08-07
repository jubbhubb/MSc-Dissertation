import json
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------------------------------
# Load themes
# ----------------------------------------------------

with open("experiments/theme_generation_document_corpus/output_files/theme_grouping_input.json", "r", encoding="utf-8") as f:
    themes = json.load(f)


theme_names = [
    t["theme"]
    for t in themes
]


theme_texts = [
    f"{t['theme']}\n\n{t['description']}"
    for t in themes
]


# ----------------------------------------------------
# Code counts from your analysis
# ----------------------------------------------------

code_counts = {
    "Trust, safety and respectful communication": 112,
    "Responsiveness, feedback loops and visible impact": 101,
    "Accessible, inclusive and low-burden feedback channels": 95,
    "Dialogue, co-creation and student agency": 90,
    "Institutional capacity, consistency and structural constraints": 68,
    "Care, belonging and whole-student support": 59,
    "Clear, relevant and actionable feedback": 53
}


# ----------------------------------------------------
# Generate embeddings
# ----------------------------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(
    theme_texts,
    normalize_embeddings=True
)


similarities = cosine_similarity(
    embeddings
)


# ----------------------------------------------------
# Create graph
# ----------------------------------------------------

G = nx.Graph()


# Add nodes

for i, theme in enumerate(theme_names):

    G.add_node(
        theme,
        size=code_counts.get(theme, 1)
    )


# Add edges
# Only include reasonably strong relationships

threshold = 0.55

for i in range(len(theme_names)):

    for j in range(i+1, len(theme_names)):

        score = similarities[i,j]

        if score >= threshold:

            G.add_edge(
                theme_names[i],
                theme_names[j],
                weight=score
            )


# ----------------------------------------------------
# Plot graph
# ----------------------------------------------------

plt.figure(figsize=(14,10))


position = nx.spring_layout(
    G,
    seed=42,
    k=1.5
)


# Node sizes

node_sizes = [
    G.nodes[node]["size"] * 20
    for node in G.nodes()
]


# Edge widths

edge_scores = [
    G[u][v]["weight"]
    for u, v in G.edges()
]

# Find range of observed similarities
min_score = min(edge_scores)
max_score = max(edge_scores)

# Rescale line widths
edge_widths = [
    2 + (
        (score - min_score) /
        (max_score - min_score)
    ) * 8
    for score in edge_scores
]

# Draw nodes

nx.draw_networkx_nodes(
    G,
    position,
    node_size=node_sizes,
    alpha=0.85
)


# Draw edges

nx.draw_networkx_edges(
    G,
    position,
    width=edge_widths,
    alpha=0.5
)


# Draw labels

nx.draw_networkx_labels(
    G,
    position,
    font_size=9
)


plt.title(
    "Semantic Relationship Network of LLM-Generated Themes"
)

plt.axis("off")

plt.tight_layout()

plt.show()