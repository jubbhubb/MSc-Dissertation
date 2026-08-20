import json
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def thematic_mindmap(input_file, folder, display = True):
    # ----------------------------------------------------
    # Load themes
    # ----------------------------------------------------

    with open(input_file, "r", encoding="utf-8") as f:
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

    for t in themes:
        t["code_count"] = len(t.get("subthemes", []))


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

    for theme in themes:
        G.add_node(
            theme["theme"],
            size=theme.get("code_count", 1)
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

    if display:
        plt.show()

    plt.savefig(folder / "thematic_mindmap.png", dpi=300)