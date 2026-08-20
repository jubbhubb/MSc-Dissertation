
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json


def theme_similarity_heatmap(input_file, folder, display=False):
    """
    Generate a semantic similarity heatmap between LLM-generated themes.

    Parameters
    ----------
    input_file : str
        JSON file containing themes.
        Each object should contain:
            - theme
            - description
            - subthemes (optional)

    Returns
    -------
    similarity_df : pandas.DataFrame
        Theme-by-theme cosine similarity matrix.
    """

    # ----------------------------------------------------
    # Load theme data
    # ----------------------------------------------------
    with open(input_file, "r", encoding="utf-8") as f:
        themes = json.load(f)


    # ----------------------------------------------------
    # Prepare text for embedding
    # ----------------------------------------------------
    texts = [
        f"{theme['theme']}\n\n{theme['description']}"
        for theme in themes
    ]


    theme_names = [
        theme["theme"]
        for theme in themes
    ]


    # ----------------------------------------------------
    # Generate embeddings
    # ----------------------------------------------------
    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )


    # ----------------------------------------------------
    # Calculate cosine similarity
    # ----------------------------------------------------
    similarity_matrix = cosine_similarity(
        embeddings
    )


    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=theme_names,
        columns=theme_names
    )


    print("\nTheme similarity matrix:")
    print(similarity_df.round(3))


    # ----------------------------------------------------
    # Print strongest relationships
    # ----------------------------------------------------
    print("\nStrongest theme relationships:")

    relationships = []

    for i in range(len(theme_names)):
        for j in range(i + 1, len(theme_names)):

            relationships.append(
                (
                    theme_names[i],
                    theme_names[j],
                    similarity_matrix[i, j]
                )
            )


    relationships.sort(
        key=lambda x: x[2],
        reverse=True
    )


    for theme_a, theme_b, score in relationships:

        print(
            f"{theme_a} <-> {theme_b}: {score:.3f}"
        )


    # ----------------------------------------------------
    # Plot heatmap
    # ----------------------------------------------------
    plt.figure(figsize=(11, 9))

    sns.heatmap(
        similarity_df,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0,
        vmax=1
    )


    plt.title(
        "Semantic Similarity Between LLM-Generated Themes"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.yticks(
        rotation=0
    )

    plt.tight_layout()

    if display:
        plt.show()

    theme_centrality = similarity_df.apply(
        lambda row: (row.sum()-1)/(len(row)-1)
    )

    with open(folder / "theme_centrality.csv", "w", encoding="utf-8") as f:
        theme_centrality.to_csv(f, header=["centrality"])

    print(theme_centrality.sort_values(ascending=False))

    plt.savefig(folder / "thematic_heatmap.png", dpi=300)

    return similarity_df



# --------------------------------------------------------
# Run analysis
# --------------------------------------------------------

if __name__ == "__main__":
    input_file = "experiments/theme_generation_document_corpus/output_files/theme_grouping_input.json"
    similarity = theme_similarity_heatmap(
        input_file, folder = "", display = True
    )

