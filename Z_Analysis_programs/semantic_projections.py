from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import umap
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

def semantic_projections(input_file, folder, display=True):
    """
    Generate semantic projections of code and explanations.

    Parameters
    ----------
    data : list of dict
        Each dict should contain 'code', 'explanation', and 'theme' keys.

    Returns
    -------
    coords : np.ndarray
        2D coordinates of the embeddings after UMAP projection.
    
    score : float
        Silhouette score of the clustering based on themes.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        texts = [
            f"{item['code']}\n\n{item['explanation']}"
            for item in data
                ]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    labels = [item["theme"] for item in data]

    embeddings = model.encode(texts)

    reducer = umap.UMAP(
        n_neighbors=10,
        min_dist=0.2,
        metric="cosine",
        random_state=42
    )

    coords = reducer.fit_transform(embeddings)

    df = pd.DataFrame({
        "x": coords[:,0],
        "y": coords[:,1],
        "theme": [item["theme"] for item in data],
        "code": [item["code"] for item in data],
        "confidence": [item["confidence"] for item in data]
    })

    print(df.head())

    plt.figure(figsize=(10,8))

    for theme in df["theme"].unique():

        subset = df[df["theme"] == theme]

        plt.scatter(
            subset["x"],
            subset["y"],
            label=theme,
            s=80,
            alpha=0.8
        )

    plt.legend(
        bbox_to_anchor=(1.05,1),
        loc="upper left"
    )

    plt.title(
        "Semantic Projection of LLM-Generated Themes"
    )

    plt.xlabel("UMAP dimension 1")
    plt.ylabel("UMAP dimension 2")

    plt.tight_layout()
    if display:
        plt.show()

    plt.savefig(folder / "semantic_projection.png", dpi=300)
    with open(folder / "semantic_projection_data.csv", "w", encoding="utf-8") as f:
        df.to_csv(f, index=False)
    score = silhouette_score(embeddings, labels, metric="cosine")
    with open(folder / "silhouette_score.txt", "w") as f:
        f.write(f"Silhouette score: {score:.3f}\n")
        f.write("Theme counts:\n")
        f.write(df["theme"].value_counts().to_string())
    print(f"Filename: {folder / 'silhouette_score.txt'}")