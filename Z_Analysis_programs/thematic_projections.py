from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
import umap
import json
import pandas as pd
import matplotlib.pyplot as plt


def thematic_projections_method(code_file, grouping_file, folder, display=True):
    """
    Generate a semantic projection of LLM-generated codes, coloured by
    overarching themes.

    Parameters
    ----------
    code_file : str
        JSON file containing the original codes. Each item should contain:
            - code
            - explanation
            - confidence
            - theme (this is the SUBTHEME)

    grouping_file : str
        JSON file containing the grouped themes. Each item should contain:
            - theme
            - description
            - subthemes

    Returns
    -------
    coords : ndarray
        2D UMAP coordinates.

    score : float
        Silhouette score using the overarching themes.
    """

    # ----------------------------------------------------
    # Load the original coded data
    # ----------------------------------------------------
    with open(code_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ----------------------------------------------------
    # Load the grouped themes
    # ----------------------------------------------------
    with open(grouping_file, "r", encoding="utf-8") as f:
        grouped_themes = json.load(f)

    # ----------------------------------------------------
    # Create a lookup:
    # subtheme -> overarching theme
    # ----------------------------------------------------
    subtheme_to_theme = {}

    for theme in grouped_themes:
        for subtheme in theme["subthemes"]:
            subtheme_to_theme[subtheme] = theme["theme"]

    # ----------------------------------------------------
    # Attach the overarching theme to each code
    # ----------------------------------------------------
    missing = []

    for item in data:
        item["overarching_theme"] = subtheme_to_theme.get(
            item["theme"],
            "Unassigned"
        )

        if item["overarching_theme"] == "Unassigned":
            missing.append(item["theme"])

    print(
        f"Mapped {len(data) - len(missing)}/{len(data)} codes successfully."
    )

    if missing:
        print("\nSubthemes that could not be matched:")
        print(sorted(set(missing)))

    # ----------------------------------------------------
    # Prepare text for embedding
    # ----------------------------------------------------
    # for item in data:
    #     print("ITEM:", item)
    #     print("ITEM KEYS:", item.keys())
    #     text = f"{item['theme']}\n\n{item['description']}"

    # for item in data:
    #     print("ITEM KEYS:", item.keys())

    # texts = [
    #     f"{item['theme']}\n\n{item['explanation']}"
    #     for item in data
    # ]
    texts = []
    for i, item in enumerate(data):
        if "theme" not in item or "explanation" not in item:
            print("BAD ITEM INDEX:", i)
            print("BAD ITEM:", item)
            print("BAD ITEM KEYS:", item.keys())
            raise ValueError("Found item without theme or explanation")

        texts.append(f"{item['theme']}\n\n{item['explanation']}")

    labels = [
        item["overarching_theme"]
        for item in data
    ]

    # ----------------------------------------------------
    # Generate embeddings
    # ----------------------------------------------------
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    # ----------------------------------------------------
    # UMAP projection
    # ----------------------------------------------------
    reducer = umap.UMAP(
        n_neighbors=10,
        min_dist=0.2,
        metric="cosine",
        random_state=42
    )

    coords = reducer.fit_transform(embeddings)

    # ----------------------------------------------------
    # Create plotting dataframe
    # ----------------------------------------------------
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "theme": [
            item["overarching_theme"]
            for item in data
        ],
        "subtheme": [
            item["theme"]
            for item in data
        ],
        "code": [
            item["code"]
            for item in data
        ],
        "confidence": [
            item["confidence"]
            for item in data
        ]
    })

    print(df.head())

    # ----------------------------------------------------
    # Plot
    # ----------------------------------------------------
    plt.figure(figsize=(12, 9))

    # for subtheme in sorted(df["subtheme"].unique()):

    #     subset = df[df["subtheme"] == subtheme]

    #     plt.scatter(
    #         subset["x"],
    #         subset["y"],
    #         label=subtheme,
    #         s=80,
    #         alpha=0.8
    #     )
    for theme in sorted(df["theme"].unique()):

        subset = df[df["theme"] == theme]

        plt.scatter(
            subset["x"],
            subset["y"],
            label=theme,
            s=80,
            alpha=0.8
        )

    plt.title("Semantic Projection of LLM-Generated Themes")

    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()
    if display:
        plt.show()

    # ----------------------------------------------------
    # Evaluate clustering
    # ----------------------------------------------------
    score = silhouette_score(
        embeddings,
        labels,
        metric="cosine"
    )
    with open(folder / "silhouette_score.txt", "w") as f:
        f.write(f"Silhouette score: {score:.3f}\n")
        f.write("Theme counts:\n")
        f.write(df["theme"].value_counts().to_string())
    print(f"Filename: {folder / 'silhouette_score.txt'}")
    print(f"Silhouette score: {score:.3f}")
    print(df["theme"].value_counts())
    return coords, score