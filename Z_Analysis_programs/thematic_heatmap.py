import matplotlib.pyplot as plt
import seaborn as sns
import json

with open("experiments/theme_generation_document_corpus/output_files/theme_grouping_input.json", "r", encoding="utf-8") as f:
    themes = json.load(f)

subtheme_records = []

for theme in themes:
    for subtheme in theme["subthemes"]:
        subtheme_records.append({
            "theme": theme["theme"],
            "subtheme": subtheme
        })

print(len(subtheme_records))
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

texts = [
    item["subtheme"]
    for item in subtheme_records
]

embeddings = model.encode(
    texts,
    normalize_embeddings=True
)
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(
    embeddings
)
import pandas as pd
import numpy as np

theme_names = [
    theme["theme"]
    for theme in themes
]

theme_similarity = pd.DataFrame(
    index=theme_names,
    columns=theme_names,
    dtype=float
)


for theme_a in theme_names:

    for theme_b in theme_names:

        values = []

        for i, item_a in enumerate(subtheme_records):

            if item_a["theme"] != theme_a:
                continue

            for j, item_b in enumerate(subtheme_records):

                if item_b["theme"] != theme_b:
                    continue

                values.append(
                    similarity_matrix[i,j]
                )

        theme_similarity.loc[
            theme_a,
            theme_b
        ] = np.mean(values)
print(theme_similarity)



plt.figure(figsize=(12,10))

sns.heatmap(
    theme_similarity,
    annot=True,
    fmt=".2f",
    cmap="viridis"
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

plt.show()