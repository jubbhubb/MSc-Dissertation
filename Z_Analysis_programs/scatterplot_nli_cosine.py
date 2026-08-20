import pandas as pd

filepath = f"experiments/coding_document"
semantic = pd.read_csv(f"{filepath}/quote_explanation_semantic_validation.csv")
nli = pd.read_csv(f"{filepath}/quote_explanation_nli_validation.csv")

data = semantic.merge(
    nli,
    on="quote",
    how="inner"
)

print(data.head())
print(len(data))

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

plt.scatter(
    data["semantic_similarity"],
    data["neutral_score"],
    alpha=0.4
)

plt.xlabel("Semantic similarity score")
plt.ylabel("NLI entailment probability")
plt.title("Semantic Similarity vs NLI Entailment")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("similarity_vs_nli.png", dpi=300)
plt.show()