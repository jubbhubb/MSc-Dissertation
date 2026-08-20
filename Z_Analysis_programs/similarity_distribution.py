import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Read the data
def similarity_histogram(experiment_name):

    df = pd.read_csv(f"{experiment_name}/quote_explanation_semantic_validation.csv")

    scores = df["semantic_similarity"].dropna()

    # Histogram
    plt.figure(figsize=(8, 5))
    counts, bins, _ = plt.hist(
        scores,
        bins=50,
        density=True,
        alpha=0.5,
        edgecolor="black",
        label="Histogram"
    )

    # KDE
    x = np.linspace(0, 1, 500)
    kde = gaussian_kde(scores)
    plt.plot(x, kde(x), linewidth=2, label="KDE")

    # Example thresholds
    for t in [0.5, 0.6, 0.7, 0.8]:
        plt.axvline(t, linestyle="--", alpha=0.6)

    plt.xlabel("Semantic Similarity")
    plt.ylabel("Density")
    plt.title("Distribution of Semantic Similarity Scores")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{experiment_name}/similarity_distribution.pdf", dpi=300)
    # plt.show()
