import json
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util

from sklearn.metrics import roc_auc_score, roc_curve
def random_explanation_baseline(experiment_folder, test_name):
    """
    Generate a random baseline for semantic similarity
    between quotes and explanations.

    Outputs:
        experiments/{test_name}/statistics/
            semantic_random_baseline.csv
            semantic_random_baseline.png
            semantic_random_baseline_stats.txt
    """
    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    N_SHUFFLES = 100

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    print("Loading model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # --------------------------------------------------
    # Load JSON
    # --------------------------------------------------

    print(f"Loading JSON...")

    JSON_FOLDER = f"{experiment_folder}/output_files"

    folder_path = f"{experiment_folder}/statistics"

    os.makedirs(folder_path, exist_ok=True)

    output_folder = f"{experiment_folder}/statistics"

    quotes = []
    explanations = []

    for filename in os.listdir(JSON_FOLDER):

        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(JSON_FOLDER, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Assuming each file contains a list of quote objects
        for item in data:

            quote = item.get("quote")
            explanation = item.get("explanation")

            if quote and explanation:
                quotes.append(quote)
                explanations.append(explanation)

    print(f"Loaded {len(quotes)} quote/explanation pairs "
        f"from {len(os.listdir(JSON_FOLDER))} files.")

    # --------------------------------------------------
    # Encode once
    # --------------------------------------------------

    print("Encoding quotes...")

    quote_embeddings = model.encode(
        quotes,
        convert_to_tensor=True,
        show_progress_bar=True
    )

    print("Encoding explanations...")

    explanation_embeddings = model.encode(
        explanations,
        convert_to_tensor=True,
        show_progress_bar=True
    )

    # --------------------------------------------------
    # Genuine similarities
    # --------------------------------------------------

    print("Computing genuine similarities...")

    true_scores = []

    for q, e in zip(quote_embeddings, explanation_embeddings):
        true_scores.append(util.cos_sim(q, e).item())

    # --------------------------------------------------
    # Random baseline
    # --------------------------------------------------

    print("Generating random baseline...")

    random_scores = []

    n = len(quotes)

    for seed in range(N_SHUFFLES):

        random.seed(seed)

        indices = list(range(n))

        # Shuffle until no quote is matched to its own explanation
        while True:
            random.shuffle(indices)

            if all(i != indices[i] for i in range(n)):
                break

        for i in range(n):

            score = util.cos_sim(
                quote_embeddings[i],
                explanation_embeddings[indices[i]]
            ).item()

            random_scores.append(score)

    print(f"Computed {len(random_scores):,} random similarities.")

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    df_true = pd.DataFrame({
        "score": true_scores,
        "type": "True"
    })

    df_random = pd.DataFrame({
        "score": random_scores,
        "type": "Random"
    })

    results = pd.concat([df_true, df_random], ignore_index=True)

    results.to_csv(output_folder + "/semantic_random_baseline.csv", index=False)

    print("Saved semantic_random_baseline.csv")

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    plt.figure(figsize=(9,6))

    plt.hist(
        true_scores,
        bins=30,
        density=True,
        alpha=0.6,
        label="True Quote–Explanation Pairs"
    )

    plt.hist(
        random_scores,
        bins=30,
        density=True,
        alpha=0.6,
        label="Random Pairings"
    )

    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.title("Semantic Similarity Distribution")

    plt.legend()

    plt.tight_layout()

    plt.savefig(output_folder + "/semantic_random_baseline.png", dpi=300)

    plt.show()

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------
    with open(output_folder + "/semantic_random_baseline_stats.txt", "w") as f:
        f.write("Statistics\n")
        f.write("----------------------------\n")
        f.write(f"True Mean   : {np.mean(true_scores):.3f}\n")
        f.write(f"Random Mean : {np.mean(random_scores):.3f}\n\n")

        f.write(f"True Std    : {np.std(true_scores):.3f}\n")
        f.write(f"Random Std  : {np.std(random_scores):.3f}\n\n")

        true_mean = np.mean(true_scores)
        random_mean = np.mean(random_scores)

        true_std = np.std(true_scores, ddof=1)
        random_std = np.std(random_scores, ddof=1)

        pooled_std = np.sqrt(
            (true_std**2 + random_std**2) / 2
        )

        cohens_d = (true_mean - random_mean) / pooled_std

        f.write(f"Cohen's d: {cohens_d:.3f}\n\n")

        # Load results
        df = pd.read_csv(output_folder + "/semantic_random_baseline.csv")

        # Labels: genuine = 1, random = 0
        labels = (df["type"] == "True").astype(int)

        # Similarity score is the predictor
        scores = df["score"]

        # Calculate AUC
        auc = roc_auc_score(labels, scores)

        f.write(f"Semantic similarity AUC: {auc:.3f}\n")

    