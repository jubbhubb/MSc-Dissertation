import pandas as pd
from sklearn.metrics import roc_auc_score

def random_semantic_calc(experiment_folder, test_name):
    # Load semantic results
    df = pd.read_csv(f"{experiment_folder}/{test_name}/statistics/semantic_random_baseline.csv")

    # Genuine pairs = 1, random pairs = 0
    labels = (df["type"] == "True").astype(int)

    # Semantic similarity score
    scores = df["score"]

    # Calculate AUC
    auc = roc_auc_score(
        labels,
        scores
    )
    with open(f"{experiment_folder}/{test_name}/statistics/semantic_random_baseline_stats.txt", "a") as f:
        f.write(f"AUC: {auc:.3f}\n")