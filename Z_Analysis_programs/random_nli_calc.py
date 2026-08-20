import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np

def random_nli_calc(experiment_folder, test_name):
    """
    Calculate AUC and Cohen's d for the NLI random baseline.
    """
    df = pd.read_csv(f"{experiment_folder}/{test_name}/statistics/nli_random_baseline.csv")

    labels = (df["type"] == "True").astype(int)

    auc = roc_auc_score(labels, df["entailment"])
    
    with open(f"{experiment_folder}/{test_name}/statistics/nli_random_baseline_stats.txt", "w") as f:
        f.write(f"AUC: {auc:.3f}\n")

    df = pd.read_csv(f"{experiment_folder}/{test_name}/statistics/nli_random_baseline.csv")

    true = df[df["type"] == "True"]["entailment"]
    random = df[df["type"] == "Random"]["entailment"]

    pooled_std = np.sqrt(
        (
            true.std(ddof=1)**2 +
            random.std(ddof=1)**2
        ) / 2
    )

    cohens_d = (
        true.mean() - random.mean()
    ) / pooled_std
    with open(f"{experiment_folder}/{test_name}/statistics/nli_random_baseline_stats.txt", "a") as f:
        f.write(f"Cohen's d: {cohens_d:.3f}\n")
    