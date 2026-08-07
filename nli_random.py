from pathlib import Path
import json
import random

import pandas as pd
import matplotlib.pyplot as plt
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

def nli_random_baseline(experiment_folder, test_name):
    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    JSON_FOLDER = Path(f"{experiment_folder}/output_files")

    output_folder = f"{experiment_folder}/statistics"

    N_SHUFFLES = 1

    MODEL_NAME = "facebook/bart-large-mnli"


    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    print("Loading NLI model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    model.eval()


    # --------------------------------------------------
    # Load all quote/explanation pairs
    # --------------------------------------------------

    print("Loading JSON files...")

    records = []

    for json_file in JSON_FOLDER.glob("*.json"):

        print(f"Reading {json_file.name}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        for item in data:

            quote = item.get("quote")
            explanation = item.get("explanation")

            if quote and explanation:

                records.append({
                    "document": json_file.stem,
                    "code": item.get("code"),
                    "quote": quote,
                    "explanation": explanation
                })

    print(f"\nLoaded {len(records)} quote/explanation pairs.")


    # --------------------------------------------------
    # NLI helper
    # --------------------------------------------------

    def run_nli(quote, explanation):

        inputs = tokenizer(
            quote.lower().strip(),          # Premise
            explanation.lower().strip(),    # Hypothesis
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0]

        return {
            "contradiction": float(probs[0]),
            "neutral": float(probs[1]),
            "entailment": float(probs[2])
        }


    # --------------------------------------------------
    # Genuine pairs
    # --------------------------------------------------

    print("\nRunning genuine pair evaluation...")

    true_results = []

    for record in records:

        scores = run_nli(
            record["quote"],
            record["explanation"]
        )

        true_results.append({

            "type": "True",

            "document": record["document"],

            "code": record["code"],

            "entailment": scores["entailment"],

            "neutral": scores["neutral"],

            "contradiction": scores["contradiction"]

        })


    # --------------------------------------------------
    # Random baseline
    # --------------------------------------------------

    print("\nRunning random baseline...")

    random_results = []

    n = len(records)

    for seed in range(N_SHUFFLES):

        random.seed(seed)

        indices = list(range(n))

        # Create a derangement
        while True:

            random.shuffle(indices)

            if all(i != indices[i] for i in range(n)):
                break

        # Uncomment if you want to verify the shuffle
        #
        # for i in range(5):
        #     print("=" * 60)
        #     print(records[i]["quote"])
        #     print()
        #     print(records[indices[i]]["explanation"])

        for i in range(n):

            scores = run_nli(
                records[i]["quote"],
                records[indices[i]]["explanation"]
            )

            random_results.append({

                "type": "Random",

                "document": records[i]["document"],

                "code": records[i]["code"],

                "entailment": scores["entailment"],

                "neutral": scores["neutral"],

                "contradiction": scores["contradiction"]

            })


    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    results = pd.DataFrame(true_results + random_results)

    results.to_csv(
        output_folder + "/nli_random_baseline.csv",
        index=False
    )

    print("\nSaved nli_random_baseline.csv")


    # --------------------------------------------------
    # Summary statistics
    # --------------------------------------------------

    with open(JSON_FOLDER / "nli_random_baseline_stats.txt", "w") as f:
        print("\nSummary Statistics", file=f)
        print("-" * 40, file=f)

        for group in ["True", "Random"]:
            subset = results[
                results["type"] == group
            ]
            print(f"\n{group}", file=f)
            print(
                subset[
                    [
                        "entailment",
                        "neutral",
                        "contradiction"
                    ]
                ].mean(),
                file=f
            )

    # --------------------------------------------------
    # Plot entailment distributions
    # --------------------------------------------------

    plt.figure(figsize=(8,5))

    for label in ["True", "Random"]:

        subset = results[
            results["type"] == label
        ]

        plt.hist(
            subset["entailment"],
            bins=30,
            density=True,
            alpha=0.5,
            label=label
        )

    plt.xlabel("Entailment Probability")
    plt.ylabel("Density")
    plt.title("NLI Entailment Distribution")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_folder + "/nli_random_baseline.png",
        dpi=300
    )

    plt.show()