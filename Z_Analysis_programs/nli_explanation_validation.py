from pathlib import Path
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def nli_explanation_validation(experiment_folder):
    """
    Validate whether explanations are supported by extracted quotes
    using a Natural Language Inference model.

    Premise:
        quote

    Hypothesis:
        explanation

    Outputs:
        experiment_folder/
            quote_explanation_nli_validation.csv
    """

    # ---------------------------
    # Load NLI model
    # ---------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/bart-large-mnli"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli"
    )

    model.eval()


    json_path = Path(experiment_folder) / "output_files"

    results = []


    # ---------------------------
    # Process JSON files
    # ---------------------------

    for json_file in json_path.glob("*.json"):

        print(f"Processing {json_file.name}")


        with open(
            json_file,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)


        if isinstance(data, dict):
            data = [data]

        for item in data:

            quote = item.get("quote")
            explanation = item.get("explanation")


            if not quote or not explanation:
                continue


            # ---------------------------
            # Cleaning
            # ---------------------------


            quote_clean = quote.lower().strip()

            explanation_clean = explanation.lower().strip()


            # ---------------------------
            # NLI prediction
            # ---------------------------

            inputs = tokenizer(
                explanation_clean,
                quote_clean,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )


            with torch.no_grad():

                outputs = model(
                    **inputs
                )


            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )[0]


            scores = {
                "contradiction": float(probabilities[0]),
                "neutral": float(probabilities[1]),
                "entailment": float(probabilities[2])
            }


            # ---------------------------
            # Store result
            # ---------------------------

            results.append({

                "document": json_file.stem,

                "code": item.get("code"),

                "quote": quote,

                "explanation": explanation,

                "entailment_score":
                    scores["entailment"],

                "neutral_score":
                    scores["neutral"],

                "contradiction_score":
                    scores["contradiction"]

            })


    # ---------------------------
    # Save results
    # ---------------------------

    results_df = pd.DataFrame(results)


    output_path = (
        Path(experiment_folder)
        / "quote_explanation_nli_validation.csv"
    )


    results_df.to_csv(
        output_path,
        index=False
    )


    print(results_df)


    return results_df



def calculate_nli_success_rate(results_file):

    df = pd.read_csv(results_file)


    return {

        "average_entailment":
            df["entailment_score"].mean(),

        "average_neutral":
            df["neutral_score"].mean(),

        "average_contradiction":
            df["contradiction_score"].mean(),

        "high_confidence_entailments":
            (
                df["entailment_score"] > 0.8
            ).mean(),

        "total_quotes":
            len(df)

    }