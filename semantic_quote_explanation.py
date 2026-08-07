from pathlib import Path
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def semantic_explanation_validation(experiment_folder):
    """
    Compare semantic similarity between extracted quotes
    and their generated explanations.

    Measures whether the explanation captures the meaning
    of the quote.

    Outputs:
        experiment_folder/
            quote_explanation_semantic_validation.csv
    """

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    json_path = Path(experiment_folder) / "output_files"

    results = []


    for json_file in json_path.glob("*.json"):

        print(f"Processing {json_file.name}")

        with open(
            json_file,
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


            quote_embedding = model.encode(
                [quote]
            )

            explanation_embedding = model.encode(
                [explanation]
            )


            similarity = cosine_similarity(
                quote_embedding,
                explanation_embedding
            )[0][0]


            results.append({

                "document": json_file.stem,

                "code": item.get("code"),

                "quote": quote,

                "explanation": explanation,

                "semantic_similarity":
                    float(similarity)

            })


    results_df = pd.DataFrame(results)


    output_path = (
        Path(experiment_folder)
        / "quote_explanation_semantic_validation.csv"
    )


    results_df.to_csv(
        output_path,
        index=False
    )


    print(results_df)

    return results_df

def calculate_explanation_semantic_score(results_file):

    df = pd.read_csv(results_file)

    overall = df["semantic_similarity"].mean()

    per_document = (
        df.groupby("document")
        ["semantic_similarity"]
        .mean()
        .to_dict()
    )

    return {
        "overall_average": overall,
        "per_document": per_document,
        "total_quotes": len(df)
    }