from pathlib import Path
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import Z_helper_functions.preprocessing as preprocessing


results = []

def cosine_similarity_between_documents(experiment_folder):
    """
    Calculate the cosine similarity between two embeddings.

    Parameters:
        original_embedding: The embedding of the original document.
        processed_embedding: The embedding of the processed document.

    Returns:
        Cosine similarity score.
    """

    model = SentenceTransformer("all-MiniLM-L6-v2")
    txt_path = Path(experiment_folder + "/input_files")
    json_path = Path(experiment_folder + "/output_files")

    for txt_file in txt_path.glob("*.txt"):

        document_name = txt_file.stem.replace("_recombined", "")
        print(f"Processing {document_name}...")
        json_file = json_path / f"{document_name}.json"


        if not json_file.exists():
            print(f"Missing JSON for {txt_file.name}")
            continue

        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()

        processed_text = preprocessing.normalise_text(
            text,
            lowercase=True
        )
        original_embedding = model.encode([processed_text])
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        processed_doc = [item["quote"] for item in data]
        processed_doc_text = " ".join(processed_doc)

        processed_embedding = model.encode([processed_doc_text])
        similarity = cosine_similarity(
            original_embedding,
            processed_embedding
        )[0][0]

        results.append({
            "document": txt_file.stem,
            "similarity": similarity,
            "original_items": 1,
            "processed_items": len(processed_doc)
        })

    results_df = pd.DataFrame(results)

    print(results_df)
    csv_path = Path(experiment_folder) / "similarity_results.csv"

    results_df.to_csv(csv_path, index=False)