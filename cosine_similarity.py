from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pathlib import Path
import preprocessing


model = SentenceTransformer("all-MiniLM-L6-v2")

# original = "Your original document..."
# processed = "Your processed document..."

# embeddings = model.encode([original, processed])

# # similarity = cosine_similarity(
# #     [embeddings[0]],
# #     [embeddings[1]]
# # )[0][0]

# print(f"Similarity: {similarity:.4f}")

def read_in_json(file_path):
    return pd.read_json(file_path)

dataframe = read_in_json("experiments/testing_006/output_files/test2.json")

source_folder="source_files"
source_path = Path(source_folder)
original_embeddings = []
for file in source_path.glob("*.txt"):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
    processed_text = preprocessing.normalise_text(
        text,
        lowercase=True
    )
    original_embeddings.append(model.encode([text]))

processed_doc = dataframe["quote"].tolist()

processed_doc_text = " ".join(processed_doc)

processed_embedding = model.encode([processed_doc_text])

similarity = cosine_similarity(
    [original_embeddings[1][0]],
    [processed_embedding[0]]
)[0][0]

print(f"Similarity: {similarity:.4f}")
    


