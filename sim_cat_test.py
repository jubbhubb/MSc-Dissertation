from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# from pathlib import Path
import matplotlib.pyplot as plt

# =====================================
# Enter your sentences here
# =====================================
sentences = [
    "The cat sat on the mat.",
    "A cat rested on a rug.",
    "The feline lay on the carpet.",
    "The dog lay on the carpet.",
    "The dog played outside.",
    "The weather is nice today."
]

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(sentences)

# Compute pairwise cosine similarities
similarity_matrix = cosine_similarity(embeddings)

# Print the matrix
plt.figure(figsize=(7,6))

plt.imshow(similarity_matrix, cmap="viridis", vmin=0, vmax=1)

plt.colorbar(label="Cosine Similarity")

plt.xticks(range(len(sentences)),
           [f"S{i+1}" for i in range(len(sentences))],
           rotation=45)

plt.yticks(range(len(sentences)),
           [f"S{i+1}" for i in range(len(sentences))])

plt.title("Sentence Similarity Heatmap")

# Show the numerical values
for i in range(len(sentences)):
    for j in range(len(sentences)):
        plt.text(j, i,
                 f"{similarity_matrix[i,j]:.2f}",
                 ha="center",
                 va="center",
                 color="white")

plt.tight_layout()
plt.show()