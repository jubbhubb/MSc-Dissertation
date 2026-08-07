from pathlib import Path
import json
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import preprocessing


def semantic_quote_validation(
    experiment_folder,
    chunk_method="words",
    window_size=40,
    step_size=10,
    adaptive=False
):
    """
    Validate whether extracted quotes are semantically supported
    by the source transcript.

    Parameters
    ----------
    chunk_method : str
        "words" or "sentences"

    window_size : int
        Size of each chunk.
            - words if chunk_method="words"
            - sentences if chunk_method="sentences"

    step_size : int
        Number of words/sentences to move between chunks.

    adaptive : bool
        If True, window size is based on quote length.
    """

    model = SentenceTransformer("all-MiniLM-L6-v2")

    txt_path = Path(experiment_folder) / "input_files"
    json_path = Path(experiment_folder) / "output_files"

    results = []

    # -------------------------------------------------
    # Sentence chunking
    # -------------------------------------------------

    def split_into_sentence_chunks(
        text,
        window_size,
        overlap
    ):

        sentences = re.split(
            r'(?<=[.!?])\s+',
            text
        )

        sentences = [
            s.strip()
            for s in sentences
            if s.strip()
        ]

        if not sentences:
            return [text]

        chunks = []

        step = max(1, window_size - overlap)

        for start in range(
            0,
            len(sentences),
            step
        ):

            chunk = " ".join(
                sentences[
                    start:start + window_size
                ]
            )

            if chunk:
                chunks.append(chunk)

            if start + window_size >= len(sentences):
                break

        return chunks

    # -------------------------------------------------
    # Word chunking
    # -------------------------------------------------

    def split_into_word_chunks(
        text,
        window_size,
        step_size
    ):

        words = text.split()

        if len(words) <= window_size:
            return [" ".join(words)]

        chunks = []

        for start in range(
            0,
            len(words) - window_size + 1,
            step_size
        ):

            chunk = " ".join(
                words[
                    start:start + window_size
                ]
            )

            chunks.append(chunk)

        # Always include document ending
        final_chunk = " ".join(
            words[-window_size:]
        )

        if final_chunk not in chunks:
            chunks.append(final_chunk)

        return chunks

    # -------------------------------------------------

    for txt_file in txt_path.glob("*.txt"):

        document_name = txt_file.stem

        print(f"Processing {document_name}...")

        json_file = json_path / f"{document_name}.json"

        if not json_file.exists():
            print(f"Missing JSON for {txt_file.name}")
            continue

        with open(txt_file, encoding="utf-8") as f:
            text = f.read()

        processed_text = preprocessing.normalise_text(
            text,
            lowercase=True
        )

        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        # ---------------------------------------------
        # Non-adaptive mode
        # ---------------------------------------------

        if not adaptive:

            if chunk_method == "words":

                chunks = split_into_word_chunks(
                    processed_text,
                    window_size,
                    step_size
                )

            else:

                overlap = max(
                    0,
                    window_size - step_size
                )

                chunks = split_into_sentence_chunks(
                    processed_text,
                    window_size,
                    overlap
                )

            chunk_embeddings = model.encode(chunks)

        # ---------------------------------------------
        # Compare every quote
        # ---------------------------------------------

        for item in data:

            quote = item.get("quote")

            if not quote:
                continue

            # Adaptive chunk sizing
            if adaptive:

                quote_length = len(
                    quote.split()
                )

                adaptive_window = max(
                    30,
                    quote_length * 2
                )

                chunks = split_into_word_chunks(
                    processed_text,
                    adaptive_window,
                    step_size
                )

                chunk_embeddings = model.encode(
                    chunks
                )

            quote_embedding = model.encode(
                [quote]
            )

            similarities = cosine_similarity(
                quote_embedding,
                chunk_embeddings
            )[0]

            best_index = similarities.argmax()

            results.append({

                "document": document_name,

                "code": item.get("code"),

                "quote": quote,

                "semantic_similarity":
                    float(similarities[best_index]),

                "matching_chunk":
                    chunks[best_index],

                "chunk_number":
                    int(best_index),

                "total_chunks":
                    len(chunks)

            })

    results_df = pd.DataFrame(results)

    csv_path = (
        Path(experiment_folder)
        / "quote_semantic_validation.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    print(results_df)

    return results_df
def calculate_semantic_success_rate(results_file):
    """
    Calculate average semantic similarity from quote validation results.

    Parameters:
        results_file:
            Path to quote_semantic_validation.csv

    Returns:
        Dictionary containing overall and per-document averages.
    """

    df = pd.read_csv(results_file)

    overall_average = df["semantic_similarity"].mean()

    per_document = (
        df.groupby("document")["semantic_similarity"]
        .mean()
        .to_dict()
    )

    return {
        "overall_average": overall_average,
        "per_document": per_document,
        "total_quotes": len(df)
    }