from pathlib import Path
from collections import defaultdict

def recombine_split_documents(input_path, output_path=None, save_combined=False):
    """
    Recombines split document sections back into full documents.

    Parameters:
        input_path (str or Path):
            Folder containing split documents

        output_path (str or Path, optional):
            Where to save recombined documents (if None, not saved)

        save_combined (bool):
            Whether to write recombined docs to disk

    Returns:
        dict:
            {doc_id: full_text}
    """

    input_path = Path(input_path)

    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)

    # -------------------------
    # 1. Group by document ID
    # -------------------------
    for file in input_path.glob("*.txt"):
        stem = file.stem

        if "_section_" in stem:
            doc_id = stem.split("_section_")[0]
        else:
            doc_id = stem

        grouped[doc_id].append(file)

    # -------------------------
    # 2. Sort sections properly
    # -------------------------
    def sort_key(path):
        name = path.stem
        if "_section_" in name:
            return int(name.split("_section_")[-1])
        return 0

    # -------------------------
    # 3. Recombine
    # -------------------------
    reconstructed = {}

    for doc_id, files in grouped.items():
        files = sorted(files, key=sort_key)

        parts = []
        for f in files:
            with open(f, "r", encoding="utf-8") as infile:
                parts.append(infile.read().strip())

        full_text = "\n".join(parts)
        reconstructed[doc_id] = full_text

        # -------------------------
        # 4. Optional save to disk
        # -------------------------
        if save_combined and output_path:
            out_file = output_path / f"{doc_id}_recombined.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(full_text)

    return reconstructed

def combine_to_corpus(
    input_path,
    save_path=None,
    filename="corpus.txt",
    include_headers=True
):
    """
    Combines recombined document files into a corpus.

    Expects files like:
        test1.txt
        test2.txt
        OR test1_recombined.txt etc.
    """

    input_path = Path(input_path)

    documents = {}

    # -------------------------
    # 1. Load files
    # -------------------------
    for file in input_path.glob("*.txt"):
        doc_id = file.stem.replace("_recombined", "")
        with open(file, "r", encoding="utf-8") as f:
            documents[doc_id] = f.read()

    # -------------------------
    # 2. Build corpus
    # -------------------------
    corpus_parts = []

    for doc_id, text in documents.items():
        if include_headers:
            corpus_parts.append(f"### {doc_id}\n{text}")
        else:
            corpus_parts.append(text)

    corpus_text = "\n\n".join(corpus_parts)

    # -------------------------
    # 3. Save if needed
    # -------------------------
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / filename, "w", encoding="utf-8") as f:
            f.write(corpus_text)

    return corpus_text