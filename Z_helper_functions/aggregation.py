from pathlib import Path
from collections import defaultdict
import json
import regex as re
import yaml

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

def combine_json_files(input_path, output_file):
    """
    Combines multiple JSON files into a single JSON file.

    Parameters:
        input_path (str or Path):
            Folder containing JSON files to combine

        output_file (str or Path):
            Path to save the combined JSON file
    """

    input_path = Path(input_path)
    combined_data = []

    print(f"Combining JSON files from {input_path} into {output_file}...")

    for file in input_path.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            combined_data.extend(data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)

def combine_json_by_document(input_folder, output_folder):
    """
    Combine section JSON files into one JSON per original document and
    generate a YAML metadata file.

    Expected filename format:
        <document_name>_section_<number>.json

    Example:
        Transcription_Group 1_S1_CZ&BI&AJ_section_1.json
        Transcription_Group 1_S1_CZ&BI&AJ_section_2.json
        ...

    Parameters
    ----------
    input_folder : str or Path
        Folder containing section JSON files.

    output_folder : str or Path
        Folder where combined JSON and YAML files will be written.
    """

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Capture:
    #   document name
    #   section number
    pattern = re.compile(r"^(.*?)_section_(\d+)\.json$")

    grouped_files = defaultdict(list)
    for file in input_folder.glob("*.json"):

        match = pattern.match(file.name)

        if not match:
            print(f"Skipping unexpected filename: {file.name}")
            continue

        document_name = match.group(1)
        section_number = int(match.group(2))

        grouped_files[document_name].append((section_number, file))

    for document_name, files in grouped_files.items():

        # Sort by section number
        files.sort(key=lambda x: x[0])

        combined_data = []
        section_metadata = []

        total_items = 0
        current_index = 0

        for section_number, file in files:

            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError(
                    f"{file.name} does not contain a JSON list."
                )

            num_items = len(data)

            combined_data.extend(data)

            section_metadata.append({
                "section": section_number,
                "file": file.name,
                "items": num_items,
                "start_item": current_index,
                "end_item": current_index + num_items - 1 if num_items else current_index
            })

            current_index += num_items
            total_items += num_items

       
        json_output = output_folder / f"{document_name}.json"

        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, indent=4, ensure_ascii=False)

        metadata = {
            "document": document_name,
            "number_of_sections": len(files),
            "total_items": total_items,
            "sections": section_metadata
        }

        yaml_output = output_folder / f"{document_name}.yaml"

        with open(yaml_output, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                metadata,
                f,
                sort_keys=False,
                allow_unicode=True
            )

        print(
            f"Saved {json_output.name} ({total_items} items) "
            f"and {yaml_output.name}"
        )