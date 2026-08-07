import json
from pathlib import Path
import time

from enum import Enum

import aggregation
from api_calls import individual_input
import save_response

class Granularity(Enum):
    SECTION="section"
    DOCUMENT="document"
    CORPUS="corpus"

class Reasoning(Enum):
    INDUCTIVE="inductive"
    DEDUCTIVE="deductive"


def chunk_list(items, chunk_size=10):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def theme_generation_pipeline_v2(stage, previous_granularity, new_granularity, reasoning, testname, experiments_folder="experiments_v2"):

    print(f"Starting theme generation pipeline for stage: {stage}, previous granularity: {previous_granularity.value}, new granularity: {new_granularity.value}")

    schema = {
        "type": "object",
        "properties": {
            "codes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "explanation": {"type": "string"},
                        "quote": {"type": "string"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "theme": {"type": "string"}
                    },
                    "required": [
                        "code",
                        "explanation",
                        "quote",
                        "confidence",
                        "theme"
                    ],
                    "additionalProperties": False
                }
            }
        },
        "required": ["codes"],
        "additionalProperties": False
    }

    with open("prompts/theme_generation_prompt_induction.txt", "r", encoding="utf-8") as f:
        prompt = f.read()

    experiment_folder = Path(f"{experiments_folder}/{testname}")
    experiment_folder.mkdir(parents=True, exist_ok=True)

    log_folder = experiment_folder / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)

    log_file = log_folder / "run_log.txt"

    print(f"Experiment name: {testname}")

    coding_folder = Path(f"{experiments_folder}/coding_{previous_granularity.value}_{reasoning.value}")

    # Determine input location

    if new_granularity == Granularity.SECTION:

        if previous_granularity == Granularity.DOCUMENT:
            input_folder = coding_folder / "output_files"

        elif previous_granularity == Granularity.SECTION:
            input_folder = coding_folder / "combined_json"

        elif previous_granularity == Granularity.CORPUS:
            input_folder = coding_folder / "output_files"


    elif new_granularity == Granularity.DOCUMENT:

        if previous_granularity == Granularity.SECTION:
            input_folder = coding_folder / "combined_json"

        elif previous_granularity == Granularity.DOCUMENT:
            input_folder = coding_folder / "output_files"

        elif previous_granularity == Granularity.CORPUS:
            raise ValueError("Corpus to document is not supported")


    elif new_granularity == Granularity.CORPUS:

        input_folder = experiment_folder / "input_files"
        input_folder.mkdir(parents=True, exist_ok=True)

        aggregation.combine_json_files(input_path=coding_folder / "output_files", output_file=input_folder / "combined.json")


    else:
        raise ValueError(f"Unsupported granularity: {new_granularity}")


    print(f"Input folder: {input_folder}")


    with open(log_file, "a", encoding="utf-8") as log:

        for file in input_folder.glob("*.json"):

            print(f"Processing file: {file.name}")

            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if new_granularity == Granularity.SECTION:
                chunks = chunk_list(data, 10)

            else:
                chunks = [data]


            for chunk_num, chunk in enumerate(chunks):

                start_time = time.perf_counter()

                try:
                    input_text = json.dumps(chunk, ensure_ascii=False)

                    response = individual_input(input_text, prompt, schema, "theme_generation_schema")

                    elapsed = time.perf_counter() - start_time

                    save_response.save_response(response=response, experiment_folder=experiment_folder, input_filename=f"{file.stem}_chunk_{chunk_num}.json", processing_time=elapsed)

                except Exception as e:

                    elapsed = time.perf_counter() - start_time

                    log.write(
                        f"Status: FAILED\n"
                        f"File: {file.name}\n"
                        f"Chunk: {chunk_num}\n"
                        f"Error: {str(e)}\n"
                        f"Time: {elapsed:.2f}s\n\n"
                    )

    print(f"Completed theme generation: {testname}")