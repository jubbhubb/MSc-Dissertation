import json
from pathlib import Path
import time

# import Z_helper_functions.aggregation as aggregation
from Z_helper_functions.aggregation import combine_json_by_document,combine_to_corpus,combine_json_files
from Z_helper_functions.api_calls import individual_input
import Z_helper_functions.save_response as save_response


def chunk_list(items, chunk_size=10):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def theme_generation_pipeline_v2(stage, previous_granularity, new_granularity, reasoning, test_folder, experiment_folder):
    print(f"Starting theme generation pipeline for stage: {stage}, previous granularity: {previous_granularity}, new granularity: {new_granularity}")
    if reasoning.value == "inductive":
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

    elif reasoning.value == "deductive":
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
                            "theme": {"type": "string"},
                            "subtheme": {"type": "string"}
                        },
                        "required": [
                            "code",
                            "explanation",
                            "quote",
                            "confidence",
                            "theme",
                            "subtheme"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["codes"],
            "additionalProperties": False
        }
        with open("prompts/subtheme_generation_prompt_deduction.txt", "r", encoding="utf-8") as f:
            prompt = f.read()

    test_folder.mkdir(parents=True, exist_ok=True)

    log_folder = test_folder / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)
    log_file = log_folder / "run_log.txt"

    print(f"Experiment name: {stage} {previous_granularity} {new_granularity}")



    coding_folder = Path(f"{experiment_folder}/coding_{reasoning.value}_{previous_granularity}")

    combine_json_files(coding_folder / "output_files", output_file = coding_folder / "combined_json/combined.json")
    # Determine input location

    if new_granularity == "S":

        if previous_granularity == "D":
            input_folder = coding_folder / "output_files"

        elif previous_granularity == "S":
            input_folder = coding_folder / "combined_json"

        elif previous_granularity == "C":
            input_folder = coding_folder / "output_files"


    elif new_granularity == "D":

        if previous_granularity == "S":
            input_folder = coding_folder / "combined_json"

        elif previous_granularity == "D":
            input_folder = coding_folder / "output_files"

        elif previous_granularity == "C":
            raise ValueError("Corpus to document is not supported")


    elif new_granularity == "C":
        input_folder = coding_folder / "combined_json"


    else:
        raise ValueError(f"Unsupported granularity: {new_granularity}")


    # print(f"Input folder: {input_folder}")


    with open(log_file, "a", encoding="utf-8") as log:
        for file in input_folder.glob("*.json"):
            print(f"Processing file: {file.name}")
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if new_granularity == "S":
                chunks = chunk_list(data, 10)
            else:
                chunks = [data]
            for chunk_num, chunk in enumerate(chunks):
                start_time = time.perf_counter()
                try:
                    input_text = json.dumps(chunk, ensure_ascii=False)
                    response = individual_input(input_text, prompt, schema, "theme_generation_schema")
                    elapsed = time.perf_counter() - start_time
                    save_response.save_response(response, test_folder, input_filename=f"{file.stem}_chunk_{chunk_num}.json", processing_time=elapsed)
                except Exception as e:
                    elapsed = time.perf_counter() - start_time
                    log.write(
                        f"Status: FAILED\n"
                        f"File: {file.name}\n"
                        f"Chunk: {chunk_num}\n"
                        f"Error: {str(e)}\n"
                        f"Time: {elapsed:.2f}s\n\n"
                    )
    print(f"Completed theme generation: {test_folder}")