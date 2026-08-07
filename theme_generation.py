import json
from pathlib import Path
import time

import aggregation
from api_calls import individual_input
import save_response

def chunk_list(items, chunk_size=10):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]

def theme_generation_pipeline(stage, previous_granularity, new_granularity, testname = None, experiments_folder = "experiments"):
    print(f"Starting theme generation pipeline for stage: {stage}, previous granularity: {previous_granularity}, new granularity: {new_granularity}")

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

    if testname is None:
        print("Error: testname must be provided for the theme generation pipeline.")
        return

    with open(
        "prompts/theme_generation_prompt_induction.txt",
        "r",
        encoding="utf-8"
    ) as f:
        prompt = f.read()

    experiment_folder = Path(f"experiments/{testname}")
    log_folder = experiment_folder / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)

    log_file = log_folder / "run_log.txt"

    print(f"Experiment name: {testname}")
    print(f"Using prompt:\n{prompt}\n")

    if new_granularity == "section":
        if previous_granularity == "document":
            input_folder = Path(
                f"experiments/coding_{previous_granularity}/output_files"
            )
        elif previous_granularity == "section":
            input_folder = Path(
                f"experiments/coding_{previous_granularity}/combined_json"
            )
        elif previous_granularity == "corpus":
            input_folder = Path(
                f"experiments/coding_{previous_granularity}/output_files"
            )

        print(f"Input folder for section granularity: {input_folder}")

        with open(log_file, "a", encoding="utf-8") as log:
            print("LOG OPENED")
            print("Input folder exists:", input_folder.exists())
            print("Contents of input folder:")
            for item in input_folder.iterdir():
                print(item)
            for file in input_folder.glob("*.json"):
                print(f"Processing file: {file.name}")
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    print(f"Loaded {len(data)} items from {file.name}")
                
                print(f"Processing file: {file.name} with {len(data)} items")

                for chunk_num, chunk in enumerate(chunk_list(data, 10)):

                    start_time = time.perf_counter()

                    try:
                        input_text = json.dumps(
                            chunk,
                            ensure_ascii=False
                        )

                        response = individual_input(
                            input_text,
                            prompt,
                            schema,
                            "theme_generation_schema"
                        )

                        elapsed = time.perf_counter() - start_time

                        save_response.save_response(
                            response=response,
                            experiment_folder=experiment_folder,
                            input_filename=f"{file.stem}_chunk_{chunk_num}.json",
                            processing_time=elapsed
                        )

                    except Exception as e:

                        elapsed = time.perf_counter() - start_time

                        log.write(
                            f"Status: FAILED\n"
                            f"File: {file.name}\n"
                            f"Chunk: {chunk_num}\n"
                            f"Error: {str(e)}\n"
                            f"Time: {elapsed:.2f}s\n\n"
                        )
    elif(new_granularity == "document"):
        if previous_granularity == "section":
            input_folder = Path(
                f"experiments/coding_{previous_granularity}/combined_json"
            )
        elif previous_granularity == "document":
            input_folder = Path(
                f"experiments/coding_{previous_granularity}/output_files"
            )
        elif previous_granularity == "corpus":
            input_folder = Path(
                f"experiments/coding_{previous_granularity}/output_files"
            )

        with open(log_file, "a", encoding="utf-8") as log:
            print("LOG OPENED")
            print("Input folder exists:", input_folder.exists())
            print("Contents of input folder:")
            for item in input_folder.iterdir():
                print(item)
            for file in input_folder.glob("*.json"):
                print(f"Processing file: {file.name}")
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    print(f"Loaded {len(data)} items from {file.name}")
                
                print(f"Processing file: {file.name} with {len(data)} items")

                for chunk_num, chunk in enumerate(chunk_list(data, 10)):

                    start_time = time.perf_counter()

                    try:
                        input_text = json.dumps(
                            chunk,
                            ensure_ascii=False
                        )

                        response = individual_input(
                            input_text,
                            prompt,
                            schema,
                            "theme_generation_schema"
                        )

                        elapsed = time.perf_counter() - start_time

                        save_response.save_response(
                            response=response,
                            experiment_folder=experiment_folder,
                            input_filename=f"{file.stem}_chunk_{chunk_num}.json",
                            processing_time=elapsed
                        )

                    except Exception as e:

                        elapsed = time.perf_counter() - start_time

                        log.write(
                            f"Status: FAILED\n"
                            f"File: {file.name}\n"
                            f"Chunk: {chunk_num}\n"
                            f"Error: {str(e)}\n"
                            f"Time: {elapsed:.2f}s\n\n"
                        )
    elif(new_granularity == "corpus"):
        corpus_input = Path(f"experiments/theme_generation_{previous_granularity}_{new_granularity}/input_files")
        corpus_input.mkdir(parents=True, exist_ok=True)
        if previous_granularity == "section":
            previous_folder = Path(f"experiments/coding_{previous_granularity}/output_files")
            aggregation.combine_json_files(
                input_path=previous_folder,
                output_file=corpus_input / "combined.json")
            input_folder = corpus_input
        elif previous_granularity == "document":
            previous_folder = Path(f"experiments/coding_{previous_granularity}/output_files")
            aggregation.combine_json_files(
                input_path=previous_folder,
                output_file=corpus_input / "combined.json")
            input_folder = corpus_input
        elif previous_granularity == "corpus":
            previous_folder = Path(f"experiments/coding_{previous_granularity}/output_files")
            aggregation.combine_json_files(
                input_path=previous_folder,
                output_file=corpus_input / "combined.json")
            input_folder = corpus_input
        with open(log_file, "a", encoding="utf-8") as log:
            chunk_num = "corpus"
            print("LOG OPENED")
            print("Input folder exists:", input_folder.exists())
            print("Contents of input folder:")
            for item in input_folder.iterdir():
                print(item)
            for file in input_folder.glob("*.json"):
                print(f"Processing file: {file.name}")
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    print(f"Loaded {len(data)} items from {file.name}")
                
                print(f"Processing file: {file.name} with {len(data)} items")
                start_time = time.perf_counter()
                try: 
                    input_text = json.dumps(data, ensure_ascii=False)
                    # print("\n--- INPUT PREVIEW ---")
                    # print(input_text[:1000])  # first 1000 characters
                    # print("--- END PREVIEW ---\n")
                    response = individual_input(
                        input_text,
                        prompt,
                        schema,
                        "theme_generation_schema"
                    )
                    print("\n--- RAW MODEL RESPONSE DEBUG ---")

                    print("Response type:", type(response))

                    print("About to access response.output")

                    raw_output = response.output

                    print("Successfully accessed response.output")
                    print("Output type:", type(raw_output))
                    print("Output repr:", repr(raw_output)[:1000])

                    print("--- END DEBUG ---\n")
                    for item in response.output:
                        if item.type == "message":
                            output_text = item.content[0].text
                            break

                    print("\n--- MODEL TEXT OUTPUT ---")
                    print(output_text[:1000] if output_text else "No message output found")
                    print("--- END MODEL TEXT OUTPUT ---\n")

                    print("Number of output items:", len(response.output))

                    print("Input records:", len(data))
                    print("Input characters:", len(input_text))
                    print("Input tokens:", response.usage.input_tokens)
                    print("Output tokens:", response.usage.output_tokens)
                    print("Status:", response.status)

                    for i, item in enumerate(response.output):
                        print(f"\nItem {i}")
                        print("Type:", item.type)
                        print("Class:", type(item))
                    elapsed = time.perf_counter() - start_time

                    save_response.save_response(
                        response=response,
                        experiment_folder=experiment_folder,
                        input_filename=f"{file.stem}_combined.json",
                        processing_time=elapsed
                    )
                
                except Exception as e:

                    elapsed = time.perf_counter() - start_time
                    log.write(
                        f"Status: FAILED\n"
                        f"File: {file.name}\n"
                        f"Chunk: {chunk_num}\n"
                        f"Error: {str(e)}\n"
                        f"Time: {elapsed:.2f}s\n\n"
                    )
    else: 
        print(f"Unsupported granularity: {new_granularity}")