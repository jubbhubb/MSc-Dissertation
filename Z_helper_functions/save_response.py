from pathlib import Path
import json
from datetime import datetime


def save_response(
    response,
    experiment_folder,
    input_filename,
    processing_time=None
):
    """
    Saves API response into experiment structure.

    Parameters:
        response:
            OpenAI Responses API object

        experiment_folder:
            Path to experiment folder

        input_filename:
            Name of input text file processed

        processing_time:
            Optional time taken for API call
    """

    experiment_folder = Path(experiment_folder)

    output_folder = experiment_folder / "output_files"
    state_folder = experiment_folder / "api_state"
    log_folder = experiment_folder / "logs"

    # Create folders if needed
    output_folder.mkdir(exist_ok=True)
    state_folder.mkdir(exist_ok=True)
    log_folder.mkdir(exist_ok=True)


    file_stem = Path(input_filename).stem


    # -----------------------------
    # Save analytical output
    # -----------------------------

    # Extract the assistant text response
    output_text = response.output[-1].content[0].text


    output_json = json.loads(output_text)
    codes = output_json["codes"]
    
    output_file = output_folder / f"{file_stem}.json"

    with open(output_file, "w", encoding="utf-8"
              ) as f:
                    json.dump(
                        codes,
                        f,
                        indent=4,
                        ensure_ascii=False
                    )


    # -----------------------------
    # Save API state
    # -----------------------------

    api_state = {
        "response_id": response.id,

        "created_at": response.created_at,

        "model": response.model,

        "instructions": response.instructions,

        "input_reference": {
            "file": input_filename
        },

        "reasoning": {
            "encrypted_content":
                get_encrypted_reasoning(response)
        },

        "raw_output": output_text,

        "status": response.status
    }


    state_file = state_folder / f"{file_stem}.json"

    with open(
        state_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            api_state,
            f,
            indent=4,
            ensure_ascii=False
        )


    # -----------------------------
    # Save log entry
    # -----------------------------

    log_file = log_folder / "run_log.txt"

    print("Saving response...")
    print("Response ID:", response.id)
    print("Output length:", len(response.output_text))

    encrypted = get_encrypted_reasoning(response)

    print(
        "Encrypted reasoning found:",
        encrypted is not None
)
     
    with open(
        log_file,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n"
            f"{datetime.now().isoformat()}\n"
            f"File: {input_filename}\n"
            f"Status: {response.status}\n"
            f"Model: {response.model}\n"
        )

        if processing_time:
            f.write(
                f"Processing time: {processing_time:.2f}s\n"
            )

        if response.usage:
            f.write(
                f"Input tokens: {response.usage.input_tokens}\n"
                f"Output tokens: {response.usage.output_tokens}\n"
                f"Total tokens: {response.usage.total_tokens}\n"
            )


def get_encrypted_reasoning(response):
    """
    Extract encrypted reasoning item for future stateful calls.
    """

    for item in response.output:

        if hasattr(item, "encrypted_content"):
            return item.encrypted_content

    return None