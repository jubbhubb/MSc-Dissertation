from pydoc import text
import config_setup
from datetime import datetime
from pathlib import Path
import time
from save_response import save_response

import openai 
with open('key.txt', 'r') as file:
    API = file.read()

def individual_input(input_text):
    with open('input_token_count.txt', 'r') as file:
        INPUT_TOKEN_COUNT = file.read()
        INPUT_TOKEN_COUNT = int(INPUT_TOKEN_COUNT)

    with open('output_token_count.txt', 'r') as file:
        OUTPUT_TOKEN_COUNT = file.read()
        OUTPUT_TOKEN_COUNT = int(OUTPUT_TOKEN_COUNT)

    client = openai.OpenAI(api_key=API)

    schema = {
    "type": "object",
    "properties": {
        "codes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string"
                    },
                    "explanation": {
                        "type": "string"
                    },
                    "quote": {
                        "type": "string"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1
                    }
                },
                "required": [
                    "code",
                    "explanation",
                    "quote",
                    "confidence"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": [
        "codes"
    ],
    "additionalProperties": False
    }


    response = client.responses.create(
        model="gpt-5.5",
        instructions="Analyse this text and provide thematic codes in the following format: {\"codes\": [{\"code\": \"thematic code\", \"explanation\": \"brief explanation of the code\", \"quote\": \"a relevant quote from the text that supports the code\", \"confidence\": number from 0 to 1}]}:" \
        "I often feel overwhelmed by my workload, but talking with colleagues helps me cope.",
        input=input_text,
        store=False,
        include=["reasoning.encrypted_content"],
        text={
            "format": {
                "type": "json_schema",
                "name": "thematic_codes",
                "schema": schema,
            }
        }
    )
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    INPUT_TOKEN_COUNT = INPUT_TOKEN_COUNT + input_tokens
    OUTPUT_TOKEN_COUNT = OUTPUT_TOKEN_COUNT + output_tokens
    with open('input_token_count.txt', 'w') as file:
        file.write(str(INPUT_TOKEN_COUNT))
    with open('output_token_count.txt', 'w') as file:
        file.write(str(OUTPUT_TOKEN_COUNT))
    return(response) 

def process_folder(experiment_folder):

    """
    Process all input txt files in an experiment folder.

    API calls are made here.
    Saving of responses is handled by save_response.py.
    """

    experiment_folder = Path(experiment_folder)

    input_folder = experiment_folder / "input_files"
    log_folder = experiment_folder / "logs"

    log_folder.mkdir(exist_ok=True)

    log_file = log_folder / "run_log.txt"


    with open(log_file, "a", encoding="utf-8") as log:

        for file in input_folder.glob("*.txt"):

            start_time = time.perf_counter()

            # log.write(
            #     f"\n{datetime.now().isoformat()}\n"
            #     f"Processing: {file.name}\n"
            # )

            try:

                # Read input text
                with open(
                    file,
                    "r",
                    encoding="utf-8"
                ) as f:
                    text = f.read()


                # Make API call
                response = individual_input(text)


                # Save all response information
                elapsed = time.perf_counter() - start_time

                save_response(
                    response=response,
                    experiment_folder=experiment_folder,
                    input_filename=file.name,
                    processing_time=elapsed
                )


                # log.write(
                #     f"Status: SUCCESS\n"
                #     f"Time: {elapsed:.2f} seconds\n"
                # )


            except Exception as e:

                elapsed = time.perf_counter() - start_time

                log.write(
                    f"Status: FAILED\n"
                    f"Error: {str(e)}\n"
                    f"Time: {elapsed:.2f} seconds\n"
                )