import json
from enum import Enum
from pathlib import Path
import openai 
with open('key.txt', 'r') as file:
    API = file.read()
from Z_helper_functions.save_response import save_response
from Z_helper_functions.aggregation import combine_json_files, combine_json_by_document
import time

class Granularity(Enum):
    SECTION="section"
    DOCUMENT="document"
    CORPUS="corpus"

def final_report(test_folder):
    prompt = ""
    with open("prompts/final_report_production_prompt.txt", "r", encoding="utf-8") as f:
        prompt = f.read()

    start_time = time.time()

    with open('input_token_count.txt', 'r') as file:
            INPUT_TOKEN_COUNT = file.read()
            INPUT_TOKEN_COUNT = int(INPUT_TOKEN_COUNT)
    
    with open('output_token_count.txt', 'r') as file:
        OUTPUT_TOKEN_COUNT = file.read()
        OUTPUT_TOKEN_COUNT = int(OUTPUT_TOKEN_COUNT)

    input_text = ""
    with open(test_folder / "codes_filtered.json", "r", encoding="utf-8") as f:
        input_data = json.load(f)

    input_text = json.dumps(input_data, indent=2)

    client = openai.OpenAI(api_key=API)
    response = client.responses.create(
        model="gpt-5.5",
        instructions=prompt,
        input=input_text,
        store=False,
        include=["reasoning.encrypted_content"],
        text={
            "format": {
                "type": "text"
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

    with open(test_folder / "logs/report_production_log.txt", 'w') as file:
        file.write(f"Input Tokens Used in report production: {input_tokens}")
        file.write(f"Output Tokens Used in report production: {output_tokens}")
        file.write(f"Time taken: {time.perf_counter() - start_time}")              

    with open (test_folder / "final_report", 'w') as file:
        file.write(str(response.output_text))

        