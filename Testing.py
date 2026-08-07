from pydoc import text
from pandas import pd
import config_setup
from datetime import datetime

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

    # print(response.output_text)
    used_tokens = response.usage.total_tokens
    print(f"Used tokens: {used_tokens}")
    input_tokens = response.usage.input_tokens
    print(f"Input tokens: {input_tokens}")
    output_tokens = response.usage.output_tokens
    print(f"Output tokens: {output_tokens}")
    INPUT_TOKEN_COUNT = INPUT_TOKEN_COUNT + input_tokens
    OUTPUT_TOKEN_COUNT = OUTPUT_TOKEN_COUNT + output_tokens
    with open('input_token_count.txt', 'w') as file:
        file.write(str(INPUT_TOKEN_COUNT))
    with open('output_token_count.txt', 'w') as file:
        file.write(str(OUTPUT_TOKEN_COUNT))
    return(response)


def file_input(file_path):
    with open(file_path, 'r') as file:
        input_text = file.read()
        print(f"Input text: {input_text}")
    return individual_input(input_text)

def corpus_run(file_path):
    input_text = file_input(file_path)
    response = individual_input(input_text)
    print(response)


def read_in_json(file_path):
    return pd.read_json(file_path, lines=True)


def main():
    # Example usage of individual_input function
    # input_text = "I often feel overwhelmed by my workload, but talking with colleagues helps me cope."
    # result = individual_input(input_text)
    # print(f"Result: {result}")

    # Example usage of file_input function
    file_path = "input.txt"  # Replace with your actual file path
    print(file_input(file_path))
    
main()