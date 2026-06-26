from pydoc import text

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
                        "code": {"type": "string"},
                        "explanation": {"type": "string"},
                        "quote": {"type": "string"}
                    },
                    "required": ["code", "explanation", "quote"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["codes"],
        "additionalProperties": False
    }

    response = client.responses.create(
        model="gpt-5.5",
        instructions="Analyse this text and provide thematic codes in the following format: {\"codes\": [{\"code\": \"thematic code\", \"explanation\": \"brief explanation of the code\", \"quote\": \"a relevant quote from the text that supports the code\"}]}:" \
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

    print(response.output_text)
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
    return(response.output_text)

individual_input("What is the capital of France?")

