from Z_helper_functions.save_response import save_response
from Z_helper_functions.unique_themes_finder import unique_themes
from Z_helper_functions.api_calls import individual_input
import time

def theme_grouping(prompt, test_folder):
    unique_themes_obj = unique_themes(test_folder / "output_files")
    schema = {
        "type": "object",
        "properties": {
            "codes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "theme": {
                            "type": "string"
                        },
                        "description": {
                            "type": "string"
                        },
                        "subthemes": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "theme",
                        "description",
                        "subthemes"
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

    theme_string = ""

    for theme in unique_themes_obj:
        theme_string += f"- {theme}\n"

    while True:
        try:
            print("Attempting theme grouping API call...")

            theme_response = individual_input(
                theme_string,
                prompt,
                schema,
                "theme_grouping_schema"
            )
            print("Theme grouping API call successful.")
            break

        except Exception as e:
            print(f"API call failed: {e}")
            print("Retrying in 10 seconds...")
            time.sleep(10)

    save_response(
         response = theme_response,
         experiment_folder=test_folder,
         input_filename="theme_grouping_input.json",
         processing_time=None
    )
    