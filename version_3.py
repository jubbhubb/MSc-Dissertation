from enum import Enum
from pathlib import Path
from datetime import datetime
from Z_helper_functions.preprocessing import create_experiment_folder
from Z_helper_functions.api_calls import process_folder
from Z_helper_functions.aggregation import combine_json_files, combine_to_corpus, combine_json_by_document, recombine_split_documents
from Z_helper_functions.theme_generation_v2 import theme_generation_pipeline_v2
from Z_helper_functions.theme_grouping_inductive import theme_grouping
from Z_helper_functions.mapping_overarching_themes import theme_mapping
# from Z_helper_functions.save_response import save_response
from Z_helper_functions.final_report_production import final_report


class Granularity(Enum):
    SECTION="S"
    DOCUMENT="D"
    CORPUS="C"

class Reasoning(Enum):
    INDUCTIVE="inductive"
    DEDUCTIVE="deductive"

def coding_log(input_tokens_used, output_tokens_used, start_time, finish_time, folders_created, folder_processed, test_path):
    with open(f"{test_path}/logs/overview_log.txt", "a", encoding="utf-8") as log:
        log.write(f"Experiment started: {start_time}\n")
        log.write(f"Folder processed (ALL API CALLS): {folder_processed}\n")
        log.write(f"Time taken to process folder: {folder_processed - folders_created}\n")
        log.write(f"Total time taken: {finish_time - start_time}\n")
        log.write(f"Input tokens used: {input_tokens_used}\n")
        log.write(f"Output tokens used: {output_tokens_used}\n")
        log.write(f"Total tokens used: {input_tokens_used+output_tokens_used}\n")


def run_experiment():
    
    granularities = [
        # Granularity.SECTION,
        # Granularity.DOCUMENT,
        Granularity.CORPUS,
    ]
    
    reasonings = [
        Reasoning.INDUCTIVE,
        Reasoning.DEDUCTIVE,
    ]
    experiment_folder = "experiments_v10"
    source_folder = "source_files"
    lowercase = True
    prompt = ""
    stage = "coding"
    start_time = datetime.now()
    for granularity in granularities:
        for reasoning in reasonings:
            if reasoning == Reasoning.INDUCTIVE:
                with open("prompts/coding_prompt_induction.txt", "r", encoding="utf-8") as f:
                    prompt = f.read()
            elif reasoning == Reasoning.DEDUCTIVE:
                with open("prompts/coding_prompt_deduction.txt", "r", encoding="utf-8") as f:
                    prompt = f.read()
            input_tokens_start = int(open('input_token_count.txt', 'r').read())
            output_tokens_start = int(open('output_token_count.txt', 'r').read())
            testname = stage + "_" + reasoning.value + "_"+ granularity.value
            print(f"Currently processing: {testname}")
            create_experiment_folder(testname, experiments_folder = experiment_folder, lowercase=lowercase, granularity=granularity.value, prompt=prompt, source_folder=source_folder)
            folders_created = datetime.now()
            process_folder(f"{experiment_folder}/{testname}", prompt = prompt)
            folder_processed = datetime.now()
            combine_json_by_document(f"{experiment_folder}/{testname}/output_files", output_folder=f"{experiment_folder}/{testname}/combined_json")
            finish_time = datetime.now()
            input_tokens_end = int(open('input_token_count.txt', 'r').read())
            output_tokens_end = int(open('output_token_count.txt', 'r').read())
            input_tokens_used = input_tokens_end-input_tokens_start
            output_tokens_used = output_tokens_end-output_tokens_start
            coding_log(input_tokens_used, output_tokens_used, start_time, finish_time, folders_created, folder_processed, test_path = experiment_folder + "/" + testname)
            print(f"Coding completed for folder {testname}")
    print("Stage One - Coding completed.")

    # =====================================
    # Stage Two- Theme Developement
    # =====================================
    stage = "theme_development"

    theme_transitions = [
    # (Granularity.SECTION, Granularity.SECTION),
    # (Granularity.SECTION, Granularity.DOCUMENT),
    # (Granularity.SECTION, Granularity.CORPUS), #Can cause problems with large datasets so disabled by default.
    #  
    # (Granularity.DOCUMENT, Granularity.SECTION),
    # (Granularity.DOCUMENT, Granularity.DOCUMENT),
    # (Granularity.DOCUMENT, Granularity.CORPUS),

    # (Granularity.CORPUS, Granularity.SECTION),
    (Granularity.CORPUS, Granularity.CORPUS),
    ]
    for previous_granularity, new_granularity in theme_transitions:
        for reasoning in reasonings:
            stage = "theme_development"
            testname = (stage + "_" + reasoning.value + "_" + previous_granularity.value + "-" + new_granularity.value)
            test_folder = Path(f"{experiment_folder}/{testname}")
            test_folder.mkdir(parents=True, exist_ok=True)
            log_folder = test_folder / "logs"
            log_folder.mkdir(parents=True, exist_ok=True)
            theme_generation_pipeline_v2(stage, previous_granularity.value, new_granularity.value, reasoning, test_folder, experiment_folder)
            stage = "theme_grouping"
            # =====================================
            # Stage Three - Theme Grouping (Inductive only)
            # =====================================
            if reasoning.value == "inductive":
                prompt = ""
                combine_json_files(test_folder / "output_files", test_folder/"combined_codes.json")
                with open("prompts/grouping_themes_prompt.txt", "r", encoding="utf-8") as f:
                    prompt = f.read()
                theme_grouping(prompt, test_folder)
                theme_mapping(test_folder)
            if reasoning.value == "deductive":
                combine_json_files(test_folder / "output_files", test_folder/"codes_filtered.json")
            final_report(test_folder, )
            
            
                

def main():
    run_experiment()

main()