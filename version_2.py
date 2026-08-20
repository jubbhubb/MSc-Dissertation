import datetime
from enum import Enum
from pathlib import Path
import time 
from preprocessing import create_experiment_folder
from cosine_sim_all import cosine_similarity_between_documents
from nli_explanation_validation import calculate_nli_success_rate, nli_explanation_validation
from nli_random import nli_random_baseline
from preprocessing import create_experiment_folder
from api_calls import individual_input, process_folder
from datetime import datetime
from aggregation import recombine_split_documents, combine_to_corpus, combine_json_by_document, combine_json_files
from quote_success_rates import calculate_quote_success_rates
from random_explanation import random_explanation_baseline
from random_nli_calc import random_nli_calc
import semantic_projections
from semantic_quote_explanation import calculate_explanation_semantic_score, semantic_explanation_validation
from semantic_quote_validation2 import calculate_semantic_success_rate, semantic_quote_validation
from semantic_random_stats import random_semantic_calc
from similarity_distribution import similarity_histogram
from quote_check import validate_quotes
from thematic_heatmap2 import theme_similarity_heatmap
from thematic_mindmap import thematic_mindmap
from theme_generation_v2 import theme_generation_pipeline_v2
from unique_themes_finder import unique_themes
from save_response import save_response
from thematic_projections import thematic_projections_method

class Granularity(Enum):
    SECTION="section"
    DOCUMENT="document"
    CORPUS="corpus"

class Reasoning(Enum):
    INDUCTIVE="inductive"
    DEDUCTIVE="deductive"

def semantic_quote_explanation_pipeline(experiment_path):

    semantic_explanation_validation(experiment_path)

    explanation_stats = calculate_explanation_semantic_score(f"{experiment_path}/quote_explanation_semantic_validation.csv")
    
    print("QUOTE EXPLANATION SEMANTIC VALIDATION")
    print( f"Average similarity: "f"{explanation_stats['overall_average']:.3f}")
    print(f"Quotes evaluated: "f"{explanation_stats['total_quotes']}")

def nli_explanation_pipeline(experiment_path):
    nli_explanation_validation(experiment_path)
    nli_stats = calculate_nli_success_rate(f"{experiment_path}/quote_explanation_nli_validation.csv")
    print("NLI EXPLANATION VALIDATION")
    print(f"Average entailment: "f"{nli_stats['average_entailment']:.3f}")
    print(f"Average contradiction: "f"{nli_stats['average_contradiction']:.3f}")
    print(f"High confidence entailments: "f"{nli_stats['high_confidence_entailments']:.2%}")
    print(f"Quotes evaluated: "f"{nli_stats['total_quotes']}")

def quote_validation_pipeline(experiment_path, chunk_method="words", window_size=23, step_size=1):
    """
    Run the full quote validation pipeline for a given experiment.

    Parameters:
        experiment_name:
            Name of the experiment folder.
        chunk_method:
            Method for chunking text ("words" or "sentences").
        window_size:
            Size of the chunking window (for "words" method).
        step_size:
            Step size for moving the chunking window (for "words" method).
    """


    # Lexical validation
    validate_quotes(experiment_path)

    # Semantic validation
    semantic_quote_validation(
        experiment_path,
        chunk_method=chunk_method,
        window_size=window_size,
        step_size=step_size
    )

    # Semantic validation
    #Sentence Chunking
    # semantic_quote_validation(
    # experiment_path,
    # chunk_method="sentences",
    # window_size=1,
    # step_size=0
    # )

    #Word chunking - best results so far
    # semantic_quote_validation(
    #     experiment_path,
    #     chunk_method="words",
    #     window_size=23,
    #     step_size=1
    # )

    #Adaptive Window size - innaccurate due to abbreviations for clarity. 
    # semantic_quote_validation(
    # experiment_path,
    # chunk_method="words",
    # adaptive=True,
    # step_size=5
    # )

    # Calculate and print statistics
    stats = calculate_quote_success_rates(
        f"{experiment_path}/quote_validation_results"
    )

    print("OVERALL")
    print(f"Total quotes: {stats['overall']['total_quotes']}")
    print(f"Strict complete match rate: "f"{stats['overall']['strict_full_rate']:.2%}")
    print(f"Relaxed complete match rate: " f"{stats['overall']['relaxed_full_rate']:.2%}")
    print(f"Strict evidence coverage: " f"{stats['overall']['strict_coverage']:.2%}")
    print(f"Relaxed evidence coverage: "f"{stats['overall']['relaxed_coverage']:.2%}")

    print("\nPER DOCUMENT")
    for file, result in stats["per_file"].items():
        print(
            f"{file}: "
            f"Strict complete {result['strict_full_rate']:.2%}, "
            f"Strict coverage {result['strict_coverage']:.2%}, "
            f"Relaxed complete {result['relaxed_full_rate']:.2%}, "
            f"Relaxed coverage {result['relaxed_coverage']:.2%}"
        )

    semantic_stats = calculate_semantic_success_rate(f"{experiment_path}/quote_semantic_validation.csv")
    
    print("SEMANTIC QUOTE VALIDATION")
    print(f"Average similarity: "f"{semantic_stats['overall_average']:.3f}")
    print(f"Quotes evaluated: "f"{semantic_stats['total_quotes']}")

    print("\nPER DOCUMENT")
    for document, score in semantic_stats["per_document"].items():
        print(f"{document}: {score:.3f}")

def main():
    
    experiments_folder = "experiments_v6"
    lowercase = True
    granularities = [
        Granularity.SECTION,
        Granularity.DOCUMENT,
        Granularity.CORPUS,
    ]

    reasonings = [
        # Reasoning.INDUCTIVE,
        Reasoning.DEDUCTIVE,
    ]

    # =====================================
    #Stage One - Coding
    # =====================================
    # prompt = ""
    # stage = "coding"

    # for granularity in granularities:
    #     for reasoning in reasonings:
    #         # if granularity == Granularity.SECTION or granularity == Granularity.DOCUMENT or granularity == Granularity.CORPUS:
    #         #     continue
    #         if reasoning == Reasoning.INDUCTIVE:
    #             with open("prompts/coding_prompt_induction.txt", "r", encoding="utf-8") as f:
    #                 prompt = f.read()
    #         elif reasoning == Reasoning.DEDUCTIVE:
    #             with open("prompts/coding_prompt_deduction.txt", "r", encoding="utf-8") as f:
    #                 prompt = f.read()
    #         input_tokens_start = int(open('input_token_count.txt', 'r').read())
    #         output_tokens_start = int(open('output_token_count.txt', 'r').read())
    #         start_time = datetime.now()
    #         testname = stage + "_" + granularity.value + "_" + reasoning.value
    #         print(f"Experiment name: {testname}")
    #         create_experiment_folder(testname, experiments_folder = experiments_folder, lowercase=lowercase, granularity=granularity.value, prompt=prompt)
    #         folders_created = datetime.now()
    #         process_folder(f"{experiments_folder}/{testname}", prompt = prompt)
    #         folder_processed = datetime.now()
    #         recombine_split_documents(f"{experiments_folder}/{testname}/input_files", save_combined=True, output_path=f"{experiments_folder}/{testname}/recombined_files")
    #         combine_to_corpus(f"{experiments_folder}/{testname}/recombined_files", f"{experiments_folder}/{testname}/corpus_files")
    #         recombine_split_documents(f"{experiments_folder}/{testname}/input_files", save_combined=True, output_path=f"{experiments_folder}/{testname}/recombined_files")
    #         combine_json_by_document(f"{experiments_folder}/{testname}/output_files", output_folder=f"{experiments_folder}/{testname}/combined_json")
    #         sections_recombined = datetime.now()
    #         cosine_similarity_between_documents(f"{experiments_folder}/{testname}")
    #         cosine_calculated = datetime.now()
    #         input_tokens_end = int(open('input_token_count.txt', 'r').read())
    #         output_tokens_end = int(open('output_token_count.txt', 'r').read())
    #         with open(f"{experiments_folder}/{testname}/logs/overview_log.txt", "a", encoding="utf-8") as log:
    #             log.write(f"Experiment started: {start_time}\n")
    #             log.write(f"Folders created: {folders_created}\n")
    #             log.write(f"Time taken to create folders: {folders_created - start_time}\n")
    #             log.write(f"Folder processed (ALL API CALLS): {folder_processed}\n")
    #             log.write(f"Time taken to process folder: {folder_processed - folders_created}\n")
    #             log.write(f"Sections recombined: {sections_recombined}\n")
    #             log.write(f"Time taken to recombine sections: {sections_recombined - folder_processed}\n")
    #             log.write(f"Cosine calculated: {cosine_calculated}\n")
    #             log.write(f"Time taken to calculate cosine similarity: {cosine_calculated - sections_recombined}\n")
    #             log.write(f"Total time taken: {cosine_calculated - start_time}\n")
    #             log.write(f"Input tokens used: {input_tokens_end - input_tokens_start}\n")
    #             log.write(f"Output tokens used: {output_tokens_end - output_tokens_start}\n")
    #             log.write(f"Total tokens used: {(input_tokens_end - input_tokens_start) + (output_tokens_end - output_tokens_start)}\n")

    #         quote_validation_pipeline(f"{experiments_folder}/{testname}", chunk_method="words", window_size=23, step_size=1)

    #         semantic_quote_explanation_pipeline(f"{experiments_folder}/{testname}")

    #         nli_explanation_pipeline(f"{experiments_folder}/{testname}")

    #         similarity_histogram(f"{experiments_folder}/{testname}")

    #         # Statistical Baselines comparing Random Pairs vs True Pairs
    #         random_explanation_baseline(experiments_folder, testname, display=False) #Semantic Baseline
    #         nli_random_baseline(experiments_folder, testname, display=False) #NLI Baseline
    #         random_nli_calc(experiments_folder, testname) 
    #         random_semantic_calc(experiments_folder, testname)

    # print("Stage One - Coding completed.")

    reasoning = Reasoning.DEDUCTIVE

    # # =====================================
    # #Stage Two - THEME GENERATION
    # # =====================================
    # #Granularity options = document, section, corpus
    if reasoning == Reasoning.DEDUCTIVE:
        theme_transitions = [
        # (Granularity.SECTION, Granularity.SECTION),
        # (Granularity.SECTION, Granularity.DOCUMENT),
        # (Granularity.SECTION, Granularity.CORPUS),

        # (Granularity.DOCUMENT, Granularity.SECTION),
        # (Granularity.DOCUMENT, Granularity.DOCUMENT),
        # (Granularity.DOCUMENT, Granularity.CORPUS),

        (Granularity.CORPUS, Granularity.SECTION),
        # (Granularity.CORPUS, Granularity.CORPUS),
        ]
        #Corpus to document is not included as there is no way to redefine document boundaries easily from a corpus. Corpus to Section is valid as the output is simply split. 

        for previous_granularity, new_granularity in theme_transitions:
            
            stage = "subtheme_generation"
            testname = (stage + "_" + reasoning.value + "_" + previous_granularity.value + "_" + new_granularity.value)
            # old_experiment_folder = Path(f"{experiments_folder}")
            experiment_folder = Path(f"{experiments_folder}/{testname}")
            experiment_folder.mkdir(parents=True, exist_ok=True)

            theme_development_folder = Path(f"{experiments_folder}/theme_development_{testname}")
            theme_development_folder.mkdir(parents=True, exist_ok=True) 

            log_folder = experiment_folder / "logs"
            log_folder.mkdir(parents=True, exist_ok=True)
            theme_generation_pipeline_v2(stage, previous_granularity.value, new_granularity.value, reasoning, testname, experiments_folder=experiments_folder)

            print(f"Experiment name: {testname}")
            #Analysis
            
            # combine_json_files(experiment_folder / "output_files", experiment_folder / "combined_output.json")
            unique_theme_obj = unique_themes(experiment_folder)
            # semantic_projections.semantic_projections(experiment_folder / "combined_output.json", experiment_folder, display=False)

    #         # =====================================
    #         # Stage Three - Theme Development
    #         # =====================================

    #         prompt = ""

    #         with open("prompts/grouping_themes_prompt.txt", "r", encoding="utf-8") as f:
    #             prompt = f.read()

    #         schema = {
    #             "type": "object",
    #             "properties": {
    #                 "codes": {
    #                     "type": "array",
    #                     "items": {
    #                         "type": "object",
    #                         "properties": {
    #                             "theme": {
    #                                 "type": "string"
    #                             },
    #                             "description": {
    #                                 "type": "string"
    #                             },
    #                             "subthemes": {
    #                                 "type": "array",
    #                                 "items": {
    #                                     "type": "string"
    #                                 }
    #                             }
    #                         },
    #                         "required": [
    #                             "theme",
    #                             "description",
    #                             "subthemes"
    #                         ],
    #                         "additionalProperties": False
    #                     }
    #                 }
    #             },
    #             "required": [
    #                 "codes"
    #             ],
    #             "additionalProperties": False
    #         }

    #         theme_string = ""

    #         for theme in unique_theme_obj:
    #             theme_string += f"- {theme}\n"


    #         # while True:
    #         #     try:
    #         #         print("Attempting theme grouping API call...")

    #         #         theme_response = individual_input(
    #         #             theme_string,
    #         #             prompt,
    #         #             schema,
    #         #             "theme_grouping_schema"
    #         #         )

    #         #         print("Theme grouping API call successful.")
    #         #         break

    #         #     except Exception as e:
    #         #         print(f"API call failed: {e}")
    #         #         print("Retrying in 10 seconds...")
    #         #         time.sleep(10)


    #         # save_response(
    #         #      response = theme_response,
    #         #      experiment_folder=theme_development_folder,
    #         #      input_filename="theme_grouping_input.json",
    #         #      processing_time=None
    #         # )

    #         # thematic_projections_method(
    #         #     experiment_folder / "combined_output.json",
    #         #     theme_development_folder / "output_files/theme_grouping_input.json",
    #         #     theme_development_folder,
    #         #     display=False
    #         # )

    #         theme_similarity_heatmap(
    #             theme_development_folder / "output_files/theme_grouping_input.json",
    #             theme_development_folder,
    #             display=False
    #         )

    #         # thematic_mindmap(
    #         #     theme_development_folder / "output_files/theme_grouping_input.json",
    #         #     theme_development_folder,
    #         #     display=False
    #         # )

main()
