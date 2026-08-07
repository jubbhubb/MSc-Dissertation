from pathlib import Path

from cosine_sim_all import cosine_similarity_between_documents
from nli_explanation_validation2 import calculate_nli_success_rate, nli_explanation_validation
from nli_random import nli_random_baseline
from preprocessing import create_experiment_folder
from api_calls import individual_input, process_folder
from datetime import datetime
from aggregation import recombine_split_documents, combine_to_corpus, combine_json_by_document, combine_json_files
from quote_check2 import validate_quotes
from quote_success_rates import calculate_quote_success_rates
from random_explanation import random_explanation_baseline
from random_nli_calc import random_nli_calc
from save_response import save_response
from semantic_projections import semantic_projections
from semantic_quote_explanation import calculate_explanation_semantic_score, semantic_explanation_validation
from semantic_quote_validation2 import calculate_semantic_success_rate, semantic_quote_validation
from semantic_random_stats import random_semantic_calc
from similarity_distribution import similarity_histogram
import thematic_projections
from theme_generation import theme_generation_pipeline
from unique_themes_finder import unique_themes
from thematic_projections import thematic_projections_method

def semantic_quote_explanation_pipeline(experiment_name):

    semantic_explanation_validation(f"experiments/{experiment_name}")

    explanation_stats = calculate_explanation_semantic_score(f"experiments/{experiment_name}/quote_explanation_semantic_validation.csv")
    
    print("QUOTE EXPLANATION SEMANTIC VALIDATION")
    print( f"Average similarity: "f"{explanation_stats['overall_average']:.3f}")
    print(f"Quotes evaluated: "f"{explanation_stats['total_quotes']}")

def nli_explanation_pipeline(experiment_name):
    nli_explanation_validation(f"experiments/{experiment_name}")
    nli_stats = calculate_nli_success_rate(f"experiments/{experiment_name}/quote_explanation_nli_validation.csv")
    print("NLI EXPLANATION VALIDATION")
    print(f"Average entailment: "f"{nli_stats['average_entailment']:.3f}")
    print(f"Average contradiction: "f"{nli_stats['average_contradiction']:.3f}")
    print(f"High confidence entailments: "f"{nli_stats['high_confidence_entailments']:.2%}")
    print(f"Quotes evaluated: "f"{nli_stats['total_quotes']}")


def quote_validation_pipeline(experiment_name, chunk_method="words", window_size=23, step_size=1):
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

    experiment_path = f"experiments/{experiment_name}"

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

    semantic_stats = calculate_semantic_success_rate(f"experiments/{experiment_name}/quote_semantic_validation.csv")
    
    print("SEMANTIC QUOTE VALIDATION")
    print(f"Average similarity: "f"{semantic_stats['overall_average']:.3f}")
    print(f"Quotes evaluated: "f"{semantic_stats['total_quotes']}")

    print("\nPER DOCUMENT")
    for document, score in semantic_stats["per_document"].items():
        print(f"{document}: {score:.3f}")


def main():
    # input_tokens_start = int(open('input_token_count.txt', 'r').read())
    # output_tokens_start = int(open('output_token_count.txt', 'r').read())
    # start_time = datetime.now()
    prompt = ""
    with open("prompts/coding_prompt_induction.txt", "r", encoding="utf-8") as f:
        prompt = f.read()

    # print(f"Using prompt:\n{prompt}\n")

    #Set up experiment parameters
    stage = "coding"
    granularity = "document"
    lowercase = True

    testname = stage + "_" + granularity

    # print(f"Experiment name: {testname}")

    # create_experiment_folder(testname, lowercase=lowercase, granularity=granularity, prompt=prompt)
    # folders_created = datetime.now()

    # process_folder(f"experiments/{testname}", prompt = prompt)
    # folder_processed = datetime.now()

    # recombine_split_documents(f"experiments/{testname}/input_files", save_combined=True, output_path=f"experiments/{testname}/recombined_files")
    # combine_to_corpus(f"experiments/{testname}/recombined_files", f"experiments/{testname}/corpus_files")
    # recombine_split_documents(f"experiments/{testname}/input_files", save_combined=True, output_path=f"experiments/{testname}/recombined_files")
    # combine_json_by_document(f"experiments/{testname}/output_files", output_folder=f"experiments/{testname}/combined_json")
    # sections_recombined = datetime.now()

    # cosine_similarity_between_documents(f"experiments/{testname}")
    # cosine_calculated = datetime.now()
    # input_tokens_end = int(open('input_token_count.txt', 'r').read())
    # output_tokens_end = int(open('output_token_count.txt', 'r').read())

    # with open(f"experiments/{testname}/logs/overview_log.txt", "a", encoding="utf-8") as log:
    #     log.write(f"Experiment started: {start_time}\n")
    #     log.write(f"Folders created: {folders_created}\n")
    #     log.write(f"Time taken to create folders: {folders_created - start_time}\n")
    #     log.write(f"Folder processed (ALL API CALLS): {folder_processed}\n")
    #     log.write(f"Time taken to process folder: {folder_processed - folders_created}\n")
    #     log.write(f"Sections recombined: {sections_recombined}\n")
    #     log.write(f"Time taken to recombine sections: {sections_recombined - folder_processed}\n")
    #     log.write(f"Cosine calculated: {cosine_calculated}\n")
    #     log.write(f"Time taken to calculate cosine similarity: {cosine_calculated - sections_recombined}\n")
    #     log.write(f"Total time taken: {cosine_calculated - start_time}\n")
    #     log.write(f"Input tokens used: {input_tokens_end - input_tokens_start}\n")
    #     log.write(f"Output tokens used: {output_tokens_end - output_tokens_start}\n")
    #     log.write(f"Total tokens used: {(input_tokens_end - input_tokens_start) + (output_tokens_end - output_tokens_start)}\n")

    # cosine_similarity_between_documents(f"experiments/{"coding_document"}")

    # quote_validation_pipeline(testname)

    # semantic_quote_explanation_pipeline(testname)

    # nli_explanation_pipeline(testname)

    # similarity_histogram(testname)

    #Statistical Baselines comparing Random Pairs vs True Pairs
    # random_explanation_baseline(testname) #Semantic Baseline
    # nli_random_baseline(testname) #NLI Baseline
    # random_nli_calc(testname) 
    # random_semantic_calc(testname)

    # =====================================
    #Stage Two - THEME GENERATION
    # =====================================
    #Granularity options = document, section, corpus
    previous_granularity = "document"
    new_granularity = "corpus"
    stage = "theme_generation"
    testname = stage + "_" + previous_granularity + "_" + new_granularity

    experiment_folder = Path(f"experiments/{testname}")
    experiment_folder.mkdir(parents=True, exist_ok=True)

    log_folder = experiment_folder / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)

    print(f"Experiment name: {testname}")

    # theme_generation_pipeline(stage, previous_granularity, new_granularity)
    # =====================================
    #Stage Two - ANALYSIS
    # =====================================
    unique_theme_obj = unique_themes(experiment_folder / "output_files")
    # combine_json_files(experiment_folder / "output_files", experiment_folder / "combined_output.json")

    # semantic_projections(experiment_folder / "combined_output.json")


    # =====================================
    #Stage Three - Theme Development
    # # =====================================
    # prompt = ""
    # with open("prompts/grouping_themes_prompt.txt", "r", encoding="utf-8") as f:
    #     prompt = f.read()
    # schema = {
    #     "type": "object",
    #     "properties": {
    #         "codes": {
    #             "type": "array",
    #             "items": {
    #                 "type": "object",
    #                 "properties": {
    #                     "theme": {
    #                         "type": "string"
    #                     },
    #                     "description": {
    #                         "type": "string"
    #                     },
    #                     "subthemes": {
    #                         "type": "array",
    #                         "items": {
    #                             "type": "string"
    #                         }
    #                     }
    #                 },
    #                 "required": [
    #                     "theme",
    #                     "description",
    #                     "subthemes"
    #                 ],
    #                 "additionalProperties": False
    #             }
    #         }
    #     },
    #     "required": [
    #         "codes"
    #     ],
    #     "additionalProperties": False
    # }
    # theme_string = ""
    # for theme in unique_theme_obj:
    #     theme_string += f"- {theme}\n"
    # theme_response = individual_input(theme_string, prompt, schema, "theme_grouping_schema")
    # save_response(theme_response, experiment_folder, "theme_grouping_input.txt", processing_time=None)

    thematic_projections_method(experiment_folder /"output_files" / "combined_combined.json", experiment_folder / "output_files" / "theme_grouping_input.json")


main()
