from preprocessing import create_experiment_folder
from api_calls import process_folder
from datetime import datetime
from aggregation import recombine_split_documents, combine_to_corpus

def main():
    create_experiment_folder("testing_012", lowercase=True, granularity="small")
    process_folder("experiments/testing_012")
    recombine_split_documents("experiments/testing_012/input_files", save_combined=True, output_path="experiments/testing_012/recombined_files")
    combine_to_corpus("experiments/testing_012/recombined_files", "experiments/testing_012/corpus_files")
main()