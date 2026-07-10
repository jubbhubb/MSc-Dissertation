from preprocessing import create_experiment_folder
from api_calls import process_folder
from datetime import datetime
from aggregation import recombine_split_documents, combine_to_corpus, combine_json_by_document, combine_json_files

def main():
    testname = "small_sections_step1_test_3"
    # create_experiment_folder(testname, lowercase=True, granularity="section")
    # process_folder(f"experiments/{testname}")
    # recombine_split_documents(f"experiments/{testname}/input_files", save_combined=True, output_path=f"experiments/{testname}/recombined_files")
    # # combine_to_corpus(f"experiments/{testname}/recombined_files", f"experiments/{testname}/corpus_files")
    # recombine_split_documents(f"experiments/{testname}/input_files", save_combined=True, output_path=f"experiments/{testname}/recombined_files")
    combine_json_by_document(f"experiments/{testname}/output_files", output_folder=f"experiments/{testname}/combined_json")

main()