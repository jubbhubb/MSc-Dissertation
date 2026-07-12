from cosine_sim_all import cosine_similarity_between_documents
from preprocessing import create_experiment_folder
from api_calls import process_folder
from datetime import datetime
from aggregation import recombine_split_documents, combine_to_corpus, combine_json_by_document, combine_json_files

def main():
    input_tokens_start = int(open('input_token_count.txt', 'r').read())
    output_tokens_start = int(open('output_token_count.txt', 'r').read())
    start_time = datetime.now()
    prompt = ""
    with open("prompt.txt", "r", encoding="utf-8") as f:
        prompt = f.read()

    print(f"Using prompt:\n{prompt}\n")

    #Set up experiment parameters
    stage = "coding"
    granularity = "section"
    lowercase = True

    testname = stage + "_" + granularity

    print(f"Experiment name: {testname}")

    create_experiment_folder(testname, lowercase=lowercase, granularity=granularity, prompt=prompt)
    folders_created = datetime.now()

    process_folder(f"experiments/{testname}", prompt)
    folder_processed = datetime.now()

    recombine_split_documents(f"experiments/{testname}/input_files", save_combined=True, output_path=f"experiments/{testname}/recombined_files")
    combine_to_corpus(f"experiments/{testname}/recombined_files", f"experiments/{testname}/corpus_files")
    recombine_split_documents(f"experiments/{testname}/input_files", save_combined=True, output_path=f"experiments/{testname}/recombined_files")
    combine_json_by_document(f"experiments/{testname}/output_files", output_folder=f"experiments/{testname}/combined_json")
    sections_recombined = datetime.now()

    cosine_similarity_between_documents(f"experiments/{testname}")
    cosine_calculated = datetime.now()
    input_tokens_end = int(open('input_token_count.txt', 'r').read())
    output_tokens_end = int(open('output_token_count.txt', 'r').read())
    with open(f"experiments/{testname}/logs/overview_log.txt", "a", encoding="utf-8") as log:
        log.write(f"Experiment started: {start_time}\n")
        log.write(f"Folders created: {folders_created}\n")
        log.write(f"Time taken to create folders: {folders_created - start_time}\n")
        log.write(f"Folder processed (ALL API CALLS): {folder_processed}\n")
        log.write(f"Time taken to process folder: {folder_processed - folders_created}\n")
        log.write(f"Sections recombined: {sections_recombined}\n")
        log.write(f"Time taken to recombine sections: {sections_recombined - folder_processed}\n")
        log.write(f"Cosine calculated: {cosine_calculated}\n")
        log.write(f"Time taken to calculate cosine similarity: {cosine_calculated - sections_recombined}\n")
        log.write(f"Total time taken: {cosine_calculated - start_time}\n")
        log.write(f"Input tokens used: {input_tokens_end - input_tokens_start}\n")
        log.write(f"Output tokens used: {output_tokens_end - output_tokens_start}\n")
        log.write(f"Total tokens used: {(input_tokens_end - input_tokens_start) + (output_tokens_end - output_tokens_start)}\n")

main()