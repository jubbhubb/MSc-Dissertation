Usage of this program requires an API key for the OpenAI.

File structure required. 
./source_files - Containing the .txt versions of the source files to be produced.
.version_3.py - Main function call
./Z_helper_functions - Helper functions required for report generation containing the following files. 
- __init__.py 
- aggregation.py
- api_calls
- final_analysis.py
- mapping_overarching_themes.py
- preprocessing.py
- save_response.py
- theme_generation_v2.py
- theme_grouping_inductive.py
- unique_themes_finder.py

./Z_Analysis_programs contains various analysis metrics ran throughout the testing of this program, however many of them are now deprecated following final refactoring of this project

version_1.py, and version_2.py are both deprecated following final refactoring of the project but remain for posterity.

Care needs to be taken in that the preprocessing is currently hard coded to remove the first 6 lines of documents, as that was the file format used in the project.

Currently the version_3.py is hardcoded to only consider the Corpus-Corpus route through both inductive and deductive processes to reduce time costs and limit token usage. A single run through of this code using chatgpt 5.5 costs around $2.50

./reports contains markdown files of outputs of an example full run of the project which were used for analysis in the project. 

