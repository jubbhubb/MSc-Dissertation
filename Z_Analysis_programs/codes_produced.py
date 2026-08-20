import json
from pathlib import Path

folder = Path("experiments_v5/coding_section_deductive/output_files")


total = 0

for file in folder.glob("*.json"):
    with open(file, "r", encoding="utf-8") as f:
        codes = json.load(f)
    total += len(codes)
    # total += 1

print(f"Total number of codes: {total}")


# COMPARISON SCRIPT

# from pathlib import Path

# folder_inductive = Path("experiments_v5/coding_section_inductive/output_files")
# folder_deductive = Path("experiments_v5/coding_section_deductive/output_files")

# inductive_files = {file.name for file in folder_inductive.glob("*.json")}
# deductive_files = {file.name for file in folder_deductive.glob("*.json")}

# missing_from_deductive = inductive_files - deductive_files
# missing_from_inductive = deductive_files - inductive_files

# print(f"Inductive files: {len(inductive_files)}")
# print(f"Deductive files: {len(deductive_files)}")

# print("\nMissing from deductive:")
# for file in sorted(missing_from_deductive):
#     print(f"  {file}")

# print("\nMissing from inductive:")
# for file in sorted(missing_from_inductive):
#     print(f"  {file}")