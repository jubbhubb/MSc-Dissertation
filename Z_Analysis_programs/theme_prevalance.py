import json
from pathlib import Path
from collections import Counter
import sys

# Folder containing the JSON files
folder = Path("experiments_v5/theme_development_theme_generation_inductive_section_document/output_files")

theme_counts = Counter()
total_codes = 0
files_processed = 0

for json_file in folder.glob("*.json"):
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                theme = item.get("theme", "Missing")
                theme_counts[theme] += 1
                total_codes += 1

        files_processed += 1

    except Exception as e:
        print(f"Skipping {json_file.name}: {e}")

print(f"\nProcessed {files_processed} JSON files")
print(f"Total coded excerpts: {total_codes}\n")

print("Theme prevalence:")
print("-" * 35)

for theme, count in theme_counts.most_common():
    percentage = (count / total_codes) * 100 if total_codes else 0
    print(f"{theme:<20} {count:>5} ({percentage:5.1f}%)")