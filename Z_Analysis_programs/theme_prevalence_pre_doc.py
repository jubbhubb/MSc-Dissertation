import json
import csv
from pathlib import Path
from collections import Counter
import sys

# Folder containing JSON files
folder = Path("experiments_v6/coding_document_deductive/output_files")
results = []
all_themes = set()

# Process each file
for json_file in sorted(folder.glob("*.json")):
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        theme_counts = Counter(item.get("theme", "Missing") for item in data)
        total = sum(theme_counts.values())

        all_themes.update(theme_counts.keys())

        results.append({
            "file": json_file.stem,
            "total": total,
            "counts": theme_counts
        })

    except Exception as e:
        print(f"Skipping {json_file.name}: {e}")

# Sort themes alphabetically for consistent columns
all_themes = sorted(all_themes)

# Print results
for r in results:
    print(f"\n{r['file']} ({r['total']} coded excerpts)")
    print("-" * 40)
    for theme in all_themes:
        count = r["counts"].get(theme, 0)
        pct = (count / r["total"] * 100) if r["total"] else 0
        print(f"{theme:<20} {count:>3} ({pct:5.1f}%)")

# Save to CSV
with open("experiments_v6/coding_document_deductive/" + "theme_prevalence_by_document.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    header = ["Document", "Total Excerpts"]
    for theme in all_themes:
        header.extend([f"{theme}_Count", f"{theme}_Percent"])
    writer.writerow(header)

    for r in results:
        row = [r["file"], r["total"]]
        for theme in all_themes:
            count = r["counts"].get(theme, 0)
            pct = (count / r["total"] * 100) if r["total"] else 0
            row.extend([count, round(pct, 2)])
        writer.writerow(row)

print("\nResults saved to theme_prevalence_by_document.csv")