import json
from pathlib import Path
def unique_themes(input_folder):
    # Folder containing your JSON files
    folder = Path(input_folder)
    print(f"Looking in: {folder.resolve()}")
    print(f"Folder exists: {folder.exists()}")

    all_themes = set() 

    # Loop through all JSON files
    for json_file in folder.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Extract themes from each document
            for item in data:
                if "theme" in item:
                    all_themes.add(item["theme"])

    # Print unique themes
    print("Unique themes:")
    for theme in sorted(all_themes):
        print(theme)

    # Optional: save to a file
    with open(folder / "unique_themes.txt", "w", encoding="utf-8") as f:
        json.dump(sorted(all_themes), f, indent=2, ensure_ascii=False)

    return sorted(all_themes)

# unique_themes()