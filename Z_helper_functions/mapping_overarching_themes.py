import json
from enum import Enum
from pathlib import Path

class Granularity(Enum):
    SECTION="section"
    DOCUMENT="document"
    CORPUS="corpus"

theme_transitions = [
        (Granularity.SECTION, Granularity.SECTION),
        (Granularity.SECTION, Granularity.DOCUMENT),
        # (Granularity.SECTION, Granularity.CORPUS),

        (Granularity.DOCUMENT, Granularity.SECTION),
        (Granularity.DOCUMENT, Granularity.DOCUMENT),
        (Granularity.DOCUMENT, Granularity.CORPUS),

        (Granularity.CORPUS, Granularity.SECTION),
        (Granularity.CORPUS, Granularity.CORPUS),
        ]

def theme_mapping(test_folder):
    theme_groups_folder = test_folder
    codes_folder = test_folder
    # Load files
    with open(test_folder / Path("output_files/theme_grouping_input.json"), "r", encoding="utf-8") as f:
        theme_groups = json.load(f)

    with open(test_folder / Path("combined_codes.json"), "r", encoding="utf-8") as f:
        codes = json.load(f)

    # Create lookup: subtheme -> overarching theme
    subtheme_lookup = {}
    for group in theme_groups:
        overarching = group["theme"]
        for subtheme in group["subthemes"]:
            subtheme_lookup[subtheme] = overarching

    # Add overarching theme to each code
    for code in codes:
        subtheme = code["theme"]
        code["overarching_theme"] = subtheme_lookup.get(subtheme)

    # Save result
    with open(test_folder / "codes_with_overarching_theme.json", "w", encoding="utf-8") as f:
        json.dump(codes, f, indent=4, ensure_ascii=False)

    with open(test_folder / "codes_with_overarching_theme.json", "r", encoding="utf-8") as f:
        codes = json.load(f)

    codes = [
        code for code in codes
        if code.get("overarching_theme") is not None
    ]

    with open(test_folder / "codes_filtered.json", "w", encoding="utf-8") as f:
        json.dump(codes, f, indent=4, ensure_ascii=False)