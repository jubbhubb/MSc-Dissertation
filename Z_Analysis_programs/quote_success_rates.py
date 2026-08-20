import json
from pathlib import Path


def calculate_quote_success_rates(results_folder):
    import json
    from pathlib import Path

    results_folder = Path(results_folder)

    file_results = {}
    all_results = []

    for json_file in results_folder.glob("*.json"):

        results = json.loads(
            json_file.read_text(encoding="utf-8")
        )

        valid_results = [
            r for r in results
            if "strict_match" in r
        ]

        if not valid_results:
            continue

        strict_full = sum(
            r["strict_match"]
            for r in valid_results
        )

        relaxed_full = sum(
            r["relaxed_match"]
            for r in valid_results
        )

        strict_coverage = sum(
            r["strict_matches"] / r["strict_total"]
            for r in valid_results
        ) / len(valid_results)

        relaxed_coverage = sum(
            r["relaxed_matches"] / r["relaxed_total"]
            for r in valid_results
        ) / len(valid_results)

        file_results[json_file.stem] = {
            "total_quotes": len(valid_results),

            "strict_full_rate": strict_full / len(valid_results),
            "relaxed_full_rate": relaxed_full / len(valid_results),

            "strict_coverage": strict_coverage,
            "relaxed_coverage": relaxed_coverage
        }

        all_results.extend(valid_results)


    total = len(all_results)

    overall = {
        "total_quotes": total,

        "strict_full_rate": sum(
            r["strict_match"]
            for r in all_results
        ) / total,

        "relaxed_full_rate": sum(
            r["relaxed_match"]
            for r in all_results
        ) / total,

        "strict_coverage": sum(
            r["strict_matches"] / r["strict_total"]
            for r in all_results
        ) / total,

        "relaxed_coverage": sum(
            r["relaxed_matches"] / r["relaxed_total"]
            for r in all_results
        ) / total
    }

    return {
        "per_file": file_results,
        "overall": overall
    }