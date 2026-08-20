import json
import re
from pathlib import Path

def validate_quotes(experiment_folder):
    """
    Validate quotes in JSON files against matching TXT files.

    Folder structure expected:

    experiment_folder/
        output_files/
            file1.json
        input_files/
            file1.txt

    Outputs:
        experiment_folder/
            quote_validation_results.json
            quote_validation_results/
                file1.json
                file2.json

    Returns:
    List of dictionaries containing strict/relaxed match results
    and match counts for grouped quotes.
    """

    experiment_folder = Path(experiment_folder)

    JSON_FOLDER = experiment_folder / "output_files"
    TXT_FOLDER = experiment_folder / "input_files"

    RESULTS_FOLDER = experiment_folder / "quote_validation_results"
    RESULTS_FOLDER.mkdir(exist_ok=True)

    def quote_to_regex(quote, normaliser):
        quote = normaliser(quote)

        pattern = re.escape(quote)

        return pattern.replace(
            r"\.\.\.",
            r".*?"
        )
    

    def split_quotes(quote):
        """
        Split grouped quotes separated by semicolons.

        Keeps the original JSON unchanged while allowing each
        quoted segment to be validated independently.
        """
        return [
            q.strip()
            for q in quote.split(";")
            if q.strip()
        ]

    def evaluate_match(document, quote, normaliser):
        """
        Evaluate one or more quotes contained in a single string.

        Returns:
            {
                "matched": bool,
                "matches": int,
                "total": int
            }
        """

        document = normaliser(document)

        quotes = split_quotes(quote)

        matches = sum(
            bool(re.search(
                quote_to_regex(q, normaliser),
                document
            ))
            for q in quotes)
        return {
            "matched": matches == len(quotes),
            "matches": matches,
            "total": len(quotes)}

    def normalise(text):
        text = text.lower()
        text = text.replace("…", "...")
        text = re.sub(r"[\"'“”‘’]", "", text)
        text = re.sub(r"[^\w\s.]","",text)
        text = re.sub(r"\s+"," ",text)
        return text.strip()


    def normalise_relaxed(text):
        """More forgiving transcript normalisation."""
        text = normalise(text)

        fillers = [
            "like",
            "um",
            "uh",
            "yeah",
            "you know"
        ]

        for filler in fillers:
            text = re.sub(
                rf"\b({filler})\s+\1\b",
                r"\1",
                text
            )

        return text


    def strict_match(document, quote):
        return evaluate_match(
            document,
            quote,
            normalise
        )


    def relaxed_match(document, quote):
        return evaluate_match(
            document,
            quote,
            normalise_relaxed
        )

    results = []

    for json_file in JSON_FOLDER.glob("*.json"):

        # Results only for this individual document
        file_results = []

        txt_file = TXT_FOLDER / f"{json_file.stem}.txt"

        if not txt_file.exists():
            error_result = {
                "file": json_file.name,
                "error": "No matching text file"
            }

            results.append(error_result)
            file_results.append(error_result)

            continue


        document = txt_file.read_text(
            encoding="utf-8"
        )

        data = json.loads(
            json_file.read_text(
                encoding="utf-8"
            )
        )
        if isinstance(data, dict):
            data = [data]
        for item in data:
            quote = item.get("quote")
            if not quote:
                continue
            strict = strict_match(document, quote)
            relaxed = relaxed_match(document, quote)

            result = {
                "file": json_file.name,
                "code": item.get("code"),
                "quote": quote,

                "strict_match": strict["matched"],
                "strict_matches": strict["matches"],
                "strict_total": strict["total"],

                "relaxed_match": relaxed["matched"],
                "relaxed_matches": relaxed["matches"],
                "relaxed_total": relaxed["total"]
            }

            # Add to both combined and individual results
            results.append(result)
            file_results.append(result)


        # Save individual document results
        individual_output = RESULTS_FOLDER / json_file.name

        individual_output.write_text(
            json.dumps(file_results, indent=2),
            encoding="utf-8"
        )


    # Save combined results
    output_path = experiment_folder / "quote_validation_results.json"

    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )
    return results