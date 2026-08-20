import re
from pathlib import Path

def summarise_api_log(log_file):
    """
    Summarise an API log containing multiple call records.

    Returns:
        dict: Aggregated totals for processing time and token usage.
    """

    text = Path(log_file).read_text(encoding="utf-8")

    processing_times = [float(x) for x in re.findall(r"Processing time:\s*([\d.]+)s", text)]
    input_tokens = [int(x) for x in re.findall(r"Input tokens:\s*(\d+)", text)]
    output_tokens = [int(x) for x in re.findall(r"Output tokens:\s*(\d+)", text)]
    total_tokens = [int(x) for x in re.findall(r"Total tokens:\s*(\d+)", text)]

    summary = {
        "api_calls": len(processing_times),
        "total_processing_time_s": round(sum(processing_times), 2),
        "total_input_tokens": sum(input_tokens),
        "total_output_tokens": sum(output_tokens),
        "total_tokens": sum(total_tokens),
    }

    return summary

def summarise_report_log(log_file):
    """
    Summarise a report-production log.

    Returns:
        dict: Aggregated processing time and token usage.
    """

    text = Path(log_file).read_text(encoding="utf-8")

    input_tokens = [
        int(x) for x in re.findall(
            r"Input Tokens Used in report production:\s*(\d+)", text
        )
    ]

    output_tokens = [
        int(x) for x in re.findall(
            r"Output Tokens Used in report production:\s*(\d+)", text
        )
    ]

    processing_times = [
        float(x) for x in re.findall(
            r"Time taken:\s*([\d.]+)", text
        )
    ]

    summary = {
        "total_processing_time_s": round(sum(processing_times), 2),
        "total_input_tokens": sum(input_tokens),
        "total_output_tokens": sum(output_tokens),
        "total_tokens": sum(input_tokens) + sum(output_tokens),
    }

    return summary

# experiments_folder = "experiments_v6" #Deductive
experiments_folder = "experiments_v5" #Inductive

reasoning_pairs = [
    ("corpus", "corpus"),
    # ("corpus", "document"),
    ("corpus", "section"),

    ("document", "corpus"),
    ("document", "document"),
    ("document", "section"),

    # ("section", "corpus"),
    ("section", "document"),
    ("section", "section"),
]
for granularity1, granularity2 in reasoning_pairs:
    stage = f"/theme_development_theme_generation_inductive_{granularity1}_{granularity2}/logs/report_production_log.txt" #Inductive
    # stage = f"/subtheme_generation_deductive_{granularity1}_{granularity2}/logs/report_production_log.txt" #Deductive
    full_filename = experiments_folder + stage
    # summary = summarise_api_log(full_filename)
    summary = summarise_report_log(full_filename)
    print(f"Test {granularity1}, {granularity2}, inductive reasoning")
    # print(f"API calls: {summary['api_calls']}") 
    print(f"Total processing time: {summary['total_processing_time_s']} s")
    print(f"Total input tokens: {summary['total_input_tokens']:,}")
    print(f"Total output tokens: {summary['total_output_tokens']:,}")
    print(f"Total tokens: {summary['total_tokens']:,}")