import re
from pathlib import Path

log_file = Path("experiments/small_sections_step1_test_3/logs/run_log.txt")

text = log_file.read_text(encoding="utf-8")

# Find all token counts
input_tokens = [int(x) for x in re.findall(r"Input tokens:\s*(\d+)", text)]
output_tokens = [int(x) for x in re.findall(r"Output tokens:\s*(\d+)", text)]

total_input = sum(input_tokens)
total_output = sum(output_tokens)

print(f"Number of tests: {len(input_tokens)}")
print(f"Total input tokens: {total_input:,}")
print(f"Total output tokens: {total_output:,}")
print(f"Total tokens: {total_input + total_output:,}")