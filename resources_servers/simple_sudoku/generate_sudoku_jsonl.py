#!/usr/bin/env python3
import json
import random
import sys


def replicate_jsonl(template_file: str, num_lines: int, output_file: str):
    """Read template JSONL and generate multiple instances."""

    # Read the template
    with open(template_file, "r") as f:
        template = json.loads(f.readline().strip())

    with open(output_file, "w") as f:
        for i in range(num_lines):
            line_data = template.copy()
            scale = random.choice([4, 9])
            if scale == 9:
                clues = random.randint(16, 48)
            else:
                clues = random.randint(6, 12)

            line_data["clues"] = clues
            line_data["scale"] = scale

            f.write(json.dumps(line_data) + "\n")

    print(f"Replicated template {num_lines} times to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python replicate_jsonl.py <template_file> <num_lines> <output_file>"
        )
        print(
            "Example: python replicate_jsonl.py data/simple_sudoku.jsonl 100 data/sudoku_batch.jsonl"
        )
        sys.exit(1)

    template_file, num_lines, output_file = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    replicate_jsonl(template_file, num_lines, output_file)
