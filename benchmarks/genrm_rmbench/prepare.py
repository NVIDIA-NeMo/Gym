# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare all nine public RM-Bench style comparisons in both candidate orders."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from resources_servers.genrm_eval.prepare import atomic_output, make_task


COMMIT = "73c52d7b27b361621361ec959ac6c0eb9bb7a689"  # pragma: allowlist secret
URL = f"https://raw.githubusercontent.com/THU-KEG/RM-Bench/{COMMIT}/data/total_dataset.json"
OUTPUT = Path(__file__).parent / "data/rmbench_validation.jsonl"


def prepare(source: Path | str | None = None, output: Path | str = OUTPUT) -> Path:
    output = Path(output)
    if source and Path(source).resolve() == output.resolve():
        raise ValueError("Source and output must be different files")
    if source:
        data = json.loads(Path(source).read_text())
    else:
        with urlopen(URL, timeout=120) as response:
            data = json.load(response)
    with atomic_output(output) as stream:
        for index, item in enumerate(data):
            if len(item["chosen"]) != 3 or len(item["rejected"]) != 3:
                raise ValueError(f"Expected three styles per response in row {index}")
            for chosen_style, chosen in enumerate(item["chosen"]):
                for rejected_style, rejected in enumerate(item["rejected"]):
                    for swap in (False, True):
                        row = make_task(
                            f"{item['domain']}:{item['id']}:{chosen_style}:{rejected_style}:{int(swap)}",
                            [{"role": "user", "content": item["prompt"]}],
                            rejected if swap else chosen,
                            chosen if swap else rejected,
                            {
                                "correct": "second" if swap else "first",
                                "domain": item["domain"],
                                # Difficulty scoring: 0=concise, 1=detailed plain text, 2=detailed Markdown.
                                "response_1_style_value": rejected_style if swap else chosen_style,
                                "response_2_style_value": chosen_style if swap else rejected_style,
                            },
                        )
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Optional downloaded total_dataset.json.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(prepare(args.source, args.output))
