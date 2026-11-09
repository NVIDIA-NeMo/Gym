# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert prepared native ClawEval references to harness-independent typed tasks."""

import argparse
import json
from pathlib import Path

from resources_servers.claweval.task_data import TaskData


def convert(source: Path, output: Path) -> None:
    """Preserve native task IDs, digests and public prompts without carrying an agent reference."""
    records = []
    for line in source.read_text().splitlines():
        row = json.loads(line)
        metadata = row["verifier_metadata"]
        data = TaskData(**metadata, prompt=row["responses_create_params"]["input"][0]["content"])
        # Dataset collation creates the typed envelope from the configured taskset.
        records.append(data.model_dump_json())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(record + "\n" for record in records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.input, args.output)
