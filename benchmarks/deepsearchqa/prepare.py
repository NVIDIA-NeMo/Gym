# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import csv
import io
import json
import urllib.request
from pathlib import Path


URL = "https://huggingface.co/datasets/google/deepsearchqa/resolve/main/DSQA-full.csv"
OUTPUT = Path(__file__).parent / "data" / "deepsearchqa_benchmark.jsonl"


def prepare() -> Path:
    data = urllib.request.urlopen(URL).read().decode()
    rows = list(csv.DictReader(io.StringIO(data)))
    OUTPUT.parent.mkdir(exist_ok=True)
    with OUTPUT.open("w") as output:
        for example_id, row in enumerate(rows):
            row["example_id"] = str(example_id)
            row["responses_create_params"] = {"input": [{"role": "user", "content": row["problem"]}]}
            output.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} tasks to {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    prepare()
