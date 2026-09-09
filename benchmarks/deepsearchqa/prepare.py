# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import csv
import io
import json
import urllib.request
from pathlib import Path


url = "https://huggingface.co/datasets/google/deepsearchqa/resolve/main/DSQA-full.csv"
data = urllib.request.urlopen(url).read().decode()
output = Path(__file__).parent / "data" / "deepsearchqa_benchmark.jsonl"
output.parent.mkdir(exist_ok=True)
with output.open("w") as file:
    for example_id, row in enumerate(csv.DictReader(io.StringIO(data))):
        row["example_id"] = str(example_id)
        row["agent_ref"] = {"type": "responses_api_agents", "name": "deepsearchqa"}
        row["responses_create_params"] = {"input": [{"role": "user", "content": row["problem"]}]}
        file.write(json.dumps(row) + "\n")
print(f"wrote {sum(1 for _ in output.open())} tasks to {output}")
