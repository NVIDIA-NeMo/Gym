# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import csv
import io
import json
import urllib.request
import zipfile
from pathlib import Path


url = "https://www.kaggle.com/api/v1/datasets/download/deepmind/deepsearchqa"
data = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read())).read("DSQA-full.csv").decode()
output = Path(__file__).parent / "data" / "example.jsonl"
output.parent.mkdir(exist_ok=True)
with output.open("w") as file:
    for row in list(csv.DictReader(io.StringIO(data)))[:5]:
        row["agent_ref"] = {"type": "responses_api_agents", "name": "deepsearchqa"}
        row["responses_create_params"] = {"input": [{"role": "user", "content": row["problem"]}]}
        file.write(json.dumps(row) + "\n")
print(f"wrote {sum(1 for _ in output.open())} tasks to {output}")
