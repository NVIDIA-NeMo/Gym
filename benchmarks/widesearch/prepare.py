# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import csv
import io
import json
import urllib.request
from pathlib import Path


DATASET_ROOT = "https://huggingface.co/datasets/ByteDance-Seed/WideSearch/resolve/main"
OUTPUT = Path(__file__).parent / "data" / "widesearch_benchmark.jsonl"


def download(path: str) -> str:
    request = urllib.request.Request(f"{DATASET_ROOT}/{path}", headers={"User-Agent": "nemo-gym"})
    return urllib.request.urlopen(request).read().decode("utf-8-sig")


def prepare() -> Path:
    rows = [json.loads(line) for line in download("widesearch.jsonl").splitlines()]
    OUTPUT.parent.mkdir(exist_ok=True)
    with OUTPUT.open("w") as output:
        for row in rows:
            evaluation = json.loads(row["evaluation"])
            gold_csv = download(f"widesearch_gold/{row['instance_id']}.csv")
            gold_answer = list(csv.DictReader(io.StringIO(gold_csv)))
            task = {
                **row,
                "evaluation": evaluation,
                "gold_answer": gold_answer,
                "responses_create_params": {"input": [{"role": "user", "content": row["query"]}]},
            }
            output.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} tasks to {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    prepare()
