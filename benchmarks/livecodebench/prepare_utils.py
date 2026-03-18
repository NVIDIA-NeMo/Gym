# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared LiveCodeBench data preparation utilities.

Two data sources are supported:

1. **Pre-prepared HF dataset** (``nvidia/nemotron-RL-coding-competitive_coding``):
   The code_gen server's validation data, built by running the official LCB runner.
   Contains test cases in ``verifier_metadata.unit_tests``. Only covers v5
   (Jul 2024–Feb 2025, 322 problems).

2. **Raw livecodebench HF dataset** (``livecodebench/code_generation_lite``):
   The original LCB dataset with private test cases encoded as base64+zlib+pickle.
   Covers all versions (v1–v6). Use this for splits not covered by the pre-prepared data.
"""

import base64
import json
import pickle
import zlib
from pathlib import Path
from typing import Optional


def _decode_test_cases(raw) -> list:
    """Decode test cases from the livecodebench HF dataset.

    Public test cases are plain JSON. Private test cases are base64+zlib+pickle encoded.
    """
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))


def prepare_from_hf_validation(
    output_path: Path,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    hf_repo: str = "nvidia/nemotron-RL-coding-competitive_coding",
    hf_filename: str = "validation.jsonl",
) -> Path:
    """Prepare LCB data from the pre-prepared code_gen validation dataset on HuggingFace.

    This dataset was built by the LCB runner (``livecodebench_accuracy_test_prep.py``)
    and contains correct test cases with fn_name. Only covers v5 (322 problems).
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    print(f"Downloading validation data from {hf_repo}...")
    source_path = hf_hub_download(hf_repo, hf_filename, repo_type="dataset")

    # Deduplicate (source has multiple rows per problem from the accuracy test)
    seen = set()
    rows = []
    with open(source_path) as f:
        for line in f:
            row = json.loads(line)
            pid = row["verifier_metadata"]["problem_id"]
            if pid in seen:
                continue
            seen.add(pid)

            prompt_input = row.get("responses_create_params", {}).get("input", [])
            question_content = ""
            for msg in prompt_input:
                if msg.get("role") == "user":
                    question_content = msg.get("content", "")
                    break

            rows.append({"question_content": question_content, "verifier_metadata": row["verifier_metadata"]})

    # Enrich with difficulty and filter by date range
    ds = load_dataset("livecodebench/code_generation_lite", "release_v5", split="test", revision="refs/pr/7")
    hf_map = {ex.get("question_id", ""): ex for ex in ds}

    enriched = []
    for row in rows:
        pid = row["verifier_metadata"]["problem_id"]
        hf_row = hf_map.get(pid)
        if hf_row:
            row["verifier_metadata"]["difficulty"] = hf_row.get("difficulty", "unknown")
            date = hf_row.get("contest_date", "")
            if date_from and date < date_from:
                continue
            if date_to and date >= date_to:
                continue
        enriched.append(row)

    return _write_rows(enriched, output_path)


def prepare_from_hf_raw(
    output_path: Path,
    release_version: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Path:
    """Prepare LCB data by decoding test cases directly from the livecodebench HF dataset.

    Works for any release version (v1–v6). Private test cases are decoded from
    base64+zlib+pickle encoding. fn_name is extracted from the metadata field.
    """
    from datasets import load_dataset

    print(f"Downloading LiveCodeBench {release_version} from HuggingFace...")
    ds = load_dataset("livecodebench/code_generation_lite", release_version, split="test", revision="refs/pr/7")

    rows = []
    for example in ds:
        contest_date = example.get("contest_date", "")
        if date_from and contest_date < date_from:
            continue
        if date_to and contest_date >= date_to:
            continue

        pub = _decode_test_cases(example.get("public_test_cases", ""))
        priv = _decode_test_cases(example.get("private_test_cases", ""))
        inputs = [tc["input"] for tc in pub] + [tc["input"] for tc in priv]
        outputs = [tc["output"] for tc in pub] + [tc["output"] for tc in priv]

        meta = example.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta) if meta else {}

        rows.append(
            {
                "question_content": example["question_content"],
                "verifier_metadata": {
                    "problem_id": example.get("question_id", ""),
                    "difficulty": example.get("difficulty", "unknown"),
                    "unit_tests": {
                        "inputs": inputs,
                        "outputs": outputs,
                        "fn_name": meta.get("func_name") or None,
                    },
                },
            }
        )

    return _write_rows(rows, output_path)


def _write_rows(rows: list, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} problems to {output_path}")
    return output_path
