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
from typing import Callable, Optional


# From LiveCodeBench lcb_runner/prompts/code_generation.py — tells the model which code style to use
_FORMATTING_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the problem"
    " and enclose your code within delimiters."
)
_FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout"
    " (do not directly test on the sample inputs). Enclose your code within delimiters"
    " as follows. Ensure that when the python program runs, it reads the inputs,"
    " runs the algorithm and writes output to STDOUT."
)


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


def _add_prompt_fields(row: dict, starter_code: str) -> None:
    """Add formatting_message and starter_code fields for prompt templating.

    Matches the logic in Skills' ``nemo_skills/dataset/livecodebench/prepare.py::clean_data()``.
    If ``starter_code`` is non-empty, the model is told to use it (LeetCode functional style).
    Otherwise, the model is told to read from stdin (Codeforces/Atcoder style).
    """
    if starter_code:
        row["formatting_message"] = _FORMATTING_WITH_STARTER_CODE
        row["starter_code"] = f"```python\n{starter_code}\n```"
    else:
        row["formatting_message"] = _FORMATTING_WITHOUT_STARTER_CODE
        row["starter_code"] = "```python\n# YOUR CODE HERE\n```"


def _add_prompt_fields_cascade(row: dict, starter_code: str) -> None:
    """Add formatting_message and starter_code fields for prompt templating.

    Matches the logic in Skills' ``nemo_skills/dataset/livecodebench/prepare.py::clean_data()``.
    If ``starter_code`` is non-empty, the model is told to use it (LeetCode functional style).
    Otherwise, the model is told to read from stdin (Codeforces/Atcoder style).
    """
    if starter_code:
        row["starter_code"] = (
            f"\n\nSolve the problem starting with the provided function header.\n\nFunction header:\n```\n{starter_code}\n```"
        )
    else:
        row["starter_code"] = ""


def prepare_from_hf_raw(
    output_path: Path,
    release_version: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    _add_prompt_fields_fn: Callable = _add_prompt_fields,
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

        row = {
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
        _add_prompt_fields_fn(row, example.get("starter_code", ""))
        rows.append(row)

    return _write_rows(rows, output_path)


def _write_rows(rows: list, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} problems to {output_path}")
    return output_path
