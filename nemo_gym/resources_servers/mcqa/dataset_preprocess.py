# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
from typing import Any, Iterable
from uuid import uuid4


def write_mcqa_jsonl(rows: Iterable[dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_row(
    question: str,
    choices: list[str],
    answer_letter: str,
    *,
    uuid: str | None = None,
    grading_mode: str = "strict_single_letter_boxed",
    metadata: dict[str, Any] | None = None,
) -> dict:
    """Build a dataset row shaped as an MCQARunRequest.

    Produces a dict that validates as `MCQARunRequest` used by the MCQA server:
    - responses_create_params: OpenAI-style request with the prompt text only (no metadata)
    - metadata (top-level, optional): arbitrary dict; pass-through only
    - options (top-level): list of {Letter: Text}
    - expected_answer (top-level): single uppercase letter
    - grading_mode (top-level): selector for the verifier parsing rules
    - uuid (top-level, optional): passthrough identifier
    """
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    content = question.strip() + "\n" + "\n".join(f"{letters[i]}: {choices[i]}" for i in range(len(choices)))

    options_list = [{letters[i]: choices[i]} for i in range(len(choices))]

    row: dict = {
        # Required by BaseRunRequest
        "responses_create_params": {
            "input": [
                {"role": "user", "content": content},
            ],
        },
        # Top-level fields from MCQARunRequest (optional in server but included here)
        "options": options_list,
        "expected_answer": answer_letter.upper(),
        "grading_mode": grading_mode,
    }

    # Pass-through metadata without combining with options/expected_answer
    # Always include metadata as a top-level dict; do not mix with options/expected
    row["metadata"] = metadata if metadata is not None else {}

    if uuid is None:
        row["uuid"] = str(uuid4())
    else:
        row["uuid"] = uuid

    return row


if __name__ == "__main__":  # pragma: no cover
    # Example: build a tiny file
    rows = [
        build_row(
            "Which of the following statements about the base pairing in DNA is correct?",
            [
                "Adenine (A) pairs with Thymine (T) via three hydrogen bonds.",
                "Guanine (G) pairs with Cytosine (C) via two hydrogen bonds.",
                "Adenine (A) pairs with Thymine (T) via two hydrogen bonds.",
                "Guanine (G) pairs with Cytosine (C) via three hydrogen bonds.",
            ],
            "C",
        ),
    ]
    write_mcqa_jsonl(rows, "data/mcqa_example.jsonl")
