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
"""Prepare Arena Hard (v0.1) benchmark data.

Downloads the arena-hard-auto v0.1 question set and the single
``gpt-4-0314`` baseline model-answer file, joins them by ``uid``, and
writes one row per question with the fields the ``arena_judge``
resources server consumes at the top level (``question``,
``baseline_answer``, ``uid``):

- Questions:
  https://github.com/lmarena/arena-hard-auto/blob/main/data/arena-hard-v0.1/question.jsonl
- Baseline (gpt-4-0314):
  https://github.com/lmarena/arena-hard-auto/blob/main/data/arena-hard-v0.1/model_answer/gpt-4-0314.jsonl

Note: arena-hard v0.1 has no real sub-categories — the upstream
``category`` field is just the dataset version string ("arena-hard-v0.1").
We drop it so ``arena_judge`` falls through to its ``default_category``
(``hard_prompt``) and uses the standard arena-hard judge prompt.
"""

import json
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "arena_hard_benchmark.jsonl"

URL_QUESTIONS = "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/data/arena-hard-v0.1/question.jsonl"
URL_BASELINE = (
    "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/data/arena-hard-v0.1/model_answer/gpt-4-0314.jsonl"
)


def _extract_answer_text(data: dict) -> str:
    """Extract the assistant answer from a baseline model's JSONL row.

    The arena-hard-auto baseline files use both shapes for the assistant
    ``content``: a plain string or a dict with an ``answer`` key.
    """
    for msg in data["messages"]:
        if msg["role"] == "assistant":
            content = msg["content"]
            return content["answer"] if isinstance(content, dict) else content
    raise ValueError("No assistant message found in the baseline row.")


def prepare() -> Path:
    """Download and write ``arena_hard_benchmark.jsonl``. Returns the path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading questions from {URL_QUESTIONS} ...")
    questions_fpath = DATA_DIR / "question.jsonl"
    urllib.request.urlretrieve(URL_QUESTIONS, questions_fpath)

    print(f"Downloading baseline from {URL_BASELINE} ...")
    baseline_fpath = DATA_DIR / "baseline_gpt-4-0314.jsonl"
    urllib.request.urlretrieve(URL_BASELINE, baseline_fpath)

    # uid -> answer_text
    baseline_answers: dict[str, str] = {}
    with open(baseline_fpath, "r", encoding="utf-8") as fin:
        for line in fin:
            row = json.loads(line)
            baseline_answers[row["uid"]] = _extract_answer_text(row)

    count = 0
    with open(questions_fpath, "r", encoding="utf-8") as fin, open(OUTPUT_FPATH, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            # arena-hard-auto stores the prompt under ``prompt`` but the
            # resource server + prompt template expect ``question``.
            row["question"] = row.pop("prompt")
            # arena-hard v0.1 has no real sub-categories — the upstream
            # ``category`` field is just the dataset version string. Drop
            # it so ``arena_judge`` falls through to its ``default_category``
            # (``hard_prompt``) and uses the standard arena-hard judge prompt.
            row.pop("category", None)
            # Fail loudly if a question's baseline answer is missing — a
            # silent skip would shrink the evaluation set.
            row["baseline_answer"] = baseline_answers[row["uid"]]
            fout.write(json.dumps(row) + "\n")
            count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
