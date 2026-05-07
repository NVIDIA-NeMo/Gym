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
"""Prepare Arena Hard v2 benchmark data.

Downloads the arena-hard-auto v2 question set and the two
category-specific baseline model-answer files, joins them by ``uid``,
and writes one row per question with the fields the ``arena_judge``
resources server consumes at the top level (``question``,
``baseline_answer``, ``category``, ``uid``, ``subcategory``):

- Questions:
  https://github.com/lmarena/arena-hard-auto/blob/main/data/arena-hard-v2.0/question.jsonl
- ``hard_prompt`` baseline (o3-mini-2025-01-31):
  https://github.com/lmarena/arena-hard-auto/blob/main/data/arena-hard-v2.0/model_answer/o3-mini-2025-01-31.jsonl
- ``creative_writing`` baseline (gemini-2.0-flash-001):
  https://github.com/lmarena/arena-hard-auto/blob/main/data/arena-hard-v2.0/model_answer/gemini-2.0-flash-001.jsonl
"""

import json
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "arena_hard_v2_benchmark.jsonl"

URL_QUESTIONS = "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/data/arena-hard-v2.0/question.jsonl"
URL_BASELINE_HARD_PROMPT = (
    "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/"
    "data/arena-hard-v2.0/model_answer/o3-mini-2025-01-31.jsonl"
)
URL_BASELINE_CREATIVE_WRITING = (
    "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/"
    "data/arena-hard-v2.0/model_answer/gemini-2.0-flash-001.jsonl"
)

CATEGORY_BASELINES = {
    "hard_prompt": URL_BASELINE_HARD_PROMPT,
    "creative_writing": URL_BASELINE_CREATIVE_WRITING,
}


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
    """Download and write ``arena_hard_v2_benchmark.jsonl``. Returns the path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading questions from {URL_QUESTIONS} ...")
    questions_fpath = DATA_DIR / "question.jsonl"
    urllib.request.urlretrieve(URL_QUESTIONS, questions_fpath)

    # uid -> {category -> answer_text}
    baseline_answers: dict[str, dict[str, str]] = {}
    for category, url in CATEGORY_BASELINES.items():
        print(f"Downloading {category} baseline from {url} ...")
        baseline_fpath = DATA_DIR / f"baseline_{category}.jsonl"
        urllib.request.urlretrieve(url, baseline_fpath)
        with open(baseline_fpath, "r", encoding="utf-8") as fin:
            for line in fin:
                row = json.loads(line)
                uid = row["uid"]
                baseline_answers.setdefault(uid, {})[category] = _extract_answer_text(row)

    count = 0
    with open(questions_fpath, "r", encoding="utf-8") as fin, open(OUTPUT_FPATH, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            # arena-hard-auto stores the prompt under ``prompt`` but the
            # resource server + prompt template expect ``question``.
            row["question"] = row.pop("prompt")
            category = row["category"]
            # Fail loudly if a question's baseline answer is missing — a
            # silent skip would shrink the evaluation set.
            row["baseline_answer"] = baseline_answers[row["uid"]][category]
            fout.write(json.dumps(row) + "\n")
            count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
