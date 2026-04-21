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

Downloads questions + category-specific baseline answers from the
arena-hard-auto repo, matching Skills'
``nemo_skills/dataset/arena-hard-v2/prepare.py`` logic exactly:

- Questions from ``lmarena/arena-hard-auto/data/arena-hard-v2.0/question.jsonl``
- ``hard_prompt`` baseline: o3-mini-2025-01-31
- ``creative_writing`` baseline: gemini-2.0-flash-001

Each output row carries the fields the ``arena_judge`` resources server
consumes (``question``, ``baseline_answer``, ``category``, ``uid``) at
the top level.
"""

import json
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "arena_hard_v2_benchmark.jsonl"

# Source URLs — must match Skills' arena-hard-v2 prepare.py byte-for-byte.
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

    Matches Skills' ``extract_answer_text`` in arena-hard-v2/prepare.py.
    Assistant ``content`` can be either a plain string or a dict with an
    ``answer`` key (the arena-hard-auto repo uses both shapes).
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
            # Skills renames `prompt` → `question` for template compatibility.
            row["question"] = row.pop("prompt")
            category = row["category"]
            # Fail loudly if a question's baseline answer is missing — matches
            # Skills' KeyError semantics.
            row["baseline_answer"] = baseline_answers[row["uid"]][category]
            fout.write(json.dumps(row) + "\n")
            count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
