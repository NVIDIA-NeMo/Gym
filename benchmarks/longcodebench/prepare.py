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

"""Prepare LongCodeBench (LongCodeQA) evaluation data for NeMo Gym.

LongCodeBench is a multi-choice QA benchmark over long code contexts, ported
1-to-1 from the NeMo Skills `longcodebench` dataset. Each row's `question`
field is the long code prompt plus the postfix that instructs the model to
emit `Answer: \\boxed{X}`. The shared `benchmarks/prompts/generic_default.yaml`
template (`user: "{question}"`) wraps it as a single user message, mirroring
Skills' `prompt_format=openai` behaviour.

The resulting Gym JSONL is consumed by the `mcqa` resource server with
`grading_mode=strict_single_letter_boxed`. We provide empty-text option dicts
purely to populate the server's `allowed_letters` set; the option text is not
used for grading because the postfix forces a `\\boxed{X}` answer.

Skills' prepare also stores a `n_tokens_cl100k_base` field counted with
tiktoken. The mcqa verifier never reads it; we omit it on the Gym side to
avoid pulling tiktoken into Gym's main dependency set just for one
benchmark's metadata column.
"""

import json
import uuid
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "longcodebench_benchmark.jsonl"
OPTION_LETTERS = ("A", "B", "C", "D")

POSTFIX = (
    "\n\nThe last line of your response should be in the following format: "
    "'Answer: \\boxed{A/B/C/D}' (e.g. 'Answer: \\boxed{A}')."
)


def prepare() -> Path:
    """Download LongCodeBench LongCodeQA from HuggingFace and write Gym JSONL."""
    from datasets import load_dataset

    print("Downloading LongCodeBench LongCodeQA from HuggingFace...")
    ds = load_dataset("json", data_files="hf://datasets/Steefano/LCB/LongCodeQA.zip")
    data = ds["train"]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Empty-text option dicts: the mcqa server only consumes the option *keys*
    # for `strict_single_letter_boxed` grading; option text is irrelevant since
    # the prompt postfix forces the model to emit `\boxed{<letter>}`.
    options = [{letter: ""} for letter in OPTION_LETTERS]

    rows = []
    for entry in data:
        question = entry["prompt"].strip() + POSTFIX
        row = {
            "question": question,
            "options": options,
            "expected_answer": entry["correct_letter"],
            "grading_mode": "strict_single_letter_boxed",
            "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, question)),
            "repo": entry["repo"],
            "prompt_goal": entry["prompt_goal"],
            "is_hard": entry["is_hard"],
        }
        rows.append(json.dumps(row) + "\n")

    with open(OUTPUT_FPATH, "w", encoding="utf-8") as f:
        f.writelines(rows)

    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
