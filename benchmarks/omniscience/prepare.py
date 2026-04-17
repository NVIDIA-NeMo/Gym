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

"""Prepare AA-Omniscience evaluation data for NeMo Gym.

Downloads the AA-Omniscience-Public dataset from HuggingFace and converts
to Gym JSONL format compatible with the omniscience resource server.

Output is raw data (no prompts baked in). Use prompt_config at rollout time
to specify the prompt, or ng_materialize_prompts to produce RL-ready data.

Source: https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public
"""

import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "omniscience_benchmark.jsonl"

SYSTEM_PROMPT = (
    "You are answering questions about {domain}, and in particular {topic}. "
    "You will be given a question, answer with JUST the answer (no explanation). "
    "If you do not know the answer, or you need more context or tools to answer "
    "the question, be clear about this - it is better that you say this than get the wrong answer."
)


def prepare() -> Path:
    """Download AA-Omniscience data and convert to Gym JSONL format."""
    from datasets import load_dataset

    print("Downloading AA-Omniscience-Public from HuggingFace...")
    ds = load_dataset("ArtificialAnalysis/AA-Omniscience-Public", split="train")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in ds:
        domain = entry["domain"]
        topic = entry["topic"]
        question = entry["question"]

        row = {
            "id": entry["question_id"],
            "domain": domain,
            "topic": topic,
            "question": question,
            "expected_answer": entry["answer"],
            "responses_create_params": {
                "input": [
                    {"role": "system", "content": SYSTEM_PROMPT.format(domain=domain, topic=topic)},
                    {"role": "user", "content": question},
                ]
            },
        }
        rows.append(json.dumps(row, ensure_ascii=False) + "\n")

    with open(OUTPUT_FPATH, "w", encoding="utf-8") as f:
        f.writelines(rows)

    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
