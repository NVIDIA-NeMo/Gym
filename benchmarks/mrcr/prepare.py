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
"""Prepare the MRCR benchmark data (mirror of NeMo Skills prepare.py).

Source: https://huggingface.co/datasets/openai/mrcr
Reference:
    https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/mrcr/prepare.py

Each row in the upstream dataset has a `prompt` field that is a JSON-stringified
list of OpenAI chat messages. We parse it into `responses_create_params.input`,
count tokens with tiktoken `o200k_base` (same as Skills / the official MRCR
grading setup), and filter to samples that fit in the model context.

The 262144-token cap matches the Nemotron-3-Super-120B context window. This
is applied symmetrically with the Skills prepare script so both pipelines
evaluate the same subset — a parity requirement.
"""

import json
from pathlib import Path

import tiktoken
from datasets import load_dataset
from tqdm import tqdm


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "mrcr_benchmark.jsonl"

# Must match run_mrcr_ns_baseline.py / prepare_mrcr_ns.py.
MAX_CONTEXT_TOKENS = 262144


def _count_tokens(messages: list[dict]) -> int:
    """Token count using the o200k_base tokenizer — same as Skills prepare."""
    enc = tiktoken.get_encoding("o200k_base")
    return sum(len(enc.encode(m["content"])) for m in messages)


def prepare() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("openai/mrcr", split="train")

    kept = 0
    skipped_tokens = 0
    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for idx, entry in tqdm(enumerate(dataset), desc="Preparing MRCR"):
            messages = json.loads(entry["prompt"])

            n_tokens = _count_tokens(messages)
            if n_tokens > MAX_CONTEXT_TOKENS:
                skipped_tokens += 1
                continue

            sample = {
                "responses_create_params": {"input": messages},
                "expected_answer": entry["answer"],
                "random_string_to_prepend": entry["random_string_to_prepend"],
                "n_needles": entry["n_needles"],
                "n_tokens": n_tokens,
            }
            fout.write(json.dumps(sample) + "\n")
            kept += 1

    print(f"Wrote {kept} samples to {OUTPUT_FPATH} (skipped {skipped_tokens} with >{MAX_CONTEXT_TOKENS} tokens)")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
