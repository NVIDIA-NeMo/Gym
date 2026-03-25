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
"""Prepare AIME 2025 benchmark data.

Downloads AIME 2025 problems from HuggingFace and converts them to the
Gym benchmark JSONL format with `question` and `expected_answer` fields.
"""

import json
from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "aime25_benchmark.jsonl"

# HuggingFace dataset for AIME 2025
HF_REPO_ID = "MathArena/aime_2025"


def prepare() -> Path:
    """Download and prepare AIME 2025 data. Returns the output file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading AIME 2025 data from {HF_REPO_ID}...")
    ds = load_dataset(HF_REPO_ID, split="train")

    count = 0
    with open(OUTPUT_FPATH, "w") as f:
        for row in ds:
            out = {
                "question": row["problem"],
                "expected_answer": str(row["answer"]),
            }
            f.write(json.dumps(out) + "\n")
            count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


import argparse
import base64
import hashlib
import json
import os

import pandas


QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

TOOLS = [
    {
        "type": "function",
        "name": "web_search",
        "description": "Search the web for a query and return up to 10 search results with <link, summary> for each result.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The term to search for"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": False,
    }
]

BROWSECOMP_CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"


def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def map_browsecomp_sample_to_rl_sample(row: dict) -> dict:
    problem = decrypt(row["problem"], row["canary"])
    answer = decrypt(row["answer"], row["canary"])

    messages = [
        {
            "role": "system",
            "content": "Please think step by step and reason about the problem. You are encouraged to use the tools provided to you to solve the problem, to make sure you can get to the right answer. You must only issue one tool call at a time. Once you are done issuing calls, then return your final answer.",
        },
        {"role": "user", "content": QUERY_TEMPLATE.format(Question=problem)},
    ]

    return {
        "responses_create_params": {"input": messages, "tools": TOOLS},
        "ground_truth": answer,
        "question": problem,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download BrowseComp dataset, decrypt, and convert to NeMo Gym format."
    )
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL file path")
    args = parser.parse_args()

    print(f"Downloading BrowseComp dataset from {BROWSECOMP_CSV_URL} ...")
    df = pandas.read_csv(BROWSECOMP_CSV_URL)
    rows = [row.to_dict() for _, row in df.iterrows()]
    print(f"Downloaded {len(rows)} samples")

    print("Decrypting and converting to NeMo Gym format ...")
    rl_samples = [map_browsecomp_sample_to_rl_sample(row) for row in rows]

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        for sample in rl_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved {len(rl_samples)} samples to {args.output_path}")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    prepare()
