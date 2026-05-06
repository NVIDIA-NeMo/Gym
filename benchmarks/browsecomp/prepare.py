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
"""Prepare Browsecomp benchmark data.

Downloads Browsecomp problems from OpenAI and converts them to the Gym benchmark JSONL format.
"""

import base64
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "browsecomp_benchmark.jsonl"


SYSTEM_PROMPT = (
    "You are a General Agent. Today's date: {date}. "
    "Your mission is to leverage a diverse set of tools to help the user conduct "
    "an in-depth investigation of their question, continuously reflect, and "
    "ultimately deliver a precise answer.\n\n"
    "Throughout the investigation, strictly observe the following principles:\n"
    "1. Whenever you encounter uncertain information, proactively invoke search "
    "tools to verify it.\n"
    "2. You can only invoke one tool in each round.\n"
    "3. Prioritize high-credibility sources (authoritative websites, academic "
    "databases, professional media) and maintain a critical stance toward "
    "low-credibility ones. Cite the source of any information you use with "
    "a format [^index^].\n"
    "4. You should not respond to the user with a counter-question, but instead "
    "do your best to provide an accurate answer.\n"
    "5. When providing the final answer, begin by explaining the reasoning "
    "process. Avoid presenting only the final answer, as this makes it "
    "difficult to understand."
)

QUERY_SUFFIX = (
    "\n\nYour response should be in the following format:\n"
    "Explanation: {your explanation for your final answer}\n"
    "Exact Answer: {your succinct, final answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)

TOOLS = [
    {
        "type": "function",
        "name": "search",
        "description": (
            "Web Search API, works like Google Search. "
            "All queries will be searched in parallel. "
            "If you want to search with multiple keywords, "
            "put them in a single query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": ("Search queries. All queries are executed in parallel."),
                }
            },
            "required": ["queries"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "browse",
        "description": (
            "Visit specific webpage(s) and return their full text content. "
            "Use this to read the complete content of web pages found "
            "during search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "URL(s) of webpage(s) to visit.",
                },
                "goal": {
                    "type": "string",
                    "description": ("What specific information you are looking for."),
                },
            },
            "required": ["urls"],
        },
        "strict": False,
    },
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

    date_str = datetime.now().strftime("%Y-%m-%d")
    base_system = SYSTEM_PROMPT.format(date=date_str)
    messages = [
        {"role": "system", "content": base_system},
        {"role": "user", "content": problem + QUERY_SUFFIX},
    ]

    return {
        "responses_create_params": {"input": messages, "tools": TOOLS},
        "ground_truth": answer,
        "question": problem,
    }


def prepare() -> Path:
    """Download and prepare AIME 2025 data. Returns the output file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading BrowseComp dataset from {BROWSECOMP_CSV_URL} ...")
    df = pandas.read_csv(BROWSECOMP_CSV_URL)
    assert len(df) == 1266, f"Expected 1266 samples, got {len(df)}"

    count = 0
    with open(OUTPUT_FPATH, "w") as f:
        for _, row in df.iterrows():
            sample = map_browsecomp_sample_to_rl_sample(row.to_dict())
            f.write(json.dumps(sample) + "\n")
            count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
