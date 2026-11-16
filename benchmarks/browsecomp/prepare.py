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
import os
import random
from datetime import datetime
from pathlib import Path

import pandas

from resources_servers.browsecomp_advanced_harness.tool_definitions import WORKSPACE_TOOLS as TOOLS


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

WORKSPACE_SYSTEM_ADDENDUM = (
    "\n\n## Tool output and the pages/ workspace\n"
    "The `search` and `browse` tools save retrieved content to local files in "
    "pages/ and return only metadata (title, URL, snippet, [Saved to] path) "
    "in the tool response. Use the `bash_command` tool (with grep, head, sed, "
    "etc.) to read or search those files. After a context reset, `ls pages/` "
    "and `cat manifest.tsv` show everything you've already retrieved."
)

BROWSECOMP_CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"

# By default prepare() writes a reproducible 400-sample subset; set BROWSECOMP_RUN_FULL=1 for the full 1266.
BROWSECOMP_SUBSET_N = 400
BROWSECOMP_SUBSET_SEED = 42


def _select_samples(df, run_full: bool):
    """1266-row df in -> the full df if run_full, else a deterministic 400-row subset (seed 42).
    Uses stdlib random.Random(seed).sample, the same selection the reference harness's
    browsecomp_eval.py performs, so both harnesses' seed-42 subset is the same 400."""
    if run_full:
        return df
    idx = random.Random(BROWSECOMP_SUBSET_SEED).sample(range(len(df)), BROWSECOMP_SUBSET_N)
    return df.iloc[idx].reset_index(drop=True)


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


def map_browsecomp_sample_to_rl_sample(row: dict, tools: list[dict] | None = TOOLS) -> dict:
    """One benchmark row. ``tools=None`` leaves the tool list out, for harnesses that take the tools over MCP."""
    problem = decrypt(row["problem"], row["canary"])
    answer = decrypt(row["answer"], row["canary"])

    date_str = datetime.now().strftime("%Y-%m-%d")
    base_system = SYSTEM_PROMPT.format(date=date_str) + WORKSPACE_SYSTEM_ADDENDUM
    messages = [
        {"role": "system", "content": base_system},
        {"role": "user", "content": problem + QUERY_SUFFIX},
    ]

    responses_create_params = {"input": messages}
    if tools is not None:
        responses_create_params["tools"] = tools
    return {
        "responses_create_params": responses_create_params,
        "ground_truth": answer,
        "question": problem,
    }


def write_benchmark(output_fpath: Path, tools: list[dict] | None) -> Path:
    """Download BrowseComp and write its rows to ``output_fpath``. Returns the output file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading BrowseComp dataset from {BROWSECOMP_CSV_URL} ...")
    df = pandas.read_csv(BROWSECOMP_CSV_URL)
    assert len(df) == 1266, f"Expected 1266 samples, got {len(df)}"

    run_full = os.environ.get("BROWSECOMP_RUN_FULL", "").lower() in ("1", "true", "yes")
    df = _select_samples(df, run_full)
    mode = "FULL 1266" if run_full else f"{BROWSECOMP_SUBSET_N}-sample subset (seed {BROWSECOMP_SUBSET_SEED})"
    print(f"BrowseComp: writing {mode} ({len(df)} rows)")

    count = 0
    with open(output_fpath, "w") as f:
        for _, row in df.iterrows():
            sample = map_browsecomp_sample_to_rl_sample(row.to_dict(), tools)
            f.write(json.dumps(sample) + "\n")
            count += 1

    print(f"Wrote {count} problems to {output_fpath}")
    return output_fpath


def prepare() -> Path:
    """Download and prepare the BrowseComp rows for browsecomp_agent. Returns the output file path."""
    return write_benchmark(OUTPUT_FPATH, tools=TOOLS)


if __name__ == "__main__":
    prepare()
