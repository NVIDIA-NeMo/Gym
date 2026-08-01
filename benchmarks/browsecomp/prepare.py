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

TOOLS = [
    {
        "type": "function",
        "name": "search",
        "description": (
            "Web Search API, works like Google Search. "
            "All queries will be searched in parallel. "
            "If you want to search with multiple keywords, "
            "put them in a single query. "
            "Each result's full raw content is saved to "
            "pages/<idx>_search_<slug>_rN.txt under the current workspace; "
            "the tool response returns the per-result title, URL, snippet, "
            "and [Saved to] path. Use bash_command to read the saved files."
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
            "Visit specific webpage(s) and save their full text content "
            "to local files. Use this to fetch the complete content of "
            "web pages found during search. Each page is written to "
            "pages/<idx>_browse_<slug>.txt under the current workspace; "
            "the tool response returns the per-URL title, [Saved to] "
            "path, byte count, and a short preview. Use bash_command "
            "to read the full content."
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
    {
        "type": "function",
        "name": "bash_command",
        "description": (
            "Run a shell command in the current sample's workspace and "
            "return stdout/stderr (truncated to ~12 KB head+tail).\n\n"
            "cwd layout (pinned per sample; do not navigate outside):\n"
            "  pages/<idx:04d>_<kind>_<slug>[_rN].txt    one file per search result or browsed page\n"
            "  manifest.tsv                              page_id<TAB>kind<TAB>source<TAB>title<TAB>bytes\n\n"
            "Useful recipes:\n"
            "  ls pages/ | wc -l                              # how many pages saved so far\n"
            "  ls pages/ | head                                # glance at filenames (slugged from query/URL)\n"
            '  grep -l "<phrase>" pages/*.txt | head           # which files mention X\n'
            '  grep -m 5 -B 2 -A 4 "<phrase>" pages/0042_*     # context around hits in one file\n'
            "  head -c 2000 pages/0042_*                       # peek at start of a file\n"
            "  sed -n '100,200p' pages/0042_*                  # specific line range\n"
            "  cut -f3 manifest.tsv | sort -u                  # all queries / URLs you've tried\n\n"
            "Each call is one-shot (cwd resets between calls). Chain with "
            "`;` or `&&` inside one keystrokes string.\n\n"
            "Read-only allow-list: only inspection commands are permitted "
            "(ls cat grep head tail sed cut sort uniq wc tr nl strings file find "
            "diff echo printf cd), composed with pipes / for-loops / conditionals. "
            "Anything else — destructive, network, interpreter, install, or "
            "redirection-to-file (rm, mv, curl, python, pip, `>`, …) — is blocked "
            "and returns `[blocked: …]`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keystrokes": {
                    "type": "string",
                    "description": "Shell command to run.",
                },
                "duration": {
                    "type": "number",
                    "description": "Max seconds to wait (1-60, default 10).",
                },
            },
            "required": ["keystrokes", "duration"],
        },
        "strict": False,
    },
]

BROWSECOMP_CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"

# By default prepare() writes a reproducible 400-sample subset; set BROWSECOMP_RUN_FULL=1 for the full 1266.
BROWSECOMP_SUBSET_N = 400
BROWSECOMP_SUBSET_SEED = 42


def _select_samples(df, run_full: bool):
    """1266-row df in -> the full df if run_full, else a deterministic 400-row subset (seed 42).
    Uses stdlib random.Random(seed).sample, the same selection the bc_frankie harness's
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


def map_browsecomp_sample_to_rl_sample(row: dict) -> dict:
    problem = decrypt(row["problem"], row["canary"])
    answer = decrypt(row["answer"], row["canary"])

    date_str = datetime.now().strftime("%Y-%m-%d")
    base_system = SYSTEM_PROMPT.format(date=date_str) + WORKSPACE_SYSTEM_ADDENDUM
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

    run_full = os.environ.get("BROWSECOMP_RUN_FULL", "").lower() in ("1", "true", "yes")
    df = _select_samples(df, run_full)
    mode = "FULL 1266" if run_full else f"{BROWSECOMP_SUBSET_N}-sample subset (seed {BROWSECOMP_SUBSET_SEED})"
    print(f"BrowseComp: writing {mode} ({len(df)} rows)")

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
