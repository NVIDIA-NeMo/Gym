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
"""Download the SecQue benchmark (nogabenyoash/SecQue), filter to questions that mention known
companies, and convert to the finance agent test format. Run from the finance_agent/ directory:
    python scripts/prepare_secque_questions.py
Output is written to data/secque_questions.jsonl.
"""

import json

from convert_questions import convert_entry
from datasets import load_dataset


# these are the companies that appear in the SecQue dataset
COMPANY_SUBSTRINGS = [
    "nvidia",
    "apple",
    "iphone",
    "tesla",
    "jpmorgan",
    "jpmc",
    "exxon",
    "johnson",
    "procter",
    "p&g",
    "coca",
    "goldman",
    "ibm",
    "amex",
    "express",
    "discover",
    "dfs",
    "general mills",
    "nextera",
    "att",
    "at&t",
    "frontier",
    "pfizer",
    "honeywell",
    "nike",
    "bank of",
    "fortinet",
    "rapid7",
    "halliburton",
    "cms",
    "smucker",
    "liveon",
    "chevron",
    "nee",
    "consumers",
    "simon",
    "vornado",
]


# check if the question mentions any of the companies in the SecQue dataset
def has_company_mention(question: str) -> bool:
    q_lower = question.lower()
    return any(s in q_lower for s in COMPANY_SUBSTRINGS)


def format_secque(data: dict) -> dict:
    return {
        "question": data["Question"],
        "expected_answer": data["ground_truth_answer"],
    }


ds = load_dataset("nogabenyoash/SecQue", split="train")

records = [format_secque(row) for row in ds]
before_count = len(records)
records = [r for r in records if has_company_mention(r["question"])]
print(f"Filtered {before_count} -> {len(records)} questions (kept questions mentioning known companies)")

converted = [convert_entry(r) for r in records]
with open("data/secque_questions.jsonl", "w") as f:
    for entry in converted:
        f.write(json.dumps(entry) + "\n")
print(f"Saved {len(converted)} questions to data/secque_questions.jsonl")
