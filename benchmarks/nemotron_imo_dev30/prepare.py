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
"""Prepare the 30-problem development set of "An Open Recipe for IMO Gold" (arXiv 2609.10712).

20 problems come from Nemotron-IMO-Bench and 10 from recent competitions (IMO-ProofBench, EGMO via
MathNet, IMO 2025 via MathArena). Ids, tiers and order follow NeMo-Skills
``recipes/nemotron-imo-tts/scripts/build_devset30.py``.
"""

import json
from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "nemotron_imo_dev30_benchmark.jsonl"

# (tier, problem id) in the paper's order. Ids without the proofbench133_ prefix are Nemotron-IMO-Bench ids.
DEVSET30 = [
    ("EASY", "G50"),
    ("EASY", "A22"),
    ("EASY", "proofbench133_101"),
    ("MEDIUM", "A48"),
    ("MEDIUM", "G17"),
    ("MEDIUM", "A3"),
    ("MEDIUM", "C5"),
    ("MEDIUM", "N20"),
    ("MEDIUM", "proofbench133_093"),
    ("MEDIUM", "proofbench133_008"),
    ("HARD", "A32"),
    ("HARD", "C32"),
    ("HARD", "proofbench133_130"),
    ("HARD", "N30"),
    ("HARD", "N39"),
    ("HARD", "proofbench133_121"),
    ("HARD", "G43"),
    ("HARD", "N14"),
    ("HARD", "proofbench133_112"),
    ("HARD", "proofbench133_103"),
    ("UNSOLVED", "C19"),
    ("UNSOLVED", "N40"),
    ("UNSOLVED", "C13"),
    ("UNSOLVED", "G18"),
    ("UNSOLVED", "N42"),
    ("UNSOLVED", "G6"),
    ("UNSOLVED", "N18"),
    ("UNSOLVED", "proofbench133_118"),
    ("UNSOLVED", "proofbench133_120"),
    ("UNSOLVED", "proofbench133_133"),
]

SOURCES = [
    {
        "repo_id": "nvidia/Nemotron-IMO-Bench",
        "revision": "223023e8406f7117cbb7e43f9bc2cd3dc21f38c3",  # pragma: allowlist secret
        "config": None,
        "split": "test",
        "id_field": "id",
        "problem_field": "problem",
        "license": "CC-BY-4.0",
        "ids": {problem_id: problem_id for _, problem_id in DEVSET30 if not problem_id.startswith("proofbench133_")},
    },
    {
        "repo_id": "Hwilner/imo-proofbench",
        "revision": "6dc351760685a4ceedb574f19da8795a96098d00",  # pragma: allowlist secret
        "config": None,
        "split": "train",
        "id_field": "Problem ID",
        "problem_field": "Problem",
        "license": "CC-BY-4.0",
        "ids": {
            "PB-Advanced-004": "proofbench133_101",
            "PB-Basic-026": "proofbench133_093",
            "PB-Advanced-024": "proofbench133_121",
            "PB-Advanced-015": "proofbench133_112",
            "PB-Advanced-006": "proofbench133_103",
            "PB-Advanced-021": "proofbench133_118",
            "PB-Advanced-023": "proofbench133_120",
        },
    },
    {
        "repo_id": "ShadenA/MathNet",
        "revision": "33e6b3bc254e6f0f0c1479b4252f5e9dd551d56c",  # pragma: allowlist secret
        "config": "European_Girls'_Mathematical_Olympiad_EGMO",
        "split": "train",
        "id_field": "id",
        "problem_field": "problem_markdown",
        "license": "CC-BY-4.0",
        "ids": {"05ed": "proofbench133_008"},
    },
    {
        "repo_id": "MathArena/imo_2025",
        "revision": "ed9cc5dbaa7bf73331b0d2cb54cac2bb4c030a10",  # pragma: allowlist secret
        "config": None,
        "split": "train",
        "id_field": "problem_idx",
        "problem_field": "problem",
        "license": "CC-BY-NC-SA-4.0",
        "ids": {"3": "proofbench133_130", "6": "proofbench133_133"},
    },
]


def collect_problems() -> dict:
    problems = {}
    for source in SOURCES:
        rows = load_dataset(source["repo_id"], source["config"], split=source["split"], revision=source["revision"])
        rows = rows.select_columns([source["id_field"], source["problem_field"]])
        by_id = {str(row[source["id_field"]]): row for row in rows}
        missing = sorted(source["ids"].keys() - by_id.keys())
        if missing:
            raise ValueError(f"{source['repo_id']} is missing ids {missing}")
        for source_id, problem_id in source["ids"].items():
            problems[problem_id] = {
                "problem": by_id[source_id][source["problem_field"]].removeprefix("Problem:\n\n"),
                "source_dataset": f"https://huggingface.co/datasets/{source['repo_id']}",
                "source_id": source_id,
                "dataset_license": source["license"],
            }
    return problems


def prepare() -> Path:
    """Download the development set and write it as benchmark JSONL. Returns the output file path."""
    problems = collect_problems()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FPATH, "w") as f:
        for tier, problem_id in DEVSET30:
            f.write(json.dumps({"problem_idx": problem_id, "tier": tier, **problems[problem_id]}) + "\n")
    print(f"Wrote {len(DEVSET30)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
