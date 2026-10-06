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
"""Prepare TimeWarp benchmark rows.

Downloads the TimeWarp task file from Hugging Face at a pinned revision and writes one row per
(task, UI version): each of the 231 version-independent goals runs in all six UI eras. The
default writes the 103-task test split (618 rows); ``prepare(split="train")`` writes the
128-task train split (768 rows).
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Sequence

from huggingface_hub import hf_hub_download


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATHS = {
    "test": DATA_DIR / "timewarp_benchmark.jsonl",
    "train": DATA_DIR / "timewarp_train.jsonl",
}

HF_REPO_ID = "sparklabutah/timewarp"
# data.json at this revision is identical to src/browsergym/timewarp/data/test.raw.json in
# github.com/sparklabutah/timewarp at commit 4978e69 (the deterministic-verifier release).
HF_REVISION = "246edb1cc9c4746df68172dad661c97164064cec"

UI_VERSIONS = (1, 2, 3, 4, 5, 6)
# The test split is tasks 1-103 and the train split 104-231, as in BrowserGym's timewarp.csv
# metadata and the dataset's test.csv / train.csv.
LAST_TEST_TASK_ID = 103

START_SITES = {"__WIKI__": "wiki", "__NEWS__": "news", "__WEBSHOP__": "webshop"}

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "name": "observe",
        "description": (
            "Show the current page: its URL, title and accessibility snapshot. Long pages are split "
            "into parts; pass a higher part number to read further."
        ),
        "parameters": {
            "type": "object",
            "properties": {"part": {"type": "integer", "minimum": 1, "description": "Part to show; 1 is the top."}},
            "required": [],
            "additionalProperties": False,
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "open_site",
        "description": "Open the home page of one of the task's websites.",
        "parameters": {
            "type": "object",
            "properties": {"site": {"type": "string", "enum": ["wiki", "news", "shop"]}},
            "required": ["site"],
            "additionalProperties": False,
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "goto",
        "description": "Open a URL on one of the task's websites.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
            "additionalProperties": False,
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "click",
        "description": "Click the element with this ref.",
        "parameters": {
            "type": "object",
            "properties": {"ref": {"type": "string", "description": "Element ref from the latest observation."}},
            "required": ["ref"],
            "additionalProperties": False,
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "fill",
        "description": "Replace the text in the input field with this ref.",
        "parameters": {
            "type": "object",
            "properties": {"ref": {"type": "string"}, "text": {"type": "string"}},
            "required": ["ref", "text"],
            "additionalProperties": False,
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "press",
        "description": "Press a key, such as Enter, while the element with this ref has focus.",
        "parameters": {
            "type": "object",
            "properties": {"ref": {"type": "string"}, "key": {"type": "string"}},
            "required": ["ref", "key"],
            "additionalProperties": False,
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "select_option",
        "description": "Choose an option, by its value or label, in the dropdown with this ref.",
        "parameters": {
            "type": "object",
            "properties": {"ref": {"type": "string"}, "option": {"type": "string"}},
            "required": ["ref", "option"],
            "additionalProperties": False,
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "go_back",
        "description": "Go back to the previous page.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": False,
    },
    {
        "type": "function",
        "name": "go_forward",
        "description": "Go forward to the next page.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": False,
    },
]


def in_split(task: Dict[str, Any], split: Literal["test", "train"]) -> bool:
    is_test = task["task_id"] <= LAST_TEST_TASK_ID
    return is_test if split == "test" else not is_test


def build_rows(tasks: Iterable[Dict[str, Any]], ui_versions: Sequence[int] = UI_VERSIONS) -> List[Dict[str, Any]]:
    """One row per (task, UI version), ordered by version and then task id.

    The train split's human-refined plans (``additional_instructions``) are left out: upstream
    appends them to the goal for teacher-trajectory collection, and they spell out the answer.
    """
    rows = []
    for ui_version in ui_versions:
        for task in sorted(tasks, key=lambda task: task["task_id"]):
            start_site = START_SITES.get(task["start_url"])
            if start_site is None:
                raise ValueError(f"task {task['task_id']}: unexpected start_url {task['start_url']!r}")
            spec = task["eval"]
            rows.append(
                {
                    "task_id": task["task_id"],
                    "ui_version": ui_version,
                    "start_site": start_site,
                    "sites": task["sites"],
                    "intent": task["intent"],
                    "verifier_metadata": {
                        "eval_types": spec["eval_types"],
                        "reference_answers": spec["reference_answers"],
                        "revision": spec.get("revision"),
                    },
                    "responses_create_params": {"tools": TOOLS},
                }
            )
    return rows


def prepare(split: Literal["test", "train"] = "test") -> Path:
    """Download the pinned task file and write the split's rows. Returns the output path."""
    source = hf_hub_download(repo_id=HF_REPO_ID, filename="data.json", repo_type="dataset", revision=HF_REVISION)
    with open(source, encoding="utf-8") as f:
        tasks = [task for task in json.load(f) if in_split(task, split)]

    rows = build_rows(tasks)
    output = OUTPUT_FPATHS[split]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rows ({len(tasks)} tasks x {len(UI_VERSIONS)} UI versions) to {output}")
    return output


if __name__ == "__main__":
    prepare()
