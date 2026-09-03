#!/usr/bin/env python3
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
"""Compute rollout scores by custom taxonomy labels supplied as JSONL."""

import argparse
from collections import defaultdict
from pathlib import Path

from resources_servers.arena.scripts.compute_rollout_scores import (
    BENCHMARK_CONFIGS,
    PROMPT_PATHS,
    load_metrics,
    load_prompts_by_question_id,
    load_rollout_tasks,
    print_table,
    read_jsonl,
    score_row,
)


def load_labels(path: Path) -> dict[str, list[str]]:
    """Load JSONL rows containing a question ID and a list of labels."""
    labels_by_question_id = {}
    for row in read_jsonl(path):
        question_id = row.get("question_id") or row.get("qid")
        labels = row.get("labels")
        valid = isinstance(question_id, str) and isinstance(labels, list)
        valid = valid and all(isinstance(label, str) for label in labels)
        if not valid:
            raise ValueError("Each row must contain a question ID and a list of string 'labels'")
        if question_id in labels_by_question_id:
            raise ValueError(f"Duplicate question ID: {question_id}")
        labels_by_question_id[question_id] = labels
    return labels_by_question_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("labels", type=Path, help='JSONL rows: {"question_id": "...", "labels": ["..."]}')
    parser.add_argument("--version", choices=BENCHMARK_CONFIGS, required=True)
    parser.add_argument("--prompts", type=Path, help="Validation JSONL used by these rollouts")
    parser.add_argument("--min-prompts", type=int, default=50)
    args = parser.parse_args()

    prompt_path = args.prompts or PROMPT_PATHS.get(args.version)
    prompts = load_prompts_by_question_id(prompt_path) if prompt_path else {}
    tasks = load_rollout_tasks(args.rollouts, args.version, prompts)
    labels_by_question_id = load_labels(args.labels)

    tasks_by_label = defaultdict(list)
    for task in tasks:
        for label in set(labels_by_question_id.get(task[0]["question_id"], [])):
            tasks_by_label[label].append(task)

    metrics = load_metrics(args.version, 1.0)
    rows = []
    for label, label_tasks in sorted(tasks_by_label.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(label_tasks) >= args.min_prompts:
            rows.append(score_row(label, len(label_tasks), metrics.compute(label_tasks)))

    print(f"benchmark: {args.version}")
    print(f"tasks: {len(tasks)}")
    print(f"mapped_tasks: {sum(task[0]['question_id'] in labels_by_question_id for task in tasks)}")
    print(f"Labels with fewer than {args.min_prompts} prompts are omitted.\n")
    if rows:
        print_table(("Label", "Prompts", "Win rate", "Win rate no SC"), rows)
    else:
        print("No labels meet the minimum prompt count.")


if __name__ == "__main__":
    main()
