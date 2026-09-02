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
"""Compute arena metrics from saved rollout JSONL."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

from resources_servers.arena.app import ArenaResourcesServerConfig
from resources_servers.arena.metrics import ArenaMetrics
from resources_servers.arena.taxonomy import (
    MIN_SLICE_PROMPTS,
    PROMPT_CATEGORY_ORDER,
    get_prompt_slices,
)


BENCHMARK_CONFIGS = {
    "lmarena_v2": Path("resources_servers/arena/configs/lmarena_v2.yaml"),
    "lmarena_v3": Path("resources_servers/arena/configs/lmarena_v3.yaml"),
}
# LMArena prompts provide taxonomy metadata used for slice reporting.
PROMPT_PATHS = {
    "lmarena_v2": Path("benchmarks/lmarena_v2/data/lmarena_v2_validation.jsonl"),
    "lmarena_v3": Path("benchmarks/lmarena_v3/data/lmarena_v3_validation.jsonl"),
}
SCORE_SLICE_ORDER = (*PROMPT_CATEGORY_ORDER, "exclude-ties")


def load_metrics(benchmark: str, max_failure_rate: float) -> ArenaMetrics:
    """Load the benchmark's production scoring settings."""
    yaml = OmegaConf.load(BENCHMARK_CONFIGS[benchmark])
    config = OmegaConf.to_container(yaml[benchmark]["resources_servers"]["arena"], resolve=True)
    config.update(
        host="0.0.0.0",
        port=8080,
        name=benchmark,
        max_rollout_failure_rate=max_failure_rate,
    )
    return ArenaMetrics(ArenaResourcesServerConfig.model_validate(config))


def read_jsonl(path: Path) -> list[dict]:
    with path.expanduser().open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_rollout_tasks(
    path: Path,
    benchmark: str,
    prompts: dict[str, dict],
    allow_unmatched_prompts: bool = False,
) -> list[list[dict]]:
    """Load rollouts, attach current prompt metadata, and group repeated samples."""
    tasks = defaultdict(list)
    unmatched_question_ids = set()
    for rollout in read_jsonl(path):
        if "_ng_task_index" not in rollout:
            raise ValueError("Rollout is missing '_ng_task_index'")
        category = rollout.get("category")
        if category != benchmark:
            raise ValueError(
                f"Rollout {rollout.get('question_id')!r} has category {category!r}, but --version is {benchmark!r}"
            )
        if prompts:
            prompt = prompts.get(rollout["question_id"])
            if prompt is None:
                unmatched_question_ids.add(rollout["question_id"])
                continue
            if prompt.get("category") != category:
                raise ValueError(
                    f"Prompt {rollout['question_id']!r} has category {prompt.get('category')!r}, "
                    f"but its rollout has category {category!r}"
                )
            rollout["prompt_slices"] = {
                namespace: sorted(labels) for namespace, labels in get_prompt_slices(prompt).items()
            }
            if benchmark == "lmarena_v3":
                rollout["style_reference_token_count"] = prompt["style_reference_token_count"]
                rollout["is_lmarena_v2_prompt"] = prompt["is_lmarena_v2_prompt"]
        tasks[rollout["_ng_task_index"]].append(rollout)
    if unmatched_question_ids and not allow_unmatched_prompts:
        examples = ", ".join(repr(question_id) for question_id in sorted(unmatched_question_ids)[:3])
        raise ValueError(
            f"{len(unmatched_question_ids)} rollout question IDs are absent from the selected prompt file "
            f"(examples: {examples}). Pass --allow-unmatched-prompts to skip them explicitly."
        )
    if not tasks:
        raise ValueError(f"No rollouts found in {path}")
    return [tasks[index] for index in sorted(tasks)]


def load_prompts_by_question_id(path: Path) -> dict[str, dict]:
    return {prompt["question_id"]: prompt for prompt in read_jsonl(path)}


def print_win_rate(metrics: dict, key: str) -> None:
    print(f"\n{key}:")
    print(f"  estimate: {metrics[key]:.1%}")
    print(f"  95% CI: [{metrics[f'{key}_ci95_lower']:.1%}, {metrics[f'{key}_ci95_upper']:.1%}]")


def score_row(name: str, prompts: int, metrics: dict) -> tuple[str, ...]:
    return name, str(prompts), f"{metrics['win_rate']:.1%}", f"{metrics['win_rate_no_SC']:.1%}"


def print_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    widths = [max([len(header), *(len(row[index]) for row in rows)]) for index, header in enumerate(headers)]

    def format_row(row: tuple[str, ...]) -> str:
        cells = [row[0].ljust(widths[0])]
        cells += [value.rjust(width) for value, width in zip(row[1:], widths[1:])]
        return "| " + " | ".join(cells) + " |"

    print(format_row(headers))
    separators = ["-" * widths[0], *("-" * (width - 1) + ":" for width in widths[1:])]
    print(format_row(tuple(separators)))
    for row in rows:
        print(format_row(row))


def print_slice_table(title: str, metrics: dict, namespace: str, order: tuple[str, ...] = ()) -> None:
    prefixes = {
        key.removeprefix(f"{namespace}/").removesuffix("/prompts")
        for key in metrics
        if key.startswith(f"{namespace}/") and key.endswith("/prompts")
    }
    if namespace == "arena" and "total_prompts" in metrics:
        prefixes.add("overall")

    def prompt_count(name: str) -> int:
        return metrics["total_prompts"] if name == "overall" else metrics[f"{namespace}/{name}/prompts"]

    names = [name for name in order if name in prefixes]
    names += sorted(prefixes - set(names), key=lambda name: (-prompt_count(name), name))
    rows = [
        score_row(
            name,
            prompt_count(name),
            {
                "win_rate": metrics[f"{namespace}/{name}/win_rate"],
                "win_rate_no_SC": metrics[f"{namespace}/{name}/win_rate_no_SC"],
            },
        )
        for name in names
    ]
    print(f"\n## {title}")
    print_table(("Category", "Prompts", "Win rate", "Win rate no SC"), rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument(
        "--version",
        dest="benchmark",
        choices=BENCHMARK_CONFIGS,
        required=True,
    )
    parser.add_argument("--prompts", type=Path, help="Validation JSONL used by these rollouts")
    parser.add_argument(
        "--allow-unmatched-prompts",
        action="store_true",
        help="Skip rollout question IDs absent from the selected prompt file",
    )
    parser.add_argument("--max-failure-rate", type=float, default=0.01)
    args = parser.parse_args()

    print(f"benchmark: {args.benchmark}")
    # Custom evaluations may use prompt IDs absent from the canonical validation set.
    prompt_path = args.prompts or PROMPT_PATHS.get(args.benchmark)
    prompts = load_prompts_by_question_id(prompt_path) if prompt_path else {}
    tasks = load_rollout_tasks(args.rollouts, args.benchmark, prompts, args.allow_unmatched_prompts)

    rollouts = [rollout for task in tasks for rollout in task]
    # Use the exact same implementation and configuration as online evaluation.
    scorer = load_metrics(args.benchmark, args.max_failure_rate)
    metrics = scorer.compute(tasks)

    print(f"n_tasks: {len(tasks)}")
    print(f"n_rollouts: {len(rollouts)}")
    diagnostic_names = [
        "rollout_failure_rate",
        "reasoning_only_response_rate",
        "missing_judgment_rate",
        "parse_failure_rate",
        "any_both_bad_rate",
        "any_tie_rate",
    ]
    if args.benchmark == "lmarena_v3":
        diagnostic_names.insert(0, "context_window_exceeded_rate")
        diagnostic_names.insert(0, "max_token_reached_rate")
    for name in diagnostic_names:
        print(f"{name}: {metrics[name]:.2%}")
    if args.benchmark == "lmarena_v3":
        print(f"verbosity_acceptance_rate: {metrics['verbosity_acceptance_rate']:.2%}")
    print(f"mean/reward: {sum(rollout['reward'] for rollout in rollouts) / len(rollouts):.1%}")

    print_win_rate(metrics, "win_rate_no_SC")
    print_win_rate(metrics, "win_rate")

    if not prompts:
        return

    # Slice metrics were computed in the same pass as the overall score.
    print(f"\nSlices with fewer than {MIN_SLICE_PROMPTS} prompts are omitted.")
    print_slice_table("Arena", metrics, "arena", ("overall", *SCORE_SLICE_ORDER))
    print_slice_table("Taxonomy language", metrics, "taxonomy-language")
    print_slice_table("Task type", metrics, "taxonomy-task-type")


if __name__ == "__main__":
    main()
