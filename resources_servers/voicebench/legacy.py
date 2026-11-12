# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay rule benchmarks in the dataset order used by the old VLMEval evaluator."""

import importlib
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from pydantic import JsonValue

from resources_servers.voicebench.scorers.base import ScoringResult
from resources_servers.voicebench.task_data import ScoringType


@dataclass(frozen=True)
class LegacyInput:
    """Saved scoring input and dataset coordinates for one rollout."""

    row: dict[str, object]
    metadata: dict[str, JsonValue]
    answer: str
    task_index: int
    rollout_index: int

    @classmethod
    def from_row(cls, row: Mapping[str, object], index: int) -> "LegacyInput | None":
        metadata = row.get("verifier_metadata")
        answer = row.get("scoring_answer", row.get("candidate_answer"))
        if not isinstance(metadata, dict) or not isinstance(answer, str):
            return None
        return cls(
            row=dict(row),
            metadata=cast(dict[str, JsonValue], metadata),
            answer=answer,
            task_index=cast(int, row.get("_ng_task_index", index)),
            rollout_index=cast(int, row.get("_ng_rollout_index", 0)),
        )


def replay_rule_rows(
    tasks: Sequence[Sequence[Mapping[str, object]]], *, scoring_type: ScoringType, seed: int | None
) -> Sequence[Sequence[Mapping[str, object]]]:
    """Recompute legacy rule statistics with one random stream per evaluation repeat.

    Cached sufficient-statistic rows and infrastructure failures have no scoring
    input and are left intact. Full rows are sorted by Gym's source task index;
    request arrival order therefore cannot change the benchmark-level fallback draws.
    """
    if scoring_type in ("open", "reference_qa"):
        return tasks
    original = [row for task in tasks for row in task]
    repeats: dict[int, list[LegacyInput]] = defaultdict(list)
    untouched = []
    for index, row in enumerate(original):
        item = LegacyInput.from_row(row, index)
        if item is None:
            untouched.append(row)
        else:
            repeats[item.rollout_index].append(item)
    if not repeats:
        return tasks
    module = importlib.import_module(f"resources_servers.voicebench.scorers.{scoring_type}")
    replayed = list(untouched)
    for repeat in sorted(repeats):
        inputs = sorted(repeats[repeat], key=lambda item: item.task_index)
        rng = random.Random(seed)
        if scoring_type == "ifeval":
            results: list[ScoringResult] = module.score_dataset(
                [item.metadata | {"response": item.answer} for item in inputs], rng=rng
            )
        else:
            results = [module.score(item.answer, item.metadata, rng=rng) for item in inputs]
        for item, result in zip(inputs, results, strict=True):
            replayed.append(
                item.row
                | {
                    "scoring_metrics": result.metrics,
                    "score_valid": result.valid,
                    "reward": result.reward,
                    "status_safe_reward": result.reward if item.row.get("candidate_valid") and result.valid else 0.0,
                }
            )
    return [replayed]
