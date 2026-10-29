# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PostTrainBench on the Harbor environment server, with the benchmark's weighted score.

Episodes run exactly as on the ``harbor`` server. This server only reads the taskset that
``benchmarks/posttrainbench/prepare.py`` wrote and adds PostTrainBench's run score to the metrics: per
benchmark, the mean task reward over the base models, weighted with upstream's factors and summed
(``scripts/aggregate.py``). An episode whose judging did not finish is masked as inconclusive, and the score
is withheld while any task has no scored rollout: ``scripts/collect.py`` refuses to aggregate unjudged runs.
A verifier that stopped before writing its reward is scored from the logs it left, by the same rules.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from fastapi import Body

from environment_servers.harbor.app import (
    HarborEnvironmentServer,
    HarborEnvironmentServerConfig,
    HarborTasksetConfig,
)
from nemo_gym import PARENT_DIR
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME
from nemo_gym.rollout_collection import NG_TASK_ID_KEY
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint


SCORE_KEY = "ptb/weighted_avg"
MISSING_KEY = "ptb/tasks_without_score"
# Per-task reward fields gym_score.py writes, counted over the scored rollouts.
COUNTED_FIELDS = ("eval_failed", "no_model", "judge_flagged", "ptb_lookup_flagged")
JUDGING_INCONCLUSIVE_KIND = "posttrainbench:judging_inconclusive"


class PostTrainBenchEnvironmentServerConfig(HarborEnvironmentServerConfig):
    # Written by the benchmark's prepare; relative paths resolve against the Gym checkout.
    taskset_file: Path


class PostTrainBenchEnvironmentServer(HarborEnvironmentServer):
    config: PostTrainBenchEnvironmentServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        path = self.config.taskset_file
        path = path if path.is_absolute() else PARENT_DIR / path
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; run `gym eval prepare --benchmark posttrainbench` first")
        taskset = json.loads(path.read_text())
        self.config.tasksets = {
            taskset["taskset"]: HarborTasksetConfig(folder=taskset["folder"], tasks=taskset["tasks"])
        }
        self._baselines: dict[str, float] = taskset["baselines"]
        self._factors: dict[str, float] = taskset["factors"]

    def _score(self, task: Any, settings: Any, result: Any, logs_dir: Path, started: float, **kwargs: Any) -> dict:
        """Harbor's reward fields; an episode whose judges did not all report is inconclusive, not a score."""
        scored = super()._score(task, settings, result, logs_dir, started, **kwargs)
        fields = _rewards(scored)
        if "baseline_accuracy" not in fields:
            # test.sh stopped before gym_score.py wrote reward.json (a timeout, a crash): Harbor's measured 0 is not
            # PostTrainBench's score. Score the verifier logs it left with the same rules, on the host.
            fields = _gym_score(task.path).score(str(logs_dir), self._baselines[task.task_id])
            scored = scored | {"reward": fields["reward"], "verifier_rewards": fields}
        if fields.get("judgement_missing") or fields.get("judgement_unreadable"):
            reason = f"PostTrainBench judging did not finish for {task.task_id} (verifier logs in {logs_dir})"
            return scored | self._masked(JUDGING_INCONCLUSIVE_KIND, reason)
        return scored

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        # Masked rows never reach compute_metrics, so a repeat whose every task was masked would vanish from it.
        self._repeats = {response.get(ROLLOUT_INDEX_KEY_NAME, 0) for response in body.verify_responses}
        return await super().aggregate_metrics(body)

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        rollouts = [rollout for task in tasks for rollout in task]
        seen = {rollout.get(ROLLOUT_INDEX_KEY_NAME, 0) for rollout in rollouts}
        repeats = sorted(seen | getattr(self, "_repeats", set())) or [0]
        scores = [weighted_score(rollouts, repeat, self._baselines, self._factors) for repeat in repeats]
        metrics: dict[str, Any] = {MISSING_KEY: sum(score[MISSING_KEY] for score in scores)}
        if all(SCORE_KEY in score for score in scores):
            metrics[SCORE_KEY] = sum(score[SCORE_KEY] for score in scores) / len(scores)
        for name in sorted({key for score in scores for key in score} - {SCORE_KEY, MISSING_KEY}):
            values = [score[name] for score in scores if name in score]
            metrics[name] = sum(values) / len(values)
        for field in COUNTED_FIELDS:
            metrics[f"ptb/{field}"] = sum(int(_rewards(rollout).get(field, 0)) for rollout in rollouts)
        return metrics

    def compute_repeat_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        rollouts = [rollout for task in tasks for rollout in task]
        repeat = rollouts[0].get(ROLLOUT_INDEX_KEY_NAME, 0) if rollouts else 0
        score = weighted_score(rollouts, repeat, self._baselines, self._factors)
        return {SCORE_KEY: score[SCORE_KEY]} if SCORE_KEY in score else {}

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        keys = {SCORE_KEY, MISSING_KEY}
        return {key: value for key, value in agent_metrics.items() if key in keys or key.startswith("mean/")}


def _rewards(rollout: dict[str, Any]) -> dict[str, Any]:
    return rollout.get("verifier_rewards") or {}


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gym_score(task_path: Path) -> ModuleType:
    """The task's tests/gym_score.py, bound to the task's copy of upstream's scripts/utils.py."""
    tests = task_path / "tests"
    previous = sys.modules.get("utils")
    sys.modules["utils"] = _load_module("posttrainbench_upstream_utils", tests / "ptb" / "scripts" / "utils.py")
    try:
        return _load_module("posttrainbench_gym_score", tests / "gym_score.py")
    finally:
        if previous is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous


def _task_name(rollout: dict[str, Any]) -> str | None:
    # The collector stores the episode's TaskId as _ng_task_id; the rollout row has none of its own.
    task_id = rollout.get(NG_TASK_ID_KEY, rollout.get("task_id"))
    return task_id.get("task_id") if isinstance(task_id, dict) else task_id


def weighted_score(
    rollouts: list[dict[str, Any]], repeat: int, baselines: dict[str, float], factors: dict[str, float]
) -> dict[str, float]:
    """PostTrainBench's run score for one repeat, with the per-benchmark means it is summed from.

    Task names are ``posttrainbench-<benchmark>-<model>``; ``baselines`` names every expected task. The score
    is left out while a task has no scored rollout (masked, or not run yet); the means cover the scored tasks.
    """
    rewards = {
        _task_name(rollout): float(rollout["reward"])
        for rollout in rollouts
        if rollout.get(ROLLOUT_INDEX_KEY_NAME, 0) == repeat
    }
    unknown = sorted({task.split("-")[1] for task in baselines} - set(factors))
    if unknown:
        raise ValueError(f"No PostTrainBench weight for benchmark(s) {unknown}")
    by_benchmark: dict[str, list[float]] = {}
    for task in baselines:
        if task in rewards:
            by_benchmark.setdefault(task.split("-")[1], []).append(rewards[task])
    means = {benchmark: sum(values) / len(values) for benchmark, values in sorted(by_benchmark.items())}
    missing = sum(task not in rewards for task in baselines)
    score: dict[str, float] = {f"ptb/{benchmark}": mean for benchmark, mean in means.items()}
    score[MISSING_KEY] = float(missing)
    if not missing:
        score[SCORE_KEY] = sum(factors[benchmark] * mean for benchmark, mean in means.items())
    return score


if __name__ == "__main__":
    PostTrainBenchEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = PostTrainBenchEnvironmentServer.run_webserver()  # noqa: F401
