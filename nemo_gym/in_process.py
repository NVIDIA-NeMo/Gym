# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a Gym benchmark from Python: plan it in one call, then run single tasks against running servers.

```python
planned = plan_benchmark("ifeval")                        # resolve, prepare, plan and collate
runner = TaskRunner(planned.rows, head_server=BaseServerConfig(host="127.0.0.1", port=11000))
result = await runner.run_task(planned.plan.tasks[0].task_id.task_id)
result.rewards                                            # one reward per repeat
```

`plan_benchmark` builds the plan `gym eval plan --benchmark NAME` writes and collates the benchmark's
rows. It reads no command-line arguments and leaves Gym's process-wide configuration untouched,
so a long-lived process can call it. The servers come from `gym env start --benchmark NAME`; each server
runs in its own process and environment either way.
"""

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from nemo_gym.config_types import BaseServerConfig, ConfigError
from nemo_gym.eval_plan import EvalPlan
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.rollout_collection import RolloutCollectionHelper
from nemo_gym.task_selection import row_task_id, select_tasks


@dataclass(frozen=True)
class PlannedBenchmark:
    """A prepared benchmark: its plan (task ids and input digests) and the collated rows to run."""

    plan: EvalPlan
    rows: list[dict[str, Any]]

    def task_ids(self) -> list[str]:
        return [task.task_id.task_id for task in self.plan.tasks]


@dataclass(frozen=True)
class TaskResult:
    """One task's rollouts: a reward per repeat and Gym's raw `/run` results, in the same order."""

    task_id: str
    rewards: list[float]
    results: list[dict[str, Any]]


def plan_benchmark(
    benchmark: str,
    *,
    task_ids: Optional[Sequence[str]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    work_dir: Optional[Path] = None,
) -> PlannedBenchmark:
    """Resolve the benchmark `NAME[/FLAVOR]`, prepare its data, build its plan and collate its rows.

    `overrides` are config keys merged on top, as `++key=value` would be on the command line.
    `work_dir` receives the collated files; a temporary directory is used when it is None.
    """
    from nemo_gym.cli.eval import PrepareBenchmarkConfig, plan_for_config
    from nemo_gym.cli.main import _asset_config_path

    global_config_dict = _benchmark_config(_asset_config_path("benchmark", benchmark), overrides)
    plan = plan_for_config(global_config_dict, PrepareBenchmarkConfig.model_validate(global_config_dict), task_ids)
    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="gym-collate-") as directory:
            rows = _collated_rows(global_config_dict, Path(directory))
    else:
        rows = _collated_rows(global_config_dict, work_dir)
    if task_ids is not None:
        rows = [row for row in rows if row_task_id(row) in set(task_ids)]
    return PlannedBenchmark(plan=plan, rows=rows)


def _benchmark_config(config_path: str, overrides: Optional[Mapping[str, Any]]) -> DictConfig:
    """The resolved config for one benchmark, built without reading sys.argv or caching it process-wide."""
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"config_paths": [config_path]},
        dict(overrides or {}),
    )
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(initial_global_config_dict=initial, skip_load_from_cli=True)
    )


def _collated_rows(global_config_dict: DictConfig, output_dir: Path) -> list[dict[str, Any]]:
    from nemo_gym.train_data_utils import TrainDataProcessor

    TrainDataProcessor().run(
        OmegaConf.merge(global_config_dict, {"mode": "train_preparation", "output_dirpath": str(output_dir)})
    )
    collated = output_dir / "benchmark.jsonl"
    if not collated.exists():
        raise ConfigError(f"Collation did not produce {collated}")
    return [json.loads(line) for line in collated.read_text().splitlines() if line.strip()]


class TaskRunner:
    """Runs single tasks of collated rows against Gym servers that are already running.

    The rows are checked and indexed by task id once. Each `run_task` call then selects one task's row,
    repeats it, sends it through the agent's `/run` and returns the rewards. Concurrent calls in one
    event loop are supported.
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]], *, head_server: BaseServerConfig):
        self._helper = RolloutCollectionHelper()
        self._head_server = head_server
        self._row_of_task = _first_row_of_each_task(rows)

    def task_ids(self) -> list[str]:
        return list(self._row_of_task)

    async def run_task(self, task_id: str, *, num_repeats: int = 1, rollout_id: Optional[str] = None) -> TaskResult:
        """Run `num_repeats` rollouts of one task.

        `rollout_id` names the rollouts for Gym's rollout correlation (e.g. verification events); repeats
        get `<rollout_id>-r<n>`.
        """
        if task_id not in self._row_of_task:
            raise ConfigError(f"No input rows for task ids {[task_id]}")
        rows = self._helper.preprocess_examples(
            [self._row_of_task[task_id]], task_ids=[task_id], num_repeats=num_repeats
        )
        if rollout_id is not None:
            rows = [_with_rollout_id(row, rollout_id, index, num_repeats) for index, row in enumerate(rows)]
        completed = [
            await rollout for rollout in self._helper.run_examples(rows, head_server_config=self._head_server)
        ]
        results = [result for _, result in completed]
        return TaskResult(task_id=task_id, rewards=[result["reward"] for result in results], results=results)


def _first_row_of_each_task(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    ids = list(dict.fromkeys(row_task_id(row) for row in rows))
    return {row_task_id(rows[index]): rows[index] for index in select_tasks(rows, ids)}


def _with_rollout_id(row: dict[str, Any], rollout_id: str, index: int, num_repeats: int) -> dict[str, Any]:
    return {**row, "_ng_rollout_id": rollout_id if num_repeats == 1 else f"{rollout_id}-r{index}"}
