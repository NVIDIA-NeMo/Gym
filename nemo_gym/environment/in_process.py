# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a Gym benchmark from Python: plan it in one call, then run single tasks against running servers.

```python
planned = plan_benchmark("ifeval")                        # resolve, prepare, plan and collate
runner = TaskRunner(planned.rows, registry_server=BaseServerConfig(host="127.0.0.1", port=11000))
result = await runner.run_task(planned.plan.tasks[0].task_id.task_id)
result.rewards                                            # one reward per repeat
```

To start the servers from Python too, serve the benchmark and give the yielded registry server to the runner:

```python
overrides = {"policy_base_url": url, "policy_api_key": key, "policy_model_name": model}
with serve_benchmark("ifeval", overrides=overrides) as registry_server:
    runner = TaskRunner(planned.rows, registry_server=registry_server)
    ...
```

`plan_benchmark` builds the plan `gym eval plan --benchmark NAME` writes and collates the benchmark's
rows. It reads no command-line arguments and leaves Gym's process-wide configuration untouched,
so a long-lived process can call it. `serve_benchmark` starts the servers `gym env start --benchmark NAME`
starts, from the same kind of config dict; each server runs in its own process and environment either way.
The registry server is the server that serves the run's config and the other servers' addresses; a `TaskRunner`
talks to it.

This is the API for callers outside Gym. They import the names `nemo_gym.environment` exports (`plan_benchmark`,
`serve_benchmark`, `TaskRunner`, `PlannedBenchmark`, `TaskResult` and `EvalPlan`) and `ConfigError` from
`nemo_gym.config`. Signatures and result fields change only additively, and tests pin them. `TaskResult.results`
holds Gym's raw `/run` responses, whose keys can change: depend on `rewards`, and use `results` for debugging.
"""

import json
import tempfile
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import BaseServerConfig, ConfigError, ignoring_command_line
from nemo_gym.config.global_config import (
    HEAD_SERVER_KEY_NAME,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    scoped_global_config_dict,
)
from nemo_gym.core.episode_types import is_materialized_task_row
from nemo_gym.data.task_selection import row_task_id, select_tasks
from nemo_gym.environment.assets import asset_config_path
from nemo_gym.environment.benchmark_preparation import PrepareBenchmarkConfig, plan_for_config
from nemo_gym.eval.eval_plan import EvalPlan
from nemo_gym.orchestration.server_supervisor import RunHelper, ServerSpinupTimeoutError, ServerStartFailedError
from nemo_gym.rollouts.rollout_collection import RolloutCollectionHelper


@dataclass(frozen=True)
class PlannedBenchmark:
    """A prepared benchmark: its plan (task ids and input digests) and the collated rows to run."""

    plan: EvalPlan
    rows: list[dict[str, Any]]

    def task_ids(self) -> list[str]:
        return [task.task_id.task_id for task in self.plan.tasks]

    def task_digests(self) -> dict[str, str]:
        """Each planned task's id mapped to the digest of its input, which changes when the input changes."""
        return {task.task_id.task_id: task.digest for task in self.plan.tasks}

    @property
    def gym_commit(self) -> Optional[str]:
        """The Gym commit the plan was made with, or None when Gym was not installed from a git checkout."""
        return self.plan.gym.commit

    def to_json(self) -> str:
        """The planned benchmark as JSON text, for handing it to another process; `from_json` reads it back."""
        return json.dumps(
            {"plan": self.plan.model_dump(mode="json", by_alias=True), "rows": self.rows}, allow_nan=False
        )

    @classmethod
    def from_json(cls, text: str) -> "PlannedBenchmark":
        saved = json.loads(text)
        return cls(plan=EvalPlan.model_validate(saved["plan"]), rows=saved["rows"])


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

    `overrides` are config keys merged on top, as a nested dict like the config file's; a dotted key is refused
    (`ConfigError`), because it would be one literal key here but a path on the command line.
    `work_dir` receives the collated files; a temporary directory is used when it is None.

    The caller's command line and active exporters are left alone. The benchmark's prepare script sees
    `overrides` as its Gym configuration, which is this process's configuration only while it runs.
    """
    global_config_dict = _benchmark_config(asset_config_path("benchmark", benchmark), overrides)
    with ignoring_command_line(), scoped_global_config_dict(global_config_dict):
        plan = plan_for_config(global_config_dict, PrepareBenchmarkConfig.model_validate(global_config_dict), task_ids)
        if work_dir is None:
            with tempfile.TemporaryDirectory(prefix="gym-collate-") as directory:
                rows = _collated_rows(global_config_dict, Path(directory))
        else:
            rows = _collated_rows(global_config_dict, work_dir)
    if task_ids is not None:
        rows = [row for row in rows if row_task_id(row) in set(task_ids)]
    return PlannedBenchmark(plan=plan, rows=rows)


@contextmanager
def serve_benchmark(
    benchmark: str,
    *,
    model_type: str = "openai_model",
    overrides: Optional[Mapping[str, Any]] = None,
    registry_port: Optional[int] = None,
) -> Iterator[BaseServerConfig]:
    """Start the benchmark's servers, and stop them on exit. Yields the registry server a `TaskRunner` talks to.

    The benchmark `NAME[/FLAVOR]` and the model type are named as in
    `gym env start --benchmark NAME --model-type TYPE`. `overrides` are config keys merged on top, as in
    `plan_benchmark`, so planning and serving read the same dict: nested dicts, and a dotted key is refused
    (`ConfigError`). The model's
    `policy_base_url`, `policy_api_key` and `policy_model_name` are among them. `registry_port` fixes the registry
    server's port, so a caller that serves from another process can know the address before the servers start;
    Gym chooses the port when it is None.

    Returns once every server is ready; raises `ConfigError` if a server or a model endpoint is not, or if
    `registry_port` is in use. Like `gym env start`, it sets this process's Gym configuration, so serve one
    benchmark per process.
    """
    config_paths = [asset_config_path("benchmark", benchmark), asset_config_path("model-type", model_type)]
    initial = OmegaConf.merge({"config_paths": config_paths}, _checked_overrides(overrides))
    if registry_port is not None:
        if not 0 < registry_port < 65536:
            raise ConfigError(f"registry_port must be a port number from 1 to 65535, got {registry_port!r}")
        initial = OmegaConf.merge(initial, {HEAD_SERVER_KEY_NAME: {"port": registry_port}})
    helper = RunHelper()
    try:
        with ignoring_command_line():
            helper.start(
                GlobalConfigDictParserConfig(
                    initial_global_config_dict=initial, skip_load_from_cli=True, skip_exporters=True
                )
            )
    except (ServerSpinupTimeoutError, ServerStartFailedError) as error:
        raise ConfigError(f"Servers of benchmark {benchmark!r} did not start: {error}") from error
    try:
        registry_server = ServerClient.load_head_server_config()
        if registry_port is not None and registry_server.port != registry_port:
            raise ConfigError(
                f"registry_port {registry_port} is in use: the registry server started on port {registry_server.port}"
            )
        yield registry_server
    finally:
        helper.shutdown()


def _checked_overrides(overrides: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """The overrides as a dict, refusing a dotted key at any depth.

    Merged as a dict, `"judge.model"` would be one key with a dot in its name, while `++judge.model=x` on the
    command line sets `model` inside `judge`. Refusing it keeps the dict and the command line from disagreeing.
    """
    checked = dict(overrides or {})
    _refuse_dotted_keys(checked)
    return checked


def _refuse_dotted_keys(node: Any) -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            if "." in str(key):
                raise ConfigError(f"Override key {key!r} is dotted: nest the keys as dicts instead.")
            _refuse_dotted_keys(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _refuse_dotted_keys(item)


def _benchmark_config(config_path: str, overrides: Optional[Mapping[str, Any]]) -> DictConfig:
    """The resolved config for one benchmark, built without reading sys.argv or caching it process-wide."""
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"config_paths": [config_path]},
        _checked_overrides(overrides),
    )
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(initial_global_config_dict=initial, skip_load_from_cli=True, skip_exporters=True)
    )


def _collated_rows(global_config_dict: DictConfig, output_dir: Path) -> list[dict[str, Any]]:
    from nemo_gym.data.dataset_preparation import TrainDataProcessor

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

    def __init__(self, rows: Sequence[Mapping[str, Any]], *, registry_server: BaseServerConfig):
        self._helper = RolloutCollectionHelper()
        self._registry_server = registry_server
        self._row_of_task = _first_row_of_each_task(rows)
        for task_id, row in self._row_of_task.items():
            if is_materialized_task_row(row):
                raise ConfigError(f"Task {task_id!r} is a materialized task row; TaskRunner runs flat task rows only.")

    def task_ids(self) -> list[str]:
        return list(self._row_of_task)

    async def run_task(self, task_id: str, *, num_repeats: int = 1, rollout_id: Optional[str] = None) -> TaskResult:
        """Run `num_repeats` rollouts of one task.

        `rollout_id` names the rollouts for Gym's rollout correlation (e.g. verification events); repeats
        get `<rollout_id>-r<n>`. It must differ between concurrent calls; a fresh one is made when it is None.
        """
        if task_id not in self._row_of_task:
            raise ConfigError(f"No input rows for task ids {[task_id]}")
        with ignoring_command_line():
            rows = self._helper.preprocess_examples(
                [self._row_of_task[task_id]], task_ids=[task_id], num_repeats=num_repeats
            )
        rollout_id = rollout_id or uuid.uuid4().hex
        rows = [_with_rollout_id(row, rollout_id, index, num_repeats) for index, row in enumerate(rows)]
        completions = self._helper.run_examples(rows, head_server_config=self._registry_server)
        try:
            completed = [await rollout for rollout in completions]
        finally:
            await completions.aclose()
        results = [result for _, result in completed]
        return TaskResult(task_id=task_id, rewards=[result["reward"] for result in results], results=results)


def _first_row_of_each_task(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    ids = list(dict.fromkeys(row_task_id(row) for row in rows))
    return {row_task_id(rows[index]): rows[index] for index in select_tasks(rows, ids)}


def _with_rollout_id(row: dict[str, Any], rollout_id: str, index: int, num_repeats: int) -> dict[str, Any]:
    return {**row, "_ng_rollout_id": rollout_id if num_repeats == 1 else f"{rollout_id}-r{index}"}
