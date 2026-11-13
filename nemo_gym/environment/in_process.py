# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a Gym benchmark from Python: plan it in one call, then run single tasks against running servers.

```python
planned = plan_benchmark("ifeval")                        # resolve, prepare, plan and collate
runner = TaskRunner(planned.rows, registry_server=BaseServerConfig(host="127.0.0.1", port=11000))
outcome = await runner.run_task(planned.plan.tasks[0].task_id.task_id)
outcome.reward                                            # None when the rollout has no reward
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
`serve_benchmark`, `TaskRunner`, `PlannedBenchmark`, `RolloutOutcome`, `TaskFailure` and `EvalPlan`) and
`ConfigError` from `nemo_gym.config`. It is experimental: signatures and result fields may change without a
deprecation period, and tests pin the current ones. `RolloutOutcome.raw` is Gym's raw `/run` response (the failure
record for a failed `/run`), whose keys can change: depend on the other fields, and use `raw` for debugging.
"""

import json
import math
import re
import tempfile
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from omegaconf import DictConfig, OmegaConf

from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import BaseServerConfig, ConfigError, ignoring_command_line
from nemo_gym.config.global_config import (
    HEAD_SERVER_KEY_NAME,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    scoped_global_config_dict,
)
from nemo_gym.config.keys import AGENT_REF_KEY_NAME
from nemo_gym.core.episode_types import is_materialized_task_row
from nemo_gym.core.failure_kinds import (
    FailureStage,
    is_evaluation_side,
    is_registered,
)
from nemo_gym.data.task_selection import row_task_id, select_tasks
from nemo_gym.environment.assets import asset_config_path
from nemo_gym.environment.benchmark_preparation import PrepareBenchmarkConfig, plan_for_config
from nemo_gym.eval.eval_plan import EvalPlan
from nemo_gym.rollouts.rollout_collection import (
    NG_ENVIRONMENT_SERVER_KEY,
    RolloutCollectionHelper,
    _metric_failure_kind,
)
from nemo_gym.rollouts.rollout_outcomes import RolloutFailureRecord
from nemo_gym.server_runtime.supervisor.server_supervisor import (
    RunHelper,
    ServerSpinupTimeoutError,
    ServerStartFailedError,
)
from nemo_gym.telemetry.gym_metrics import record_rollout_completed, record_rollout_reward
from nemo_gym.utils.decorators import experimental


# The bounds `gym eval run` puts on a `/run` call: a connection that was made and dropped is sent twice, and one that
# was refused is resent for a minute (`dispatch_connect_grace_s`). Without them a dead server is retried for ever.
_MAX_RUN_TRIES = 2
_CONNECT_GRACE_S = 60.0


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
class TaskFailure:
    """What Gym observed when a rollout failed. It reports; what to do about it is the caller's decision.

    `kind` is Gym's failure kind (the vocabulary is in `nemo_gym/core/failure_kinds.py`; a `<server>:<kind>` kind is an
    environment's own): the cause when the reporter named one, else `environment_server_failed` (the environment
    reported a failure without a kind) or `agent_request_failed` (the `/run` call failed). `domain` is derived from
    it and `evaluation_side` classifies it.

    `source` says who reported the failure. `delivery` is Gym's evidence about whether the `/run` request reached the
    agent: `not_sent` (every connection attempt was refused, so resending cannot start a second episode),
    `possibly_delivered` (it may have been processed) or `delivered`. It is the evidence to use for a retry decision;
    `terminal` is only the reporter's statement that the episode must not be attempted again, and `False` does not mean
    it is safe to retry. `stage` is where in the episode it happened, when the reporter knows. `reason`, `http_status`
    and `exception_type` are diagnostics.
    """

    kind: str
    source: Literal["environment", "collector"]
    delivery: Literal["not_sent", "possibly_delivered", "delivered"]
    stage: Optional[FailureStage] = None
    terminal: bool = False
    reason: Optional[str] = None
    http_status: Optional[int] = None
    exception_type: Optional[str] = None

    @property
    def domain(self) -> str:
        """The layer that observed the failure: the part of `kind` before its first `_` or `:`.

        Names are `<domain>_<condition>` and an environment's own are `<server>:<kind>`, so `agent_run_error` is in
        `agent`, `judge_failed` in `judge` and `myenv:bad_input` in `myenv`. A name with neither, such as `cancelled`,
        is its own domain. It names a layer, not whose fault the failure was: `agent_timeout` and `agent_run_error` are
        both in `agent`. Use `evaluation_side` for that.
        """
        return re.split(r"[_:]", self.kind, maxsplit=1)[0]

    @property
    def evaluation_side(self) -> Optional[bool]:
        """Whether the kind is an evaluation-side fault (True) or the model's own outcome (False).

        None for a kind that is not in Gym's vocabulary, such as an environment's own `<server>:<kind>`. This is the
        kind's own class; a recorded HTTP status can refine an occurrence of it (see `failure_kinds.py`).
        """
        return is_evaluation_side(self.kind) if is_registered(self.kind) else None


@dataclass(frozen=True)
class RolloutOutcome:
    """One rollout's outcome: its reward as the verifier returned it, and the failure if one was reported.

    `reward` is None when no reward exists, as for a failed `/run`: an infrastructure failure is not a score of zero.
    A masked rollout keeps the reward the verifier returned. `masked` is True when the verifier says that reward is not
    a valid measurement of the evaluated system (for example a lost session reports 0 and masks it), so a caller that
    computes scores must leave masked rollouts out. A failed `/run` has no sample, so its `masked` is False and its
    `reward` is None. `prompt` and `output` are `responses_create_params.input` and `response.output` as JSON text, for
    display and debugging, or None when the result has no such shape (for example a native episode result). `raw` is
    Gym's own `/run` response (the failure record for a failed `/run`), whose keys can change: use it for debugging only.
    """

    reward: Optional[float]
    failure: Optional[TaskFailure] = None
    masked: bool = False
    prompt: Optional[str] = None
    output: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


@experimental(once=True)
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


@experimental(once=True)
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
    """Runs single tasks of collated rows, flat or materialized, against Gym servers that are already running.

    The rows are checked and indexed by task id once. Each `run_task` call then selects one task's row, sends it
    through the agent's `/run`, or a materialized row through the environment server its taskset routes to, and
    returns the rollout's outcome. The running servers' configuration is read from the registry server once and shared
    by every call, flat or materialized, so build a new `TaskRunner` after the deployment's servers change.
    Concurrent calls in one event loop are supported.
    """

    @experimental(once=True)
    def __init__(self, rows: Sequence[Mapping[str, Any]], *, registry_server: BaseServerConfig):
        self._helper = RolloutCollectionHelper(reuse_server_client=True)
        self._registry_server = registry_server
        self._row_of_task = _first_row_of_each_task(rows)

    def task_ids(self) -> list[str]:
        return list(self._row_of_task)

    @experimental(once=True)
    async def run_task(self, task_id: str, *, rollout_id: Optional[str] = None) -> RolloutOutcome:
        """Run one rollout of a task.

        `rollout_id` names the rollout for Gym's rollout correlation (e.g. verification events). It must differ between
        concurrent calls; a fresh one is made when it is None.

        A `/run` call that fails (a client error, a bad JSON body, a timeout or an unusable reply) does not raise: the
        outcome has a `failure` and no reward. A refused connection is resent for a minute, and one that dropped is sent
        twice, as `gym eval run` does. Any other exception, such as a programming error, and cancellation
        propagate.
        """
        if task_id not in self._row_of_task:
            raise ConfigError(f"No input rows for task ids {[task_id]}")
        task_row = self._row_of_task[task_id]
        global_config_dict = (
            self._helper.setup_server_client(self._registry_server).global_config_dict
            if is_materialized_task_row(task_row)
            else None
        )
        with ignoring_command_line():
            rows = self._helper.preprocess_examples(
                [task_row], task_ids=[task_id], num_repeats=1, global_config_dict=global_config_dict
            )
        row = {**rows[0], "_ng_rollout_id": rollout_id or uuid.uuid4().hex}
        completions = self._helper.run_outcomes(
            [row],
            head_server_config=self._registry_server,
            max_num_tries=_MAX_RUN_TRIES,
            connect_grace_s=_CONNECT_GRACE_S,
        )
        try:
            (completion,) = completions
            resolved_row, result = await completion
        finally:
            await completions.aclose()
        outcome = _outcome(result)
        _record_outcome(outcome, _dispatch_label(self._helper, resolved_row))
        return outcome


def _dispatch_label(helper: RolloutCollectionHelper, row: Mapping[str, Any]) -> str:
    """The dispatch name of the row, or `unknown` when it names neither an environment server nor an agent.

    The rollout has finished by now, so a label must not cost the caller its outcome.
    """
    if NG_ENVIRONMENT_SERVER_KEY in row or AGENT_REF_KEY_NAME in row:
        return helper._dispatch_name(dict(row))
    return "unknown"


def _record_outcome(outcome: RolloutOutcome, dispatch_name: str) -> None:
    """Count the rollout in `gym.rollout.completed_total` and, when it is scored, in `gym.rollout.rewarded_total`.

    A failure makes the outcome `failed` and labels it with its kind, whatever the reward is. Without a failure the
    outcome is `scored` when there is a reward and `unscored` when there is none. Only a scored rollout with a finite
    reward is counted in `rewarded_total`.
    """
    failure = outcome.failure
    if failure is not None:
        record_rollout_completed(
            "failed",
            dispatch_name=dispatch_name,
            failure_kind=_metric_failure_kind(failure.kind),
            failure_type=failure.exception_type,
        )
        return
    if outcome.reward is None:
        record_rollout_completed("unscored", dispatch_name=dispatch_name)
        return
    record_rollout_completed("scored", dispatch_name=dispatch_name)
    if math.isfinite(outcome.reward):
        record_rollout_reward(outcome.reward, dispatch_name=dispatch_name, masked=outcome.masked)


def _raw(result: Mapping[str, Any] | RolloutFailureRecord) -> dict[str, Any]:
    return result.model_dump(mode="json") if isinstance(result, RolloutFailureRecord) else dict(result)


def _outcome(result: Mapping[str, Any] | RolloutFailureRecord) -> RolloutOutcome:
    if isinstance(result, RolloutFailureRecord):
        return RolloutOutcome(reward=None, failure=_failure_of_record(result), raw=_raw(result))
    kind, reason = result.get("failure_kind"), result.get("failure_reason")
    failure = None
    if kind:
        failure = TaskFailure(
            kind=str(kind),
            source="environment",
            delivery="delivered",
            reason=None if reason is None else str(reason),
        )
    prompt = (result.get("responses_create_params") or {}).get("input")
    output = (result.get("response") or {}).get("output")
    return RolloutOutcome(
        reward=result.get("reward"),
        failure=failure,
        masked=bool(result.get("mask_sample")),
        prompt=None if prompt is None else json.dumps(prompt, default=str),
        output=None if output is None else json.dumps(output, default=str),
        raw=_raw(result),
    )


def _failure_of_record(record: RolloutFailureRecord) -> TaskFailure:
    return TaskFailure(
        kind=record.sidecar_failure_class,
        source=record.source,
        delivery=record.delivery,
        stage=record.failure.stage,
        terminal=record.failure.terminal,
        reason=record.failure.failure_reason,
        http_status=record.http_status,
        exception_type=record.exception_type,
    )


def _first_row_of_each_task(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    ids = list(dict.fromkeys(row_task_id(row) for row in rows))
    return {row_task_id(rows[index]): rows[index] for index in select_tasks(rows, ids)}
