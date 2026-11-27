# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run single tasks of planned rows against Gym servers that are already running."""

import math
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from nemo_gym.config.config_types import BaseServerConfig, ConfigError, ignoring_command_line
from nemo_gym.config.global_config import (
    AGENT_REF_KEY_NAME,
)
from nemo_gym.core.episode_types import is_materialized_task_row
from nemo_gym.data.task_selection import row_task_id, select_tasks
from nemo_gym.environment.in_process.outcomes import RolloutOutcome, _outcome
from nemo_gym.rollouts.rollout_collection import (
    NG_ENVIRONMENT_SERVER_KEY,
    RolloutCollectionHelper,
    _metric_failure_kind,
)
from nemo_gym.telemetry.gym_metrics import record_rollout_completed, record_rollout_reward
from nemo_gym.utils.decorators import experimental


# The bounds `gym eval run` puts on a `/run` call: a connection that was made and dropped is sent twice, and one that
# was refused is resent for a minute (`dispatch_connect_grace_s`). Without them a dead server is retried for ever.
_MAX_RUN_TRIES = 2


_CONNECT_GRACE_S = 60.0


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


def _first_row_of_each_task(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    ids = list(dict.fromkeys(row_task_id(row) for row in rows))
    return {row_task_id(rows[index]): rows[index] for index in select_tasks(rows, ids)}
