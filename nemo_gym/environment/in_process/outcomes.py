# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a rollout returned: its reward as the verifier gave it, and the failure Gym observed, if any."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from nemo_gym.core.failure_kinds import (
    FailureStage,
    is_evaluation_side,
    is_registered,
)
from nemo_gym.rollouts.rollout_outcomes import RolloutFailureRecord


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
