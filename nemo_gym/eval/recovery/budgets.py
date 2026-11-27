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
"""The run's retry, attempt and walltime budgets as the recovery decision reads and extends them."""

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.eval.recovery.evidence import (
    CollectedEvidence,
    ManagedRecoveryFact,
)


def _with_attempt_extension(
    facts: list[ManagedRecoveryFact] | None, extend_attempts: int
) -> list[ManagedRecoveryFact] | None:
    """The Store's facts as they read once ``extend_attempts`` more attempts are recorded for every rollout.

    Only a rollout the Store would run again under some cap (``replayable``) gets the higher cap, and it is pending
    while it has attempts left. Any other rollout keeps its cap and state: a model failure, a judge failure that kept
    its saved answer (its re-judgment budget is the saved cap), a saved result.
    """
    if not extend_attempts or facts is None:
        return facts
    return [
        fact.model_copy(
            update={
                "max_attempts": fact.max_attempts + extend_attempts,
                "pending": fact.pending or fact.attempts < fact.max_attempts + extend_attempts,
            }
        )
        if fact.replayable
        else fact
        for fact in facts
    ]


def _saved_attempt_cap(collected: "CollectedEvidence") -> int | None:
    """The run's saved per-rollout attempt cap, from the Store's facts; None when they do not say."""
    applied = collected.attempt_extension_applied
    for fact in collected.managed_selection or []:
        if fact.replayable is None or applied is None:
            return None
        return fact.max_attempts - applied if fact.replayable else fact.max_attempts
    return None


def retry_budget(collected: "CollectedEvidence") -> dict[str, dict[str, int | None]]:
    """The run's two retry budgets as the gate reports them, each its configured cap and its recorded extension.

    ``infra`` is the chain's infra-retry cap (``--extend-retries``); ``attempts`` is every rollout's attempt cap
    (``--extend-attempts``). A value that could not be read is None.
    """
    recorded = {"absent": 0, "ok": collected.infra_retry_extension}.get(collected.infra_retry_extension_state)
    attempts = {"absent": 0, "ok": collected.attempt_extension}.get(collected.attempt_extension_state)
    return {
        "infra": {
            "configured": collected.infra_retries_configured,
            "extension": None if recorded is None else int(recorded),
        },
        "attempts": {"configured": _saved_attempt_cap(collected), "extension": attempts},
    }


def _extend_attempts_refusal(collected: "CollectedEvidence") -> str | None:
    """Why this run cannot take an attempt extension, or None when it can."""
    if collected.attempt_extension_state == "malformed":
        return "the recorded attempt extension (artifacts/rollouts_attempt_extension) is unreadable"
    if not collected.managed_store or not collected.managed_selection:
        return "only a run written by a rollout store, whose saved selection was read, keeps a per-rollout attempt cap"
    if collected.attempt_extension_applied is None or any(
        fact.replayable is None for fact in collected.managed_selection
    ):
        return (
            "the run's pinned Gym predates attempt extensions, so its Store would not run the extra attempts; repin the "
            "run to a Gym that has them (--repin COMMIT --repin-reason TEXT) in the same resubmit"
        )
    if collected.attempt_extension_applied != (collected.attempt_extension or 0):
        return "the recorded attempt extension changed while it was read; inspect again"
    return None


def _rerun_masked_refusal(collected: "CollectedEvidence") -> str | None:
    """Why this run cannot run masked results without an artifact again, or None when it can."""
    if not collected.managed_store or not collected.managed_selection:
        return "only a run written by a rollout store, whose saved selection was read, can run a masked result again"
    if collected.rerun_masked_applied is not True:
        return (
            "the run's pinned Gym predates the rerun of masked results, so its Store would not run them; repin the "
            "run to a Gym that has it (--repin COMMIT --repin-reason TEXT) in the same resubmit"
        )
    return None


class OperatorBudget(BaseModel):
    """An operator's explicit budget for a run whose chain accounting is unknown.

    ``retries`` infra retries remain from now (``--extend-retries``, one by default) and ``max_walltime`` is the
    walltime cap (``--max-walltime``, raised as a recovery raises it). It only fills counters that are unknown: a
    readable counter is never lowered or reset.
    """

    model_config = ConfigDict(extra="forbid")

    max_walltime: str
    reason: str = Field(min_length=1)
    retries: int = Field(default=1, ge=1, le=1000)
    user: str


class RecoveryBudget(BaseModel):
    """The run's stored infra-retry and walltime budget and what its job chain used, as the chain files record them.

    ``max_infra_retries`` and ``max_walltime_s`` come from the benchmark's saved ``resume_config``;
    ``infra_retries_used`` from ``.gym_infra_retries`` and ``walltime_used_s`` from ``.gym_accumulated_walltime``
    (see ``resume_script``). The chain records admitted retries and allows another only while
    ``infra_retries_used < max_infra_retries``. ``None`` means the evidence is unavailable and stops the decision.
    """

    max_infra_retries: int | None = Field(default=None, ge=0)
    infra_retries_used: int | None = Field(default=None, ge=0)
    max_walltime_s: float | None = Field(default=None, gt=0, description="None: the run sets no walltime cap.")
    walltime_capped: bool = Field(default=False, description="Whether resume_config sets max_walltime.")
    walltime_used_s: float | None = Field(default=None, ge=0)
    live_job: bool | None = Field(default=None, description="A job runs or waits in this job dir.")
    live_job_name: str | None = Field(default=None, description="The live chain job, as 'id (STATE)', when known.")
    job_chain: list[str] = Field(default_factory=list, description="Job ids already run in this job dir.")
