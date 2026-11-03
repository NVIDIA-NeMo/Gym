# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A provider-confirmed memory kill of the agent's own sandbox during the agent's command is a final zero."""

from unittest.mock import AsyncMock

import pytest

from nemo_gym.artifacts import NG_FAILURE_CLASS_KEY
from nemo_gym.core.failure_records import FailureBoundary, SandboxFailureDetails
from nemo_gym.core.failure_transport import FailureHeader
from nemo_gym.core.typed_exceptions import NgFailureEvent, failure_report_for
from nemo_gym.rollouts import NG_TERMINAL_KEY
from nemo_gym.rollouts.rollout_collection import NG_FAILURE_COUNTS_AS_ZERO_KEY, RolloutCollectionHelper
from nemo_gym.rollouts.rollout_recovery import agent_phase_oom
from tests.unit_tests.rollouts.test_rollout_collection import FakeResponse, failing_row, install_fake_server_client


def _sandbox_event(*, kind="provider_oom_killed", stage="agent", reason="OOMKilled") -> NgFailureEvent:
    return NgFailureEvent(
        boundary=FailureBoundary.SANDBOX,
        failure_kind=kind,
        failure_reason=f"The provider ended the sandbox ({reason})",
        details=SandboxFailureDetails(sandbox_id="box-1", provider="opensandbox", provider_status_reason=reason),
        operation="agent.sandbox_exec",
        stage=stage,
    )


def _headers(event: NgFailureEvent) -> dict:
    return FailureHeader.encode(failure_report_for(event))


@pytest.mark.parametrize(
    "event, expected",
    [
        (_sandbox_event(), True),
        (_sandbox_event(stage="verification"), False),  # a verifier's sandbox is infrastructure
        (_sandbox_event(stage="seed"), False),  # so is setup
        (_sandbox_event(kind="provider_unavailable", reason="Evicted"), False),  # an eviction is not a memory kill
    ],
)
def test_only_an_agent_stage_memory_kill_is_the_models_result(event, expected) -> None:
    assert agent_phase_oom(failure_report_for(event).events) is expected


def test_no_events_or_unreadable_events_are_not_an_oom() -> None:
    assert agent_phase_oom(None) is False
    assert agent_phase_oom({}) is False
    assert (
        agent_phase_oom({"e": {"boundary": "agent", "failure_kind": "agent_route_failed", "stage": "agent"}}) is False
    )


async def test_a_raised_failure_with_an_agent_stage_oom_is_a_final_zero(monkeypatch) -> None:
    reply = FakeResponse(500, {"detail": "agent failed"})
    reply.headers = _headers(_sandbox_event())
    install_fake_server_client(monkeypatch, AsyncMock(return_value=reply))
    _, result = await next(RolloutCollectionHelper().run_examples([failing_row()], route_failures_to_sidecar=True))
    assert result[NG_TERMINAL_KEY] is True and result[NG_FAILURE_COUNTS_AS_ZERO_KEY] is True


async def test_a_failure_class_result_with_an_agent_stage_oom_is_a_final_zero(monkeypatch) -> None:
    reply = FakeResponse(200, {NG_FAILURE_CLASS_KEY: "harbor_failed", "reward": 0.0, "failure_reason": "trial failed"})
    reply.headers = _headers(_sandbox_event())
    install_fake_server_client(monkeypatch, AsyncMock(return_value=reply))
    _, result = await next(RolloutCollectionHelper().run_examples([failing_row()], route_failures_to_sidecar=True))
    assert result[NG_TERMINAL_KEY] is True and result[NG_FAILURE_COUNTS_AS_ZERO_KEY] is True


@pytest.mark.parametrize(
    "event",
    [_sandbox_event(stage="verification"), _sandbox_event(kind="provider_unavailable", reason="Error")],
    ids=["verification-stage", "non-memory-reason"],
)
async def test_any_other_sandbox_ending_stays_a_retryable_failure(monkeypatch, event) -> None:
    reply = FakeResponse(500, {"detail": "agent failed"})
    reply.headers = _headers(event)
    install_fake_server_client(monkeypatch, AsyncMock(return_value=reply))
    _, result = await next(RolloutCollectionHelper().run_examples([failing_row()], route_failures_to_sidecar=True))
    assert NG_TERMINAL_KEY not in result and NG_FAILURE_COUNTS_AS_ZERO_KEY not in result
