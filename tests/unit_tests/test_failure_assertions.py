# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`raises_agent_failure` accepts only an agent route failure linked to its cause, so a lifecycle test cannot pass
on the wrong failure."""

import pytest

from nemo_gym.core.failure_kinds import AGENT_RUN_ERROR
from nemo_gym.server_runtime.episode_types import FailureBoundary
from nemo_gym.testing.failure_assertions import raises_agent_failure
from nemo_gym.typed_exceptions import NgFailureEvent, agent_route_failure


def test_accepts_a_route_failure_and_exposes_its_cause() -> None:
    cause = RuntimeError("sandbox lost")
    with raises_agent_failure(RuntimeError, match="sandbox lost") as caught:
        raise agent_route_failure(cause, operation="agent.responses") from cause
    assert caught.value is cause


def test_rejects_a_failure_of_another_kind() -> None:
    cause = RuntimeError("sandbox lost")
    other = NgFailureEvent(
        boundary=FailureBoundary.AGENT,
        failure_kind=AGENT_RUN_ERROR,
        failure_reason="run failed",
        operation="agent.responses",
    )
    with pytest.raises(pytest.fail.Exception, match=f"got {AGENT_RUN_ERROR}"):
        with raises_agent_failure(RuntimeError):
            raise other from cause


def test_rejects_a_route_failure_without_a_cause() -> None:
    with pytest.raises(AssertionError):
        with raises_agent_failure(RuntimeError):
            raise agent_route_failure(RuntimeError("sandbox lost"), operation="agent.responses")


def test_rejects_the_raw_error_without_the_boundary() -> None:
    with pytest.raises(pytest.fail.Exception, match="got RuntimeError"):
        with raises_agent_failure(RuntimeError):
            raise RuntimeError("sandbox lost")


def test_rejects_a_block_that_raises_nothing() -> None:
    with pytest.raises(pytest.fail.Exception, match="did not raise"):
        with raises_agent_failure(RuntimeError):
            pass


def test_a_cause_of_another_type_is_not_the_expected_error() -> None:
    cause = ValueError("bad input")
    with pytest.raises(ValueError, match="bad input"):
        with raises_agent_failure(RuntimeError):
            raise agent_route_failure(cause, operation="agent.responses") from cause
