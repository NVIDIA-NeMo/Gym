# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""simple_agent in an episode whose steps a caller drives: the step limit it reads, and the one invocation it keeps."""

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from nemo_gym.agents import AGENT_EPISODE_STEP_LIMIT_HEADER
from nemo_gym.trajectory.trajectory_record import AgentInvocation, ModelCallRef
from responses_api_agents.simple_agent.app import _episode_step_limit, _extend_episode_invocation


def _request(headers: dict[str, str]) -> Request:
    return Request({"type": "http", "headers": [(name.encode(), value.encode()) for name, value in headers.items()]})


def _invocation(invocation_id: str, *calls: str) -> AgentInvocation:
    return AgentInvocation(
        invocation_id=invocation_id, model_calls=[ModelCallRef(model_call_id=call) for call in calls]
    )


def test_the_step_limit_is_the_callers_or_none() -> None:
    assert _episode_step_limit(_request({})) is None
    assert _episode_step_limit(_request({AGENT_EPISODE_STEP_LIMIT_HEADER: "50"})) == 50


@pytest.mark.parametrize("raw", ["0", "-3", "fifty", ""])
def test_a_step_limit_that_is_not_a_positive_integer_is_refused(raw: str) -> None:
    with pytest.raises(HTTPException) as refused:
        _episode_step_limit(_request({AGENT_EPISODE_STEP_LIMIT_HEADER: raw}))
    assert refused.value.status_code == 422
    assert AGENT_EPISODE_STEP_LIMIT_HEADER in refused.value.detail


def test_a_steps_invocation_extends_the_episodes_record_with_its_model_calls() -> None:
    other = _invocation("subagent", "s1")
    records = [_invocation("root", "m1"), other]

    left = _extend_episode_invocation(records, [_invocation("root", "m2"), _invocation("helper", "h1")])

    assert [record.invocation_id for record in records] == ["root", "subagent"]
    assert [call.model_call_id for call in records[0].model_calls] == ["m1", "m2"]
    assert records[1] is other
    assert [invocation.invocation_id for invocation in left] == ["helper"]


def test_the_first_steps_invocation_is_appended_as_it_is() -> None:
    first = [_invocation("root", "m1")]

    assert _extend_episode_invocation([], first) is first
    assert _extend_episode_invocation([_invocation("subagent", "s1")], first) is first
    assert _extend_episode_invocation([_invocation("root", "m1")], []) == []
