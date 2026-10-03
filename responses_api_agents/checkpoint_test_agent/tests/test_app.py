# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import time
from unittest.mock import MagicMock

import pytest

from nemo_gym._checkpoint.agent import _Session
from nemo_gym._checkpoint.control import CheckpointRequest
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.checkpoint_test_agent.app import (
    HOLD_FIRST_MUTATED_BOUNDARY_ENV,
    PREFIX_MIN_TOKENS_ENV,
    WORKPLACE_PREFIX_AFTER_MUTATION_ENV,
    CheckpointTestAgent,
    CheckpointTestParticipant,
)
from responses_api_agents.simple_agent.app import SimpleAgentConfig


def _agent() -> CheckpointTestAgent:
    config = SimpleAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="checkpoint_test_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy"),
        resources_server=ResourcesServerRef(
            type="resources_servers",
            name="resources",
        ),
        max_steps=2,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": False}
    return CheckpointTestAgent(config=config, server_client=client)


def _participant_with_session(
    monkeypatch: pytest.MonkeyPatch, *, hold: bool
) -> tuple[CheckpointTestParticipant, _Session]:
    if hold:
        monkeypatch.setenv(HOLD_FIRST_MUTATED_BOUNDARY_ENV, "1")
    else:
        monkeypatch.delenv(HOLD_FIRST_MUTATED_BOUNDARY_ENV, raising=False)
    participant = CheckpointTestParticipant(hooks=None)
    session = _Session(key="session-1", episode_id=EpisodeId(rollout_id="rollout-1"))
    session.state = "running"
    participant._sessions[session.key] = session
    return participant, session


def _loop_state(*, step: int, pending_tools: bool):
    return lambda: {"step": step, "pending_tools": pending_tools}


def _request() -> CheckpointRequest:
    return CheckpointRequest(checkpoint_id="save-1", deadline_ts=time.time() + 60)


@pytest.mark.parametrize(
    ("hold", "step", "pending_tools"),
    [
        (False, 1, False),  # hold not requested
        (True, 0, False),  # before the first model call
        (True, 1, True),  # after a model call, before its tools ran
    ],
)
async def test_boundaries_that_do_not_qualify_pass_straight_through(
    monkeypatch: pytest.MonkeyPatch, hold: bool, step: int, pending_tools: bool
) -> None:
    participant, session = _participant_with_session(monkeypatch, hold=hold)

    await asyncio.wait_for(participant.at_boundary(session, _loop_state(step=step, pending_tools=pending_tools)), 1)

    assert session.state == "running"
    assert session.boundary is not None


async def test_first_post_tool_boundary_parks_for_a_real_checkpoint_and_stays_parked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    participant, session = _participant_with_session(monkeypatch, hold=True)
    snapshot = _loop_state(step=1, pending_tools=False)
    held = asyncio.create_task(participant.at_boundary(session, snapshot))

    # Held at the boundary without parking until a checkpoint asks for it.
    await asyncio.sleep(0.05)
    assert not held.done()
    assert session.state == "running" and session.boundary is None

    await participant.close_admission(_request())
    for _ in range(40):
        if session.state == "at_boundary":
            break
        await asyncio.sleep(0.05)
    assert session.state == "at_boundary"
    assert session.boundary is snapshot

    # Resuming the checkpoint does not release it: later checkpoints still export this boundary.
    await participant.open_admission()
    await asyncio.sleep(0.05)
    assert not held.done()
    assert session.state == "at_boundary"

    # Only the first qualifying boundary is held.
    other = _Session(key="session-2", episode_id=EpisodeId(rollout_id="rollout-2"))
    other.state = "running"
    participant._sessions[other.key] = other
    await asyncio.wait_for(participant.at_boundary(other, snapshot), 1)

    held.cancel()
    with pytest.raises(asyncio.CancelledError):
        await held


def test_workplace_prefix_mode_rewrites_only_later_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(WORKPLACE_PREFIX_AFTER_MUTATION_ENV, "1")
    monkeypatch.setenv(PREFIX_MIN_TOKENS_ENV, "512")
    agent = _agent()
    request = NeMoGymResponseCreateParamsNonStreaming(
        input="create the event",
        tools=[
            {
                "type": "function",
                "name": "calendar_create_event",
                "description": "Create a calendar event.",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
        max_output_tokens=64,
        metadata={"existing": "value"},
    )

    first = agent._prepare_model_request_for_turn(request, turn_index=1)
    assert first == request

    second = agent._prepare_model_request_for_turn(request, turn_index=2)
    assert second.tools == request.tools
    assert second.tool_choice == "none"
    assert second.parallel_tool_calls is False
    assert second.max_output_tokens == 512
    assert second.metadata["existing"] == "value"
    assert json.loads(second.metadata["extra_body"]) == {"min_tokens": 512}


def test_workplace_prefix_mode_rejects_invalid_min_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(WORKPLACE_PREFIX_AFTER_MUTATION_ENV, "1")
    monkeypatch.setenv(PREFIX_MIN_TOKENS_ENV, "zero")

    with pytest.raises(ValueError, match="must be an integer"):
        _agent()._prepare_model_request_for_turn(
            NeMoGymResponseCreateParamsNonStreaming(input="test"),
            turn_index=2,
        )
