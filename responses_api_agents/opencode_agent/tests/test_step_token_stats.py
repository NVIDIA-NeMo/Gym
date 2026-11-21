# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-call token stats read from OpenCode's step-finish parts, without HTTP capture."""

from nemo_gym.config import ModelServerRef
from nemo_gym.rollout_observability import TrajectoryRecord
from responses_api_agents.opencode_agent.app import _parse_opencode_session
from responses_api_agents.opencode_agent.tests.test_app import _session_db
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import setup  # noqa: F401  (fixture)


POLICY = ModelServerRef(type="responses_api_models", name="policy")


def _assistant(created, tokens, *, reason="tool-calls", model="test-model"):
    message = {"role": "assistant", "modelID": model, "time": {"created": created, "completed": created + 500}}
    parts = [
        {"type": "step-start"},
        {"type": "text", "text": f"turn at {created}"},
        {"type": "step-finish", "reason": reason, "tokens": tokens},
    ]
    return (message, parts)


def _parse(db, **kwargs):
    trajectory = TrajectoryRecord(task_id="t", rollout_id="0-0")
    _parse_opencode_session(db, "fallback", trajectory, model_ref=POLICY, **kwargs)
    return trajectory


def test_step_finish_usage_becomes_one_model_call_per_turn(tmp_path):
    db = _session_db(
        tmp_path,
        [
            _assistant(1000, {"input": 100, "output": 10, "reasoning": 5, "cache": {"read": 20, "write": 0}}),
            _assistant(3000, {"input": 300, "output": 40, "cache": {"read": 0, "write": 7}}, reason="stop"),
        ],
    )
    trajectory = _parse(db, step_token_stats=True)
    assert len(trajectory.turns) == len(trajectory.model_calls) == 2
    first, second = trajectory.model_calls
    assert first.token_stats.prompt_tokens == 120  # input + cache read + cache write
    assert first.token_stats.completion_tokens == 15  # output + reasoning
    assert first.token_stats.reasoning_tokens == 5
    assert first.token_stats.cached_tokens == 20
    assert first.token_stats.total_tokens == 135
    assert second.token_stats.prompt_tokens == 307
    assert second.token_stats.reasoning_tokens == 0
    # The peak context is simply the last call's prompt once no compaction runs.
    assert max(call.token_stats.prompt_tokens for call in trajectory.model_calls) == 307
    assert [turn.model_calls[0].model_call_id for turn in trajectory.turns] == [
        c.model_call_id for c in (first, second)
    ]
    assert first.client_assistant_message_id == trajectory.turns[0].source_message_id
    assert first.client_session_id == trajectory.turns[0].invocation_id == "root"
    assert (first.started_at, first.completed_at) == (1.0, 1.5)
    assert first.response_metadata.model == "test-model"
    assert (first.response_metadata.finish_reason, second.response_metadata.finish_reason) == ("tool-calls", "stop")


def test_step_token_stats_off_by_default_leaves_turns_unlinked(tmp_path):
    db = _session_db(tmp_path, [_assistant(1000, {"input": 100, "output": 10})])
    trajectory = _parse(db)
    assert trajectory.model_calls == []
    assert trajectory.turns[0].model_calls == []


def test_step_finish_without_usage_yields_empty_stats_not_zeros(tmp_path):
    message = {"role": "assistant", "time": {"created": 1000, "completed": 1500}}
    db = _session_db(
        tmp_path, [(message, [{"type": "step-start"}, {"type": "text", "text": "x"}, {"type": "step-finish"}])]
    )
    trajectory = _parse(db, step_token_stats=True)
    [call] = trajectory.model_calls
    assert call.token_stats.prompt_tokens is None
    assert call.response_metadata.model is None


def test_sandbox_receipt_carries_step_usage_when_capture_is_off(setup, tmp_path):
    from fastapi.testclient import TestClient

    from responses_api_agents.opencode_agent.tests.test_continuation import (
        activate,
        close,
        install_artifact_runner,
        start,
    )

    agent, sandbox = setup
    agent.config.step_token_stats = True
    install_artifact_runner(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        request = start(client)
        activate(client, 1, "delta-1")
        receipt = close(client, request)
    trajectory = TrajectoryRecord.model_validate(receipt["trajectory"])
    assert trajectory.model_calls, "step-finish usage should be on the trajectory when nothing else captures calls"
    assert all(len(turn.model_calls) == 1 for turn in trajectory.turns)
    by_id = {call.model_call_id: call for call in trajectory.model_calls}
    for turn in trajectory.turns:
        call = by_id[turn.model_calls[0].model_call_id]
        assert call.client_assistant_message_id == turn.source_message_id
        # The fake runner's step-finish parts carry no usage, so the stats stay empty, never zero.
        assert call.token_stats.prompt_tokens is None
