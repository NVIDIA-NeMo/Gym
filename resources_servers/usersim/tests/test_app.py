# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import ServerClient
from resources_servers.usersim.app import UserSimResourcesServer, UserSimResourcesServerConfig


USERSIM_REVISION = "b3381ae021baac2a6fb314b5f08a55845243017d"  # pragma: allowlist secret


def _resolved_row(probe_type: str = "safety_agentic") -> dict:
    return {
        "probe_type": probe_type,
        "probe_family": probe_type,
        "probe_variant": "default",
        "persona": {"first_name": "Morgan", "age": 42},
        "locale": "en_US",
        "conversation_language": "English",
        "trajectory_id": f"usersim-{probe_type}",
        "usersim_provenance": {"code_sha": USERSIM_REVISION},
        "usersim_config": {"random_seed": 42},
    }


def _seed_body(row: dict | None = None) -> dict:
    return {
        "resources_session_id": "resources-session-0",
        "episode_id": {"rollout_id": "0-0", "attempt": 0},
        "task_id": {"taskset": "usersim:example", "task_id": "0"},
        "task_data": {"resolved_row": row or _resolved_row()},
    }


def _tool_session(*, tools: bool = False) -> MagicMock:
    assistant_tools = (
        [
            {
                "type": "function",
                "function": {
                    "name": "safe_action",
                    "description": "Perform an action.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        if tools
        else []
    )
    session = MagicMock()
    session.assistant_tools = assistant_tools
    session.close = AsyncMock()
    return session


def _app(*, with_scorer_model: bool = False) -> UserSimResourcesServer:
    config = UserSimResourcesServerConfig(
        host="127.0.0.1",
        port=12345,
        entrypoint="app.py",
        name="usersim",
        probe_scorer_model=(
            ModelServerRef(type="responses_api_models", name="support") if with_scorer_model else None
        ),
    )
    return UserSimResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _verify_body(row: dict | None = None) -> dict:
    resolved_row = row or _resolved_row()
    return {
        "episode_id": {"rollout_id": "0-0", "attempt": 0},
        "task_id": {"taskset": "usersim:example", "task_id": "0"},
        "verification_input": {
            "resolved_row": resolved_row,
            "usersim_result": {
                "conversation_messages": [
                    {"role": "user", "content": "Can you help?"},
                    {"role": "assistant", "content": "Yes."},
                ],
                "conversation_status": True,
                "simulation_outcome": {"status": "completed"},
            },
            "invocations": [],
            "episode_interaction_protocol": "usersim.ConversationLoop",
        },
    }


def _assistant_eval(*, scorer_pass: bool = True) -> dict:
    return {
        "envelope": {
            "axes": ["helpfulness", "accuracy", "coherence"],
            "scorers": ["safety_agentic"],
        },
        "axes": {
            "helpfulness": {"judge_model": {"score": 5, "reasoning": "Helpful."}},
            "accuracy": {"judge_model": {"score": 4, "reasoning": "Accurate."}},
            "coherence": {"judge_model": {"score": 3, "reasoning": "Coherent."}},
        },
        "scorers": {
            "safety_agentic": {
                "status_proposal": scorer_pass,
            }
        },
        "skipped": False,
        "skipped_reason": None,
    }


def test_resources_constructs_runtime_only_from_unchanged_resolved_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _resolved_row()
    tool_session = _tool_session()
    from_resolved_row = MagicMock(return_value=tool_session)
    monkeypatch.setattr(
        "usersim.engine.external.ProbeToolSession.from_resolved_row",
        from_resolved_row,
    )

    with TestClient(_app().setup_webserver()) as client:
        response = client.post("/seed_session", json=_seed_body(row))

    assert response.status_code == 200
    assert response.json()["resolved_row"] == row
    from_resolved_row.assert_called_once()
    assert from_resolved_row.call_args.args == (row,)
    assert set(from_resolved_row.call_args.kwargs) == {"models"}


def test_seed_rejects_wrong_revision_before_runtime_construction() -> None:
    row = _resolved_row()
    row["usersim_provenance"]["code_sha"] = "a" * 40

    with TestClient(_app().setup_webserver()) as client:
        response = client.post("/seed_session", json=_seed_body(row))

    assert response.status_code == 422
    assert response.json()["detail"] == "Resolved row does not match the configured UserSim revision"


def test_named_tool_route_separates_plain_output_from_hidden_native_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _tool_session(tools=True)
    evidence = {
        "turn_id": "turn-1",
        "rounds": [
            {
                "turn_id": "turn-1",
                "round_id": "round-1",
                "receipts": [
                    {
                        "tool_call_id": "call-1",
                        "tool_name": "safe_action",
                        "arguments": {"value": "one"},
                        "raw_tool_call": {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "safe_action", "arguments": '{"value":"one"}'},
                        },
                        "payload": "plain safety payload",
                        "turn_idx": 4,
                        "call_idx": 0,
                        "effect_state": {},
                    }
                ],
                "limit_reached": False,
            }
        ],
        "final_effect_state": {"metadata": {}, "outcome": {}},
    }
    session.execute_call = AsyncMock(
        return_value=SimpleNamespace(
            payload="plain safety payload",
            evidence=SimpleNamespace(to_dict=lambda: evidence),
            limit_reached=False,
        )
    )
    monkeypatch.setattr(UserSimResourcesServer, "_create_tool_session", lambda *_args, **_kwargs: session)

    assistant_response = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "safe_action", "arguments": '{"value":"one"}'},
            }
        ],
    }
    request_body = {
        "arguments": {"value": "one"},
        "tool_call_context": {
            "turn_id": "turn-1",
            "first_tool_turn_idx": 4,
            "max_tool_calls": 2,
            "state_snapshot": {"messages": [], "metadata": {}, "outcome": {}},
        },
        "tool_call_id": "call-1",
        "round_id": "round-1",
        "assistant_response": assistant_response,
    }
    with TestClient(_app().setup_webserver()) as client:
        assert client.post("/seed_session", json=_seed_body()).status_code == 200
        result = client.post("/safe_action", json=request_body)
        old_runtime_route = client.post("/runtime/tool_calls", json={})

    assert result.status_code == 200
    assert result.json() == {
        "output": "plain safety payload",
        "tool_call_context": evidence,
        "limit_reached": False,
    }
    assert old_runtime_route.status_code == 404
    native_request = session.execute_call.await_args.args[0]
    assert native_request.tool_name == "safe_action"
    assert native_request.arguments == {"value": "one"}
    assert native_request.assistant_response == assistant_response


def test_close_session_closes_owned_tool_session(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _tool_session()
    monkeypatch.setattr(UserSimResourcesServer, "_create_tool_session", lambda *_args, **_kwargs: session)

    with TestClient(_app().setup_webserver()) as client:
        seed = client.post("/seed_session", json=_seed_body()).json()
        closed = client.post(
            "/close_session",
            json={
                "resources_session_id": seed["resources_session_id"],
                "episode_id": {"rollout_id": "0-0", "attempt": 0},
            },
        )

    assert closed.status_code == 200
    session.close.assert_awaited_once()


def test_verify_uses_one_native_evaluator_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _tool_session()
    runtime = SimpleNamespace(evaluate=AsyncMock(return_value=_assistant_eval()))
    runtime_type = MagicMock(return_value=runtime)
    monkeypatch.setattr(UserSimResourcesServer, "_create_tool_session", lambda *_args, **_kwargs: session)
    monkeypatch.setattr("usersim.engine.external.TrajectoryEvaluatorRuntime", runtime_type)

    with TestClient(_app(with_scorer_model=True).setup_webserver()) as client:
        assert client.post("/seed_session", json=_seed_body()).status_code == 200
        verified = client.post("/verify", json=_verify_body())

    assert verified.status_code == 200
    result = verified.json()
    assert result["reward"] == pytest.approx(0.8)
    assert result["mask_sample"] is False
    assert result["scenario_completed"] is True
    assert result["reward_components"] == {
        "participants_completed": 1.0,
        "native_conversation_status": 1.0,
        "native_scorer_applied": 1.0,
        "native_scorer_pass": 1.0,
        "trajectory_evaluator_applied": 1.0,
        "assistant_quality": pytest.approx(0.8),
        "quality.helpfulness": 1.0,
        "quality.accuracy": 0.8,
        "quality.coherence": 0.6,
    }
    assert result["verifier_data"]["native_scorer_name"] == "safety_agentic"
    assert result["verifier_data"]["native_scores"] == {"status_proposal": True}
    assert result["verifier_data"]["assistant_eval"] == _assistant_eval()
    runtime.evaluate.assert_awaited_once()
    evaluated_row = runtime.evaluate.await_args.args[0]
    assert evaluated_row["probe_type"] == "safety_agentic"
    assert evaluated_row["conversation_status"] is True


def test_verify_treats_native_scorer_rejection_as_policy_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _tool_session()
    runtime = SimpleNamespace(evaluate=AsyncMock(return_value=_assistant_eval(scorer_pass=False)))
    monkeypatch.setattr(UserSimResourcesServer, "_create_tool_session", lambda *_args, **_kwargs: session)
    monkeypatch.setattr("usersim.engine.external.TrajectoryEvaluatorRuntime", MagicMock(return_value=runtime))

    with TestClient(_app(with_scorer_model=True).setup_webserver()) as client:
        assert client.post("/seed_session", json=_seed_body()).status_code == 200
        verified = client.post("/verify", json=_verify_body())

    assert verified.status_code == 200
    assert verified.json()["reward"] == 0.0
    assert verified.json()["mask_sample"] is False
    assert verified.json()["scenario_completed"] is False


def test_verify_converts_native_evaluator_exception_to_masked_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _tool_session()
    runtime = SimpleNamespace(evaluate=AsyncMock(side_effect=ValueError("judge payload was invalid")))
    monkeypatch.setattr(UserSimResourcesServer, "_create_tool_session", lambda *_args, **_kwargs: session)
    monkeypatch.setattr("usersim.engine.external.TrajectoryEvaluatorRuntime", MagicMock(return_value=runtime))

    with TestClient(_app(with_scorer_model=True).setup_webserver()) as client:
        assert client.post("/seed_session", json=_seed_body()).status_code == 200
        verified = client.post("/verify", json=_verify_body())

    assert verified.status_code == 200
    result = verified.json()
    assert result["reward"] == 0.0
    assert result["mask_sample"] is True
    assert result["failure_kind"] == "judge_failed"
    assert result["failure_reason"] == "ValueError: judge payload was invalid"
