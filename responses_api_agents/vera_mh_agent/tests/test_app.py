# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY
from nemo_gym.server_utils import ServerClient
from responses_api_agents.vera_mh_agent.app import (
    DEFAULT_PROVIDER_SYSTEM_PROMPT,
    DEFAULT_START_PROMPT,
    REMINDER_PATH,
    SIMULATION_FAILURE_CLASS,
    TERMINATION_SIGNAL,
    VeraMHAgent,
    VeraMHAgentConfig,
    VeraMHAgentRunRequest,
    build_messages,
    ensure_provider_has_last_turn,
    format_transcript,
)


REMINDER = REMINDER_PATH.read_text(encoding="utf-8")


def _response_payload(text: str, *, model: str, response_id: str, incomplete: bool = False) -> dict:
    return {
        "id": response_id,
        "created_at": 0,
        "model": model,
        "object": "response",
        "output": [
            {"type": "reasoning", "id": f"{response_id}-r", "summary": [{"type": "summary_text", "text": "thinking"}]},
            {
                "type": "message",
                "id": f"{response_id}-m",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            },
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "incomplete_details": {"reason": "max_output_tokens"} if incomplete else None,
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 10,
            "output_tokens_details": {"reasoning_tokens": 3},
            "total_tokens": 110,
        },
    }


class _HTTP:
    ok = True
    status = 200

    def __init__(self, payload):
        self.payload = payload
        self.cookies = {}

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _agent(persona_script, provider_script, *, observability=False, verify_result=None, max_turns=4, **overrides):
    """Scripts are callables ``(messages) -> text`` (or raise) for the persona and provider model servers."""
    config = VeraMHAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="vera_mh_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="vera_mh"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        user_model_servers={"gpt-5.2": ModelServerRef(type="responses_api_models", name="vera_mh_user_gpt52")},
        max_turns=max_turns,
        **overrides,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": observability}
    calls = []
    counters = {"n": 0}

    async def _post(server_name, url_path, json=None, cookies=None, **kwargs):
        calls.append((server_name, url_path, json))
        if url_path == "/seed_session":
            return _HTTP({})
        if url_path == "/verify":
            return _HTTP(verify_result if verify_result is not None else {**json, "reward": 0.75, "ratings": {}})
        if url_path == "/aggregate_metrics":
            return _HTTP({})
        counters["n"] += 1
        messages = json.input
        if server_name == "vera_mh_user_gpt52":
            text = persona_script(messages)
            return _HTTP(_response_payload(text, model="gpt-5.2-2025-12-11", response_id=f"persona-{counters['n']}"))
        assert server_name == "policy_model"
        text = provider_script(messages)
        incomplete = text.endswith("[TRUNCATED]")
        return _HTTP(
            _response_payload(
                text, model="moonshotai/Kimi-K3", response_id=f"provider-{counters['n']}", incomplete=incomplete
            )
        )

    client.post = AsyncMock(side_effect=_post)
    return VeraMHAgent(config=config, server_client=client), calls


def _request(**overrides) -> VeraMHAgentRunRequest:
    fields = dict(
        id="vera_mh__Abigail__gpt-5.2",
        persona_system_prompt="You are roleplaying as a human user...",
        user_simulator="gpt-5.2",
        max_turns=None,
        responses_create_params={
            "input": [{"role": "system", "content": "You are a helpful AI assistant."}],
            "temperature": 0.3,
        },
    )
    fields.update(overrides)
    return VeraMHAgentRunRequest(**fields)


def _fastapi_request():
    request = MagicMock()
    request.cookies = {}
    return request


def test_turn_rounding_matches_upstream():
    assert ensure_provider_has_last_turn(30, True) == 30
    assert ensure_provider_has_last_turn(29, True) == 30
    assert ensure_provider_has_last_turn(30, False) == 31
    assert ensure_provider_has_last_turn(31, False) == 31


def test_message_building_mirrors_upstream_roles_and_reminder():
    conversation = [
        {"turn": 1, "speaker": "persona", "input": DEFAULT_START_PROMPT, "response": "hi", "early_termination": False},
        {"turn": 2, "speaker": "provider", "input": "hi", "response": "hello, what's up?", "early_termination": False},
    ]
    persona_view = build_messages("persona", conversation, REMINDER)
    assert persona_view == [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": REMINDER + "hello, what's up?"},
    ]
    provider_view = build_messages("provider", conversation, REMINDER)
    assert provider_view == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello, what's up?"}]
    # the reminder is only added when the latest turn is the provider's
    assert build_messages("persona", conversation[:1], REMINDER) == [{"role": "assistant", "content": "hi"}]
    assert build_messages("persona", [{"turn": 0, "response": DEFAULT_START_PROMPT}], REMINDER) == [
        {"role": "user", "content": DEFAULT_START_PROMPT}
    ]


def test_transcript_format_matches_upstream():
    conversation = [
        {"turn": 1, "speaker": "persona", "response": "hi", "early_termination": False},
        {"turn": 2, "speaker": "provider", "response": "hello", "early_termination": False},
        {"turn": 3, "speaker": "persona", "response": f"bye {TERMINATION_SIGNAL}", "early_termination": True},
    ]
    assert format_transcript(conversation) == (
        f"user: hi\n\nchatbot: hello\n\nuser: bye {TERMINATION_SIGNAL}\n[CONVERSATION ENDED - persona signaled termination]\n"
    )


async def test_full_conversation_alternates_and_routes_each_role_to_its_model_server():
    persona_turns = iter(["i feel awful", "nothing helps"])
    provider_turns = iter(["I'm here for you.", "Would you like to talk about what happened?"])
    agent, calls = _agent(lambda m: next(persona_turns), lambda m: next(provider_turns), max_turns=4)
    result = await agent.run(_fastapi_request(), _request())
    model_calls = [(name, params) for name, path, params in calls if path == "/v1/responses"]
    assert [name for name, _ in model_calls] == [
        "vera_mh_user_gpt52",
        "policy_model",
        "vera_mh_user_gpt52",
        "policy_model",
    ]
    first_persona = model_calls[0][1].input
    assert first_persona == [
        {"role": "system", "content": "You are roleplaying as a human user..."},
        {"role": "user", "content": DEFAULT_START_PROMPT},
    ]
    first_provider = model_calls[1][1].input
    assert first_provider == [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "i feel awful"},
    ]
    assert model_calls[1][1].temperature == 0.3  # provider sampling comes from the dataset row
    assert model_calls[0][1].temperature is None  # persona uses its own (default) settings
    second_persona = model_calls[2][1].input
    assert second_persona[1] == {"role": "assistant", "content": "i feel awful"}
    assert second_persona[2]["role"] == "user" and second_persona[2]["content"].startswith("REMINDER:")
    assert second_persona[2]["content"].endswith("I'm here for you.")
    verify_payload = next(params for name, path, params in calls if path == "/verify")
    assert verify_payload["turn_count"] == 4 and verify_payload["early_termination"] is False
    assert verify_payload["transcript"].startswith(
        "user: i feel awful\n\nchatbot: I'm here for you.\n\nuser: nothing helps"
    )
    assert [turn["speaker"] for turn in verify_payload["conversation"]] == [
        "persona",
        "provider",
        "persona",
        "provider",
    ]
    assert verify_payload["conversation"][0]["input"] == DEFAULT_START_PROMPT
    assert verify_payload["conversation"][1]["input"] == "i feel awful"
    assert (
        verify_payload["response"]["output"][-1]["content"][0]["text"] == "Would you like to talk about what happened?"
    )
    assert verify_payload["response"]["usage"]["input_tokens"] == 200  # provider usage only (2 calls)
    assert verify_payload["simulation"]["persona_usage"]["input_tokens"] == 200
    assert verify_payload["simulation"]["max_turns_effective"] == 4
    assert verify_payload["simulation"]["provider_empty_turns"] == 0
    assert verify_payload["responses_create_params"]["input"][0]["role"] == "system"
    assert result.reward == 0.75 and NG_FAILURE_CLASS_KEY not in result.model_dump()


async def test_termination_signal_from_persona_ends_the_conversation():
    persona_turns = iter(["hi", f"thanks, bye {TERMINATION_SIGNAL}"])
    agent, calls = _agent(lambda m: next(persona_turns), lambda m: "hello", max_turns=10)
    await agent.run(_fastapi_request(), _request())
    verify_payload = next(params for name, path, params in calls if path == "/verify")
    assert verify_payload["turn_count"] == 3 and verify_payload["early_termination"] is True
    assert verify_payload["conversation"][-1]["early_termination"] is True
    assert verify_payload["transcript"].endswith("[CONVERSATION ENDED - persona signaled termination]\n")


async def test_provider_does_not_terminate_and_odd_max_turns_is_rounded_up():
    agent, calls = _agent(lambda m: "hi", lambda m: f"bye {TERMINATION_SIGNAL}", max_turns=3)
    await agent.run(_fastapi_request(), _request())
    verify_payload = next(params for name, path, params in calls if path == "/verify")
    assert verify_payload["turn_count"] == 4 and verify_payload["early_termination"] is False


async def test_dataset_row_max_turns_overrides_config():
    agent, calls = _agent(lambda m: "hi", lambda m: "hello", max_turns=10)
    await agent.run(_fastapi_request(), _request(max_turns=2))
    verify_payload = next(params for name, path, params in calls if path == "/verify")
    assert verify_payload["turn_count"] == 2


async def test_simulator_failure_becomes_a_sidecar_row_not_a_safety_failure():
    def persona(messages):
        raise RuntimeError("simulator quota exhausted")

    agent, calls = _agent(persona, lambda m: "hello")
    result = await agent.run(_fastapi_request(), _request())
    dumped = result.model_dump()
    assert dumped[NG_FAILURE_CLASS_KEY] == SIMULATION_FAILURE_CLASS
    assert dumped["failure_stage"] == "persona_turn_1" and "quota" in dumped["error"]
    assert result.reward == 0.0
    assert not any(path == "/verify" for _, path, _ in calls)


async def test_unknown_simulator_is_a_configuration_failure():
    agent, _ = _agent(lambda m: "hi", lambda m: "hello")
    result = await agent.run(_fastapi_request(), _request(user_simulator="claude-opus-4-5"))
    assert result.model_dump()[NG_FAILURE_CLASS_KEY] == SIMULATION_FAILURE_CLASS


async def test_empty_and_truncated_provider_turns_are_counted():
    provider_turns = iter(["", "partial [TRUNCATED]"])
    agent, calls = _agent(lambda m: "hi", lambda m: next(provider_turns), max_turns=4)
    await agent.run(_fastapi_request(), _request())
    verify_payload = next(params for name, path, params in calls if path == "/verify")
    assert verify_payload["simulation"]["provider_empty_turns"] == 1
    assert verify_payload["simulation"]["provider_truncated_turns"] == 1


async def test_trajectory_is_attached_when_observability_is_enabled(monkeypatch):
    agent, calls = _agent(lambda m: "hi", lambda m: "hello", observability=True, max_turns=2)
    monkeypatch.setattr(VeraMHAgent, "rollout_id_from_run", lambda self, body: "0-0")
    result = await agent.run(_fastapi_request(), _request())
    trajectory = result.model_dump()["ng_trajectory"]
    assert [inv["invocation_id"] for inv in trajectory["invocations"]] == ["root", "user_simulator"]
    assert trajectory["invocations"][0]["model_calls"][0]["model_ref"]["name"] == "policy_model"
    assert trajectory["invocations"][1]["model_calls"][0]["model_ref"]["name"] == "vera_mh_user_gpt52"
    assert len(trajectory["turns"]) == 1 and trajectory["turns"][0]["turn_no"] == 1
    assert trajectory["turns"][0]["reasoning_content"] is not None


async def test_provider_system_prompt_defaults_when_row_has_none():
    agent, calls = _agent(lambda m: "hi", lambda m: "hello", max_turns=2)
    await agent.run(_fastapi_request(), _request(responses_create_params={"input": []}))
    provider_call = next(params for name, path, params in calls if name == "policy_model")
    assert provider_call.input[0] == {"role": "system", "content": DEFAULT_PROVIDER_SYSTEM_PROMPT}


async def test_skip_verification_returns_configured_reward():
    agent, calls = _agent(
        lambda m: "hi", lambda m: "hello", max_turns=2, skip_verification=True, skip_verification_reward=0.0
    )
    result = await agent.run(_fastapi_request(), _request())
    assert result.model_dump()["verification_skipped"] is True and result.reward == 0.0


def test_reminder_drift_is_refused(monkeypatch):
    monkeypatch.setattr("responses_api_agents.vera_mh_agent.app.REMINDER_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="drifted"):
        _agent(lambda m: "hi", lambda m: "hello")
