# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_fabric import Fabric
from omegaconf import OmegaConf

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import (
    OBSERVABILITY_ENABLED_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    SKILLS_REF_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nemo_fabric_agent.app import (
    NeMoFabricAgent,
    NeMoFabricAgentConfig,
    NeMoFabricAgentRunRequest,
    _content_text,
    _extract_request_input,
    _fabric_output_items,
    _mapping,
    _normalized_usage,
    _skill_paths,
    _turns_used,
    _usage_value,
)


def _config(**kwargs) -> NeMoFabricAgentConfig:
    kwargs.setdefault("resources_server", ResourcesServerRef(type="resources_servers", name="resources"))
    kwargs.setdefault("model_server", ModelServerRef(type="responses_api_models", name="policy_model"))
    kwargs.setdefault("adapter_id", "nvidia.fabric.hermes")
    kwargs.setdefault("model_provider", "openai")
    kwargs.setdefault("model", "gym-policy-model")
    return NeMoFabricAgentConfig(host="0.0.0.0", port=8080, entrypoint="", name="fabric", **kwargs)


def _agent(**kwargs) -> NeMoFabricAgent:
    return NeMoFabricAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))


class _FakeHttpResponse:
    ok = True

    def __init__(self, payload: dict, cookies: dict | None = None) -> None:
        self.payload = payload
        self.cookies = cookies or {}

    async def read(self) -> bytes:
        import orjson

        return orjson.dumps(self.payload)


class _FakeOutput(dict):
    def to_mapping(self) -> dict:
        return dict(self)


class _FakeResult:
    status = "succeeded"
    error = None

    def __init__(self) -> None:
        self.output = _FakeOutput(
            response="fabric answer",
            model="gym-policy-model",
            api_calls=2,
            usage={"input_tokens": 3, "output_tokens": 4, "cached_input_tokens": 1},
        )

    def to_mapping(self) -> dict:
        return {
            "status": self.status,
            "output": dict(self.output),
            "request_id": "request-1",
        }


def test_extract_request_input_keeps_multiturn_messages_structured() -> None:
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {
            "input": [
                {"role": "system", "content": "system", "type": "message"},
                {"role": "user", "content": "old", "type": "message"},
                {"role": "developer", "content": "developer", "type": "message"},
                {"role": "user", "content": "latest", "type": "message"},
            ]
        }
    )

    request_input, instructions = _extract_request_input(body.input)
    assert [message["role"] for message in request_input] == ["user", "user"]
    assert instructions == "system\n\ndeveloper"


def test_extract_request_input_preserves_function_call_replay() -> None:
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {
            "input": [
                {"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call-1", "output": "result"},
            ]
        }
    )

    request_input, instructions = _extract_request_input(body.input)
    assert [item["type"] for item in request_input] == ["function_call", "function_call_output"]
    assert instructions is None


def test_content_and_response_normalizers_handle_generic_values() -> None:
    part = MagicMock(text="object")

    assert _content_text(None) == ""
    assert _content_text([{"text": "dict"}, part, {"ignored": True}]) == "dictobject"
    assert _extract_request_input("plain prompt") == ("plain prompt", None)
    assert _mapping(_FakeOutput(answer=1)) == {"answer": 1}
    assert _mapping(None) == {}
    assert _usage_value({"first": True, "second": -1, "third": 7}, "first", "second", "third") == 7
    assert _usage_value({}, "missing") == 0


def test_normalized_usage_supports_canonical_and_adapter_shapes() -> None:
    canonical = _normalized_usage(
        {
            "usage": {
                "input_tokens": 2,
                "output_tokens": 3,
                "metadata": {"reasoning_tokens": 1},
                "extensions": {"cached_input_tokens": 4},
            }
        },
        {},
    )
    adapter = _normalized_usage(
        {},
        {
            "usage": {
                "total": {
                    "inputTokens": 11,
                    "outputTokens": 7,
                    "reasoningOutputTokens": 5,
                    "cachedInputTokens": 4,
                    "totalTokens": 18,
                }
            }
        },
    )

    assert canonical == {
        "input_tokens": 2,
        "output_tokens": 3,
        "reasoning_tokens": 1,
        "cached_input_tokens": 4,
        "metadata": {"reasoning_tokens": 1},
        "extensions": {"cached_input_tokens": 4},
    }
    assert adapter["reasoningOutputTokens"] == 5


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ({"num_turns": 4}, 4),
        ({"api_calls": 5}, 5),
        ({"usage": {"api_calls": 6}}, 6),
        ({"turns_used": 3, "api_calls": 8}, 3),
        ({"num_turns": True}, 1),
        ({}, 1),
    ],
)
def test_turns_used_supports_adapter_shapes(output: dict, expected: int) -> None:
    assert _turns_used(output) == expected


def test_skill_paths_expands_gym_variant(tmp_path: Path) -> None:
    for name in ("alpha", "beta"):
        skill = tmp_path / name
        skill.mkdir()
        (skill / "SKILL.md").write_text(f"---\nname: {name}\n---\n")
    (tmp_path / "ignored").mkdir()

    assert _skill_paths(str(tmp_path)) == [str(tmp_path / "alpha"), str(tmp_path / "beta")]


def test_skill_paths_accepts_one_skill_and_rejects_invalid_roots(tmp_path: Path) -> None:
    assert _skill_paths(None) == []
    with pytest.raises(ValueError, match="not a directory"):
        _skill_paths(str(tmp_path / "missing"))

    (tmp_path / "SKILL.md").write_text("---\nname: root\n---\n")
    assert _skill_paths(str(tmp_path)) == [str(tmp_path)]
    (tmp_path / "SKILL.md").unlink()
    with pytest.raises(ValueError, match="contains no SKILL.md"):
        _skill_paths(str(tmp_path))


def test_fabric_config_injects_model_workspace_mcp_and_skills(tmp_path: Path) -> None:
    agent = _agent(
        system_prompt="agent instruction",
        harness_settings={"sandbox": "danger-full-access"},
        fabric_config={
            "runtime": {"max_turns": 3},
            "instructions": {"system": {"mode": "append"}, "project": {"enabled": False}},
            "mcp": {
                "servers": {"static": {"transport": "stdio", "url": "x"}},
                "tool_policy": {"default": "allow"},
            },
            "skills": {"validation": "strict"},
        },
    )

    config = agent._fabric_config(
        model_base_url="http://model/ng-rollout/1/v1",
        workspace=tmp_path,
        system_prompt="dataset instruction",
        mcp_servers={"dynamic": {"transport": "streamable-http", "url": "http://resources/mcp"}},
        skills=[str(tmp_path / "skill")],
    ).to_mapping()

    assert config["harness"] == {
        "adapter_id": "nvidia.fabric.hermes",
        "settings": {"sandbox": "danger-full-access"},
    }
    assert config["models"]["default"]["base_url"] == "http://model/ng-rollout/1/v1"
    assert config["environment"]["workspace"] == str(tmp_path)
    assert config["runtime"] == {"timeout_seconds": 600.0, "max_turns": 3}
    assert set(config["mcp"]["servers"]) == {"static", "dynamic"}
    assert config["mcp"]["tool_policy"] == {"default": "allow"}
    assert config["skills"]["paths"] == [str(tmp_path / "skill")]
    assert config["skills"]["validation"] == "strict"
    assert config["instructions"]["system"]["content"] == "dataset instruction"
    assert config["instructions"]["project"] == {"enabled": False}


@pytest.mark.parametrize(
    "preset_name",
    [
        "nemo_fabric_claude",
        "nemo_fabric_codex",
        "nemo_fabric_deepagents",
        "nemo_fabric_hermes",
        "nemo_fabric_mini_swe_agent",
    ],
)
def test_harness_presets_pass_fabric_planning(preset_name: str, tmp_path: Path) -> None:
    configs_dir = Path(__file__).parents[1] / "configs"
    base = OmegaConf.load(configs_dir / "nemo_fabric_agent.yaml").nemo_fabric_agent
    preset = OmegaConf.load(configs_dir / f"{preset_name}.yaml")[preset_name]
    preset.pop("_inherit_from")
    OmegaConf.set_struct(base, True)
    merged = OmegaConf.to_container(OmegaConf.merge(base, preset), resolve=True)["responses_api_agents"][
        "nemo_fabric_agent"
    ]
    agent = _agent(
        adapter_id=merged["adapter_id"],
        model_provider=merged["model_provider"],
        harness_settings=merged.get("harness_settings", {}),
        fabric_config=merged.get("fabric_config", {}),
    )

    config = agent._fabric_config(
        model_base_url="http://model/v1",
        workspace=tmp_path,
        system_prompt=None,
        mcp_servers={},
        skills=[],
    )

    assert Fabric().plan(config, base_dir=tmp_path).adapter.adapter_id == merged["adapter_id"]


def test_rollout_mcp_metadata_is_optional_and_headers_are_optional(caplog) -> None:
    agent = _agent()
    agent.server_client.global_config_dict = {
        "resources": {"resources_servers": {"resources": {"host": "127.0.0.1", "port": 9001}}}
    }
    agent.server_client._build_server_base_url.return_value = "http://127.0.0.1:9001"

    assert agent._rollout_mcp_servers({}) == {}
    assert agent._rollout_mcp_servers({"mcp": {}}) == {
        "resources": {"transport": "streamable-http", "url": "http://127.0.0.1:9001/mcp"}
    }
    assert "has no session headers" in caplog.text


@pytest.mark.parametrize(
    ("answer_key", "completed", "finished_naturally"),
    [("response", True, True), ("output", False, False)],
)
def test_run_preserves_fabric_result_and_verifies(
    answer_key: str, completed: bool, finished_naturally: bool, tmp_path: Path
) -> None:
    agent = _agent(cwd=str(tmp_path))
    agent.server_client.global_config_dict = {
        "resources": {"resources_servers": {"resources": {"host": "127.0.0.1", "port": 9001}}},
        "policy_model": {"responses_api_models": {"policy_model": {"host": "127.0.0.1", "port": 9002}}},
        OBSERVABILITY_ENABLED_KEY_NAME: True,
    }
    agent.server_client._build_server_base_url.side_effect = lambda config: f"http://{config['host']}:{config['port']}"

    async def post(server_name, url_path, json=None, cookies=None, **kwargs):
        if url_path == "/seed_session":
            return _FakeHttpResponse(
                {
                    "mcp": {
                        "server_name": "resources",
                        "url_path": "/mcp",
                        "headers": {"X-NeMo-Gym-Session-Token": "token"},
                    }
                },
                cookies={"session": "seeded"},
            )
        assert url_path == "/verify"
        assert json["response"]["output"][0]["content"][0]["text"] == "fabric answer"
        return _FakeHttpResponse(json | {"reward": 1.0})

    agent.server_client.post = AsyncMock(side_effect=post)
    body = NeMoFabricAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": "question"},
            SKILLS_REF_KEY_NAME: None,
            TASK_INDEX_KEY_NAME: 3,
            ROLLOUT_INDEX_KEY_NAME: 7,
        }
    )
    request = MagicMock()
    request.cookies = {}

    fabric = MagicMock()
    fabric_result = _FakeResult()
    if answer_key == "output":
        fabric_result.output["output"] = fabric_result.output.pop("response")
    fabric_result.output.update(completed=completed, failed=not completed)
    fabric.run = AsyncMock(return_value=fabric_result)
    with patch("responses_api_agents.nemo_fabric_agent.app.Fabric", return_value=fabric):
        result = asyncio.run(agent.run(request, body))

    assert result.reward == 1.0
    assert result.fabric_result["request_id"] == "request-1"
    assert result.turns_used == 2
    assert result.finished_naturally is finished_naturally
    assert result.response.usage.input_tokens == 3
    assert result.response.usage.output_tokens == 4
    called_config = fabric.run.call_args.args[0].to_mapping()
    called_request = fabric.run.call_args.kwargs["request"]
    assert called_request.request_id == "3-7"
    assert called_request.context == {"nemo_gym_rollout_id": "3-7"}
    dynamic = called_config["mcp"]["servers"]["resources"]
    assert dynamic["transport"] == "streamable-http"
    assert dynamic["custom_headers"] == {"X-NeMo-Gym-Session-Token": "token"}


def test_failed_fabric_result_raises(tmp_path: Path) -> None:
    agent = _agent(cwd=str(tmp_path))
    agent.server_client.global_config_dict = {
        "policy_model": {"responses_api_models": {"policy_model": {"host": "127.0.0.1", "port": 9002}}}
    }
    agent.server_client._build_server_base_url.side_effect = lambda config: f"http://{config['host']}:{config['port']}"
    failed = _FakeResult()
    failed.status = "failed"
    failed.error = MagicMock(message="adapter exploded")
    fabric = MagicMock()
    fabric.run = AsyncMock(return_value=failed)

    with patch("responses_api_agents.nemo_fabric_agent.app.Fabric", return_value=fabric):
        with pytest.raises(RuntimeError, match="adapter exploded"):
            asyncio.run(agent._create_response(NeMoGymResponseCreateParamsNonStreaming(input="question")))


def test_responses_forwards_rollout_id_and_invalid_cwd_is_rejected(tmp_path: Path) -> None:
    agent = _agent(cwd=str(tmp_path / "missing"))
    agent.server_client.global_config_dict = {
        "policy_model": {"responses_api_models": {"policy_model": {"host": "127.0.0.1", "port": 9002}}}
    }
    agent.server_client._build_server_base_url.return_value = "http://127.0.0.1:9002"
    with pytest.raises(ValueError, match="configured cwd is not a directory"):
        asyncio.run(agent._create_response(NeMoGymResponseCreateParamsNonStreaming(input="question")))

    expected = MagicMock()
    agent._create_response = AsyncMock(return_value=(expected, {}))
    request = MagicMock()
    request.path_params = {"rollout_id": "rollout-7"}
    result = asyncio.run(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="question")))

    assert result is expected
    assert agent._create_response.await_args.kwargs == {"rollout_id": "rollout-7"}


def test_fabric_output_items_preserve_tool_calls() -> None:
    """A harness tool call must reach the Responses trajectory, not just fabric_result."""
    output = {
        "response": "final",
        "messages": [
            {"role": "human", "content": "q"},
            {"role": "ai", "content": "", "tool_calls": [{"id": "call-1", "name": "task", "args": {"a": 1}}]},
            {"role": "tool", "content": "subagent result"},
            {"role": "ai", "content": "final"},
        ],
    }
    items = [item.model_dump() for item in _fabric_output_items(output, "final")]
    assert [item["type"] for item in items] == ["function_call", "function_call_output", "message"]
    assert items[0]["name"] == "task"
    assert items[0]["arguments"] == '{"a": 1}'
    assert items[0]["call_id"] == items[1]["call_id"] == "call-1"
    assert items[1]["output"] == "subagent result"
    assert items[2]["content"][0]["text"] == "final"


def test_fabric_output_items_single_turn_is_one_message() -> None:
    """No tool calls must still yield exactly one assistant message (no duplication)."""
    output = {"response": "42", "messages": [{"role": "human", "content": "q"}, {"role": "ai", "content": "42"}]}
    items = [item.model_dump() for item in _fabric_output_items(output, "42")]
    assert [item["type"] for item in items] == ["message"]
    assert items[0]["content"][0]["text"] == "42"


def test_fabric_output_items_reads_openai_tool_call_shape() -> None:
    """hermes reports {call_id, function: {name, arguments}} rather than LangChain's {id, name, args}."""
    output = {
        "response": "done",
        "messages": [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"call_id": "c9", "function": {"name": "execute_code", "arguments": '{"code": "1+1"}'}}
                ],
            },
            {"role": "tool", "content": "2"},
            {"role": "assistant", "content": "done"},
        ],
    }
    items = [item.model_dump() for item in _fabric_output_items(output, "done")]
    assert [item["type"] for item in items] == ["function_call", "function_call_output", "message"]
    assert items[0]["name"] == "execute_code"
    assert items[0]["arguments"] == '{"code": "1+1"}'
    assert items[0]["call_id"] == items[1]["call_id"] == "c9"


def test_turns_used_counts_assistant_messages_when_adapter_reports_none() -> None:
    output = {
        "messages": [
            {"role": "human", "content": "q"},
            {"role": "ai", "content": "", "tool_calls": [{"id": "c1", "name": "t", "args": {}}]},
            {"role": "tool", "content": "r"},
            {"role": "ai", "content": "done"},
        ],
        "usage": {"completion_tokens": 10, "prompt_tokens": 20},
    }
    assert _turns_used(output) == 2


def test_turns_used_prefers_adapter_reported_count() -> None:
    output = {
        "api_calls": 4,
        "messages": [{"role": "ai", "content": "x"}],
    }
    assert _turns_used(output) == 4


def test_fabric_output_items_reads_claude_sdk_event_blocks() -> None:
    """Claude SDK content blocks carry no ``type``; they are identified by their keys."""
    output = {
        "response": "done",
        "events": [
            {"type": "SystemMessage", "message": {"content": None}},
            {
                "type": "AssistantMessage",
                "message": {"content": [{"id": "tu1", "name": "Bash", "input": {"command": "echo hi"}}]},
            },
            {
                "type": "UserMessage",
                "message": {"content": [{"tool_use_id": "tu1", "content": "hi", "is_error": False}]},
            },
            {"type": "AssistantMessage", "message": {"content": [{"text": "done"}]}},
        ],
    }
    items = [item.model_dump() for item in _fabric_output_items(output, "done")]
    assert [item["type"] for item in items] == ["function_call", "function_call_output", "message"]
    assert items[0]["name"] == "Bash"
    assert items[0]["arguments"] == '{"command": "echo hi"}'
    assert items[0]["call_id"] == items[1]["call_id"] == "tu1"
    assert items[1]["output"] == "hi"
