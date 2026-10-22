# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_fabric import Fabric
from nemo_fabric.types import RunResult
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


@pytest.mark.parametrize("adapter_id", ["nvidia.fabric.mini-swe-agent", "nemo-gym.mini-swe-agent"])
@pytest.mark.parametrize("sandbox_bundle", [False, True])
@pytest.mark.parametrize("authored_path", [None, "/authored/source:/authored/dependencies"])
def test_mini_child_site_preserves_paths_and_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adapter_id: str, sandbox_bundle: bool, authored_path: str | None
) -> None:
    bundle = tmp_path / "relocated-runtime"
    site = bundle / "mini-mcp-site"
    site.mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "other-prefix") if sandbox_bundle else str(bundle))
    monkeypatch.setattr(sys, "executable", "/launcher/wrapper.sh")
    monkeypatch.delenv("NGSWE_AGENT_DEPS_DIR", raising=False)
    if sandbox_bundle:
        monkeypatch.setenv("NGSWE_AGENT_DEPS_DIR", str(bundle))
    monkeypatch.setenv("PYTHONPATH", "/inherited/source:/inherited/dependencies")
    env = {"KEEP": "value"}
    if authored_path is not None:
        env["PYTHONPATH"] = authored_path
    agent = _agent(adapter_id=adapter_id, fabric_config={"environment": {"env": env}})
    before = dict(os.environ)
    config = agent._fabric_config(
        model_base_url="http://model/v1", workspace=tmp_path, system_prompt=None, mcp_servers={}, skills=[]
    )
    child = config.to_mapping()["environment"]["env"]
    assert child["PYTHONPATH"] == str(site) + os.pathsep + (authored_path or before["PYTHONPATH"])
    assert child["KEEP"] == "value"
    assert agent.config.fabric_config["environment"]["env"] == env
    assert dict(os.environ) == before
    planned = Fabric().plan(config, base_dir=tmp_path).adapter
    assert planned.adapter_id == "nemo-gym.mini-swe-agent"
    assert planned.to_mapping()["runner"]["module"] == "responses_api_agents.nemo_fabric_agent.mini_swe_adapter"
    deep = _agent(adapter_id="nvidia.fabric.deepagents", fabric_config={"environment": {"env": env}})
    deep_config = deep._fabric_config(
        model_base_url="http://model/v1", workspace=tmp_path, system_prompt=None, mcp_servers={}, skills=[]
    ).to_mapping()
    assert deep_config["environment"]["env"].get("PYTHONPATH") == authored_path


@pytest.mark.parametrize("native", [False, True])
def test_installer_uses_separate_mini_target(tmp_path: Path, native: bool) -> None:
    helper = tmp_path / "portable.sh"
    helper.write_text(
        "install_portable_python() { :; }\n"
        "install_nemo_gym_deps() { :; }\n"
        'install_python_packages() { printf "%s\\n" "$@" >> "$INSTALL_LOG"; printf "\\n" >> "$INSTALL_LOG"; }\n'
        f"portable_python_can_run() {{ {'true' if native else 'false'}; }}\n"
    )
    executable = "python3" if native else "uv"
    binary = tmp_path / "bin" / executable
    binary.parent.mkdir()
    binary.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$INSTALL_LOG"\nprintf "\\n" >> "$INSTALL_LOG"\n')
    binary.chmod(0o755)
    log = tmp_path / "install.jsonl"
    script = Path(__file__).parents[1] / "scripts/nemo_fabric_agent_deps.sh"
    subprocess.run(
        ["bash", str(script)],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PORTABLE_PYTHON_SH": str(helper),
            "DEPS_DIR": str(tmp_path),
            "NEMO_GYM_ROOT": str(script.parents[3]),
            "INSTALL_LOG": str(log),
            "PATH": str(binary.parent) + os.pathsep + os.environ["PATH"],
            "ARCH": "x86_64-unknown-linux-gnu",
        },
    )
    commands = [command.splitlines() for command in log.read_text().split("\n\n") if command]
    # A second resolution without Gym would upgrade Anthropic beyond Gym's cap.
    assert any(command[0] == str(script.parents[3]) and "-r" in command for command in commands)
    target = next(command for command in commands if "--target" in command)
    assert target[target.index("--target") + 1] == str(tmp_path / "mini-mcp-site")
    assert {"mcp==2.3.0", "mcp-types==2.3.0"} <= set(target)
    assert "--prefix" not in target
    assert "--no-deps" in target
    if not native:
        assert target[target.index("--python-platform") + 1] == "x86_64-unknown-linux-gnu"
        assert target[target.index("--python-version") + 1] == "3.13.14"


@pytest.mark.parametrize(
    ("preset_name", "adapter_package"),
    [
        ("nemo_fabric_claude", "claude"),
        ("nemo_fabric_codex", "codex"),
        ("nemo_fabric_deepagents", "deepagents"),
        ("nemo_fabric_hermes", "hermes"),
        ("nemo_fabric_mini_swe_agent", "mini-swe-agent"),
    ],
)
def test_harness_presets_pass_fabric_planning(preset_name: str, adapter_package: str, tmp_path: Path) -> None:
    configs_dir = Path(__file__).parents[1] / "configs"
    base = OmegaConf.load(configs_dir / "nemo_fabric_agent.yaml").nemo_fabric_agent
    preset = OmegaConf.load(configs_dir / f"{preset_name}.yaml")[preset_name]
    preset.pop("_inherit_from")
    OmegaConf.set_struct(base, True)
    merged = OmegaConf.to_container(OmegaConf.merge(base, preset), resolve=True)["responses_api_agents"][
        "nemo_fabric_agent"
    ]
    package = distribution(f"nemo-fabric-adapters-{adapter_package}")
    descriptor = next(path for path in package.files if str(path).endswith(".fabric-adapter.json"))
    fabric_config = merged.get("fabric_config", {})
    fabric_config["discovery"] = {"local_paths": [str(package.locate_file(descriptor))]}
    agent = _agent(
        adapter_id=merged["adapter_id"],
        model_provider=merged["model_provider"],
        harness_settings=merged.get("harness_settings", {}),
        fabric_config=fabric_config,
    )

    config = agent._fabric_config(
        model_base_url="http://model/v1",
        workspace=tmp_path,
        system_prompt=None,
        mcp_servers={},
        skills=[],
    )

    planned = Fabric().plan(config, base_dir=tmp_path).adapter
    assert planned.adapter_id == merged["adapter_id"]
    if adapter_package == "mini-swe-agent":
        assert planned.to_mapping()["runner"]["module"] == "responses_api_agents.nemo_fabric_agent.mini_swe_adapter"
        assert config.to_mapping()["models"]["default"]["base_url"] == "http://model/v1"


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
    failed.output["messages"] = [{"role": "assistant", "content": "partial model response"}]
    fabric = MagicMock()
    fabric.run = AsyncMock(return_value=failed)
    trace_path = tmp_path / "logs/agent/fabric-result.json"

    with (
        patch("responses_api_agents.nemo_fabric_agent.app.Fabric", return_value=fabric),
        patch("responses_api_agents.nemo_fabric_agent.app._FABRIC_RESULT_PATH", trace_path, create=True),
    ):
        with pytest.raises(RuntimeError, match="adapter exploded"):
            asyncio.run(agent._create_response(NeMoGymResponseCreateParamsNonStreaming(input="question")))
    assert json.loads(trace_path.read_text()) == failed.to_mapping()


@pytest.fixture
def mini_limit_mapping() -> dict:
    """V6's captured failed-result shape, with conversation content replaced by test data."""
    messages = []
    for turn in range(8):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": f"call-{turn}", "function": {"name": "bash", "arguments": '{"command":"true"}'}}
                    ],
                    "extra": {
                        "response": {"thinking_blocks": [{"type": "thinking", "signature": "fixture-signature"}]}
                    },
                },
                {"role": "tool", "tool_call_id": f"call-{turn}", "content": "fixture output"},
            ]
        )
    messages.append(
        {"role": "exit", "content": "LimitsExceeded", "extra": {"exit_status": "LimitsExceeded", "submission": ""}}
    )
    return {
        "agent_name": "fabric",
        "harness": "nemo-gym.mini-swe-agent",
        "adapter_id": "nemo-gym.mini-swe-agent",
        "adapter_kind": "python",
        "runtime_id": "runtime-fixture",
        "invocation_id": "invoke-fixture",
        "request_id": "request-fixture",
        "status": "failed",
        "error": {
            "stage": "invoke",
            "code": "mini_swe_agent_incomplete",
            "message": "mini-SWE-agent stopped before submitting a final output",
            "retryable": False,
        },
        "output": {
            "messages": messages,
            "model": "azure/anthropic/claude-opus-5-5",
            "output": "",
            "usage": {
                "api_calls": 8,
                "cost": None,
                "input_tokens": 31961,
                "output_tokens": 1076,
                "total_tokens": 33037,
            },
        },
        "usage": {"input_tokens": 263609, "output_tokens": 1885, "total_tokens": 265494, "metadata": {}},
    }


@pytest.mark.parametrize("adapter_id", ["nvidia.fabric.mini-swe-agent", "nemo-gym.mini-swe-agent"])
def test_mini_limit_returns_real_partial_response(tmp_path: Path, mini_limit_mapping: dict, adapter_id: str) -> None:
    agent = _agent(adapter_id=adapter_id, cwd=str(tmp_path))
    agent.server_client.global_config_dict = {
        "policy_model": {"responses_api_models": {"policy_model": {"host": "model", "port": 9002}}}
    }
    agent.server_client._build_server_base_url.return_value = "http://model/v1"
    native = RunResult.from_mapping(mini_limit_mapping)
    fabric = MagicMock(run=AsyncMock(return_value=native))
    trace = tmp_path / "fabric-result.json"
    with (
        patch("responses_api_agents.nemo_fabric_agent.app.Fabric", return_value=fabric),
        patch("responses_api_agents.nemo_fabric_agent.app._FABRIC_RESULT_PATH", trace),
    ):
        response, result = asyncio.run(
            agent._create_response(NeMoGymResponseCreateParamsNonStreaming(input="question"))
        )
    assert response.status == "incomplete"
    assert response.metadata["stop_reason"] == "turn_limit"
    assert response.metadata["fabric_status"] == "failed"
    assert json.loads(response.metadata["fabric_result"]) == result == native.to_mapping()
    assert json.loads(trace.read_text()) == result
    assert response.usage.input_tokens == 263609
    assert response.usage.output_tokens == 1885
    assert response.usage.total_tokens == 265494
    assert [item.type for item in response.output].count("function_call") == 8
    assert [item.type for item in response.output].count("function_call_output") == 8
    assert response.output[-1].type == "function_call_output"
    assert not any(item.type == "message" for item in response.output)
    assert result["output"]["messages"] == mini_limit_mapping["output"]["messages"]


@pytest.mark.parametrize(
    "mismatch", ["adapter", "api_error", "unknown_exit", "last_role", "unknown_status", "empty_messages"]
)
def test_mini_limit_still_raises_other_failures(tmp_path: Path, mini_limit_mapping: dict, mismatch: str) -> None:
    adapter = "nemo-gym.mini-swe-agent"
    if mismatch == "adapter":
        adapter = "nvidia.fabric.deepagents"
    elif mismatch == "api_error":
        mini_limit_mapping["error"]["code"] = "BadRequestError"
    elif mismatch == "unknown_exit":
        mini_limit_mapping["output"]["messages"][-1]["extra"]["exit_status"] = "UnknownExit"
    elif mismatch == "last_role":
        mini_limit_mapping["output"]["messages"][-1]["role"] = "assistant"
    elif mismatch == "unknown_status":
        mini_limit_mapping["status"] = "unknown"
    else:
        mini_limit_mapping["output"]["messages"] = []
    agent = _agent(adapter_id=adapter, cwd=str(tmp_path))
    agent.server_client.global_config_dict = {
        "policy_model": {"responses_api_models": {"policy_model": {"host": "model", "port": 9002}}}
    }
    agent.server_client._build_server_base_url.return_value = "http://model/v1"
    native = RunResult.from_mapping(mini_limit_mapping)
    fabric = MagicMock(run=AsyncMock(return_value=native))
    trace = tmp_path / "fabric-result.json"
    with (
        patch("responses_api_agents.nemo_fabric_agent.app.Fabric", return_value=fabric),
        patch("responses_api_agents.nemo_fabric_agent.app._FABRIC_RESULT_PATH", trace),
    ):
        with pytest.raises(RuntimeError, match="NeMo Fabric invocation failed"):
            asyncio.run(agent._create_response(NeMoGymResponseCreateParamsNonStreaming(input="question")))
    assert json.loads(trace.read_text()) == native.to_mapping()


def test_mini_limit_reaches_resource_verification(tmp_path: Path, mini_limit_mapping: dict) -> None:
    agent = _agent(adapter_id="nemo-gym.mini-swe-agent", cwd=str(tmp_path))
    agent.server_client.global_config_dict = {
        "policy_model": {"responses_api_models": {"policy_model": {"host": "model", "port": 9002}}}
    }
    agent.server_client._build_server_base_url.return_value = "http://model/v1"
    native = RunResult.from_mapping(mini_limit_mapping)
    trace = tmp_path / "fabric-result.json"

    async def post(server_name, url_path, json=None, **kwargs):
        if url_path == "/seed_session":
            return _FakeHttpResponse({})
        assert url_path == "/verify"
        assert trace.is_file()
        assert json["response"]["status"] == "incomplete"
        assert json["response"]["output"][-1]["type"] == "function_call_output"
        return _FakeHttpResponse(json | {"reward": 0.0})

    agent.server_client.post = AsyncMock(side_effect=post)
    fabric = MagicMock(run=AsyncMock(return_value=native))
    request = MagicMock(cookies={}, path_params={})
    with (
        patch("responses_api_agents.nemo_fabric_agent.app.Fabric", return_value=fabric),
        patch("responses_api_agents.nemo_fabric_agent.app._FABRIC_RESULT_PATH", trace),
    ):
        result = asyncio.run(
            agent.run(request, NeMoFabricAgentRunRequest(responses_create_params={"input": "question"}))
        )
    assert agent.server_client.post.await_count == 2
    assert result.finished_naturally is False
    assert result.turns_used == 8
    assert result.fabric_result == native.to_mapping()


@pytest.mark.parametrize("failing_method", ["mkdir", "write_text"])
@pytest.mark.parametrize("api_error", [False, True])
def test_trace_oserror_preserves_native_outcome(
    tmp_path: Path, mini_limit_mapping: dict, caplog: pytest.LogCaptureFixture, failing_method: str, api_error: bool
) -> None:
    if api_error:
        mini_limit_mapping["error"].update(code="BadRequestError", message="fixture API error")
    agent = _agent(adapter_id="nemo-gym.mini-swe-agent", cwd=str(tmp_path))
    agent.server_client.global_config_dict = {
        "policy_model": {"responses_api_models": {"policy_model": {"host": "model", "port": 9002}}}
    }
    agent.server_client._build_server_base_url.return_value = "http://model/v1"
    native = RunResult.from_mapping(mini_limit_mapping)
    fabric = MagicMock(run=AsyncMock(return_value=native))
    trace = tmp_path / "logs/agent/fabric-result.json"
    with (
        patch("responses_api_agents.nemo_fabric_agent.app.Fabric", return_value=fabric),
        patch("responses_api_agents.nemo_fabric_agent.app._FABRIC_RESULT_PATH", trace),
        patch.object(Path, failing_method, side_effect=PermissionError("fixture read-only filesystem")),
        caplog.at_level("WARNING", logger="responses_api_agents.nemo_fabric_agent.app"),
    ):
        body = NeMoGymResponseCreateParamsNonStreaming(input="question")
        if api_error:
            with pytest.raises(RuntimeError, match="fixture API error") as raised:
                asyncio.run(agent._create_response(body))
            assert "BadRequestError" in str(raised.value)
            assert "Result saved" not in str(raised.value)
        else:
            response, result = asyncio.run(agent._create_response(body))
            assert response.status == "incomplete"
            assert response.metadata["stop_reason"] == "turn_limit"
            assert json.loads(response.metadata["fabric_result"]) == result == native.to_mapping()
            assert response.output[-1].type == "function_call_output"
    assert not trace.exists()
    assert "Could not save Fabric result" in caplog.text


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


def test_fabric_output_items_preserve_mini_swe_reasoning_and_submission() -> None:
    output = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Verify the result before submitting.",
                "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n42"},
            {"role": "exit", "content": "42"},
        ],
    }
    items = [item.model_dump() for item in _fabric_output_items(output, "42")]
    assert [item["type"] for item in items] == ["reasoning", "function_call", "function_call_output", "message"]
    assert items[0]["summary"][0]["text"] == "Verify the result before submitting."
    assert items[1]["call_id"] == items[2]["call_id"] == "c1"
    assert items[2]["output"] == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n42"
    assert items[3]["content"][0]["text"] == "42"


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
