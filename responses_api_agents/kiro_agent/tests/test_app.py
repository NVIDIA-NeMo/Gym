# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.kiro_agent.app import (
    KiroAgent,
    KiroAgentConfig,
    _extract_instruction,
    parse_kiro_events,
)


def make_config(tmp_path: Path, **overrides) -> KiroAgentConfig:
    fields = {
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "app.py",
        "name": "kiro_agent",
        "resources_server": ResourcesServerRef(type="resources_servers", name="environment"),
        "command": sys.executable,
        "workspace_root": str(tmp_path),
    }
    fields.update(overrides)
    return KiroAgentConfig(**fields)


def make_agent(tmp_path: Path, **overrides) -> KiroAgent:
    return KiroAgent(config=make_config(tmp_path, **overrides), server_client=MagicMock(spec=ServerClient))


def session_update(update: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {"sessionId": "session-1", "update": update},
    }


def test_extract_instruction() -> None:
    user, system = _extract_instruction(
        [
            NeMoGymEasyInputMessage(role="system", content="Be precise"),
            NeMoGymEasyInputMessage(role="user", content="Solve this"),
        ]
    )
    assert user == "Solve this"
    assert system == "Be precise"


def test_parse_kiro_events_preserves_tool_trajectory() -> None:
    events = [
        session_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "messageId": "message-1",
                "content": {"type": "text", "text": "I will calculate it."},
            }
        ),
        session_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call-1",
                "title": "Run Python",
                "kind": "execute",
                "status": "pending",
                "rawInput": {"command": "python -c 'print(42)'"},
            }
        ),
        session_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call-1",
                "status": "completed",
                "rawOutput": {"stdout": "42\n", "exitCode": 0},
            }
        ),
        session_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "messageId": "message-2",
                "content": {"type": "text", "text": "The answer is 42."},
            }
        ),
        session_update({"sessionUpdate": "usage_update", "used": 120, "size": 1000}),
    ]

    output, usage = parse_kiro_events(events)

    assert [item.type for item in output] == ["message", "function_call", "function_call_output", "message"]
    assert output[0].content[0].text == "I will calculate it."
    assert output[1].name == "execute"
    assert json.loads(output[1].arguments) == {"command": "python -c 'print(42)'"}
    assert json.loads(output[2].output) == {"stdout": "42\n", "exitCode": 0}
    assert output[3].content[0].text == "The answer is 42."
    assert usage == {"input_tokens": 120, "output_tokens": 0}


def test_parse_kiro_events_merges_message_chunks_and_content_output() -> None:
    events = [
        session_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "messageId": "message-1",
                "content": {"type": "text", "text": "hello "},
            }
        ),
        session_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "messageId": "message-1",
                "content": {"type": "text", "text": "world"},
            }
        ),
        session_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call-1",
                "title": "Read file",
                "rawInput": {"path": "README.md"},
            }
        ),
        session_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call-1",
                "status": "completed",
                "content": [{"type": "content", "content": {"type": "text", "text": "contents"}}],
            }
        ),
    ]
    output, _ = parse_kiro_events(events)
    assert output[0].content[0].text == "hello world"
    assert output[1].name == "read_file"
    assert output[2].output == "contents"


def test_workspace_command_and_environment(tmp_path: Path) -> None:
    agent = make_agent(
        tmp_path,
        api_key="secret",
        model="claude-sonnet-4",
        effort="high",
        agent_engine="v3",
        env={"EXTRA": "value"},
    )
    work_dir, kiro_home, agent_name = agent._workspace("Use tools")
    config = json.loads((work_dir / ".kiro" / "agents" / "nemo-gym.json").read_text())
    assert config["prompt"] == "Use tools"
    assert config["allowedTools"] == ["*"]
    assert agent._env(kiro_home)["KIRO_API_KEY"] == "secret"
    assert agent._env(kiro_home)["EXTRA"] == "value"
    assert agent._command(agent_name) == [
        sys.executable,
        "acp",
        "--agent-engine",
        "v3",
        "--agent",
        "nemo-gym",
        "--model",
        "claude-sonnet-4",
        "--effort",
        "high",
        "--trust-all-tools",
    ]


@pytest.mark.asyncio
async def test_permission_request_selects_allow_option(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = make_agent(tmp_path)
    proc = MagicMock()
    payloads = []

    async def capture_send(self, process, payload):
        payloads.append(payload)

    monkeypatch.setattr(KiroAgent, "_send", capture_send)
    await agent._handle_agent_request(
        proc,
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "session/request_permission",
            "params": {
                "options": [
                    {"optionId": "reject", "kind": "reject_once"},
                    {"optionId": "allow", "kind": "allow_once"},
                ]
            },
        },
    )
    assert payloads[0]["result"]["outcome"] == {"outcome": "selected", "optionId": "allow"}


@pytest.mark.asyncio
async def test_responses_combines_system_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = make_agent(tmp_path, system_prompt="Configured prompt")
    captured = {}
    output, _ = parse_kiro_events(
        [
            session_update(
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "done"},
                }
            )
        ]
    )

    async def run_kiro(self, instruction, system_prompt):
        captured.update(instruction=instruction, system_prompt=system_prompt)
        return output, {"input_tokens": 3, "output_tokens": 1}, "kiro-model", "end_turn"

    monkeypatch.setattr(KiroAgent, "_run_kiro", run_kiro)
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            NeMoGymEasyInputMessage(role="system", content="Input prompt"),
            NeMoGymEasyInputMessage(role="user", content="Solve it"),
        ]
    )
    response = await agent.responses(MagicMock(), body)

    assert captured == {"instruction": "Solve it", "system_prompt": "Configured prompt\n\nInput prompt"}
    assert response.model == "kiro-model"
    assert response.output[0].content[0].text == "done"
    assert response.usage.total_tokens == 4


def test_config_yaml_parses() -> None:
    path = Path("responses_api_agents/kiro_agent/configs/kiro_agent.yaml")
    config = yaml.safe_load(path.read_text())
    agent = config["kiro_agent"]["responses_api_agents"]["kiro_agent"]
    assert agent["command"] == "kiro-cli"
    assert agent["model"] == "${oc.env:KIRO_MODEL,null}"
    assert "model_server" not in agent
