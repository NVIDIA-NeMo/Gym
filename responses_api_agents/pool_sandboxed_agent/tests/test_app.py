# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from unittest.mock import MagicMock

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.pool_sandboxed_agent.app import (
    PoolSandboxedAgent,
    PoolSandboxedAgentConfig,
    parse_pool_events,
)


EVENTS = "\n".join(
    json.dumps(event)
    for event in [
        {"type": "assistantMessage", "message": "\n\n"},
        {"type": "reasoning", "reasoning": "Look at the file first."},
        {"type": "thought", "thought": "Look at the file first."},
        {"type": "toolCall", "name": "read", "args": {"path": "/testbed/a.py"}},
        {"type": "toolCallResult", "result": "print('hi')"},
        {"type": "toolCall", "name": "bash", "args": {"command": "ls"}},
        {"type": "toolCallResult", "err": "boom"},
        {"type": "assistantMessage", "message": "Done."},
    ]
)


def _agent() -> PoolSandboxedAgent:
    config = PoolSandboxedAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="pool_sandboxed_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        model_server=ModelServerRef(type="responses_api_models", name=""),
        pool_version="v1.0.16",
        pool_max_context_window=1000,
        pool_extra_args=["--verbose"],
        sandbox_provider="",
        sandbox_config=dict(),
        sandbox_timeout=0,
        token_id_capture=True,
    )
    return PoolSandboxedAgent(config=config, server_client=MagicMock(spec=ServerClient))


def test_parse_pool_events_emits_reasoning_items_in_order_and_pairs_tool_calls() -> None:
    items, metadata = parse_pool_events(EVENTS)

    assert [item.type for item in items] == [
        "reasoning",
        "function_call",
        "function_call_output",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert items[0].summary[0].text == "Look at the file first."
    assert items[1].name == "read" and json.loads(items[1].arguments) == {"path": "/testbed/a.py"}
    assert items[2].call_id == items[1].call_id and items[2].output == "print('hi')"
    assert items[4].output == "[error] boom"
    assert items[5].content[0].text == "Done."
    assert metadata == {}


def test_parse_pool_events_collects_errors() -> None:
    items, metadata = parse_pool_events('{"error": "approval required"}\n')
    assert items == []
    assert metadata == {"errors": ["approval required"]}


def test_build_command_installs_and_runs_pool_under_its_own_home() -> None:
    home = "/tmp/pool-home"
    command = _agent()._build_command(home)

    assert f"POOL_INSTALL_ACCEPT_EULA=1 POOL_INSTALL_DIR={home}/bin" in command
    assert 'sh "$installer" v1.0.16' in command
    assert f"XDG_STATE_HOME={home}/state" in command
    assert "pool exec -o json --sandbox disabled --unsafe-auto-allow" in command
    assert f"--agent-config-file {home}/agent_config.json" in command
    assert f"-f {home}/prompt.txt --verbose" in command
    assert f"> {home}/events.jsonl" in command
    assert 'echo "pool run finished rc=$?"' in command


def test_pool_agent_config_is_non_streaming_and_scales_compaction() -> None:
    agent = _agent()
    agent.config.pool_agent_config = {"model": {"max_completion_retries": 5}}

    config = agent._pool_agent_config("http://gym:8000/ng-rollout/r1/v1")

    assert config["model"]["provider"]["openai"] == {
        "base_url": "http://gym:8000/ng-rollout/r1/v1",
        "api_key": "dummy_key",  # pragma: allowlist secret
        "model_id": "dummy_model",
        "use_streaming": False,
    }
    assert config["model"]["max_completion_retries"] == 5
    assert config["memory"]["compact"]["TriggerCompressionTokenCount"] == 800
    assert config["memory"]["compact"]["MaxSummarizeTokenCount"] == 700
    assert "exit" in config["enabled_tools"]
