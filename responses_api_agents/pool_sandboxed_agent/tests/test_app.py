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
        {"type": "reasoning", "reasoning": "Look at the file first."},
        {"type": "thought", "thought": "Look at the file first."},
        {"type": "toolCall", "name": "read", "args": {"path": "/testbed/a.py"}},
        {"type": "toolCallResult", "result": "print('hi')"},
        {"type": "toolCall", "name": "bash", "args": {"command": "ls"}},
        {"type": "toolCallResult", "err": "boom"},
        {"type": "assistantMessage", "message": "Done."},
    ]
)


def _config() -> PoolSandboxedAgentConfig:
    return PoolSandboxedAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="pool_sandboxed_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        model_server=ModelServerRef(type="responses_api_models", name=""),
        pool_version="1.0.16",
        pool_max_context_window=1234,
        pool_extra_args=["--verbose"],
        sandbox_provider="",
        sandbox_config=dict(),
        sandbox_timeout=0,
        token_id_capture=True,
    )


def test_parse_pool_events_pairs_tool_calls_and_prepends_reasoning() -> None:
    items, metadata = parse_pool_events(EVENTS)

    assert [item.type for item in items] == [
        "function_call",
        "function_call_output",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert items[0].name == "read" and json.loads(items[0].arguments) == {"path": "/testbed/a.py"}
    assert items[1].call_id == items[0].call_id and items[1].output == "print('hi')"
    assert items[3].output == "[error] boom"
    assert items[4].content[0].text == "<think>\nLook at the file first.\n</think>\n\nDone."
    assert metadata == {}


def test_parse_pool_events_collects_errors() -> None:
    items, metadata = parse_pool_events('{"error": "approval required"}\n')
    assert items == []
    assert metadata == {"errors": ["approval required"]}


def test_build_command_installs_and_runs_pool_outside_the_workdir() -> None:
    agent = PoolSandboxedAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
    command = agent._build_command("/tmp/nemo-gym-pool-x", "http://gym:8000/ng-rollout/r1/v1")

    assert "POOL_INSTALL_ACCEPT_EULA=1 POOL_INSTALL_DIR=/tmp/nemo-gym-pool-x/bin" in command
    assert 'sh "$installer" 1.0.16' in command
    assert "POOLSIDE_STANDALONE_BASE_URL=http://gym:8000/ng-rollout/r1/v1" in command
    assert "POOLSIDE_STANDALONE_MODEL=dummy_model" in command
    assert "POOLSIDE_STANDALONE_CONTEXT_LENGTH=1234" in command
    assert "XDG_STATE_HOME=/tmp/nemo-gym-pool-x/state" in command
    assert "pool exec -o json --sandbox disabled --unsafe-auto-allow" in command
    assert "-f /tmp/nemo-gym-pool-x/prompt.txt --verbose > /tmp/nemo-gym-pool-x/events.jsonl" in command
    assert 'echo "pool run finished rc=$?"' in command
