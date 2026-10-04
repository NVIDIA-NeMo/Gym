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
from pathlib import Path

import yaml

from responses_api_agents.mimoagent.app import PROFILES_DIR, _output_items


def test_output_items_maps_chat_messages() -> None:
    items = _output_items(
        [
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": "looking",
                "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "a.py"},
        ]
    )
    assert [i["type"] for i in items] == ["message", "function_call", "function_call_output"]
    assert items[1]["call_id"] == items[2]["call_id"] == "c1"
    assert items[2]["output"] == "a.py"


def test_every_mimoagent_harness_has_a_profile() -> None:
    types = {
        (yaml.safe_load(p.read_text()).get("agent") or {}).get("type", "default") for p in PROFILES_DIR.glob("*.yaml")
    }
    expected = {
        "default",
        "bashonly-agent",
        "cc-agent",
        "codex-agent",
        "mimocode-agent",
        "claude-code",
        "codex",
        "mimocode",
        "pi",
        "grok",
        "kimi-code",
        "kimi-cli",
        "kilocode",
        "openclaw",
        "opencode",
        "omp",
        "hermes",
        "dsh",
        "mini-swe-agent",
    }
    assert types == expected
    assert {Path(p).stem for p in PROFILES_DIR.glob("*.yaml")} == expected


def test_model_url_carries_the_rollout_capture_prefix() -> None:
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from nemo_gym.server_utils import ServerClient
    from responses_api_agents.mimoagent.app import MimoAgent, MimoAgentConfig

    config = MimoAgentConfig(
        host="",
        port=0,
        entrypoint="",
        name="a",
        profile="cc-agent",
        resources_server={"type": "resources_servers", "name": "r"},
        model_server={"type": "responses_api_models", "name": "p"},
    )
    agent = MimoAgent(config=config, server_client=MagicMock(spec=ServerClient))
    seen = []
    object.__setattr__(agent, "_run_agent", lambda body, prefix: (seen.append(prefix), ("Idle", "done", []))[1])
    body = SimpleNamespace(model="m")
    rollout = SimpleNamespace(
        path_params={"rollout_id": "r1"}, url=SimpleNamespace(path="/ng-rollout/r1/v1/responses")
    )
    runner = SimpleNamespace(path_params={}, url=SimpleNamespace(path=""))
    asyncio.run(agent.responses(rollout, body))
    asyncio.run(agent.responses(runner, body))
    assert seen[0].endswith("r1")
    assert seen[1] == ""
