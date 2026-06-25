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

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.opencode_agent.app import (
    OpenCodeAgent,
    OpenCodeAgentConfig,
    _extract_instruction,
    parse_opencode_session,
)


def _config(**kwargs) -> OpenCodeAgentConfig:
    return OpenCodeAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        **kwargs,
    )


def _make_agent(**kwargs) -> OpenCodeAgent:
    with patch("responses_api_agents.opencode_agent.app.OpenCodeAgent.model_post_init"):
        agent = OpenCodeAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))
    agent.sem = asyncio.Semaphore(agent.config.concurrency)
    return agent


def _session_db(tmp_path, messages) -> Path:
    """Build a minimal opencode sqlite db. messages is a list of (role, [part_dicts])."""
    import sqlite3

    db = tmp_path / "opencode.db"
    con = sqlite3.connect(db)
    con.execute("create table message (id text, data text, time_created integer)")
    con.execute("create table part (id text, message_id text, data text, time_created integer)")
    t = 0
    for mi, (role, parts) in enumerate(messages):
        mid = f"m{mi}"
        con.execute("insert into message values (?,?,?)", (mid, json.dumps({"role": role}), mi))
        for p in parts:
            con.execute("insert into part values (?,?,?,?)", (f"p{t}", mid, json.dumps(p), t))
            t += 1
    con.commit()
    con.close()
    return db


class TestSanity:
    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.concurrency == 8
        assert cfg.command == "opencode"
        assert cfg.thinking is True
        assert cfg.command_parts == ["opencode"]

    def test_semaphore_initialized(self) -> None:
        agent = _make_agent(concurrency=4)
        assert agent.sem._value == 4


class TestExtractInstruction:
    def test_user_only(self) -> None:
        user, system = _extract_instruction([NeMoGymEasyInputMessage(role="user", content="hello")])
        assert user == "hello"
        assert system is None

    def test_system_plus_user(self) -> None:
        items = [
            NeMoGymEasyInputMessage(role="system", content="be concise"),
            NeMoGymEasyInputMessage(role="user", content="hi"),
        ]
        user, system = _extract_instruction(items)
        assert user == "hi"
        assert system == "be concise"

    def test_empty(self) -> None:
        user, system = _extract_instruction([])
        assert user == ""
        assert system is None


class TestParseOpencodeSession:
    def test_missing_db(self, tmp_path) -> None:
        items, usage = parse_opencode_session(tmp_path / "nope.db")
        assert items == []
        assert usage == {"input_tokens": 0, "output_tokens": 0}

    def test_assistant_text(self, tmp_path) -> None:
        db = _session_db(tmp_path, [("assistant", [{"type": "text", "text": "the answer is 4"}])])
        items, _ = parse_opencode_session(db)
        assert len(items) == 1
        assert isinstance(items[0], NeMoGymResponseOutputMessage)
        assert items[0].content[0].text == "the answer is 4"

    def test_user_parts_ignored(self, tmp_path) -> None:
        db = _session_db(tmp_path, [("user", [{"type": "text", "text": "hi"}])])
        items, _ = parse_opencode_session(db)
        assert items == []

    def test_tool_call_and_output(self, tmp_path) -> None:
        db = _session_db(
            tmp_path,
            [
                (
                    "assistant",
                    [
                        {
                            "type": "tool",
                            "callID": "c1",
                            "tool": "bash",
                            "state": {"input": {"command": "echo 6"}, "output": "6\n"},
                        },
                        {"type": "text", "text": "answer is 6"},
                    ],
                )
            ],
        )
        items, _ = parse_opencode_session(db)
        assert isinstance(items[0], NeMoGymResponseFunctionToolCall)
        assert items[0].name == "bash"
        assert json.loads(items[0].arguments)["command"] == "echo 6"
        assert isinstance(items[1], NeMoGymFunctionCallOutput)
        assert items[1].call_id == "c1"
        assert "6" in items[1].output
        assert isinstance(items[2], NeMoGymResponseOutputMessage)

    def test_step_finish_usage(self, tmp_path) -> None:
        db = _session_db(
            tmp_path,
            [("assistant", [{"type": "step-finish", "tokens": {"input": 100, "output": 20, "cache": {"read": 5}}}])],
        )
        _, usage = parse_opencode_session(db)
        assert usage["input_tokens"] == 105
        assert usage["output_tokens"] == 20


class TestDeepMerge:
    def test_nested_merge(self) -> None:
        base = {"a": {"b": 1, "c": 2}}
        OpenCodeAgent._deep_merge(base, {"a": {"c": 3, "d": 4}})
        assert base == {"a": {"b": 1, "c": 3, "d": 4}}


class TestEnv:
    def test_env_passthrough(self) -> None:
        agent = _make_agent(openai_api_key="k", openai_base_url="https://x/v1", env={"FOO": "bar", "EMPTY": ""})
        # model_server unset -> request is unused, so a dummy request is fine.
        env = agent._env(MagicMock(), "/tmp/data")
        assert env["OPENAI_API_KEY"] == "k"
        assert env["OPENAI_BASE_URL"] == "https://x/v1"
        assert env["XDG_DATA_HOME"] == "/tmp/data"
        assert env["FOO"] == "bar"
        assert "EMPTY" not in env


class TestRunScoping:
    """When model_server is set, opencode's model calls are routed to the run-scoped model-server url
    so the model server buffers their token ids."""

    @staticmethod
    def _scoped_agent(**kwargs) -> OpenCodeAgent:
        return _make_agent(
            model="nvinf/some/model",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            opencode_config={
                "provider": {
                    "nvinf": {
                        "npm": "@ai-sdk/openai-compatible",
                        "options": {"baseURL": "http://placeholder/v1", "apiKey": "x"},
                        "models": {"some/model": {}},
                    }
                }
            },
            **kwargs,
        )

    @staticmethod
    def _patch_base_url():
        # harness_base_url is the model-server (run-scoped when run() supplied a token) base url.
        # patch on the class (Pydantic blocks per-instance method assignment).
        return patch.object(OpenCodeAgent, "harness_base_url", return_value="http://model:9000/runs/abc123")

    def test_run_base_url_none_without_model_server(self) -> None:
        agent = _make_agent()
        assert agent._run_base_url(MagicMock()) is None

    def test_run_base_url_scoped(self) -> None:
        agent = self._scoped_agent()
        with self._patch_base_url():
            assert agent._run_base_url(MagicMock()) == "http://model:9000/runs/abc123/v1"

    def test_model_arg_plain(self) -> None:
        agent = _make_agent(model="nvinf/some/model")
        assert agent._model_arg() == "nvinf/some/model"

    def test_model_arg_scoped_keeps_provider(self) -> None:
        agent = self._scoped_agent()
        assert agent._model_arg() == "nvinf/policy_model"

    def test_env_scoped_overrides_base_url(self) -> None:
        agent = self._scoped_agent(openai_api_key="k", openai_base_url="https://x/v1")
        with self._patch_base_url():
            env = agent._env(MagicMock(), "/tmp/data")
        assert env["OPENAI_BASE_URL"] == "http://model:9000/runs/abc123/v1"
        assert env["OPENAI_API_KEY"] == "gym"

    def test_write_opencode_config_scoped(self, tmp_path) -> None:
        agent = self._scoped_agent()
        with self._patch_base_url():
            agent._write_opencode_config(MagicMock(), tmp_path)
        written = json.loads((tmp_path / "opencode.json").read_text())
        prov = written["provider"]["nvinf"]
        assert prov["options"]["baseURL"] == "http://model:9000/runs/abc123/v1"
        assert prov["options"]["apiKey"] == "gym"
        assert prov["models"] == {"policy_model": {}}

    def test_write_opencode_config_plain_unscoped(self, tmp_path) -> None:
        agent = _make_agent(opencode_config={"provider": {"nvinf": {"options": {"baseURL": "http://placeholder/v1"}}}})
        agent._write_opencode_config(MagicMock(), tmp_path)
        written = json.loads((tmp_path / "opencode.json").read_text())
        assert written["provider"]["nvinf"]["options"]["baseURL"] == "http://placeholder/v1"


class TestConfigYaml:
    def test_module_parses(self) -> None:
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        compile(app_path.read_text(), str(app_path), "exec")

    def test_config_yaml_parses(self) -> None:
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "opencode_agent.yaml"
        data = yaml.safe_load(cfg_path.read_text())
        assert "opencode_agent" in data
        inner = data["opencode_agent"]["responses_api_agents"]["opencode_agent"]
        assert inner["entrypoint"] == "app.py"
        assert inner["concurrency"] == 8
        assert inner["command"] == "opencode"
