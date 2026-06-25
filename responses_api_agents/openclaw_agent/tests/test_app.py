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

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.openclaw_agent.app import (
    OpenClawAgent,
    OpenClawAgentConfig,
    ResourcesServerRef,
    _decode_last_json_dict_suffix,
    _extract_instruction,
    _text_from_openclaw_payloads,
    parse_openclaw_output,
    parse_openclaw_session,
)


def _provider_cfg() -> dict:
    return {
        "models": {
            "providers": {
                "nvinf": {
                    "baseUrl": "http://placeholder/v1",
                    "apiKey": "orig",  # pragma: allowlist secret
                    "models": [{"id": "orig", "name": "orig"}],
                }
            }
        }
    }


def _request(run_token=None) -> MagicMock:
    req = MagicMock()
    req.headers = {"x-nemo-gym-run-token": run_token} if run_token else {}
    return req


def _config(**kwargs) -> OpenClawAgentConfig:
    return OpenClawAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        **kwargs,
    )


def _make_agent(**kwargs) -> OpenClawAgent:
    with patch("responses_api_agents.openclaw_agent.app.OpenClawAgent.model_post_init"):
        agent = OpenClawAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))
    agent.sem = asyncio.Semaphore(agent.config.concurrency)
    return agent


def _envelope(payloads, usage=None, final_text=None) -> str:
    meta = {}
    if usage is not None:
        meta["agentMeta"] = {"usage": usage}
    if final_text is not None:
        meta["finalAssistantVisibleText"] = final_text
    return json.dumps({"payloads": payloads, "meta": meta})


class TestSanity:
    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.concurrency == 32
        assert cfg.command == "openclaw"
        assert cfg.thinking == "off"
        assert cfg.command_parts == ["openclaw"]

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


class TestDecodeJsonSuffix:
    def test_plain_json(self) -> None:
        assert _decode_last_json_dict_suffix('{"a": 1}') == {"a": 1}

    def test_log_lines_before_json(self) -> None:
        raw = 'INFO booting\nWARN slow\n{"a": 1, "b": {"c": 2}}'
        assert _decode_last_json_dict_suffix(raw) == {"a": 1, "b": {"c": 2}}

    def test_empty_returns_none(self) -> None:
        assert _decode_last_json_dict_suffix("   ") is None

    def test_no_json_returns_none(self) -> None:
        assert _decode_last_json_dict_suffix("just logs, no json") is None


class TestTextFromPayloads:
    def test_plain_text(self) -> None:
        env = {"payloads": [{"text": "hello"}, {"text": "world"}]}
        assert _text_from_openclaw_payloads(env) == "hello\n\nworld"

    def test_falls_back_to_final_visible_text(self) -> None:
        env = {"payloads": [], "meta": {"finalAssistantVisibleText": "final"}}
        assert _text_from_openclaw_payloads(env) == "final"

    def test_no_payloads(self) -> None:
        assert _text_from_openclaw_payloads({}) == ""


class TestParseOpenclawOutput:
    def test_empty(self) -> None:
        items, usage = parse_openclaw_output("")
        assert items == []
        assert usage == {"input_tokens": 0, "output_tokens": 0}

    def test_text_message_and_usage(self) -> None:
        raw = _envelope(
            [{"text": "the answer is 4"}],
            usage={"input": 100, "output": 20, "cacheRead": 5},
        )
        items, usage = parse_openclaw_output(raw)
        assert len(items) == 1
        assert isinstance(items[0], NeMoGymResponseOutputMessage)
        assert items[0].content[0].text == "the answer is 4"
        assert usage["input_tokens"] == 105
        assert usage["output_tokens"] == 20

    def test_no_text_no_items(self) -> None:
        raw = _envelope([], usage={"input": 1, "output": 0})
        items, usage = parse_openclaw_output(raw)
        assert items == []
        assert usage["input_tokens"] == 1

    def test_log_prefix_tolerated(self) -> None:
        raw = "INFO starting agent\n" + _envelope([{"text": "ok"}])
        items, _ = parse_openclaw_output(raw)
        assert len(items) == 1
        assert items[0].content[0].text == "ok"


class TestParseOpenclawSession:
    def _msg(self, role, content, **extra) -> str:
        return json.dumps({"type": "message", "message": {"role": role, "content": content, **extra}})

    def test_tool_call_and_result_and_final_text(self) -> None:
        lines = "\n".join(
            [
                json.dumps({"type": "session", "id": "s1"}),  # non-message, ignored
                self._msg("user", [{"type": "text", "text": "compute 2*3"}]),
                self._msg(
                    "assistant",
                    [
                        {
                            "type": "toolCall",
                            "id": "c1",
                            "name": "exec",
                            "arguments": {"command": "python3 -c 'print(6)'"},
                        }
                    ],
                ),
                self._msg("toolResult", [{"type": "text", "text": "6\n"}], toolCallId="c1", toolName="exec"),
                self._msg("assistant", [{"type": "text", "text": "The answer is 6."}]),
            ]
        )
        items = parse_openclaw_session(lines)
        assert len(items) == 3
        assert isinstance(items[0], NeMoGymResponseFunctionToolCall)
        assert items[0].name == "exec"
        assert json.loads(items[0].arguments)["command"] == "python3 -c 'print(6)'"
        assert isinstance(items[1], NeMoGymFunctionCallOutput)
        assert items[1].call_id == "c1"
        assert "6" in items[1].output
        assert isinstance(items[2], NeMoGymResponseOutputMessage)
        assert items[2].content[0].text == "The answer is 6."

    def test_user_messages_ignored(self) -> None:
        line = self._msg("user", [{"type": "text", "text": "hi"}])
        assert parse_openclaw_session(line) == []

    def test_malformed_lines_skipped(self) -> None:
        line = "not-json\n" + self._msg("assistant", [{"type": "text", "text": "ok"}])
        items = parse_openclaw_session(line)
        assert len(items) == 1


class TestBuildOpenclawConfig:
    def test_headless_message_tool_denied(self) -> None:
        agent = _make_agent()
        cfg = agent._build_openclaw_config({})
        assert "message" in cfg["tools"]["deny"]

    def test_preserves_setup_config(self) -> None:
        agent = _make_agent()
        base = {"gateway": {"mode": "local", "auth": {"token": "abc"}}, "agents": {"defaults": {"workspace": "/w"}}}
        cfg = agent._build_openclaw_config(base)
        assert cfg["gateway"]["auth"]["token"] == "abc"
        assert cfg["agents"]["defaults"]["workspace"] == "/w"

    def test_existing_denies_preserved_and_deduped(self) -> None:
        agent = _make_agent()
        cfg = agent._build_openclaw_config({"tools": {"deny": ["message", "gateway"]}})
        assert cfg["tools"]["deny"] == ["message", "gateway"]

    def test_user_openclaw_config_merged(self) -> None:
        agent = _make_agent(
            openclaw_config={"models": {"providers": {"nvinf": {"baseUrl": "https://x/v1"}}}, "extra": {"k": "v"}}
        )
        cfg = agent._build_openclaw_config({"gateway": {"mode": "local"}})
        assert cfg["models"]["providers"]["nvinf"]["baseUrl"] == "https://x/v1"
        assert cfg["extra"] == {"k": "v"}
        assert cfg["gateway"]["mode"] == "local"

    def test_env_passthrough(self) -> None:
        agent = _make_agent(env={"NVIDIA_API_KEY": "k", "EMPTY": ""})
        env = agent._env(Path("/tmp/h"))
        assert env["NVIDIA_API_KEY"] == "k"
        assert env["HOME"] == "/tmp/h"
        assert "EMPTY" not in env


class TestDeepMerge:
    def test_nested_merge(self) -> None:
        base = {"a": {"b": 1, "c": 2}}
        OpenClawAgent._deep_merge(base, {"a": {"c": 3, "d": 4}})
        assert base == {"a": {"b": 1, "c": 3, "d": 4}}


class TestApplyRunScope:
    def test_no_model_server_is_noop(self) -> None:
        agent = _make_agent(model="nvinf/foo/bar")
        cfg = _provider_cfg()
        model_arg = agent._apply_run_scope(cfg, _request(run_token="tok"))
        # plain eval: configured model returned and the provider config is left untouched
        assert model_arg == "nvinf/foo/bar"
        assert cfg["models"]["providers"]["nvinf"]["baseUrl"] == "http://placeholder/v1"
        assert cfg["models"]["providers"]["nvinf"]["models"][0]["id"] == "orig"

    def test_run_scoped_rewrites_provider_and_model(self) -> None:
        agent = _make_agent(
            model="nvinf/foo/bar",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        )
        cfg = _provider_cfg()
        with patch.object(OpenClawAgent, "model_server_base_url", return_value="http://ms:9000"):
            model_arg = agent._apply_run_scope(cfg, _request(run_token="tok123"))
        prov = cfg["models"]["providers"]["nvinf"]
        assert prov["baseUrl"] == "http://ms:9000/runs/tok123/v1"
        assert prov["apiKey"] == "gym"  # pragma: allowlist secret
        assert prov["models"][0]["id"] == "policy_model"
        assert prov["models"][0]["name"] == "policy_model"
        assert model_arg == "nvinf/policy_model"

    def test_run_scoped_without_token_uses_plain_model_server_url(self) -> None:
        agent = _make_agent(
            model="nvinf/foo/bar",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        )
        cfg = _provider_cfg()
        with patch.object(OpenClawAgent, "model_server_base_url", return_value="http://ms:9000"):
            model_arg = agent._apply_run_scope(cfg, _request())
        # no run token -> no /runs/<token> segment, but still rewrites to the model server
        assert cfg["models"]["providers"]["nvinf"]["baseUrl"] == "http://ms:9000/v1"
        assert model_arg == "nvinf/policy_model"


class TestRunOpenclaw:
    """Exercise _run_openclaw's config-writing path: the harness must write a complete openclaw.json
    itself and must not hard-fail when `openclaw setup` produces nothing or exits nonzero."""

    def _drive(self, agent: OpenClawAgent, request, *, setup_code=0, setup_writes_config=False):
        """Run _run_openclaw with _run_exec mocked. Captures the openclaw.json written before the
        `agent` subcommand runs. Returns (written_config_dict, model_arg)."""
        captured: dict = {}

        async def fake_run_exec(args, *, cwd, env, timeout):
            home = Path(cwd) / ".openclaw-home"
            config_path = home / ".openclaw" / "openclaw.json"
            if args[1] == "setup":
                if setup_writes_config:
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    config_path.write_text(json.dumps({"gateway": {"mode": "local"}}))
                return setup_code, "", "boom" if setup_code else ""
            # the `agent` subcommand: snapshot the config the harness wrote, capture --model arg
            captured["config"] = json.loads(config_path.read_text())
            captured["model_arg"] = args[args.index("--model") + 1]
            return 0, json.dumps({"payloads": [{"text": "hi"}], "meta": {}}), ""

        with patch.object(OpenClawAgent, "_run_exec", side_effect=fake_run_exec):
            asyncio.run(agent._run_openclaw(request, "do a thing", None))
        return captured["config"], captured["model_arg"]

    def test_setup_no_config_still_writes_valid_config(self) -> None:
        # the reported cluster bug: `openclaw setup` produces no config. must not raise.
        agent = _make_agent(
            model="nvinf/foo/bar",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            openclaw_config=_provider_cfg(),
        )
        with patch.object(OpenClawAgent, "model_server_base_url", return_value="http://ms:9000"):
            cfg, model_arg = self._drive(agent, _request(run_token="tok123"), setup_writes_config=False)
        prov = cfg["models"]["providers"]["nvinf"]
        assert prov["baseUrl"] == "http://ms:9000/runs/tok123/v1"
        assert prov["apiKey"] == "gym"  # pragma: allowlist secret
        assert prov["models"][0]["id"] == "policy_model"
        assert "message" in cfg["tools"]["deny"]
        assert model_arg == "nvinf/policy_model"

    def test_setup_nonzero_exit_tolerated(self) -> None:
        agent = _make_agent(model="nvinf/foo/bar", openclaw_config=_provider_cfg())
        cfg, model_arg = self._drive(agent, _request(), setup_code=1, setup_writes_config=False)
        # plain eval (no model_server): provider config preserved verbatim, configured model used
        assert cfg["models"]["providers"]["nvinf"]["baseUrl"] == "http://placeholder/v1"
        assert model_arg == "nvinf/foo/bar"
        assert "message" in cfg["tools"]["deny"]

    def test_setup_config_merged_when_present(self) -> None:
        agent = _make_agent(model="nvinf/foo/bar", openclaw_config=_provider_cfg())
        cfg, _ = self._drive(agent, _request(), setup_writes_config=True)
        # config seeded by setup is preserved and the wrapper layers its config on top
        assert cfg["gateway"]["mode"] == "local"
        assert cfg["models"]["providers"]["nvinf"]["baseUrl"] == "http://placeholder/v1"


class TestConfigYaml:
    def test_module_parses(self) -> None:
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        compile(app_path.read_text(), str(app_path), "exec")

    def test_config_yaml_parses(self) -> None:
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "openclaw_agent.yaml"
        data = yaml.safe_load(cfg_path.read_text())
        assert "openclaw_agent" in data
        inner = data["openclaw_agent"]["responses_api_agents"]["openclaw_agent"]
        assert inner["entrypoint"] == "app.py"
        assert inner["concurrency"] == 32
        assert inner["command"] == "openclaw"
