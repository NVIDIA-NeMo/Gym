# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import signal
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle, ObservationGap
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_agent import runner
from responses_api_agents.hermes_agent.app import (
    HermesAgent,
    HermesAgentConfig,
    HermesAgentRunRequest,
    ModelServerRef,
    ResourcesServerRef,
    _split_chat_messages,
)
from responses_api_agents.hermes_agent.observability import build_hermes_observations
from responses_api_agents.hermes_agent.setup_hermes import HERMES_COMMIT, HERMES_RELEASE, HERMES_VERSION


class _FakeResponse:
    ok = True

    def __init__(self, payload: dict, cookies: dict | None = None) -> None:
        self.payload = payload
        self.cookies = cookies or {}

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _config(**kwargs) -> HermesAgentConfig:
    return HermesAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        model_server=ModelServerRef(type="responses_api_models", name=""),
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _skip_managed_runtime_install(monkeypatch) -> None:
    monkeypatch.setattr("responses_api_agents.hermes_agent.app.ensure_hermes", lambda: Path(sys.executable))


class TestSanity:
    def test_construct(self) -> None:
        HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))

    def test_upstream_release_api_matches_adapter(self) -> None:
        assert HERMES_RELEASE == "v2026.8.19"
        assert HERMES_VERSION == "0.20.5"
        assert HERMES_COMMIT == "fcbd1076a93841fa88855acce810e342a5b78101"  # pragma: allowlist secret

    def test_concurrency_semaphore_initialized(self) -> None:
        agent = HermesAgent(config=_config(concurrency=4), server_client=MagicMock(spec=ServerClient))
        assert agent.sem._value == 4

    def test_training_token_capture_disabled_by_default(self) -> None:
        assert _config().token_id_capture is False

    def test_model_defaults_to_server_name(self) -> None:
        agent = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        assert agent._model_name() == ""

    def test_configured_model_overrides_server_name(self) -> None:
        agent = HermesAgent(config=_config(model="Qwen3.6-35B-A3B"), server_client=MagicMock(spec=ServerClient))
        assert agent._model_name() == "Qwen3.6-35B-A3B"


class TestRolloutMCPServers:
    def _agent_with_resources_server(self) -> HermesAgent:
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {
            "example_mcp_weather": {
                "resources_servers": {
                    "example_mcp_weather": {
                        "host": "127.0.0.1",
                        "port": 8123,
                    }
                }
            }
        }
        server_client._build_server_base_url.side_effect = lambda config: (f"http://{config['host']}:{config['port']}")
        return HermesAgent(
            config=HermesAgentConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                name="hermes",
                resources_server=ResourcesServerRef(
                    type="resources_servers",
                    name="example_mcp_weather",
                ),
                model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            ),
            server_client=server_client,
        )

    def test_no_metadata_preserves_verifier_only_behavior(self) -> None:
        agent = self._agent_with_resources_server()
        assert agent._hermes_mcp_servers_from_seed({}) is None
        assert "mcp_servers" not in yaml.safe_load(agent._build_config())

    def test_builds_streamable_http_entry_with_session_header(self) -> None:
        agent = self._agent_with_resources_server()
        servers = agent._hermes_mcp_servers_from_seed(
            {
                "mcp": {
                    "server_name": "example_mcp_weather",
                    "url_path": "/mcp",
                    "transport": "http",
                    "headers": {"X-NeMo-Gym-Session-Token": "secret-token"},
                }
            }
        )

        assert servers == {
            "example_mcp_weather": {
                "url": "http://127.0.0.1:8123/mcp",
                "headers": {"X-NeMo-Gym-Session-Token": "secret-token"},
            }
        }
        assert yaml.safe_load(agent._build_config(servers))["mcp_servers"] == servers

    def test_rejects_non_http_transport(self) -> None:
        agent = self._agent_with_resources_server()
        with pytest.raises(ValueError, match="not supported"):
            agent._hermes_mcp_servers_from_seed({"mcp": {"transport": "stdio"}})

    def test_preserves_sse_transport(self) -> None:
        agent = self._agent_with_resources_server()
        servers = agent._hermes_mcp_servers_from_seed(
            {
                "mcp": {
                    "server_name": "example_mcp_weather",
                    "transport": "sse",
                    "headers": {"X-NeMo-Gym-Session-Token": "token"},
                }
            }
        )

        assert servers["example_mcp_weather"]["transport"] == "sse"

    @pytest.mark.parametrize(
        "metadata",
        [
            "not-an-object",
            {"server_name": 42},
            {"url_path": 42},
            {"transport": 42},
            {"headers": "not-an-object"},
        ],
    )
    def test_rejects_malformed_metadata(self, metadata) -> None:
        agent = self._agent_with_resources_server()
        with pytest.raises(ValueError):
            agent._hermes_mcp_servers_from_seed({"mcp": metadata})

    def test_rejects_malformed_headers_without_exposing_token(self, caplog) -> None:
        secret = "do-not-log-this-token"  # pragma: allowlist secret
        agent = self._agent_with_resources_server()

        with pytest.raises(ValueError) as exc_info:
            agent._hermes_mcp_servers_from_seed({"mcp": {"headers": {"Authorization": {"token": secret}}}})

        assert "scalar values" in str(exc_info.value)
        assert secret not in str(exc_info.value)
        assert secret not in caplog.text

    def test_run_passes_rollout_mcp_config_and_session_cookie(self, monkeypatch) -> None:
        agent = self._agent_with_resources_server()
        response = NeMoGymResponse.model_validate(
            {
                "id": "resp-1",
                "created_at": 1,
                "model": "model",
                "object": "response",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "mcp__example_mcp_weather__get_weather",
                        "arguments": '{"city":"Seattle"}',
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            }
        )

        create_response = AsyncMock(return_value=response)
        monkeypatch.setattr(agent, "_create_response", create_response)
        captured: dict = {}

        async def post(server_name, url_path, json=None, cookies=None, **kwargs):
            if url_path == "/seed_session":
                return _FakeResponse(
                    {
                        "mcp": {
                            "server_name": "example_mcp_weather",
                            "url_path": "/mcp",
                            "transport": "http",
                            "headers": {"X-NeMo-Gym-Session-Token": "rollout-token"},
                            "tool_names": ["get_weather"],
                        }
                    },
                    {"session": "rollout-cookie"},
                )
            if url_path == "/verify":
                captured["verify_cookies"] = cookies
                return _FakeResponse(json | {"reward": 1.0})
            raise AssertionError(f"unexpected post: {server_name} {url_path}")

        agent.server_client.post = AsyncMock(side_effect=post)
        request = MagicMock()
        request.cookies = {}
        body = HermesAgentRunRequest.model_validate({"responses_create_params": {"input": "use the tool"}})

        result = asyncio.run(agent.run(request, body))

        assert result.reward == 1.0
        assert create_response.await_args.kwargs["hermes_mcp_servers"] == {
            "example_mcp_weather": {
                "url": "http://127.0.0.1:8123/mcp",
                "headers": {"X-NeMo-Gym-Session-Token": "rollout-token"},
            }
        }
        assert captured["verify_cookies"] == {"session": "rollout-cookie"}
        assert agent.server_client.post.await_args_list[-1].kwargs["json"]["mcp_tool_call_provenance"] == {
            "call-1": {
                "server_name": "example_mcp_weather",
                "tool_name": "get_weather",
            }
        }
        assert result.model_dump(mode="json")["mcp_tool_call_provenance"] == {
            "call-1": {
                "server_name": "example_mcp_weather",
                "tool_name": "get_weather",
            }
        }


class _FakeProcess:
    def __init__(self) -> None:
        self.returncode = None
        self.signal = None

    def send_signal(self, sig: signal.Signals) -> None:
        self.signal = sig


class TestSigtermHandler:
    """Regression tests for the concurrency-safe SIGTERM dispatcher.

    The old per-call add_signal_handler/remove_signal_handler approach raced: concurrent responses()
    calls clobbered each other's handler and the first to finish removed the only one left, so a
    later SIGTERM interrupted nobody. The fix registers a single dispatcher over a shared set of
    in-flight agents.
    """

    def test_active_processes_initialized_empty(self) -> None:
        agent = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        assert agent.active_processes == set()
        assert agent.sigterm_installed is False

    def test_handler_installed_once_and_signals_all_in_flight(self) -> None:
        agent = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))

        registered: list = []
        loop = asyncio.new_event_loop()
        loop.add_signal_handler = lambda sig, cb, *a: registered.append(cb)  # type: ignore[method-assign]
        asyncio.set_event_loop(loop)
        try:
            agent._ensure_sigterm_handler()
            assert agent.sigterm_installed is True
            assert len(registered) == 1  # exactly one dispatcher registered

            # Idempotent: a second concurrent call must NOT register another handler.
            agent._ensure_sigterm_handler()
            assert len(registered) == 1

            dispatch = registered[0]

            # Two concurrent child processes: SIGTERM must reach both.
            a, b = _FakeProcess(), _FakeProcess()
            agent.active_processes.update({a, b})
            dispatch()
            assert a.signal == signal.SIGTERM
            assert b.signal == signal.SIGTERM

            # Once a child finishes (discarded), a later SIGTERM no longer touches it.
            a.signal = None
            b.signal = None
            agent.active_processes.discard(a)
            dispatch()
            assert a.signal is None
            assert b.signal == signal.SIGTERM
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def test_handler_install_survives_unsupported_platform(self) -> None:
        # On platforms where add_signal_handler raises (e.g. non-main thread), install is a no-op
        # rather than an error, and the agent stays usable.
        agent = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))

        loop = asyncio.new_event_loop()

        def _raise(*_a, **_k):
            raise NotImplementedError

        loop.add_signal_handler = _raise  # type: ignore[method-assign]
        asyncio.set_event_loop(loop)
        try:
            agent._ensure_sigterm_handler()
            assert agent.sigterm_installed is False
        finally:
            asyncio.set_event_loop(None)
            loop.close()


class TestManagedSubprocessIsolation:
    def test_concurrent_rollouts_use_distinct_homes_and_clean_up(self, monkeypatch) -> None:
        agent = HermesAgent(config=_config(concurrency=3), server_client=MagicMock(spec=ServerClient))
        monkeypatch.setattr(agent, "_ensure_sigterm_handler", lambda: None)

        processes = []
        homes: list[Path] = []
        workdirs: list[Path] = []
        configs: list[dict] = []
        request_flags: list[bool] = []
        all_started = asyncio.Event()
        max_active = 0

        class FakeProcess:
            returncode = None

            def __init__(self, response_path: Path) -> None:
                self.response_path = response_path

            async def communicate(self):
                nonlocal max_active
                max_active = max(max_active, len(agent.active_processes))
                if len(processes) == 3:
                    all_started.set()
                await all_started.wait()
                self.response_path.write_text(
                    json.dumps(
                        {
                            "result": {"messages": [{"role": "assistant", "content": "ok"}]},
                            "observations": None,
                        }
                    ),
                    encoding="utf-8",
                )
                self.returncode = 0
                return b"", b""

        async def create_process(*args, **kwargs):
            home = Path(kwargs["env"]["HERMES_HOME"])
            workdir = Path(kwargs["cwd"])
            assert home.is_dir()
            assert home.parent == workdir
            assert Path(args[2]).is_file()
            assert Path(args[3]).parent == workdir
            configs.append(yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")))
            request_flags.append(json.loads(Path(args[2]).read_text(encoding="utf-8"))["mcp_enabled"])
            process = FakeProcess(Path(args[3]))
            processes.append(process)
            homes.append(home)
            workdirs.append(workdir)
            return process

        monkeypatch.setattr(
            "responses_api_agents.hermes_agent.app.asyncio.create_subprocess_exec",
            create_process,
        )

        async def run_concurrently():
            return await asyncio.gather(
                *(
                    agent._run_hermes_subprocess(
                        {"rollout": index},
                        hermes_mcp_servers={
                            "workplace": {
                                "url": "http://resources/mcp",
                                "headers": {"X-NeMo-Gym-Session-Token": f"token-{index}"},
                            }
                        },
                    )
                    for index in range(3)
                )
            )

        results = asyncio.run(run_concurrently())

        assert len(results) == 3
        assert max_active == 3
        assert len(set(homes)) == 3
        assert len(set(workdirs)) == 3
        assert agent.active_processes == set()
        assert all(process.returncode == 0 for process in processes)
        assert all(not home.exists() for home in homes)
        assert all(not workdir.exists() for workdir in workdirs)
        assert request_flags == [True, True, True]
        assert {config["mcp_servers"]["workplace"]["headers"]["X-NeMo-Gym-Session-Token"] for config in configs} == {
            "token-0",
            "token-1",
            "token-2",
        }

    def test_failed_rollout_cleans_up_token_config(self, monkeypatch) -> None:
        agent = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        monkeypatch.setattr(agent, "_ensure_sigterm_handler", lambda: None)
        homes: list[Path] = []
        workdirs: list[Path] = []

        class FakeProcess:
            returncode = 1

            async def communicate(self):
                return b"", b"runtime failed"

        async def create_process(*args, **kwargs):
            home = Path(kwargs["env"]["HERMES_HOME"])
            homes.append(home)
            workdirs.append(Path(kwargs["cwd"]))
            config = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
            assert config["mcp_servers"]["workplace"]["headers"] == {"X-NeMo-Gym-Session-Token": "secret-token"}
            return FakeProcess()

        monkeypatch.setattr(
            "responses_api_agents.hermes_agent.app.asyncio.create_subprocess_exec",
            create_process,
        )

        with pytest.raises(RuntimeError, match="runtime failed"):
            asyncio.run(
                agent._run_hermes_subprocess(
                    {},
                    hermes_mcp_servers={
                        "workplace": {
                            "url": "http://resources/mcp",
                            "headers": {"X-NeMo-Gym-Session-Token": "secret-token"},
                        }
                    },
                )
            )

        assert agent.active_processes == set()
        assert all(not home.exists() for home in homes)
        assert all(not workdir.exists() for workdir in workdirs)


class TestSplitChatMessages:
    def test_user_only(self) -> None:
        items = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        user, history, system = _split_chat_messages(items)
        assert user == "hi"
        assert history == []
        assert system is None

    def test_system_plus_user(self) -> None:
        items = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ]
        user, history, system = _split_chat_messages(items)
        assert user == "hi"
        assert history == []
        assert system == "be helpful"

    def test_history_then_user(self) -> None:
        items = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "follow-up"},
        ]
        user, history, system = _split_chat_messages(items)
        assert user == "follow-up"
        assert history == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]
        assert system is None

    def test_resumed_ends_on_assistant(self) -> None:
        items = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        user, history, system = _split_chat_messages(items)
        assert user == ""
        assert history == [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]

    def test_dict_inputs(self) -> None:
        items = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "ok"}]
        user, history, system = _split_chat_messages(items)
        assert user == "ok"
        assert history == []
        assert system == "be brief"

    def test_preserves_tool_call_history(self) -> None:
        tool_call = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}],
        }
        items = [
            {"role": "user", "content": "list files"},
            tool_call,
            {"role": "tool", "tool_call_id": "c1", "content": "file.txt"},
            {"role": "user", "content": "continue"},
        ]

        user, history, system = _split_chat_messages(items)

        assert user == "continue"
        assert history == items[:-1]
        assert system is None


class TestResponsesConversion:
    def test_uses_shared_converter_for_input_history_and_output(self, monkeypatch) -> None:
        import nemo_gym.base_responses_api_agent as base_agent

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        agent = HermesAgent(config=_config(), server_client=server_client)
        seen: dict = {}

        async def run_runtime(payload):
            seen.update(payload)
            return (
                {
                    "messages": [
                        *payload["history"],
                        {"role": "user", "content": payload["user_message"]},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "c2",
                                    "type": "function",
                                    "function": {"name": "terminal", "arguments": '{"cmd":"pwd"}'},
                                }
                            ],
                        },
                        {"role": "tool", "tool_call_id": "c2", "content": "/workspace"},
                        {"role": "assistant", "content": "done"},
                    ]
                },
                None,
            )

        monkeypatch.setattr(agent, "_run_hermes_subprocess", run_runtime)
        body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            {
                "input": [
                    {"role": "user", "type": "message", "content": "list files"},
                    {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "terminal",
                        "arguments": '{"cmd":"ls"}',
                    },
                    {"type": "function_call_output", "call_id": "c1", "output": "file.txt"},
                    {"role": "user", "type": "message", "content": "where am I?"},
                ]
            }
        )

        response = asyncio.run(agent._create_response(body))

        assert seen["user_message"] == "where am I?"
        assert [message["role"] for message in seen["history"]] == ["user", "assistant", "tool"]
        assert seen["history"][1]["tool_calls"][0]["function"]["name"] == "terminal"
        assert [item.type for item in response.output] == ["function_call", "function_call_output", "message"]
        assert response.output[0].name == "terminal"
        assert response.output[1].output == "/workspace"
        assert response.output[2].content[0].text == "done"


class TestRolloutCorrelation:
    def test_responses_applies_rollout_prefix(self, monkeypatch) -> None:
        from fastapi.testclient import TestClient

        import nemo_gym.base_responses_api_agent as base_agent
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        agent = HermesAgent(config=_config(), server_client=server_client)
        monkeypatch.setattr(agent, "_ensure_sigterm_handler", lambda: None)
        run_runtime = AsyncMock(
            return_value=(
                {"messages": [{"role": "assistant", "content": "ok"}]},
                AgentObservationBundle(source="hermes"),
            )
        )
        monkeypatch.setattr(agent, "_run_hermes_subprocess", run_runtime)
        client = TestClient(agent.setup_webserver())

        assert client.post("/ng-rollout/rid/v1/responses", json={"input": "hi"}).status_code == 200
        assert run_runtime.await_args.args[0]["base_url"] == "http://h:1/ng-rollout/rid/v1"

        direct = asyncio.run(agent.responses(request=None, body=NeMoGymResponseCreateParamsNonStreaming(input="hi")))
        assert run_runtime.await_args.args[0]["base_url"] == "http://h:1/v1"
        assert "_ng_agent_observations" not in direct.model_dump(mode="json")

        episode = asyncio.run(
            agent._create_episode(
                body=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
                rollout_id="rid",
            )
        )
        assert run_runtime.await_args.args[0]["base_url"] == "http://h:1/ng-rollout/rid/v1"
        assert episode.observations.source == "hermes"


class TestMaxTokens:
    def _agent_and_seen(self, monkeypatch, **config_kwargs) -> tuple[HermesAgent, dict]:
        import nemo_gym.base_responses_api_agent as base_agent

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        agent = HermesAgent(config=_config(**config_kwargs), server_client=server_client)
        monkeypatch.setattr(agent, "_ensure_sigterm_handler", lambda: None)
        seen: dict = {}

        async def run_runtime(payload):
            seen.update(payload)
            return {"messages": [{"role": "assistant", "content": "ok"}]}, None

        monkeypatch.setattr(agent, "_run_hermes_subprocess", run_runtime)
        return agent, seen

    def test_max_tokens_passed_to_ai_agent(self, monkeypatch) -> None:
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        agent, seen = self._agent_and_seen(monkeypatch, max_tokens=4096)
        asyncio.run(agent.responses(request=None, body=NeMoGymResponseCreateParamsNonStreaming(input="hi")))
        assert seen["max_tokens"] == 4096

    def test_max_tokens_defaults_to_none(self, monkeypatch) -> None:
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        agent, seen = self._agent_and_seen(monkeypatch)
        asyncio.run(agent.responses(request=None, body=NeMoGymResponseCreateParamsNonStreaming(input="hi")))
        assert seen["max_tokens"] is None

    def test_uses_request_overrides(self, monkeypatch) -> None:
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        agent, seen = self._agent_and_seen(monkeypatch, temperature=0.25)
        asyncio.run(agent.responses(request=None, body=NeMoGymResponseCreateParamsNonStreaming(input="hi")))

        assert seen["request_overrides"] == {
            "temperature": 0.25,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "truncate_history_thinking": False,
                }
            },
        }

    def test_can_disable_chat_template_overrides(self, monkeypatch) -> None:
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        agent, seen = self._agent_and_seen(
            monkeypatch,
            temperature=None,
            chat_template_kwargs_enabled=False,
        )
        asyncio.run(agent.responses(request=None, body=NeMoGymResponseCreateParamsNonStreaming(input="hi")))

        assert seen["request_overrides"] == {}


class TestManagedRunner:
    def test_uses_upstream_constructor_contract(self, monkeypatch) -> None:
        seen: dict = {}

        class _StubAIAgent:
            def __init__(self, **kwargs) -> None:
                assert seen["mcp_prepared"] is True
                seen.update(kwargs)
                seen["agent"] = self

            def run_conversation(self, *args) -> dict:
                seen["run_args"] = args
                return {"messages": [{"role": "assistant", "content": "ok"}]}

            def interrupt(self, reason: str) -> None:
                seen["interrupt"] = reason

            def close(self) -> None:
                seen["closed"] = True

        monkeypatch.setattr(runner, "_prepare_mcp_tools", lambda: seen.__setitem__("mcp_prepared", True))
        monkeypatch.setattr(runner, "_load_ai_agent", lambda: _StubAIAgent)
        result, observations = runner.run(
            {
                "base_url": "http://model/v1",
                "api_key": "gym",  # pragma: allowlist secret
                "model": "model",
                "max_iterations": 7,
                "max_tokens": 4096,
                "enabled_toolsets": ["terminal"],
                "disabled_toolsets": ["browser"],
                "request_overrides": {"temperature": 0.2},
                "user_message": "question",
                "system_message": "system",
                "history": [{"role": "user", "content": "earlier"}],
                "capture_observations": True,
                "mcp_enabled": True,
            }
        )

        assert result["messages"][-1]["content"] == "ok"
        assert seen["base_url"] == "http://model/v1"
        assert seen["max_tokens"] == 4096
        assert seen["request_overrides"] == {"temperature": 0.2}
        assert seen["skip_background_review"] is True
        assert seen["agent"]._persist_disabled is True
        assert seen["agent"]._disable_streaming is True
        assert seen["run_args"] == (
            "question",
            "system",
            [{"role": "user", "content": "earlier"}],
        )
        assert observations is not None
        assert observations["source"] == "hermes"
        assert observations["invocations"][0]["messages"][-1]["content"] == "ok"
        json.dumps(observations)
        bundle = build_hermes_observations(
            observations,
            model_ref=ModelServerRef(type="responses_api_models", name="policy"),
        )
        root = next(record for record in bundle.records if record.kind == "agent_invocation")
        assert root.conversation[-1].content[0].text == "ok"
        assert seen["closed"] is True
        assert seen["mcp_prepared"] is True
        assert "use_streaming" not in seen
        assert "insert_reasoning" not in seen
        assert "persist_session" not in seen


class TestObservability:
    @pytest.mark.parametrize(
        ("terminal_backend", "runtime_gap"),
        [("local", "no_sandbox_runtime"), ("docker", "sandbox_observation_unavailable")],
    )
    def test_observation_failure_does_not_change_response(self, monkeypatch, terminal_backend, runtime_gap) -> None:
        import nemo_gym.base_responses_api_agent as base_agent
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        agent = HermesAgent(config=_config(terminal_backend=terminal_backend), server_client=server_client)
        runtime_result = {
            "completed": True,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
            ],
        }

        async def run_runtime(payload):
            observations = (
                AgentObservationBundle(
                    source="hermes",
                    gaps=[ObservationGap(code="observation_capture_failed")],
                )
                if payload["capture_observations"]
                else None
            )
            return runtime_result, observations

        monkeypatch.setattr(agent, "_run_hermes_subprocess", run_runtime)
        body = NeMoGymResponseCreateParamsNonStreaming(input="hi")
        baseline = asyncio.run(agent.responses(request=None, body=body))
        episode = asyncio.run(agent._create_episode(body=body, rollout_id="rid"))

        assert [item.model_dump(exclude={"id"}) for item in episode.response.output] == [
            item.model_dump(exclude={"id"}) for item in baseline.output
        ]
        assert episode.response.usage == baseline.usage
        assert [gap.code for gap in episode.observations.gaps] == [
            "observation_capture_failed",
            runtime_gap,
        ]

    def test_run_returns_observations_without_leaking_internal_attachment(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {"observability_enabled": True}
        agent = HermesAgent(config=_config(), server_client=server_client)
        response = NeMoGymResponse.model_validate(
            {
                "id": "resp-1",
                "created_at": 1,
                "model": "model",
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        observed_response = AsyncMock(
            return_value=AgentEpisode(
                response=response,
                observations=AgentObservationBundle(source="hermes"),
            )
        )

        async def post(server_name, url_path, json=None, cookies=None, **kwargs):
            if url_path == "/seed_session":
                return _FakeResponse({}, {"session": "1"})
            if url_path.endswith("/v1/responses"):
                response = await agent.responses(MagicMock(path_params={"rollout_id": "1-2"}), json)
                return _FakeResponse(response.model_dump(mode="json"), cookies)
            return _FakeResponse(json | {"reward": 1.0})

        server_client.post = AsyncMock(side_effect=post)
        request = MagicMock()
        request.cookies = {}
        body = HermesAgentRunRequest.model_validate(
            {
                "responses_create_params": {"input": "solve"},
                "_ng_task_index": 1,
                "_ng_rollout_index": 2,
            }
        )

        with patch.object(HermesAgent, "_create_episode", observed_response):
            result = asyncio.run(agent.run(request, body))

        assert result.ng_agent_observations is not None
        assert result.ng_agent_observations.source == "hermes"
        verify_json = server_client.post.await_args_list[-1].kwargs["json"]
        assert "_ng_agent_observations" not in verify_json["response"]
        assert "rollout_id" not in verify_json

    def test_observer_failure_does_not_mask_agent_exception(self, monkeypatch) -> None:
        import nemo_gym.base_responses_api_agent as base_agent
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        agent = HermesAgent(config=_config(), server_client=server_client)

        async def run_runtime(_payload):
            raise ValueError("agent failed")

        monkeypatch.setattr(agent, "_run_hermes_subprocess", run_runtime)

        with pytest.raises(ValueError, match="agent failed"):
            asyncio.run(
                agent._create_episode(
                    body=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
                    rollout_id="rid",
                )
            )
