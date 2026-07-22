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
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.cvdp_agent.app import (
    CVDPAgent,
    CVDPAgentConfig,
    CVDPAgentRunRequest,
    deps_build_env,
)


class TestApp:
    def test_sanity(self) -> None:
        config = CVDPAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            simple_agent=True,
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
        )
        CVDPAgent(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_responses(self, monkeypatch: MonkeyPatch) -> None:
        config = CVDPAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            simple_agent=True,
            model_server=ModelServerRef(
                type="responses_api_models",
                name="my server name",
            ),
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
        )
        server = CVDPAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        mock_response_data = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Hello! How can I help you today?",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        dotjson_mock = AsyncMock()
        dotjson_mock.read.return_value = json.dumps(mock_response_data)
        dotjson_mock.cookies = MagicMock()
        server.server_client.post.return_value = dotjson_mock

        # No model provided should use the one from the config
        res_no_model = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert res_no_model.status_code == 200
        server.server_client.post.assert_called_with(
            server_name="my server name",
            url_path="/v1/responses",
            json=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(content="hello", role="user", type="message")]
            ),
            cookies=None,
        )

        actual_responses_dict = res_no_model.json()
        expected_responses_dict = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": None,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Hello! How can I help you today?",
                            "type": "output_text",
                            "logprobs": None,
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "temperature": None,
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "background": None,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "reasoning": None,
            "service_tier": None,
            "status": None,
            "text": None,
            "top_logprobs": None,
            "truncation": None,
            "usage": None,
            "user": None,
            "conversation": None,
            "prompt_cache_key": None,
            "safety_identifier": None,
        }
        assert expected_responses_dict == actual_responses_dict


class TestAgenticSandboxSpec:
    def _agent(self, **overrides) -> CVDPAgent:
        values = {
            "image": "docker://ghcr.io/hdl/sim/osvb",
            "sandbox_provider": {"opensandbox": {}},
            **overrides,
        }
        config = CVDPAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="cvdp_agent",
            simple_agent=False,
            resources_server=ResourcesServerRef(type="resources_servers", name="cvdp"),
            **values,
        )
        return CVDPAgent(config=config, server_client=MagicMock(spec=ServerClient))

    @staticmethod
    def _run_request() -> CVDPAgentRunRequest:
        return CVDPAgentRunRequest.model_validate(
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": "build rtl"}],
                    "model": "test-model",
                },
                "verifier_metadata": {
                    "task_id": "task-1",
                    "categories": ["cid001"],
                    "difficulty": "easy",
                },
            }
        )

    @staticmethod
    def _sandbox_context(box):
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=box)
        context.__aexit__ = AsyncMock(return_value=False)
        return context

    def test_open_sandbox_uses_oci_ref_without_provider_options(self) -> None:
        agent = self._agent(deps_provision="archive")
        agent._model_url = MagicMock(return_value="https://model.example/v1")
        body = MagicMock()
        body.responses_create_params.model = "model"
        spec = agent._build_spec(body, "build rtl", {}, agent._resolve_image())
        assert spec.image == "ghcr.io/hdl/sim/osvb"
        assert spec.provider_options == {}
        assert spec.env["NV_AGENT_HOME"] == "/code/.home"

    def test_local_sif_rejected_for_open_sandbox(self) -> None:
        agent = self._agent(image="/tmp/cvdp.sif")
        with pytest.raises(ValueError, match="Apptainer"):
            agent._resolve_image()

    async def test_harness_failure_returns_zero_reward(self) -> None:
        agent = self._agent(deps_provision="baked")
        agent._image = "ghcr.io/hdl/sim/osvb"
        box = MagicMock()
        box.start = AsyncMock()
        box.exec = AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout="", stderr=""),
                SimpleNamespace(return_code=7, stdout="", stderr="harness failed"),
            ]
        )

        with patch(
            "responses_api_agents.cvdp_agent.app.AsyncSandbox",
            return_value=self._sandbox_context(box),
        ):
            result = await agent.run(MagicMock(cookies={}), self._run_request())

        assert result.reward == 0.0
        assert result.task_id == "task-1"
        assert result.container_exit_code == 7
        assert "harness failed" in result.container_stderr
        agent.server_client.post.assert_not_called()

    async def test_missing_harness_response_returns_zero_reward(self) -> None:
        agent = self._agent(deps_provision="baked")
        agent._image = "ghcr.io/hdl/sim/osvb"
        box = MagicMock()
        box.start = AsyncMock()
        box.exec = AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout="", stderr=""),
                SimpleNamespace(return_code=0, stdout="", stderr=""),
            ]
        )
        box.download = AsyncMock(side_effect=FileNotFoundError("response.json"))

        with patch(
            "responses_api_agents.cvdp_agent.app.AsyncSandbox",
            return_value=self._sandbox_context(box),
        ):
            result = await agent.run(MagicMock(cookies={}), self._run_request())

        assert result.reward == 0.0
        assert result.task_id == "task-1"
        assert result.container_exit_code is None
        assert "response.json" in result.container_stderr
        agent.server_client.post.assert_not_called()

    def test_deps_archive_contains_runtime_root(self, tmp_path) -> None:
        agent = self._agent()
        deps = tmp_path / "hermes"
        (deps / "bin").mkdir(parents=True)
        (deps / "bin" / "python").write_text("runtime")
        (deps / ".installed").write_text("recipe")
        archive = agent._archive_deps(deps)
        with tarfile.open(archive, "r:gz") as tar:
            assert "bin/python" in tar.getnames()

    def test_deps_build_uses_private_host_state(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", "/full-home")
        deps = tmp_path / "deps" / "claude_code_agent"
        env = deps_build_env(deps)

        build_root = deps.parent / ".claude_code_agent-build"
        assert env["HOME"] == str(build_root / "home")
        assert env["PIP_CACHE_DIR"] == str(build_root / "cache" / "pip")
        assert env["NPM_CONFIG_CACHE"] == str(build_root / "cache" / "npm")
        assert env["TMPDIR"] == str(build_root / "tmp")
        assert env["PYTHONPATH"] == ""
        assert Path(env["TMPDIR"]).is_dir()

    def test_named_provider_config_is_resolved(self) -> None:
        config = CVDPAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="cvdp_agent",
            simple_agent=False,
            resources_server=ResourcesServerRef(type="resources_servers", name="cvdp"),
            sandbox_provider="sandbox",
        )
        client = ServerClient.model_construct(
            global_config_dict={
                "sandbox": {
                    "opensandbox": {"connection": {"domain": "sandbox.example"}},
                    "default_metadata": {"sandbox-api": "opensandbox-sdk"},
                }
            }
        )
        agent = CVDPAgent(config=config, server_client=client)
        assert "opensandbox" in agent._sandbox_provider
        assert agent._sandbox_metadata == {"sandbox-api": "opensandbox-sdk"}

    async def test_responses_continues_on_reasoning_only(self, monkeypatch: MonkeyPatch) -> None:
        config = CVDPAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            simple_agent=True,
            model_server=ModelServerRef(
                type="responses_api_models",
                name="my server name",
            ),
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
        )
        server = CVDPAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        mock_response_reasoning_data = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "summary": [
                        {
                            "text": "I'm thinking how to respond",
                            "type": "summary_text",
                        }
                    ],
                    "status": "completed",
                    "type": "reasoning",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        mock_response_chat_data = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Hello! How can I help you today?",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        dotjson_mock = AsyncMock()
        dotjson_mock.read.side_effect = [json.dumps(mock_response_reasoning_data), json.dumps(mock_response_chat_data)]
        dotjson_mock.cookies = MagicMock()
        server.server_client.post.return_value = dotjson_mock

        # No model provided should use the one from the config
        res_no_model = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert res_no_model.status_code == 200

        expected_calls = [
            call(
                server_name="my server name",
                url_path="/v1/responses",
                json=NeMoGymResponseCreateParamsNonStreaming(
                    input=[NeMoGymEasyInputMessage(content="hello", role="user", type="message")]
                ),
                cookies=None,
            ),
            call().ok.__bool__(),
            call().read(),
            call(
                server_name="my server name",
                url_path="/v1/responses",
                json=NeMoGymResponseCreateParamsNonStreaming(
                    input=[
                        NeMoGymEasyInputMessage(content="hello", role="user", type="message"),
                        NeMoGymResponseReasoningItem(
                            id="msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                            summary=[NeMoGymSummary(text="I'm thinking how to respond", type="summary_text")],
                            type="reasoning",
                            encrypted_content=None,
                            status="completed",
                        ),
                    ]
                ),
                cookies=dotjson_mock.cookies,
            ),
            call().ok.__bool__(),
            call().read(),
            call().cookies.items(),
            call().cookies.items().__iter__(),
            call().cookies.items().__len__(),
        ]
        server.server_client.post.assert_has_calls(expected_calls)

        actual_responses_dict = res_no_model.json()
        expected_responses_dict = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": None,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "encrypted_content": None,
                    "summary": [
                        {
                            "text": "I'm thinking how to respond",
                            "type": "summary_text",
                        }
                    ],
                    "type": "reasoning",
                },
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Hello! How can I help you today?",
                            "type": "output_text",
                            "logprobs": None,
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                },
            ],
            "parallel_tool_calls": True,
            "temperature": None,
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "background": None,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "reasoning": None,
            "service_tier": None,
            "status": None,
            "text": None,
            "top_logprobs": None,
            "truncation": None,
            "usage": None,
            "user": None,
            "conversation": None,
            "prompt_cache_key": None,
            "safety_identifier": None,
        }
        assert expected_responses_dict == actual_responses_dict
