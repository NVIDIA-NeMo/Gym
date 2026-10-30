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

from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import (
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


def _drop_nulls(value):
    """Remove dictionary entries with a value of ``None`` recursively.

    SDK releases can add optional response fields at any depth.
    Exact comparisons should ignore these unset fields.
    Expected non-null values remain part of the comparison.
    """
    if isinstance(value, dict):
        return {k: _drop_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_nulls(v) for v in value]
    return value


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
        assert _drop_nulls(expected_responses_dict) == _drop_nulls(actual_responses_dict)


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

    def test_sandbox_model_url_carries_the_current_span(self, monkeypatch) -> None:
        from nemo_gym import server_utils

        segment = "/ng-trace/00-" + "a" * 32 + "-" + "b" * 16 + "-01"
        monkeypatch.setattr(server_utils, "trace_context_path_segment", lambda: segment)
        agent = self._agent(deps_provision="archive")
        agent._model_url = MagicMock(return_value="http://model.example")
        spec = agent._build_spec(self._run_request(), "build rtl", {}, agent._resolve_image())
        assert spec.env["NV_MODEL_URL"] == f"http://model.example{segment}"

    def test_sandbox_model_url_carries_the_rollout_prefix_when_capture_is_on(self, monkeypatch) -> None:
        from nemo_gym import server_utils

        segment = "/ng-trace/00-" + "a" * 32 + "-" + "b" * 16 + "-01"
        monkeypatch.setattr(server_utils, "trace_context_path_segment", lambda: segment)
        agent = self._agent(deps_provision="archive")
        agent.server_client.global_config_dict = {"observability_enabled": True}
        agent._model_url = MagicMock(return_value="http://model.example")
        body = CVDPAgentRunRequest.model_validate(
            {
                "responses_create_params": {"input": [{"role": "user", "content": "build rtl"}], "model": "m"},
                "_ng_task_index": 3,
                "_ng_rollout_index": 0,
            }
        )
        spec = agent._build_spec(body, "build rtl", {}, agent._resolve_image())
        assert spec.env["NV_MODEL_URL"] == f"http://model.example/ng-rollout/3-0{segment}"

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
        env = deps_build_env(deps, {"ARCH": "x86_64-unknown-linux-gnu", "NODE_ARCH": "linux-x64"})

        build_root = deps.parent / ".claude_code_agent-build"
        assert env["HOME"] == str(build_root / "home")
        assert env["PIP_CACHE_DIR"] == str(build_root / "cache" / "pip")
        assert env["NPM_CONFIG_CACHE"] == str(build_root / "cache" / "npm")
        assert env["TMPDIR"] == str(build_root / "tmp")
        assert env["PYTHONPATH"] == ""
        assert env["ARCH"] == "x86_64-unknown-linux-gnu"
        assert env["NODE_ARCH"] == "linux-x64"
        assert Path(env["TMPDIR"]).is_dir()

    def test_opensandbox_deps_default_to_amd64_even_on_arm_host(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("responses_api_agents.cvdp_agent.app.os.uname", lambda: SimpleNamespace(machine="aarch64"))
        agent = self._agent(sandbox_provider={"opensandbox": {}})

        assert agent._deps_target_arch() == ("amd64", "x86_64-unknown-linux-gnu", "linux-x64")
        assert agent._deps_needs_sandbox_build() is True

    def test_opensandbox_deps_honor_explicit_arm64_platform(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("responses_api_agents.cvdp_agent.app.os.uname", lambda: SimpleNamespace(machine="aarch64"))
        agent = self._agent(
            sandbox_provider={"opensandbox": {}},
            sandbox_spec={"provider_options": {"platform": {"os": "linux", "arch": "arm64"}}},
        )

        assert agent._deps_target_arch() == ("arm64", "aarch64-unknown-linux-gnu", "linux-arm64")
        assert agent._deps_needs_sandbox_build() is False

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
                    "content": None,
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
        assert _drop_nulls(expected_responses_dict) == _drop_nulls(actual_responses_dict)


class TestSimpleRunAttemptEvidence:
    """/run owns the model calls of every attempt and accounts their usage, without changing what it dispatches."""

    @staticmethod
    def _reply(response_id: str, text: str | None, usage: tuple[int, int] | None) -> dict:
        output = []
        if text is not None:
            output.append(
                {
                    "id": f"msg_{response_id}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            )
        body = {
            "id": response_id,
            "created_at": 1.0,
            "model": "synthetic-model",
            "object": "response",
            "output": output,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
        if usage is not None:
            body["usage"] = {
                "input_tokens": usage[0],
                "output_tokens": usage[1],
                "total_tokens": sum(usage),
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            }
        return body

    def _run(
        self, model_replies: list, parse_failed: list[bool], *, capture: bool = True, max_steps: int | None = None
    ):
        """Run /run in process: model and resources servers are scripted, the self-call reaches this agent's app."""
        config = CVDPAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="cvdp_agent",
            simple_agent=True,
            max_steps=max_steps,
            resources_server=ResourcesServerRef(type="resources_servers", name="cvdp"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        )
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {"observability_enabled": capture}
        agent = CVDPAgent(config=config, server_client=server_client)
        client = TestClient(agent.setup_webserver())
        model_requests, verify_payloads, self_replies = [], [], []
        model_replies, parse_failed = list(model_replies), list(parse_failed)

        def reply(status: int, payload: dict) -> MagicMock:
            response = MagicMock()
            response.ok = status < 400
            response.status = status
            response.cookies = {}
            response.read = AsyncMock(return_value=json.dumps(payload).encode())
            response.raise_for_status.side_effect = None if status < 400 else RuntimeError(f"HTTP {status}")
            return response

        async def post(server_name, url_path, json=None, cookies=None, **kwargs):
            payload = json.model_dump(mode="json", exclude_unset=True) if hasattr(json, "model_dump") else json
            if server_name == "cvdp_agent":
                result = client.post(url_path, json=payload)
                self_replies.append(result.json())
                return reply(result.status_code, result.json())
            if server_name == "policy_model":
                model_requests.append(payload)
                body = model_replies.pop(0)
                return reply(500, {"error": "synthetic"}) if body is None else reply(200, body)
            if url_path == "/seed_session":
                return reply(200, {})
            verify_payloads.append(payload)
            return reply(200, payload | {"reward": 0.0, "parse_failed": parse_failed.pop(0)})

        server_client.post.side_effect = post
        row = {
            "_ng_task_index": 3,
            "_ng_rollout_index": 1,
            "responses_create_params": {"input": [{"role": "user", "content": "write module a"}], "temperature": 0.6},
            "verifier_metadata": {"task_id": "synthetic_0001"},
        }
        result = client.post("/run", json=row)
        assert result.status_code == 200, result.text
        return result.json(), model_requests, verify_payloads, self_replies

    def test_parse_retry_owns_both_calls_and_reports_the_episode_usage(self) -> None:
        first, last = self._reply("resp_a", "no code", (10, 7)), self._reply("resp_b", "module a; endmodule", (10, 9))
        result, model_requests, verify_payloads, self_replies = self._run([first, last], [True, False])

        assert len(model_requests) == 2 and model_requests[0] == model_requests[1]
        assert [p["response"]["id"] for p in verify_payloads] == ["resp_a", "resp_b"]
        assert all("_ng_trajectory" not in p["response"] for p in verify_payloads)
        assert [p["response"]["usage"]["output_tokens"] for p in verify_payloads] == [7, 9]
        assert all("_ng_trajectory" in r for r in self_replies)
        assert result["response"]["id"] == "resp_b"
        assert result["response"]["output"][0]["content"][0]["text"] == "module a; endmodule"
        assert (result["response"]["usage"]["input_tokens"], result["response"]["usage"]["output_tokens"]) == (20, 16)
        trajectory = result["ng_trajectory"]
        assert (trajectory["task_id"], trajectory["rollout_id"]) == ("3", "3-1")
        assert [ref["response_id"] for ref in trajectory["invocations"][0]["model_calls"]] == ["resp_a", "resp_b"]
        assert [(t["turn_no"], [r["response_id"] for r in t["model_calls"]]) for t in trajectory["turns"]] == [
            (1, ["resp_a"]),
            (2, ["resp_b"]),
        ]
        assert trajectory["gaps"] == []

    def test_single_call_keeps_its_usage_and_owns_its_call(self) -> None:
        result, model_requests, verify_payloads, _ = self._run(
            [self._reply("resp_a", "module a; endmodule", (10, 5))], [False]
        )

        assert len(model_requests) == 1 and len(verify_payloads) == 1
        assert result["response"]["usage"] == verify_payloads[0]["response"]["usage"]
        assert [ref["response_id"] for ref in result["ng_trajectory"]["invocations"][0]["model_calls"]] == ["resp_a"]

    def test_attempt_without_a_reply_is_a_missing_reference_not_an_invented_one(self) -> None:
        result, model_requests, verify_payloads, _ = self._run(
            [None, self._reply("resp_b", "module a; endmodule", (10, 6))], [False]
        )

        assert len(model_requests) == 2 and len(verify_payloads) == 1
        assert (result["response"]["usage"]["input_tokens"], result["response"]["usage"]["output_tokens"]) == (10, 6)
        trajectory = result["ng_trajectory"]
        assert [ref["response_id"] for ref in trajectory["invocations"][0]["model_calls"]] == ["resp_b"]
        assert [(g["code"], g["detail"]) for g in trajectory["gaps"]] == [
            ("model_call_reference_unavailable", "attempt:1")
        ]

    def test_missing_attempt_usage_makes_the_episode_usage_unknown(self) -> None:
        result, *_ = self._run(
            [self._reply("resp_a", "no code", None), self._reply("resp_b", "module a; endmodule", (10, 9))],
            [True, False],
        )

        assert result["response"]["usage"] is None

    def test_without_capture_the_reply_and_result_carry_no_trajectory(self) -> None:
        result, _, verify_payloads, self_replies = self._run(
            [self._reply("resp_a", "no code", (10, 7)), self._reply("resp_b", "module a; endmodule", (10, 9))],
            [True, False],
            capture=False,
        )

        assert "ng_trajectory" not in result
        assert all("_ng_trajectory" not in r for r in self_replies)
        assert all("_ng_trajectory" not in p["response"] for p in verify_payloads)
        assert (result["response"]["usage"]["input_tokens"], result["response"]["usage"]["output_tokens"]) == (20, 16)

    def test_multi_step_call_without_usage_keeps_the_verify_payload_and_reports_unknown_usage(self) -> None:
        # One attempt, two model calls: the first returns no message and no usage, so the loop continues.
        result, model_requests, verify_payloads, _ = self._run(
            [self._reply("resp_a", None, None), self._reply("resp_b", "module a; endmodule", (10, 9))], [False]
        )

        assert len(model_requests) == 2 and len(verify_payloads) == 1
        # /verify receives the same partial sum it received before this agent reported evidence.
        assert (
            verify_payloads[0]["response"]["usage"]["input_tokens"],
            verify_payloads[0]["response"]["usage"]["output_tokens"],
        ) == (10, 9)
        assert set(verify_payloads[0]["response"]) & {"_ng_trajectory", "_ng_usage_complete"} == set()
        # After verification the episode usage is unknown, not the partial sum.
        assert result["response"]["usage"] is None
        assert [ref["response_id"] for ref in result["ng_trajectory"]["invocations"][0]["model_calls"]] == [
            "resp_a",
            "resp_b",
        ]

    def test_an_evidence_error_is_a_gap_and_never_another_attempt(self, monkeypatch: MonkeyPatch) -> None:
        import responses_api_agents.cvdp_agent.app as cvdp_app

        def refuse(*args, **kwargs):
            raise ValueError("synthetic evidence failure")

        monkeypatch.setattr(cvdp_app.TrajectoryRecord, "model_validate", refuse)
        result, model_requests, verify_payloads, _ = self._run(
            [self._reply("resp_a", "module a; endmodule", (10, 5))], [False]
        )

        assert len(model_requests) == 1 and len(verify_payloads) == 1
        assert result["reward"] == 0.0 and result["response"]["id"] == "resp_a"
        assert [(g["code"], g["detail"]) for g in result["ng_trajectory"]["gaps"]] == [
            ("agent_observations_invalid", "attempt:1:ValueError")
        ]
        assert result["ng_trajectory"]["invocations"][0]["model_calls"] == []
        assert result["ng_trajectory"]["invocations"][0]["status"] == "unknown"

    @pytest.mark.parametrize("usage", [(10, 5), None], ids=["usage", "missing_usage"])
    @pytest.mark.parametrize(
        ("outcome", "expected_status", "expected_stop_reason"),
        [
            ("iteration_limit", "incomplete", "max_iterations"),
            ("failed", "failed", None),
            ("incomplete", "incomplete", "max_output_tokens"),
            ("completed", "completed", None),
            ("legacy", "completed", None),
        ],
    )
    def test_invocation_outcome_preserves_requests_verification_and_results(
        self, outcome: str, expected_status: str, expected_stop_reason: str | None, usage: tuple[int, int] | None
    ) -> None:
        reply = self._reply("resp_a", "module a; endmodule", usage)
        if outcome == "iteration_limit":
            reply["status"] = "completed"
            reply["output"] = [{"id": "reason_a", "type": "reasoning", "summary": []}]
        elif outcome == "failed":
            reply["status"] = "failed"
            reply["error"] = {"code": "server_error", "message": "synthetic failure with partial output"}
        elif outcome == "incomplete":
            reply["status"] = "incomplete"
            reply["incomplete_details"] = {"reason": "max_output_tokens"}
        elif outcome == "completed":
            reply["status"] = "completed"

        result, requests, verified, _ = self._run([reply], [False], max_steps=1)
        without_capture, plain_requests, plain_verified, _ = self._run([reply], [False], capture=False, max_steps=1)

        invocation = result["ng_trajectory"]["invocations"][0]
        assert (invocation["status"], invocation["stop_reason"]) == (expected_status, expected_stop_reason)
        assert [ref["response_id"] for ref in invocation["model_calls"]] == ["resp_a"]
        assert len(requests) == len(verified) == 1
        assert requests == plain_requests
        assert verified == plain_verified
        assert {key: value for key, value in result.items() if key != "ng_trajectory"} == without_capture
        assert result["response"] == verified[0]["response"]
        assert result["reward"] == 0.0

    def test_recovered_iteration_limit_uses_final_attempt_outcome_and_keeps_both_calls(self) -> None:
        first = self._reply("resp_a", None, None)
        first["output"] = [{"id": "reason_a", "type": "reasoning", "summary": []}]
        last = self._reply("resp_b", "module a; endmodule", (10, 5))
        result, requests, verified, _ = self._run([first, last], [True, False], max_steps=1)
        invocation = result["ng_trajectory"]["invocations"][0]
        assert (invocation["status"], invocation["stop_reason"]) == ("completed", None)
        assert [ref["response_id"] for ref in invocation["model_calls"]] == ["resp_a", "resp_b"]
        assert len(requests) == len(verified) == 2
        assert requests[0] == requests[1]
        assert result["response"]["usage"] is None
        assert result["response"]["output"] == verified[-1]["response"]["output"]
        assert result["reward"] == 0.0
