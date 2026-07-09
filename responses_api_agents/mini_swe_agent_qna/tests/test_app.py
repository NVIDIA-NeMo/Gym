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
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import yaml
from pytest import approx

import responses_api_agents.mini_swe_agent_qna.app as app_module
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.mini_swe_agent_qna.app import (
    MiniSWEAgentQna,
    MiniSWEAgentQnaConfig,
    MiniSWEAgentQnaRunRequest,
    _answer_message_item,
    _default_response_object,
    _json_dict_from_metadata,
    _message_content_to_text,
    _resolve_image,
    _responses_create_params_to_model_kwargs,
    _run_mini_swe_qna,
    _split_trajectory_for_responses,
    _strip_extra,
)


MODULE = "responses_api_agents.mini_swe_agent_qna.app"


def _make_config(**over: Any) -> MiniSWEAgentQnaConfig:
    base: dict[str, Any] = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="swe_atlas_qna_mini_swe_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        resources_server=ResourcesServerRef(type="resources_servers", name="swe_atlas_qna_resources_server"),
        sandbox_provider={"apptainer": {}},
        concurrency=2,
    )
    base.update(over)
    return MiniSWEAgentQnaConfig(**base)


def _make_server(server_client: Any = None) -> MiniSWEAgentQna:
    if server_client is None:
        server_client = MagicMock(spec=ServerClient)
    return MiniSWEAgentQna(config=_make_config(), server_client=server_client)


def _make_run_request(**over: Any) -> MiniSWEAgentQnaRunRequest:
    metadata = over.pop(
        "verifier_metadata",
        {
            "instance_id": "task-1",
            "problem_statement": "How does X work?",
            "rubrics": [
                {"id": "r1", "title": "states X", "annotations": {"type": "positive", "importance": "must have"}}
            ],
            "docker_image": "ghcr.io/scaleapi/swe-atlas:img_1.0",
        },
    )
    return MiniSWEAgentQnaRunRequest(
        instance_id=over.pop("instance_id", "task-1"),
        verifier_metadata=metadata,
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[], temperature=over.pop("temperature", 0.7)
        ),
    )


def _verify_resp(payload: dict[str, Any]) -> Any:
    resp = AsyncMock()
    resp.ok = True
    resp.read = AsyncMock(return_value=json.dumps(payload).encode())
    return resp


class TestResolveImage:
    def test_template(self) -> None:
        img = _resolve_image("/sifs/{sif_basename}", {"sif_basename": "a.sif"})
        assert img == "/sifs/a.sif"

    def test_docker_image_fallback(self) -> None:
        assert _resolve_image(None, {"docker_image": "docker://x"}) == "docker://x"

    def test_missing_raises(self) -> None:
        try:
            _resolve_image(None, {})
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


class TestModelKwargs:
    def test_basic_params(self) -> None:
        kw = _responses_create_params_to_model_kwargs({"temperature": 0.5, "top_p": 0.9, "max_output_tokens": 128})
        assert kw["temperature"] == 0.5
        assert kw["top_p"] == 0.9
        assert kw["max_tokens"] == 128

    def test_extra_body_and_chat_template(self) -> None:
        kw = _responses_create_params_to_model_kwargs(
            {"metadata": {"extra_body": {"a": 1}, "chat_template_kwargs": {"enable_thinking": True}}}
        )
        assert kw["extra_body"]["a"] == 1
        assert kw["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}

    def test_empty(self) -> None:
        assert _responses_create_params_to_model_kwargs({}) == {}


class TestJsonDictFromMetadata:
    def test_none(self) -> None:
        assert _json_dict_from_metadata(None, field_name="x") == {}

    def test_dict(self) -> None:
        assert _json_dict_from_metadata({"a": 1}, field_name="x") == {"a": 1}

    def test_json_string(self) -> None:
        assert _json_dict_from_metadata('{"a": 1}', field_name="x") == {"a": 1}

    def test_non_dict_raises(self) -> None:
        for bad in (5, "[1, 2]"):
            try:
                _json_dict_from_metadata(bad, field_name="x")
                raise AssertionError("expected ValueError")
            except ValueError:
                pass


class TestTrajectoryHelpers:
    def test_message_content_to_text(self) -> None:
        assert _message_content_to_text("hi") == "hi"
        assert _message_content_to_text([{"text": "a"}, "b"]) == "a\nb"
        assert _message_content_to_text(None) == ""

    def test_strip_extra(self) -> None:
        assert _strip_extra({"a": 1, "extra": 2}) == {"a": 1}
        assert _strip_extra("raw")["content"] == "raw"

        class Dumpable:
            def model_dump(self) -> dict[str, Any]:
                return {"a": 1, "extra": 9}

        assert _strip_extra(Dumpable()) == {"a": 1}

    def test_split_trajectory(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"text": "problem"}]},
            {
                "role": "assistant",
                "content": "thinking",
                "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "out"},
            {"type": "function_call_output", "call_id": "c2", "output": "o", "extra": 1},
            {"object": "response", "output": [{"type": "message", "extra": 1}], "extra": 2},
        ]
        input_messages, output_items, responses = _split_trajectory_for_responses(messages)
        assert [m["role"] for m in input_messages] == ["system", "user"]
        types = [o.get("type") for o in output_items]
        assert "message" in types and "function_call" in types and "function_call_output" in types
        assert len(responses) == 1

    def test_default_response_and_answer_item(self) -> None:
        resp = _default_response_object()
        assert resp["object"] == "response"
        item = _answer_message_item("my answer")
        assert item["content"][0]["text"] == "my answer"


class TestVerifyAnswer:
    async def test_success(self) -> None:
        server = _make_server()
        server.server_client.post = AsyncMock(
            return_value=_verify_resp({"reward": 1.0, "agg_score": 0.8, "passed": True, "rubric_scores": []})
        )
        reward, extra = await server._verify_answer(_make_run_request(), "answer")
        assert reward == approx(1.0)
        assert extra["agg_score"] == 0.8
        assert extra["passed"] is True

    async def test_failure_scores_zero(self) -> None:
        server = _make_server()
        server.server_client.post = AsyncMock(side_effect=RuntimeError("verify down"))
        reward, extra = await server._verify_answer(_make_run_request(), "answer")
        assert reward == approx(0.0)
        assert extra == {}


class TestRun:
    def _patches(self, to_thread_result: Any):
        p_sc = patch(f"{MODULE}.ServerClient")
        p_gfscd = patch(f"{MODULE}.get_first_server_config_dict", return_value={"host": "0.0.0.0", "port": 8080})
        p_rpc = patch(f"{MODULE}.resolve_provider_config", return_value={"apptainer": {}})
        p_rpm = patch(f"{MODULE}.resolve_provider_metadata", return_value={})
        p_runner = patch(f"{MODULE}.runner_ray_remote")
        p_thread = patch(f"{MODULE}.asyncio.to_thread", new=AsyncMock(**to_thread_result))
        return p_sc, p_gfscd, p_rpc, p_rpm, p_runner, p_thread

    async def test_happy_path(self) -> None:
        server = _make_server()
        server.server_client.post = AsyncMock(
            return_value=_verify_resp({"reward": 1.0, "agg_score": 1.0, "passed": True})
        )
        result = {
            "task-1": {
                "answer": "the answer",
                "input_messages": [{"type": "message", "role": "user", "content": "q"}],
                "response_output": [_answer_message_item("the answer")],
                "responses": [],
                "exit_status": "Submitted",
            }
        }
        p_sc, p_gfscd, p_rpc, p_rpm, p_runner, p_thread = self._patches({"return_value": result})
        with p_sc as m_sc, p_gfscd, p_rpc, p_rpm, p_runner, p_thread:
            m_sc.load_from_global_config.return_value.global_config_dict = {"policy_model_name": "m"}
            resp = await server.run(_make_run_request())
        assert resp.reward == approx(1.0)
        assert resp.answer == "the answer"
        assert resp.agg_score == 1.0
        assert len(resp.responses_create_params.input) == 1

    async def test_run_error_still_verifies(self) -> None:
        server = _make_server()
        server.server_client.post = AsyncMock(return_value=_verify_resp({"reward": 0.0}))
        p_sc, p_gfscd, p_rpc, p_rpm, p_runner, p_thread = self._patches({"side_effect": RuntimeError("ray boom")})
        with p_sc as m_sc, p_gfscd, p_rpc, p_rpm, p_runner, p_thread:
            m_sc.load_from_global_config.return_value.global_config_dict = {"policy_model_name": "m"}
            resp = await server.run(_make_run_request())
        assert resp.reward == approx(0.0)
        assert resp.answer == ""
        assert resp.run_error is not None


class TestRunMiniSweQna:
    def test_runner(self, monkeypatch, tmp_path) -> None:
        holder: dict[str, Any] = {}

        class FakeEnv:
            def __init__(self, config: dict[str, Any]) -> None:
                self.config = config
                self.cleaned = False

            def execute(self, command: str, is_eval: bool = False) -> dict[str, Any]:
                assert command.startswith("cat ")
                return {"output": "<<FINAL_ANSWER>>\nreal\n<<FINAL_ANSWER>>", "returncode": 0}

            def cleanup(self) -> None:
                self.cleaned = True
                holder["cleaned"] = True

        class FakeAgent:
            def __init__(self, model: Any, env: FakeEnv, **agent_config: Any) -> None:
                holder["agent_config"] = agent_config

            def run(self, problem_statement: str) -> dict[str, Any]:
                assert problem_statement == "the question"
                return {"exit_status": "Submitted"}

            def save(self, path: Any, metadata: dict[str, Any]) -> dict[str, Any]:
                return {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "the question"},
                        {"role": "assistant", "content": "reasoning"},
                    ]
                }

        modules = {
            "minisweagent.agents": ModuleType("minisweagent.agents"),
            "minisweagent.agents.default": ModuleType("minisweagent.agents.default"),
            "minisweagent.environments": ModuleType("minisweagent.environments"),
            "minisweagent.models": ModuleType("minisweagent.models"),
        }
        modules["minisweagent.agents.default"].DefaultAgent = FakeAgent
        modules["minisweagent.environments"].get_environment = lambda config: FakeEnv(config)
        modules["minisweagent.models"].get_model = lambda config: SimpleNamespace(config=config)
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)

        config_path = tmp_path / "mswea_qa.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "agent": {"system_template": "s", "instance_template": "{{task}}", "step_limit": 5},
                    "model": {"model_kwargs": {"temperature": 1.0}},
                    "environment": {"cwd": "/app"},
                }
            )
        )
        monkeypatch.setattr(app_module, "get_config_path", lambda _c: config_path)

        params = dict(
            instance_dict={"instance_id": "task-1"},
            instance_id="task-1",
            problem_statement="the question",
            output=str(tmp_path / "out"),
            model="hosted_vllm/m",
            api_key="dummy_key",  # pragma: allowlist secret
            base_url="http://x/v1",
            config=str(config_path),
            image="docker://img",
            provider={"apptainer": {}},
            spec={},
            sandbox_environment_kwargs={"cwd": "/app", "user": "root"},
            model_kwargs={"temperature": 0.5},
            step_timeout=900,
            step_limit=5,
            answer_path="/logs/agent/answer.txt",
        )
        result = _run_mini_swe_qna(**params)["task-1"]
        assert result["answer"] == "<<FINAL_ANSWER>>\nreal\n<<FINAL_ANSWER>>"
        assert result["exit_status"] == "Submitted"
        assert [m["role"] for m in result["input_messages"]] == ["system", "user"]
        assert holder["cleaned"] is True

    def test_runner_missing_answer_file(self, monkeypatch, tmp_path) -> None:
        class FakeEnv:
            def __init__(self, config: dict[str, Any]) -> None:
                pass

            def execute(self, command: str, is_eval: bool = False) -> dict[str, Any]:
                return {"output": "no such file", "returncode": 1}

            def cleanup(self) -> None:
                pass

        class FakeAgent:
            def __init__(self, *a: Any, **k: Any) -> None:
                pass

            def run(self, _ps: str) -> dict[str, Any]:
                return {"exit_status": "LimitReached"}

            def save(self, _p: Any, _m: dict[str, Any]) -> dict[str, Any]:
                return {"messages": []}

        modules = {
            "minisweagent.agents": ModuleType("minisweagent.agents"),
            "minisweagent.agents.default": ModuleType("minisweagent.agents.default"),
            "minisweagent.environments": ModuleType("minisweagent.environments"),
            "minisweagent.models": ModuleType("minisweagent.models"),
        }
        modules["minisweagent.agents.default"].DefaultAgent = FakeAgent
        modules["minisweagent.environments"].get_environment = lambda config: FakeEnv(config)
        modules["minisweagent.models"].get_model = lambda config: SimpleNamespace()
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)

        config_path = tmp_path / "c.yaml"
        config_path.write_text(yaml.safe_dump({"agent": {}, "model": {}, "environment": {}}))
        monkeypatch.setattr(app_module, "get_config_path", lambda _c: config_path)

        params = dict(
            instance_dict={"instance_id": "t"},
            instance_id="t",
            problem_statement="q",
            output=str(tmp_path / "out"),
            model="m",
            api_key="k",
            base_url="u",
            config=str(config_path),
            image="img",
            provider={"apptainer": {}},
            spec={},
            sandbox_environment_kwargs=None,
            model_kwargs=None,
            step_timeout=1,
            step_limit=1,
            answer_path="/logs/agent/answer.txt",
        )
        result = _run_mini_swe_qna(**params)["t"]
        assert result["answer"] == ""


class TestMetrics:
    def test_score_fn(self) -> None:
        assert MiniSWEAgentQna._score_fn({"reward": 1.0, "agg_score": 0.5}) == {"pass": 1.0, "agg_score": 0.5}
        assert MiniSWEAgentQna._score_fn({}) == {"pass": 0.0, "agg_score": 0.0}

    def test_compute_and_key_metrics(self) -> None:
        server = _make_server()
        tasks = [
            [{"reward": 1.0, "agg_score": 1.0}],
            [{"reward": 0.0, "agg_score": 0.5, "run_error": "boom"}],
        ]
        metrics = server.compute_metrics(tasks)
        assert metrics["task_count"] == 2
        assert metrics["rollout_count"] == 2
        assert metrics["run_error_count"] == 1
        key = server.get_key_metrics({**metrics, "mean/reward": 0.5, "pass@1[avg-of-2]/pass": 0.5})
        assert key["task_count"] == 2
        assert key["run_error_count"] == 1
