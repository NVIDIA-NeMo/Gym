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
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from fastapi.testclient import TestClient

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.sandbox.observability import SandboxRecorder, use_recorder
from nemo_gym.server_utils import ServerClient
from responses_api_agents.mini_swe_agent import app as mini_swe_app_module
from responses_api_agents.mini_swe_agent.app import (
    MiniSWEAgent,
    MiniSWEAgentConfig,
    MiniSWEAgentRunRequest,
    MiniSWEAgentVerifyResponse,
    _agentic_router_program_id,
    _AutoToolRetryModel,
    _barrier_file_name,
    _is_missing_tool_call_error,
    _json_dict_from_metadata,
    _message_content_to_text,
    _ObservedModel,
    _responses_create_params_to_model_kwargs,
    _run_swegym_v2,
    _sandbox_spec_for_instance,
    _swebench_config_path,
    _swebench_image_name,
    _wait_for_sandbox_ready_barrier,
    run_swegym_with_optional_sandbox,
)


DEFAULT_RUN_SWEGYM_RESULT = {
    "test_instance_123": {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Fix this bug."},
            {"role": "assistant", "content": "I'll help you fix the bug."},
            {"role": "user", "content": "Thank you!"},
        ],
        "responses": [
            {
                "choices": [],
                "provider_specific_fields": {
                    "prompt_token_ids": [],
                    "generation_token_ids": [],
                    "generation_log_probs": [],
                },
            }
        ],
        "eval_report": {
            "eval_report": {
                "test_instance_123": {
                    "resolved": True,
                    "tests_status": {
                        "FAIL_TO_PASS": {"success": ["test1"], "failure": []},
                        "PASS_TO_PASS": {"success": ["test2"], "failure": []},
                    },
                }
            }
        },
    }
}

DEFAULT_CONFIG_YAML = """
model:
  model_kwargs:
    temperature: 0.5
    top_p: 0.8
"""

DEFAULT_CHAT_COMPLETION = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "test_model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
}


def create_test_config(
    host: str = "0.0.0.0",
    port: int = 8080,
    model_name: str = "test_model",
    env: str = "singularity",
    cache_dir_template: str = "/tmp/cache/gym.sif",
) -> MiniSWEAgentConfig:
    return MiniSWEAgentConfig(
        name="mini_swe_agent",
        host=host,
        port=port,
        entrypoint="",
        model_server=ModelServerRef(
            type="responses_api_models",
            name=model_name,
        ),
        env=env,
        concurrency=1,
        cache_dir_template=cache_dir_template,
    )


def setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict):
    mock_server_client_instance = MagicMock()
    mock_server_client_instance.global_config_dict = {"policy_model_name": "test_model"}
    mock_load_from_global_config.return_value = mock_server_client_instance

    mock_get_first_server_config_dict.return_value = {
        "host": "0.0.0.0",
        "port": 8080,
    }


def setup_config_path_mock(mock_get_config_path, config_yaml: str = DEFAULT_CONFIG_YAML):
    mock_config_path = MagicMock()
    mock_config_path.read_text.return_value = config_yaml
    mock_get_config_path.return_value = mock_config_path


def setup_run_swegym_mock(
    mock_to_thread,
    mock_runner_ray_remote,
    run_swegym_result: Dict[str, Any] = None,
):
    """Setup mock for Ray-based run_swegym execution"""
    if run_swegym_result is None:
        run_swegym_result = DEFAULT_RUN_SWEGYM_RESULT

    # Mock the Ray remote function to return a future-like object
    mock_future = MagicMock()
    mock_runner_ray_remote.remote.return_value = mock_future

    # Mock asyncio.to_thread (which calls ray.get) to return the result
    mock_to_thread.return_value = run_swegym_result


def create_run_request(
    instance_id: str = "test_instance_123",
    temperature: float = 0.5,
    top_p: float = 0.8,
    max_output_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
    subset: str = "gym",
    split: str = "train",
    input_data: list = None,
) -> MiniSWEAgentRunRequest:
    """Create a test run request with default values."""
    if input_data is None:
        input_data = []

    return MiniSWEAgentRunRequest(
        instance_id=instance_id,
        subset=subset,
        split=split,
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            metadata=metadata,
            input=input_data,
        ),
    )


def create_chat_completion_request(
    model: str = "test_model",
    messages: list = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> NeMoGymChatCompletionCreateParamsNonStreaming:
    if messages is None:
        messages = [{"role": "user", "content": "Hello!"}]

    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    return NeMoGymChatCompletionCreateParamsNonStreaming(**kwargs)


def assert_run_response(
    response: MiniSWEAgentVerifyResponse,
    expected_reward: float = 1.0,
    expected_temperature: float = 0.5,
    expected_top_p: float = 0.8,
    expected_input_length: int = 2,
):
    assert isinstance(response, MiniSWEAgentVerifyResponse)
    assert response.reward == expected_reward
    assert response.responses_create_params.temperature == expected_temperature
    assert response.responses_create_params.top_p == expected_top_p
    assert len(response.responses_create_params.input) == expected_input_length

    if expected_input_length >= 2:
        assert response.responses_create_params.input[0]["role"] == "system"
        assert response.responses_create_params.input[1]["role"] == "user"


def assert_run_swegym_called(
    mock_to_thread,
    subset: str = "gym",
    split: str = "train",
    instance_id: str = "test_instance_123",
):
    mock_to_thread.assert_called_once()
    call_args = mock_to_thread.call_args
    args = call_args[0]
    assert len(args) >= 1


def _otel_attributes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attrs = {}
    for row in rows:
        value = row["value"]
        if "stringValue" in value:
            attrs[row["key"]] = value["stringValue"]
        elif "boolValue" in value:
            attrs[row["key"]] = value["boolValue"]
        elif "intValue" in value:
            attrs[row["key"]] = int(value["intValue"])
        elif "doubleValue" in value:
            attrs[row["key"]] = value["doubleValue"]
    return attrs


def _otel_spans(output_dir: Path) -> list[dict[str, Any]]:
    trace_payload = json.loads((output_dir / "traces" / "otel_traces.json").read_text())
    return [
        span
        for resource_span in trace_payload["resourceSpans"]
        for scope_span in resource_span["scopeSpans"]
        for span in scope_span["spans"]
    ]


class FormatError(Exception):
    def __init__(self, content: str = "No tool calls found in the response.") -> None:
        self.messages = ({"role": "user", "content": content, "extra": {"interrupt_type": "FormatError"}},)
        super().__init__(content)


class _FakeModelConfig:
    def __init__(self, tool_choice: Any) -> None:
        self.model_kwargs = {"tool_choice": tool_choice}


class LitellmModel:
    __module__ = "minisweagent.models.litellm_model"

    def __init__(self, *, tool_choice: Any = "auto", error: Exception | None = None) -> None:
        self.config = _FakeModelConfig(tool_choice)
        self.calls = []
        self.error = error or FormatError()

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.calls.append(self.config.model_kwargs["tool_choice"])
        if len(self.calls) == 1:
            raise self.error
        return {"role": "assistant", "content": "", "extra": {"actions": [{"command": "pwd"}]}}


class TestApp:
    def test_sanity(self) -> None:
        config = create_test_config(model_name="", cache_dir_template="/")
        MiniSWEAgent(config=config, server_client=MagicMock(spec=ServerClient))

    def test_auto_tool_retry_uses_single_registered_tool_then_restores_auto(self) -> None:
        model = LitellmModel(tool_choice="auto")

        message = _AutoToolRetryModel(model).query([])

        assert message["extra"]["actions"] == [{"command": "pwd"}]
        assert model.calls == ["auto", {"type": "function", "function": {"name": "bash"}}]
        assert model.config.model_kwargs["tool_choice"] == "auto"

    def test_auto_tool_retry_does_not_override_explicit_tool_choice(self) -> None:
        model = LitellmModel(tool_choice={"type": "function", "function": {"name": "custom"}})

        with pytest.raises(FormatError):
            _AutoToolRetryModel(model).query([])

        assert model.calls == [{"type": "function", "function": {"name": "custom"}}]

    def test_observed_model_records_llm_span(self, tmp_path: Path) -> None:
        class QueryModel:
            def __init__(self) -> None:
                self.config = SimpleNamespace(model_kwargs={"temperature": 0.7, "top_p": 0.95, "max_tokens": 128})
                self.calls = []

            def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
                self.calls.append((messages, kwargs))
                return {"role": "assistant", "content": "ok"}

        recorder = SandboxRecorder(output_dir=tmp_path / "observability", otel={"enabled": False})
        model = QueryModel()
        with use_recorder(recorder):
            with mini_swe_app_module.event_context(trajectory_id="task-1"):
                assert _ObservedModel(model, model_name="hosted_vllm/qwen").query([{"role": "user", "content": "hi"}])
        recorder.finalize()

        spans = _otel_spans(recorder.output_dir)
        llm_span = next(span for span in spans if span["name"] == "llm.request")
        attrs = _otel_attributes(llm_span["attributes"])

        assert attrs["operation.name"] == "llm.request"
        assert attrs["span.section"] == "rollout"
        assert attrs["model"] == "hosted_vllm/qwen"
        assert attrs["message_count"] == 1

    def test_observed_model_records_auto_tool_retry_as_successful_llm_span(self, tmp_path: Path) -> None:
        model = LitellmModel(tool_choice="auto")
        observed = _ObservedModel(_AutoToolRetryModel(model), model_name="hosted_vllm/qwen")

        recorder = SandboxRecorder(output_dir=tmp_path / "observability", otel={"enabled": False})
        with use_recorder(recorder):
            with mini_swe_app_module.event_context(trajectory_id="task-1"):
                message = observed.query([{"role": "user", "content": "hi"}])
        recorder.finalize()

        spans = _otel_spans(recorder.output_dir)
        llm_span = next(span for span in spans if span["name"] == "llm.request")
        attrs = _otel_attributes(llm_span["attributes"])

        assert message["extra"]["actions"] == [{"command": "pwd"}]
        assert model.calls == ["auto", {"type": "function", "function": {"name": "bash"}}]
        assert attrs["status"] == "ok"
        assert not llm_span.get("events")

    def test_response_param_helpers_cover_metadata_and_tool_choice_modes(self) -> None:
        assert _json_dict_from_metadata(None, field_name="extra_body") == {}
        assert _json_dict_from_metadata({"top_k": 20}, field_name="extra_body") == {"top_k": 20}

        kwargs = _responses_create_params_to_model_kwargs(
            {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_output_tokens": 123,
                "metadata": {
                    "extra_body": json.dumps({"top_k": 20}),
                    "chat_template_kwargs": json.dumps({"enable_thinking": True}),
                },
                "tool_choice": {"type": "function", "function": {"name": "python"}},
            }
        )

        assert kwargs == {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 123,
            "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}},
            "tool_choice": {"type": "function", "function": {"name": "python"}},
        }
        assert _responses_create_params_to_model_kwargs({"tool_choice": "bash"})["tool_choice"] == {
            "type": "function",
            "function": {"name": "bash"},
        }
        assert (
            _responses_create_params_to_model_kwargs({"tool_choice": "auto"}, default_tool_choice="none")[
                "tool_choice"
            ]
            == "none"
        )

        with pytest.raises(ValueError, match="extra_body"):
            _json_dict_from_metadata("[]", field_name="extra_body")

    def test_auto_tool_retry_edge_cases_and_forwarded_attributes(self) -> None:
        assert _is_missing_tool_call_error(RuntimeError("No tool calls found")) is False
        assert _is_missing_tool_call_error(FormatError("Different format error")) is False
        non_dict_error = FormatError()
        non_dict_error.messages = ("not-a-dict",)
        assert _is_missing_tool_call_error(non_dict_error) is False
        wrong_interrupt_error = FormatError()
        wrong_interrupt_error.messages = ({"content": "No tool calls found", "extra": {"interrupt_type": "Other"}},)
        assert _is_missing_tool_call_error(wrong_interrupt_error) is False

        model = LitellmModel(tool_choice="auto")
        model.extra_attr = "forwarded"
        assert _AutoToolRetryModel(model).extra_attr == "forwarded"

        class OtherLitellmModel(LitellmModel):
            __module__ = "custom.model"

        with pytest.raises(FormatError):
            _AutoToolRetryModel(OtherLitellmModel(tool_choice="auto")).query([])

    def test_sandbox_resource_profiles_override_static_resources(self) -> None:
        spec = _sandbox_spec_for_instance(
            {"resources": {"cpu": "1", "memory": "8Gi", "ephemeral-storage": "20Gi"}},
            resource_profiles=[
                {"cpu": "250m", "memory": "3Gi", "ephemeral-storage": "1Gi"},
                {"cpu": "500m", "memory": "4Gi", "ephemeral-storage": "1Gi"},
            ],
            instance_id="django__django-12345",
        )

        assert spec["resources"] in (
            {"cpu": "250m", "memory": "3Gi", "ephemeral-storage": "1Gi"},
            {"cpu": "500m", "memory": "4Gi", "ephemeral-storage": "1Gi"},
        )
        assert _sandbox_spec_for_instance(None, resource_profiles=None, instance_id="task") == {}

    def test_misc_mini_swe_helpers(self, monkeypatch, tmp_path) -> None:
        assert _agentic_router_program_id("", "task-1") == "task-1"
        assert _agentic_router_program_id("mini", "mini:task-1") == "mini:task-1"
        assert _agentic_router_program_id("mini", "task-1") == "mini:task-1"
        assert _barrier_file_name("bad/value:with spaces") == "bad_value_with_spaces"
        assert _barrier_file_name("") == "unknown"
        assert _swebench_image_name({"instance_id": "django__django-1"}, "verified") == (
            "swebench/sweb.eval.x86_64.django_1776_django-1:latest"
        )
        assert _swebench_image_name({"instance_id": "django__django-1"}, "lite") == (
            "xingyaoww/sweb.eval.x86_64.django_s_django-1:latest"
        )
        assert _swebench_image_name({"instance_id": "x", "image_name": "custom:image"}, "verified") == "custom:image"
        assert _message_content_to_text("hello") == "hello"
        assert _message_content_to_text(None) == ""
        assert _message_content_to_text([{"text": "one"}, {"content": "two"}, 3]) == "one\ntwo\n3"

        builtin_dir = tmp_path / "configs"
        benchmark_dir = builtin_dir / "benchmarks"
        benchmark_dir.mkdir(parents=True)
        (benchmark_dir / "swebench.yaml").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(mini_swe_app_module, "builtin_config_dir", builtin_dir)
        assert _swebench_config_path() == benchmark_dir / "swebench.yaml"
        monkeypatch.setattr(mini_swe_app_module, "builtin_config_dir", tmp_path / "missing")
        assert _swebench_config_path() == tmp_path / "missing" / "extra" / "swebench.yaml"

    def test_sandbox_ready_barrier_waits_for_all_ready_files(self, tmp_path) -> None:
        barrier_dir = tmp_path / "_sandbox_ready_barriers" / "run"
        barrier_dir.mkdir(parents=True)
        (barrier_dir / "second.ready").write_text("{}")

        _wait_for_sandbox_ready_barrier(
            output_dir=tmp_path,
            barrier_id="run",
            instance_id="first",
            count=2,
            timeout_s=1.0,
            poll_s=0.1,
        )

        assert (barrier_dir / "first.ready").exists()

    def test_sandbox_ready_barrier_timeout(self, tmp_path) -> None:
        _wait_for_sandbox_ready_barrier(
            output_dir=tmp_path,
            barrier_id="run",
            instance_id="single",
            count=1,
            timeout_s=0,
            poll_s=0.1,
        )
        with pytest.raises(TimeoutError, match="Timed out waiting for sandbox-ready barrier"):
            _wait_for_sandbox_ready_barrier(
                output_dir=tmp_path,
                barrier_id="run",
                instance_id="first",
                count=2,
                timeout_s=0,
                poll_s=0.1,
            )

    def test_run_swegym_records_completion_and_errors(self, monkeypatch) -> None:
        monkeypatch.setattr(
            mini_swe_app_module,
            "run_swegym_v1",
            lambda **_params: {
                "task-1": {
                    "eval_report": {
                        "task-1": {"resolved": True},
                    }
                }
            },
        )
        monkeypatch.setattr(mini_swe_app_module.MiniSWEAgentUtils, "is_resolved", lambda *_args: True)
        assert run_swegym_with_optional_sandbox(env="docker", instance_id="task-1") == {
            "task-1": {"eval_report": {"task-1": {"resolved": True}}}
        }

        def fail_runner(**_params):
            raise RuntimeError("boom")

        monkeypatch.setattr(mini_swe_app_module, "run_swegym_v1", fail_runner)
        with pytest.raises(RuntimeError, match="boom"):
            run_swegym_with_optional_sandbox(env="docker", instance_id="task-1")

        env_module = ModuleType("minisweagent.environments")
        env_module.ENV_MAP = {}
        monkeypatch.setitem(sys.modules, "minisweagent.environments", env_module)
        monkeypatch.setattr(mini_swe_app_module, "run_swegym_v1", None)
        monkeypatch.setattr(
            mini_swe_app_module,
            "_run_swegym_v2",
            lambda **_params: {"task-1": {"eval_report": {"task-1": {"resolved": False}}}},
        )
        monkeypatch.setattr(mini_swe_app_module.MiniSWEAgentUtils, "is_resolved", lambda *_args: False)
        assert run_swegym_with_optional_sandbox(env="sandbox", instance_id="task-1") == {
            "task-1": {"eval_report": {"task-1": {"resolved": False}}}
        }
        assert env_module.ENV_MAP["sandbox"].__name__ == "MiniSWESandboxEnvironment"

        monkeypatch.setattr(mini_swe_app_module, "run_swegym_v1", lambda **_params: {"task-1": "bad"})
        assert run_swegym_with_optional_sandbox(env="docker", instance_id="task-1") == {"task-1": "bad"}

        monkeypatch.setattr(
            mini_swe_app_module,
            "run_swegym_v1",
            lambda **_params: {"task-1": {"eval_report": {"task-1": {"resolved": True}}}},
        )

        def raise_is_resolved(*_args: Any) -> bool:
            raise ValueError("bad report")

        monkeypatch.setattr(mini_swe_app_module.MiniSWEAgentUtils, "is_resolved", raise_is_resolved)
        assert run_swegym_with_optional_sandbox(env="docker", instance_id="task-1") == {
            "task-1": {"eval_report": {"task-1": {"resolved": True}}}
        }

    def test_run_swegym_v2_success_and_golden_paths(self, monkeypatch, tmp_path) -> None:
        holder: dict[str, Any] = {}

        class FakeLogger:
            def info(self, _message: str) -> None:
                return None

        def setup_logger(_instance_id: str, _log_file: Path) -> FakeLogger:
            return FakeLogger()

        def make_test_spec(instance: dict[str, Any]) -> SimpleNamespace:
            return SimpleNamespace(
                instance_id=instance["instance_id"],
                eval_script="#!/bin/bash\npytest -q",
            )

        def get_eval_report(*, test_spec: SimpleNamespace, prediction: dict[str, Any], log_path: Path, **_kwargs: Any):
            assert log_path.exists()
            return {test_spec.instance_id: {"resolved": True, "prediction": prediction}}

        class FakeEnv:
            def __init__(self, config: dict[str, Any]) -> None:
                self.config = config
                self.commands: list[tuple[str, bool]] = []
                self.cleaned = False

            def execute(self, command: str, is_eval: bool = False) -> dict[str, Any]:
                self.commands.append((command, is_eval))
                return {"output": "tests passed", "returncode": 0}

            def cleanup(self) -> None:
                self.cleaned = True

        class FakeAgent:
            def __init__(self, model: Any, env: FakeEnv, **agent_config: Any) -> None:
                self.model = model
                self.env = env
                self.agent_config = agent_config
                holder["agent_config"] = agent_config

            def run(self, problem_statement: str) -> dict[str, Any]:
                assert problem_statement == "Fix the bug"
                return {"exit_status": "submitted", "submission": "diff --git a/file b/file"}

            def save(self, path: Path | None, metadata: dict[str, Any]) -> dict[str, Any]:
                holder["save_path"] = path
                holder["save_metadata"] = metadata
                if path is not None:
                    path.write_text("{}", encoding="utf-8")
                return {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": [{"text": "problem"}]},
                        {
                            "role": "assistant",
                            "content": "answer",
                            "extra": {"response": {"id": "resp-1"}},
                        },
                        {"role": "tool", "content": "tool output"},
                    ]
                }

        def get_environment(config: dict[str, Any]) -> FakeEnv:
            env = FakeEnv(config)
            holder["env"] = env
            return env

        def get_model(config: dict[str, Any]) -> SimpleNamespace:
            holder["model_config"] = config
            return SimpleNamespace(config=config)

        module_specs = {
            "swegym": ModuleType("swegym"),
            "swegym.harness": ModuleType("swegym.harness"),
            "swegym.harness.constants": ModuleType("swegym.harness.constants"),
            "swegym.harness.docker_build": ModuleType("swegym.harness.docker_build"),
            "swegym.harness.grading": ModuleType("swegym.harness.grading"),
            "swegym.harness.test_spec": ModuleType("swegym.harness.test_spec"),
            "minisweagent.agents": ModuleType("minisweagent.agents"),
            "minisweagent.agents.default": ModuleType("minisweagent.agents.default"),
            "minisweagent.environments": ModuleType("minisweagent.environments"),
            "minisweagent.models": ModuleType("minisweagent.models"),
        }
        module_specs["swegym.harness.constants"].SWEbenchInstance = dict
        module_specs["swegym.harness.docker_build"].setup_logger = setup_logger
        module_specs["swegym.harness.grading"].get_eval_report = get_eval_report
        module_specs["swegym.harness.test_spec"].make_test_spec = make_test_spec
        module_specs["minisweagent.agents.default"].DefaultAgent = FakeAgent
        module_specs["minisweagent.environments"].get_environment = get_environment
        module_specs["minisweagent.models"].get_model = get_model
        for name, module in module_specs.items():
            monkeypatch.setitem(sys.modules, name, module)

        config_path = tmp_path / "swebench.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "model": {"model_kwargs": {"max_output_tokens": 99}},
                    "environment": {},
                    "agent": {"step_limit": 1, "collapse_limit": 3},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(mini_swe_app_module, "get_config_path", lambda _config: config_path)
        monkeypatch.setattr(mini_swe_app_module, "uuid4", lambda: "uuid")
        monkeypatch.setattr(mini_swe_app_module.time, "time", lambda: 1234)

        params = {
            "instance_dict": {
                "instance_id": "django__django-123",
                "problem_statement": "Fix the bug",
                "patch": "gold",
            },
            "instance_id": "django__django-123",
            "output": str(tmp_path / "out"),
            "config": "swebench",
            "model": "hosted/model",
            "api_key": "key",
            "base_url": "http://model/v1",
            "subset": "verified",
            "step_timeout": 30,
            "eval_timeout": 60,
            "env": "sandbox",
            "step_limit": 7,
            "auto_tool_retry": True,
            "run_golden": False,
        }

        result = _run_swegym_v2(**params)

        env = holder["env"]
        assert env.cleaned is True
        assert env.config["environment_class"].endswith("MiniSWESandboxEnvironment")
        assert env.config["image"] == "swebench/sweb.eval.x86_64.django_1776_django-123:latest"
        assert holder["model_config"]["model_kwargs"]["max_tokens"] == 99
        assert holder["agent_config"]["step_limit"] == 7
        assert holder["save_metadata"] == {"instance_id": "django__django-123"}
        assert result["django__django-123"]["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "problem"},
            {"role": "assistant", "content": "answer"},
        ]
        assert result["django__django-123"]["responses"] == [{"id": "resp-1"}]

        golden_params = params | {"env": "docker", "run_golden": True}
        result = _run_swegym_v2(**golden_params)

        env = holder["env"]
        assert env.cleaned is True
        assert env.config["environment_class"] == "docker"
        assert [command for command, _ in env.commands[:4]] == [
            "cat > patch.diff <<'EOF'\ngold\n\nEOF",
            "git status --porcelain",
            "git apply --check patch.diff",
            "git apply patch.diff",
        ]
        assert result["django__django-123"]["exit_status"] == "Gold Patch Applied"

        string_params = params | {
            "instance_dict": json.dumps(
                {"instance_id": "django__django-123", "problem_statement": "Fix the bug", "patch": "gold"}
            ),
            "sandbox_ready_barrier_id": "ready",
            "sandbox_ready_barrier_count": 1,
        }
        assert "django__django-123" in _run_swegym_v2(**string_params)

        with pytest.raises(ValueError, match="instance_dict"):
            _run_swegym_v2(**(params | {"instance_dict": None}))

    async def test_release_agentic_router_program_uses_model_server_endpoint(self, monkeypatch) -> None:
        calls: list[tuple[str, str, dict[str, Any]]] = []

        async def fake_request(method: str, url: str, *, json: dict[str, Any]) -> object:
            calls.append((method, url, json))
            return object()

        async def fake_raise_for_status(_response: object) -> None:
            return None

        async def fake_get_response_json(_response: object) -> dict[str, Any]:
            return {"released": True}

        monkeypatch.setattr(mini_swe_app_module, "server_request", fake_request)
        monkeypatch.setattr(mini_swe_app_module, "raise_for_status", fake_raise_for_status)
        monkeypatch.setattr(mini_swe_app_module, "get_response_json", fake_get_response_json)

        server = MiniSWEAgent(config=create_test_config(), server_client=MagicMock(spec=ServerClient))
        result = await server._release_agentic_router_program(
            {"host": "model-host", "port": 1234},
            "mini:task-1",
        )

        assert result == {"released": True}
        assert calls == [
            (
                "POST",
                "http://model-host:1234/agentic_router/release",
                {"program_id": "mini:task-1"},
            )
        ]

    @patch("responses_api_agents.mini_swe_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.mini_swe_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.mini_swe_agent.app.get_config_path")
    @patch("responses_api_agents.mini_swe_agent.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_successful_execution(
        self,
        mock_to_thread,
        mock_runner_ray_remote,
        mock_get_config_path,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
    ) -> None:
        """Test successful execution of the run method with mocked run_swegym."""

        config = create_test_config()
        mock_server_client = MagicMock(spec=ServerClient)
        server = MiniSWEAgent(config=config, server_client=mock_server_client)

        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        setup_config_path_mock(mock_get_config_path)
        setup_run_swegym_mock(mock_to_thread, mock_runner_ray_remote)

        run_request = create_run_request()

        response = await server.run(run_request)

        assert_run_response(response)

        assert_run_swegym_called(mock_to_thread)

    @patch("responses_api_agents.mini_swe_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.mini_swe_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.mini_swe_agent.app.get_config_path")
    @patch("responses_api_agents.mini_swe_agent.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_writes_generation_params_to_config(
        self,
        mock_to_thread,
        mock_runner_ray_remote,
        mock_get_config_path,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
        tmp_path,
        monkeypatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        config = create_test_config()
        config.tool_choice = "bash"
        mock_server_client = MagicMock(spec=ServerClient)
        server = MiniSWEAgent(config=config, server_client=mock_server_client)

        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        setup_config_path_mock(mock_get_config_path)
        setup_run_swegym_mock(mock_to_thread, mock_runner_ray_remote)

        run_request = create_run_request(
            temperature=0.6,
            top_p=0.95,
            max_output_tokens=49152,
            metadata={
                "extra_body": '{"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}',
                "chat_template_kwargs": '{"enable_thinking":true}',
            },
        )

        await server.run(run_request)

        call_args = mock_runner_ray_remote.options.return_value.remote.call_args
        params = call_args.args[1]
        generated_config = yaml.safe_load(Path(params["config"]).read_text())
        model_kwargs = generated_config["model"]["model_kwargs"]
        assert model_kwargs["temperature"] == 0.6
        assert model_kwargs["top_p"] == 0.95
        assert model_kwargs["max_tokens"] == 49152
        assert "max_output_tokens" not in model_kwargs
        assert model_kwargs["tool_choice"] == {"type": "function", "function": {"name": "bash"}}
        assert model_kwargs["extra_body"] == {
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": True},
        }

    @patch("responses_api_agents.mini_swe_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.mini_swe_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.mini_swe_agent.app.get_config_path")
    @patch("responses_api_agents.mini_swe_agent.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_writes_thunderagent_program_id_to_config(
        self,
        mock_to_thread,
        mock_runner_ray_remote,
        mock_get_config_path,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
        tmp_path,
        monkeypatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        config = create_test_config()
        config.agentic_router_program_id = True
        config.agentic_router_program_id_prefix = "mini_swe"
        config.agentic_router_release_program = False
        server = MiniSWEAgent(config=config, server_client=MagicMock(spec=ServerClient))

        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        setup_config_path_mock(mock_get_config_path)
        setup_run_swegym_mock(mock_to_thread, mock_runner_ray_remote)

        run_request = create_run_request(instance_id="django__django-12345")

        await server.run(run_request)

        call_args = mock_runner_ray_remote.options.return_value.remote.call_args
        params = call_args.args[1]
        generated_config = yaml.safe_load(Path(params["config"]).read_text())
        assert generated_config["model"]["model_kwargs"]["extra_body"]["program_id"] == (
            "mini_swe:django__django-12345"
        )

    @patch("responses_api_agents.mini_swe_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.mini_swe_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.mini_swe_agent.app.get_config_path")
    @patch("responses_api_agents.mini_swe_agent.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_defaults_to_auto_tool_choice(
        self,
        mock_to_thread,
        mock_runner_ray_remote,
        mock_get_config_path,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
        tmp_path,
        monkeypatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        config = create_test_config()
        server = MiniSWEAgent(config=config, server_client=MagicMock(spec=ServerClient))

        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        setup_config_path_mock(mock_get_config_path)
        setup_run_swegym_mock(mock_to_thread, mock_runner_ray_remote)

        await server.run(create_run_request())

        call_args = mock_runner_ray_remote.options.return_value.remote.call_args
        params = call_args.args[1]
        generated_config = yaml.safe_load(Path(params["config"]).read_text())
        assert generated_config["model"]["model_kwargs"]["tool_choice"] == "auto"
        assert params["auto_tool_retry"] is False

    @patch("responses_api_agents.mini_swe_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.mini_swe_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.mini_swe_agent.app.get_config_path")
    @patch("responses_api_agents.mini_swe_agent.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_releases_thunderagent_program(
        self,
        mock_to_thread,
        mock_runner_ray_remote,
        mock_get_config_path,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
        tmp_path,
        monkeypatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        config = create_test_config()
        config.agentic_router_program_id = True
        config.agentic_router_program_id_prefix = "mini_swe"
        server = MiniSWEAgent(config=config, server_client=MagicMock(spec=ServerClient))
        server._release_agentic_router_program = AsyncMock(return_value={"released": True})

        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        setup_config_path_mock(mock_get_config_path)
        setup_run_swegym_mock(mock_to_thread, mock_runner_ray_remote)

        run_request = create_run_request(instance_id="django__django-12345")

        response = await server.run(run_request)

        server._release_agentic_router_program.assert_awaited_once_with(
            model_server_config={"host": "0.0.0.0", "port": 8080},
            program_id="mini_swe:django__django-12345",
        )
        assert response.metadata["agentic_router_release"] == {"released": True}

    @patch("responses_api_agents.mini_swe_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.mini_swe_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.mini_swe_agent.app.get_config_path")
    @patch("responses_api_agents.mini_swe_agent.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_failed_execution(
        self,
        mock_to_thread,
        mock_runner_ray_remote,
        mock_get_config_path,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
    ) -> None:
        """Test run method when run_swegym fails."""

        config = create_test_config(env="docker")
        mock_server_client = MagicMock(spec=ServerClient)
        server = MiniSWEAgent(config=config, server_client=mock_server_client)

        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        setup_config_path_mock(mock_get_config_path)

        # Mock Ray remote function
        mock_future = MagicMock()
        mock_runner_ray_remote.remote.return_value = mock_future

        # Mock asyncio.to_thread (ray.get) to raise an exception
        mock_to_thread.side_effect = Exception("run_swegym failed")

        run_request = create_run_request(instance_id="test_instance_456", temperature=0.3, top_p=0.95)

        response = await server.run(run_request)

        assert_run_response(
            response,
            expected_reward=0.0,
            expected_temperature=0.3,
            expected_top_p=0.95,
            expected_input_length=0,
        )

        assert_run_swegym_called(mock_to_thread, instance_id="test_instance_456")

    @patch("responses_api_agents.mini_swe_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.mini_swe_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.mini_swe_agent.app.get_config_path")
    @patch("responses_api_agents.mini_swe_agent.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_swegym_not_found(
        self,
        mock_to_thread,
        mock_runner_ray_remote,
        mock_get_config_path,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
    ) -> None:
        config = create_test_config(env="docker")
        mock_server_client = MagicMock(spec=ServerClient)
        server = MiniSWEAgent(config=config, server_client=mock_server_client)

        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        setup_config_path_mock(mock_get_config_path)

        # Mock Ray remote function
        mock_future = MagicMock()
        mock_runner_ray_remote.remote.return_value = mock_future

        # Mock asyncio.to_thread (ray.get) to raise FileNotFoundError
        mock_to_thread.side_effect = FileNotFoundError("run_swegym not found")

        run_request = create_run_request(instance_id="test_instance_789", temperature=0.2, top_p=1.0)

        response = await server.run(run_request)

        assert_run_response(
            response,
            expected_reward=0.0,
            expected_temperature=0.2,
            expected_top_p=1.0,
            expected_input_length=0,
        )

        assert_run_swegym_called(mock_to_thread, instance_id="test_instance_789")

    async def test_responses_not_implemented(self) -> None:
        config = create_test_config(env="docker")
        mock_server_client = MagicMock(spec=ServerClient)
        server = MiniSWEAgent(config=config, server_client=mock_server_client)

        request_body = NeMoGymResponseCreateParamsNonStreaming(temperature=0.7, top_p=0.9, input=[])

        with pytest.raises(NotImplementedError):
            await server.responses(request_body)

    def test_endpoints_registration(self) -> None:
        config = create_test_config(env="docker")
        mock_server_client = MagicMock(spec=ServerClient)
        server = MiniSWEAgent(config=config, server_client=mock_server_client)

        app = server.setup_webserver()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post("/v1/responses", json={"temperature": 0.7, "top_p": 0.9, "input": []})
        assert response.status_code == 500

        run_response = client.post("/run", json={})
        assert run_response.status_code != 404

        aggregate_response = client.post("/aggregate_metrics", json={"verify_responses": []})
        assert aggregate_response.status_code == 200
