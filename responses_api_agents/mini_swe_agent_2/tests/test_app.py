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
from unittest.mock import MagicMock, patch

import pytest
import yaml
from fastapi.testclient import TestClient

from nemo_gym.config_types import AggregateMetricsRequest, ModelServerRef
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient


try:
    __import__("minisweagent.config")
except ModuleNotFoundError as exc:
    if exc.name not in {"minisweagent", "minisweagent.config"}:
        raise
    minisweagent_module = ModuleType("minisweagent")
    minisweagent_module.__path__ = []
    minisweagent_config_module = ModuleType("minisweagent.config")
    minisweagent_config_module.builtin_config_dir = Path("/tmp/minisweagent/config")
    minisweagent_config_module.get_config_path = Path
    sys.modules["minisweagent"] = minisweagent_module
    sys.modules["minisweagent.config"] = minisweagent_config_module

from responses_api_agents.mini_swe_agent_2 import app as mini_swe_app_module
from responses_api_agents.mini_swe_agent_2.app import (
    OPENSANDBOX_API_KEY_ENV,
    MiniSWEAgent,
    MiniSWEAgentConfig,
    MiniSWEAgentRunRequest,
    MiniSWEAgentVerifyResponse,
    _is_resolved,
    _json_dict_from_metadata,
    _message_content_to_text,
    _responses_create_params_to_model_kwargs,
    _run_mini_swe_v2,
    _sandbox_provider_for_config_dump,
    _sandbox_runtime_env,
    _sandbox_spec_for_instance,
    _split_trajectory_for_responses,
    _swebench_config_path,
    _swebench_image_name,
    run_mini_swe_with_sandbox,
)


DEFAULT_RUN_MINI_SWE_RESULT = {
    "test_instance_123": {
        "input_messages": [
            {"type": "message", "role": "system", "content": "You are a helpful assistant."},
            {"type": "message", "role": "user", "content": "Fix this bug."},
        ],
        "response_output": [
            {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "I'll help you fix the bug.", "annotations": []}],
            }
        ],
        "responses": [
            {
                "id": "resp-1",
                "object": "response",
                "output": [],
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
) -> MiniSWEAgentConfig:
    return MiniSWEAgentConfig(
        name="mini_swe_agent_2",
        host=host,
        port=port,
        entrypoint="",
        model_server=ModelServerRef(
            type="responses_api_models",
            name=model_name,
        ),
        env="sandbox",
        concurrency=1,
        sandbox_provider={"opensandbox": {}},
        sandbox_spec={},
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


def setup_run_mini_swe_mock(
    mock_to_thread,
    mock_runner_ray_remote,
    run_mini_swe_result: Dict[str, Any] = None,
):
    """Setup mock for Ray-based run_mini_swe execution"""
    if run_mini_swe_result is None:
        run_mini_swe_result = DEFAULT_RUN_MINI_SWE_RESULT

    # Mock the Ray remote function to return a future-like object
    mock_future = MagicMock()
    mock_runner_ray_remote.remote.return_value = mock_future
    mock_runner_ray_remote.options.return_value.remote.return_value = mock_future

    # Mock asyncio.to_thread (which calls ray.get) to return the result
    mock_to_thread.return_value = run_mini_swe_result


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


def assert_run_mini_swe_called(
    mock_to_thread,
    subset: str = "gym",
    split: str = "train",
    instance_id: str = "test_instance_123",
):
    mock_to_thread.assert_called_once()
    call_args = mock_to_thread.call_args
    args = call_args[0]
    assert len(args) >= 1


class TestApp:
    def test_sanity(self) -> None:
        config = create_test_config(model_name="")
        MiniSWEAgent(config=config, server_client=MagicMock(spec=ServerClient))

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

    def test_sandbox_resource_profiles_override_static_resources(self) -> None:
        spec = _sandbox_spec_for_instance(
            {"resources": {"cpu": 1, "memory_mib": 8192, "disk_gib": 20}},
            resource_profiles=[
                {"cpu": 0.25, "memory_mib": 3072, "disk_gib": 1},
                {"cpu": 0.5, "memory_mib": 4096, "disk_gib": 1},
            ],
            instance_id="django__django-12345",
        )

        assert spec["resources"] in (
            {"cpu": 0.25, "memory_mib": 3072, "disk_gib": 1},
            {"cpu": 0.5, "memory_mib": 4096, "disk_gib": 1},
        )
        assert _sandbox_spec_for_instance(None, resource_profiles=None, instance_id="task") == {}

    def test_sandbox_provider_config_dump_strips_api_key(self) -> None:
        provider = {
            "opensandbox": {
                "connection": {
                    "domain": "sandbox.example",
                    "api_key": "fixture-value",  # pragma: allowlist secret
                }
            }
        }

        provider_for_disk = _sandbox_provider_for_config_dump(provider)
        assert "api_key" not in provider_for_disk["opensandbox"]["connection"]
        assert provider["opensandbox"]["connection"]["api_key"] == "fixture-value"  # pragma: allowlist secret
        assert _sandbox_runtime_env(provider)["env_vars"] == {
            OPENSANDBOX_API_KEY_ENV: "fixture-value"  # pragma: allowlist secret
        }

    def test_split_trajectory_and_resolution_helpers_cover_edge_cases(self) -> None:
        input_messages, output_items, raw_responses = _split_trajectory_for_responses(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "user"},
                {
                    "role": "assistant",
                    "content": "answer",
                    "tool_calls": [{"id": "call-1", "function": {"name": "bash", "arguments": '{"command":"pwd"}'}}],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "tool output"},
                {"type": "function_call_output", "call_id": "call-2", "output": "raw", "extra": {"ignored": True}},
                {"object": "response", "output": [{"type": "message", "content": "raw"}], "extra": {"ignored": True}},
            ]
        )

        assert input_messages == [
            {"type": "message", "role": "system", "content": "sys"},
            {"type": "message", "role": "user", "content": "user"},
        ]
        assert any(item["type"] == "function_call" and item["call_id"] == "call-1" for item in output_items)
        assert any(item["type"] == "function_call_output" and item["call_id"] == "call-1" for item in output_items)
        assert any(item["type"] == "function_call_output" and item["call_id"] == "call-2" for item in output_items)
        assert raw_responses == [{"object": "response", "output": [{"type": "message", "content": "raw"}]}]

        assert not _is_resolved("task", {})
        assert not _is_resolved("task", {"eval_report": {"task": {"resolved": True}}})
        assert not _is_resolved(
            "task",
            {
                "eval_report": {
                    "task": {
                        "resolved": True,
                        "tests_status": {"FAIL_TO_PASS": {"success": [], "failure": []}},
                    }
                }
            },
        )

    def test_misc_mini_swe_helpers(self, monkeypatch, tmp_path) -> None:
        assert _swebench_image_name({"instance_id": "django__django-1"}, "verified") == (
            "docker.io/swebench/sweb.eval.x86_64.django_1776_django-1:latest"
        )
        assert _swebench_image_name({"instance_id": "django__django-1"}, "lite") == (
            "docker.io/xingyaoww/sweb.eval.x86_64.django_s_django-1:latest"
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

    def test_run_mini_swe_records_completion_and_errors(self, monkeypatch) -> None:
        monkeypatch.setattr(
            mini_swe_app_module,
            "_run_mini_swe_v2",
            lambda **_params: {
                "task-1": {
                    "eval_report": {
                        "task-1": {"resolved": True},
                    }
                }
            },
        )
        assert run_mini_swe_with_sandbox(
            env="sandbox",
            instance_id="task-1",
        ) == {"task-1": {"eval_report": {"task-1": {"resolved": True}}}}

        def fail_runner(**_params):
            raise RuntimeError("boom")

        monkeypatch.setattr(mini_swe_app_module, "_run_mini_swe_v2", fail_runner)
        with pytest.raises(RuntimeError, match="boom"):
            run_mini_swe_with_sandbox(env="sandbox", instance_id="task-1")

        monkeypatch.setattr(
            mini_swe_app_module,
            "_run_mini_swe_v2",
            lambda **_params: {"task-1": {"eval_report": {"task-1": {"resolved": False}}}},
        )
        assert run_mini_swe_with_sandbox(env="sandbox", instance_id="task-1") == {
            "task-1": {"eval_report": {"task-1": {"resolved": False}}}
        }

        monkeypatch.setattr(mini_swe_app_module, "_run_mini_swe_v2", lambda **_params: {"task-1": "bad"})
        assert run_mini_swe_with_sandbox(env="sandbox", instance_id="task-1") == {"task-1": "bad"}

    def test_run_mini_swe_v2_success_and_golden_paths(self, monkeypatch, tmp_path) -> None:
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

        def get_eval_report(
            *,
            test_spec: SimpleNamespace,
            prediction: dict[str, Any],
            test_log_path: str,
            **_kwargs: Any,
        ):
            assert Path(test_log_path).exists()
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
                            "id": "resp-1",
                            "object": "response",
                            "output": [
                                {
                                    "id": "msg-1",
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "completed",
                                    "content": [{"type": "output_text", "text": "answer", "annotations": []}],
                                },
                                {
                                    "type": "function_call",
                                    "name": "bash",
                                    "call_id": "call-1",
                                    "arguments": json.dumps({"command": "echo hi"}),
                                },
                            ],
                            "extra": {"actions": [{"command": "echo hi", "tool_call_id": "call-1"}]},
                        },
                        {
                            "type": "function_call_output",
                            "call_id": "call-1",
                            "output": "tool output",
                            "extra": {"raw_output": "tool output"},
                        },
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
            "swebench": ModuleType("swebench"),
            "swebench.harness": ModuleType("swebench.harness"),
            "swebench.harness.constants": ModuleType("swebench.harness.constants"),
            "swebench.harness.docker_build": ModuleType("swebench.harness.docker_build"),
            "swebench.harness.grading": ModuleType("swebench.harness.grading"),
            "swebench.harness.test_spec": ModuleType("swebench.harness.test_spec"),
            "swebench.harness.test_spec.test_spec": ModuleType("swebench.harness.test_spec.test_spec"),
            "minisweagent.agents": ModuleType("minisweagent.agents"),
            "minisweagent.agents.default": ModuleType("minisweagent.agents.default"),
            "minisweagent.environments": ModuleType("minisweagent.environments"),
            "minisweagent.models": ModuleType("minisweagent.models"),
        }
        module_specs["swebench.harness.constants"].SWEbenchInstance = dict
        module_specs["swebench.harness.docker_build"].setup_logger = setup_logger
        module_specs["swebench.harness.grading"].get_eval_report = get_eval_report
        module_specs["swebench.harness.test_spec.test_spec"].make_test_spec = make_test_spec
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
                    "environment": {"provider": {"opensandbox": {"connection": {}}}},
                    "agent": {"step_limit": 1, "collapse_limit": 3},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(mini_swe_app_module, "get_config_path", lambda _config: config_path)
        monkeypatch.setattr(mini_swe_app_module, "uuid4", lambda: "uuid")
        monkeypatch.setattr(mini_swe_app_module.time, "time", lambda: 1234)
        monkeypatch.setenv(OPENSANDBOX_API_KEY_ENV, "worker-value")  # pragma: allowlist secret

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
            "api_key": "key",  # pragma: allowlist secret
            "base_url": "http://model/v1",
            "subset": "verified",
            "step_timeout": 30,
            "eval_timeout": 60,
            "env": "sandbox",
            "step_limit": 7,
            "run_golden": False,
        }

        result = _run_mini_swe_v2(**params)

        env = holder["env"]
        assert env.cleaned is True
        assert env.config["environment_class"].endswith("MiniSWESandboxEnvironment")
        assert (
            env.config["provider"]["opensandbox"]["connection"]["api_key"]
            == "worker-value"  # pragma: allowlist secret
        )
        assert env.config["image"] == "docker.io/swebench/sweb.eval.x86_64.django_1776_django-123:latest"
        assert holder["model_config"]["model_class"] == "litellm"
        assert holder["model_config"]["model_name"] == "hosted/model"
        assert holder["model_config"]["model_kwargs"]["max_tokens"] == 99
        assert holder["model_config"]["model_kwargs"]["base_url"] == "http://model/v1"
        assert "api_base" not in holder["model_config"]["model_kwargs"]
        assert holder["agent_config"]["step_limit"] == 7
        assert holder["save_metadata"] == {"instance_id": "django__django-123"}
        assert result["django__django-123"]["input_messages"] == [
            {"type": "message", "role": "system", "content": "sys"},
            {"type": "message", "role": "user", "content": "problem"},
        ]
        assert result["django__django-123"]["response_output"] == [
            {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "answer", "annotations": []}],
            },
            {
                "type": "function_call",
                "name": "bash",
                "call_id": "call-1",
                "arguments": json.dumps({"command": "echo hi"}),
            },
            {"type": "function_call_output", "call_id": "call-1", "output": "tool output"},
        ]
        assert result["django__django-123"]["responses"] == [
            {
                "id": "resp-1",
                "object": "response",
                "output": [
                    {
                        "id": "msg-1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "answer", "annotations": []}],
                    },
                    {
                        "type": "function_call",
                        "name": "bash",
                        "call_id": "call-1",
                        "arguments": json.dumps({"command": "echo hi"}),
                    },
                ],
            }
        ]

        golden_params = params | {"run_golden": True}
        result = _run_mini_swe_v2(**golden_params)

        env = holder["env"]
        assert env.cleaned is True
        assert env.config["environment_class"].endswith("MiniSWESandboxEnvironment")
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
        }
        assert "django__django-123" in _run_mini_swe_v2(**string_params)

        with pytest.raises(ValueError, match="instance_dict"):
            _run_mini_swe_v2(**(params | {"instance_dict": None}))

    @patch("responses_api_agents.mini_swe_agent_2.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.mini_swe_agent_2.app.get_first_server_config_dict")
    @patch("responses_api_agents.mini_swe_agent_2.app.get_config_path")
    @patch("responses_api_agents.mini_swe_agent_2.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_successful_execution(
        self,
        mock_to_thread,
        mock_runner_ray_remote,
        mock_get_config_path,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
    ) -> None:
        """Test successful execution of the run method with mocked run_mini_swe."""

        config = create_test_config()
        mock_server_client = MagicMock(spec=ServerClient)
        server = MiniSWEAgent(config=config, server_client=mock_server_client)

        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        setup_config_path_mock(mock_get_config_path)
        setup_run_mini_swe_mock(mock_to_thread, mock_runner_ray_remote)

        run_request = create_run_request()

        response = await server.run(run_request)

        assert_run_response(response)

        assert_run_mini_swe_called(mock_to_thread)

    @patch("responses_api_agents.mini_swe_agent_2.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.mini_swe_agent_2.app.get_first_server_config_dict")
    @patch("responses_api_agents.mini_swe_agent_2.app.get_config_path")
    @patch("responses_api_agents.mini_swe_agent_2.app.runner_ray_remote")
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
        config.sandbox_provider = {
            "opensandbox": {
                "connection": {
                    "domain": "sandbox.example",
                    "api_key": "fixture-value",  # pragma: allowlist secret
                }
            }
        }
        mock_server_client = MagicMock(spec=ServerClient)
        server = MiniSWEAgent(config=config, server_client=mock_server_client)

        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        setup_config_path_mock(mock_get_config_path)
        setup_run_mini_swe_mock(mock_to_thread, mock_runner_ray_remote)

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

        runtime_env = mock_runner_ray_remote.options.call_args.kwargs["runtime_env"]
        assert runtime_env["env_vars"] == {OPENSANDBOX_API_KEY_ENV: "fixture-value"}  # pragma: allowlist secret
        call_args = mock_runner_ray_remote.options.return_value.remote.call_args
        params = call_args.args[1]
        generated_config = yaml.safe_load(Path(params["config"]).read_text())
        assert "api_key" not in generated_config["environment"]["provider"]["opensandbox"]["connection"]
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

    @patch("responses_api_agents.mini_swe_agent_2.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.mini_swe_agent_2.app.get_first_server_config_dict")
    @patch("responses_api_agents.mini_swe_agent_2.app.get_config_path")
    @patch("responses_api_agents.mini_swe_agent_2.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_resolves_named_sandbox_provider_reference(
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
        config.sandbox_provider = "sandbox"
        mock_server_client = MagicMock(spec=ServerClient)
        server = MiniSWEAgent(config=config, server_client=mock_server_client)

        mock_server_client_instance = MagicMock()
        mock_server_client_instance.global_config_dict = {
            "policy_model_name": "test_model",
            "sandbox": {
                "opensandbox": {
                    "connection": {
                        "domain": "sandbox.example",
                        "api_key": "fixture-value",  # pragma: allowlist secret
                    }
                }
            },
        }
        mock_load_from_global_config.return_value = mock_server_client_instance
        mock_get_first_server_config_dict.return_value = {"host": "0.0.0.0", "port": 8080}
        setup_config_path_mock(mock_get_config_path)
        setup_run_mini_swe_mock(mock_to_thread, mock_runner_ray_remote)

        await server.run(create_run_request())

        runtime_env = mock_runner_ray_remote.options.call_args.kwargs["runtime_env"]
        assert runtime_env["env_vars"] == {OPENSANDBOX_API_KEY_ENV: "fixture-value"}  # pragma: allowlist secret
        call_args = mock_runner_ray_remote.options.return_value.remote.call_args
        params = call_args.args[1]
        generated_config = yaml.safe_load(Path(params["config"]).read_text())
        provider = generated_config["environment"]["provider"]["opensandbox"]
        assert provider["connection"]["domain"] == "sandbox.example"
        assert "api_key" not in provider["connection"]

    @patch("responses_api_agents.mini_swe_agent_2.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.mini_swe_agent_2.app.get_first_server_config_dict")
    @patch("responses_api_agents.mini_swe_agent_2.app.get_config_path")
    @patch("responses_api_agents.mini_swe_agent_2.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_failed_execution(
        self,
        mock_to_thread,
        mock_runner_ray_remote,
        mock_get_config_path,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
    ) -> None:
        """Test run method when run_mini_swe fails."""

        config = create_test_config()
        mock_server_client = MagicMock(spec=ServerClient)
        server = MiniSWEAgent(config=config, server_client=mock_server_client)

        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        setup_config_path_mock(mock_get_config_path)

        # Mock Ray remote function
        mock_future = MagicMock()
        mock_runner_ray_remote.remote.return_value = mock_future

        # Mock asyncio.to_thread (ray.get) to raise an exception
        mock_to_thread.side_effect = Exception("run_mini_swe failed")

        run_request = create_run_request(instance_id="test_instance_456", temperature=0.3, top_p=0.95)

        response = await server.run(run_request)

        assert_run_response(
            response,
            expected_reward=0.0,
            expected_temperature=0.3,
            expected_top_p=0.95,
            expected_input_length=0,
        )

        assert_run_mini_swe_called(mock_to_thread, instance_id="test_instance_456")

    @patch("responses_api_agents.mini_swe_agent_2.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.mini_swe_agent_2.app.get_first_server_config_dict")
    @patch("responses_api_agents.mini_swe_agent_2.app.get_config_path")
    @patch("responses_api_agents.mini_swe_agent_2.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_mini_swe_not_found(
        self,
        mock_to_thread,
        mock_runner_ray_remote,
        mock_get_config_path,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
    ) -> None:
        config = create_test_config()
        mock_server_client = MagicMock(spec=ServerClient)
        server = MiniSWEAgent(config=config, server_client=mock_server_client)

        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        setup_config_path_mock(mock_get_config_path)

        # Mock Ray remote function
        mock_future = MagicMock()
        mock_runner_ray_remote.remote.return_value = mock_future

        # Mock asyncio.to_thread (ray.get) to raise FileNotFoundError
        mock_to_thread.side_effect = FileNotFoundError("run_mini_swe not found")

        run_request = create_run_request(instance_id="test_instance_789", temperature=0.2, top_p=1.0)

        response = await server.run(run_request)

        assert_run_response(
            response,
            expected_reward=0.0,
            expected_temperature=0.2,
            expected_top_p=1.0,
            expected_input_length=0,
        )

        assert_run_mini_swe_called(mock_to_thread, instance_id="test_instance_789")

    async def test_responses_not_implemented(self) -> None:
        config = create_test_config()
        mock_server_client = MagicMock(spec=ServerClient)
        server = MiniSWEAgent(config=config, server_client=mock_server_client)

        request_body = NeMoGymResponseCreateParamsNonStreaming(temperature=0.7, top_p=0.9, input=[])

        with pytest.raises(NotImplementedError):
            await server.responses(request_body)

    async def test_aggregate_metrics_includes_eval_results(self) -> None:
        config = create_test_config()
        mock_server_client = MagicMock(spec=ServerClient)
        server = MiniSWEAgent(config=config, server_client=mock_server_client)

        responses = [
            {
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "instance_id": "task-a",
                "reward": 1.0,
                "metadata": {
                    "instance_id": "task-a",
                    "eval_report": {
                        "task-a": {
                            "resolved": True,
                            "patch_successfully_applied": True,
                            "tests_status": {
                                "FAIL_TO_PASS": {"success": ["test-a"], "failure": []},
                                "PASS_TO_PASS": {"success": ["test-b"], "failure": []},
                            },
                        }
                    },
                },
            },
            {
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 1,
                "instance_id": "task-a",
                "reward": 0.0,
                "metadata": {
                    "instance_id": "task-a",
                    "eval_report": {
                        "task-a": {
                            "resolved": False,
                            "patch_successfully_applied": True,
                            "tests_status": {
                                "FAIL_TO_PASS": {"success": [], "failure": ["test-a"]},
                                "PASS_TO_PASS": {"success": ["test-b"], "failure": []},
                            },
                        }
                    },
                },
            },
            {
                TASK_INDEX_KEY_NAME: 1,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "instance_id": "task-b",
                "reward": 0.0,
                "metadata": {"error": "boom"},
            },
            {
                TASK_INDEX_KEY_NAME: 1,
                ROLLOUT_INDEX_KEY_NAME: 1,
                "instance_id": "task-b",
                "reward": 0.0,
                "metadata": {"error": "boom"},
            },
        ]

        result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=responses))

        assert result.agent_metrics["pass@2/accuracy"] == pytest.approx(50.0)
        assert result.agent_metrics["resolved_task_count"] == 1
        assert result.agent_metrics["eval_error_rollout_count"] == 2
        assert result.agent_metrics["tests_status/fail_to_pass_success"] == 1
        assert result.key_metrics["pass@2/accuracy"] == pytest.approx(50.0)

        groups = {group[TASK_INDEX_KEY_NAME]: group for group in result.group_level_metrics}
        assert groups[0]["instance_id"] == "task-a"
        assert groups[0]["resolved"] is True
        assert groups[0]["tests_status_rollout_count"] == 2
        assert groups[1]["instance_id"] == "task-b"
        assert groups[1]["eval_error_rollout_count"] == 2

    def test_endpoints_registration(self) -> None:
        config = create_test_config()
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
