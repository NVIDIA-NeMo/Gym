# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the OSWorld rollout client.

The fake env and fake agent keep these tests independent from OSWorld,
Docker, QEMU, and model servers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List

import pytest

from responses_api_agents.osworld_agent import client as osworld_client


class FakeController:
    def __init__(self) -> None:
        self.started = 0
        self.ended_paths: List[str] = []

    def start_recording(self) -> None:
        self.started += 1
        return None

    def end_recording(self, path: str) -> None:
        self.ended_paths.append(path)
        return None


class FakeEnv:
    instances: List["FakeEnv"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.controller = FakeController()
        self.vm_ip = "127.0.0.1"
        self.actions: List[Any] = []
        FakeEnv.instances.append(self)

    def reset(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        self.task_config = task_config
        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        return {"screenshot": b"not-black", "accessibility_tree": "<tree />"}

    def step(self, action: Any, _pause: float):
        self.actions.append(action)
        done = action == "DONE" or (isinstance(action, dict) and action.get("action_type") == "DONE")
        return self._get_obs(), 1.0 if done else 0.0, done, {"done": done}

    def evaluate(self) -> float:
        return 1.0 if self.actions else 0.0

    def close(self) -> None:
        self.closed = True


class FakePromptAgent:
    call_llm_responses: List[str] = []
    next_actions: List[Any] = [{"action_type": "DONE"}]

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.reset_calls = []

    def reset(self, *_args: Any, **kwargs: Any) -> None:
        self.reset_calls.append(kwargs)

    def predict(self, instruction: str, obs: Dict[str, Any]):
        response = self.call_llm(
            {
                "model": self.kwargs["model"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                        ],
                    }
                ],
                "max_tokens": self.kwargs["max_tokens"],
                "temperature": self.kwargs["temperature"],
                "top_p": self.kwargs["top_p"],
            }
        )
        FakePromptAgent.call_llm_responses.append(response)
        assert obs["screenshot"] == b"not-black"
        return response, list(self.next_actions)


class FakePointerAgent:
    instances: List["FakePointerAgent"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.reset_calls = []
        self.predict_calls = 0
        self.log_usage_calls = 0
        FakePointerAgent.instances.append(self)

    def reset(self, instruction: str, task_logger: Any, task_results_dir: str) -> None:
        self.reset_calls.append(
            {
                "instruction": instruction,
                "task_logger": task_logger,
                "task_results_dir": task_results_dir,
            }
        )

    def predict(self, obs: Dict[str, Any]):
        self.predict_calls += 1
        assert obs["screenshot"] == b"not-black"
        return "Pointer response", [{"action_type": "DONE"}]

    def log_usage(self) -> None:
        self.log_usage_calls += 1


class FakeM3Agent:
    instances: List["FakeM3Agent"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.reset_calls = []
        self.api_log_dirs: List[str] = []
        self.predict_calls = 0
        FakeM3Agent.instances.append(self)

    def reset(self, *_args: Any, **kwargs: Any) -> None:
        self.reset_calls.append(kwargs)

    def set_api_log_dir(self, path: str) -> None:
        self.api_log_dirs.append(path)

    def predict(self, instruction: str, obs: Dict[str, Any]):
        self.predict_calls += 1
        assert instruction == "Use official M3Agent."
        assert obs["screenshot"] == b"not-black"
        return "M3 response", ["DONE"]


class FakePointerEnv(FakeEnv):
    def evaluate(self, eval_logger: Any) -> float:
        assert eval_logger is not None
        return 1.0 if self.actions else 0.0


def _patch_client_for_fake_runtime(monkeypatch) -> None:
    FakeEnv.instances.clear()
    FakePromptAgent.call_llm_responses.clear()
    FakePromptAgent.next_actions = [{"action_type": "DONE"}]
    FakePointerAgent.instances.clear()
    FakeM3Agent.instances.clear()

    def fake_load_attr(import_path: str):
        if import_path == "fake.FakeEnv":
            return FakeEnv
        if import_path == "fake.FakePointerEnv":
            return FakePointerEnv
        if import_path == "fake.FakePromptAgent":
            return FakePromptAgent
        if import_path == "fake.FakePointerAgent":
            return FakePointerAgent
        if import_path == "fake.FakeM3Agent":
            return FakeM3Agent
        raise AssertionError(f"unexpected import path: {import_path}")

    monkeypatch.setattr(osworld_client, "load_attr", fake_load_attr)
    monkeypatch.setattr(osworld_client.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("OSWORLD_COLD_BOOT_MIN_PNG_BYTES", "1")


def test_prompt_agent_template_escape_preserves_json_and_password_placeholder() -> None:
    template = """Password: {CLIENT_PASSWORD}
{
    "action_type": "click",
    "x": 1
}
"""

    escaped = osworld_client._escape_prompt_agent_format_template(template)

    assert (
        escaped.format(CLIENT_PASSWORD="pw")  # pragma: allowlist secret
        == """Password: pw
{
    "action_type": "click",
    "x": 1
}
"""
    )


def test_prompt_agent_computer_13_action_normalization() -> None:
    assert osworld_client._normalize_prompt_agent_computer_13_action(
        {"action_type": "LEFT_CLICK", "x": 753, "varies": 45}
    ) == {"action_type": "CLICK", "parameters": {"x": 753, "y": 45, "button": "left"}}

    assert osworld_client._normalize_prompt_agent_computer_13_action(
        {"action_type": "TYPE", "parameters": {"text": "hello"}}
    ) == {"action_type": "TYPING", "parameters": {"text": "hello"}}

    assert osworld_client._normalize_prompt_agent_computer_13_action(
        {"action_type": "CLICK", "parameters": {"click_type": "RIGHT", "x": 1, "y": 2}}
    ) == {"action_type": "CLICK", "parameters": {"x": 1, "y": 2, "button": "right"}}

    assert osworld_client._normalize_prompt_agent_computer_13_action(
        {"action_type": "TRIPLE_CLICK", "parameters": {"x": 1, "y": 2}}
    ) == {"action_type": "CLICK", "parameters": {"x": 1, "y": 2, "button": "left", "num_clicks": 3}}


def test_m3_native_tool_use_is_translated_to_upstream_text_protocol() -> None:
    raw_response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="Inspect the screenshot."),
            SimpleNamespace(
                type="tool_use",
                name="computer",
                input={"action": "left_click", "coordinate": [516, 71]},
            ),
            SimpleNamespace(type="text", text="Action: Click the address bar."),
        ]
    )

    class Agent:
        def _call_llm(self, _messages):
            return "Action: Click the address bar.", raw_response

    agent = Agent()
    osworld_client._patch_m3_native_tool_use(agent)
    osworld_client._patch_m3_native_tool_use(agent)

    text, returned_response = agent._call_llm([])

    assert returned_response is raw_response
    assert text.count("<tool_call>") == 1
    assert '"name": "computer"' in text
    assert '"arguments": {"action": "left_click", "coordinate": [516, 71]}' in text


def test_gym_policy_runner_preserves_existing_pyautogui_flow(monkeypatch) -> None:
    _patch_client_for_fake_runtime(monkeypatch)

    def model_fn(_system: str, instruction: str, history: List[Dict[str, Any]]) -> str:
        assert instruction == "Finish the task."
        assert len(history) == 1
        return "```DONE```"

    result = osworld_client.run_osworld_task(
        {"id": "task-1", "instruction": "Finish the task."},
        model_fn=model_fn,
        env_class_path="fake.FakeEnv",
        sleep_after_execution=0,
        task_timeout=10,
    )

    assert result.reward == 1.0
    assert result.finished is True
    assert result.steps[0].actions == ["DONE"]
    assert FakeEnv.instances[0].kwargs["action_space"] == "pyautogui"
    assert FakeEnv.instances[0].actions == ["DONE"]


def test_recording_can_be_limited_to_selected_task_ids(monkeypatch, tmp_path: Path) -> None:
    _patch_client_for_fake_runtime(monkeypatch)
    monkeypatch.setenv("OSWORLD_RECORD_VIDEO_DIR", str(tmp_path / "videos"))
    selected_path = tmp_path / "selected_task_ids.txt"
    selected_path.write_text("record-me\n", encoding="utf-8")
    monkeypatch.setenv("OSWORLD_RECORD_VIDEO_TASK_IDS_FILE", str(selected_path))

    def model_fn(_system: str, _instruction: str, _history: List[Dict[str, Any]]) -> str:
        return "```DONE```"

    selected = osworld_client.run_osworld_task(
        {"id": "record-me", "instruction": "Finish the task."},
        model_fn=model_fn,
        env_class_path="fake.FakeEnv",
        sleep_after_execution=0,
        task_timeout=10,
    )
    skipped = osworld_client.run_osworld_task(
        {"id": "skip-me", "instruction": "Finish the task."},
        model_fn=model_fn,
        env_class_path="fake.FakeEnv",
        sleep_after_execution=0,
        task_timeout=10,
    )

    assert selected.reward == 1.0
    assert skipped.reward == 1.0
    assert FakeEnv.instances[0].controller.started == 1
    assert FakeEnv.instances[0].controller.ended_paths == [str(tmp_path / "videos" / "record-me.mp4")]
    assert FakeEnv.instances[1].controller.started == 0
    assert FakeEnv.instances[1].controller.ended_paths == []


def test_prompt_agent_runner_routes_native_agent_messages_to_policy_model(monkeypatch) -> None:
    _patch_client_for_fake_runtime(monkeypatch)
    calls: List[Dict[str, Any]] = []

    def model_fn(_system: str, _instruction: str, _history: List[Dict[str, Any]]) -> str:
        raise AssertionError("prompt_agent runner should use messages_model_fn")

    def messages_model_fn(messages: List[Dict[str, Any]], payload: Dict[str, Any]) -> str:
        calls.append({"messages": messages, "payload": payload})
        return "native prompt agent response"

    result = osworld_client.run_osworld_task(
        {"id": "task-2", "instruction": "Use native PromptAgent."},
        model_fn=model_fn,
        runner_name="prompt_agent_computer_13",
        env_class_path="fake.FakeEnv",
        agent_class_path="fake.FakePromptAgent",
        messages_model_fn=messages_model_fn,
        policy_model_name="policy-under-test",
        policy_max_tokens=123,
        policy_temperature=0.4,
        policy_top_p=None,
        sleep_after_execution=0,
        task_timeout=10,
    )

    assert result.reward == 1.0
    assert result.finished is True
    assert result.steps[0].model_text == "native prompt agent response"
    assert result.steps[0].actions == [{"action_type": "DONE"}]
    assert FakeEnv.instances[0].kwargs["action_space"] == "computer_13"
    assert FakeEnv.instances[0].actions == [{"action_type": "DONE"}]
    assert calls[0]["payload"]["model"] == "policy-under-test"
    assert calls[0]["payload"]["max_tokens"] == 123
    assert calls[0]["payload"]["temperature"] == 0.4
    assert calls[0]["payload"]["top_p"] is None


def test_prompt_agent_runner_normalizes_computer_13_actions(monkeypatch) -> None:
    _patch_client_for_fake_runtime(monkeypatch)
    FakePromptAgent.next_actions = [{"action_type": "LEFT_CLICK", "x": 753, "varies": 45}]

    result = osworld_client.run_osworld_task(
        {"id": "task-normalize", "instruction": "Normalize native PromptAgent action."},
        model_fn=lambda *_args: "unused",
        runner_name="prompt_agent_computer_13",
        env_class_path="fake.FakeEnv",
        agent_class_path="fake.FakePromptAgent",
        messages_model_fn=lambda _messages, _payload: "native response",
        max_steps=1,
        sleep_after_execution=0,
        task_timeout=10,
    )

    assert result.error is None
    assert result.steps[0].actions == [{"action_type": "CLICK", "parameters": {"x": 753, "y": 45, "button": "left"}}]
    assert FakeEnv.instances[0].actions == [
        {"action_type": "CLICK", "parameters": {"x": 753, "y": 45, "button": "left"}}
    ]


def test_prompt_agent_runner_strips_thinking_before_native_agent_parse(monkeypatch) -> None:
    _patch_client_for_fake_runtime(monkeypatch)

    result = osworld_client.run_osworld_task(
        {"id": "task-thinking", "instruction": "Strip thinking before native parse."},
        model_fn=lambda *_args: "unused",
        runner_name="prompt_agent_computer_13",
        env_class_path="fake.FakeEnv",
        agent_class_path="fake.FakePromptAgent",
        messages_model_fn=lambda _messages, _payload: "<think>private reasoning</think>```DONE```",
        sleep_after_execution=0,
        task_timeout=10,
    )

    assert result.reward == 1.0
    assert result.steps[0].model_text == "```DONE```"
    assert FakePromptAgent.call_llm_responses == ["```DONE```"]


@pytest.mark.parametrize(
    "runner_name",
    [
        "prompt_agent",
        "prompt_agent_a11y_tree_pyautogui",
        "prompt_agent_som_pyautogui",
    ],
)
def test_prompt_agent_runners_requiring_a11y_enable_env_a11y_tree(monkeypatch, runner_name: str) -> None:
    _patch_client_for_fake_runtime(monkeypatch)

    result = osworld_client.run_osworld_task(
        {"id": "task-a11y", "instruction": "Use native PromptAgent with richer observations."},
        model_fn=lambda *_args: "unused",
        runner_name=runner_name,
        env_class_path="fake.FakeEnv",
        agent_class_path="fake.FakePromptAgent",
        messages_model_fn=lambda _messages, _payload: "native response",
        sleep_after_execution=0,
        task_timeout=10,
    )

    assert result.reward == 1.0
    assert FakeEnv.instances[0].kwargs["require_a11y_tree"] is True


def test_pointer_agent_runner_uses_native_pointer_predict_loop(monkeypatch, tmp_path) -> None:
    _patch_client_for_fake_runtime(monkeypatch)
    monkeypatch.setenv("OSWORLD_POINTER_RESULTS_DIR", str(tmp_path))

    result = osworld_client.run_osworld_task(
        {"id": "task-pointer", "instruction": "Use Pointer."},
        model_fn=lambda *_args: (_ for _ in ()).throw(AssertionError("pointer_agent should not use model_fn")),
        runner_name="pointer_agent",
        env_class_path="fake.FakeEnv",
        agent_class_path="fake.FakePointerAgent",
        agent_kwargs={"provider_name": "anthropic"},
        policy_base_url="https://inference-api.nvidia.com",
        policy_api_key="test-key",  # pragma: allowlist secret
        policy_model_name="azure/anthropic/claude-opus-4-7",
        sleep_after_execution=0,
        task_timeout=10,
    )

    assert result.reward == 1.0
    assert result.finished is True
    assert result.steps[0].model_text == "Pointer response"
    assert result.steps[0].actions == [{"action_type": "DONE"}]
    assert FakeEnv.instances[0].kwargs["action_space"] == "pyautogui"
    assert FakeEnv.instances[0].actions == [{"action_type": "DONE"}]
    pointer = FakePointerAgent.instances[0]
    assert pointer.kwargs["env"] is FakeEnv.instances[0]
    assert pointer.kwargs["provider_name"] == "anthropic"
    assert pointer.reset_calls[0]["instruction"] == "Use Pointer."
    assert pointer.predict_calls == 1
    assert pointer.log_usage_calls == 1
    assert (Path(pointer.reset_calls[0]["task_results_dir"]) / "pointer.log").exists()


def test_m3_agent_runner_uses_messages_endpoint_and_native_predict_loop(monkeypatch, tmp_path) -> None:
    _patch_client_for_fake_runtime(monkeypatch)
    monkeypatch.setenv("OSWORLD_M3_RESULTS_DIR", str(tmp_path))

    result = osworld_client.run_osworld_task(
        {"id": "task-m3", "instruction": "Use official M3Agent."},
        model_fn=lambda *_args: (_ for _ in ()).throw(AssertionError("m3_agent should not use model_fn")),
        runner_name="m3_agent",
        env_class_path="fake.FakeEnv",
        agent_class_path="fake.FakeM3Agent",
        policy_base_url="https://inference-api.nvidia.com/v1/messages",
        policy_api_key="test-key",  # pragma: allowlist secret
        policy_model_name="nvidia/minimaxai/minimax-m3",
        policy_max_tokens=8192,
        policy_temperature=0.6,
        policy_top_p=None,
        max_trajectory_length=10,
        sleep_after_execution=0,
        task_timeout=10,
    )

    assert result.reward == 1.0
    assert result.finished is True
    assert result.steps[0].model_text == "M3 response"
    assert result.steps[0].actions == ["DONE"]
    assert FakeEnv.instances[0].kwargs["action_space"] == "pyautogui"
    assert FakeEnv.instances[0].actions == ["DONE"]

    agent = FakeM3Agent.instances[0]
    assert agent.kwargs == {
        "platform": "ubuntu",
        "model": "nvidia/minimaxai/minimax-m3",
        "max_tokens": 8192,
        "top_p": None,
        "temperature": 0.6,
        "action_space": "pyautogui",
        "observation_type": "screenshot",
        "max_trajectory_length": 10,
        "client_password": "password",  # pragma: allowlist secret
        "base_url": "https://inference-api.nvidia.com",
        "api_key": "test-key",  # pragma: allowlist secret
    }
    assert agent.predict_calls == 1
    assert len(agent.api_log_dirs) == 1
    assert Path(agent.api_log_dirs[0]).name == "api_logs"
    assert Path(agent.api_log_dirs[0]).parent.name.endswith("-task-m3")
    assert Path(agent.api_log_dirs[0]).is_relative_to(tmp_path)


def test_pointer_agent_runner_sets_optional_parallel_placeholder_when_key_missing(monkeypatch, tmp_path) -> None:
    _patch_client_for_fake_runtime(monkeypatch)
    monkeypatch.setenv("OSWORLD_POINTER_RESULTS_DIR", str(tmp_path))
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    result = osworld_client.run_osworld_task(
        {"id": "task-pointer-ih", "instruction": "Use Pointer through InferenceHub."},
        model_fn=lambda *_args: (_ for _ in ()).throw(AssertionError("pointer_agent should not use model_fn")),
        runner_name="pointer_agent",
        env_class_path="fake.FakeEnv",
        agent_class_path="fake.FakePointerAgent",
        policy_base_url="https://inference-api.nvidia.com/v1",
        policy_api_key="test-key",
        policy_model_name="azure/anthropic/claude-opus-4-7",
        sleep_after_execution=0,
        task_timeout=10,
    )

    assert result.reward == 1.0
    assert result.finished is True
    assert osworld_client.os.environ["ANTHROPIC_API_KEY"] == "test-key"
    assert osworld_client.os.environ["ANTHROPIC_BASE_URL"] == "https://inference-api.nvidia.com"
    assert (
        osworld_client.os.environ["PARALLEL_API_KEY"]
        == "__nemo_gym_parallel_tools_disabled__"  # pragma: allowlist secret
    )
    pointer = FakePointerAgent.instances[0]
    assert pointer.kwargs["provider_name"] == "anthropic"
    assert "disable_parallel_tools" not in pointer.kwargs
    assert "use_policy_endpoint" not in pointer.kwargs


def test_pointer_optional_parallel_patch_removes_web_tools(monkeypatch) -> None:
    class Tool:
        def __init__(self, name: str) -> None:
            self.schema = {"name": name}

    mm_agents = ModuleType("mm_agents")
    pointer_pkg = ModuleType("mm_agents.pointer")
    gate_module = ModuleType("mm_agents.pointer.agent_feasibility_gate")
    planner_module = ModuleType("mm_agents.pointer.agent_planner")

    gate_module.GATE_TOOLS = [Tool("probe_bash"), Tool("web_search"), Tool("web_fetch")]
    gate_module._TOOL_DISPATCH = {"probe_bash": object(), "web_search": object(), "web_fetch": object()}
    planner_module.PLANNER_TOOLS = [Tool("submit_plan"), Tool("web_search")]
    planner_module._TOOL_DISPATCH = {"submit_plan": object(), "web_search": object()}
    pointer_pkg.agent_feasibility_gate = gate_module
    pointer_pkg.agent_planner = planner_module

    monkeypatch.setitem(sys.modules, "mm_agents", mm_agents)
    monkeypatch.setitem(sys.modules, "mm_agents.pointer", pointer_pkg)
    monkeypatch.setitem(sys.modules, "mm_agents.pointer.agent_feasibility_gate", gate_module)
    monkeypatch.setitem(sys.modules, "mm_agents.pointer.agent_planner", planner_module)

    osworld_client._patch_pointer_optional_parallel_tools(disable_parallel_tools=True)

    assert [tool.schema["name"] for tool in gate_module.GATE_TOOLS] == ["probe_bash"]
    assert gate_module._TOOL_DISPATCH == {"probe_bash": gate_module._TOOL_DISPATCH["probe_bash"]}
    assert [tool.schema["name"] for tool in planner_module.PLANNER_TOOLS] == ["submit_plan"]
    assert planner_module._TOOL_DISPATCH == {"submit_plan": planner_module._TOOL_DISPATCH["submit_plan"]}


def test_pointer_anthropic_client_options_are_configurable(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    monkeypatch.setenv("POINTER_ANTHROPIC_MAX_RETRIES", "6")
    monkeypatch.setenv("POINTER_ANTHROPIC_TIMEOUT_SECONDS", "45.5")

    assert osworld_client._pointer_anthropic_client_options(
        "https://inference-api.nvidia.com/",
        api_key="client-key",  # pragma: allowlist secret
    ) == {
        "api_key": "client-key",
        "base_url": "https://inference-api.nvidia.com",
        "max_retries": 6,
        "timeout": 45.5,
    }

    assert (
        osworld_client._pointer_anthropic_client_options("https://inference-api.nvidia.com")["api_key"]
        == "env-key"  # pragma: allowlist secret
    )


def test_pointer_config_sync_uses_anthropic_provider_for_policy_endpoint(monkeypatch) -> None:
    class FakeAPIProvider:
        ANTHROPIC = "anthropic"
        BEDROCK = "bedrock"

    class FakePointerConfig:
        provider = FakeAPIProvider.BEDROCK
        executor_model = "claude-opus-4-7"
        gate_model = "claude-sonnet-4-6"
        planner_model = "claude-sonnet-4-6"
        verifier_model = "claude-sonnet-4-6"
        summarization_model = "claude-haiku-4-5"

    config_module = ModuleType("mm_agents.pointer.config")
    utils_module = ModuleType("mm_agents.pointer.utils")
    config_module.config = FakePointerConfig()
    utils_module.APIProvider = FakeAPIProvider

    monkeypatch.setitem(sys.modules, "mm_agents.pointer.config", config_module)
    monkeypatch.setitem(sys.modules, "mm_agents.pointer.utils", utils_module)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://inference-api.nvidia.com")
    monkeypatch.setenv("POINTER_GATE_AGENT_MODEL", "azure/anthropic/claude-sonnet-4-6")
    monkeypatch.setenv("POINTER_PLANNER_AGENT_MODEL", "azure/anthropic/claude-sonnet-4-6")
    monkeypatch.setenv("POINTER_VERIFIER_AGENT_MODEL", "azure/anthropic/claude-sonnet-4-6")
    monkeypatch.setenv("POINTER_SUMMARIZATION_MODEL", "azure/anthropic/claude-haiku-4-5")

    osworld_client._sync_pointer_config("azure/anthropic/claude-opus-4-7")

    pointer_config = config_module.config
    assert pointer_config.provider == FakeAPIProvider.ANTHROPIC
    assert pointer_config.executor_model == "azure/anthropic/claude-opus-4-7"
    assert pointer_config.gate_model == "azure/anthropic/claude-sonnet-4-6"
    assert pointer_config.planner_model == "azure/anthropic/claude-sonnet-4-6"
    assert pointer_config.verifier_model == "azure/anthropic/claude-sonnet-4-6"
    assert pointer_config.summarization_model == "azure/anthropic/claude-haiku-4-5"


def test_pointer_env_evaluate_receives_eval_logger(monkeypatch, tmp_path) -> None:
    _patch_client_for_fake_runtime(monkeypatch)
    monkeypatch.setenv("OSWORLD_POINTER_RESULTS_DIR", str(tmp_path))

    result = osworld_client.run_osworld_task(
        {"id": "task-pointer-eval", "instruction": "Evaluate through Pointer env."},
        model_fn=lambda *_args: (_ for _ in ()).throw(AssertionError("pointer_agent should not use model_fn")),
        runner_name="pointer_agent",
        env_class_path="fake.FakePointerEnv",
        agent_class_path="fake.FakePointerAgent",
        policy_base_url="https://inference-api.nvidia.com/v1",
        policy_api_key="test-key",
        policy_model_name="azure/anthropic/claude-opus-4-7",
        sleep_after_execution=0,
        task_timeout=10,
    )

    assert result.reward == 1.0
    assert result.score == 1.0
    assert result.error is None
