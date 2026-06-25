# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the OSWorld rollout client.

The fake env and fake agent keep these tests independent from OSWorld,
Docker, QEMU, and model servers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from responses_api_agents.osworld_agent import client as osworld_client


class FakeController:
    def start_recording(self) -> None:
        return None

    def end_recording(self, _path: str) -> None:
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
        return response, [{"action_type": "DONE"}]


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


def _patch_client_for_fake_runtime(monkeypatch) -> None:
    FakeEnv.instances.clear()
    FakePromptAgent.call_llm_responses.clear()
    FakePointerAgent.instances.clear()

    def fake_load_attr(import_path: str):
        if import_path == "fake.FakeEnv":
            return FakeEnv
        if import_path == "fake.FakePromptAgent":
            return FakePromptAgent
        if import_path == "fake.FakePointerAgent":
            return FakePointerAgent
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

    assert escaped.format(CLIENT_PASSWORD="pw") == """Password: pw
{
    "action_type": "click",
    "x": 1
}
"""


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
        policy_api_key="test-key",
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
