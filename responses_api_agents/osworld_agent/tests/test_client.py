# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the OSWorld rollout client.

The fake env and fake agent keep these tests independent from OSWorld,
Docker, QEMU, and model servers.
"""

from __future__ import annotations

import json
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


class FakeNemotronAgent:
    instances: List["FakeNemotronAgent"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.reset_calls = 0
        FakeNemotronAgent.instances.append(self)

    def reset(self, *_args: Any, **_kwargs: Any) -> None:
        self.reset_calls += 1

    def predict(self, instruction: str, obs: Dict[str, Any]):
        response = self.call_llm(
            {
                "model": self.kwargs["model"],
                "messages": [{"role": "user", "content": instruction}],
                "_nemo_gym_return_message": True,
            },
            self.kwargs["model"],
        )
        assert obs["screenshot"] == b"not-black"
        return response["content"], ["DONE"], {"thought": response["reasoning_content"]}


class FakeQwenAgent:
    instances: List["FakeQwenAgent"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.reset_calls = 0
        FakeQwenAgent.instances.append(self)

    def reset(self, *_args: Any, **_kwargs: Any) -> None:
        self.reset_calls += 1

    def predict(self, instruction: str, obs: Dict[str, Any]):
        response = self.call_llm(
            {
                "model": self.kwargs["model"],
                "messages": [{"role": "user", "content": instruction}],
            },
            self.kwargs["model"],
        )
        assert obs["screenshot"] == b"not-black"
        return response, ["pyautogui.click(1, 2)", "pyautogui.press('enter')", "DONE"]


class FakePointerEnv(FakeEnv):
    def evaluate(self, eval_logger: Any) -> float:
        assert eval_logger is not None
        return 1.0 if self.actions else 0.0


class FakeSetupScoreZeroEnv(FakeEnv):
    def reset(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        raise osworld_client._EvaluatorScoreZero("declared zero")


def _patch_client_for_fake_runtime(monkeypatch) -> None:
    FakeEnv.instances.clear()
    FakePromptAgent.call_llm_responses.clear()
    FakePromptAgent.next_actions = [{"action_type": "DONE"}]
    FakePointerAgent.instances.clear()
    FakeM3Agent.instances.clear()
    FakeNemotronAgent.instances.clear()
    FakeQwenAgent.instances.clear()

    def fake_load_attr(import_path: str):
        if import_path == "fake.FakeEnv":
            return FakeEnv
        if import_path == "fake.FakePointerEnv":
            return FakePointerEnv
        if import_path == "fake.FakeSetupScoreZeroEnv":
            return FakeSetupScoreZeroEnv
        if import_path == "fake.FakePromptAgent":
            return FakePromptAgent
        if import_path == "fake.FakePointerAgent":
            return FakePointerAgent
        if import_path == "fake.FakeM3Agent":
            return FakeM3Agent
        if import_path == "fake.FakeNemotronAgent":
            return FakeNemotronAgent
        if import_path == "fake.FakeQwenAgent":
            return FakeQwenAgent
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
    assert FakeEnv.instances[0].kwargs["enable_proxy"] is False
    assert FakeEnv.instances[0].actions == ["DONE"]


def test_proxy_required_task_is_masked_without_starting_an_environment(monkeypatch) -> None:
    _patch_client_for_fake_runtime(monkeypatch)

    result = osworld_client.run_osworld_task(
        {"id": "proxy-disabled", "instruction": "Open the website.", "proxy": True},
        model_fn=lambda *_args: pytest.fail("model must not run"),
        env_class_path="fake.FakeEnv",
        enable_proxy=False,
    )

    assert result.reward == 0.0
    assert result.mask_sample is True
    assert result.termination_reason == "proxy_required_but_disabled"
    assert "ProxyRequiredButDisabled" in (result.error or "")
    assert FakeEnv.instances == []


def test_proxy_required_task_passes_enablement_and_config_to_osworld(monkeypatch, tmp_path: Path) -> None:
    _patch_client_for_fake_runtime(monkeypatch)
    proxy_path = tmp_path / "proxy.json"
    proxy_path.write_text('[{"host":"proxy.example.com","port":3128}]\n', encoding="utf-8")

    result = osworld_client.run_osworld_task(
        {"id": "proxy-enabled", "instruction": "Open the website.", "proxy": True},
        model_fn=lambda *_args: "```DONE```",
        env_class_path="fake.FakeEnv",
        enable_proxy=True,
        proxy_config_file=str(proxy_path),
        sleep_after_execution=0,
        task_timeout=10,
    )

    assert result.reward == 1.0
    assert result.mask_sample is False
    assert FakeEnv.instances[0].kwargs["enable_proxy"] is True
    assert osworld_client.os.environ["PROXY_CONFIG_FILE"] == str(proxy_path)


def test_proxy_reset_failure_has_specific_masked_termination(monkeypatch, tmp_path: Path) -> None:
    _patch_client_for_fake_runtime(monkeypatch)
    proxy_path = tmp_path / "proxy.json"
    proxy_path.write_text('[{"host":"proxy.example.com","port":3128}]\n', encoding="utf-8")

    def fail_reset(_self, task_config):
        raise RuntimeError("tinyproxy setup failed")

    monkeypatch.setattr(FakeEnv, "reset", fail_reset)
    result = osworld_client.run_osworld_task(
        {"id": "proxy-setup-failure", "instruction": "Open the website.", "proxy": True},
        model_fn=lambda *_args: pytest.fail("model must not run"),
        env_class_path="fake.FakeEnv",
        enable_proxy=True,
        proxy_config_file=str(proxy_path),
        sleep_after_execution=0,
        task_timeout=10,
    )

    assert result.reward == 0.0
    assert result.mask_sample is True
    assert result.termination_reason == "proxy_setup_error"
    assert "tinyproxy setup failed" in (result.error or "")


def test_raw_reward_mode_preserves_partial_osworld_score(monkeypatch) -> None:
    _patch_client_for_fake_runtime(monkeypatch)
    monkeypatch.setattr(FakeEnv, "evaluate", lambda _self: 0.4)

    result = osworld_client.run_osworld_task(
        {"id": "partial-score", "instruction": "Finish the task."},
        model_fn=lambda _system, _instruction, _history: "```DONE```",
        env_class_path="fake.FakeEnv",
        sleep_after_execution=0,
        task_timeout=10,
        reward_mode="raw",
    )

    assert result.score == 0.4
    assert result.reward == 0.4
    assert result.finished is True


def test_setup_score_zero_returns_valid_unmasked_zero(monkeypatch) -> None:
    _patch_client_for_fake_runtime(monkeypatch)

    result = osworld_client.run_osworld_task(
        {"id": "setup-zero", "instruction": "This task is already known to score zero."},
        model_fn=lambda _system, _instruction, _history: pytest.fail("model must not run"),
        env_class_path="fake.FakeSetupScoreZeroEnv",
        sleep_after_execution=0,
        task_timeout=10,
    )

    assert result.score == 0.0
    assert result.reward == 0.0
    assert result.finished is True
    assert result.mask_sample is False
    assert result.error is None
    assert result.termination_reason == "setup_score_zero"


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


def test_task_artifacts_capture_logs_trajectory_screenshots_and_result(monkeypatch, tmp_path: Path) -> None:
    _patch_client_for_fake_runtime(monkeypatch)
    artifact_root = tmp_path / "task-artifacts"
    monkeypatch.setenv("OSWORLD_TASK_ARTIFACT_ROOT", str(artifact_root))
    adapter_logger = osworld_client.logging.getLogger("nemo_gym.osworld_agent")
    handlers_before = list(adapter_logger.handlers)

    result = osworld_client.run_osworld_task(
        {
            "id": "artifact-task",
            "snapshot": "chrome",
            "instruction": "Capture complete task logs.",
        },
        model_fn=lambda _system, _instruction, _history: "```DONE```",
        env_class_path="fake.FakeEnv",
        policy_model_name="model-under-test",
        sleep_after_execution=0,
        task_timeout=10,
    )

    artifact_dir = artifact_root / "chrome" / "artifact-task"
    assert result.artifact_dir == str(artifact_dir)
    assert adapter_logger.handlers == handlers_before
    expected_files = {
        "manifest.json",
        "result.json",
        "run.json",
        "runtime.log",
        "step_000.png",
        "step_001.png",
        "task.json",
        "traj.jsonl",
        "worker.log",
    }
    assert expected_files.issubset({path.name for path in artifact_dir.iterdir()})

    trajectory = [
        osworld_client.json.loads(line)
        for line in (artifact_dir / "traj.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["event"] for record in trajectory] == ["initial_state", "step", "evaluation"]
    assert all(record["schema_version"] == 2 for record in trajectory)
    assert all(record["task_id"] == "artifact-task" for record in trajectory)
    assert all(record["domain"] == "chrome" for record in trajectory)
    assert all(record["event_id"] for record in trajectory)
    assert trajectory[0]["screenshot_sha256"]
    assert trajectory[1]["actions"] == ["DONE"]
    assert trajectory[1]["screenshot_file"] == "step_001.png"
    assert trajectory[1]["screenshot_sha256"] == trajectory[0]["screenshot_sha256"]
    assert trajectory[2]["status"] == "completed"
    assert trajectory[2]["agent_terminal_action"] == "DONE"
    assert trajectory[2]["agent_declared_success"] is True
    assert trajectory[2]["evaluation_error"] is None

    result_payload = osworld_client.json.loads((artifact_dir / "result.json").read_text(encoding="utf-8"))
    assert result_payload["reward"] == 1.0
    assert result_payload["score"] == 1.0
    assert result_payload["step_count"] == 1
    assert result_payload["termination_reason"] == "agent_done"
    assert "Starting OSWorld rollout" in (artifact_dir / "worker.log").read_text(encoding="utf-8")
    assert "Starting OSWorld rollout" in (artifact_dir / "runtime.log").read_text(encoding="utf-8")


def test_task_artifact_directory_is_collision_safe(monkeypatch, tmp_path: Path) -> None:
    _patch_client_for_fake_runtime(monkeypatch)
    monkeypatch.setenv("OSWORLD_TASK_ARTIFACT_ROOT", str(tmp_path))
    task = {"id": "same-task", "snapshot": "calc", "instruction": "Repeat safely."}
    kwargs = {
        "model_fn": lambda _system, _instruction, _history: "```DONE```",
        "env_class_path": "fake.FakeEnv",
        "sleep_after_execution": 0,
        "task_timeout": 10,
    }

    first = osworld_client.run_osworld_task(task, **kwargs)
    second = osworld_client.run_osworld_task(task, **kwargs)

    assert first.artifact_dir == str(tmp_path / "calc" / "same-task")
    assert second.artifact_dir is not None
    assert second.artifact_dir != first.artifact_dir
    assert Path(second.artifact_dir).name.startswith("same-task--")


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


def test_nemotron_v3_nano_omni_runner_uses_gym_messages_transport(monkeypatch) -> None:
    _patch_client_for_fake_runtime(monkeypatch)
    calls: List[Dict[str, Any]] = []

    def messages_model_fn(messages: List[Dict[str, Any]], payload: Dict[str, Any]) -> Dict[str, str]:
        calls.append({"messages": messages, "payload": payload})
        return {"content": "Nemotron response", "reasoning_content": "Inspect then finish."}

    result = osworld_client.run_osworld_task(
        {"id": "task-omni-mini", "instruction": "Use the Nemotron Omni scaffold."},
        model_fn=lambda *_args: (_ for _ in ()).throw(AssertionError("Nemotron should use messages_model_fn")),
        runner_name="nemotron_v3_nano_omni_agent",
        env_class_path="fake.FakeEnv",
        agent_class_path="fake.FakeNemotronAgent",
        messages_model_fn=messages_model_fn,
        policy_model_name="nemotron-3-nano-omni-under-test",
        policy_max_tokens=8192,
        policy_temperature=0.6,
        policy_top_p=0.95,
        max_steps=100,
        max_trajectory_length=3,
        sleep_after_execution=0,
        task_timeout=10,
    )

    assert result.reward == 1.0
    assert result.finished is True
    assert result.steps[0].model_text == "Nemotron response"
    assert result.steps[0].actions == ["DONE"]
    assert result.steps[0].info["agent"]["thought"] == "Inspect then finish."
    assert calls[0]["payload"]["_nemo_gym_return_message"] is True
    assert FakeNemotronAgent.instances[0].kwargs["model"] == "nemotron-3-nano-omni-under-test"
    assert FakeNemotronAgent.instances[0].kwargs["max_steps"] == 100


def test_qwen3_omni_runner_retries_and_merges_adjacent_pyautogui_actions(monkeypatch) -> None:
    _patch_client_for_fake_runtime(monkeypatch)
    responses = [
        "No tool call yet.",
        '<tool_call>\n{"name":"computer_use","arguments":{"action":"left_click"}}\n</tool_call>',
    ]
    calls = 0

    def messages_model_fn(_messages: List[Dict[str, Any]], _payload: Dict[str, Any]) -> str:
        nonlocal calls
        response = responses[calls]
        calls += 1
        return response

    result = osworld_client.run_osworld_task(
        {"id": "task-qwen", "instruction": "Use Qwen3-Omni."},
        model_fn=lambda *_args: (_ for _ in ()).throw(AssertionError("Qwen should use messages_model_fn")),
        runner_name="qwen3_omni_agent",
        env_class_path="fake.FakeEnv",
        agent_class_path="fake.FakeQwenAgent",
        agent_kwargs={"model_call_retries": 3, "require_tool_call": True},
        messages_model_fn=messages_model_fn,
        policy_model_name="qwen3-omni-under-test",
        policy_max_tokens=32768,
        policy_temperature=0.0,
        policy_top_p=0.9,
        max_steps=100,
        max_trajectory_length=4,
        sleep_after_execution=0,
        task_timeout=10,
    )

    assert calls == 2
    assert result.reward == 1.0
    assert result.steps[0].actions == ["pyautogui.click(1, 2)\npyautogui.press('enter')", "DONE"]
    assert FakeEnv.instances[0].actions == ["pyautogui.click(1, 2)\npyautogui.press('enter')", "DONE"]
    qwen = FakeQwenAgent.instances[0]
    assert qwen.kwargs["model"] == "qwen3-omni-under-test"
    assert qwen.kwargs["history_n"] == 4


def test_stage_setup_cache_links_task_artifacts_without_patching_osworld(monkeypatch, tmp_path: Path) -> None:
    task_id = "task-with-prestaged-files"
    url = "https://example.test/reference.pdf"
    cache_name = f"{osworld_client.uuid.uuid5(osworld_client.uuid.NAMESPACE_URL, url)}_reference.pdf"
    source_dir = tmp_path / "prestage" / task_id
    source_dir.mkdir(parents=True)
    (source_dir / cache_name).write_bytes(b"pdf")
    monkeypatch.setenv("OSWORLD_SETUP_CACHE_DIR", str(tmp_path / "prestage"))

    linked = osworld_client._stage_setup_cache(
        {
            "id": task_id,
            "instruction": "Use the reference.",
            "config": [
                {
                    "type": "download",
                    "parameters": {"files": [{"url": url, "path": "/home/oai/share/reference.pdf"}]},
                }
            ],
        },
        str(tmp_path / "cache"),
    )

    destination = tmp_path / "cache" / task_id / cache_name
    assert linked == 1
    assert destination.is_symlink()
    assert destination.read_bytes() == b"pdf"


def test_stage_setup_cache_does_not_link_evaluator_outputs(monkeypatch, tmp_path: Path) -> None:
    task_id = "task-with-archive-evaluator"
    url = "https://example.test/input.zip"
    cache_name = f"{osworld_client.uuid.uuid5(osworld_client.uuid.NAMESPACE_URL, url)}_input.zip"
    source_dir = tmp_path / "prestage" / task_id
    source_dir.mkdir(parents=True)
    (source_dir / cache_name).write_bytes(b"zip")
    (source_dir / "result_pred").mkdir()
    (source_dir / "result_gold").mkdir()
    monkeypatch.setenv("OSWORLD_SETUP_CACHE_DIR", str(tmp_path / "prestage"))

    linked = osworld_client._stage_setup_cache(
        {
            "id": task_id,
            "config": [
                {
                    "type": "download",
                    "parameters": {"files": [{"url": url, "path": "/home/oai/share/input.zip"}]},
                }
            ],
        },
        str(tmp_path / "cache"),
    )

    task_cache = tmp_path / "cache" / task_id
    assert linked == 1
    assert (task_cache / cache_name).is_symlink()
    assert not (task_cache / "result_pred").exists()
    assert not (task_cache / "result_gold").exists()


def test_stage_setup_cache_supports_flat_spreadsheet_download_cache(monkeypatch, tmp_path: Path) -> None:
    task_id = "spreadsheetbench-task"
    url = "https://example.test/input.xlsx"
    destination_path = "/home/oai/share/input.xlsx"
    cache_name = f"{osworld_client.uuid.uuid5(osworld_client.uuid.NAMESPACE_URL, url)}_input.xlsx"
    source_dir = tmp_path / "spreadsheet-prestage"
    source_dir.mkdir()
    (source_dir / cache_name).write_bytes(b"xlsx")
    monkeypatch.setenv("SPREADSHEETBENCH_SETUP_CACHE_DIR", str(source_dir))

    linked = osworld_client._stage_setup_cache(
        {
            "id": task_id,
            "config": [
                {
                    "type": "download",
                    "parameters": {"files": [{"url": url, "path": destination_path}]},
                }
            ],
        },
        str(tmp_path / "cache"),
    )

    destination = tmp_path / "cache" / task_id / cache_name
    assert linked == 1
    assert destination.is_symlink()
    assert destination.read_bytes() == b"xlsx"


def _install_fake_setup_module(monkeypatch, result: Dict[str, Any]):
    desktop_env = ModuleType("desktop_env")
    controllers = ModuleType("desktop_env.controllers")
    setup_module = ModuleType("desktop_env.controllers.setup")
    original_calls: List[Dict[str, Any]] = []
    post_calls: List[Dict[str, Any]] = []

    class SetupController:
        def __init__(self, cache_dir: str) -> None:
            self.client_password = "password"  # pragma: allowlist secret
            self.screen_width = 1920
            self.screen_height = 1080
            self.http_server = "http://vm"
            self.cache_dir = cache_dir

        def _execute_setup(
            self,
            command,
            stdout="",
            stderr="",
            shell=False,
            until=None,
        ):
            original_calls.append(
                {
                    "command": command,
                    "stdout": stdout,
                    "stderr": stderr,
                    "shell": shell,
                    "until": until,
                }
            )
            return "upstream"

    class Response:
        status_code = 200

        def json(self):
            return dict(result)

    def post(url, **kwargs):
        post_calls.append({"url": url, **kwargs})
        return Response()

    setup_module.SetupController = SetupController
    setup_module.requests = SimpleNamespace(
        post=post,
        exceptions=SimpleNamespace(RequestException=RuntimeError),
    )
    controllers.setup = setup_module
    desktop_env.controllers = controllers
    monkeypatch.setitem(sys.modules, "desktop_env", desktop_env)
    monkeypatch.setitem(sys.modules, "desktop_env.controllers", controllers)
    monkeypatch.setitem(sys.modules, "desktop_env.controllers.setup", setup_module)
    return SetupController, original_calls, post_calls


def test_setup_returncode_contract_is_opt_in_and_accepts_declared_codes(monkeypatch, tmp_path: Path) -> None:
    controller_class, original_calls, post_calls = _install_fake_setup_module(
        monkeypatch,
        {"returncode": 2, "output": "expected", "error": ""},
    )
    osworld_client._patch_setup_execute_contract()
    controller = controller_class(str(tmp_path))

    assert controller._execute_setup(["legacy"]) == "upstream"
    assert len(original_calls) == 1
    result = controller._execute_setup(
        ["tool", "{SCREEN_WIDTH_HALF}"],
        expected_returncodes=[0, 2],
    )

    assert result["returncode"] == 2
    assert len(post_calls) == 1
    assert '"960"' in post_calls[0]["data"]


def test_setup_on_nonzero_score_zero_is_a_valid_evaluator_outcome(monkeypatch, tmp_path: Path) -> None:
    controller_class, _, _ = _install_fake_setup_module(
        monkeypatch,
        {"returncode": 3, "output": "not present", "error": ""},
    )
    osworld_client._patch_setup_execute_contract()
    controller = controller_class(str(tmp_path))

    with pytest.raises(osworld_client._EvaluatorScoreZero, match="return code 3"):
        controller._execute_setup(["check"], on_nonzero="score_zero")

    class Evaluator:
        def evaluate(self):
            raise osworld_client._EvaluatorScoreZero("expected zero")

    assert osworld_client._evaluate_osworld_env(
        Evaluator(),
        osworld_client.logging.getLogger("test-evaluator"),
        disable_gpu=False,
    ) == pytest.approx(0.0)


def test_docker_port_lock_timeout_is_configurable(monkeypatch) -> None:
    desktop_env = ModuleType("desktop_env")
    providers = ModuleType("desktop_env.providers")
    docker_package = ModuleType("desktop_env.providers.docker")
    provider = ModuleType("desktop_env.providers.docker.provider")
    provider.LOCK_TIMEOUT = 10
    docker_package.provider = provider
    providers.docker = docker_package
    desktop_env.providers = providers
    monkeypatch.setitem(sys.modules, "desktop_env", desktop_env)
    monkeypatch.setitem(sys.modules, "desktop_env.providers", providers)
    monkeypatch.setitem(sys.modules, "desktop_env.providers.docker", docker_package)
    monkeypatch.setitem(sys.modules, "desktop_env.providers.docker.provider", provider)

    osworld_client._configure_docker_port_lock_timeout(45.0)
    assert provider.LOCK_TIMEOUT == 45.0
    with pytest.raises(ValueError, match="must be positive"):
        osworld_client._configure_docker_port_lock_timeout(0)


def test_extension_alias_patch_updates_both_metric_exports(monkeypatch) -> None:
    desktop_env = ModuleType("desktop_env")
    evaluators = ModuleType("desktop_env.evaluators")
    metrics = ModuleType("desktop_env.evaluators.metrics")
    chrome_metrics = ModuleType("desktop_env.evaluators.metrics.chrome")
    calls: List[tuple[Any, Any]] = []

    def original(installed: Any, expected: Any) -> float:
        calls.append((installed, expected))
        return float(installed == expected["expected"])

    chrome_metrics.is_expected_installed_extensions = original
    metrics.is_expected_installed_extensions = original
    metrics.chrome = chrome_metrics
    evaluators.metrics = metrics
    desktop_env.evaluators = evaluators
    monkeypatch.setitem(sys.modules, "desktop_env", desktop_env)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators", evaluators)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators.metrics", metrics)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators.metrics.chrome", chrome_metrics)

    osworld_client._patch_extension_name_aliases()
    osworld_client._patch_extension_name_aliases()

    result = metrics.is_expected_installed_extensions(
        ["Speechify — Text to Speech"],
        {"expected": ["Speechify — Voice AI Assistant"]},
    )
    assert result == 1.0
    assert calls == [
        (
            ["Speechify — Voice AI Assistant"],
            {"expected": ["Speechify — Voice AI Assistant"]},
        )
    ]
    assert metrics.is_expected_installed_extensions is chrome_metrics.is_expected_installed_extensions


def test_pdf_evaluator_cleanup_patch_preserves_zero_score(monkeypatch) -> None:
    desktop_env = ModuleType("desktop_env")
    evaluators = ModuleType("desktop_env.evaluators")
    metrics = ModuleType("desktop_env.evaluators.metrics")
    chrome_metrics = ModuleType("desktop_env.evaluators.metrics.chrome")

    def original(_actual: Any, _expected: Any) -> float:
        raise FileNotFoundError(2, "missing", "cache/task/temp_pdf_comparison")

    chrome_metrics.compare_pdf_images = original
    metrics.compare_pdf_images = original
    metrics.chrome = chrome_metrics
    evaluators.metrics = metrics
    desktop_env.evaluators = evaluators
    monkeypatch.setitem(sys.modules, "desktop_env", desktop_env)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators", evaluators)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators.metrics", metrics)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators.metrics.chrome", chrome_metrics)

    osworld_client._patch_pdf_image_evaluator_cleanup()
    osworld_client._patch_pdf_image_evaluator_cleanup()

    assert metrics.compare_pdf_images("actual.pdf", "gold.pdf") == 0.0
    assert metrics.compare_pdf_images is chrome_metrics.compare_pdf_images


def test_pdf_evaluator_cleanup_patch_reraises_unrelated_missing_file(monkeypatch) -> None:
    desktop_env = ModuleType("desktop_env")
    evaluators = ModuleType("desktop_env.evaluators")
    metrics = ModuleType("desktop_env.evaluators.metrics")
    chrome_metrics = ModuleType("desktop_env.evaluators.metrics.chrome")

    def original(_actual: Any, _expected: Any) -> float:
        raise FileNotFoundError(2, "missing", "cache/task/receipt.pdf")

    chrome_metrics.compare_pdf_images = original
    metrics.compare_pdf_images = original
    metrics.chrome = chrome_metrics
    evaluators.metrics = metrics
    desktop_env.evaluators = evaluators
    monkeypatch.setitem(sys.modules, "desktop_env", desktop_env)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators", evaluators)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators.metrics", metrics)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators.metrics.chrome", chrome_metrics)

    osworld_client._patch_pdf_image_evaluator_cleanup()

    with pytest.raises(FileNotFoundError, match="receipt.pdf"):
        metrics.compare_pdf_images("actual.pdf", "gold.pdf")


def test_evaluator_result_artifacts_records_zero_byte_output(monkeypatch, tmp_path: Path) -> None:
    task_id = "pdf-task"
    task_cache = tmp_path / task_id
    task_cache.mkdir()
    (task_cache / "receipt.pdf").write_bytes(b"")
    task = {
        "id": task_id,
        "evaluator": {
            "func": "compare_pdf_images",
            "result": {"type": "vm_file", "path": "/home/user/Desktop/receipt.pdf", "dest": "receipt.pdf"},
        },
    }

    records = osworld_client._evaluator_result_artifacts(task, str(tmp_path))

    assert records == [
        {
            "destination": "receipt.pdf",
            "cache_path": str(task_cache / "receipt.pdf"),
            "within_task_cache": True,
            "exists": True,
            "is_file": True,
            "bytes": 0,
            "sha256": osworld_client.hashlib.sha256(b"").hexdigest(),
        }
    ]


def test_evaluator_gpu_visibility_is_restored(monkeypatch) -> None:
    observed_cuda_values: List[str | None] = []
    observed_easyocr_gpu: List[bool] = []
    easyocr_module = ModuleType("easyocr")

    def reader(_languages: List[str], *, gpu: bool = True) -> object:
        observed_easyocr_gpu.append(gpu)
        return object()

    easyocr_module.Reader = reader
    monkeypatch.setitem(sys.modules, "easyocr", easyocr_module)

    class Env:
        def evaluate(self) -> float:
            observed_cuda_values.append(osworld_client.os.environ.get("CUDA_VISIBLE_DEVICES"))
            easyocr_module.Reader(["en"], gpu=True)
            return 1.0

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    assert osworld_client._evaluate_osworld_env(Env(), osworld_client.LOG, disable_gpu=True) == 1.0
    assert observed_cuda_values == [""]
    assert observed_easyocr_gpu == [False]
    assert easyocr_module.Reader is reader
    assert osworld_client.os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"


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
        policy_api_key="test-key",  # pragma: allowlist secret
        policy_model_name="azure/anthropic/claude-opus-4-7",
        sleep_after_execution=0,
        task_timeout=10,
    )

    assert result.reward == 1.0
    assert result.finished is True
    assert (
        osworld_client.os.environ["ANTHROPIC_API_KEY"] == "test-key"  # pragma: allowlist secret
    )
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
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")  # pragma: allowlist secret
    monkeypatch.setenv("POINTER_ANTHROPIC_MAX_RETRIES", "6")
    monkeypatch.setenv("POINTER_ANTHROPIC_TIMEOUT_SECONDS", "45.5")

    assert osworld_client._pointer_anthropic_client_options(
        "https://inference-api.nvidia.com/",
        api_key="client-key",  # pragma: allowlist secret
    ) == {
        "api_key": "client-key",  # pragma: allowlist secret
        "base_url": "https://inference-api.nvidia.com",
        "max_retries": 6,
        "timeout": 45.5,
    }

    assert (
        osworld_client._pointer_anthropic_client_options("https://inference-api.nvidia.com")["api_key"]
        == "env-key"  # pragma: allowlist secret
    )


def test_pointer_anthropic_proxy_logs_schema_v2_request_and_response(tmp_path: Path) -> None:
    class Response:
        def model_dump(self, *, mode: str = "python") -> Dict[str, Any]:
            assert mode == "json"
            return {"id": "msg-1", "content": [{"type": "text", "text": "done"}]}

    class Messages:
        def create(self, **kwargs: Any) -> Response:
            content = kwargs["messages"][0]["content"]
            if isinstance(content, list):
                assert content[0]["source"]["data"] == "base64-pixels"
            return Response()

        def count_tokens(self, **_kwargs: Any) -> int:
            return 123

    class Beta:
        messages = Messages()

    class Client:
        beta = Beta()
        messages = Messages()

    log_path = tmp_path / "model-io-agent.jsonl"
    context = osworld_client._PointerModelIOContext(
        log_path=str(log_path),
        identity={
            "run_id": "run-20",
            "adapter": "gym",
            "task_id": "task-1",
            "domain": "chrome",
            "served_model": "azure/anthropic/claude-opus-4-7",
        },
        step_ref=[7],
    )
    proxy = osworld_client._PointerAnthropicClientProxy(
        Client(),
        context=context,
        agent_role="executor",
    )
    response = proxy.beta.messages.create(
        model="azure/anthropic/claude-opus-4-7",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "base64-pixels"},
                    }
                ],
            }
        ],
        max_tokens=4096,
        extra_headers={
            "authorization": "Bearer secret",  # pragma: allowlist secret
            "x-trace-id": "trace-1",
        },
    )
    second_response = proxy.messages.create(
        model="azure/anthropic/claude-opus-4-7",
        messages=[{"role": "user", "content": "verify"}],
        max_tokens=128,
    )

    assert isinstance(response, Response)
    assert isinstance(second_response, Response)
    assert proxy.beta.messages.count_tokens(messages=[]) == 123
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == [
        "model_request",
        "model_response",
        "model_request",
        "model_response",
    ]
    assert all(record["schema_version"] == 2 for record in records)
    assert all(record["task_id"] == "task-1" for record in records)
    assert all(record["agent_role"] == "executor" for record in records)
    assert all(record["step"] == 7 for record in records)
    assert [record["call_index"] for record in records] == [1, 1, 2, 2]
    assert records[0]["api_surface"] == "beta.messages"
    assert records[2]["api_surface"] == "messages"
    request = records[0]["anthropic_request"]
    assert request["kwargs"]["messages"][0]["content"][0]["source"]["data"] == "base64-pixels"
    assert request["kwargs"]["extra_headers"] == {
        "authorization": "<redacted>",
        "x-trace-id": "trace-1",
    }
    assert records[0]["embedded_images"][0]["encoded_chars"] == len("base64-pixels")
    assert records[1]["anthropic_response"]["id"] == "msg-1"
    assert records[0]["call_id"] == records[1]["call_id"]
    assert records[2]["call_id"] == records[3]["call_id"]
    assert records[0]["anthropic_request_sha256"] == osworld_client._pointer_io_sha256(request)


def test_pointer_anthropic_patch_wraps_clients_only_in_logging_context(monkeypatch, tmp_path: Path) -> None:
    class APIProvider:
        ANTHROPIC = "anthropic"

    class LLMClient:
        name = "executor"
        api_key = "test-key"  # pragma: allowlist secret

        def _create_client(self, provider: Any) -> Any:
            return ("original", provider)

    class LLMContextManager:
        def _get_counting_client(self) -> str:
            return "original-counting-client"

    class Anthropic:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.messages = SimpleNamespace(create=lambda **_kwargs: "response")

    mm_agents = ModuleType("mm_agents")
    pointer_package = ModuleType("mm_agents.pointer")
    llm_client_module = ModuleType("mm_agents.pointer.llm_client")
    context_manager_module = ModuleType("mm_agents.pointer.llm_context_manager")
    utils_module = ModuleType("mm_agents.pointer.utils")
    anthropic_module = ModuleType("anthropic")
    llm_client_module.LLMClient = LLMClient
    context_manager_module.LLMContextManager = LLMContextManager
    utils_module.APIProvider = APIProvider
    anthropic_module.Anthropic = Anthropic
    pointer_package.llm_client = llm_client_module
    pointer_package.llm_context_manager = context_manager_module
    pointer_package.utils = utils_module
    mm_agents.pointer = pointer_package
    monkeypatch.setitem(sys.modules, "mm_agents", mm_agents)
    monkeypatch.setitem(sys.modules, "mm_agents.pointer", pointer_package)
    monkeypatch.setitem(sys.modules, "mm_agents.pointer.llm_client", llm_client_module)
    monkeypatch.setitem(sys.modules, "mm_agents.pointer.llm_context_manager", context_manager_module)
    monkeypatch.setitem(sys.modules, "mm_agents.pointer.utils", utils_module)
    monkeypatch.setitem(sys.modules, "anthropic", anthropic_module)

    osworld_client._patch_pointer_anthropic_client("https://inference-api.nvidia.com/")
    client = LLMClient()
    unlogged = client._create_client(APIProvider.ANTHROPIC)
    assert isinstance(unlogged, Anthropic)
    assert unlogged.kwargs["base_url"] == "https://inference-api.nvidia.com"
    assert client._create_client("other") == ("original", "other")

    context = osworld_client._PointerModelIOContext(
        log_path=str(tmp_path / "model-io-agent.jsonl"),
        identity={"task_id": "task-1"},
        step_ref=[1],
    )
    token = osworld_client._POINTER_MODEL_IO_CONTEXT.set(context)
    try:
        logged = client._create_client(APIProvider.ANTHROPIC)
    finally:
        osworld_client._POINTER_MODEL_IO_CONTEXT.reset(token)

    assert isinstance(logged, osworld_client._PointerAnthropicClientProxy)
    assert logged.messages.create(model="model", messages=[]) == "response"
    records = [
        json.loads(line)
        for line in (tmp_path / "model-io-agent.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["event"] for record in records] == ["model_request", "model_response"]


def test_pointer_anthropic_proxy_logs_errors_and_preserves_exception(tmp_path: Path) -> None:
    class Messages:
        def create(self, **_kwargs: Any) -> Any:
            raise RuntimeError("provider unavailable")

    class Client:
        messages = Messages()

    log_path = tmp_path / "model-io-agent.jsonl"
    context = osworld_client._PointerModelIOContext(
        log_path=str(log_path),
        identity={"task_id": "task-error"},
        step_ref=[3],
    )
    proxy = osworld_client._PointerAnthropicClientProxy(
        Client(),
        context=context,
        agent_role="planner",
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        proxy.messages.create(model="model", messages=[])

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["model_request", "model_error"]
    assert records[1]["error_type"] == "RuntimeError"
    assert records[0]["call_id"] == records[1]["call_id"]


def test_pointer_model_io_log_failure_does_not_change_sdk_result(tmp_path: Path) -> None:
    class Messages:
        def create(self, **_kwargs: Any) -> str:
            return "response"

    class Client:
        messages = Messages()

    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("file", encoding="utf-8")
    context = osworld_client._PointerModelIOContext(
        log_path=str(blocked_parent / "model-io-agent.jsonl"),
        identity={"task_id": "task-logging-failure"},
        step_ref=[1],
    )
    proxy = osworld_client._PointerAnthropicClientProxy(
        Client(),
        context=context,
        agent_role="verifier",
    )

    assert proxy.messages.create(model="model", messages=[]) == "response"


def test_pointer_model_io_serialization_failure_does_not_change_sdk_result(monkeypatch, tmp_path: Path) -> None:
    class Messages:
        def create(self, **_kwargs: Any) -> str:
            return "response"

    class Client:
        messages = Messages()

    context = osworld_client._PointerModelIOContext(
        log_path=str(tmp_path / "model-io-agent.jsonl"),
        identity={"task_id": "task-serialization-failure"},
        step_ref=[1],
    )
    proxy = osworld_client._PointerAnthropicClientProxy(
        Client(),
        context=context,
        agent_role="verifier",
    )

    def fail_hash(_value: Any) -> str:
        raise RuntimeError("diagnostic serialization failed")

    monkeypatch.setattr(osworld_client, "_pointer_io_sha256", fail_hash)

    assert proxy.messages.create(model="model", messages=[]) == "response"


def test_pointer_model_io_context_is_opt_in_and_task_scoped(monkeypatch, tmp_path: Path) -> None:
    step_ref = [0]
    monkeypatch.delenv("OSWORLD_MODEL_IO_LOG", raising=False)
    assert (
        osworld_client._pointer_model_io_context(
            {"run_id": "run-20", "task_id": "task-1"},
            endpoint="https://inference-api.nvidia.com",
            served_model="azure/anthropic/claude-opus-4-7",
            step_ref=step_ref,
        )
        is None
    )

    log_path = tmp_path / "model-io-agent.jsonl"
    monkeypatch.setenv("OSWORLD_MODEL_IO_LOG", str(log_path))
    monkeypatch.setenv("NEMO_GYM_SOURCE_COMMIT", "09a895d")
    context = osworld_client._pointer_model_io_context(
        {"run_id": "run-20", "task_id": "task-1", "domain": "chrome"},
        endpoint="https://inference-api.nvidia.com",
        served_model="azure/anthropic/claude-opus-4-7",
        step_ref=step_ref,
    )

    assert context is not None
    assert context.log_path == str(log_path)
    assert context.step_ref is step_ref
    assert context.identity == {
        "run_id": "run-20",
        "task_id": "task-1",
        "domain": "chrome",
        "adapter": "gym",
        "endpoint": "https://inference-api.nvidia.com",
        "served_model": "azure/anthropic/claude-opus-4-7",
        "source_commit": "09a895d",
    }


def test_pointer_rollout_sets_and_resets_model_io_context(monkeypatch, tmp_path: Path) -> None:
    _patch_client_for_fake_runtime(monkeypatch)
    monkeypatch.setenv("OSWORLD_POINTER_RESULTS_DIR", str(tmp_path / "pointer"))
    monkeypatch.setenv("OSWORLD_MODEL_IO_LOG", str(tmp_path / "model-io-agent.jsonl"))
    observed_contexts: List[osworld_client._PointerModelIOContext | None] = []
    monkeypatch.setattr(
        osworld_client,
        "_patch_pointer_anthropic_client",
        lambda _base_url: observed_contexts.append(osworld_client._POINTER_MODEL_IO_CONTEXT.get()),
    )

    result = osworld_client.run_osworld_task(
        {"id": "task-pointer-context", "instruction": "Use Pointer."},
        model_fn=lambda *_args: "unused",
        runner_name="pointer_agent",
        env_class_path="fake.FakeEnv",
        agent_class_path="fake.FakePointerAgent",
        policy_base_url="https://inference-api.nvidia.com",
        policy_api_key="test-key",  # pragma: allowlist secret
        policy_model_name="azure/anthropic/claude-opus-4-7",
        log_context={"run_id": "run-20", "task_attempt": 2},
        sleep_after_execution=0,
        task_timeout=10,
    )

    assert result.error is None
    assert len(observed_contexts) == 1
    assert observed_contexts[0] is not None
    assert observed_contexts[0].identity["task_id"] == "task-pointer-context"
    assert observed_contexts[0].identity["task_attempt"] == 2
    assert observed_contexts[0].step_ref == [1]
    assert osworld_client._POINTER_MODEL_IO_CONTEXT.get() is None


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
        policy_api_key="test-key",  # pragma: allowlist secret
        policy_model_name="azure/anthropic/claude-opus-4-7",
        sleep_after_execution=0,
        task_timeout=10,
    )

    assert result.reward == 1.0
    assert result.score == 1.0
    assert result.error is None
