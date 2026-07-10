# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for adapter-owned OSWorld model scaffolds."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from responses_api_agents.osworld_agent.native_agents import (
    NemotronOmniAgent,
    NemotronV3Agent,
    normalize_python_code_newlines,
    normalize_response_content,
    parse_nemotron_response,
    project_pyautogui_coordinates,
)


@pytest.mark.parametrize(
    ("coordinate_type", "code", "expected"),
    [
        ("relative", "pyautogui.click(0.5, 0.25)", "pyautogui.click(960, 270)"),
        ("absolute", "pyautogui.moveTo(x=12, y=34)", "pyautogui.moveTo(x=12, y=34)"),
        ("qwen25", "pyautogui.dragTo(500, 250, duration=1)", "pyautogui.dragTo(960, 270, duration=1)"),
    ],
)
def test_project_pyautogui_coordinates(coordinate_type: str, code: str, expected: str) -> None:
    assert (
        project_pyautogui_coordinates(
            code,
            screen_width=1920,
            screen_height=1080,
            coordinate_type=coordinate_type,
        )
        == expected
    )


def test_parse_nemotron_response_preserves_reasoning_and_projects_action() -> None:
    response = {
        "reasoning_content": "The target is in the middle of the screen.",
        "content": """## Action:
Click the target.
## Code:
```python
pyautogui.click(0.5, 0.25)
```""",
    }

    action, commands, info = parse_nemotron_response(
        response,
        screen_size=(1920, 1080),
        coordinate_type="relative",
        thinking=True,
    )

    assert action == "Click the target."
    assert commands == ["pyautogui.click(960, 270)"]
    assert info["thought"] == "The target is in the middle of the screen."
    assert info["original_code"] == "pyautogui.click(0.5, 0.25)"


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ('{"name": "computer.wait", "arguments": {}}', "WAIT"),
        ('{"name": "computer.terminate", "arguments": {"status": "success"}}', "DONE"),
        ('{"name": "computer.terminate", "arguments": {"status": "failure"}}', "FAIL"),
    ],
)
def test_parse_nemotron_control_actions(code: str, expected: str) -> None:
    response = f"""## Action:
Control the task.
## Code:
```code
{code}
```"""

    _action, commands, info = parse_nemotron_response(
        response,
        screen_size=(1920, 1080),
        coordinate_type="relative",
        thinking=False,
    )

    assert commands == [expected]
    assert info["code"] == expected


def test_parse_nemotron_terminate_requires_explicit_status() -> None:
    response = """## Action:
Stop.
## Code:
```code
computer.terminate()
```"""

    action, commands, _info = parse_nemotron_response(
        response,
        screen_size=(1920, 1080),
        coordinate_type="relative",
        thinking=False,
    )

    assert action.startswith("<Error>")
    assert commands == ["FAIL"]


def test_parse_nemotron_does_not_infer_status_from_answer_text() -> None:
    response = """## Action:
Stop.
## Code:
```json
{"name":"computer.terminate","arguments":{"answer":"looks successful"}}
```"""

    action, commands, _info = parse_nemotron_response(
        response,
        screen_size=(1920, 1080),
        coordinate_type="relative",
        thinking=False,
    )

    assert action.startswith("<Error>")
    assert commands == ["FAIL"]


def test_parse_nemotron_accepts_unfenced_code_section() -> None:
    response = """## Action:
Click the visible target.
## Code:
pyautogui.click(0.25, 0.75)
"""

    action, commands, _info = parse_nemotron_response(
        response,
        screen_size=(1920, 1080),
        coordinate_type="relative",
        thinking=False,
    )

    assert action == "Click the visible target."
    assert commands == ["pyautogui.click(480, 810)"]


def test_parse_nemotron_normalizes_literal_newlines_outside_code() -> None:
    response = {
        "content": (
            "\\n## Action:\\nClick the Change button.\\n## Code:\n```python\npyautogui.click(0.664,0.308)\n```"
        ),
        "reasoning": "The Change button is visible.",
    }
    action, commands, info = parse_nemotron_response(
        response, screen_size=(1920, 1080), coordinate_type="relative", thinking=True
    )
    assert action == "Click the Change button."
    assert commands == ["pyautogui.click(1275, 333)"]
    assert info["thought"] == "The Change button is visible."


def test_normalize_response_preserves_literal_newline_inside_code() -> None:
    content = "\\n## Action:\\nType two lines.\\n## Code:\n```python\npyautogui.write('first\\nsecond')\n```"
    normalized = normalize_response_content(content)
    assert "\n## Action:\nType two lines.\n## Code:\n" in normalized
    assert "pyautogui.write('first\\nsecond')" in normalized


def test_normalize_python_code_newlines_restores_only_structural_escapes() -> None:
    code = "\\npyautogui.click(0.5, 0.5)\\npyautogui.write('first\\nsecond')\\n"

    normalized = normalize_python_code_newlines(code)

    assert normalized == "\npyautogui.click(0.5, 0.5)\npyautogui.write('first\\nsecond')\n"
    compile(normalized, "<test-action>", "exec")


def test_parse_nemotron_response_repairs_structural_code_newlines() -> None:
    response = {
        "content": (
            "## Action:\nClick and type.\n## Code:\n```python\n"
            "\\npyautogui.click(0.5, 0.5)\\npyautogui.write('first\\nsecond')\\n\n```"
        )
    }

    _action, commands, info = parse_nemotron_response(
        response, screen_size=(1920, 1080), coordinate_type="relative", thinking=False
    )

    assert commands == ["pyautogui.click(960, 540)\npyautogui.write('first\\nsecond')"]
    assert info["raw_code"].startswith("\\n")
    compile(commands[0], "<test-action>", "exec")


def test_nemotron_agent_routes_messages_and_compacts_old_images() -> None:
    agent = NemotronV3Agent(
        model="policy-under-test",
        max_steps=3,
        max_image_history_length=2,
        max_tokens=4096,
        temperature=0.6,
        top_p=0.95,
    )
    payloads: List[Dict[str, Any]] = []
    responses = [
        {
            "reasoning_content": "First thought",
            "content": "## Action:\nClick.\n## Code:\n```python\npyautogui.click(0.5, 0.5)\n```",
        },
        {
            "reasoning_content": "Second thought",
            "content": "## Action:\nWait.\n## Code:\n```code\ncomputer.wait()\n```",
        },
        {
            "reasoning_content": "Done",
            "content": (
                "## Action:\nFinish.\n## Code:\n```code\n"
                '{"name": "computer.terminate", "arguments": {"status": "success"}}\n```'
            ),
        },
    ]

    def call_llm(payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        payloads.append(payload)
        return responses[len(payloads) - 1]

    agent.call_llm = call_llm  # type: ignore[method-assign]
    obs = {"screenshot": b"fake-png"}

    assert agent.predict("Complete the task.", obs)[1] == ["pyautogui.click(960, 540)"]
    assert agent.predict("Complete the task.", obs)[1] == ["WAIT"]
    assert agent.predict("Complete the task.", obs)[1] == ["DONE"]

    # At step three only the most recent historical image is retained. The
    # older step is represented as text, so total images remain bounded at 2.
    third_messages = payloads[2]["messages"]
    image_parts = [
        part
        for message in third_messages
        for part in message.get("content", [])
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]
    assert len(image_parts) == 2
    assert any("# Previous History Actions" in str(message.get("content")) for message in third_messages)
    assert payloads[0]["_nemo_gym_return_message"] is True


def test_nemotron_agent_turns_last_nonterminal_step_into_fail() -> None:
    agent = NemotronV3Agent(model="policy", max_steps=1)
    agent.call_llm = lambda _payload, _model: {  # type: ignore[method-assign]
        "content": "## Action:\nClick.\n## Code:\n```python\npyautogui.click(1, 2)\n```",
        "reasoning_content": "Try once.",
    }

    _response, actions, info = agent.predict("Try the task.", {"screenshot": b"fake-png"})

    assert actions == ["FAIL"]
    assert info["code"] == "FAIL"


def test_nemotron_agent_retries_invalid_python_action() -> None:
    agent = NemotronV3Agent(model="policy", max_steps=2, parse_retries=2)
    responses = [
        {
            "content": ("## Action:\nClick.\n## Code:\n```python\npyautogui.click(]\n```"),
            "reasoning_content": "The first response is not repairable Python.",
        },
        {
            "content": ("## Action:\nClick.\n## Code:\n```python\npyautogui.click(0.5, 0.5)\n```"),
            "reasoning_content": "Retry with valid Python.",
        },
    ]
    calls = 0

    def call_llm(_payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        nonlocal calls
        response = responses[calls]
        calls += 1
        return response

    agent.call_llm = call_llm  # type: ignore[method-assign]
    _response, actions, _info = agent.predict("Click.", {"screenshot": b"fake-png"})

    assert calls == 2
    assert actions == ["pyautogui.click(960, 540)"]


def test_omni_agent_retries_invalid_python_with_feedback_and_lower_temperature(monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "model-io-agent.jsonl"
    monkeypatch.setenv("OSWORLD_MODEL_IO_LOG", str(log_path))
    agent = NemotronOmniAgent(
        model="policy",
        max_steps=2,
        parse_retries=2,
        parse_error_feedback=True,
        parse_retry_temperature=0.2,
        pre_done_checklist=True,
        temperature=0.6,
    )
    payloads: List[Dict[str, Any]] = []
    responses = [
        {
            "content": "## Action:\nType a URL.\n## Code:\n```python\npyautogui.write(\"unterminated)\n```",
            "reasoning_content": "The first response contains invalid Python.",
        },
        {
            "content": "## Action:\nType a URL.\n## Code:\n```python\npyautogui.write('valid')\n```",
            "reasoning_content": "Correct the string quoting.",
        },
    ]

    def call_llm(payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        payloads.append(payload)
        return responses[len(payloads) - 1]

    agent.call_llm = call_llm  # type: ignore[method-assign]
    _response, actions, _info = agent.predict("Type the URL.", {"screenshot": b"fake-png"})

    assert actions == ["pyautogui.write('valid')"]
    assert [payload["temperature"] for payload in payloads] == [0.6, 0.2]
    retry_messages = payloads[1]["messages"]
    assert [message["role"] for message in retry_messages[-2:]] == ["assistant", "user"]
    assert "unterminated string literal" in retry_messages[-1]["content"]
    assert "do not repeat the invalid code" in " ".join(retry_messages[-1]["content"].split())
    image_parts = [
        part
        for message in retry_messages
        for part in message.get("content", [])
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]
    assert len(image_parts) == 1
    first_user_text = payloads[0]["messages"][-1]["content"][-1]["text"]
    assert "Before returning computer.terminate" in first_user_text
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["event"] == "agent_parse_error"
    assert rows[0]["will_retry"] is True
    assert rows[0]["retry_feedback_injected_next"] is True
    assert rows[0]["retry_temperature_next"] == 0.2
    assert rows[1]["event"] == "agent_parse"
    assert rows[1]["parse_feedback_injected"] is True
    assert rows[1]["pre_done_checklist_injected"] is True


def test_omni_agent_warns_after_repeated_nontrivial_action() -> None:
    agent = NemotronOmniAgent(
        model="policy",
        max_steps=10,
        repeated_action_warning_threshold=3,
        repeated_action_window=6,
    )
    agent.actions = ["Scroll the settings."] * 3
    agent.cots = [{"code": "pyautogui.scroll(-3)"}] * 3

    messages = agent._messages("Change the setting.", {"screenshot": b"fake-png"})
    user_text = messages[-1]["content"][-1]["text"]

    assert "same executable action appeared 3 times" in user_text
    assert "choose a different verifiable action" in user_text


def test_nemotron_agent_logs_parse_error_and_success(monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "model-io-agent.jsonl"
    monkeypatch.setenv("OSWORLD_MODEL_IO_LOG", str(log_path))
    agent = NemotronV3Agent(
        model="policy",
        max_steps=2,
        parse_retries=2,
        log_context={
            "run_id": "run-001",
            "adapter": "gym",
            "task_id": "task-001",
            "domain": "chrome",
            "task_attempt": 1,
        },
    )
    responses = [
        {
            "content": "## Action:\nClick.\n## Code:\n```python\npyautogui.click(]\n```",
            "reasoning_content": "Invalid first attempt.",
        },
        {
            "content": "## Action:\nClick.\n## Code:\n```python\npyautogui.click(0.5, 0.5)\n```",
            "reasoning_content": "Valid retry.",
        },
    ]

    agent.call_llm = lambda _payload, _model: responses.pop(0)  # type: ignore[method-assign]
    _response, actions, _info = agent.predict("Click.", {"screenshot": b"fake-png"})

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert actions == ["pyautogui.click(960, 540)"]
    assert [row["event"] for row in rows] == ["agent_parse_error", "agent_parse"]
    assert rows[0]["attempt"] == 1
    assert rows[1]["attempt"] == 2
    assert all(row["task_id"] == "task-001" for row in rows)
    assert all(row["step"] == 1 for row in rows)
    assert [row["parse_attempt"] for row in rows] == [1, 2]
    assert rows[1]["parsed_actions"] == ["pyautogui.click(960, 540)"]


def test_omni_mini_agent_sends_one_image_and_keeps_text_history() -> None:
    agent = NemotronOmniAgent(
        model="nemotron-3-nano-omni",
        max_steps=3,
        max_text_history_length=2,
        max_tokens=8192,
        temperature=0.6,
        top_p=0.95,
    )
    payloads: List[Dict[str, Any]] = []
    responses = [
        {
            "reasoning_content": "Open settings.",
            "content": "## Action:\nClick settings.\n## Code:\n```python\npyautogui.click(0.5, 0.5)\n```",
        },
        {
            "reasoning_content": "The requested state is visible.",
            "content": (
                "## Action:\nFinish.\n## Code:\n```json\n"
                '{"name":"computer.terminate","arguments":{"status":"success"}}\n```'
            ),
        },
    ]

    def call_llm(payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        payloads.append(payload)
        return responses[len(payloads) - 1]

    agent.call_llm = call_llm  # type: ignore[method-assign]
    obs = {"screenshot": b"fake-png"}

    assert "absolute pixel coordinates from the 1920x1080" in agent.system_prompt
    assert agent.predict("Complete the task.", obs)[1] == ["pyautogui.click(960, 540)"]
    assert agent.predict("Complete the task.", obs)[1] == ["DONE"]

    for payload in payloads:
        image_parts = [
            part
            for message in payload["messages"]
            for part in message.get("content", [])
            if isinstance(part, dict) and part.get("type") == "image_url"
        ]
        assert len(image_parts) == 1
    assert "Previous interactions (text only" in payloads[1]["messages"][0]["content"]
    assert "Click settings." in payloads[1]["messages"][0]["content"]


def test_nemotron_agent_sends_current_image_and_full_text_history() -> None:
    agent = NemotronV3Agent(
        model="nemotron-3-nano-omni",
        max_steps=100,
        max_image_history_length=1,
        max_tokens=8192,
        temperature=0.6,
        top_p=0.95,
    )
    payloads: List[Dict[str, Any]] = []
    responses = [
        {
            "reasoning_content": "First thought",
            "content": "## Action:\nFirst action.\n## Code:\n```python\npyautogui.click(0.5, 0.5)\n```",
        },
        {
            "reasoning_content": "Second thought",
            "content": "## Action:\nSecond action.\n## Code:\n```python\npyautogui.click(0.4, 0.4)\n```",
        },
        {
            "reasoning_content": "Finish",
            "content": ("## Action:\nDone.\n## Code:\n```code\ncomputer.terminate(status='success')\n```"),
        },
    ]

    def call_llm(payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        payloads.append(payload)
        return responses[len(payloads) - 1]

    agent.call_llm = call_llm  # type: ignore[method-assign]
    obs = {"screenshot": b"fake-png"}
    assert agent.predict("Complete the task.", obs)[1] == ["pyautogui.click(960, 540)"]
    assert agent.predict("Complete the task.", obs)[1] == ["pyautogui.click(768, 432)"]
    assert agent.predict("Complete the task.", obs)[1] == ["DONE"]

    final_messages = payloads[-1]["messages"]
    image_parts = [
        part
        for message in final_messages
        for part in message.get("content", [])
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]
    assert len(image_parts) == 1
    final_user_text = str(final_messages[-1]["content"])
    assert "# Step 1:" in final_user_text
    assert "# Step 2:" in final_user_text
    assert "First thought" in final_user_text
    assert "Second thought" in final_user_text
    assert "First action." in final_user_text
    assert "Second action." in final_user_text
    assert "pyautogui.click(0.5, 0.5)" in final_user_text
    assert "pyautogui.click(0.4, 0.4)" in final_user_text
