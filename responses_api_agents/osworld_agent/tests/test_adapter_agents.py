# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for adapter-owned OSWorld model scaffolds."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from responses_api_agents.osworld_agent.adapter_agents import (
    NemotronV3NanoOmniAgent,
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
    ("content", "expected_action", "expected_command"),
    [
        (
            "## Action: Click the target.\n## Code:\n```python\npyautogui.click(0.5, 0.25)\n```",
            "Click the target.",
            "pyautogui.click(960, 270)",
        ),
        (
            "## Action:\nClick the target.\n## Code: pyautogui.click(0.5, 0.25)",
            "Click the target.",
            "pyautogui.click(960, 270)",
        ),
        (
            "## Action: Click the target.\n## Code: pyautogui.click(0.5, 0.25)\n"
            "```python\npyautogui.click(0.5, 0.25)\n```",
            "Click the target.",
            "pyautogui.click(960, 270)",
        ),
    ],
)
def test_parse_nemotron_accepts_inline_and_multiline_sections(
    content: str, expected_action: str, expected_command: str
) -> None:
    action, commands, _info = parse_nemotron_response(
        {"content": content, "reasoning_content": "Choose the visible target."},
        screen_size=(1920, 1080),
        coordinate_type="relative",
        thinking=True,
    )

    assert action == expected_action
    assert commands == [expected_command]


def test_parse_nemotron_accepts_explicit_code_without_descriptive_action() -> None:
    action, commands, info = parse_nemotron_response(
        {"content": "## Code: pyautogui.click(0.5, 0.25)"},
        screen_size=(1920, 1080),
        coordinate_type="relative",
        thinking=True,
    )

    assert action == "Execute the provided code."
    assert commands == ["pyautogui.click(960, 270)"]
    assert info["action_inferred"] is True


def test_parse_nemotron_accepts_inline_thought_when_reasoning_is_not_separate() -> None:
    action, commands, info = parse_nemotron_response(
        {"content": ("## Thought: The target is visible.\n## Action: Click it.\n## Code: pyautogui.click(0.5, 0.25)")},
        screen_size=(1920, 1080),
        coordinate_type="relative",
        thinking=False,
    )

    assert action == "Click it."
    assert commands == ["pyautogui.click(960, 270)"]
    assert info["thought"] == "The target is visible."


def test_parse_nemotron_rejects_unlabelled_global_code_block() -> None:
    action, commands, _info = parse_nemotron_response(
        {"content": "Here is an example:\n```python\npyautogui.click(0.5, 0.25)\n```"},
        screen_size=(1920, 1080),
        coordinate_type="relative",
        thinking=True,
    )

    assert action == "<Error>: no explicit ## Code section found"
    assert commands == []


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
    assert commands == []


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
    assert commands == []


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
    agent = NemotronV3NanoOmniAgent(
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
    assert any(
        message.get("content") == "<think>\nSecond thought\n</think>\n## Action:\nWait.\n"
        for message in third_messages
    )
    serialized_messages = str(third_messages)
    assert "pyautogui.click(0.5, 0.5)" not in serialized_messages
    assert "computer.wait()" not in serialized_messages
    assert payloads[0]["_nemo_gym_return_message"] is True


def test_nemotron_automatically_records_exact_calls_with_bounded_images() -> None:
    agent = NemotronV3NanoOmniAgent(
        model="policy-under-test",
        max_steps=4,
        max_image_history_length=3,
        parse_retries=1,
    )
    payloads: List[Dict[str, Any]] = []
    responses = []
    for index in range(4):
        thought = f"Thought {index + 1}"
        raw_content = (
            f"<think>{thought}</think>## Action:\nClick.\n## Code:\n```python\npyautogui.click(0.5, 0.5)\n```"
        )
        responses.append(
            {
                "content": ("## Action:\nClick.\n## Code:\n```python\npyautogui.click(0.5, 0.5)\n```"),
                "reasoning_content": thought,
                "raw_content": raw_content,
                "prompt_token_ids": [10, 11, index],
                "generation_token_ids": [20 + index],
                "generation_log_probs": [-0.1],
            }
        )

    def call_llm(payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        payloads.append(payload)
        return responses[len(payloads) - 1]

    agent.call_llm = call_llm  # type: ignore[method-assign]
    model_call_infos = []
    for index in range(4):
        _, actions, info = agent.predict(
            "Complete the task.",
            {"screenshot": f"png-{index + 1}".encode()},
        )
        # max_steps termination is runner-owned; the adapter must preserve a
        # valid model-authored action even on its configured last step.
        assert actions == ["pyautogui.click(960, 540)"]
        model_call_infos.append(info["model_calls"][0])

    image_counts = [
        sum(
            part.get("type") == "image_url"
            for message in payload["messages"]
            for part in message.get("content", [])
            if isinstance(part, dict)
        )
        for payload in payloads
    ]
    assert image_counts == [1, 2, 3, 3]
    assert "cG5nLTE=" not in str(payloads[-1]["messages"])
    assert "# Previous History Actions" in str(payloads[-1]["messages"])
    assert model_call_infos[-1]["response"]["generation_token_ids"] == [23]
    assert model_call_infos[-1]["prompt_messages"] == payloads[-1]["messages"]
    assert all(call["accepted"] for call in model_call_infos)


def test_nemotron_snapshot_window_accumulates_from_three_to_ten_then_compacts() -> None:
    agent = NemotronV3NanoOmniAgent(
        model="policy-under-test",
        max_steps=20,
        max_image_history_length=3,
        max_live_images=10,
        parse_retries=1,
    )
    payloads: List[Dict[str, Any]] = []

    def call_llm(payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        index = len(payloads)
        payloads.append(payload)
        return {
            "content": "## Action:\nClick.\n## Code:\n```python\npyautogui.click(0.5, 0.5)\n```",
            "reasoning_content": f"Thought {index + 1}",
            "raw_content": f"raw completion {index + 1}",
            "prompt_token_ids": [10, index],
            "generation_token_ids": [20 + index],
            "generation_log_probs": [-0.1],
        }

    agent.call_llm = call_llm  # type: ignore[method-assign]
    model_calls = []
    for index in range(12):
        _, actions, info = agent.predict(
            "Complete the task.",
            {"screenshot": f"png-{index + 1}".encode()},
        )
        assert actions == ["pyautogui.click(960, 540)"]
        model_calls.append(info["model_calls"][0])

    image_counts = [
        sum(
            part.get("type") == "image_url"
            for message in payload["messages"]
            for part in message.get("content", [])
            if isinstance(part, dict)
        )
        for payload in payloads
    ]
    assert image_counts == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 3, 4]
    assert model_calls[9]["snapshot_compaction_triggered"] is False
    assert model_calls[10]["snapshot_compaction_triggered"] is True
    assert model_calls[10]["snapshot_window_start"] == 8
    assert model_calls[10]["prompt_snapshot_count"] == 3
    assert model_calls[11]["snapshot_compaction_triggered"] is False
    assert model_calls[11]["prompt_snapshot_count"] == 4
    assert "cG5nLTE=" not in str(payloads[10]["messages"])
    assert "cG5nLTk=" in str(payloads[10]["messages"])
    assert "# Previous History Actions" in str(payloads[10]["messages"])

    agent.reset()
    assert agent.compacted_turns == ()


def test_nemotron_snapshot_window_rejects_high_water_below_low_water() -> None:
    with pytest.raises(ValueError, match="max_live_images"):
        NemotronV3NanoOmniAgent(
            model="policy-under-test",
            max_steps=3,
            max_image_history_length=3,
            max_live_images=2,
        )


def test_nemotron_default_prompt_views_may_rewrite_between_calls() -> None:
    agent = NemotronV3NanoOmniAgent(
        model="policy-under-test",
        max_steps=2,
        max_image_history_length=1,
        parse_retries=1,
    )
    payloads: List[Dict[str, Any]] = []
    responses = [
        {
            "content": "## Action:\nClick.\n## Code:\n```python\npyautogui.click(0.5, 0.5)\n```",
            "reasoning_content": "First thought",
            "raw_content": "first raw completion",
            "prompt_token_ids": [10, 11],
            "generation_token_ids": [20],
            "generation_log_probs": [-0.1],
        },
        {
            "content": "## Action:\nFinish.\n## Code:\n```code\ncomputer.terminate(status='success')\n```",
            "reasoning_content": "Second thought",
            "raw_content": "second raw completion",
            "prompt_token_ids": [99, 100],
            "generation_token_ids": [101],
            "generation_log_probs": [-0.2],
        },
    ]

    def call_llm(payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        payloads.append(payload)
        return responses[len(payloads) - 1]

    agent.call_llm = call_llm  # type: ignore[method-assign]
    _, first_actions, first_info = agent.predict("Complete the task.", {"screenshot": b"first-png"})
    _, second_actions, second_info = agent.predict("Complete the task.", {"screenshot": b"second-png"})

    assert first_actions == ["pyautogui.click(960, 540)"]
    assert second_actions == ["DONE"]
    assert first_info["model_calls"][0]["accepted"] is True
    assert second_info["model_calls"][0]["accepted"] is True
    assert "Zmlyc3QtcG5n" not in str(payloads[1]["messages"])
    assert "c2Vjb25kLXBuZw==" in str(payloads[1]["messages"])


def test_nemotron_missing_token_metadata_does_not_break_benchmarking() -> None:
    agent = NemotronV3NanoOmniAgent(
        model="policy-under-test",
        max_steps=1,
        parse_retries=1,
    )
    agent.call_llm = lambda _payload, _model: {  # type: ignore[method-assign]
        "content": "## Action:\nFinish.\n## Code:\n```code\ncomputer.terminate(status='success')\n```",
        "raw_content": "raw response",
    }

    content, actions, info = agent.predict("Complete the task.", {"screenshot": b"fake-png"})

    assert actions == ["DONE"]
    assert "Finish" in content
    assert info["model_calls"][0]["accepted"] is True
    assert "prompt_token_ids" not in info["model_calls"][0]["response"]


def test_nemotron_preserves_model_call_when_python_is_invalid() -> None:
    agent = NemotronV3NanoOmniAgent(
        model="policy-under-test",
        max_steps=1,
        parse_retries=1,
    )
    response = {
        "content": "## Action:\nType.\n## Code:\n```python\npyautogui.write('truncated)\n```",
        "reasoning_content": "Attempt the action.",
        "raw_content": "<think>Attempt the action.</think>\n## Action:\nType.\n"
        "## Code:\n```python\npyautogui.write('truncated)\n```",
        "prompt_token_ids": [10, 11],
        "generation_token_ids": [20, 21],
        "generation_log_probs": [-0.1, -0.2],
    }
    agent.call_llm = lambda _payload, _model: response  # type: ignore[method-assign]

    error, actions, info = agent.predict("Type the text.", {"screenshot": b"fake-png"})

    assert actions == []
    assert "unterminated string literal" in error
    # A model that emitted a stop token mid-string is a different defect from a
    # rejected request or a truncated one; the outcome must say which.
    assert info["agent_outcome"] == "model_response_unparseable"
    assert info["agent_outcome_family"] == "model_response_invalid"
    assert info["parse_failure"]["last_failure_kind"] == "unparseable"
    assert info["stop_rollout"] is True
    assert info["model_call_completed"] is True
    assert info["parse_failure"]["last_failure_stage"] == "response_parse"
    assert "mask_sample" not in info
    assert "termination_reason" not in info
    assert info["model_calls"][0]["response"] == response
    assert info["model_calls"][0]["prompt_messages"][-1]["role"] == "user"
    assert info["model_calls"][0]["accepted"] is False


def test_nemotron_rejects_removed_training_switches() -> None:
    with pytest.raises(ValueError, match="training-specific agent switches"):
        NemotronV3NanoOmniAgent(
            model="policy-under-test",
            max_steps=1,
            training_turn_strategy="last",
        )


def test_nemotron_agent_uses_the_maintained_checkpoint_prompt_contract() -> None:
    agent = NemotronV3NanoOmniAgent(
        model="policy-under-test",
        max_steps=1,
        client_password="test-password",  # pragma: allowlist secret
    )

    assert "The passoword of the computer is test-password." in agent.system_prompt
    assert "The password of the computer is" not in agent.system_prompt


def test_nemotron_agent_preserves_last_nonterminal_action_for_runner() -> None:
    agent = NemotronV3NanoOmniAgent(model="policy", max_steps=1)
    agent.call_llm = lambda _payload, _model: {  # type: ignore[method-assign]
        "content": "## Action:\nClick.\n## Code:\n```python\npyautogui.click(1, 2)\n```",
        "reasoning_content": "Try once.",
    }

    _response, actions, info = agent.predict("Try the task.", {"screenshot": b"fake-png"})

    assert actions == ["pyautogui.click(1, 2)"]
    assert info["code"] == "pyautogui.click(1, 2)"
    assert "mask_sample" not in info
    assert "termination_reason" not in info


def test_nemotron_agent_reports_exhausted_length_response_without_deciding_admission() -> None:
    agent = NemotronV3NanoOmniAgent(model="policy", max_steps=2, parse_retries=1)

    def truncated(_payload, _model):
        return {
            "content": "## Action:\nClick.\n## Code:\n```python\npyautogui.click(1, 2)",
            "raw_content": "truncated sampled response",
            "prompt_token_ids": [1],
            "generation_token_ids": [2],
            "generation_log_probs": [-0.1],
            "finish_reason": "length",
        }

    agent.call_llm = truncated  # type: ignore[method-assign]

    response, actions, info = agent.predict("Try the task.", {"screenshot": b"fake-png"})

    assert "finish_reason='length'" in response
    assert actions == []
    # finish_reason is checked before the parser runs, so this is the only
    # failure kind that really means "the sampler hit its token budget".
    assert info["agent_outcome"] == "model_output_truncated"
    assert info["agent_outcome_family"] == "model_response_invalid"
    assert info["parse_failure"]["last_failure_kind"] == "output_truncated"
    assert info["stop_rollout"] is True
    assert info["model_call_completed"] is True
    assert info["parse_failure"]["last_failure_stage"] == "response_parse"
    assert "mask_sample" not in info
    assert "termination_reason" not in info
    assert info["model_calls"][0]["accepted"] is False
    assert info["model_calls"][0]["response"]["finish_reason"] == "length"


def test_nemotron_agent_reports_model_transport_failure_as_a_fact() -> None:
    agent = NemotronV3NanoOmniAgent(model="policy", max_steps=2, parse_retries=1)

    def unavailable(_payload, _model):
        raise ConnectionError("policy endpoint unreachable")

    agent.call_llm = unavailable  # type: ignore[method-assign]

    response, actions, info = agent.predict("Try the task.", {"screenshot": b"fake-png"})

    assert response == "policy endpoint unreachable"
    assert actions == []
    assert info["agent_outcome"] == "model_call_failed"
    assert info["agent_outcome_family"] == "model_response_invalid"
    assert info["parse_failure"]["last_failure_kind"] == "transport_error"
    assert info["stop_rollout"] is True
    assert info["model_call_completed"] is False
    assert info["parse_failure"] == {
        "attempt_count": 1,
        "completed_model_call_count": 0,
        "last_failure_stage": "model_call",
        "last_failure_kind": "transport_error",
        "failure_kind_counts": {"transport_error": 1},
        "prompt_shrink_events": [],
        "last_error_type": "ConnectionError",
        "last_error": "policy endpoint unreachable",
    }
    assert "mask_sample" not in info


def test_nemotron_agent_retries_invalid_python_action() -> None:
    agent = NemotronV3NanoOmniAgent(model="policy", max_steps=2, parse_retries=2)
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
    _response, actions, info = agent.predict("Click.", {"screenshot": b"fake-png"})

    assert calls == 2
    assert actions == ["pyautogui.click(960, 540)"]
    assert [call["accepted"] for call in info["model_calls"]] == [False, True]
    assert [call["parse_attempt"] for call in info["model_calls"]] == [1, 2]


def test_nemotron_agent_retries_invalid_python_with_feedback_and_lower_temperature(monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "model-io-agent.jsonl"
    monkeypatch.setenv("OSWORLD_MODEL_IO_LOG", str(log_path))
    agent = NemotronV3NanoOmniAgent(
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
            "content": '## Action:\nType a URL.\n## Code:\n```python\npyautogui.write("unterminated)\n```',
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


def test_nemotron_agent_warns_after_repeated_nontrivial_action() -> None:
    agent = NemotronV3NanoOmniAgent(
        model="policy",
        max_steps=10,
        max_image_history_length=1,
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
    agent = NemotronV3NanoOmniAgent(
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


def test_nemotron_agent_single_image_mode_keeps_text_history() -> None:
    agent = NemotronV3NanoOmniAgent(
        model="nemotron-3-nano-omni",
        max_steps=3,
        max_image_history_length=1,
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
    second_user_text = payloads[1]["messages"][-1]["content"][-1]["text"]
    assert "# Previous History Actions" in second_user_text
    assert "Click settings." in second_user_text


def test_nemotron_agent_sends_current_image_and_full_text_history() -> None:
    agent = NemotronV3NanoOmniAgent(
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
    assert "pyautogui.click(0.5, 0.5)" not in final_user_text
    assert "pyautogui.click(0.4, 0.4)" not in final_user_text
    assert "## Code:" not in final_user_text
