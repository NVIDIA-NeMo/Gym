# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from responses_api_agents.osworld_agent import canonical_agent
from responses_api_agents.osworld_agent import client as osworld_client
from responses_api_agents.osworld_agent.adapter_agents import _encode_image, normalize_nemotron_agent_options
from responses_api_agents.osworld_agent.canonical_agent import (
    CanonicalNemotronAgent,
    normalize_canonical_harness,
    retained_image_indices,
)


AGENT_SOURCE = """
SYSTEM_PROMPT_THINKING = "Think first. The password is {password}."
SYSTEM_PROMPT_NON_THINKING = "Act. The password is {password}."
PARSE_CALLS = []


def parse_response_to_cot_and_action(response, screen_size, coordinate_type, thinking):
    PARSE_CALLS.append((response, screen_size, coordinate_type, thinking))
    if "DONE" in response["content"]:
        return "Done", ["DONE"], {"code": "DONE", "thought": response["reasoning_content"]}
    return "Click", ["pyautogui.click(10, 20)"], {"code": "pyautogui.click(10, 20)", "thought": "t"}
"""

FEEDBACK_SOURCE = """
import json


def normalize_execution_result(result, model_command):
    return {"command": model_command, "returncode": result.get("returncode")}


def render_feedback_batch(items):
    return "result:\\n" + json.dumps(items)
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def harness(tmp_path: Path) -> Dict[str, Any]:
    for name in ("_nemo_gym_canonical_nemotron_agent", "_nemo_gym_canonical_execution_feedback"):
        sys.modules.pop(name, None)
    agent_path = tmp_path / "mm_agents" / "nvidia" / "nemotron_agent.py"
    feedback_path = tmp_path / "desktop_env" / "execution_feedback.py"
    agent_path.parent.mkdir(parents=True)
    feedback_path.parent.mkdir(parents=True)
    agent_path.write_text(AGENT_SOURCE)
    feedback_path.write_text(FEEDBACK_SOURCE)
    return {
        "root": str(tmp_path),
        "agent_sha256": _sha256(agent_path),
        "feedback_sha256": _sha256(feedback_path),
        "execution_feedback": True,
    }


def _options(harness: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    return {
        "canonical_harness": harness,
        "assistant_history_replay": "raw_full",
        "legacy_response_parser": True,
        "context_overflow_policy": "stop",
        "unterminated_episode_policy": "fail",
        "train_invalid_terminal_call": True,
        "validate_python_syntax": False,
        "parse_retries": 1,
        **overrides,
    }


def _agent(harness: Dict[str, Any], **overrides: Any) -> CanonicalNemotronAgent:
    return CanonicalNemotronAgent(model="policy", max_steps=30, client_password="pw", **_options(harness, **overrides))


def _reply(content: str) -> Dict[str, Any]:
    return {
        "content": content,
        "reasoning_content": "Reason",
        "raw_content": f"<think>Reason</think>{content}",
        "finish_reason": "stop",
    }


def _run_turns(agent: CanonicalNemotronAgent, count: int) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []

    def call_llm(payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        payloads.append(payload)
        return _reply(f"## Action:\nClick {len(payloads)}.")

    agent.call_llm = call_llm  # type: ignore[method-assign]
    for turn in range(count):
        agent.predict("Open the file.", {"screenshot": f"png-{turn}".encode()})
    return payloads


def _image_url(turn: int) -> str:
    return f"data:image/png;base64,{_encode_image(f'png-{turn}'.encode())}"


@pytest.mark.parametrize(
    ("image_count", "expected"),
    [
        (1, {0}),
        (8, set(range(8))),
        (9, {0, 6, 7, 8}),
        (16, {0, *range(6, 16)}),
        (17, {0, 14, 15, 16}),
        (25, {0, 22, 23, 24}),
    ],
)
def test_k8_s3_window(image_count: int, expected: set[int]) -> None:
    assert retained_image_indices(image_count, block_k=8, boundary_s=3) == expected


def test_k_s_window_never_exceeds_k_plus_s_images() -> None:
    counts = [len(retained_image_indices(n, block_k=8, boundary_s=3)) for n in range(1, 200)]
    assert max(counts) == 11


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"surprise": 1}, "Unknown canonical_harness"),
        ({"root": ""}, "root"),
        ({"agent_sha256": "abc"}, "agent_sha256"),
        ({"feedback_sha256": None}, "feedback_sha256"),
        ({"execution_feedback": "yes"}, "execution_feedback"),
        ({"image_block_k": 2, "image_boundary_s": 3}, "image_block_k"),
        ({"omission_marker": ""}, "omission_marker"),
    ],
)
def test_harness_options_are_validated(harness: Dict[str, Any], overrides: Dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        normalize_canonical_harness({**harness, **overrides})


def test_feedback_pin_is_optional_without_feedback(harness: Dict[str, Any]) -> None:
    normalized = normalize_canonical_harness({**harness, "execution_feedback": False, "feedback_sha256": None})
    assert "feedback_sha256" not in normalized


def test_contract_records_the_pins_and_window_but_not_the_path(harness: Dict[str, Any]) -> None:
    agent = _agent(harness)
    expected = normalize_nemotron_agent_options(_options(harness))

    assert agent.agent_contract["agent_options"] == expected
    assert expected["canonical_harness"] == {
        "agent_sha256": harness["agent_sha256"],
        "feedback_sha256": harness["feedback_sha256"],
        "execution_feedback": True,
        "image_block_k": 8,
        "image_boundary_s": 3,
        "omission_marker": "[Image omitted for text cleaning]",
    }


def test_a_changed_harness_file_is_rejected(harness: Dict[str, Any]) -> None:
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _agent({**harness, "agent_sha256": "0" * 64})


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"assistant_history_replay": "normalized"}, "raw_full"),
        ({"legacy_response_parser": False}, "legacy_response_parser"),
        ({"context_overflow_policy": "shrink", "train_invalid_terminal_call": False}, "context_overflow_policy"),
    ],
)
def test_incompatible_base_options_are_rejected(
    harness: Dict[str, Any], overrides: Dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _agent(harness, **overrides)


def test_first_turn_uses_the_harness_prompt_and_task_layout(harness: Dict[str, Any]) -> None:
    agent = _agent(harness)
    payloads = _run_turns(agent, 1)

    messages = payloads[0]["messages"]
    assert messages[0] == {"role": "system", "content": "Think first. The password is pw."}
    assert messages[1] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Task: Open the file.\n"},
            {"type": "image_url", "image_url": {"url": _image_url(0)}},
        ],
    }
    parsed_input, screen_size, coordinate_type, thinking = sys.modules[
        "_nemo_gym_canonical_nemotron_agent"
    ].PARSE_CALLS[-1]
    assert parsed_input == {"content": "## Action:\nClick 1.", "reasoning_content": "Reason"}
    assert (screen_size, coordinate_type, thinking) == ((1920, 1080), "relative", True)


def test_later_turns_wrap_the_screenshot_with_the_previous_step_feedback(harness: Dict[str, Any]) -> None:
    agent = _agent(harness)
    payloads: List[Dict[str, Any]] = []

    def call_llm(payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        payloads.append(payload)
        return _reply("## Action:\nClick.")

    agent.call_llm = call_llm  # type: ignore[method-assign]
    agent.predict("Open the file.", {"screenshot": b"png-0"})
    agent.record_execution_result("pyautogui.click(10, 20)", {"returncode": 0})
    agent.record_execution_result("pyautogui.press('enter')", None)
    agent.predict("Open the file.", {"screenshot": b"png-1"})
    agent.predict("Open the file.", {"screenshot": b"png-2"})

    feedback = 'result:\n[{"command": "pyautogui.click(10, 20)", "returncode": 0}]'
    second = payloads[1]["messages"]
    assert second[2] == {"role": "assistant", "content": "<think>Reason</think>## Action:\nClick."}
    assert second[3]["content"] == [
        {"type": "text", "text": f"Computer observation:\n{feedback}\n<tool_response>\n"},
        {"type": "image_url", "image_url": {"url": _image_url(1)}},
        {"type": "text", "text": "\n</tool_response>"},
    ]
    third = payloads[2]["messages"]
    assert third[3] == second[3]
    assert third[5]["content"][0] == {"type": "text", "text": canonical_agent._UNKNOWN_RESULT_PREFIX}


def test_feedback_names_the_command_the_model_wrote(harness: Dict[str, Any]) -> None:
    agent = _agent(harness)
    payloads = _run_turns(agent, 1)
    agent.record_execution_result("pyautogui.click(10, 20)  # rewritten by DesktopEnv", {"returncode": 0})
    agent.record_execution_result("pyautogui.press('enter')", {"returncode": 1})
    payloads.extend(_run_turns(agent, 1))

    feedback = (
        'result:\n[{"command": "pyautogui.click(10, 20)", "returncode": 0}, '
        '{"command": "pyautogui.press(\'enter\')", "returncode": 1}]'
    )
    assert payloads[1]["messages"][3]["content"][0]["text"] == f"Computer observation:\n{feedback}\n<tool_response>\n"


def test_without_feedback_every_later_turn_reports_an_unknown_result(harness: Dict[str, Any]) -> None:
    agent = _agent({**harness, "execution_feedback": False})
    agent.record_execution_result("pyautogui.click(10, 20)", {"returncode": 0})
    payloads = _run_turns(agent, 2)
    assert payloads[1]["messages"][3]["content"][0]["text"] == canonical_agent._UNKNOWN_RESULT_PREFIX


def test_dropped_screenshots_become_the_omission_marker_in_place(harness: Dict[str, Any]) -> None:
    agent = _agent(harness)
    payloads = _run_turns(agent, 9)

    ninth = payloads[8]["messages"]
    users = [message for message in ninth if message["role"] == "user"]
    assert len(users) == 9
    images = [index for index, user in enumerate(users) if any(p["type"] == "image_url" for p in user["content"])]
    assert images == [0, 6, 7, 8]
    assert users[3]["content"][1] == {"type": "text", "text": "[Image omitted for text cleaning]"}
    assert agent.snapshot_window["prompt_snapshot_count"] == 4
    assert agent.snapshot_window["snapshot_compaction_triggered"] is True
    assert agent.compacted_turns == (1, 2, 3, 4, 5)
    assert [m["role"] for m in ninth].count("assistant") == 8


def test_an_unparseable_reply_executes_fail(harness: Dict[str, Any]) -> None:
    agent = _agent(harness)
    agent.call_llm = lambda _payload, _model: _reply("no action here")  # type: ignore[method-assign]
    agent._parser = lambda *_args, **_kwargs: ("", [], {})

    _, actions, info = agent.predict("Open the file.", {"screenshot": b"png-0"})

    assert actions == ["FAIL"]
    assert info["forced_terminal"] == "model_response_invalid"


def test_runner_reports_each_execute_reply_to_the_agent() -> None:
    recorded: List[tuple[str, Any]] = []
    controller = SimpleNamespace(execute_python_command=lambda command: {"returncode": 0, "command": command})

    osworld_client._report_execution_results(controller, lambda command, result: recorded.append((command, result)))

    assert controller.execute_python_command("pyautogui.click(1, 2)") == {
        "returncode": 0,
        "command": "pyautogui.click(1, 2)",
    }
    assert recorded == [("pyautogui.click(1, 2)", {"returncode": 0, "command": "pyautogui.click(1, 2)"})]
