# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io

import pytest
from PIL import Image

from nemo_gym.web.models import WebImage, WebObservation
from responses_api_agents.web_agent.qwen_computer_use import (
    COLLAPSED_SCREENSHOT_TEXT,
    QwenPolicyState,
    build_system_prompt,
    parse_qwen_action,
    smart_resize,
)


def _screenshot() -> str:
    stream = io.BytesIO()
    Image.new("RGB", (64, 32), color=(17, 34, 51)).save(stream, format="PNG", compress_level=0)
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


SCREENSHOT = _screenshot()


def _xml(action: str, **parameters) -> str:
    serialized = "\n".join(f"<parameter={name}>\n{value}\n</parameter>" for name, value in parameters.items())
    return (
        f"Action: {action}\n<tool_call>\n<function=computer_use>\n"
        f"<parameter=action>\n{action}\n</parameter>\n{serialized}\n"
        "</function>\n</tool_call>"
    )


def test_qwen_prompt_uses_the_reference_xml_and_relative_coordinate_contract() -> None:
    prompt = build_system_prompt(1920, 1088, "relative")

    assert "<tool_call>" in prompt
    assert "<function=example_function_name>" in prompt
    assert "The screen's resolution is 1000x1000." in prompt
    assert '"name": "computer_use"' in prompt


def test_qwen_parser_normalizes_relative_coordinates_and_actions() -> None:
    action = parse_qwen_action(
        _xml("left_click", coordinate="[500, 250]"),
        coordinate_type="relative",
        original_size=(1920, 1080),
        processed_size=(1920, 1088),
    )

    computer_action = action.arguments["calls"][0]["arguments"]["actions"][0]
    assert computer_action["action"] == "left_click"
    assert computer_action["coordinate"] == pytest.approx([500 / 999, 250 / 999])
    assert action.metadata["policy_protocol"] == "qwen_xml_computer_use"


def test_qwen_parser_clamps_scroll_and_emits_terminal_answer() -> None:
    scroll = parse_qwen_action(
        _xml("scroll", pixels="100000"),
        coordinate_type="relative",
        original_size=(1920, 1080),
        processed_size=(1920, 1088),
    )
    parameters = scroll.arguments["calls"][0]["arguments"]["actions"][0]["scroll_parameters"]
    assert parameters == {"scroll_direction": "up", "scroll_amount": 50}

    terminal = parse_qwen_action(
        _xml("terminate", status="success", answer="done"),
        coordinate_type="relative",
        original_size=(1920, 1080),
        processed_size=(1920, 1088),
    )
    assert terminal.terminal is True
    assert terminal.answer == "done"


def test_qwen_parser_matches_reference_terminal_and_action_summary_semantics() -> None:
    no_action_line = _xml("left_click", coordinate="[500, 250]").replace("Action: left_click\n", "")
    action = parse_qwen_action(
        no_action_line,
        coordinate_type="relative",
        original_size=(1920, 1080),
        processed_size=(1920, 1088),
    )
    assert action.metadata["natural_language_action"] == "left click at (500, 250)"

    click_then_terminate = no_action_line + "\n" + _xml("terminate", status="success", answer="done")
    terminal = parse_qwen_action(
        click_then_terminate,
        coordinate_type="relative",
        original_size=(1920, 1080),
        processed_size=(1920, 1088),
    )
    assert terminal.terminal is True
    assert [call["name"] for call in terminal.arguments["calls"]] == ["terminate"]


def test_qwen_parser_preserves_reference_fallback_for_unknown_or_empty_key_actions() -> None:
    for text in (_xml("unsupported"), _xml("key", keys="[]")):
        action = parse_qwen_action(
            text,
            coordinate_type="relative",
            original_size=(1920, 1080),
            processed_size=(1920, 1088),
        )
        assert action.terminal is True
        assert action.arguments["calls"][0]["name"] == "terminate"


@pytest.mark.parametrize("keys", ['["enter"]', "enter", "ctrl+l"])
def test_qwen_parser_accepts_json_list_scalar_and_chord_keys(keys: str) -> None:
    action = parse_qwen_action(
        _xml("key", keys=keys),
        coordinate_type="relative",
        original_size=(1920, 1080),
        processed_size=(1920, 1088),
    )

    computer_action = action.arguments["calls"][0]["arguments"]["actions"][0]
    expected = ["enter"] if keys != "ctrl+l" else ["ctrl", "l"]
    assert computer_action == {"action": "key_press", "keys": expected}


def test_qwen_toolless_answer_is_judged_without_desktop_specific_phrase_rules() -> None:
    action = parse_qwen_action(
        "The requested product is not available in this country.",
        coordinate_type="relative",
        original_size=(1920, 1080),
        processed_size=(1920, 1088),
    )

    assert action.terminal is True
    assert action.arguments["calls"][0]["arguments"]["status"] == "success"
    assert action.answer == "The requested product is not available in this country."


def test_qwen_explicit_refusal_and_call_user_are_terminal_failures() -> None:
    refusal = parse_qwen_action(
        "I cannot complete this task because the page never loaded.",
        coordinate_type="relative",
        original_size=(1920, 1080),
        processed_size=(1920, 1088),
    )
    call_user = parse_qwen_action(
        _xml("call_user", text="Please provide an account"),
        coordinate_type="relative",
        original_size=(1920, 1080),
        processed_size=(1920, 1088),
    )

    assert refusal.arguments["calls"][0]["arguments"]["status"] == "failure"
    assert call_user.arguments["calls"][0]["arguments"] == {
        "status": "failure",
        "answer": "Please provide an account",
    }


def test_qwen_history_folds_old_screenshots_without_dropping_actions() -> None:
    state = QwenPolicyState(instruction="Find the answer", max_image_history=2, fold_size=1, history_n=10)
    for index in range(4):
        state.append_observation(WebObservation(screenshot=WebImage(data_url=SCREENSHOT)))
        if index < 3:
            state.record_response(
                f"Action: wait {index}",
                parse_qwen_action(
                    _xml("wait", time="1"),
                    coordinate_type="relative",
                    original_size=state.original_size,
                    processed_size=state.processed_size,
                ),
            )

    messages = state.messages()
    text = str([message.model_dump(mode="json") for message in messages])
    assert text.count(COLLAPSED_SCREENSHOT_TEXT) == 2
    assert text.count("data:image/png;base64") == 2
    assert "Action: wait 0" in text


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [(1080, 1920, (1088, 1920)), (32, 64, (64, 96))],
)
def test_qwen_smart_resize_uses_factor_32(height, width, expected) -> None:
    assert smart_resize(height, width) == expected
