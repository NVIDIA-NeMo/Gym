# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.web.actions import ActionParseError, parse_model_action
from nemo_gym.web.models import WebActionProfile


def test_parses_fenced_browsergym_action() -> None:
    action = parse_model_action(
        "Thought: open the result\nAction:\n```python\nclick('a42')\n```",
        WebActionProfile.BROWSERGYM_HIGHLEVEL,
    )

    assert action.name == "click"
    assert action.script == "click('a42')"
    assert action.arguments["args"] == ["a42"]
    assert not action.terminal


def test_rejects_non_literal_or_arbitrary_python() -> None:
    with pytest.raises(ActionParseError, match="literal"):
        parse_model_action("click(get_target())", WebActionProfile.BROWSERGYM_HIGHLEVEL)
    with pytest.raises(ActionParseError, match="direct function call"):
        parse_model_action("import os", WebActionProfile.BROWSERGYM_HIGHLEVEL)


def test_translates_webvoyager_type_and_submit() -> None:
    action = parse_model_action("Action: Type [17]; [vegetarian lasagna]", WebActionProfile.WEBVOYAGER_LEGACY)

    assert action.name == "multi_action"
    assert action.script == "fill('17', 'vegetarian lasagna')\nkeyboard_press('Enter')"
    assert action.arguments["calls"][1]["name"] == "keyboard_press"


def test_translates_webvoyager_answer_to_terminal_action() -> None:
    action = parse_model_action("Action: ANSWER; [The result is 42]", WebActionProfile.WEBVOYAGER_LEGACY)

    assert action.name == "send_msg_to_user"
    assert action.terminal
    assert action.answer == "The result is 42"
