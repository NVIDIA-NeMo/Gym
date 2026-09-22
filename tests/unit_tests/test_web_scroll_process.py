# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nemo_gym.web.actions import ActionParseError, parse_nano_omni_tool_calls
from nemo_gym.web.computer_use import nano_omni_tools
from nemo_gym.web.scroll_process import main, run_scroll


def _call(amount):
    return {
        "type": "function_call",
        "name": "computer",
        "arguments": {
            "actions": [
                {"action": "scroll", "scroll_parameters": {"scroll_direction": "down", "scroll_amount": amount}}
            ]
        },
    }


def test_relaxed_scroll_preserves_parser_value_and_default_is_strict():
    call = _call(123456)
    with pytest.raises(ActionParseError, match=r"\[0, 50\]"):
        parse_nano_omni_tool_calls([call])
    action = parse_nano_omni_tool_calls([call], max_scroll_amount=None)
    assert action.arguments["calls"][0]["arguments"] == call["arguments"]


@pytest.mark.parametrize("amount", [-1, True, 2.5, "100"])
def test_relaxed_scroll_still_rejects_invalid_amounts(amount):
    with pytest.raises(ActionParseError):
        parse_nano_omni_tool_calls([_call(amount)], max_scroll_amount=None)
    with pytest.raises(ValueError):
        run_scroll("up", amount, (100, 100), timeout=1)


def test_schema_is_selected_not_mutated():
    def amount(tools):
        action = next(t for t in tools if t["name"] == "computer")["parameters"]["properties"]["actions"]["items"]
        return action["properties"]["scroll_parameters"]["anyOf"][0]["properties"]["scroll_amount"]

    assert "maximum" not in amount(nano_omni_tools(max_scroll_amount=None))
    assert amount(nano_omni_tools())["maximum"] == 50
    assert amount(nano_omni_tools(max_scroll_amount=80))["maximum"] == 80


def test_subprocess_receives_exact_amount_and_deadline(monkeypatch):
    execute = Mock()
    monkeypatch.setattr(subprocess, "run", execute)
    run_scroll("left", 123456, (200, 300), timeout=7)
    args, kwargs = execute.call_args
    assert args[0] == [sys.executable, "-m", "nemo_gym.web.scroll_process"]
    assert json.loads(kwargs["input"]) == {"direction": "left", "amount": 123456, "point": [200, 300]}
    assert kwargs["timeout"] == 7 and kwargs["check"]
    assert "shell" not in kwargs


def test_subprocess_timeout_is_terminal_error_not_a_retry(monkeypatch):
    execute = Mock(side_effect=subprocess.TimeoutExpired("scroll", 7))
    monkeypatch.setattr(subprocess, "run", execute)
    with pytest.raises(TimeoutError, match="partial action not retried"):
        run_scroll("down", 100000, (100, 100), timeout=7)
    assert execute.call_count == 1


def test_child_failure_is_visible(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", Mock(side_effect=subprocess.CalledProcessError(1, "scroll", stderr="failed display"))
    )
    with pytest.raises(RuntimeError, match="failed display"):
        run_scroll("down", 60, (100, 100), timeout=7)
    with pytest.raises(ValueError, match="direction"):
        run_scroll("sideways", 60, (100, 100), timeout=7)


@pytest.mark.parametrize(
    "direction,signed,method",
    [("up", 81, "scroll"), ("down", -81, "scroll"), ("left", -81, "hscroll"), ("right", 81, "hscroll")],
)
def test_child_direction_and_amount(monkeypatch, direction, signed, method):
    gui = SimpleNamespace(moveTo=Mock(), scroll=Mock(), hscroll=Mock())
    monkeypatch.setitem(sys.modules, "pyautogui", gui)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"direction": direction, "amount": 81, "point": [1, 2]})))
    main()
    gui.moveTo.assert_called_once_with(1, 2)
    getattr(gui, method).assert_called_once_with(signed)
    assert gui.PAUSE == 0 and gui.FAILSAFE is False
