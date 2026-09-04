# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from nemo_gym.web.actions import (
    ActionParseError,
    _json_container_balance,
    parse_nano_omni_tool_calls,
)


def test_parses_native_computer_and_terminal_tool_calls() -> None:
    action = parse_nano_omni_tool_calls(
        [
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "computer",
                "arguments": '{"actions":[{"action":"left_click","coordinate":[0.25,0.75]}]}',
            },
            {
                "type": "function_call",
                "call_id": "call-2",
                "name": "terminate",
                "arguments": '{"status":"success","answer":"done"}',
            },
        ]
    )

    assert action.name == "computer_use_tool_calls"
    assert action.arguments["calls"][0]["arguments"]["actions"][0]["action"] == "left_click"
    assert action.terminal is True
    assert action.answer == "done"
    assert action.metadata["nano_omni_parse"]["recovered"] is False


def test_native_robust_mode_decodes_actions_string_and_records_recovery() -> None:
    action = parse_nano_omni_tool_calls(
        [
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "computer",
                "arguments": '{"actions":"[{\\"action\\":\\"left_click\\",\\"coordinate\\":[0.25,0.75]}]"}',
            }
        ],
        recovery="decode_string",
    )

    assert action.arguments["calls"][0]["arguments"]["actions"][0]["action"] == "left_click"
    assert action.metadata["nano_omni_parse"]["recovered"] is True
    assert action.metadata["nano_omni_parse"]["calls"][0]["recovery_mode"] == "decoded_inner_string"


def test_native_robust_mode_repairs_only_one_missing_closing_bracket() -> None:
    action = parse_nano_omni_tool_calls(
        [
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "computer",
                "arguments": (
                    '{"actions":"[{\\"action\\":\\"left_click\\",'
                    '\\"coordinate\\":[0.325,0.5155]},'
                    '{\\"action\\":\\"type\\",\\"text\\":\\"Stockholm\\"}"}'
                ),
            }
        ],
        recovery="repair_single_closing_bracket",
    )

    actions = action.arguments["calls"][0]["arguments"]["actions"]
    assert [item["action"] for item in actions] == ["left_click", "type"]
    assert action.metadata["nano_omni_parse"]["calls"][0]["recovery_mode"] == "closed_one_missing_bracket"


def test_native_alignment_mode_still_rejects_actions_string() -> None:
    with pytest.raises(ActionParseError, match="non-empty actions list"):
        parse_nano_omni_tool_calls(
            [
                {
                    "type": "function_call",
                    "name": "computer",
                    "arguments": '{"actions":"[{\\"action\\":\\"wait\\",\\"duration\\":1}]"}',
                }
            ]
        )


def test_native_alias_recovery_is_opt_in_and_records_nested_click_conversion() -> None:
    item = {
        "type": "function_call",
        "call_id": "call-click",
        "name": "computer",
        "arguments": '{"actions":[{"action":"click","coordinate":[0.25,0.75]}]}',
    }
    with pytest.raises(ActionParseError, match="unsupported native computer action"):
        parse_nano_omni_tool_calls([item])

    action = parse_nano_omni_tool_calls([item], alias_recovery="webvoyager_v3")

    call = action.arguments["calls"][0]
    assert call["name"] == "computer"
    assert call["arguments"]["actions"] == [{"action": "left_click", "coordinate": [0.25, 0.75]}]
    record = action.metadata["nano_omni_parse"]["calls"][0]
    assert record["original_tool"] == "computer"
    assert record["alias_recovery_modes"] == ["computer.click_to_left_click"]
    assert action.metadata["nano_omni_parse"]["recovered"] is True


@pytest.mark.parametrize("duration,expected", [(-4, 0.0), (45, 30.0)])
def test_native_alias_recovery_clamps_nested_wait_and_records_values(duration, expected) -> None:
    item = {
        "type": "function_call",
        "call_id": "call-wait",
        "name": "computer",
        "arguments": json.dumps({"actions": [{"action": "wait", "duration": duration}]}),
    }
    with pytest.raises(ActionParseError, match=r"must be in \[0, 30\]"):
        parse_nano_omni_tool_calls([item])

    action = parse_nano_omni_tool_calls([item], alias_recovery="webvoyager_v3")

    assert action.arguments["calls"][0]["arguments"]["actions"] == [{"action": "wait", "duration": expected}]
    record = action.metadata["nano_omni_parse"]["calls"][0]
    assert record["alias_recovery_modes"] == ["computer.wait_duration_clamped"]
    assert record["alias_recovery_details"] == [
        {
            "field": "computer.actions[0].duration",
            "original": duration,
            "normalized": expected,
            "minimum": 0,
            "maximum": 30,
        }
    ]


@pytest.mark.parametrize("duration", [True, "45", float("nan"), float("inf")])
def test_native_alias_recovery_rejects_unsafe_nested_wait_values(duration) -> None:
    item = {
        "type": "function_call",
        "name": "computer",
        "arguments": json.dumps({"actions": [{"action": "wait", "duration": duration}]}),
    }

    with pytest.raises(ActionParseError):
        parse_nano_omni_tool_calls([item], alias_recovery="webvoyager_v3")


@pytest.mark.parametrize(
    "name,arguments,expected_action,expected_payload,expected_mode",
    [
        (
            "click",
            '{"x":"0.25","y":"0.75"}',
            "left_click",
            {"coordinate": [0.25, 0.75]},
            "tool.click_xy_to_computer_left_click",
        ),
        (
            "left_click",
            '{"coordinate":"[0.4, 0.6]"}',
            "left_click",
            {"coordinate": [0.4, 0.6]},
            "tool.left_click_coordinate_to_computer_left_click",
        ),
        (
            "type",
            '{"text":"hello"}',
            "type",
            {"text": "hello"},
            "tool.type_to_computer_type",
        ),
        (
            "wait",
            '{"duration":"2"}',
            "wait",
            {"duration": 2.0},
            "tool.wait_to_computer_wait",
        ),
    ],
)
def test_native_alias_recovery_wraps_unambiguous_top_level_actions(
    name, arguments, expected_action, expected_payload, expected_mode
) -> None:
    with pytest.raises(ActionParseError, match="unsupported Nano Omni browser tool"):
        parse_nano_omni_tool_calls(
            [{"type": "function_call", "call_id": "call-alias", "name": name, "arguments": arguments}]
        )

    action = parse_nano_omni_tool_calls(
        [{"type": "function_call", "call_id": "call-alias", "name": name, "arguments": arguments}],
        alias_recovery="webvoyager_v3",
    )

    call = action.arguments["calls"][0]
    assert call["name"] == "computer"
    computer_action = call["arguments"]["actions"][0]
    assert computer_action.pop("action") == expected_action
    assert computer_action == expected_payload
    record = action.metadata["nano_omni_parse"]["calls"][0]
    assert record["original_tool"] == name
    assert record["alias_recovery_modes"] == [expected_mode]


@pytest.mark.parametrize("duration,expected", [("-2", 0.0), (90, 30.0)])
def test_native_alias_recovery_clamps_top_level_wait_and_records_values(duration, expected) -> None:
    action = parse_nano_omni_tool_calls(
        [
            {
                "type": "function_call",
                "call_id": "call-wait",
                "name": "wait",
                "arguments": json.dumps({"duration": duration}),
            }
        ],
        alias_recovery="webvoyager_v3",
    )

    assert action.arguments["calls"][0]["arguments"]["actions"] == [{"action": "wait", "duration": expected}]
    record = action.metadata["nano_omni_parse"]["calls"][0]
    assert record["alias_recovery_modes"] == [
        "tool.wait_to_computer_wait",
        "tool.wait_duration_clamped",
    ]
    assert record["alias_recovery_details"] == [
        {
            "field": "tool.wait.duration",
            "original": duration,
            "normalized": expected,
            "minimum": 0,
            "maximum": 30,
        }
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        '{"target":"Buy button"}',
        '{"x":"500","y":"300"}',
        '{"action":"double_click","coordinate":[0.4,0.6]}',
        '{"action":"type","coordinate":"[0.4,0.6]","text":"query"}',
    ],
)
def test_native_alias_recovery_rejects_ambiguous_or_non_normalized_clicks(arguments) -> None:
    with pytest.raises(ActionParseError):
        parse_nano_omni_tool_calls(
            [{"type": "function_call", "name": "click", "arguments": arguments}],
            alias_recovery="webvoyager_v3",
        )


def test_native_recovery_rejects_non_local_json_damage() -> None:
    with pytest.raises(ActionParseError, match="not eligible"):
        parse_nano_omni_tool_calls(
            [
                {
                    "type": "function_call",
                    "name": "computer",
                    "arguments": '{"actions":"[{\\"action\\":\\"left_click\\",\\"coordinate\\":[0.2,0.3}"}',
                }
            ],
            recovery="repair_single_closing_bracket",
        )


def test_nano_omni_parser_validates_complete_batch_and_batch_limit() -> None:
    with pytest.raises(ActionParseError, match=r"action\[1\].*coordinate"):
        parse_nano_omni_tool_calls(
            [
                {
                    "type": "function_call",
                    "name": "computer",
                    "arguments": (
                        '{"actions":['
                        '{"action":"left_click","coordinate":[0.2,0.3]},'
                        '{"action":"left_click","coordinate":[2,3]}]}'
                    ),
                }
            ]
        )
    with pytest.raises(ActionParseError, match="2-action batch limit"):
        parse_nano_omni_tool_calls(
            [
                {
                    "type": "function_call",
                    "name": "computer",
                    "arguments": (
                        '{"actions":['
                        '{"action":"wait","duration":1},'
                        '{"action":"wait","duration":1},'
                        '{"action":"wait","duration":1}]}'
                    ),
                }
            ],
            max_computer_actions=2,
        )


@pytest.mark.parametrize(
    "name,arguments,match",
    [
        ("navigate", '{"url":"example.com"}', "must use http"),
        ("tabs_focus", '{"tab_id":-1}', "non-negative integer"),
        ("terminate", '{"status":"done"}', "success or failure"),
    ],
)
def test_nano_omni_parser_validates_tool_arguments(name, arguments, match) -> None:
    with pytest.raises(ActionParseError, match=match):
        parse_nano_omni_tool_calls([{"type": "function_call", "name": name, "arguments": arguments}])


@pytest.mark.parametrize(
    "item,match",
    [
        ({"type": "function_call", "name": "shell", "arguments": "{}"}, "unsupported Nano Omni browser tool"),
        (
            {"type": "function_call", "name": "computer", "arguments": '{"actions":[{"action":"exec"}]}'},
            "unsupported native computer action",
        ),
    ],
)
def test_rejects_unsafe_native_tool_calls(item, match) -> None:
    with pytest.raises(ActionParseError, match=match):
        parse_nano_omni_tool_calls([item])


def test_json_container_balance_handles_escapes_and_early_closing() -> None:
    assert _json_container_balance('[{"text":"a\\"b"}') == (1, 0, False)
    assert _json_container_balance("]") == (-1, 0, False)


def _native_item(name, arguments):
    return {"type": "function_call", "name": name, "arguments": json.dumps(arguments)}


@pytest.mark.parametrize(
    ("item", "kwargs", "match"),
    [
        (_native_item("computer", {"actions": "not-json"}), {"recovery": "decode_string"}, "invalid JSON"),
        (
            _native_item("computer", {"actions": "[invalid"}),
            {"recovery": "repair_single_closing_bracket"},
            "remains invalid",
        ),
        (_native_item("computer", {"actions": ["click"]}), {}, "must be an object"),
        (_native_item("computer", {"actions": [{"action": "left_click", "coordinate": [True, 0.2]}]}), {}, "number"),
        (_native_item("computer", {"actions": [{"action": "left_click", "coordinate": [0.2]}]}), {}, "x and y"),
        (_native_item("computer", {"actions": [{"action": "type", "text": 3}]}), {}, "text must be a string"),
        (_native_item("computer", {"actions": [{"action": "key_press", "keys": []}]}), {}, "non-empty string list"),
        (
            _native_item("computer", {"actions": [{"action": "scroll", "scroll_parameters": None}]}),
            {},
            "must be an object",
        ),
        (
            _native_item(
                "computer",
                {
                    "actions": [
                        {"action": "scroll", "scroll_parameters": {"scroll_direction": "around", "scroll_amount": 1}}
                    ]
                },
            ),
            {},
            "direction is unsupported",
        ),
        (
            _native_item(
                "computer",
                {
                    "actions": [
                        {"action": "scroll", "scroll_parameters": {"scroll_direction": "down", "scroll_amount": -1}}
                    ]
                },
            ),
            {},
            "non-negative integer",
        ),
        (_native_item("navigate", {"url": ""}), {}, "non-empty string"),
        (_native_item("navigate", {"url": "back", "tab_id": True}), {}, "tab_id"),
        (_native_item("tabs_create", {"url": "file:///tmp/x"}), {}, "about:blank"),
        (_native_item("terminate", {"status": "success", "answer": 3}), {}, "answer must be a string"),
    ],
)
def test_nano_omni_parser_rejects_additional_invalid_shapes(item, kwargs, match) -> None:
    with pytest.raises(ActionParseError, match=match):
        parse_nano_omni_tool_calls([item], **kwargs)


def test_nano_omni_parser_accepts_drag_scroll_and_default_arguments() -> None:
    action = parse_nano_omni_tool_calls(
        [
            _native_item(
                "computer",
                {
                    "actions": [
                        {"action": "left_click_drag", "start_coordinate": [0.1, 0.2], "coordinate": [0.8, 0.9]},
                        {
                            "action": "scroll",
                            "coordinate": [0.5, 0.5],
                            "scroll_parameters": {"scroll_direction": "down", "scroll_amount": 2},
                        },
                    ]
                },
            ),
            {"type": "function_call", "name": "tabs_create", "arguments": None},
        ]
    )
    assert action.name == "computer_use_tool_calls"


def test_nano_omni_parser_rejects_transport_and_sequence_errors() -> None:
    with pytest.raises(ActionParseError, match="did not contain"):
        parse_nano_omni_tool_calls([{"type": "message", "content": "ignored"}])
    with pytest.raises(ActionParseError, match="invalid JSON arguments"):
        parse_nano_omni_tool_calls([{"type": "function_call", "name": "navigate", "arguments": "{"}])
    with pytest.raises(ActionParseError, match="arguments must be an object"):
        parse_nano_omni_tool_calls([{"type": "function_call", "name": "navigate", "arguments": "[]"}])
    with pytest.raises(ActionParseError, match="1-call limit"):
        parse_nano_omni_tool_calls(
            [_native_item("navigate", {"url": "back"}), _native_item("navigate", {"url": "forward"})],
            max_calls=1,
        )
    with pytest.raises(ActionParseError, match="terminate must be the final"):
        parse_nano_omni_tool_calls(
            [_native_item("terminate", {"status": "success"}), _native_item("navigate", {"url": "back"})]
        )


@pytest.mark.parametrize(
    ("name", "arguments", "match"),
    [
        ("click", {"x": "not-a-number", "y": 0.5}, "must be a number"),
        ("left_click", {"coordinate": "not-json"}, "JSON coordinate array"),
        ("left_click", {"coordinate": [0.5]}, "x and y"),
        ("wait", {"duration": "not-a-number"}, "must be a number"),
    ],
)
def test_native_alias_recovery_rejects_malformed_numeric_aliases(name, arguments, match) -> None:
    with pytest.raises(ActionParseError, match=match):
        parse_nano_omni_tool_calls(
            [_native_item(name, arguments)],
            alias_recovery="webvoyager_v3",
        )
