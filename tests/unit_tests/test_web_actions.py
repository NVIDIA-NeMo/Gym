# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from nemo_gym.web.actions import ActionParseError, parse_nano_omni_tool_calls


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
    assert action.metadata["nano_omni_parse"]["calls"][0]["computer_actions"] == 1


def test_nano_omni_parser_does_not_decode_quoted_actions() -> None:
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


def test_nano_omni_parser_does_not_complete_missing_json_delimiters() -> None:
    with pytest.raises(ActionParseError, match="invalid JSON arguments"):
        parse_nano_omni_tool_calls(
            [
                {
                    "type": "function_call",
                    "name": "computer",
                    "arguments": '{"actions":[{"action":"wait","duration":1}',
                }
            ]
        )


def test_nano_omni_parser_does_not_rewrite_nested_action_aliases() -> None:
    item = {
        "type": "function_call",
        "call_id": "call-click",
        "name": "computer",
        "arguments": '{"actions":[{"action":"click","coordinate":[0.25,0.75]}]}',
    }
    with pytest.raises(ActionParseError, match="unsupported native computer action"):
        parse_nano_omni_tool_calls([item])


@pytest.mark.parametrize("name", ["click", "left_click", "type", "wait"])
def test_nano_omni_parser_does_not_rewrite_top_level_tool_aliases(name) -> None:
    with pytest.raises(ActionParseError, match="unsupported Nano Omni browser tool"):
        parse_nano_omni_tool_calls(
            [
                {
                    "type": "function_call",
                    "name": name,
                    "arguments": "{}",
                }
            ]
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


def _native_item(name, arguments):
    return {"type": "function_call", "name": name, "arguments": json.dumps(arguments)}


@pytest.mark.parametrize(
    ("item", "kwargs", "match"),
    [
        (_native_item("computer", {"actions": "not-json"}), {}, "non-empty actions list"),
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
            "integer in",
        ),
        (
            _native_item(
                "computer",
                {
                    "actions": [
                        {"action": "scroll", "scroll_parameters": {"scroll_direction": "down", "scroll_amount": 51}}
                    ]
                },
            ),
            {},
            "integer in",
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


def test_nano_omni_reference_profile_accepts_more_than_eight_unmodified_calls() -> None:
    # A real rollout returned eleven valid calls in one response. A response
    # count limit is a profile choice, not a tool-protocol validity rule.
    calls = [
        {"type": "function_call", "call_id": f"call-{index}", "name": "navigate", "arguments": '{ "url": "back" }'}
        for index in range(11)
    ]
    before = json.dumps(calls)
    with pytest.raises(ActionParseError, match="8-call limit"):
        parse_nano_omni_tool_calls(calls)
    action = parse_nano_omni_tool_calls(calls, max_calls=None)
    assert len(action.arguments["calls"]) == 11
    assert [call["id"] for call in action.arguments["calls"]] == [call["call_id"] for call in calls]
    assert json.dumps(calls) == before
    assert action.terminal is False
    # Removing the count cap must not enable JSON repair or weaker validation.
    with pytest.raises(ActionParseError, match="invalid JSON arguments"):
        parse_nano_omni_tool_calls(
            calls + [{"type": "function_call", "name": "computer", "arguments": "{"}], max_calls=None
        )


@pytest.mark.parametrize("invalid_arguments", ['{"actions":"[]"}', '{"actions":['])
def test_invalid_later_call_exposes_only_complete_validated_prefix(invalid_arguments) -> None:
    calls = [
        {
            "type": "function_call",
            "call_id": "first",
            "name": "computer",
            "arguments": '{ "actions": [{"action":"left_click","coordinate":[0.2,0.3]}] }',
        },
        {"type": "function_call", "call_id": "bad", "name": "computer", "arguments": invalid_arguments},
        {"type": "function_call", "call_id": "unreached", "name": "navigate", "arguments": '{"url":"back"}'},
    ]
    before = json.dumps(calls)
    with pytest.raises(ActionParseError) as error:
        parse_nano_omni_tool_calls(calls, max_calls=None)
    prefix = error.value.validated_prefix
    assert prefix is not None
    assert prefix.terminal is False
    assert prefix.arguments["calls"] == [
        {"id": "first", "name": "computer", "arguments": json.loads(calls[0]["arguments"])}
    ]
    assert json.dumps(calls) == before


@pytest.mark.parametrize("prefix_kind", ["none", "at_limit", "over_limit", "terminate", "invalid_nested_action"])
def test_invalid_batch_does_not_expose_unsafe_or_over_budget_prefix(prefix_kind) -> None:
    good = {"type": "function_call", "name": "navigate", "arguments": '{"url":"back"}'}
    bad = {"type": "function_call", "name": "computer", "arguments": '{"actions":"[]"}'}
    prefix = {
        "none": [],
        "at_limit": [good],
        "over_limit": [good, good],
        "terminate": [{"type": "function_call", "name": "terminate", "arguments": '{"status":"success"}'}],
        "invalid_nested_action": [
            {
                "type": "function_call",
                "name": "computer",
                "arguments": '{"actions":[{"action":"wait","duration":0},{"action":"wait","duration":-1}]}',
            }
        ],
    }[prefix_kind]
    with pytest.raises(ActionParseError) as error:
        parse_nano_omni_tool_calls(prefix + [bad], max_calls=1 if prefix_kind in {"at_limit", "over_limit"} else None)
    assert error.value.validated_prefix is None
