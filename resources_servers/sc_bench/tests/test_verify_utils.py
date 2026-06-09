# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from resources_servers.sc_bench.verify_utils import (
    _assert_no_reasoning,
    compute_episode_reward,
    extract_tool_trace_from_response,
    extract_trade_order_id,
    get_verifier_fields,
    parse_tool_arguments,
)


class TestAssertNoReasoning:
    def test_passes_clean_text(self) -> None:
        _assert_no_reasoning('{"order_id": "T1001"}')
        _assert_no_reasoning("")

    @pytest.mark.parametrize(
        "text",
        [
            '<think>reasoning</think>{"order_id": "T1"}',
            'hidden</think>{"order_id": "T1"}',
            '<thinking>plan</thinking>{"order_id": "T1"}',
            'hidden</thinking>{"order_id": "T1"}',
        ],
    )
    def test_raises_on_reasoning_tags(self, text: str) -> None:
        with pytest.raises(AssertionError, match="reasoning tags"):
            _assert_no_reasoning(text)

    def test_parse_arguments_rejects_thinking(self) -> None:
        with pytest.raises(AssertionError, match="reasoning tags"):
            parse_tool_arguments('<thinking>plan</thinking>{"order_id": "T1003"}')


class TestVerifierMetadata:
    def test_get_verifier_fields(self) -> None:
        meta = {
            "trade_order_id": "T1001",
            "gt_lines": [{"trade_order_id": "T1001"}],
            "expected_result": {"fulfillments": []},
        }
        gt_lines, expected = get_verifier_fields(meta)
        assert len(gt_lines) == 1
        assert expected == {"fulfillments": []}

    def test_get_verifier_fields_empty(self) -> None:
        gt_lines, expected = get_verifier_fields(None)
        assert gt_lines == []
        assert expected == {}

    def test_get_verifier_fields_invalid_types(self) -> None:
        gt_lines, expected = get_verifier_fields({"gt_lines": "bad", "expected_result": "bad"})
        assert gt_lines == []
        assert expected == {}


class TestToolTrace:
    def test_extract_trade_order_id(self) -> None:
        assert extract_trade_order_id("Check trade order T1042 status") == "T1042"
        assert extract_trade_order_id("") is None
        assert extract_trade_order_id("no order here") is None

    def test_parse_tool_arguments_edge_cases(self) -> None:
        assert parse_tool_arguments({"order_id": "T1"}) == {"order_id": "T1"}
        assert parse_tool_arguments(42) == {}
        assert parse_tool_arguments("not-json") == {}

    def test_extract_tool_trace_pairs_calls(self) -> None:
        class Item:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        output = [
            Item(
                type="function_call",
                call_id="c1",
                name="query_buyer_and_related",
                arguments=json.dumps({"order_id": "T1003"}),
            ),
            Item(
                type="function_call_output",
                call_id="c1",
                output=json.dumps({"buyer_id": {"id": 1}, "related_item": []}),
            ),
        ]
        trace = extract_tool_trace_from_response(output)
        assert len(trace) == 1
        assert trace[0]["arguments"] == {"order_id": "T1003"}

    def test_extract_tool_trace_rejects_thinking_in_arguments(self) -> None:
        output = [
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "query_buyer_and_related",
                "arguments": "<thinking>x</thinking>" + json.dumps({"order_id": "T1003"}),
            },
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": json.dumps({"buyer_id": {"id": 1}, "related_item": []}),
            },
        ]
        with pytest.raises(AssertionError, match="reasoning tags"):
            extract_tool_trace_from_response(output)

    def test_extract_tool_trace_dict_output(self) -> None:
        output = [
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "query_buyer_and_related",
                "arguments": '{"order_id": "T1"}',
            },
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": {"buyer_id": {"id": 1}, "related_item": []},
            },
        ]
        trace = extract_tool_trace_from_response(output)
        assert trace[0]["output"] == {"buyer_id": {"id": 1}, "related_item": []}

    def test_extract_tool_trace_skips_orphan_output(self) -> None:
        output = [
            {"type": "function_call_output", "call_id": "missing", "output": "{}"},
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "query_buyer_and_related",
                "arguments": '{"order_id": "T1"}',
            },
            {"type": "function_call_output", "call_id": "c1", "output": "not-valid-json"},
        ]
        trace = extract_tool_trace_from_response(output)
        assert len(trace) == 1
        assert trace[0]["output"] == {"raw_output": "not-valid-json"}

    def test_compute_episode_reward(self) -> None:
        output = [
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "query_buyer_and_related",
                "arguments": '{"order_id": "T1001"}',
            },
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": json.dumps({"buyer_id": {"id": 1}, "related_item": []}),
            },
        ]
        assert compute_episode_reward(output, []) == 0.0
        assert compute_episode_reward([], [{"trade_order_id": "T1001"}]) == 0.0
