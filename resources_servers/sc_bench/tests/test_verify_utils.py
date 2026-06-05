# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from resources_servers.sc_bench.verify_utils import (
    compute_episode_reward,
    extract_tool_trace_from_response,
    extract_trade_order_id,
    get_verifier_fields,
    parse_tool_arguments,
    strip_thinking_traces,
)


class TestThinkingStrip:
    def test_strip_closed_tags(self) -> None:
        text = '<think>reasoning</think>{"order_id": "T1001"}'
        assert strip_thinking_traces(text) == '{"order_id": "T1001"}'

    def test_strip_orphan_close_tag(self) -> None:
        text = 'hidden reasoning</thinking>{"fulfillment_id": "FO1"}'
        assert strip_thinking_traces(text) == '{"fulfillment_id": "FO1"}'

    def test_parse_arguments_with_thinking(self) -> None:
        args = '<thinking>plan</thinking>{"order_id": "T1003"}'
        assert parse_tool_arguments(args) == {"order_id": "T1003"}


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
                arguments="<thinking>x</thinking>" + json.dumps({"order_id": "T1003"}),
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
