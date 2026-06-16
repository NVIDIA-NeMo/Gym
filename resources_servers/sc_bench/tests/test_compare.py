# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.sc_bench.compare import (
    _fulfillment_match,
    _warehouse_match,
    compute_episode_reward_full,
    structures_match,
)


class TestStructuresMatch:
    def test_match_success(self) -> None:
        expected = {
            "fulfillments": [
                {
                    "fulfillment_order_id": "FO2001",
                    "biz_status": "canceled",
                    "reason_text": "buyer canceled",
                    "warehouse_orders": [{"warehouse_order_id": "WO3001", "status": "packing_in_progress"}],
                }
            ]
        }
        predicted = {
            "fulfillments": [
                {
                    "fulfillment_order_id": "FO2001",
                    "biz_status": "canceled",
                    "cancelErrorMsg": "The buyer canceled order",
                    "warehouse_orders": [{"warehouse_order_id": "WO3001", "status": "packing_in_progress"}],
                }
            ]
        }
        assert structures_match(expected, predicted)

    def test_match_fail_status(self) -> None:
        expected = {"fulfillments": [{"fulfillment_order_id": "FO1", "biz_status": "error", "warehouse_orders": []}]}
        predicted = {
            "fulfillments": [{"fulfillment_order_id": "FO1", "biz_status": "delivered", "warehouse_orders": []}]
        }
        assert not structures_match(expected, predicted)

    def test_empty_structures(self) -> None:
        assert not structures_match({}, {"fulfillments": []})
        assert not structures_match({"fulfillments": []}, {})
        assert structures_match({"fulfillments": []}, {"fulfillments": [{"fulfillment_order_id": "FO1"}]}) is False

    def test_warehouse_match_branches(self) -> None:
        assert _warehouse_match({"warehouse_order_id": "WO1", "status": None}, [])
        expected_wh = {"warehouse_order_id": "WO1", "status": "delivered"}
        assert not _warehouse_match(expected_wh, [])
        assert not _warehouse_match(expected_wh, [{"warehouse_order_id": "WO1", "status": "error"}])
        assert not _warehouse_match(
            {"warehouse_order_id": "WO1", "error_code": "E1"},
            [{"warehouse_order_id": "WO1", "status": "delivered", "error": "E2"}],
        )
        assert _warehouse_match(
            {"warehouse_order_id": "WO1", "status": None},
            [{"warehouse_order_id": "WO2", "status": "delivered"}],
        )

    def test_fulfillment_match_branches(self) -> None:
        base = {"fulfillment_order_id": "FO1", "warehouse_orders": []}
        assert not _fulfillment_match({**base, "cancel_type": "buyer"}, {**base, "cancel_type": "seller"})
        assert not _fulfillment_match(
            {**base, "reason_text": "buyer canceled"},
            {**base, "reason_text": "seller timeout"},
        )
        assert not _fulfillment_match(
            {**base, "warehouse_orders": [{"warehouse_order_id": "WO1", "status": "delivered"}]},
            {**base, "warehouse_orders": [{"warehouse_order_id": "WO1", "status": "error"}]},
        )


class TestComputeEpisodeRewardFull:
    def _make_response(self, tool_trace: list) -> NeMoGymResponse:
        output = []
        for step in tool_trace:
            call_id = f"call_{step['step']}"
            output.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": step["name"],
                    "arguments": json.dumps(step["arguments"]),
                }
            )
            output.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(step["output"]),
                }
            )
        return NeMoGymResponse(
            id="",
            object="response",
            created_at=0.0,
            model="test",
            output=output,
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=False,
        )

    def test_wrong_trace_returns_zero(self) -> None:
        trace = [
            {
                "step": 1,
                "name": "query_buyer_and_related",
                "arguments": {"order_id": "T1001"},
                "output": {"buyer_id": {"id": 1}, "related_item": []},
            }
        ]
        response = self._make_response(trace)
        reward = compute_episode_reward_full(
            response.output,
            [
                {
                    "trade_order_id": "T1001",
                    "fulfillment_id": "FO1",
                    "warehouse_order_id": "WO1",
                    "warehouse_order_status": "delivered",
                }
            ],
            None,
        )
        assert reward == 0.0

    def test_expected_result_fallback(self) -> None:
        trace = [
            {
                "step": 1,
                "name": "query_buyer_and_related",
                "arguments": {"order_id": "T1003"},
                "output": {
                    "buyer_id": {"id": 90002},
                    "related_item": [{"fulfillment_id": "FO2007", "warehouse_order_id": "WO3013"}],
                },
            },
            {
                "step": 2,
                "name": "get_fulfillment_status",
                "arguments": {"fulfillment_id": "FO2007"},
                "output": {"status": "packing_in_progress"},
            },
            {
                "step": 3,
                "name": "get_warehouse_status",
                "arguments": {"fulfillment_id": "FO2007", "warehouse_order_id": "WO3013"},
                "output": {"status": "packing_in_progress", "error": None},
            },
        ]
        response = self._make_response(trace)
        expected = {
            "trade_order_id": "T1003",
            "buyer_id": {"id": 90002},
            "fulfillments": [
                {
                    "fulfillment_order_id": "FO2007",
                    "biz_status": "packing_in_progress",
                    "warehouse_orders": [{"warehouse_order_id": "WO3013", "status": "packing_in_progress"}],
                }
            ],
        }
        reward = compute_episode_reward_full(response.output, [], expected)
        assert reward == 1.0

    def test_empty_output_returns_zero(self) -> None:
        response = NeMoGymResponse(
            id="",
            object="response",
            created_at=0.0,
            model="test",
            output=[],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=False,
        )
        assert compute_episode_reward_full(response.output, [], {"fulfillments": []}) == 0.0
