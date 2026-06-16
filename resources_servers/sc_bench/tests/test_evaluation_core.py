# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from resources_servers.sc_bench.evaluation import (
    compute_reward_from_tool_trace,
    is_nullish,
    normalize_status,
    standard_object_to_eval_prediction,
    tool_trace_to_standard_object,
)


class TestEvaluationHelpers:
    def test_is_nullish(self) -> None:
        assert is_nullish(None)
        assert is_nullish("null")
        assert not is_nullish("packing_in_progress")

    def test_normalize_status(self) -> None:
        assert normalize_status("In-Transit") == "in_transit"


class TestToolTraceToStandard:
    def test_reconstructs_trade_and_fulfillment(self) -> None:
        trace = [
            {
                "name": "query_buyer_and_related",
                "arguments": {"order_id": "T1003"},
                "output": {
                    "buyer_id": {"id": 90002},
                    "related_item": [{"fulfillment_id": "FO2007", "warehouse_order_id": "WO3013"}],
                },
            },
            {
                "name": "get_fulfillment_status",
                "arguments": {"fulfillment_id": "FO2007"},
                "output": {"status": "packing_in_progress"},
            },
            {
                "name": "get_warehouse_status",
                "arguments": {"fulfillment_id": "FO2007", "warehouse_order_id": "WO3013"},
                "output": {"status": "packing_in_progress", "error": None},
            },
        ]
        std = tool_trace_to_standard_object(trace)
        assert std["trade_order_id"] == "T1003"
        assert len(std["fulfillments"]) == 1
        assert std["fulfillments"][0]["biz_status"] == "packing_in_progress"

    def test_standard_object_to_eval_prediction(self) -> None:
        std = tool_trace_to_standard_object(
            [
                {
                    "name": "query_buyer_and_related",
                    "arguments": {"order_id": "T1003"},
                    "output": {
                        "buyer_id": {"id": 90002},
                        "related_item": [{"fulfillment_id": "FO2007", "warehouse_order_id": "WO3013"}],
                    },
                }
            ]
        )
        pred = standard_object_to_eval_prediction(std)
        assert pred["trade_order_id"] == "T1003"
        assert pred["buyer_id"]["id"] == 90002

    def test_compute_reward_from_tool_trace(self) -> None:
        trace = [
            {
                "name": "query_buyer_and_related",
                "arguments": {"order_id": "T1003"},
                "output": {
                    "buyer_id": {"id": 90002},
                    "related_item": [{"fulfillment_id": "FO2007", "warehouse_order_id": "WO3013"}],
                },
            },
            {
                "name": "get_warehouse_status",
                "arguments": {"fulfillment_id": "FO2007", "warehouse_order_id": "WO3013"},
                "output": {"status": "packing_in_progress", "error": None},
            },
        ]
        gt_lines = [
            {
                "trade_order_id": "T1003",
                "fulfillment_id": "FO2007",
                "cancel_scene": None,
                "buyer_cancel_reason": None,
                "warehouse_order_id": "WO3013",
                "warehouse_order_status": "packing_in_progress",
            }
        ]
        assert compute_reward_from_tool_trace(trace, gt_lines) == 1.0
