# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from resources_servers.sc_bench.evaluation import (
    compute_reward_from_tool_trace,
    evaluate,
    is_nullish,
    iter_prediction_files,
    load_ground_truth,
    load_ground_truth_lines,
    load_predictions,
    load_questions,
    normalize_scene,
    normalize_simple_text,
    parse_jsonl_line,
    parse_tool_step,
    preprocess_line_nan_to_null,
    run_evaluation_for_file,
    standard_object_to_eval_prediction,
    text_includes,
    tool_trace_to_standard_object,
)


class TestJsonAndTextHelpers:
    def test_preprocess_line_nan_to_null(self) -> None:
        assert preprocess_line_nan_to_null('{"x": NaN}') == '{"x": null}'

    def test_parse_jsonl_line(self) -> None:
        assert parse_jsonl_line("") is None
        assert parse_jsonl_line("{bad json") is None
        assert parse_jsonl_line('{"a": 1}') == {"a": 1}

    def test_is_nullish_float_nan(self) -> None:
        assert is_nullish(float("nan"))

    def test_normalize_scene_and_text(self) -> None:
        assert normalize_scene("buyer-cancel") == "BUYER_CANCEL"
        assert normalize_simple_text("Hello, World!") == "hello world"

    def test_text_includes(self) -> None:
        assert text_includes(None, "anything")
        assert not text_includes("expected", None)
        assert text_includes("size received", "The size received was wrong")
        assert text_includes("", "non-empty")


class TestParseToolStep:
    def test_query_and_cancel_tools(self) -> None:
        trade, ful, wh = parse_tool_step(
            "query_buyer_and_related",
            {"order_id": "T1001"},
            {"buyer_id": {"id": 1}, "related_item": []},
        )
        assert trade["trade_order_id"] == "T1001"
        assert ful is None
        assert wh is None

        _, ful, _ = parse_tool_step(
            "get_fulfillment_status",
            {"fulfillment_id": "FO2001"},
            {"status": "cancelled"},
        )
        assert ful["status"] == "canceled"

        _, ful, _ = parse_tool_step(
            "get_cancel_scenes",
            {"fulfillment_id": "FO2001"},
            {"cancelType": "buyer"},
        )
        assert ful["cancel_type"] == "BUYER"

        _, ful, _ = parse_tool_step(
            "get_cancel_error_code",
            {"fulfillment_id": "FO2001"},
            {"cancelErrorCode": "X", "cancelErrorMsg": "reason"},
        )
        assert ful["reason_code"] == "X"

        _, ful, _ = parse_tool_step(
            "get_error_reason",
            {"fulfillment_id": "FO2018"},
            {"code": "FAKE_SHIP", "text": "flagged"},
        )
        assert ful["errorCode"] == "FAKE_SHIP"

        _, _, wh = parse_tool_step(
            "get_warehouse_status",
            {"fulfillment_id": "FO2001", "warehouse_order_id": "WO3001"},
            {"status": "packing_in_progress", "error": None},
        )
        assert wh["status"] == "packing_in_progress"

        _, _, wh = parse_tool_step(
            "get_warehouse_error_details",
            {"fulfillment_id": "FO2009", "warehouse_order_id": "WO3016"},
            {"code": "PACK_MISSING_LABEL", "text": "missing label"},
        )
        assert wh["errorCode"] == "PACK_MISSING_LABEL"

        assert parse_tool_step("check_fake_shipping", {"fulfillment_id": "FO1"}, {}) == (None, None, None)


class TestToolTraceReconstruction:
    def test_non_list_trace(self) -> None:
        assert tool_trace_to_standard_object({})["trade_order_id"] is None

    def test_string_arguments_and_error_fields(self) -> None:
        trace = [
            {
                "name": "query_buyer_and_related",
                "arguments": json.dumps({"order_id": "T1003"}),
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
                "output": {"status": "packing_in_progress", "error": "WARN"},
            },
            {
                "name": "get_warehouse_error_details",
                "arguments": {"fulfillment_id": "FO2007", "warehouse_order_id": "WO3013"},
                "output": {"code": "DETAIL", "text": "detail"},
            },
        ]
        std = tool_trace_to_standard_object(trace)
        assert std["trade_order_id"] == "T1003"
        wh = std["fulfillments"][0]["warehouse_orders"][0]
        assert wh["error_code"] == "DETAIL"

    def test_standard_object_without_tid(self) -> None:
        assert standard_object_to_eval_prediction({"buyer_id": {"id": 1}}) == {}


class TestLoaders:
    def test_load_ground_truth_lines_formats(self, tmp_path: Path) -> None:
        flat_path = tmp_path / "flat.jsonl"
        flat_path.write_text(
            json.dumps(
                {
                    "trade_order_id": "T1001",
                    "fulfillment_id": "FO2001",
                    "warehouse_order_id": "WO3001",
                    "warehouse_order_status": "packing_in_progress",
                    "buyer_cancel_reason": "size issue",
                    "cancel_scene": "BUYER",
                }
            )
            + "\n"
        )
        gt_by_id, gt_lines = load_ground_truth_lines(flat_path)
        assert len(gt_lines) == 1
        assert gt_by_id["T1001"]["fulfillment_id"] == "FO2001"

        legacy_path = tmp_path / "legacy.jsonl"
        legacy_path.write_text(
            json.dumps(
                {
                    "trade_order_id": "T1002",
                    "result": {
                        "trade_order_id": "T1002",
                        "fulfillments": [
                            {
                                "fulfillment_id": "FO2003",
                                "cancelScene": "BUYER",
                                "cancelErrorMsg": "mismatch",
                                "warehouse_orders": [{"warehouse_order_id": "WO3003", "status": "delivered"}],
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        _, legacy_lines = load_ground_truth_lines(legacy_path)
        assert legacy_lines[0]["warehouse_order_status"] == "delivered"

        nested_path = tmp_path / "nested.jsonl"
        nested_path.write_text(
            json.dumps(
                {
                    "trade_order_id": "T1003",
                    "fulfillments": [
                        {
                            "fulfillment_order_id": "FO2007",
                            "cancel_type": "BUYER",
                            "reason_text": "reason",
                            "warehouse_orders": [{"warehouse_order_id": "WO3013", "status": "packing_in_progress"}],
                        }
                    ],
                }
            )
            + "\n"
        )
        _, nested_lines = load_ground_truth_lines(nested_path)
        assert nested_lines[0]["trade_order_id"] == "T1003"

    def test_load_ground_truth_legacy_result(self, tmp_path: Path) -> None:
        gt_path = tmp_path / "gt.jsonl"
        gt_path.write_text(
            json.dumps(
                {
                    "trade_order_id": "T1004",
                    "result": {
                        "buyer_id": {},
                        "fulfillments": [
                            {
                                "fulfillment_id": "FO2009",
                                "status": "error",
                                "errorCode": None,
                                "errorText": None,
                                "warehouse_order_ids": ["WO3016"],
                                "warehouse_orders": [
                                    {
                                        "warehouse_order_id": "WO3016",
                                        "status": "error",
                                        "errorCode": "PACK_MISSING_LABEL",
                                        "errorText": None,
                                    }
                                ],
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        gt_map = load_ground_truth(gt_path)
        assert "T1004" in gt_map
        assert gt_map["T1004"]["buyer_id"]["id"] is None

    def test_load_predictions_and_questions(self, tmp_path: Path) -> None:
        pred_path = tmp_path / "pred.jsonl"
        pred_path.write_text(
            json.dumps(
                {
                    "tool_trace": [
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
                            "arguments": {
                                "fulfillment_id": "FO2007",
                                "warehouse_order_id": "WO3013",
                            },
                            "output": {"status": "packing_in_progress", "error": None},
                        },
                    ]
                }
            )
            + "\n"
        )
        preds = load_predictions(pred_path)
        assert preds[0]["trade_order_id"] == "T1003"

        q_path = tmp_path / "questions.jsonl"
        q_path.write_text(json.dumps({"trade_order_id": "T1003", "question": "Status for T1003?"}) + "\n")
        assert load_questions(q_path)["T1003"] == "Status for T1003?"


class TestEvaluatePipeline:
    def _gt_lines(self) -> list[dict]:
        return [
            {
                "trade_order_id": "T1001",
                "fulfillment_id": "FO2001",
                "cancel_scene": None,
                "buyer_cancel_reason": "size received",
                "warehouse_order_id": "WO3001",
                "warehouse_order_status": "packing_in_progress",
            }
        ]

    def _pred_map(self) -> dict:
        return {
            "T1001": {
                "trade_order_id": "T1001",
                "buyer_id": {"id": 90000},
                "fulfillments": [
                    {
                        "fulfillment_id": "FO2001",
                        "cancelScene": None,
                        "cancelErrorMsg": "The size received was incorrect",
                        "warehouse_orders": [{"warehouse_order_id": "WO3001", "status": "packing_in_progress"}],
                    }
                ],
            }
        }

    def test_evaluate_match_and_mismatch(self) -> None:
        report = evaluate(self._pred_map(), {}, self._gt_lines(), {"T1001": "question"})
        assert report["metrics"]["line_match_rate"] == 1.0

        bad_pred = {
            "T1001": {
                "trade_order_id": "T1001",
                "fulfillments": [
                    {
                        "fulfillment_id": "FO2001",
                        "warehouse_orders": [{"warehouse_order_id": "WO3001", "status": "delivered"}],
                    }
                ],
            }
        }
        bad_report = evaluate(bad_pred, {}, self._gt_lines())
        assert bad_report["metrics"]["line_match_rate"] == 0.0
        assert bad_report["mismatches"]

    def test_evaluate_skips_malformed_gt_line(self) -> None:
        report = evaluate(self._pred_map(), {}, [{"trade_order_id": None}])
        assert report["counts"]["ground_truth_total_lines"] == 1
        assert report["metrics"]["line_match_rate"] == 0.0

    def test_compute_reward_from_tool_trace_edges(self) -> None:
        assert compute_reward_from_tool_trace([], []) == 0.0
        trace = [
            {
                "name": "query_buyer_and_related",
                "arguments": {"order_id": "T1001"},
                "output": {"buyer_id": {"id": 1}, "related_item": []},
            }
        ]
        assert compute_reward_from_tool_trace(trace, self._gt_lines()) == 0.0

    def test_run_evaluation_for_file(self, tmp_path: Path) -> None:
        pred_path = tmp_path / "pred.jsonl"
        pred_path.write_text(
            json.dumps(
                {
                    "toolTrace": [
                        {
                            "name": "query_buyer_and_related",
                            "arguments": {"order_id": "T1001"},
                            "output": {
                                "buyer_id": {"id": 90000},
                                "related_item": [{"fulfillment_id": "FO2001", "warehouse_order_id": "WO3001"}],
                            },
                        },
                        {
                            "name": "get_cancel_error_code",
                            "arguments": {"fulfillment_id": "FO2001"},
                            "output": {
                                "cancelErrorCode": None,
                                "cancelErrorMsg": "The size received was incorrect",
                            },
                        },
                        {
                            "name": "get_warehouse_status",
                            "arguments": {
                                "fulfillment_id": "FO2001",
                                "warehouse_order_id": "WO3001",
                            },
                            "output": {"status": "packing_in_progress", "error": None},
                        },
                    ]
                }
            )
            + "\n"
        )
        report = run_evaluation_for_file(
            pred_path,
            {},
            self._gt_lines(),
            {"T1001": "question"},
        )
        assert report["metrics"]["line_match_rate"] == 1.0

    def test_iter_prediction_files(self, tmp_path: Path) -> None:
        pred_file = tmp_path / "a.jsonl"
        pred_file.write_text("{}\n")
        assert iter_prediction_files(pred_file) == [pred_file]
        assert iter_prediction_files(tmp_path) == [pred_file]
        assert iter_prediction_files(tmp_path / "missing.jsonl") == []


class TestEvaluationCoverageGaps:
    def test_helper_edge_cases(self) -> None:
        assert not is_nullish(0)
        assert normalize_simple_text(None) is None
        assert text_includes("---", "anything predicted here")

    def test_parse_tool_step_null_status_and_scene(self) -> None:
        _, ful, _ = parse_tool_step(
            "get_fulfillment_status",
            {"fulfillment_id": "FO1"},
            {"status": None},
        )
        assert ful["status"] is None

        _, ful, _ = parse_tool_step(
            "get_cancel_scenes",
            {"fulfillment_id": "FO1"},
            {"cancelType": None},
        )
        assert ful["cancel_type"] is None

    def test_tool_trace_malformed_steps_and_cancelled_status(self) -> None:
        assert tool_trace_to_standard_object("not-a-list")["fulfillments"] == []

        malformed = [
            {
                "name": "query_buyer_and_related",
                "arguments": "not-json",
                "output": {
                    "buyer_id": {"id": 1},
                    "related_item": [{"fulfillment_id": None, "warehouse_order_id": None}],
                },
            },
            {
                "name": "get_fulfillment_status",
                "arguments": ["not", "a", "dict"],
                "output": {"status": "cancelled"},
            },
        ]
        assert tool_trace_to_standard_object(malformed)["fulfillments"] == []

        cancelled = [
            {
                "name": "query_buyer_and_related",
                "arguments": {"order_id": "T1001"},
                "output": {
                    "buyer_id": {"id": 1},
                    "related_item": [{"fulfillment_id": "FO2001", "warehouse_order_id": "WO3001"}],
                },
            },
            {
                "name": "get_fulfillment_status",
                "arguments": {"fulfillment_id": "FO2001"},
                "output": {"status": "cancelled"},
            },
            {
                "name": "get_cancel_scenes",
                "arguments": {"fulfillment_id": "FO2001"},
                "output": {"cancelType": "BUYER"},
            },
            {
                "name": "get_warehouse_status",
                "arguments": {"fulfillment_id": "FO2001", "warehouse_order_id": "WO3001"},
                "output": {"status": "cancelled", "error": None},
            },
        ]
        std = tool_trace_to_standard_object(cancelled)
        assert std["fulfillments"][0]["biz_status"] == "canceled"
        assert std["fulfillments"][0]["warehouse_orders"][0]["status"] == "canceled"

    def test_standard_object_skips_invalid_children(self) -> None:
        pred = standard_object_to_eval_prediction(
            {
                "trade_order_id": "T1",
                "buyer_id": {"id": 1},
                "fulfillments": [
                    "bad",
                    {"fulfillment_order_id": None},
                    {
                        "fulfillment_order_id": "FO1",
                        "warehouse_orders": [
                            "bad",
                            {"warehouse_order_id": None},
                            {"warehouse_order_id": "WO1", "status": "ok"},
                        ],
                    },
                ],
            }
        )
        assert pred["fulfillments"][0]["fulfillment_id"] == "FO1"
        assert len(pred["fulfillments"][0]["warehouse_orders"]) == 1

    def test_compute_reward_without_trade_order_id(self) -> None:
        trace = [
            {
                "name": "get_warehouse_status",
                "arguments": {"fulfillment_id": "FO1", "warehouse_order_id": "WO1"},
                "output": {"status": "ok", "error": None},
            }
        ]
        assert (
            compute_reward_from_tool_trace(
                trace,
                [
                    {
                        "trade_order_id": "T1",
                        "fulfillment_id": "FO1",
                        "warehouse_order_id": "WO1",
                        "warehouse_order_status": "ok",
                    }
                ],
            )
            == 0.0
        )

    def test_loader_skip_paths(self, tmp_path: Path) -> None:
        pred_path = tmp_path / "pred.jsonl"
        pred_path.write_text("\n" + json.dumps({"tool_trace": "not-a-list"}) + "\n")
        assert load_predictions(pred_path) == []

        gt_path = tmp_path / "gt.jsonl"
        gt_path.write_text(json.dumps({"trade_order_id": "T9"}) + "\n" + json.dumps({"result": {}}) + "\n")
        assert load_ground_truth(gt_path) == {}

        embedded_path = tmp_path / "embedded.jsonl"
        embedded_path.write_text(
            json.dumps(
                {
                    "result": {
                        "trade_order_id": "T8",
                        "fulfillments": [{"fulfillment_id": "FO1", "warehouse_orders": []}],
                    }
                }
            )
            + "\n"
        )
        assert "T8" in load_ground_truth(embedded_path)

        lines_path = tmp_path / "lines.jsonl"
        lines_path.write_text(
            json.dumps([{"not": "a dict"}])
            + "\n"
            + json.dumps(
                {
                    "trade_order_id": "T7",
                    "result": {
                        "trade_order_id": "T7",
                        "fulfillments": [
                            {"fulfillment_id": None, "warehouse_orders": [{"warehouse_order_id": "WO1"}]}
                        ],
                    },
                }
            )
            + "\n"
            + json.dumps(
                {
                    "trade_order_id": "T6",
                    "fulfillments": [
                        {
                            "fulfillment_order_id": None,
                            "warehouse_orders": [{"warehouse_order_id": "WO1", "status": "ok"}],
                        },
                        {
                            "fulfillment_order_id": "FO2",
                            "warehouse_orders": [{"warehouse_order_id": None, "status": "ok"}],
                        },
                    ],
                }
            )
            + "\n"
            + json.dumps({"trade_order_id": "T5", "fulfillment_id": "FO1"})
            + "\n"
        )
        _, lines = load_ground_truth_lines(lines_path)
        assert lines == []

        q_path = tmp_path / "q.jsonl"
        q_path.write_text(json.dumps(["bad"]) + "\n" + json.dumps({"trade_order_id": "T1", "question": 123}) + "\n")
        assert load_questions(q_path) == {}

    def test_run_evaluation_for_file_skip_invalid_rows(self, tmp_path: Path) -> None:
        pred_path = tmp_path / "pred.jsonl"
        pred_path.write_text(
            json.dumps({"tool_trace": []})
            + "\n"
            + json.dumps(
                {
                    "tool_trace": [
                        {
                            "name": "query_buyer_and_related",
                            "arguments": {"order_id": None},
                            "output": {"buyer_id": {"id": 1}, "related_item": []},
                        },
                        {
                            "name": "get_fulfillment_status",
                            "arguments": {"fulfillment_id": None},
                            "output": {"status": "ok"},
                        },
                        {
                            "name": "get_warehouse_status",
                            "arguments": {"fulfillment_id": "FO1", "warehouse_order_id": None},
                            "output": {"status": "ok", "error": None},
                        },
                    ]
                }
            )
            + "\n"
        )
        report = run_evaluation_for_file(pred_path, {}, [], {})
        assert report["counts"]["ground_truth_total_lines"] == 0
