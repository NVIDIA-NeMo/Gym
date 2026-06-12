# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from resources_servers.sc_bench.evaluation import (
    compute_reward_from_tool_trace,
    is_nullish,
    normalize_scene,
    normalize_status,
    text_includes,
)


def _normalize_fulfillment(ful: Dict[str, Any]) -> Dict[str, Any]:
    fid = ful.get("fulfillment_order_id") or ful.get("fulfillment_id")
    return {
        "fulfillment_order_id": str(fid) if fid else None,
        "biz_status": normalize_status(ful.get("biz_status") or ful.get("status")),
        "cancel_type": normalize_scene(ful.get("cancel_type") or ful.get("cancelScene")),
        "reason_text": ful.get("reason_text") or ful.get("cancelErrorMsg"),
        "warehouse_orders": ful.get("warehouse_orders") or [],
    }


def _warehouse_match(expected_wh: Dict[str, Any], predicted_wh: Dict[str, Any]) -> bool:
    wid = expected_wh.get("warehouse_order_id")
    pred_wh = None
    for wh in predicted_wh if isinstance(predicted_wh, list) else []:
        if str(wh.get("warehouse_order_id")) == str(wid):
            pred_wh = wh
            break
    if pred_wh is None:
        return is_nullish(expected_wh.get("status"))

    expected_status = normalize_status(expected_wh.get("status"))
    predicted_status = normalize_status(pred_wh.get("status"))
    if not is_nullish(expected_status) and predicted_status != expected_status:
        return False

    expected_err = expected_wh.get("error_code") or expected_wh.get("errorCode")
    predicted_err = pred_wh.get("error_code") or pred_wh.get("errorCode") or pred_wh.get("error")
    if not is_nullish(expected_err) and str(predicted_err) != str(expected_err):
        return False
    return True


def _fulfillment_match(expected: Dict[str, Any], predicted: Dict[str, Any]) -> bool:
    exp = _normalize_fulfillment(expected)
    pred = _normalize_fulfillment(predicted)

    if not is_nullish(exp["biz_status"]) and pred["biz_status"] != exp["biz_status"]:
        return False
    if not is_nullish(exp["cancel_type"]) and pred["cancel_type"] != exp["cancel_type"]:
        return False
    if not is_nullish(exp["reason_text"]) and not text_includes(exp["reason_text"], pred["reason_text"]):
        return False

    for wh in exp["warehouse_orders"]:
        if not _warehouse_match(wh, pred["warehouse_orders"]):
            return False
    return True


def structures_match(expected: Dict[str, Any], predicted: Dict[str, Any]) -> bool:
    """Compare get_results-style fulfillment structures."""
    if not expected or not predicted:
        return False

    expected_fuls = expected.get("fulfillments") or []
    predicted_fuls = predicted.get("fulfillments") or []
    if not expected_fuls:
        return not predicted_fuls

    for exp_f in expected_fuls:
        fid = exp_f.get("fulfillment_order_id") or exp_f.get("fulfillment_id")
        pred_f = None
        for candidate in predicted_fuls:
            cand_id = candidate.get("fulfillment_order_id") or candidate.get("fulfillment_id")
            if str(cand_id) == str(fid):
                pred_f = candidate
                break
        if pred_f is None or not _fulfillment_match(exp_f, pred_f):
            return False
    return True


def compute_episode_reward_full(
    response_output: List[Any],
    gt_lines: List[Dict[str, Any]],
    expected_result: Optional[Dict[str, Any]],
) -> float:
    from resources_servers.sc_bench.evaluation import tool_trace_to_standard_object
    from resources_servers.sc_bench.verify_utils import extract_tool_trace_from_response

    tool_trace = extract_tool_trace_from_response(response_output)
    if not tool_trace:
        return 0.0

    if gt_lines:
        reward = compute_reward_from_tool_trace(tool_trace, gt_lines)
        if reward == 1.0:
            return 1.0

    if expected_result:
        predicted = tool_trace_to_standard_object(tool_trace)
        if structures_match(expected_result, predicted):
            return 1.0

    return 0.0
