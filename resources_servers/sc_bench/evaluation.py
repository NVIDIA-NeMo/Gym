# SPDX-FileCopyrightText: Copyright (c) 2025 AIDC-SupplyChain-AI
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/Damon-GSY/SC-bench/scripts/evaluation.py

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


WORD_NAN_PATTERN = re.compile(r'(?<!")\bNaN\b')
NAN_VALUE = math.nan


def preprocess_line_nan_to_null(line: str) -> str:
    """
    Replace standalone NaN tokens with JSON-null to allow strict JSON parsing.
    """
    return WORD_NAN_PATTERN.sub("null", line)


def parse_jsonl_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single JSONL line with lenient handling of NaN tokens.
    Returns parsed dict or None if the line is empty/invalid.
    """
    stripped = line.strip()
    if not stripped:
        return None
    patched = preprocess_line_nan_to_null(stripped)
    try:
        return json.loads(patched)
    except json.JSONDecodeError:
        return None


def is_nullish(value: Any) -> bool:
    """
    Determine if a value should be considered null/empty for evaluation.
    """
    if value is None:
        return True
    if isinstance(value, float):
        # NaN check: NaN != NaN
        return value != value
    if isinstance(value, str):
        txt = value.strip().lower()
        return txt in ("", "null", "none", "nan", "na")
    return False


def normalize_status(status: Optional[str]) -> Optional[str]:
    """
    Normalize status strings to a canonical form (lowercase, underscores).
    """
    if status is None:
        return None
    s = str(status).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s


def normalize_scene(scene: Optional[str]) -> Optional[str]:
    """
    Normalize cancel scene/type to canonical uppercase with underscores.
    """
    if scene is None:
        return None
    s = str(scene).strip().upper()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s


def normalize_simple_text(text: Optional[str]) -> Optional[str]:
    """
    Normalize free-form text for inclusion checks:
      - lowercased
      - non-alphanumeric chars converted to single spaces
      - collapsed consecutive spaces
    """
    if text is None:
        return None
    s = str(text).strip().lower()
    # Replace any sequence of non a-z0-9 with a single space
    s = re.sub(r"[^a-z0-9]+", " ", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def text_includes(expected: Optional[str], predicted: Optional[str]) -> bool:
    """
    Return True if predicted text includes the expected text (after normalization).
    Rules:
      - If expected is nullish or normalizes to empty, consider inclusion satisfied
        as long as predicted is not nullish.
      - Otherwise, the normalized expected must be a substring of normalized predicted.
    """
    if is_nullish(expected):
        # No constraint when ground truth is nullish; any predicted value is acceptable.
        return True
    if is_nullish(predicted):
        return False
    ne = normalize_simple_text(expected)
    np = normalize_simple_text(predicted)
    if not ne:
        # Expected normalizes to empty string; accept any non-nullish predicted text
        return not is_nullish(predicted)
    return (np or "").find(ne) != -1


def parse_tool_step(
    function_call_name: str,
    arguments: Dict[str, Any],
    output: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Parse a single tool step from the tool_trace and extract
    trade_order, fulfillment_order, and warehouse_order records as applicable.

    This function returns a tuple (trade_order, fulfillment_order, warehouse_order),
    where each element is either a populated dict or None. It uses ONLY the current
    step's inputs/outputs and does not maintain cross-step state.
    """
    trade_order: Optional[Dict[str, Any]] = None
    fulfillment_order: Optional[Dict[str, Any]] = None
    warehouse_order: Optional[Dict[str, Any]] = None

    name = (function_call_name or "").strip()

    # Helper: normalize "cancelled" -> "canceled" to match US spelling in the standard
    def _normalize_canceled(status: Optional[str]) -> Optional[str]:
        if status is None:
            return None
        s = str(status).strip().lower()
        if s == "cancelled":
            return "canceled"
        return s

    # Helper: uppercase scene/type with underscores
    def _normalize_scene(scene: Optional[str]) -> Optional[str]:
        if scene is None:
            return None
        s = str(scene).strip().upper().replace("-", "_").replace(" ", "_")
        return re.sub(r"_+", "_", s)

    out = output or {}
    args = arguments or {}

    if name == "query_buyer_and_related":
        tid = args.get("order_id")
        trade_order = {
            "trade_order_id": str(tid) if tid is not None else None,
            "buyer_id": out.get("buyer_id"),
        }

    elif name == "get_fulfillment_status":
        fid = args.get("fulfillment_id")
        status = _normalize_canceled(out.get("status"))
        fulfillment_order = {
            "fulfillment_id": str(fid) if fid is not None else None,
            "trade_order_id": None,  # can be filled by aggregator using earlier mapping
            "status": status,
            "errorCode": None,
            "errorText": None,
            "cancel_type": None,
            "reason_code": None,
            "reason_text": None,
            "timeout_flag": None,
        }

    elif name == "get_cancel_scenes":
        fid = args.get("fulfillment_id")
        cancel_type = _normalize_scene(out.get("cancelType"))
        fulfillment_order = {
            "fulfillment_id": str(fid) if fid is not None else None,
            "trade_order_id": None,
            "status": None,
            "errorCode": None,
            "errorText": None,
            "cancel_type": cancel_type,
            "reason_code": None,
            "reason_text": None,
            "timeout_flag": None,  # not provided by tools; keep null
        }

    elif name == "get_cancel_error_code":
        fid = args.get("fulfillment_id")
        reason_code = out.get("cancelErrorCode")
        reason_text = out.get("cancelErrorMsg")
        # In canceled flows, align errorCode/errorText with reason fields for the standard
        fulfillment_order = {
            "fulfillment_id": str(fid) if fid is not None else None,
            "trade_order_id": None,
            "status": None,
            "errorCode": reason_code,
            "errorText": reason_text,
            "cancel_type": None,
            "reason_code": reason_code,
            "reason_text": reason_text,
            "timeout_flag": None,
        }

    elif name == "get_error_reason":
        fid = args.get("fulfillment_id")
        fulfillment_order = {
            "fulfillment_id": str(fid) if fid is not None else None,
            "trade_order_id": None,
            "status": None,
            "errorCode": out.get("code"),
            "errorText": out.get("text"),
            "cancel_type": None,
            "reason_code": None,
            "reason_text": None,
            "timeout_flag": None,
        }

    elif name == "check_fake_shipping":
        # Not part of the requested standard output schema; ignore
        pass

    elif name == "get_warehouse_status":
        fid = args.get("fulfillment_id")
        wid = args.get("warehouse_order_id")
        warehouse_order = {
            "warehouse_order_id": str(wid) if wid is not None else None,
            "fulfillment_id": str(fid) if fid is not None else None,
            "status": _normalize_canceled(out.get("status")),
            "errorCode": out.get("error"),  # column is named 'error' in tool output
            "errorText": None,
        }

    elif name == "get_warehouse_error_details":
        fid = args.get("fulfillment_id")
        wid = args.get("warehouse_order_id")
        warehouse_order = {
            "warehouse_order_id": str(wid) if wid is not None else None,
            "fulfillment_id": str(fid) if fid is not None else None,
            "status": None,
            "errorCode": out.get("code"),
            "errorText": out.get("text"),
        }

    return (trade_order, fulfillment_order, warehouse_order)


def tool_trace_to_standard_object(tool_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reconstruct structured trade/fulfillment/warehouse state from a tool trace."""
    trace = tool_trace or []
    if not isinstance(trace, list):
        return {"trade_order_id": None, "buyer_id": {"id": None}, "fulfillments": []}

    current_tid: Optional[str] = None
    buyer_id_value: Optional[Any] = None
    fid_to_tid: Dict[str, str] = {}
    fulfills: Dict[str, Dict[str, Any]] = {}
    warehouses: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def ensure_fulfillment(fid: Optional[str]) -> Dict[str, Any]:
        fid_str = str(fid) if fid is not None else None
        if fid_str is None:
            return {
                "fulfillment_order_id": None,
                "warehouse_orders": [],
                "cancelScene": None,
                "cancelErrorMsg": None,
            }
        if fid_str not in fulfills:
            fulfills[fid_str] = {
                "fulfillment_order_id": fid_str,
                "trade_order_id": None,
                "biz_status": None,
                "warehouse_orders": [],
                "cancelScene": None,
                "cancelErrorMsg": None,
            }
        return fulfills[fid_str]

    def ensure_warehouse(fid: Optional[str], wid: Optional[str]) -> Dict[str, Any]:
        fid_str = str(fid) if fid is not None else None
        wid_str = str(wid) if wid is not None else None
        if fid_str is None or wid_str is None:
            return {
                "warehouse_order_id": None,
                "fulfillment_order_id": None,
                "status": None,
                "error_code": NAN_VALUE,
            }
        key = (fid_str, wid_str)
        if key not in warehouses:
            warehouses[key] = {
                "warehouse_order_id": wid_str,
                "fulfillment_order_id": fid_str,
                "status": None,
                "error_code": NAN_VALUE,
            }
        return warehouses[key]

    for step in trace:
        name = step.get("name")
        args = step.get("arguments", {}) or {}
        out = step.get("output") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        if not isinstance(args, dict):
            args = {}
        _ = parse_tool_step(name, arguments=args, output=out)

        if name == "query_buyer_and_related":
            tid = args.get("order_id")
            current_tid = str(tid) if tid is not None else current_tid
            buyer_id_value = (out or {}).get("buyer_id")
            if isinstance(buyer_id_value, dict) and "id" in buyer_id_value:
                buyer_id_value = buyer_id_value["id"]
            for it in (out or {}).get("related_item", []) or []:
                fid = it.get("fulfillment_id")
                wid = it.get("warehouse_order_id")
                if fid is not None:
                    fid_str = str(fid)
                    if current_tid is not None:
                        fid_to_tid[fid_str] = current_tid
                    ensure_fulfillment(fid_str)
                if fid is not None and wid is not None:
                    ensure_warehouse(fid, wid)
        elif name == "get_fulfillment_status":
            fid = args.get("fulfillment_id")
            fo = ensure_fulfillment(fid)
            status = normalize_status(out.get("status"))
            if status == "cancelled":
                status = "canceled"
            fo["biz_status"] = status
        elif name == "get_cancel_scenes":
            fid = args.get("fulfillment_id")
            fo = ensure_fulfillment(fid)
            fo["cancelScene"] = normalize_scene(out.get("cancelType"))
        elif name == "get_cancel_error_code":
            fid = args.get("fulfillment_id")
            fo = ensure_fulfillment(fid)
            fo["cancelErrorMsg"] = out.get("cancelErrorMsg")
        elif name == "get_warehouse_status":
            fid = args.get("fulfillment_id")
            wid = args.get("warehouse_order_id")
            wo = ensure_warehouse(fid, wid)
            status = normalize_status(out.get("status"))
            if status == "cancelled":
                status = "canceled"
            wo["status"] = status
            err = out.get("error")
            if err not in (None, ""):
                wo["error_code"] = err
        elif name == "get_warehouse_error_details":
            fid = args.get("fulfillment_id")
            wid = args.get("warehouse_order_id")
            wo = ensure_warehouse(fid, wid)
            code = out.get("code")
            if code not in (None, ""):
                wo["error_code"] = code

    for fid, fo in fulfills.items():
        if (fo.get("trade_order_id") in (None, "")) and (fid in fid_to_tid):
            fo["trade_order_id"] = fid_to_tid[fid]
    for (fid, wid), wh in warehouses.items():
        fulfills.setdefault(fid, ensure_fulfillment(fid))
        fulfills[fid]["warehouse_orders"].append(wh)
    if current_tid is None and fid_to_tid:
        current_tid = next(iter(fid_to_tid.values()))

    return {
        "trade_order_id": current_tid,
        "buyer_id": {"id": buyer_id_value},
        "fulfillments": list(fulfills.values()),
    }


def standard_object_to_eval_prediction(std_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a standard tool-trace object into evaluate() pred_map entry format."""
    tid = std_obj.get("trade_order_id")
    if tid is None:
        return {}
    tid_s = str(tid)
    buyer_id_obj = std_obj.get("buyer_id") or {}
    buyer_id_val = buyer_id_obj.get("id") if isinstance(buyer_id_obj, dict) else buyer_id_obj

    fuls_out: List[Dict[str, Any]] = []
    for f in std_obj.get("fulfillments") or []:
        if not isinstance(f, dict):
            continue
        fid = f.get("fulfillment_order_id") or f.get("fulfillment_id")
        if not fid:
            continue
        f_rec: Dict[str, Any] = {
            "fulfillment_id": str(fid),
            "status": f.get("biz_status"),
            "cancelScene": f.get("cancelScene"),
            "cancelErrorMsg": f.get("cancelErrorMsg"),
        }
        wh_list: List[Dict[str, Any]] = []
        for wh in f.get("warehouse_orders") or []:
            if not isinstance(wh, dict):
                continue
            wid = wh.get("warehouse_order_id")
            if not wid:
                continue
            wh_list.append(
                {
                    "warehouse_order_id": str(wid),
                    "status": wh.get("status"),
                    "errorCode": wh.get("error_code"),
                    "errorText": None,
                }
            )
        if wh_list:
            f_rec["warehouse_orders"] = wh_list
        fuls_out.append(f_rec)

    return {
        "trade_order_id": tid_s,
        "buyer_id": {"id": buyer_id_val},
        "fulfillments": fuls_out,
    }


def compute_reward_from_tool_trace(
    tool_trace: List[Dict[str, Any]],
    gt_lines: List[Dict[str, Any]],
) -> float:
    """Return 1.0 when all ground-truth flat lines match the tool trace, else 0.0."""
    if not gt_lines:
        return 0.0
    std_obj = tool_trace_to_standard_object(tool_trace)
    pred_entry = standard_object_to_eval_prediction(std_obj)
    tid = pred_entry.get("trade_order_id")
    if not tid:
        return 0.0
    pred_map = {tid: pred_entry}
    report = evaluate(pred_map, {}, gt_lines)
    metrics = report.get("metrics") or {}
    return 1.0 if metrics.get("line_match_rate") == 1.0 else 0.0


def load_predictions(pred_path: Path) -> List[Dict[str, Any]]:
    """
    Transform predictions JSONL with tool_trace into a list of standardized objects
    matching the desired schema per input line:

    {
      "trade_order_id": "T1003",
      "buyer_id": {"id": 90002},
      "fulfillments": [
        {
          "fulfillment_order_id": "FO2007",
          "trade_order_id": "T1003",
          "biz_status": "packing_in_progress",
          "warehouse_orders": [
            {
              "warehouse_order_id": "WO3013",
              "fulfillment_order_id": "FO2007",
              "status": "packing_in_progress",
              "error_code": NaN
            }
          ]
        }
      ]
    }
    """
    results: List[Dict[str, Any]] = []

    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = parse_jsonl_line(line)
            if not record:
                continue

            trace = record.get("tool_trace") or record.get("toolTrace") or []
            if not isinstance(trace, list):
                continue
            results.append(tool_trace_to_standard_object(trace))

    return results


def load_ground_truth(gt_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load ground truth JSONL into mapping by trade_order_id -> result dict, normalizing statuses.
    """
    gt_results: Dict[str, Dict[str, Any]] = {}
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = parse_jsonl_line(line)
            if not record:
                continue
            trade_id = record.get("trade_order_id")
            result = record.get("result")
            if not result:
                continue

            # Normalize statuses and IDs in result
            fulfillments = result.get("fulfillments") or []
            for ful in fulfillments:
                ful["status"] = normalize_status(ful.get("status"))
                f_ec = ful.get("errorCode")
                f_et = ful.get("errorText")
                ful["errorCode"] = None if is_nullish(f_ec) else str(f_ec)
                ful["errorText"] = None if is_nullish(f_et) else str(f_et)
                ids = ful.get("warehouse_order_ids") or []
                ful["warehouse_order_ids"] = [str(x) for x in ids]
                for wh in ful.get("warehouse_orders") or []:
                    wh["status"] = normalize_status(wh.get("status"))
                    ec = wh.get("errorCode")
                    et = wh.get("errorText")
                    wh["errorCode"] = None if is_nullish(ec) else str(ec)
                    wh["errorText"] = None if is_nullish(et) else str(et)

            # Ensure buyer_id.id exists
            buyer = result.get("buyer_id") or {}
            if "id" not in buyer:
                result["buyer_id"] = {"id": None}

            # Ensure embedded trade_order_id present
            if "trade_order_id" not in result:
                result["trade_order_id"] = trade_id

            if trade_id:
                gt_results[str(trade_id)] = result
            else:
                embedded_id = result.get("trade_order_id")
                if embedded_id:
                    gt_results[str(embedded_id)] = result
    return gt_results


def load_ground_truth_lines(gt_path: Path) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load ground truth JSONL preserving per-line records in a simple, flat schema:
    {
      "trade_order_id": "...",
      "fulfillment_id": "...",
      "cancel_scene": "BUYER",
      "buyer_cancel_reason": "...",
      "warehouse_order_id": "...",
      "warehouse_order_status": "packing_in_progress"
    }
    Also supports legacy 'result' payloads by expanding them to the flat schema.
    Returns (gt_by_id, gt_lines).
    """
    gt_by_id: Dict[str, Dict[str, Any]] = {}
    gt_lines: List[Dict[str, Any]] = []
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = parse_jsonl_line(line)
            if not record:
                continue

            # Support both dict and list records per line
            records = record if isinstance(record, list) else [record]
            for rec in records:
                if not isinstance(rec, dict):
                    continue

                if "result" in rec:
                    # Expand legacy rich result to flat records
                    trade_id = rec.get("trade_order_id") or (rec.get("result") or {}).get("trade_order_id")
                    result = rec.get("result") or {}
                    fulfillments = result.get("fulfillments") or []
                    for ful in fulfillments:
                        fid = ful.get("fulfillment_id")
                        if not fid:
                            continue
                        cancel_scene = normalize_scene(ful.get("cancelScene"))
                        buyer_reason = ful.get("cancelErrorMsg")
                        whs = ful.get("warehouse_orders") or []
                        for wh in whs:
                            wid = wh.get("warehouse_order_id")
                            if not wid:
                                continue
                            flat = {
                                "trade_order_id": str(trade_id) if trade_id is not None else None,
                                "fulfillment_id": str(fid),
                                "cancel_scene": cancel_scene,
                                "buyer_cancel_reason": None if is_nullish(buyer_reason) else str(buyer_reason),
                                "warehouse_order_id": str(wid),
                                "warehouse_order_status": normalize_status(wh.get("status")),
                            }
                            gt_lines.append(flat)
                            if trade_id is not None:
                                gt_by_id[str(trade_id)] = flat
                    continue

                # New standard format: expand nested fulfillments/warehouse_orders
                if "fulfillments" in rec and isinstance(rec.get("fulfillments"), list):
                    tid_std = rec.get("trade_order_id")
                    fulfillments_std = rec.get("fulfillments") or []
                    for ful in fulfillments_std:
                        fid_std = ful.get("fulfillment_order_id") or ful.get("fulfillment_id")
                        if not fid_std:
                            continue
                        cancel_scene = normalize_scene(ful.get("cancel_type"))
                        buyer_reason = ful.get("reason_text")
                        whs = ful.get("warehouse_orders") or []
                        for wh in whs:
                            wid_std = wh.get("warehouse_order_id")
                            if not wid_std:
                                continue
                            flat = {
                                "trade_order_id": str(tid_std) if tid_std is not None else None,
                                "fulfillment_id": str(fid_std),
                                "cancel_scene": cancel_scene,
                                "buyer_cancel_reason": None if is_nullish(buyer_reason) else str(buyer_reason),
                                "warehouse_order_id": str(wid_std),
                                "warehouse_order_status": normalize_status(wh.get("status")),
                            }
                            gt_lines.append(flat)
                            if tid_std is not None:
                                gt_by_id[str(tid_std)] = flat
                    # After expanding, skip to next record
                    continue

                # Preferred flat schema
                tid = rec.get("trade_order_id")
                fid = rec.get("fulfillment_id")
                wid = rec.get("warehouse_order_id")
                if not tid or not fid or not wid:
                    continue
                flat = {
                    "trade_order_id": str(tid),
                    "fulfillment_id": str(fid),
                    "cancel_scene": normalize_scene(rec.get("cancel_scene")),
                    "buyer_cancel_reason": None
                    if is_nullish(rec.get("buyer_cancel_reason"))
                    else str(rec.get("buyer_cancel_reason")),
                    "warehouse_order_id": str(wid),
                    "warehouse_order_status": normalize_status(rec.get("warehouse_order_status")),
                }
                gt_lines.append(flat)
                gt_by_id[str(tid)] = flat
    return gt_by_id, gt_lines


def load_questions(q_path: Path) -> Dict[str, str]:
    """
    Load mapping from trade_order_id -> question string from a JSONL file.
    Supports lines that are either single dicts or lists of dicts.
    """
    mapping: Dict[str, str] = {}
    with q_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = parse_jsonl_line(line)
            if not rec:
                continue
            items = rec if isinstance(rec, list) else [rec]
            for item in items:
                if not isinstance(item, dict):
                    continue
                tid = item.get("trade_order_id")
                q = item.get("question")
                if tid is not None and isinstance(q, str):
                    mapping[str(tid)] = q.strip()
    return mapping


def evaluate(
    pred_map: Dict[str, Dict[str, Any]],
    gt_map: Dict[str, Dict[str, Any]],
    gt_lines: List[Dict[str, Any]],
    questions_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Compare predictions to ground truth per JSONL line in the flat schema and
    collect simplified mismatch entries that only record:
      - question (if available)
      - ground_truth (flat record with expected values)
      - predict (flat record with predicted values for the same ids)
    """
    ground_truth_total_lines = len(gt_lines)
    matched_lines = 0
    unmatched_lines = 0
    mismatches: List[Dict[str, Any]] = []
    questions_map = questions_map or {}

    for rec in gt_lines:
        tid = rec.get("trade_order_id")
        fid = rec.get("fulfillment_id")
        wid = rec.get("warehouse_order_id")
        if tid is None or fid is None or wid is None:
            # skip malformed gt line
            continue
        tid = str(tid)
        fid = str(fid)
        wid = str(wid)

        expected_scene = normalize_scene(rec.get("cancel_scene"))
        expected_reason = rec.get("buyer_cancel_reason")
        if isinstance(expected_reason, str):
            expected_reason = expected_reason.strip()
        expected_wh_status = normalize_status(rec.get("warehouse_order_status"))

        gt_summary = {
            "fulfillment_id": fid,
            "cancel_scene": expected_scene,
            "buyer_cancel_reason": expected_reason,
            "warehouse_order_id": wid,
            "warehouse_order_status": expected_wh_status,
        }

        pred_summary = {
            "fulfillment_id": fid,
            "cancel_scene": None,
            "buyer_cancel_reason": None,
            "warehouse_order_id": wid,
            "warehouse_order_status": None,
        }

        pred = pred_map.get(tid)
        if pred is not None:
            # locate predicted fulfillment
            fobj = None
            for f in pred.get("fulfillments") or []:
                if str((f or {}).get("fulfillment_id")) == fid:
                    fobj = f
                    break
            if fobj is not None:
                pred_scene = normalize_scene(fobj.get("cancelScene"))
                pred_reason = fobj.get("cancelErrorMsg")
                pred_reason = pred_reason.strip() if isinstance(pred_reason, str) else pred_reason
                pred_summary["cancel_scene"] = pred_scene
                pred_summary["buyer_cancel_reason"] = pred_reason

                # locate predicted warehouse order entry
                wh_entry = None
                for wh in fobj.get("warehouse_orders") or []:
                    if str((wh or {}).get("warehouse_order_id")) == wid:
                        wh_entry = wh
                        break
                if wh_entry is not None:
                    pred_wh_status = normalize_status(wh_entry.get("status"))
                    pred_summary["warehouse_order_status"] = pred_wh_status

        # Matching logic per user's rule:
        # - If a ground truth field is null/empty, we do NOT penalize differences; prediction may have extra info.
        # - If ground truth provides a non-empty value, prediction MUST provide the key and its value must match
        #   (for free-form text, prediction must include the ground truth text; for enums/status, exact match after normalization).
        matched_scene = True if is_nullish(expected_scene) else (pred_summary["cancel_scene"] == expected_scene)
        matched_reason = (
            True
            if is_nullish(expected_reason)
            else (
                not is_nullish(pred_summary["buyer_cancel_reason"])
                and text_includes(expected_reason, pred_summary["buyer_cancel_reason"])
            )
        )
        matched_wh_status = (
            True if is_nullish(expected_wh_status) else (pred_summary["warehouse_order_status"] == expected_wh_status)
        )

        all_equal = matched_scene and matched_reason and matched_wh_status

        if all_equal:
            matched_lines += 1
        else:
            unmatched_lines += 1
            mismatches.append(
                {
                    "trade_order_id": tid,
                    "question": questions_map.get(tid),
                    "ground_truth": gt_summary,
                    "predict": pred_summary,
                }
            )

    line_match_rate = (matched_lines / ground_truth_total_lines) if ground_truth_total_lines else 0.0
    line_mismatch_rate = 1.0 - line_match_rate if ground_truth_total_lines else 0.0

    return {
        "counts": {
            "ground_truth_total_lines": ground_truth_total_lines,
            "matched_lines": matched_lines,
            "unmatched_lines": unmatched_lines,
        },
        "metrics": {
            "line_match_rate": line_match_rate,
            "line_mismatch_rate": line_mismatch_rate,
        },
        "mismatches": mismatches,
    }


def run_evaluation_for_file(
    pred_path: Path,
    gt_map_by_id: Dict[str, Dict[str, Any]],
    gt_lines: List[Dict[str, Any]],
    questions_map: Dict[str, str],
) -> Dict[str, Any]:
    """
    Run evaluation pipeline for a single predictions file:
      - load predictions
      - convert to evaluate()-compatible map
      - compute report via evaluate()
    """
    # Load predictions as a list of standardized dicts (one per input line)
    pred_list = load_predictions(pred_path)

    # Convert list-of-standard dicts to the evaluate() expected mapping by trade_order_id
    pred_map: Dict[str, Dict[str, Any]] = {}
    for obj in pred_list or []:
        if not isinstance(obj, dict):
            continue

        tid = obj.get("trade_order_id")
        if tid is None:
            continue
        tid_s = str(tid)

        buyer_id_obj = obj.get("buyer_id") or {}
        buyer_id_val = buyer_id_obj.get("id")

        fuls_input = obj.get("fulfillments") or []
        fuls_out: List[Dict[str, Any]] = []

        for f in fuls_input:
            if not isinstance(f, dict):
                continue
            fid = f.get("fulfillment_order_id")
            if not fid:
                continue

            f_rec: Dict[str, Any] = {
                "fulfillment_id": str(fid),
                "status": f.get("biz_status"),
                # Fields expected by evaluate(); extract from fulfillment object
                "cancelScene": f.get("cancelScene"),
                "cancelErrorMsg": f.get("cancelErrorMsg"),
            }

            wh_list: List[Dict[str, Any]] = []
            for wh in f.get("warehouse_orders") or []:
                if not isinstance(wh, dict):
                    continue
                wid = wh.get("warehouse_order_id")
                if not wid:
                    continue
                wh_list.append(
                    {
                        "warehouse_order_id": str(wid),
                        "status": wh.get("status"),
                        "errorCode": wh.get("error_code"),
                        "errorText": None,
                    }
                )
            if wh_list:
                f_rec["warehouse_orders"] = wh_list

            fuls_out.append(f_rec)

        pred_map[tid_s] = {
            "trade_order_id": tid_s,
            "buyer_id": {"id": buyer_id_val},
            "fulfillments": fuls_out,
        }

    # Compute report for this predictions file
    return evaluate(pred_map, gt_map_by_id, gt_lines, questions_map)


def iter_prediction_files(pred_input: Path, glob_pattern: str = "*.jsonl") -> List[Path]:
    """
    Resolve prediction inputs:
      - If a file path, return [file]
      - If a directory, return all matching files by glob (sorted)
      - Else, return empty list
    """
    if pred_input.is_dir():
        return sorted([p for p in pred_input.glob(glob_pattern) if p.is_file()])
    elif pred_input.is_file():
        return [pred_input]
    else:
        return []
