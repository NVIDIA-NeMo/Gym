import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from resources_servers.sc_bench.supchain_tools import (
    check_fake_shipping,
    get_cancel_error_code,
    get_cancel_scenes,
    get_error_reason,
    get_fulfillment_status,
    get_warehouse_error_details,
    get_warehouse_status,
    query_buyer_and_related,
)


def extract_trade_order_id(question: str) -> Optional[str]:
    """
    Extract trade_order_id from natural language question.
    Expected format like 'T1001', 'T123456', case-insensitive.
    """
    if not question:
        return None
    # Normalize and search for pattern like T12345
    # Use word boundary to avoid matching part of other tokens
    m = re.search(r"T\d+", question.upper())
    if m:
        return m.group(0)
    return None


def _group_warehouse_ids(related_items: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Group warehouse_order_id by fulfillment_id from related_item list returned by query_buyer_and_related.
    related_items element example: {'fulfillment_id': 'FO2001', 'warehouse_order_id': 'W3001'}
    """
    grouped: Dict[str, List[str]] = defaultdict(list)
    for it in related_items:
        fid = it.get("fulfillment_id")
        wid = it.get("warehouse_order_id")
        if fid and wid:
            grouped[fid].append(wid)
        elif fid and wid is None:
            # Ensure fulfillment_id key exists even without warehouse id
            _ = grouped[fid]
    return dict(grouped)


def process_fulfillment(
    fulfillment_id: str, warehouse_order_ids: List[str], trade_order_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute conditional tool call logic for a single fulfillment_id.

    Flow:
      - 1 -> 2 -> 3 -> 4
        If status == 'cancelled': Get cancellation scene, then get cancellation error code.
        Additionally, if cancelType is None/unknown/other, check fake shipping (6).

      - 1 -> 2 -> 5
        If status == 'error': Get error reason.

      - Always execute:
        For each warehouse_order_id, get warehouse-level status and error code via tool.get_warehouse_status,
        and include in results.
    """
    result: Dict[str, Any] = {
        "fulfillment_order_id": fulfillment_id,
        "trade_order_id": trade_order_id,
    }

    # Step 2: Status
    status_resp = get_fulfillment_status(fulfillment_id)
    status = (status_resp or {}).get("status")
    # Normalize British "cancelled" to American "canceled" for consistency
    if isinstance(status, str) and status.strip().lower() == "cancelled":
        status = "canceled"
    # Match database field name
    result["biz_status"] = status

    if status == "canceled":
        # Step 3: Cancellation scene
        scene_resp = get_cancel_scenes(fulfillment_id) or {}
        cancel_type = scene_resp.get("cancelType")
        result["cancel_type"] = cancel_type
        result["timeout_flag"] = None

        # Step 4: Cancellation reason
        cancel_reason_resp = get_cancel_error_code(fulfillment_id) or {}
        result["reason_code"] = cancel_reason_resp.get("cancelErrorCode")
        result["reason_text"] = cancel_reason_resp.get("cancelErrorMsg")

        # Optional Step 6: If scene is unknown/other/None, check fake shipping
        normalized_type = str(cancel_type).strip().lower() if cancel_type is not None else None
        if normalized_type in (None, "", "unknown", "other", "others", "other", "unknown"):
            _ = check_fake_shipping(fulfillment_id) or {}
            # Intentionally omit non-schema fields; do not include 'fakeShippingFlag' in output

    elif status == "error":
        # Step 5: Error reason
        err_resp = get_error_reason(fulfillment_id) or {}
        # Match ErrorLogs database field name
        result["code"] = err_resp.get("code")
        result["text"] = err_resp.get("text")

    # Step 7: Warehouse order status + error for each warehouse_order_id
    warehouse_details: List[Dict[str, Any]] = []
    for wid in warehouse_order_ids or []:
        w_resp = get_warehouse_status(fulfillment_id, wid) or {}
        # Prefer error details from ErrorLogs via composite key; fallback to WarehouseOrders.error_code
        err_detail = get_warehouse_error_details(fulfillment_id, wid) or {}
        err_code = err_detail.get("code") if err_detail.get("code") not in (None, "") else w_resp.get("error")
        warehouse_details.append(
            {
                "warehouse_order_id": wid,
                "fulfillment_order_id": fulfillment_id,
                "status": w_resp.get("status"),
                "error_code": err_code,
            }
        )

    if warehouse_details:
        result["warehouse_orders"] = warehouse_details

    return result


def process_fulfillment_standard(
    fulfillment_id: str, trade_order_id: Optional[str], warehouse_order_ids: List[str]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Build standard format fulfillment and warehouse order entries based on tool output.

    Standard fields:
      Fulfillment order:
        - fulfillment_id, trade_order_id, status, errorCode, errorText
        - cancel_type, reason_code, reason_text, timeout_flag
      Warehouse order:
        - warehouse_order_id, fulfillment_id, status, errorCode, errorText
    """

    # Status normalization helper function
    def _norm_status(val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        s = str(val).strip().lower()
        return "canceled" if s == "cancelled" else s

    # cancel_type normalization helper function (uppercase with underscores, consistent with evaluation.parse_tool_step)
    def _norm_scene(scene: Optional[str]) -> Optional[str]:
        if scene is None:
            return None
        s = str(scene).strip().upper()
        s = s.replace("-", "_").replace(" ", "_")
        return re.sub(r"_+", "_", s)

    # Fulfillment order base structure
    fulfillment: Dict[str, Any] = {
        "fulfillment_id": fulfillment_id,
        "trade_order_id": trade_order_id,
        "status": None,
        "errorCode": None,
        "errorText": None,
        "cancel_type": None,
        "reason_code": None,
        "reason_text": None,
        "timeout_flag": None,
    }

    # Step: Status
    status_resp = get_fulfillment_status(fulfillment_id) or {}
    status = _norm_status(status_resp.get("status"))
    fulfillment["status"] = status

    # Branch: Canceled
    if status == "canceled":
        scene_resp = get_cancel_scenes(fulfillment_id) or {}
        fulfillment["cancel_type"] = _norm_scene(scene_resp.get("cancelType"))

        reason_resp = get_cancel_error_code(fulfillment_id) or {}
        fulfillment["reason_code"] = reason_resp.get("cancelErrorCode")
        fulfillment["reason_text"] = reason_resp.get("cancelErrorMsg")

        # Align errorCode/errorText with cancellation reason in cancellation flow
        fulfillment["errorCode"] = fulfillment["reason_code"]
        fulfillment["errorText"] = fulfillment["reason_text"]

        # Optional fake shipping check not included in standard output structure

    # Branch: Error
    elif status == "error":
        err_resp = get_error_reason(fulfillment_id) or {}
        fulfillment["errorCode"] = err_resp.get("code")
        fulfillment["errorText"] = err_resp.get("text")

    # Warehouse order details
    warehouse_entries: List[Dict[str, Any]] = []
    for wid in warehouse_order_ids or []:
        w_resp = get_warehouse_status(fulfillment_id, wid) or {}
        err_detail = get_warehouse_error_details(fulfillment_id, wid) or {}
        # Prefer detailed error code/text from logs; fallback to WarehouseOrders.error_code via get_warehouse_status "error"
        err_code = err_detail.get("code") if err_detail.get("code") not in (None, "") else w_resp.get("error")
        wh = {
            "warehouse_order_id": wid,
            "fulfillment_id": fulfillment_id,
            "status": _norm_status(w_resp.get("status")),
            "errorCode": err_code,
            "errorText": err_detail.get("text"),
        }
        warehouse_entries.append(wh)

    return fulfillment, warehouse_entries


def get_results_standard(question: str) -> Dict[str, Any]:
    """
    Return standardized structure:
    {
      "trade_orders": [
        {"trade_order_id": "T1006", "buyer_id": 90005}
      ],
      "fulfillment_orders": [
        {
          "fulfillment_id": "FO2016",
          "trade_order_id": "T1006",
          "status": "canceled",
          "errorCode": "INVENTORY_SHORTAGE",
          "errorText": "Seller canceled due to inventory shortage",
          "cancel_type": "BUYER",
          "reason_code": null,
          "reason_text": null,
          "timeout_flag": null
        }
      ],
      "warehouse_orders": [
        {
          "warehouse_order_id": "WO3032",
          "fulfillment_id": "FO2016",
          "status": "canceled",
          "errorCode": null,
          "errorText": null
        }
      ]
    }
    """
    trade_order_id = extract_trade_order_id(question)
    # If trade id is missing, return default empty standard object
    if not trade_order_id:
        return {
            "trade_orders": [{"trade_order_id": None, "buyer_id": None}],
            "fulfillment_orders": [],
            "warehouse_orders": [],
        }

    # Step 1: Buyer + related items
    rel_resp = query_buyer_and_related(trade_order_id) or {}
    buyer_id = rel_resp.get("buyer_id")
    related_items = rel_resp.get("related_item", []) or []

    # Group warehouse ids by fulfillment id
    grouped_by_fid = _group_warehouse_ids(related_items)

    fulfillment_orders: List[Dict[str, Any]] = []
    warehouse_orders: List[Dict[str, Any]] = []

    # Build fulfillment order + warehouse order lists
    for fid, wids in grouped_by_fid.items():
        fulfillment, wh_list = process_fulfillment_standard(fid, trade_order_id, wids)
        fulfillment_orders.append(fulfillment)
        warehouse_orders.extend(wh_list)

    return {
        "trade_orders": [{"trade_order_id": trade_order_id, "buyer_id": buyer_id}],
        "fulfillment_orders": fulfillment_orders,
        "warehouse_orders": warehouse_orders,
    }


def get_results(question: str) -> Dict[str, Any]:
    """
    Orchestrate tool calls for given question:
    - Extract trade_order_id
    - 1) Query buyer and related items (fulfillment_id, warehouse_order_id)
    - 2) Get status for each fulfillment_id
    - Based on status:
      * cancelled: 3) Get cancellation scene, 4) Get cancellation error code, optional 6) Check fake shipping
      * error: 5) Get error reason
      * other: Return status
    """
    trade_order_id = extract_trade_order_id(question)
    if not trade_order_id:
        return {
            "trade_order_id": None,
            "buyer_id": None,
            "fulfillments": [],
            "error": "trade_order_id_not_found",
        }

    # Step 1: Buyer + related items
    rel_resp = query_buyer_and_related(trade_order_id) or {}
    buyer_id = rel_resp.get("buyer_id")
    related_items = rel_resp.get("related_item", []) or []

    # Group warehouse order ids by fulfillment id
    grouped_by_fid = _group_warehouse_ids(related_items)

    # Return buyer_id and empty fulfillments even if no related items found
    fulfillments: List[Dict[str, Any]] = []
    for fid, wids in grouped_by_fid.items():
        fulfillments.append(process_fulfillment(fid, wids, trade_order_id))

    # If 'related_item' entries are zero but fulfillment IDs without warehouse orders exist,
    # try to discover fulfillment IDs from related_items missing wid captured in grouping above.
    # If grouping is empty but related_items contains fulfillment IDs without warehouse orders, they still exist as keys.
    # The grouping function above ensures key existence in this case; no additional handling needed here.

    return {
        "trade_order_id": trade_order_id,
        "buyer_id": buyer_id,
        "fulfillments": fulfillments,
    }


if __name__ == "__main__":
    # Simple manual test
    # Example questions:
    #   "Please help me check the fulfillment status of T1001"
    #   "What's going on with order T1005?"
    import sys

    q = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else "Please help me check the fulfillment status of T1023"
    from pprint import pprint

    res = get_results(q)
    pprint(res)

    # for i in range(1001, 1100 + 1):
    #     res = get_results(f"T{i}")
    #     # print(res)
    #     # break
    #     # store the result into a jsonl file
    #     with open("all_results.jsonl", "a", encoding="utf-8") as f:
    #         f.write(json.dumps(res) + "\n")
