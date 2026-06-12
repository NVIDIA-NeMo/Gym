# SPDX-FileCopyrightText: Copyright (c) 2025 AIDC-SupplyChain-AI
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/Damon-GSY/SC-bench (SupChain-Bench) src/tool.py

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


_DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "csv"
_data_dir: Path = _DEFAULT_DATA_DIR
_cache_data_dir: Path | None = None
_csv_cache: Dict[Tuple[str, Tuple[str, ...]], pd.DataFrame] = {}


def configure_data_dir(data_dir: Path) -> None:
    """Point tool implementations at a directory containing SC-bench CSV tables."""
    global _data_dir, _cache_data_dir, _csv_cache
    resolved = Path(data_dir)
    if _cache_data_dir != resolved:
        _csv_cache.clear()
        _cache_data_dir = resolved
    _data_dir = resolved


def clear_csv_cache() -> None:
    """Clear cached CSV tables (used in tests)."""
    global _csv_cache
    _csv_cache.clear()


def get_data_dir() -> Path:
    return _data_dir


def _files() -> Dict[str, Path]:
    base = _data_dir
    return {
        "TradeOrders": base / "TradeOrders.csv",
        "FulfillmentOrders": base / "FulfillmentOrders.csv",
        "WarehouseOrders": base / "WarehouseOrders.csv",
        "ErrorLogs": base / "ErrorLogs.csv",
        "CancellationContext": base / "CancellationContext.csv",
    }


def _read_csv(path: Path, usecols: List[str]) -> pd.DataFrame:
    cache_key = (str(path), tuple(usecols))
    if cache_key not in _csv_cache:
        _csv_cache[cache_key] = pd.read_csv(path, usecols=usecols)
    return _csv_cache[cache_key]


def _as_category(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def _parse_buyer(val: Any) -> Any:
    if pd.isna(val):
        return None
    if isinstance(val, (dict, list)):
        return val
    s = str(val).strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(s)
            return parsed
        except Exception:
            continue
    return s


def _map_status(raw) -> str:
    s = str(raw).strip().upper()
    if s in ("RECEIVING", "PICKING", "PACKING", "1"):
        return "packing_in_progress"
    if s in ("PACKED", "2"):
        return "packing_done"
    if s in ("SHIPPED", "DISPATCHED", "3"):
        return "dispatched"
    if s in ("IN_TRANSIT", "DELIVERING", "4"):
        return "in_transit"
    if s in ("DELIVERED", "5"):
        return "delivered"
    if s in ("ERROR", "FAIL", "9"):
        return "error"
    return "packing_in_progress"


# 1) order_id -> buyer_id + related items(fulfillment_id, warehouse_order_id)
def query_buyer_and_related(order_id: str) -> Dict[str, Any]:
    files = _files()
    t = _read_csv(files["TradeOrders"], ["trade_order_id", "buyer_id"])
    f = _read_csv(files["FulfillmentOrders"], ["fulfillment_order_id", "trade_order_id"])
    w = _read_csv(files["WarehouseOrders"], ["warehouse_order_id", "fulfillment_order_id"])

    _as_category(t, ["trade_order_id"])
    _as_category(f, ["fulfillment_order_id", "trade_order_id"])
    _as_category(w, ["warehouse_order_id", "fulfillment_order_id"])

    buyer_id = None
    fulfillment_ids: List[str] = []

    # Case 1: input is trade_order_id
    hit_t = t.loc[t["trade_order_id"] == order_id]
    if not hit_t.empty:
        buyer_id = _parse_buyer(hit_t["buyer_id"].iloc[0])
        fulfillment_ids = f.loc[f["trade_order_id"] == order_id, "fulfillment_order_id"].dropna().tolist()
    else:
        return {"buyer_id": None, "related_item": []}

    related: List[Dict[str, Any]] = []
    for fid in fulfillment_ids:
        wid_list = w.loc[w["fulfillment_order_id"] == fid, "warehouse_order_id"].dropna().tolist()
        for wid in wid_list:
            related.append({"fulfillment_id": fid, "warehouse_order_id": wid})

    return {"buyer_id": buyer_id, "related_item": related}


# 2) fulfillment_id -> aggregated status
def get_fulfillment_status(fulfillment_id: str) -> Dict[str, Any]:
    f = _read_csv(_files()["FulfillmentOrders"], ["fulfillment_order_id", "biz_status"])
    _as_category(f, ["fulfillment_order_id", "biz_status"])
    row = f.loc[f["fulfillment_order_id"] == fulfillment_id]
    if not row.empty:
        biz_status_str = str(row["biz_status"].iloc[0]).upper()
        if "CANCEL" in biz_status_str:
            return {"status": "cancelled"}
        # Check fulfillment-order-level ERROR status
        if "ERROR" in biz_status_str:
            return {"status": "error"}

    w = _read_csv(_files()["WarehouseOrders"], ["warehouse_order_id", "fulfillment_order_id", "status"])
    _as_category(w, ["warehouse_order_id", "fulfillment_order_id"])
    wsub = w.loc[w["fulfillment_order_id"] == fulfillment_id]
    mapped = [_map_status(s) for s in wsub["status"].tolist()]

    if any(s == "error" for s in mapped):
        return {"status": "error"}
    if any(s == "in_transit" for s in mapped):
        return {"status": "in_transit"}
    if any(s == "dispatched" for s in mapped):
        return {"status": "dispatched"}
    if len(mapped) > 0 and all(s == "delivered" for s in mapped):
        return {"status": "delivered"}
    if any(s == "packing_done" for s in mapped) and not any(s == "packing_in_progress" for s in mapped):
        return {"status": "packing_done"}
    return {"status": "packing_in_progress"}


# 3) fulfillment_id -> cancellation scene
def get_cancel_scenes(fulfillment_id: str) -> Dict[str, Any]:
    cc = _read_csv(_files()["CancellationContext"], ["entity_type", "entity_id", "cancel_type"])
    _as_category(cc, ["entity_type", "entity_id", "cancel_type"])
    rows = cc.loc[(cc["entity_type"] == "fulfillment_order") & (cc["entity_id"] == fulfillment_id)]
    if rows.empty:
        return {"cancelType": None}
    r = rows.iloc[0]
    return {"cancelType": r.get("cancel_type")}


# 4) fulfillment_id -> cancellation error code (specific cancellation reason)
def get_cancel_error_code(fulfillment_id: str) -> Dict[str, Any]:
    cc = _read_csv(_files()["CancellationContext"], ["entity_type", "entity_id", "reason_code", "reason_text"])
    _as_category(cc, ["entity_type", "entity_id", "reason_code"])
    rows = cc.loc[(cc["entity_type"] == "fulfillment_order") & (cc["entity_id"] == fulfillment_id)]
    if rows.empty:
        return {"cancelErrorCode": None, "cancelErrorMsg": None}
    r = rows.iloc[0]
    return {"cancelErrorCode": r.get("reason_code"), "cancelErrorMsg": r.get("reason_text")}


# 5) fulfillment_id -> error reason
def get_error_reason(fulfillment_id: str) -> Dict[str, Any]:
    """
    Get fulfillment-order-level error details.
    Uses composite key (entity_type, warehouse_order_id, fulfillment_order_id, code, text)
    from the ErrorLogs table.
    """
    logs = _read_csv(
        _files()["ErrorLogs"], ["entity_type", "warehouse_order_id", "fulfillment_order_id", "code", "text"]
    )
    _as_category(logs, ["entity_type", "warehouse_order_id", "fulfillment_order_id", "code"])
    rows = logs.loc[(logs["entity_type"] == "fulfillment_order") & (logs["fulfillment_order_id"] == fulfillment_id)]
    if not rows.empty:
        r = rows.iloc[0]
        return {"code": r.get("code"), "text": r.get("text")}
    return {"code": None, "text": None}


# 6) check fake shipping
def check_fake_shipping(fulfillment_id: str) -> Dict[str, Any]:
    """
    Heuristic check: detect fake shipping error codes in fulfillment-order-level logs.
    Filters ErrorLogs by fulfillment_order_id.
    """
    logs = _read_csv(
        _files()["ErrorLogs"], ["entity_type", "warehouse_order_id", "fulfillment_order_id", "code", "text"]
    )
    _as_category(logs, ["entity_type", "warehouse_order_id", "fulfillment_order_id", "code"])
    rows = logs.loc[(logs["entity_type"] == "fulfillment_order") & (logs["fulfillment_order_id"] == fulfillment_id)]
    if not rows.empty:
        codes = rows["code"].astype(str).str.upper().tolist()
        texts = rows["text"].astype(str).str.upper().tolist()
        if any("FAKE_SHIP" in c for c in codes) or any(("FAKE" in t and "SHIP" in t) for t in texts):
            return {"exceptionFlag": True}
    return {"exceptionFlag": False}


# 7) fulfillment_id + warehouse_order_id -> warehouse order status
def get_warehouse_status(fulfillment_id: str, warehouse_order_id: str) -> Dict[str, Any]:
    """
    Get status and error code for a specific warehouse order under a fulfillment order.
    Returns: {"status": str, "error": str|None}
    """
    w = _read_csv(_files()["WarehouseOrders"], ["warehouse_order_id", "fulfillment_order_id", "status", "error_code"])
    _as_category(w, ["warehouse_order_id", "fulfillment_order_id"])
    rows = w.loc[(w["warehouse_order_id"] == warehouse_order_id) & (w["fulfillment_order_id"] == fulfillment_id)]
    if rows.empty:
        return {"status": None, "error": None}
    r = rows.iloc[0]
    status = _map_status(r.get("status"))
    error_code = r.get("error_code")
    if pd.isna(error_code):
        error_code = None
    return {"status": status, "error": error_code}


# 8) fulfillment_id + warehouse_order_id -> warehouse error details
def get_warehouse_error_details(fulfillment_id: str, warehouse_order_id: str) -> Dict[str, Any]:
    """
    Get warehouse-level error details using composite key (warehouse_order_id + fulfillment_order_id).
    Looks up entity_type='warehouse_order' records in ErrorLogs.
    Returns: {"code": str|None, "text": str|None}
    """
    logs = _read_csv(
        _files()["ErrorLogs"],
        ["entity_type", "warehouse_order_id", "fulfillment_order_id", "code", "text"],
    )
    _as_category(logs, ["entity_type", "warehouse_order_id", "fulfillment_order_id", "code"])
    rows = logs.loc[
        (logs["entity_type"] == "warehouse_order")
        & (logs["warehouse_order_id"] == warehouse_order_id)
        & (logs["fulfillment_order_id"] == fulfillment_id)
    ]
    if not rows.empty:
        r = rows.iloc[0]
        return {"code": r.get("code"), "text": r.get("text")}
    return {"code": None, "text": None}


# ----------------------------------------------------------------------
# OpenAI function tools schema and registry
# ----------------------------------------------------------------------
TOOL_REGISTRY = {
    "query_buyer_and_related": query_buyer_and_related,
    "get_fulfillment_status": get_fulfillment_status,
    "get_cancel_scenes": get_cancel_scenes,
    "get_cancel_error_code": get_cancel_error_code,
    "get_error_reason": get_error_reason,
    "check_fake_shipping": check_fake_shipping,
    "get_warehouse_status": get_warehouse_status,
    "get_warehouse_error_details": get_warehouse_error_details,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "query_buyer_and_related",
            "description": "Given a trade_order_id, return buyer info and related fulfillment/warehouse order IDs.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "Trade order ID to query"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fulfillment_status",
            "description": "Get aggregated status for a fulfillment order.",
            "parameters": {
                "type": "object",
                "properties": {"fulfillment_id": {"type": "string", "description": "fulfillment order ID"}},
                "required": ["fulfillment_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cancel_scenes",
            "description": "Get cancellation scene info (initiator/timeout) for a fulfillment order.",
            "parameters": {
                "type": "object",
                "properties": {"fulfillment_id": {"type": "string", "description": "fulfillment order ID"}},
                "required": ["fulfillment_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cancel_error_code",
            "description": "Get cancellation error code and message for a fulfillment order.",
            "parameters": {
                "type": "object",
                "properties": {"fulfillment_id": {"type": "string", "description": "fulfillment order ID"}},
                "required": ["fulfillment_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_error_reason",
            "description": "Get the error code and text for a fulfillment order, if exists.",
            "parameters": {
                "type": "object",
                "properties": {"fulfillment_id": {"type": "string", "description": "fulfillment order ID"}},
                "required": ["fulfillment_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_fake_shipping",
            "description": "Check whether the fulfillment order is flagged for fake shipping.",
            "parameters": {
                "type": "object",
                "properties": {"fulfillment_id": {"type": "string", "description": "fulfillment order ID"}},
                "required": ["fulfillment_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_warehouse_status",
            "description": "Get status and error code for a specific warehouse order under a fulfillment order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fulfillment_id": {"type": "string", "description": "fulfillment order ID"},
                    "warehouse_order_id": {"type": "string", "description": "warehouse orders ID"},
                },
                "required": ["fulfillment_id", "warehouse_order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_warehouse_error_details",
            "description": "Get error details (code, text) for a specific warehouse order under a fulfillment order using composite key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fulfillment_id": {"type": "string", "description": "fulfillment order ID"},
                    "warehouse_order_id": {"type": "string", "description": "warehouse orders ID"},
                },
                "required": ["fulfillment_id", "warehouse_order_id"],
                "additionalProperties": False,
            },
        },
    },
]


def to_nemo_gym_tools() -> List[Dict[str, Any]]:
    """Convert SC-bench OpenAI tool schemas to NeMo Gym Responses API format."""
    nemo_tools: List[Dict[str, Any]] = []
    for tool in tools:
        fn = tool["function"]
        nemo_tools.append(
            {
                "type": "function",
                "name": fn["name"],
                "description": fn["description"],
                "parameters": fn["parameters"],
                "strict": True,
            }
        )
    return nemo_tools


SYSTEM_PROMPT = (
    "You are a helpful assistant. Your job is to extract the status of the request order "
    "as much as possible using the available tools."
)
