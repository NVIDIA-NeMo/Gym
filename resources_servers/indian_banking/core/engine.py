# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""In-process banking world: per-episode DB copies and tool dispatch.

Each episode owns an independent copy of the customer DB (seed_world); tool
calls run against that copy (apply_tool) by binding it into
banking_tools' contextvars for the duration of the call. Contextvars
propagate per asyncio task, so concurrent rollouts never see each other's DB.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, Optional

from resources_servers.indian_banking.core import banking_tools


_SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_SERVER_DIR, "data")
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "db.json")
DEFAULT_KB_PATH = os.path.join(DATA_DIR, "kb.json")

# Frozen sim clock and id seed: rollouts and gold replays must produce identical ids/timestamps.
# Changing either invalidates the shipped example rollouts.
SIM_CLOCK = "2026-07-16"
SIM_SEED = "tau2-banking"

# Mirrors the tool list in banking_tools.py; keep in sync if a tool is added.
# transfer_to_human_agents is handled separately in apply_tool, not as a banking_tools call.
MCP_TOOL_NAMES: list[str] = [
    "search_knowledge_base",
    "get_account_balance",
    "get_account_details",
    "get_transaction_history",
    "get_fd_details",
    "get_rd_details",
    "get_deposit_loan_rates",
    "calculate_fd_maturity",
    "calculate_rd_maturity",
    "create_fd",
    "create_rd",
    "get_deposit_closure_quote",
    "close_deposit",
    "update_deposit_renewal",
    "get_loan_details",
    "get_loan_foreclosure_quote",
    "calculate_emi",
    "get_gold_rate",
    "calculate_gold_loan_ltv",
    "get_card_details",
    "toggle_card_freeze",
    "block_card",
    "set_card_controls",
    "show_mandates",
    "cancel_mandate",
    "stop_cheque_payment",
    "request_cheque_book",
    "request_duplicate_statement",
    "update_address",
    "raise_request",
    "get_request_status",
    "get_insurance_details",
    "get_products_and_offers",
]

WRITE_TOOLS = {
    "create_fd",
    "create_rd",
    "close_deposit",
    "update_deposit_renewal",
    "toggle_card_freeze",
    "block_card",
    "set_card_controls",
    "cancel_mandate",
    "stop_cheque_payment",
    "update_address",
    "raise_request",
}
SOFT_WRITE_TOOLS = {"request_cheque_book", "request_duplicate_statement"}

TRANSFER_TOOL = "transfer_to_human_agents"

_DB_PATH: str = DEFAULT_DB_PATH
_KB_PATH: str = DEFAULT_KB_PATH
_CONFIGURED = False
_BASE_DB: Optional[dict[str, Any]] = None
_BASE_DB_PICKLE: Optional[bytes] = None
# search_knowledge_base is deterministic and state-free: memoized by argument JSON.
_KB_CACHE: dict[str, str] = {}


def configure(db_path: Optional[str] = None, kb_path: Optional[str] = None) -> None:
    """Point the engine at a customer DB / knowledge base and reset all caches.

    Called once by the resources server at startup (and by tests). Paths default
    to data/db.json and data/kb.json next to this package.
    """
    global _DB_PATH, _KB_PATH, _CONFIGURED, _BASE_DB, _BASE_DB_PICKLE
    _DB_PATH = db_path or DEFAULT_DB_PATH
    _KB_PATH = kb_path or DEFAULT_KB_PATH
    banking_tools.CUSTOMER_DB_PATH = _DB_PATH
    banking_tools.KB_PATH = _KB_PATH
    banking_tools.PERSIST_WRITES = False
    banking_tools.SIM_CLOCK = SIM_CLOCK
    banking_tools.SIM_SEED = SIM_SEED
    banking_tools._DB_CACHE = None
    banking_tools._KB_CACHE = None
    _BASE_DB = None
    _BASE_DB_PICKLE = None
    _KB_CACHE.clear()
    _CONFIGURED = True


def _ensure_configured() -> None:
    if not _CONFIGURED:
        configure()


def get_base_db() -> dict[str, Any]:
    global _BASE_DB
    _ensure_configured()
    if _BASE_DB is None:
        if not os.path.exists(_DB_PATH):
            raise FileNotFoundError(
                f"{_DB_PATH} not found. db.json (and kb.json) ship with the environment's "
                "dataset (same repository as train.jsonl/validation.jsonl); download them "
                "and place them in resources_servers/indian_banking/data/."
            )
        with open(_DB_PATH, encoding="utf-8") as f:
            _BASE_DB = json.load(f)
    return _BASE_DB


def _fresh_db() -> dict[str, Any]:
    """Independent copy of the base DB. Pickle round-trip off one cached
    buffer instead of deepcopy (much cheaper, verified identical output,
    since the DB is plain JSON-safe types)."""
    global _BASE_DB_PICKLE
    if _BASE_DB_PICKLE is None:
        _BASE_DB_PICKLE = pickle.dumps(get_base_db(), protocol=pickle.HIGHEST_PROTOCOL)
    return pickle.loads(_BASE_DB_PICKLE)


def seed_world(active_customer: Optional[str]) -> dict[str, Any]:
    db = _fresh_db()
    db["active_customer"] = active_customer
    return {
        "db": db,
        "customer": active_customer,
        "calls": [],  # [{name, arguments, result, error}]
        "transferred": False,
    }


def apply_tool(world: dict[str, Any], name: str, arguments: dict[str, Any]) -> str:
    """Execute one tool call and record it in ``world["calls"]``. Returns the raw
    string result; tools return error JSON rather than raising."""
    if name == TRANSFER_TOOL:
        world["transferred"] = True
        result = "Transfer successful"
        world["calls"].append({"name": name, "arguments": arguments, "result": result, "error": False})
        return result

    _ensure_configured()
    fn = banking_tools.TOOLS.get(name)
    if fn is None or name not in MCP_TOOL_NAMES:
        result = json.dumps({"error_code": "UNKNOWN_TOOL", "error_message": f"No tool named {name}"})
        world["calls"].append({"name": name, "arguments": arguments, "result": result, "error": True})
        return result

    # Cache hit path.
    cache_key = None
    if name == "search_knowledge_base":
        cache_key = json.dumps(arguments, sort_keys=True, default=str)
        hit = _KB_CACHE.get(cache_key)
        if hit is not None:
            world["calls"].append(
                {"name": name, "arguments": arguments, "result": hit, "error": _looks_like_error(hit)}
            )
            return hit

    tokens = banking_tools.bind_db(world["db"], world["customer"])
    try:
        try:
            result = str(fn(**arguments))
            error = _looks_like_error(result)
        except TypeError as exc:  # bad/missing kwargs, surface like the server would
            result = json.dumps({"error_code": "VALIDATION_ERROR", "error_message": str(exc)})
            error = True
        except Exception as exc:  # defensive: never kill the rollout
            result = json.dumps({"error_code": "TOOL_ERROR", "error_message": f"{type(exc).__name__}: {exc}"})
            error = True
    finally:
        banking_tools.reset_db(*tokens)

    # Never cache a not-configured result.
    if cache_key is not None and not error and "not configured" not in result:
        _KB_CACHE[cache_key] = result

    world["calls"].append({"name": name, "arguments": arguments, "result": result, "error": error})
    return result


def _looks_like_error(result: str) -> bool:
    return '"error_code"' in result or "VALIDATION_ERROR" in result or "NOT_FOUND" in result
