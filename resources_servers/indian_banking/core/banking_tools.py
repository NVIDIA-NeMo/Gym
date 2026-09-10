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
"""The 33 Indian retail-banking tools, executed in-process.

Each tool is a plain function taking keyword arguments and returning a JSON string (a result
object or ``{"error_code": ..., "error_message": ...}``). Tools never raise; ``engine.apply_tool``
is the only caller. ``@mcp_server.tool()`` registers each function in :data:`TOOLS`; the
per-episode DB is bound through contextvars (see ``engine``).
"""

from __future__ import annotations

import contextvars
import copy
import functools
import hashlib
import json
import os
import random
from datetime import datetime, timedelta
from typing import Any, Literal, Optional


class _ToolRegistry:
    """Minimal tool registry: ``@mcp_server.tool()`` registers the function in
    :data:`TOOLS` and returns it unchanged for direct in-process calls."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, *args: Any, **kwargs: Any):
        def _decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return _decorator


mcp_server = _ToolRegistry()
TOOLS: dict[str, Any] = mcp_server.tools

# Per-rollout DB binding (see module docstring). Set via :func:`bind_db`.
_BOUND_DB: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "indian_banking_bound_db", default=None
)
_BOUND_CUSTOMER: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "indian_banking_bound_customer", default=None
)


def bind_db(raw_db: dict[str, Any], active_customer: Optional[str]) -> tuple[Any, Any]:
    token_db = _BOUND_DB.set(raw_db)
    token_cid = _BOUND_CUSTOMER.set(active_customer)
    return token_db, token_cid


def reset_db(token_db: Any, token_cid: Any) -> None:
    _BOUND_DB.reset(token_db)
    _BOUND_CUSTOMER.reset(token_cid)


# Simulation clock (ISO datetime) and id seed. Frozen by ``engine`` so an agent
# rollout and the gold replay produce identical timestamps and content-hashed ids.
SIM_CLOCK: Optional[str] = None
SIM_SEED: Optional[str] = None

_HERE = os.path.dirname(os.path.abspath(__file__))
# Overridden by ``engine`` to point at resources_servers/indian_banking/data/.
CUSTOMER_DB_PATH = os.path.join(_HERE, "customer_db.json")
KB_PATH = os.path.join(_HERE, "knowledge_base.json")
PERSIST_WRITES = False

# BANK_CHAOS_TOOLS="tool:n,..." opt-in failure injection: the first n calls to `tool` fail.
_CHAOS_CONFIG: dict[str, int] | None = None
_CHAOS_CALLS: dict[str, int] = {}


def _chaos_config() -> dict[str, int]:
    global _CHAOS_CONFIG
    if _CHAOS_CONFIG is None:
        _CHAOS_CONFIG = {}
        for part in os.environ.get("BANK_CHAOS_TOOLS", "").split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            name, count = part.split(":", 1)
            try:
                _CHAOS_CONFIG[name.strip()] = int(count)
            except ValueError:
                pass
    return _CHAOS_CONFIG


def _maybe_chaos_fail(tool_name: str) -> Optional[str]:
    fail_count = _chaos_config().get(tool_name)
    if not fail_count:
        return None
    seen = _CHAOS_CALLS.get(tool_name, 0)
    _CHAOS_CALLS[tool_name] = seen + 1
    if seen < fail_count:
        return _error(
            "SERVICE_UNAVAILABLE",
            "The service is temporarily unavailable. Please try again in a moment.",
            retriable=True,
        )
    return None


_orig_mcp_tool = mcp_server.tool


def _chaos_aware_tool(*dargs: Any, **dkwargs: Any):
    orig_decorator = _orig_mcp_tool(*dargs, **dkwargs)

    def wrapper(fn):
        @functools.wraps(fn)
        def chaos_checked(*fargs: Any, **fkwargs: Any):
            err = _maybe_chaos_fail(fn.__name__)
            return err if err is not None else fn(*fargs, **fkwargs)

        return orig_decorator(chaos_checked)

    return wrapper


mcp_server.tool = _chaos_aware_tool

_IFSC_BANK_MAP: dict[str, str] | None = None


def _ifsc_bank_map() -> dict[str, str]:
    global _IFSC_BANK_MAP
    if _IFSC_BANK_MAP is None:
        path = os.path.join(_HERE, "ifsc_bank_map.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                _IFSC_BANK_MAP = {str(k).upper(): str(v) for k, v in json.load(fh).items()}
        else:
            _IFSC_BANK_MAP = {}
    return _IFSC_BANK_MAP


def _bank_from_ifsc(ifsc: Any, fallback: Any = None) -> Optional[str]:
    if fallback:
        return str(fallback)
    code = str(ifsc or "").strip().upper()[:4]
    if not code:
        return None
    return _ifsc_bank_map().get(code) or f"Unknown Bank ({code})"


def _dry() -> bool:
    # The engine binds a per-episode DB copy; the session-scoped dry-run copy is unused.
    return False


DEPOSIT_PENALTY_RATE = 1.0  # percentage points deducted from contract rate
GOLD_LTV_CAP = 75.0  # RBI cap

RATE_CARDS: dict[str, dict[str, Any]] = {
    "fd": {
        "unit": "tenure_months",
        "compounding": "quarterly",
        "slabs": [
            {"label": "7 days - 3 months", "min_months": 0, "max_months": 3, "general_rate": 4.5, "senior_rate": 5.0},
            {"label": "3 - 6 months", "min_months": 3, "max_months": 6, "general_rate": 5.5, "senior_rate": 6.0},
            {"label": "6 - 12 months", "min_months": 6, "max_months": 12, "general_rate": 6.8, "senior_rate": 7.3},
            {"label": "12 - 24 months", "min_months": 12, "max_months": 24, "general_rate": 7.1, "senior_rate": 7.6},
            {"label": "24 - 36 months", "min_months": 24, "max_months": 36, "general_rate": 7.3, "senior_rate": 7.8},
            {"label": "36 - 60 months", "min_months": 36, "max_months": 60, "general_rate": 7.0, "senior_rate": 7.5},
        ],
    },
    "rd": {
        "unit": "tenure_months",
        "compounding": "quarterly",
        "slabs": [
            {"label": "6 - 12 months", "min_months": 6, "max_months": 12, "general_rate": 6.5, "senior_rate": 7.0},
            {"label": "12 - 24 months", "min_months": 12, "max_months": 24, "general_rate": 6.8, "senior_rate": 7.3},
            {"label": "24 - 36 months", "min_months": 24, "max_months": 36, "general_rate": 7.0, "senior_rate": 7.5},
            {"label": "36 - 60 months", "min_months": 36, "max_months": 60, "general_rate": 6.8, "senior_rate": 7.3},
        ],
    },
    "home_loan": {
        "unit": "amount",
        "slabs": [
            {"label": "up to 30 lakh", "min_amount": 0, "max_amount": 3000000, "general_rate": 8.5},
            {"label": "30 - 75 lakh", "min_amount": 3000000, "max_amount": 7500000, "general_rate": 8.35},
            {"label": "above 75 lakh", "min_amount": 7500000, "max_amount": 1_000_000_000, "general_rate": 8.6},
        ],
    },
    "personal_loan": {
        "unit": "amount",
        "slabs": [
            {"label": "up to 5 lakh", "min_amount": 0, "max_amount": 500000, "general_rate": 12.5},
            {"label": "5 - 15 lakh", "min_amount": 500000, "max_amount": 1500000, "general_rate": 11.5},
            {"label": "above 15 lakh", "min_amount": 1500000, "max_amount": 1_000_000_000, "general_rate": 10.75},
        ],
    },
    "gold_loan": {
        "unit": "amount",
        "slabs": [
            {"label": "up to 3 lakh", "min_amount": 0, "max_amount": 300000, "general_rate": 8.5},
            {"label": "3 - 10 lakh", "min_amount": 300000, "max_amount": 1000000, "general_rate": 8.0},
            {"label": "above 10 lakh", "min_amount": 1000000, "max_amount": 1_000_000_000, "general_rate": 7.5},
        ],
    },
}

GOLD_RATES_PER_GRAM = {18: 4600, 22: 5850, 24: 6300}

PRODUCT_CATALOG = [
    {
        "type": "deposit",
        "code": "FD",
        "name": "Fixed Deposit",
        "summary": "Lump-sum deposit with guaranteed returns; flexible tenure 7 days to 10 years.",
    },
    {
        "type": "deposit",
        "code": "RD",
        "name": "Recurring Deposit",
        "summary": "Systematic monthly savings from ₹500/month; interest paid at maturity.",
    },
    {
        "type": "loan",
        "code": "HL",
        "name": "Home Loan",
        "summary": "Financing for purchase/construction; floating rates linked to external benchmark.",
    },
    {
        "type": "loan",
        "code": "PL",
        "name": "Personal Loan",
        "summary": "Unsecured loan for any need; quick disbursal, minimal documentation.",
    },
    {
        "type": "loan",
        "code": "GL",
        "name": "Gold Loan",
        "summary": "Loan against gold ornaments up to the RBI LTV cap; fast processing.",
    },
    {
        "type": "card",
        "code": "DC",
        "name": "Debit Card",
        "summary": "Linked to your account with configurable per-channel limits.",
    },
    {
        "type": "card",
        "code": "CC",
        "name": "Credit Card",
        "summary": "Revolving credit with rewards and EMI conversion options.",
    },
    {
        "type": "insurance",
        "code": "LI",
        "name": "Life Insurance (Bancassurance)",
        "summary": "Term and endowment cover distributed through the bank.",
    },
]

_DB_CACHE: dict[str, Any] | None = None
_KB_CACHE: list[dict[str, Any]] | None = None


def _load_db() -> dict[str, Any]:
    global _DB_CACHE
    bound = _BOUND_DB.get()
    if bound is not None:
        return bound
    if _DB_CACHE is None:
        if os.path.exists(CUSTOMER_DB_PATH):
            if CUSTOMER_DB_PATH.lower().endswith((".xlsx", ".xls")):
                _DB_CACHE = _load_db_from_excel(CUSTOMER_DB_PATH)
            else:
                with open(CUSTOMER_DB_PATH, "r", encoding="utf-8") as fh:
                    _DB_CACHE = json.load(fh)
        else:
            _DB_CACHE = {"customers": {}}
    return _DB_CACHE


def _load_db_from_excel(path: str) -> dict[str, Any]:
    import pandas as pd  # avoid a hard pandas dependency for the JSON-only path

    book = pd.read_excel(path, sheet_name=None, dtype=object)

    def sheet(name: str) -> list[dict[str, Any]]:
        df = book.get(name)
        if df is None:
            return []
        return df.where(df.notna(), None).to_dict("records")

    def num(v: Any) -> Any:
        if v is None or v == "":
            return None
        try:
            f = float(v)
            return int(f) if f.is_integer() else f
        except (TypeError, ValueError):
            return v

    def truthy(v: Any) -> bool:
        return str(v).strip().lower() in ("true", "1", "yes", "y")

    def nominee(name: Any, rel: Any) -> Optional[dict[str, Any]]:
        return {"name": name, "relationship": rel} if name else None

    customers: dict[str, Any] = {}

    def cust(cid: str) -> dict[str, Any]:
        return customers.setdefault(
            cid,
            {
                "profile": {},
                "login_context": {
                    "customer_id": cid,
                    "linked_accounts": [],
                    "linked_products": [],
                    "linked_cards": [],
                },
                "accounts": {},
                "deposits": {},
                "loans": {},
                "cards": {},
                "mandates": [],
                "insurance": {},
                "requests": {},
                "offers": [],
            },
        )

    for r in sheet("customers"):
        c = cust(r["customer_id"])
        c["profile"] = {
            "name": r.get("name"),
            "bank_name": r.get("bank_name") or None,
            "age": num(r.get("age")),
            "gender": r.get("gender"),
            "senior_citizen": truthy(r.get("senior_citizen")),
            "segment": r.get("segment"),
            "kyc_status": r.get("kyc_status"),
            "email": r.get("email"),
            "phone": r.get("phone"),
            "communication_address": {
                "line1": r.get("comm_line1"),
                "line2": r.get("comm_line2"),
                "city": r.get("comm_city"),
                "state": r.get("comm_state"),
                "pincode": r.get("comm_pincode"),
            },
            "permanent_address": {
                "line1": r.get("perm_line1"),
                "line2": r.get("perm_line2"),
                "city": r.get("perm_city"),
                "state": r.get("perm_state"),
                "pincode": r.get("perm_pincode"),
            },
        }

    for r in sheet("accounts"):
        c = cust(r["customer_id"])
        aid = r["account_id"]
        ifsc = r.get("ifsc")
        bank_name = _bank_from_ifsc(ifsc, r.get("bank_name"))
        c["accounts"][aid] = {
            "account_id": aid,
            "account_type": r.get("account_type"),
            "status": r.get("status"),
            "ifsc": ifsc,
            "bank_name": bank_name,
            "home_branch": r.get("home_branch"),
            "holders": [h.strip() for h in str(r.get("holders") or "").split(";") if h.strip()],
            "nominee": nominee(r.get("nominee_name"), r.get("nominee_relationship")),
            "open_date": r.get("open_date"),
            "available_balance": num(r.get("available_balance")) or 0.0,
            "hold_amount": num(r.get("hold_amount")) or 0.0,
            "currency": r.get("currency") or "INR",
            "transactions": [],
            "cheques": [],
        }
        c["login_context"]["linked_accounts"].append(aid)
        if bank_name and not c["profile"].get("bank_name"):
            c["profile"]["bank_name"] = bank_name
        elif not c["profile"].get("bank_name") and bank_name:
            c["profile"]["bank_name"] = bank_name

    for r in sheet("transactions"):
        acc = customers.get(r["customer_id"], {}).get("accounts", {}).get(r["account_id"])
        if acc is not None:
            acc["transactions"].append(
                {
                    "reference_id": r.get("reference_id"),
                    "date": r.get("date"),
                    "description": r.get("description"),
                    "amount": num(r.get("amount")),
                    "direction": r.get("direction"),
                    "channel": r.get("channel"),
                    "balance_after": num(r.get("balance_after")),
                }
            )

    for r in sheet("cheques"):
        acc = customers.get(r["customer_id"], {}).get("accounts", {}).get(r["account_id"])
        if acc is not None:
            acc["cheques"].append({"cheque_number": str(r.get("cheque_number")), "status": r.get("status")})

    for r in sheet("deposits"):
        c = cust(r["customer_id"])
        did = r["deposit_id"]
        kind = r.get("kind")
        dep: dict[str, Any] = {
            "deposit_id": did,
            "kind": kind,
            "rate": num(r.get("rate")),
            "tenure_months": num(r.get("tenure_months")),
            "start_date": r.get("start_date"),
            "maturity_date": r.get("maturity_date"),
            "maturity_amount": num(r.get("maturity_amount")),
            "maturity_instruction": r.get("maturity_instruction"),
            "status": r.get("status"),
            "payout_account": r.get("payout_account"),
            "nominee": nominee(r.get("nominee_name"), r.get("nominee_relationship")),
        }
        if kind == "RD":
            dep.update(
                {
                    "monthly_installment": num(r.get("monthly_installment")),
                    "installments_paid": num(r.get("installments_paid")),
                    "total_deposited": num(r.get("total_deposited")),
                    "source_account": r.get("source_account"),
                }
            )
        else:
            dep.update({"principal": num(r.get("principal")), "payout_type": r.get("payout_type")})
        c["deposits"][did] = dep
        c["login_context"]["linked_products"].append(did)

    for r in sheet("loans"):
        c = cust(r["customer_id"])
        lid = r["loan_id"]
        c["loans"][lid] = {
            "loan_id": lid,
            "loan_type": r.get("loan_type"),
            "outstanding": num(r.get("outstanding")),
            "rate": num(r.get("rate")),
            "emi": num(r.get("emi")),
            "remaining_tenure_months": num(r.get("remaining_tenure_months")),
            "next_due_date": r.get("next_due_date"),
            "status": r.get("status"),
        }
        c["login_context"]["linked_products"].append(lid)

    def region(enabled: Any, limit: Any) -> dict[str, Any]:
        blk: dict[str, Any] = {"enabled": truthy(enabled)}
        if blk["enabled"] and num(limit) is not None:
            blk["daily_limit"] = num(limit)
        return blk

    for r in sheet("cards"):
        c = cust(r["customer_id"])
        cardid = r["card_id"]
        c["cards"][cardid] = {
            "card_id": cardid,
            "card_type": r.get("card_type"),
            "network": r.get("network"),
            "status": r.get("status"),
            "credit_limit": num(r.get("credit_limit")) or 0,
            "available_limit": num(r.get("available_limit")) or 0,
            "linked_account": r.get("linked_account"),
            "controls": {
                "atm": {
                    "domestic": region(r.get("atm_dom_enabled"), r.get("atm_dom_limit")),
                    "international": region(r.get("atm_intl_enabled"), r.get("atm_intl_limit")),
                },
                "online": {
                    "domestic": region(r.get("online_dom_enabled"), r.get("online_dom_limit")),
                    "international": region(r.get("online_intl_enabled"), r.get("online_intl_limit")),
                },
                "pos": {
                    "domestic": region(r.get("pos_dom_enabled"), r.get("pos_dom_limit")),
                    "international": region(r.get("pos_intl_enabled"), r.get("pos_intl_limit")),
                },
            },
        }
        c["login_context"]["linked_cards"].append(cardid)

    for r in sheet("mandates"):
        c = cust(r["customer_id"])
        c["mandates"].append(
            {
                "mandate_id": r.get("mandate_id"),
                "account_id": r.get("account_id"),
                "payee": r.get("payee"),
                "amount": num(r.get("amount")),
                "frequency": r.get("frequency"),
                "next_debit_date": r.get("next_debit_date"),
                "status": r.get("status"),
                "type": r.get("type"),
            }
        )

    for r in sheet("insurance"):
        c = cust(r["customer_id"])
        pid = r["policy_id"]
        c["insurance"][pid] = {
            "policy_id": pid,
            "policy_name": r.get("policy_name"),
            "insurer": r.get("insurer"),
            "sum_assured": num(r.get("sum_assured")),
            "premium": num(r.get("premium")),
            "frequency": r.get("frequency"),
            "status": r.get("status"),
            "next_premium_date": r.get("next_premium_date"),
            "nominee": nominee(r.get("nominee_name"), None),
        }
        c["login_context"]["linked_products"].append(pid)

    for r in sheet("offers"):
        c = cust(r["customer_id"])
        c["offers"].append(
            {
                "offer_id": r.get("offer_id"),
                "type": r.get("type"),
                "title": r.get("title"),
                "detail": r.get("detail"),
                "indicative": truthy(r.get("indicative")),
            }
        )

    active = next(iter(customers)) if customers else None
    return {"active_customer": active, "customers": customers}


def _persist_db() -> None:
    # Episode DBs are in-memory copies bound via contextvars; never write to disk.
    return None


def _load_kb() -> list[dict[str, Any]]:
    global _KB_CACHE
    if _KB_CACHE is None:
        if os.path.exists(KB_PATH):
            with open(KB_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            _KB_CACHE = data.get("articles", data) if isinstance(data, dict) else data
        else:
            _KB_CACHE = []
    return _KB_CACHE


def _active_customer_id() -> Optional[str]:
    db = _load_db()
    customers = db.get("customers", {})
    if not customers:
        return None
    cid = _BOUND_CUSTOMER.get() or db.get("active_customer")
    if cid and cid in customers:
        return cid
    if len(customers) == 1:
        return next(iter(customers))
    return cid  # may be invalid; caller resolves that to CUSTOMER_NOT_FOUND


_SESSION_CUSTOMERS: dict[str, dict[str, Any]] = {}


def _active_customer() -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """(customer_record, error_json) - exactly one is non-None."""
    db = _load_db()
    if not db.get("customers"):
        return None, _error("NO_CUSTOMER_DB", "No customer database configured.")
    cid = _active_customer_id()
    cust = db.get("customers", {}).get(cid) if cid else None
    if not cust:
        return None, _error("CUSTOMER_NOT_FOUND", "No active customer resolved.")
    if _dry():
        # Session-scoped in-memory copy per customer; never written to disk.
        if cid not in _SESSION_CUSTOMERS:
            _SESSION_CUSTOMERS[cid] = copy.deepcopy(cust)
        return _SESSION_CUSTOMERS[cid], None
    return cust, None


def _login_context(cust: dict[str, Any]) -> dict[str, Any]:
    ctx = cust.get("login_context", {})
    return {
        "linked_accounts": ctx.get("linked_accounts", list(cust.get("accounts", {}).keys())),
        "linked_products": ctx.get(
            "linked_products",
            list(cust.get("deposits", {}).keys())
            + list(cust.get("loans", {}).keys())
            + list(cust.get("insurance", {}).keys()),
        ),
        "linked_cards": ctx.get("linked_cards", list(cust.get("cards", {}).keys())),
    }


def _error(code: str, message: str, retriable: bool = False) -> str:
    return json.dumps({"error_code": code, "error_message": message, "retriable": retriable})


def _now() -> datetime:
    """Wall clock, or a frozen clock when ``SIM_CLOCK`` is set (ISO datetime).
    Needed so the gold-action replay used by the DB check produces the same
    timestamps as the agent's rollout instead of failing DB equality."""
    sim = SIM_CLOCK
    if sim:
        return datetime.fromisoformat(sim)  # let a malformed value raise, not silently fall back
    return datetime.now()


def _today_str() -> str:
    return _now().strftime("%Y-%m-%d")


def _ref(prefix: str, digits: int = 8, key: Optional[str] = None) -> str:
    """Random, unless ``SIM_SEED`` and a semantic `key` are both given, in which
    case the id is derived from seed+prefix+key so the agent run and the gold
    replay (no shared RNG) land on the same id. Only pass `key` for ids
    persisted into ordered structures (e.g. linked_products)."""
    seed = SIM_SEED
    if seed and key is not None:
        digest = hashlib.sha1(f"{seed}:{prefix}:{key}".encode("utf-8")).hexdigest()
        span = 10**digits - 10 ** (digits - 1)
        return f"{prefix}{10 ** (digits - 1) + int(digest, 16) % span}"
    return f"{prefix}{random.randint(10 ** (digits - 1), 10**digits - 1)}"


# 1. Knowledge base


@mcp_server.tool()
def search_knowledge_base(
    query: str,
    bank_id: Optional[str] = None,
    category: Optional[str] = None,
    top_k: Optional[int] = None,
) -> str:
    """bank_id is accepted but ignored (single local corpus, keyword search only).
    score is a raw keyword-overlap count, not a calibrated probability."""

    articles = _load_kb()
    if not articles:
        return json.dumps(
            {
                "query": query,
                "total_results": 0,
                "results": [],
                "note": "Knowledge base not configured.",
            }
        )

    pool = articles
    if category:
        cat_l = category.strip().lower()
        narrowed = [a for a in pool if cat_l in str(a.get("category", "")).lower()]
        if narrowed:
            pool = narrowed

    terms = [t for t in query.lower().split() if len(t) > 2]

    def score(a: dict[str, Any]) -> int:
        hay = " ".join(
            [
                str(a.get("title", "")),
                str(a.get("content", a.get("snippet", ""))),
                " ".join(a.get("keywords", []) or []),
            ]
        ).lower()
        return sum(hay.count(t) for t in terms)

    k = top_k or 5
    ranked = sorted(pool, key=score, reverse=True)
    matched = [a for a in ranked if score(a) > 0][:k] or ranked[: min(3, k)]

    results = [
        {
            "title": a.get("title", ""),
            "text": a.get("content", a.get("snippet", "")),
            "category": a.get("category", ""),
            "source_url": a.get("source", a.get("id", "")),
            "score": score(a),
        }
        for a in matched
    ]
    return json.dumps({"query": query, "total_results": len(results), "results": results})


# 2-4. Accounts


@mcp_server.tool()
def get_account_balance(account_ids: list[str]) -> str:
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    accounts = cust.get("accounts", {})
    out, errors = [], []
    for aid in account_ids:
        if aid not in ctx["linked_accounts"] or aid not in accounts:
            errors.append({"account_id": aid, "error": "Account not linked to logged-in customer"})
            continue
        acc = accounts[aid]
        out.append(
            {
                "account_id": aid,
                "available_balance": acc.get("available_balance", 0.0),
                "hold_amount": acc.get("hold_amount", 0.0),
                "currency": acc.get("currency", "INR"),
                "as_of": _now().isoformat(),
            }
        )
    return json.dumps({"balances": out, "errors": errors})


@mcp_server.tool()
def get_account_details(account_ids: list[str]) -> str:
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    accounts = cust.get("accounts", {})
    out, errors = [], []
    for aid in account_ids:
        if aid not in ctx["linked_accounts"] or aid not in accounts:
            errors.append({"account_id": aid, "error": "Account not linked to logged-in customer"})
            continue
        acc = accounts[aid]
        out.append(
            {
                "account_id": aid,
                "account_type": acc.get("account_type"),
                "status": acc.get("status", "ACTIVE"),
                "bank_name": acc.get("bank_name") or _bank_from_ifsc(acc.get("ifsc")),
                "ifsc": acc.get("ifsc"),
                "home_branch": acc.get("home_branch"),
                "holders": acc.get("holders", []),
                "nominee": acc.get("nominee"),
                "open_date": acc.get("open_date"),
                "currency": acc.get("currency", "INR"),
            }
        )
    return json.dumps({"accounts": out, "errors": errors})


@mcp_server.tool()
def get_transaction_history(
    account_id: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    type: Literal["debit", "credit", "all"] = "all",
    channel: Literal["upi", "neft", "imps", "rtgs", "atm", "pos", "cheque", "all"] = "all",
    limit: int = 10,
    offset: int = 0,
) -> str:
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    if account_id not in ctx["linked_accounts"] or account_id not in cust.get("accounts", {}):
        return _error("VALIDATION_ERROR", f"Account {account_id} not linked to logged-in customer")
    txns = list(cust["accounts"][account_id].get("transactions", []))
    if from_date:
        txns = [t for t in txns if t.get("date", "") >= from_date]
    if to_date:
        txns = [t for t in txns if t.get("date", "") <= to_date]
    if type != "all":
        txns = [t for t in txns if t.get("direction") == type]
    if channel != "all":
        txns = [t for t in txns if t.get("channel") == channel]
    txns.sort(key=lambda t: t.get("date", ""), reverse=True)
    total = len(txns)
    page = txns[offset : offset + limit]
    return json.dumps(
        {
            "account_id": account_id,
            "total_records": total,
            "returned": len(page),
            "offset": offset,
            "limit": limit,
            "transactions": page,
        }
    )


# 5-6. Deposit read


def _get_deposits(cust: dict[str, Any], deposit_ids: list[str], kind: Optional[str]) -> tuple[list, list]:
    ctx = _login_context(cust)
    deposits = cust.get("deposits", {})
    out, errors = [], []
    for did in deposit_ids:
        dep = deposits.get(did)
        if not dep or did not in ctx["linked_products"]:
            errors.append({"deposit_id": did, "error": "Deposit not linked to logged-in customer"})
            continue
        if kind and dep.get("kind") != kind:
            errors.append({"deposit_id": did, "error": f"{did} is not a {kind}"})
            continue
        out.append(dep)
    return out, errors


@mcp_server.tool()
def get_fd_details(deposit_ids: list[str]) -> str:
    cust, err = _active_customer()
    if err:
        return err
    fds, errors = _get_deposits(cust, deposit_ids, "FD")
    return json.dumps({"fds": fds, "errors": errors})


@mcp_server.tool()
def get_rd_details(deposit_ids: list[str]) -> str:
    cust, err = _active_customer()
    if err:
        return err
    rds, errors = _get_deposits(cust, deposit_ids, "RD")
    return json.dumps({"rds": rds, "errors": errors})


# 7. Rate card


@mcp_server.tool()
def get_deposit_loan_rates(
    product_type: Literal["fd", "rd", "home_loan", "personal_loan", "gold_loan"],
    tenure_months: Optional[int] = None,
    amount: Optional[float] = None,
) -> str:
    """Sole source of truth for rates; calculators and bookings must use this,
    not a hardcoded value."""
    card = RATE_CARDS.get(product_type)
    if not card:
        return _error("VALIDATION_ERROR", f"No rate card for product {product_type}")
    slabs = card["slabs"]
    if card["unit"] == "tenure_months" and tenure_months is not None:
        slabs = [s for s in slabs if s["min_months"] < tenure_months <= s["max_months"]] or slabs
    if card["unit"] == "amount" and amount is not None:
        slabs = [s for s in slabs if s["min_amount"] <= amount < s["max_amount"]] or slabs
    return json.dumps(
        {
            "product_type": product_type,
            "effective_date": _today_str(),
            "unit": card["unit"],
            "compounding": card.get("compounding"),
            "slabs": slabs,
        }
    )


# 8-9. Deposit calculators (pure)


@mcp_server.tool()
def calculate_fd_maturity(
    principal_amount: float,
    rate: float,
    tenure_months: int,
    payout_type: Literal["cumulative", "monthly", "quarterly"] = "cumulative",
) -> str:
    years = tenure_months / 12.0
    if payout_type == "cumulative":
        n = 4  # quarterly compounding
        maturity = principal_amount * (1 + rate / 100 / n) ** (n * years)
        total_interest = maturity - principal_amount
        return json.dumps(
            {
                "principal_amount": principal_amount,
                "rate": rate,
                "tenure_months": tenure_months,
                "payout_type": payout_type,
                "compounding": "quarterly",
                "maturity_amount": round(maturity, 2),
                "total_interest": round(total_interest, 2),
            }
        )
    # non-cumulative: interest paid out periodically, principal unchanged at maturity
    periods_per_year = 12 if payout_type == "monthly" else 4
    total_interest = principal_amount * rate / 100 * years
    payout_each = principal_amount * rate / 100 / periods_per_year
    return json.dumps(
        {
            "principal_amount": principal_amount,
            "rate": rate,
            "tenure_months": tenure_months,
            "payout_type": payout_type,
            "periodic_payout": round(payout_each, 2),
            "num_payouts": int(round(periods_per_year * years)),
            "total_interest": round(total_interest, 2),
            "maturity_amount": round(principal_amount, 2),
        }
    )


@mcp_server.tool()
def calculate_rd_maturity(monthly_installment: float, rate: float, tenure_months: int) -> str:
    quarterly_rate = rate / 100 / 4
    balance = 0.0
    for month in range(1, tenure_months + 1):
        balance += monthly_installment
        if month % 3 == 0:
            balance += balance * quarterly_rate
    # trailing partial quarter
    trailing = tenure_months % 3
    if trailing:
        balance += balance * quarterly_rate * (trailing / 3)
    total_deposited = monthly_installment * tenure_months
    return json.dumps(
        {
            "monthly_installment": monthly_installment,
            "rate": rate,
            "tenure_months": tenure_months,
            "compounding": "quarterly",
            "total_deposited": round(total_deposited, 2),
            "maturity_amount": round(balance, 2),
            "total_interest": round(balance - total_deposited, 2),
        }
    )


# 10-11. Deposit booking (mutation)


def _debit_account(
    cust: dict[str, Any], account_id: str, amount: float, description: str, channel: str = "neft"
) -> Optional[str]:
    acc = cust.get("accounts", {}).get(account_id)
    if not acc:
        return _error("VALIDATION_ERROR", f"Source account {account_id} not found")
    balance = acc.get("available_balance", 0.0)
    if balance < amount:
        return _error("INSUFFICIENT_FUNDS", f"Available balance {balance} is less than required {amount}")
    acc["available_balance"] = round(balance - amount, 2)
    acc.setdefault("transactions", []).append(
        {
            "reference_id": _ref("TXN", 10),
            "date": _today_str(),
            "description": description,
            "amount": -abs(amount),
            "direction": "debit",
            "channel": channel,
            "balance_after": acc["available_balance"],
        }
    )
    return None


@mcp_server.tool()
def create_fd(
    principal_amount: float,
    tenure_months: int,
    source_account: str,
    payout_type: Literal["cumulative", "monthly", "quarterly"] = "cumulative",
    maturity_instruction: Literal["no_renewal", "renew_principal", "renew_principal_interest"] = "no_renewal",
    nominee: Optional[dict[str, Any]] = None,
) -> str:
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    if source_account not in ctx["linked_accounts"]:
        return _error("VALIDATION_ERROR", f"Source account {source_account} not linked to customer")
    senior = bool(cust.get("profile", {}).get("senior_citizen"))
    slabs = [s for s in RATE_CARDS["fd"]["slabs"] if s["min_months"] < tenure_months <= s["max_months"]]
    rate = (slabs[0]["senior_rate"] if senior else slabs[0]["general_rate"]) if slabs else 7.0
    debit_err = _debit_account(cust, source_account, principal_amount, "FD booking", "internal")
    if debit_err:
        return debit_err
    maturity = principal_amount * (1 + rate / 100 / 4) ** (4 * tenure_months / 12)
    start = _now()
    maturity_date = start + timedelta(days=int(tenure_months * 30.4375))
    fd_id = _ref(
        "FD",
        6,
        key=f"{cust.get('customer_id')}|{principal_amount}|{tenure_months}|{source_account}",
    )
    record = {
        "deposit_id": fd_id,
        "kind": "FD",
        "principal": principal_amount,
        "rate": rate,
        "tenure_months": tenure_months,
        "start_date": start.strftime("%Y-%m-%d"),
        "maturity_date": maturity_date.strftime("%Y-%m-%d"),
        "maturity_amount": round(maturity, 2),
        "payout_type": payout_type,
        "maturity_instruction": maturity_instruction,
        "status": "ACTIVE",
        "payout_account": source_account,
        "nominee": nominee,
    }
    cust.setdefault("deposits", {})[fd_id] = record
    cust.setdefault("login_context", {}).setdefault("linked_products", []).append(fd_id)
    _persist_db()
    return json.dumps(
        {
            "status": "BOOKED",
            "deposit_id": fd_id,
            "principal_amount": principal_amount,
            "rate": rate,
            "tenure_months": tenure_months,
            "payout_type": payout_type,
            "maturity_date": record["maturity_date"],
            "maturity_amount": round(maturity, 2),
            "source_account": source_account,
            "maturity_instruction": maturity_instruction,
            "receipt_ref": _ref("RCPT", 8),
            "booked_at": _now().isoformat(),
        }
    )


@mcp_server.tool()
def create_rd(
    monthly_installment: float,
    tenure_months: int,
    source_account: str,
    maturity_instruction: Literal["no_renewal", "renew_principal", "renew_principal_interest"] = "no_renewal",
    nominee: Optional[dict[str, Any]] = None,
) -> str:
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    if source_account not in ctx["linked_accounts"]:
        return _error("VALIDATION_ERROR", f"Source account {source_account} not linked to customer")
    senior = bool(cust.get("profile", {}).get("senior_citizen"))
    slabs = [s for s in RATE_CARDS["rd"]["slabs"] if s["min_months"] < tenure_months <= s["max_months"]]
    rate = (slabs[0]["senior_rate"] if senior else slabs[0]["general_rate"]) if slabs else 6.8
    debit_err = _debit_account(cust, source_account, monthly_installment, "RD first installment", "internal")
    if debit_err:
        return debit_err
    quarterly_rate = rate / 100 / 4
    balance = 0.0
    for month in range(1, tenure_months + 1):
        balance += monthly_installment
        if month % 3 == 0:
            balance += balance * quarterly_rate
    start = _now()
    maturity_date = start + timedelta(days=int(tenure_months * 30.4375))
    rd_id = _ref(
        "RD",
        6,
        key=f"{cust.get('customer_id')}|{monthly_installment}|{tenure_months}|{source_account}",
    )
    record = {
        "deposit_id": rd_id,
        "kind": "RD",
        "monthly_installment": monthly_installment,
        "rate": rate,
        "tenure_months": tenure_months,
        "installments_paid": 1,
        "total_deposited": monthly_installment,
        "start_date": start.strftime("%Y-%m-%d"),
        "maturity_date": maturity_date.strftime("%Y-%m-%d"),
        "maturity_amount": round(balance, 2),
        "maturity_instruction": maturity_instruction,
        "status": "ACTIVE",
        "source_account": source_account,
        "payout_account": source_account,
        "nominee": nominee,
    }
    cust.setdefault("deposits", {})[rd_id] = record
    cust.setdefault("login_context", {}).setdefault("linked_products", []).append(rd_id)
    _persist_db()
    return json.dumps(
        {
            "status": "BOOKED",
            "deposit_id": rd_id,
            "monthly_installment": monthly_installment,
            "rate": rate,
            "tenure_months": tenure_months,
            "first_installment_debited": monthly_installment,
            "recurring_commitment": f"₹{monthly_installment:.0f}/month for {tenure_months} months",
            "maturity_date": record["maturity_date"],
            "projected_maturity_amount": round(balance, 2),
            "source_account": source_account,
            "maturity_instruction": maturity_instruction,
            "receipt_ref": _ref("RCPT", 8),
            "booked_at": _now().isoformat(),
        }
    )


# 12-14. Deposit closure / renewal


def _closure_quote_for(dep: dict[str, Any]) -> dict[str, Any]:
    today = _now()
    start = datetime.strptime(dep["start_date"], "%Y-%m-%d")
    maturity_date = datetime.strptime(dep["maturity_date"], "%Y-%m-%d")
    matured = today >= maturity_date
    if dep.get("kind") == "RD":
        principal = dep.get("total_deposited", dep.get("monthly_installment", 0) * dep.get("installments_paid", 0))
    else:
        principal = dep.get("principal", 0)
    contract_rate = dep.get("rate", 0.0)
    if matured:
        penalty_rate = 0.0
        payout = dep.get("maturity_amount", principal)
        interest = payout - principal
        penalty_amount = 0.0
    else:
        penalty_rate = DEPOSIT_PENALTY_RATE
        effective_rate = max(contract_rate - penalty_rate, 0.0)
        days_held = max((today - start).days, 0)
        interest = principal * effective_rate / 100 * days_held / 365
        penalty_amount = 0.0  # penalty is expressed as the rate reduction
        payout = principal + interest
    return {
        "deposit_id": dep["deposit_id"],
        "kind": dep.get("kind"),
        "matured": matured,
        "principal": round(principal, 2),
        "contract_rate": contract_rate,
        "penalty_rate": penalty_rate,
        "effective_rate": round(max(contract_rate - penalty_rate, 0.0), 2) if not matured else contract_rate,
        "interest_payable": round(interest, 2),
        "penalty_amount": round(penalty_amount, 2),
        "net_payout": round(payout, 2),
        "quote_date": _today_str(),
    }


@mcp_server.tool()
def get_deposit_closure_quote(deposit_ids: list[str]) -> str:
    """Read-only; penalty applies pre-maturity, none after."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    deposits = cust.get("deposits", {})
    quotes, errors = [], []
    for did in deposit_ids:
        dep = deposits.get(did)
        if not dep or did not in ctx["linked_products"]:
            errors.append({"deposit_id": did, "error": "Deposit not linked to logged-in customer"})
            continue
        quotes.append(_closure_quote_for(dep))
    return json.dumps({"quotes": quotes, "errors": errors})


@mcp_server.tool()
def close_deposit(deposit_id: str, payout_account: Optional[str] = None) -> str:
    """Works premature (penalty) or post-maturity; for an RD also permanently
    stops future installment debits."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    dep = cust.get("deposits", {}).get(deposit_id)
    if not dep or deposit_id not in ctx["linked_products"]:
        return _error("VALIDATION_ERROR", f"Deposit {deposit_id} not linked to logged-in customer")
    if dep.get("status") == "CLOSED":
        return _error("ALREADY_CLOSED", f"Deposit {deposit_id} is already closed")
    quote = _closure_quote_for(dep)
    credit_account = payout_account or dep.get("payout_account")
    if credit_account and credit_account in cust.get("accounts", {}):
        acc = cust["accounts"][credit_account]
        acc["available_balance"] = round(acc.get("available_balance", 0.0) + quote["net_payout"], 2)
        acc.setdefault("transactions", []).append(
            {
                "reference_id": _ref("TXN", 10),
                "date": _today_str(),
                "description": f"{dep.get('kind')} {deposit_id} closure proceeds",
                "amount": quote["net_payout"],
                "direction": "credit",
                "channel": "internal",
                "balance_after": acc["available_balance"],
            }
        )
    dep["status"] = "CLOSED"
    dep["closed_on"] = _today_str()
    _persist_db()
    return json.dumps(
        {
            "status": "CLOSED",
            "deposit_id": deposit_id,
            "kind": dep.get("kind"),
            "premature": not quote["matured"],
            "penalty_rate": quote["penalty_rate"],
            "net_payout": quote["net_payout"],
            "payout_account": credit_account,
            "future_installments_stopped": dep.get("kind") == "RD",
            "closed_at": _now().isoformat(),
        }
    )


@mcp_server.tool()
def update_deposit_renewal(
    deposit_id: str,
    instruction: Literal["no_renewal", "renew_principal", "renew_principal_interest"],
) -> str:
    """Takes effect at maturity, not immediately."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    dep = cust.get("deposits", {}).get(deposit_id)
    if not dep or deposit_id not in ctx["linked_products"]:
        return _error("VALIDATION_ERROR", f"Deposit {deposit_id} not linked to logged-in customer")
    previous = dep.get("maturity_instruction")
    dep["maturity_instruction"] = instruction
    _persist_db()
    return json.dumps(
        {
            "status": "UPDATED",
            "deposit_id": deposit_id,
            "previous_instruction": previous,
            "new_instruction": instruction,
            "effective_on": dep.get("maturity_date"),
            "updated_at": _now().isoformat(),
        }
    )


# 15-17. Loans


@mcp_server.tool()
def get_loan_details(loan_ids: list[str]) -> str:
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    loans = cust.get("loans", {})
    out, errors = [], []
    for lid in loan_ids:
        loan = loans.get(lid)
        if not loan or lid not in ctx["linked_products"]:
            errors.append({"loan_id": lid, "error": "Loan not linked to logged-in customer"})
            continue
        out.append(loan)
    return json.dumps({"loans": out, "errors": errors})


@mcp_server.tool()
def get_loan_foreclosure_quote(loan_ids: list[str], as_of_date: Optional[str] = None) -> str:
    """Informational only. Also use when a customer asks to pause/stop/reduce
    EMIs, since no such capability exists."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    loans = cust.get("loans", {})
    value_date = as_of_date or _today_str()
    quotes, errors = [], []
    for lid in loan_ids:
        loan = loans.get(lid)
        if not loan or lid not in ctx["linked_products"]:
            errors.append({"loan_id": lid, "error": "Loan not linked to logged-in customer"})
            continue
        outstanding = loan.get("outstanding", 0.0)
        rate = loan.get("rate", 0.0)
        accrued_interest = outstanding * rate / 100 * 30 / 365
        # Foreclosure charge: nil for floating home loans, else ~2%.
        charge_rate = 0.0 if loan.get("loan_type") == "home_loan" else 2.0
        foreclosure_charges = outstanding * charge_rate / 100
        gst = foreclosure_charges * 0.18
        total = outstanding + accrued_interest + foreclosure_charges + gst
        quotes.append(
            {
                "loan_id": lid,
                "as_of_date": value_date,
                "loan_type": loan.get("loan_type"),
                "outstanding_principal": round(outstanding, 2),
                "accrued_interest": round(accrued_interest, 2),
                "foreclosure_charges": round(foreclosure_charges, 2),
                "gst_on_charges": round(gst, 2),
                "total_payable": round(total, 2),
            }
        )
    return json.dumps(
        {
            "quotes": quotes,
            "errors": errors,
            "note": "No tool closes a loan; route the actual foreclosure via raise_request.",
        }
    )


@mcp_server.tool()
def calculate_emi(principal: float, rate: float, tenure_months: int) -> str:
    r = rate / 100 / 12
    if r == 0:
        emi = principal / tenure_months
    else:
        emi = principal * r * (1 + r) ** tenure_months / ((1 + r) ** tenure_months - 1)
    schedule = []
    balance = principal
    for month in range(1, tenure_months + 1):
        interest_component = balance * r
        principal_component = emi - interest_component
        balance = balance - principal_component
        if month == tenure_months:
            principal_component += balance  # absorb rounding so it closes at 0
            balance = 0.0
        schedule.append(
            {
                "month": month,
                "emi": round(emi, 2),
                "principal_component": round(principal_component, 2),
                "interest_component": round(interest_component, 2),
                "closing_balance": round(max(balance, 0.0), 2),
            }
        )
    total_payment = emi * tenure_months
    return json.dumps(
        {
            "principal": principal,
            "rate": rate,
            "tenure_months": tenure_months,
            "emi": round(emi, 2),
            "total_interest": round(total_payment - principal, 2),
            "total_repayment": round(total_payment, 2),
            "amortisation_schedule": schedule,
        }
    )


# 18-19. Gold


@mcp_server.tool()
def get_gold_rate(purity_karat: Literal[18, 22, 24]) -> str:
    rate = GOLD_RATES_PER_GRAM.get(purity_karat)
    if rate is None:
        return _error("VALIDATION_ERROR", f"Unsupported purity {purity_karat}")
    return json.dumps(
        {
            "purity_karat": purity_karat,
            "gold_rate_per_gram": rate,
            "currency": "INR",
            "date": _today_str(),
            "source": "MCX",
            "valid_till": (_now() + timedelta(hours=8)).isoformat(),
        }
    )


@mcp_server.tool()
def calculate_gold_loan_ltv(gold_grams: float, purity_karat: Literal[18, 22, 24], gold_rate_per_gram: float) -> str:
    gross_value = gold_grams * gold_rate_per_gram
    eligible = gross_value * GOLD_LTV_CAP / 100
    return json.dumps(
        {
            "gold_grams": gold_grams,
            "purity_karat": purity_karat,
            "gold_rate_per_gram": gold_rate_per_gram,
            "gross_value": round(gross_value, 2),
            "ltv_cap_percent": GOLD_LTV_CAP,
            "max_eligible_loan_amount": round(eligible, 2),
        }
    )


# 20-23. Cards


@mcp_server.tool()
def get_card_details(card_ids: list[str]) -> str:
    """Use to diagnose card declines."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    cards = cust.get("cards", {})
    out, errors = [], []
    for cid in card_ids:
        card = cards.get(cid)
        if not card or cid not in ctx["linked_cards"]:
            errors.append({"card_id": cid, "error": "Card not linked to logged-in customer"})
            continue
        out.append(card)
    return json.dumps({"cards": out, "errors": errors})


@mcp_server.tool()
def toggle_card_freeze(card_id: str, state: Literal["freeze", "unfreeze"]) -> str:
    """Cannot unfreeze a blocked or expired card."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    card = cust.get("cards", {}).get(card_id)
    if not card or card_id not in ctx["linked_cards"]:
        return _error("VALIDATION_ERROR", f"Card {card_id} not linked to logged-in customer")
    status = card.get("status")
    if status in ("blocked", "expired"):
        return _error("INVALID_STATE", f"Cannot {state} a card whose status is {status}")
    card["status"] = "frozen" if state == "freeze" else "active"
    _persist_db()
    return json.dumps(
        {
            "status": "UPDATED",
            "card_id": card_id,
            "action": state,
            "new_card_status": card["status"],
            "updated_at": _now().isoformat(),
        }
    )


@mcp_server.tool()
def block_card(card_id: str, reason: Literal["lost", "stolen", "fraud", "damaged"]) -> str:
    """Irreversible; request a replacement via raise_request."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    card = cust.get("cards", {}).get(card_id)
    if not card or card_id not in ctx["linked_cards"]:
        return _error("VALIDATION_ERROR", f"Card {card_id} not linked to logged-in customer")
    if card.get("status") == "blocked":
        return _error("ALREADY_BLOCKED", f"Card {card_id} is already blocked")
    card["status"] = "blocked"
    card["block_reason"] = reason
    _persist_db()
    return json.dumps(
        {
            "status": "BLOCKED",
            "card_id": card_id,
            "reason": reason,
            "reversible": False,
            "note": "Card permanently blocked. Raise a service request for a replacement.",
            "blocked_at": _now().isoformat(),
        }
    )


@mcp_server.tool()
def set_card_controls(
    card_id: str,
    atm: Optional[dict[str, Any]] = None,
    online: Optional[dict[str, Any]] = None,
    pos: Optional[dict[str, Any]] = None,
) -> str:
    """daily_limit is ignored if enabled=false."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    card = cust.get("cards", {}).get(card_id)
    if not card or card_id not in ctx["linked_cards"]:
        return _error("VALIDATION_ERROR", f"Card {card_id} not linked to logged-in customer")
    controls = card.setdefault("controls", {})
    for channel, payload in (("atm", atm), ("online", online), ("pos", pos)):
        if payload is None:
            continue
        chan_ctrl = controls.setdefault(channel, {})
        for region in ("domestic", "international"):
            region_payload = payload.get(region)
            if region_payload is None:
                continue
            enabled = bool(region_payload.get("enabled"))
            new_region = {"enabled": enabled}
            if enabled and "daily_limit" in region_payload:
                new_region["daily_limit"] = region_payload["daily_limit"]
            chan_ctrl[region] = new_region
    _persist_db()
    return json.dumps(
        {
            "status": "UPDATED",
            "card_id": card_id,
            "controls": controls,
            "updated_at": _now().isoformat(),
        }
    )


# 24-25. Mandates


@mcp_server.tool()
def show_mandates(
    account_id: Optional[str] = None,
    status: Literal["active", "paused", "cancelled", "all"] = "all",
) -> str:
    cust, err = _active_customer()
    if err:
        return err
    mandates = list(cust.get("mandates", []))
    if account_id:
        mandates = [m for m in mandates if m.get("account_id") == account_id]
    if status != "all":
        mandates = [m for m in mandates if m.get("status") == status]
    return json.dumps({"total": len(mandates), "mandates": mandates})


@mcp_server.tool()
def cancel_mandate(mandate_id: str) -> str:
    cust, err = _active_customer()
    if err:
        return err
    for mandate in cust.get("mandates", []):
        if mandate.get("mandate_id") == mandate_id:
            if mandate.get("status") == "cancelled":
                return _error("ALREADY_CANCELLED", f"Mandate {mandate_id} is already cancelled")
            mandate["status"] = "cancelled"
            mandate["cancelled_on"] = _today_str()
            _persist_db()
            return json.dumps(
                {
                    "status": "CANCELLED",
                    "mandate_id": mandate_id,
                    "payee": mandate.get("payee"),
                    "future_debits_stopped": True,
                    "cancelled_at": _now().isoformat(),
                }
            )
    return _error("VALIDATION_ERROR", f"Mandate {mandate_id} not found for logged-in customer")


# 26-27. Cheques


@mcp_server.tool()
def stop_cheque_payment(
    account_id: str,
    cheque_numbers: list[str],
    reason: Optional[Literal["lost", "dispute", "other"]] = None,
) -> str:
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    acc = cust.get("accounts", {}).get(account_id)
    if not acc or account_id not in ctx["linked_accounts"]:
        return _error("VALIDATION_ERROR", f"Account {account_id} not linked to logged-in customer")
    cheques = {c.get("cheque_number"): c for c in acc.get("cheques", [])}
    results = []
    for num in cheque_numbers:
        cheque = cheques.get(num)
        if cheque is None:
            # unknown cheque number: still register the stop optimistically
            results.append(
                {"cheque_number": num, "outcome": "STOP_PLACED", "note": "no record found; stop registered"}
            )
            acc.setdefault("cheques", []).append({"cheque_number": num, "status": "stopped"})
        elif cheque.get("status") == "cleared":
            results.append({"cheque_number": num, "outcome": "CANNOT_STOP", "note": "cheque already cleared"})
        else:
            cheque["status"] = "stopped"
            results.append({"cheque_number": num, "outcome": "STOP_PLACED"})
    _persist_db()
    return json.dumps(
        {
            "account_id": account_id,
            "reason": reason or "other",
            "reference": _ref("STP", 8),
            "results": results,
        }
    )


@mcp_server.tool()
def request_cheque_book(
    account_id: str,
    leaves: int = 25,
    delivery_address: Optional[str] = None,
) -> str:
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    if account_id not in ctx["linked_accounts"]:
        return _error("VALIDATION_ERROR", f"Account {account_id} not linked to logged-in customer")
    return json.dumps(
        {
            "status": "REQUESTED",
            "request_id": _ref("CHQ", 8),
            "account_id": account_id,
            "leaves": leaves,
            "delivery_address": delivery_address or "registered address",
            "expected_delivery": (_now() + timedelta(days=7)).strftime("%Y-%m-%d"),
            "requested_at": _now().isoformat(),
        }
    )


# 28. Duplicate statement


@mcp_server.tool()
def request_duplicate_statement(
    account_id: str,
    from_date: str,
    to_date: str,
    delivery_mode: Literal["email", "download"] = "email",
) -> str:
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    if account_id not in ctx["linked_accounts"]:
        return _error("VALIDATION_ERROR", f"Account {account_id} not linked to logged-in customer")
    ref = _ref("STMT", 8)
    resp: dict[str, Any] = {
        "status": "REQUESTED",
        "request_id": ref,
        "account_id": account_id,
        "from_date": from_date,
        "to_date": to_date,
        "delivery_mode": delivery_mode,
        "requested_at": _now().isoformat(),
    }
    if delivery_mode == "email":
        resp["delivered_to"] = cust.get("profile", {}).get("email", "registered email")
    else:
        resp["download_url"] = f"https://bank.example.com/statements/{ref}.pdf"
        resp["expiry"] = (_now() + timedelta(days=7)).isoformat()
    return json.dumps(resp)


# 29. Address update


@mcp_server.tool()
def update_address(
    address_type: Literal["communication", "permanent"],
    line1: str,
    city: str,
    state: str,
    pincode: str,
    line2: Optional[str] = None,
) -> str:
    """Effective only after verification; never tell the customer it's already
    updated."""
    cust, err = _active_customer()
    if err:
        return err
    requested = {"line1": line1, "line2": line2, "city": city, "state": state, "pincode": pincode}
    cust.setdefault("pending_address_updates", []).append(
        {
            "address_type": address_type,
            "address": requested,
            "requested_at": _now().isoformat(),
            "status": "PENDING_VERIFICATION",
        }
    )
    _persist_db()
    return json.dumps(
        {
            "status": "REQUEST_LOGGED",
            "request_id": _ref("ADR", 8),
            "address_type": address_type,
            "requested_address": requested,
            "verification_status": "PENDING_VERIFICATION",
            "note": "Request logged; change is effective only after bank verification.",
            "logged_at": _now().isoformat(),
        }
    )


# 30-31. Service requests


@mcp_server.tool()
def raise_request(
    description: str,
    category: Literal[
        "failed_transaction",
        "unauthorized_debit",
        "service_quality",
        "app_issue",
        "charges_dispute",
        "general_query",
        "other",
    ] = "general_query",
    related_transaction_id: Optional[str] = None,
    account_id: Optional[str] = None,
) -> str:
    """Escalation channel for anything no other tool handles. Infer `category`
    and `related_transaction_id` from the conversation; don't ask the customer
    for backend category codes."""
    cust, err = _active_customer()
    if err:
        return err
    if category in ("failed_transaction", "unauthorized_debit") and not related_transaction_id:
        return _error("VALIDATION_ERROR", f"related_transaction_id is required for category {category}")
    urgent = category == "unauthorized_debit"
    request_id = _ref("SR-", 8)
    eta_days = 1 if urgent else 5
    record = {
        "request_id": request_id,
        "category": category,
        "description": description,
        "related_transaction_id": related_transaction_id,
        "account_id": account_id,
        "status": "OPEN",
        "priority": "URGENT" if urgent else "NORMAL",
        "created_at": _now().isoformat(),
        "eta": (_now() + timedelta(days=eta_days)).strftime("%Y-%m-%d"),
    }
    cust.setdefault("requests", {})[request_id] = record
    _persist_db()
    return json.dumps(
        {
            "status": "RAISED",
            "request_id": request_id,
            "category": category,
            "priority": record["priority"],
            "eta": record["eta"],
            "note": "Request routed for review; no specific outcome is promised.",
            "created_at": record["created_at"],
        }
    )


@mcp_server.tool()
def get_request_status(request_id: str) -> str:
    cust, err = _active_customer()
    if err:
        return err
    record = cust.get("requests", {}).get(request_id)
    if not record:
        return _error("VALIDATION_ERROR", f"Service request {request_id} not found")
    return json.dumps(
        {
            "request_id": request_id,
            "category": record.get("category"),
            "status": record.get("status", "OPEN"),
            "priority": record.get("priority"),
            "description": record.get("description"),
            "created_at": record.get("created_at"),
            "eta": record.get("eta"),
        }
    )


# 32. Insurance


@mcp_server.tool()
def get_insurance_details(policy_ids: list[str]) -> str:
    """No premium-payment or claim tools exist; route those via raise_request."""
    cust, err = _active_customer()
    if err:
        return err
    ctx = _login_context(cust)
    policies = cust.get("insurance", {})
    out, errors = [], []
    for pid in policy_ids:
        pol = policies.get(pid)
        if not pol or pid not in ctx["linked_products"]:
            errors.append({"policy_id": pid, "error": "Policy not linked to logged-in customer"})
            continue
        out.append(pol)
    return json.dumps(
        {"policies": out, "errors": errors, "note": "Premium payment and claims are handled via raise_request."}
    )


# 33. Products & offers


@mcp_server.tool()
def get_products_and_offers(
    type: Literal["deposit", "loan", "card", "insurance", "offer", "all"] = "all",
) -> str:
    """Offers are indicative, never guaranteed. Get rates from
    get_deposit_loan_rates, not here."""
    cust, _ = _active_customer()
    products = PRODUCT_CATALOG
    offers = cust.get("offers", []) if cust else []
    if type == "offer":
        products = []
    elif type != "all":
        products = [p for p in products if p["type"] == type]
        offers = [o for o in offers if o.get("type") == type]
    return json.dumps(
        {
            "filter": type,
            "products": products,
            "offers": offers,
            "disclaimer": "Offers are indicative and subject to eligibility verification. "
            "Rates must be confirmed via get_deposit_loan_rates.",
        }
    )
