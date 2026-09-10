# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Shared helpers for domain noise generators.

Fairness rule: noise belongs in DATA/transactional worksheets, never in
reference/lookup/policy worksheets. A reference table holds fixed config the agent
reads BY KEY (tier premiums, SLA targets, approver routing, scoring weights); adding
noise rows there either collides with a real key or creates ambiguity, making a
correct solution's lookup wrong. So the injectors skip these worksheets entirely.
"""

_REFERENCE_WS_TOKENS = (
    "tier",
    "target",
    "polic",
    "criteria",
    "premium",
    "weight",
    "rout",
    "approver",
    "discount",
    "trial_limit",
    "owner",
    "_rule",
    "rules",
    "_model",
    "cpl",
    "sla",
    "pricing",
    "threshold",
    "matrix",
    "lookup",
    "config",
    "override",
    "mapping",
    "classification",
    # per-entity lookup/ledger tables the agent reads by name/key
    "exposure",
    "drawdown",
    "correction",
    "limit",
    "reconcil",
    "ledger",
    "weighting",
    "band",
    "schedule_rule",
)

# A worksheet this small is almost certainly a fixed reference/config table, not a
# transactional data table — injecting 15 noise rows would dominate it and, if it is a
# lookup table, collide with real keys. Skip noise for tables at or below this size.
_REFERENCE_WS_MAX_ROWS = 8


def is_reference_ws(ws: dict) -> bool:
    """True if a worksheet looks like a reference/lookup/policy table (skip noise there)."""
    ref = (str(ws.get("id", "")) + " " + str(ws.get("title", ""))).lower()
    if any(tok in ref for tok in _REFERENCE_WS_TOKENS):
        return True
    rows = ws.get("rows", [])
    return isinstance(rows, list) and 0 < len(rows) <= _REFERENCE_WS_MAX_ROWS
