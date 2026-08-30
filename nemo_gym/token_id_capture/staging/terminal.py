# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Heuristic terminal-call selection for rollouts without a declared terminal.

A harness that reports the response id it kept gives receipt assembly an
authoritative terminal row. A harness that reports nothing leaves the ledger
manifest as the only evidence. This module infers the terminal call from the
manifest's explicit parent links alone: it never reads tokens, so a wrong
inference can only select a shorter verified chain, never corrupt one —
``verify_and_linearize`` still checks every digest downstream.

Selection is fail-closed. Any shape with two defensible answers (a retry of
the final call, divergent extended branches) returns no terminal and a reason
the caller records as the rollout's failure reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from nemo_gym.token_id_capture.staging.records import (
    TERMINAL_AMBIGUOUS,
    TERMINAL_DUPLICATE_CALL_ID,
    TERMINAL_NO_RECORDS,
    TERMINAL_NO_ROOT,
    TERMINAL_ORPHANED_ROW,
    TERMINAL_SELECTED,
    CallRecord,
)


# Local aliases: the selection vocabulary lives with the wire schemas.
SELECTED = TERMINAL_SELECTED
NO_RECORDS = TERMINAL_NO_RECORDS
DUPLICATE_CALL_ID = TERMINAL_DUPLICATE_CALL_ID
ORPHANED_ROW = TERMINAL_ORPHANED_ROW
NO_ROOT = TERMINAL_NO_ROOT
AMBIGUOUS_TERMINAL = TERMINAL_AMBIGUOUS


@dataclass(frozen=True)
class TerminalSelection:
    """The inferred terminal call, or the reason none could be chosen."""

    terminal_model_call_id: str | None
    reason: str


def _root_order_key(records: Sequence[CallRecord]):
    """Order candidate roots by admission time, unstamped rows last.

    ``admitted_at`` is absent on rows written before the column existed and on
    any path that fails to thread it; those rows sort after every stamped row.
    Manifest position breaks exact-timestamp ties (ledger append order is the
    commit order), which also makes the all-unstamped case deterministic.
    """
    index_by_id = {record.model_call_id: position for position, record in enumerate(records)}

    def key(record: CallRecord) -> tuple:
        return (
            record.admitted_at is None,
            record.admitted_at if record.admitted_at is not None else 0.0,
            index_by_id[record.model_call_id],
        )

    return key


def select_terminal_call(records: Sequence[CallRecord]) -> TerminalSelection:
    """Infer the terminal call of an undeclared rollout from parent links.

    The manifest's committed rows form a forest. The main root is the
    earliest-admitted root, except that a root extended by children beats an
    abandoned (childless) sibling root. The walk descends the fork structure,
    eliminating abandoned retries the same way: a childless child loses to an
    extended sibling. One surviving child continues the chain; several
    surviving candidates are ambiguous and mask the rollout — a retry of the
    final call is indistinguishable from its survivor, so it always masks.
    """
    if not records:
        return TerminalSelection(None, NO_RECORDS)

    by_id = {record.model_call_id: record for record in records}
    if len(by_id) != len(records):
        return TerminalSelection(None, DUPLICATE_CALL_ID)
    for record in records:
        if record.parent_call_id is not None and record.parent_call_id not in by_id:
            return TerminalSelection(None, ORPHANED_ROW)

    children: dict[str, list[CallRecord]] = {}
    roots: list[CallRecord] = []
    for record in records:
        if record.parent_call_id is None:
            roots.append(record)
        else:
            children.setdefault(record.parent_call_id, []).append(record)
    if not roots:
        # Every row names a present parent: the graph is cyclic.
        return TerminalSelection(None, NO_ROOT)

    key = _root_order_key(records)

    def survivors(candidates: list[CallRecord]) -> list[CallRecord]:
        extended = [record for record in candidates if children.get(record.model_call_id)]
        return extended or candidates

    root_pool = survivors(roots)
    node = min(root_pool, key=key)
    while True:
        candidates = children.get(node.model_call_id) or []
        if not candidates:
            return TerminalSelection(node.model_call_id, SELECTED)
        pool = survivors(candidates)
        if len(pool) > 1:
            return TerminalSelection(None, AMBIGUOUS_TERMINAL)
        node = pool[0]
