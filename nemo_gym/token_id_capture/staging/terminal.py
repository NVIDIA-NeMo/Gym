# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select a terminal call when the harness did not identify one.

The selector follows ``parent_call_id`` links in the manifest.
It ignores an abandoned leaf when a sibling has descendants.
It returns no result when more than one terminal call remains possible.

This function does not inspect token data.
``verify_and_linearize`` validates the selected chain before training.
``verify_and_linearize_all`` reuses the same helpers to walk every other root
(subagent sessions) down to its own leaf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

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


def root_order_key(records: Sequence[CallRecord]) -> Callable[[CallRecord], tuple]:
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


def children_by_parent(records: Sequence[CallRecord]) -> dict[str, list[CallRecord]]:
    """Map each call to its token-linked children, in manifest order."""
    children: dict[str, list[CallRecord]] = {}
    for record in records:
        if record.parent_call_id is not None:
            children.setdefault(record.parent_call_id, []).append(record)
    return children


def survivors(
    candidates: Sequence[CallRecord],
    children: Mapping[str, Sequence[CallRecord]],
) -> list[CallRecord]:
    """Keep the candidates that have token-linked children; fall back to all of them.

    An abandoned sibling is only eliminated when another candidate was extended.
    """
    extended = [record for record in candidates if children.get(record.model_call_id)]
    return extended or list(candidates)


def descend_to_leaf(root_id: str, children: Mapping[str, Sequence[CallRecord]]) -> str | None:
    """Walk token links from ``root_id`` to a unique leaf; ``None`` when ambiguous.

    At each level children with descendants are preferred over childless
    siblings; more than one surviving child is ambiguous.
    """
    node_id = root_id
    while True:
        candidates = children.get(node_id) or []
        if not candidates:
            return node_id
        pool = survivors(candidates, children)
        if len(pool) > 1:
            return None
        node_id = pool[0].model_call_id


def select_terminal_call(records: Sequence[CallRecord]) -> TerminalSelection:
    """Select a terminal call from parent links when the harness did not provide one.

    Records may contain multiple roots and branches.
    A root with children is preferred over roots without children.
    Remaining roots are ordered by admission time and then manifest position.
    At each subsequent level, children with descendants are preferred over childless siblings.
    Selection continues when exactly one preferred child remains.
    If multiple preferred children remain, no terminal call is returned.
    """
    if not records:
        return TerminalSelection(None, NO_RECORDS)

    by_id = {record.model_call_id: record for record in records}
    if len(by_id) != len(records):
        return TerminalSelection(None, DUPLICATE_CALL_ID)
    for record in records:
        if record.parent_call_id is not None and record.parent_call_id not in by_id:
            return TerminalSelection(None, ORPHANED_ROW)

    children = children_by_parent(records)
    roots = [record for record in records if record.parent_call_id is None]
    if not roots:
        # Every row names a present parent: the graph is cyclic.
        return TerminalSelection(None, NO_ROOT)

    key = root_order_key(records)
    node = min(survivors(roots, children), key=key)
    while True:
        candidates = children.get(node.model_call_id) or []
        if not candidates:
            return TerminalSelection(node.model_call_id, SELECTED)
        pool = survivors(candidates, children)
        if len(pool) > 1:
            return TerminalSelection(None, AMBIGUOUS_TERMINAL)
        node = pool[0]
