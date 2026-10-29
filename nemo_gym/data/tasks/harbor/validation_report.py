# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarise a gym dataset validate run from its rollout and failure files.

Reads the rows a validation run wrote, keys them by task, reports which tasks passed, which
scored below one and which were never run, and returns the text and the overall verdict as a
ValidationReport. The task digest lookup the resume checks share lives here as well. cli.py
imports it for validate_target and the eval run resume conflict check.
"""

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_gym.config.config_types import ConfigError
from nemo_gym.data.tasks.harbor.digest import DIGEST_KEY
from nemo_gym.utils.path_utils import failures_path_for


@dataclass(frozen=True)
class ValidationReport:
    text: str
    ok: bool


def _dig(mapping: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(mapping, dict):
            return None
        mapping = mapping.get(key)
    return mapping


def _task_name(row: dict[str, Any]) -> str:
    identity = row.get("_ng_task_id") or row.get("task_id") or {}
    return str(identity.get("task_id", "?")) if isinstance(identity, dict) else str(identity)


def _validation_key(row: dict[str, object]) -> tuple[str, str, object, object]:
    identity = row.get("_ng_task_id") or row.get("task_id") or {}
    taskset = str(identity.get("taskset", "")) if isinstance(identity, dict) else ""
    return taskset, _task_name(row), row.get("_ng_task_index"), row.get("_ng_rollout_index")


def _iter_validation_rows(path: Path) -> Iterator[dict[str, object]]:
    """The records of a rollouts, failures or materialized inputs file, in order (none when there is no file).

    One record is read and parsed at a time, so a reader that keeps a projection of each row (the run
    summary) never holds a long run's responses together. A run killed while appending leaves a final line
    without a newline that does not parse (the cut may land inside a multibyte character, so it may not
    decode either). The collector drops that line from the rollouts and failures files when it resumes
    (``_drop_truncated_tail``), so it is skipped here with the same tolerance, and a resume that reads the
    cache first still reaches that repair. A complete final record without a newline is read like any
    other. A line that does not parse anywhere else is damage the collector does not repair, and is reported
    by file and line rather than resumed over.
    """
    if not path.is_file():
        return
    with path.open("rb") as handle:
        for number, line in enumerate(handle, 1):
            terminated = line.endswith(b"\n")
            line = line[:-1] if terminated else line
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:  # JSONDecodeError, or UnicodeDecodeError for bytes that are not UTF-8
                if not terminated:
                    return  # the unterminated last line
                detail = f"{exc.msg}, column {exc.colno}" if isinstance(exc, json.JSONDecodeError) else str(exc)
                raise ConfigError(f"{path}, line {number}: not a JSON record ({detail}).") from exc
            yield row


def _validation_rows(path: Path) -> list[dict[str, object]]:
    """:func:`_iter_validation_rows` as a list, for the readers that need every record at once."""
    return list(_iter_validation_rows(path))


def summarize_validation(
    rollouts: Path,
    unvalidated: list[str],
    *,
    expected: list[dict[str, object]] | None = None,
    skipped: dict[str, str] | None = None,
) -> ValidationReport:
    """Report every selected task/repeat, reconciling results with failed attempts.

    The collector stores verify responses directly and keeps historical failed attempts in a
    sidecar. A result supersedes failures for the same task/repeat, but never another repeat.
    ``expected`` contains the collector's selected, expanded input rows (after limit/repeats).
    """
    lines = ["task_id\tstatus\treward\tverify_s\tnetwork"]
    ok = True
    selected = {_validation_key(row): row for row in expected} if expected is not None else None
    results = {_validation_key(row): row for row in _validation_rows(rollouts)}
    failures = {_validation_key(row): row for row in _validation_rows(failures_path_for(rollouts))}
    keys = selected if selected is not None else dict.fromkeys([*results, *failures])
    if selected is not None and not selected:
        # Nothing ran: no selected task has a solution. Reporting that as a pass would let
        # `gym dataset validate X && publish` go ahead with no reference solution checked.
        ok = False
        lines.append("-\t0 tasks validated (no selected task has a solution/)\t-\t-\t-")
    for key in keys:
        _, task_id, _, repeat = key
        label = f"{task_id} [repeat {repeat}]" if repeat is not None else task_id
        if task_id in unvalidated:
            continue
        row = results.get(key)
        if row is None:
            ok = False
            failure = failures.get(key)
            if failure is None:
                lines.append(f"{label}\tmissing result\t-\t-\t-")
            else:
                message = failure.get("_ng_failure_message") or failure.get("_ng_failure_class") or "failed"
                lines.append(f"{label}\tfailed: {str(message)[:120]}\t-\t-\t-")
            continue
        oracle = _dig(row, "response", "metadata", "oracle")
        reward = row.get("reward")
        # What the verifier cost and used, per task: its wall time and the network it ran under.
        verify_s = row.get("verifier_seconds")
        verify_note = (
            f"{verify_s:.1f}" if isinstance(verify_s, (int, float)) and not isinstance(verify_s, bool) else "-"
        )
        network = row.get("verifier_network") or "-"
        if row.get("mask_sample"):
            ok = False
            lines.append(f"{label}\tmasked ({row.get('failure_kind')})\t-\t{verify_note}\t{network}")
        elif oracle == "unvalidated":
            # A selected reference solution disappearing is not a successful validation.
            if selected is not None:
                ok = False
            lines.append(f"{label}\tunvalidated (no solution/)\t-\t-\t-")
        else:
            # Only the oracle agent stamps `response.metadata.oracle` (``solved`` or ``failed`` here). A row
            # without it came from another producer, such as a model run that wrote to this file, and a
            # status the agent never emits is no validation either: neither may pass the task.
            if oracle not in ("solved", "failed"):
                oracle = "missing" if oracle is None else f"status {oracle!r}"
                oracle += " (not an oracle result)"
            scored = isinstance(reward, (int, float)) and not isinstance(reward, bool) and math.isfinite(reward)
            if oracle != "solved" or not scored or reward < 1.0:
                ok = False
            kind = row.get("failure_kind")
            note = f" ({kind})" if kind else ""
            lines.append(f"{label}\toracle {oracle or 'ran'}{note}\t{reward}\t{verify_note}\t{network}")
    if selected is None and not rollouts.is_file():
        ok = False
        lines.append(f"-\tno rollouts written at {rollouts}\t-\t-\t-")
    for task_id in unvalidated:
        lines.append(f"{task_id}\tunvalidated (no solution/)\t-\t-\t-")
    for task_id, reason in (skipped or {}).items():
        ok = False
        lines.append(f"{task_id}\tskipped: {reason}\t-\t-\t-")
    return ValidationReport(text="\n".join(lines), ok=ok)


def _task_digests(rows: list[dict[str, object]], key: Any) -> dict[Any, object]:
    """``key(row) -> task digest`` for materialized rows; shared by the eval and validation resume checks."""
    return {key(row): _dig(row, "task_input", "task_data", DIGEST_KEY) for row in rows}
