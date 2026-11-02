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

"""Read rollout rows and what each row says about itself.

Strict per-line JSONL parsing of the main file and the failures sidecar, with the last failure
per attempt folded in as a masked row, and the per-row signals: the mask reason, the measured
failure kind, the agent error and the strict numeric reward. summarize_side reduces one side's
rows for a task to a TaskSide.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Optional

from nemo_gym.config.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.rollout_compare.join import _lookup, task_id_of
from nemo_gym.rollout_compare.schema import (
    FAILURES_SUFFIX,
    SIDECAR_FLAG,
    TASK_KEY_CANDIDATES,
    CompareInputError,
    TaskSide,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise CompareInputError(f"cannot read {path}: {exc.strerror or exc}") from exc
    rows: list[dict[str, Any]] = []
    # "\n" only: str.splitlines would also break a line at U+2028, U+2029 or U+0085 inside a string field.
    for lineno, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise CompareInputError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
        if not isinstance(row, dict):
            raise CompareInputError(f"{path}:{lineno}: expected a JSON object per line")
        try:
            _reward_of(row)
        except CompareInputError as exc:
            raise CompareInputError(f"{path}:{lineno}: {exc} (task {_row_task_label(row)})") from exc
        rows.append(row)
    return rows


def _row_task_label(row: dict[str, Any]) -> str:
    """Best-effort task name for an error about one row, before the join key is known."""
    for candidate in TASK_KEY_CANDIDATES:
        if _lookup(row, candidate) is not None:
            return task_id_of(row, candidate, with_taskset=True)
    return "unknown"


def _failure_as_masked_row(failure: dict[str, Any]) -> dict[str, Any]:
    """A `<name>_failures.jsonl` record becomes a masked row with no reward."""
    row = dict(failure)
    row[SIDECAR_FLAG] = True
    row.setdefault("reward", None)
    row["mask_sample"] = True
    row.setdefault("failure_kind", failure.get("_ng_failure_class") or "failure")
    row.setdefault("failure_reason", failure.get("_ng_failure_message"))
    return row


def _attempt_key(row: dict[str, Any]) -> Optional[tuple[Any, Any]]:
    """The collector's (task index, rollout index) identity of a row, or None when the row does not carry it."""
    if TASK_INDEX_KEY_NAME in row and ROLLOUT_INDEX_KEY_NAME in row:
        return row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]
    return None


def current_failures(failures: list[dict[str, Any]], scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The sidecar records that still stand: the collector writes one record per attempt and a retried rollout
    can end as a scored row in the main file, so only the last record per (task index, rollout index) counts and
    none whose rollout was scored after all. Records without the indices are kept as they are."""
    scored_keys = {key for key in map(_attempt_key, scored) if key is not None}
    latest: dict[tuple[Any, Any], dict[str, Any]] = {}
    unkeyed: list[dict[str, Any]] = []
    for record in failures:  # file order is attempt order, so a later record overwrites an earlier one
        key = _attempt_key(record)
        if key is None:
            unkeyed.append(record)
        else:
            latest[key] = record
    return [record for key, record in latest.items() if key not in scored_keys] + unkeyed


def load_rollouts(path: str | Path) -> list[dict[str, Any]]:
    """Read a rollouts JSONL plus, when present, its sibling `<name>_failures.jsonl` of masked failures. A failure
    superseded by a later attempt (another sidecar record or a scored row) is left out. A `record_omitted_rollouts`
    record (`_ng_omitted_rollout`) is never a failure: an interrupted attempt still supersedes older ones and leaves
    the rollout missing, and a row never started is skipped."""
    path = Path(path)
    rows = _read_jsonl(path)
    if path.suffix == ".jsonl":
        failures_path = path.with_name(path.stem + FAILURES_SUFFIX)
        if failures_path.is_file():
            attempts = [
                record
                for record in _read_jsonl(failures_path)
                if not (record.get("_ng_omitted_rollout") and record.get("_ng_dispatch_drained"))
            ]
            failures = [record for record in current_failures(attempts, rows) if not record.get("_ng_omitted_rollout")]
            rows.extend(_failure_as_masked_row(f) for f in failures)
    return rows


def mask_reason(row: dict[str, Any]) -> Optional[str]:
    """Why a row does not count: `mask_sample`, no reward, or a NaN/infinite reward.

    A `failure_kind` alone does not mask: with `mask_sample` false it labels a measured reward (e.g. the Harbor
    server's scored zero for a verifier timeout), which counts like any other reward; see `measured_failure_kind`.
    """
    if row.get("mask_sample"):
        kind = row.get("failure_kind")
        return f"mask_sample (failure_kind={kind})" if kind else "mask_sample"
    reward = row.get("reward")
    if reward is None:
        return "no reward"
    if isinstance(reward, float) and not math.isfinite(reward):
        return f"non-finite reward ({reward!r})"
    return None


def measured_failure_kind(row: dict[str, Any]) -> Optional[str]:
    """The `failure_kind` of a row that still counts (not masked): a scored failure the report labels but does
    not set aside, so a flip it causes is a real flip."""
    kind = row.get("failure_kind")
    return str(kind) if kind and not mask_reason(row) else None


def agent_error(row: dict[str, Any]) -> Optional[str]:
    """An agent-level error the row reports without being masked: the legacy agent's `error` field or an
    `*error_type` entry in the response metadata (e.g. Terminus's `terminus2_error_type`)."""
    error = row.get("error")
    if error:
        last_line = str(error).strip().splitlines()[-1] if str(error).strip() else str(error)
        return f"error: {last_line[:120]}"
    metadata = (row.get("response") or {}).get("metadata") or {}
    if isinstance(metadata, dict):
        for name, value in metadata.items():
            if name.endswith("error_type") and value:
                return f"{name}={value}"
    return None


def _reward_of(row: dict[str, Any]) -> Optional[float]:
    """The row's reward as a float (None when absent; NaN/inf pass through for `mask_reason` to catch). Booleans
    and strings are rejected rather than coerced: a `"1.0"` string means the producer is broken."""
    reward = row.get("reward")
    if reward is None:
        return None
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise CompareInputError(f"reward {reward!r} is not a number")
    return float(reward)


def _finite_reward_of(row: dict[str, Any]) -> Optional[float]:
    reward = _reward_of(row)
    return reward if reward is not None and math.isfinite(reward) else None


def summarize_side(rows: list[dict[str, Any]]) -> TaskSide:
    reasons = [reason for reason in (mask_reason(row) for row in rows) if reason]
    # A masked row's reward is a placeholder (the Harbor server writes 0.0 beside `mask_sample: true`), so only
    # the counted rows feed the mean.
    counted = [row for row in rows if not mask_reason(row)]
    rewards = [reward for reward in (_finite_reward_of(row) for row in counted) if reward is not None]
    errors = [err for err in (agent_error(row) for row in rows) if err]
    kinds = [kind for kind in (measured_failure_kind(row) for row in rows) if kind]
    masked = None
    if reasons:
        masked = reasons[0] if len(rows) == 1 else f"{len(reasons)}/{len(rows)} rows masked: {reasons[0]}"
    return TaskSide(
        rows=rows,
        reward=statistics.fmean(rewards) if rewards else None,
        masked=masked,
        agent_error=errors[0] if errors else None,
        reward_rows=len(rewards),
        failure_kind=kinds[0] if kinds else None,
    )
