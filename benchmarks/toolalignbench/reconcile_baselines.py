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
"""Reconcile selective ToolAlignBench reruns into complete Gym result artifacts.

Earlier four-model rollouts exposed provider-specific prompted-tool syntax that the
original parser did not execute. Affected trajectories cannot be repaired by re-verifying
their final text: missing a tool result changes subsequent model turns. This utility finds
every affected ``(_ng_task_index, _ng_rollout_index)`` pair from the original rollout,
requires one clean rerun for each pair, preserves all unaffected rows byte-for-byte at the
JSON-object level, and writes complete rollout plus materialized-input files for profiling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from responses_api_agents.toolalignbench_agent.xml_tool_calls import (
    extract_tool_calls,
    has_unparsed_tool_call_markup,
)


EXPECTED_TASKS = 128
EXPECTED_REPEATS = 5
EXPECTED_ROWS = EXPECTED_TASKS * EXPECTED_REPEATS
RECOVERY_SOURCES = frozenset({"xml_recovery", "harmony_open"})
INPUT_FIELDS = (
    "responses_create_params",
    "id",
    "domain",
    "scenario_type",
    "prompt_condition",
    "tool_names",
    "remaining_documents",
    "agent_ref",
    "_ng_task_index",
    "_ng_rollout_index",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assistant_texts(row: dict[str, Any]) -> Iterable[str]:
    for item in (row.get("response") or {}).get("output") or []:
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue
        yield "".join(
            part.get("text") or "" for part in item.get("content") or [] if part.get("type") == "output_text"
        )


def rerun_reasons(row: dict[str, Any]) -> set[str]:
    """Return every reason this trajectory must be regenerated, not reverified."""
    reasons = set()
    if row.get("mask_sample"):
        reasons.add("masked_sample")
    if row.get("num_unparsed_tool_call_replies"):
        reasons.add("unparsed_tool_call")
    for text in _assistant_texts(row):
        calls = extract_tool_calls(text)
        if any(call.source in RECOVERY_SOURCES for call in calls):
            reasons.add("parser_recovery_changes_execution")
        if has_unparsed_tool_call_markup(text):
            reasons.add("unparsed_tool_call")

    response = row.get("response") or {}
    if row.get("failure_kind") or row.get("failure_reason"):
        reasons.add("prior_failure_reason")
    if row.get("episode_timed_out"):
        reasons.add("episode_timed_out")
    if row.get("model_incomplete") or response.get("status") == "incomplete" or response.get("incomplete_details"):
        reasons.add("model_incomplete")
    if row.get("num_documents") != row.get("num_documents_completed"):
        reasons.add("incomplete_episode")
    return reasons


def _key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["_ng_task_index"]), int(row["_ng_rollout_index"])


def _replacement_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["rerun_target_task_index"]), int(row["rerun_target_rollout_index"])


def _validate_clean_replacement(row: dict[str, Any], path: Path) -> None:
    response = row.get("response") or {}
    problems = []
    if row.get("mask_sample"):
        problems.append("mask_sample")
    if row.get("failure_kind") or row.get("failure_reason"):
        problems.append("failure")
    if row.get("episode_timed_out"):
        problems.append("episode_timed_out")
    if row.get("model_incomplete") or response.get("status") != "completed" or response.get("incomplete_details"):
        problems.append("model_incomplete")
    if row.get("num_documents") != row.get("num_documents_completed"):
        problems.append("incomplete_episode")
    if row.get("num_unparsed_tool_call_replies"):
        problems.append("unparsed_tool_call")
    if problems:
        raise ValueError(f"{path}: replacement {_replacement_key(row)} is not scoreable: {sorted(set(problems))}")


def _restore_identity(row: dict[str, Any], key: tuple[int, int]) -> dict[str, Any]:
    task_index, rollout_index = key
    restored = dict(row)
    restored["_ng_task_index"] = task_index
    restored["_ng_rollout_index"] = rollout_index
    restored.pop("rerun_target_task_index", None)
    restored.pop("rerun_target_rollout_index", None)
    rollout_id = f"{task_index}-{rollout_index}"
    capture = restored.get("ng_model_call_capture")
    if isinstance(capture, dict):
        capture = dict(capture)
        capture["rollout_id"] = rollout_id
        restored["ng_model_call_capture"] = capture
    trajectory = restored.get("ng_trajectory")
    if isinstance(trajectory, dict):
        trajectory = dict(trajectory)
        trajectory["rollout_id"] = rollout_id
        trajectory["task_id"] = str(task_index)
        trajectory["turns"] = [
            {
                **turn,
                "task_id": str(task_index),
                "rollout_id": rollout_id,
            }
            if isinstance(turn, dict)
            else turn
            for turn in trajectory.get("turns") or []
        ]
        restored["ng_trajectory"] = trajectory
    return restored


def _validate_complete(rows: list[dict[str, Any]]) -> None:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    counts = Counter(row["_ng_task_index"] for row in rows)
    if len(counts) != EXPECTED_TASKS or set(counts.values()) != {EXPECTED_REPEATS}:
        raise ValueError(
            f"expected {EXPECTED_TASKS} tasks x {EXPECTED_REPEATS} repeats; got {Counter(counts.values())}"
        )
    keys = [_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate task/rollout identity")


def reconcile(
    original_path: Path,
    replacement_paths: Path | list[Path],
    output_dir: Path,
    lane: str,
) -> dict[str, Any]:
    originals = _read_jsonl(original_path)
    _validate_complete(originals)
    required = {_key(row): sorted(rerun_reasons(row)) for row in originals if rerun_reasons(row)}
    all_original_keys = {_key(row) for row in originals}

    if isinstance(replacement_paths, Path):
        replacement_paths = [replacement_paths]
    replacements = []
    replacement_by_key = {}
    for replacement_path in replacement_paths:
        for row in _read_jsonl(replacement_path):
            _validate_clean_replacement(row, replacement_path)
            key = _replacement_key(row)
            if key in replacement_by_key:
                raise ValueError(f"{replacement_path}: duplicate replacement {key}")
            replacement_by_key[key] = _restore_identity(row, key)
            replacements.append(row)

    replacement_keys = set(replacement_by_key)
    if replacement_keys not in (set(required), all_original_keys):
        expected_keys = all_original_keys if len(replacement_keys) > len(required) else set(required)
        missing = sorted(expected_keys - replacement_keys)
        extra = sorted(replacement_keys - expected_keys)
        raise ValueError(f"replacement key mismatch; missing={missing[:20]} extra={extra[:20]}")

    merged = [replacement_by_key.get(_key(row), row) for row in originals]
    merged.sort(key=_key)
    _validate_complete(merged)

    rollout_path = output_dir / f"{lane}_toolalignbench_rollouts.jsonl"
    inputs_path = output_dir / f"{lane}_toolalignbench_materialized_inputs.jsonl"
    _write_jsonl(rollout_path, merged)
    _write_jsonl(inputs_path, ({field: row[field] for field in INPUT_FIELDS} for row in merged))

    rewards = Counter(str(row["reward"]) for row in merged)
    models = sorted({(row.get("response") or {}).get("model") for row in merged})
    reasons = Counter(reason for value in required.values() for reason in value)
    summary = {
        "lane": lane,
        "models": models,
        "expected_rows": EXPECTED_ROWS,
        "scored_rows": len(merged),
        "reused_rows": len(merged) - len(replacements),
        "replaced_rows": len(replacements),
        "masked_rows": sum(bool(row.get("mask_sample")) for row in merged),
        "failure_rows": sum(bool(row.get("failure_kind") or row.get("failure_reason")) for row in merged),
        "unparsed_rows": sum(bool(row.get("num_unparsed_tool_call_replies")) for row in merged),
        "timed_out_rows": sum(bool(row.get("episode_timed_out")) for row in merged),
        "incomplete_rows": sum(bool(row.get("model_incomplete")) for row in merged),
        "unknown_tool_rows": sum(bool(row.get("num_unknown_tool_calls")) for row in merged),
        "reward_counts": dict(rewards),
        "mean_reward": sum(float(row["reward"]) for row in merged) / len(merged),
        "rerun_reason_counts": dict(reasons),
        "original": {"path": str(original_path), "sha256": _sha256(original_path)},
        "replacements": [
            {"path": str(replacement_path), "sha256": _sha256(replacement_path)}
            for replacement_path in replacement_paths
        ],
        "rollouts": {"path": str(rollout_path), "sha256": _sha256(rollout_path)},
        "materialized_inputs": {"path": str(inputs_path), "sha256": _sha256(inputs_path)},
    }
    reconciliation_path = output_dir / f"{lane}_toolalignbench_reconciliation.json"
    reconciliation_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--replacement",
        action="append",
        required=True,
        metavar="LANE=PATH",
        help="Complete selective-rerun JSONL for one lane; repeat for every lane.",
    )
    args = parser.parse_args()
    replacements: dict[str, list[Path]] = {}
    for item in args.replacement:
        lane, separator, path = item.partition("=")
        if not separator or not lane or not path:
            parser.error(f"invalid --replacement {item!r}; expected LANE=PATH")
        replacements.setdefault(lane, []).append(Path(path))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for lane, replacement_paths in sorted(replacements.items()):
        original_path = args.original_dir / f"{lane}_toolalignbench.jsonl"
        summaries[lane] = reconcile(original_path, replacement_paths, args.output_dir, lane)
    combined = args.output_dir / "toolalignbench_reconciliation.json"
    combined.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
