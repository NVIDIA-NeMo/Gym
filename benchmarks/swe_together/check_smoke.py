# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit a complete, one-repeat SWE-Together smoke from durable rollout records.

A valid zero is healthy. Missing grades, failed auxiliary measurements and absent
cleanup evidence fail qualification independently of the candidate's reward.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from environment_servers.swe_together.evaluation import PASS_THRESHOLD, derive_score
from environment_servers.swe_together.task import MANIFEST


def inspect_row(row: dict, *, asset_root: Path) -> list[str]:
    """Check measurement evidence without selecting an attempt based on its score."""
    errors = []
    task_id = row.get("task_id")
    score = row.get("judge_score")
    if row.get("mask_sample") or row.get("failure_kind"):
        errors.append("failed_or_masked")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
        errors.append("missing_judge_score")
    else:
        if row.get("reward") != float(score >= PASS_THRESHOLD):
            errors.append("reward_mismatch")
        try:
            rubric = json.loads((asset_root / task_id / "canonical_goals.json").read_text())
            if derive_score(rubric, row["verdict"])["judge_score"] != score:
                errors.append("frozen_score_mismatch")
        except (ValueError, KeyError, TypeError, OSError):
            errors.append("missing_frozen_goal_evidence")
    turns = row.get("ng_activations") or []
    if not turns or [t.get("activation_id") for t in turns] != list(range(1, len(turns) + 1)):
        errors.append("invalid_activation_sequence")
    sessions = set()
    for turn in turns:
        response = turn.get("response") or {}
        native_id = (response.get("metadata") or {}).get("harness_session_id")
        if not isinstance(native_id, str) or not native_id:
            errors.append("native_conversation_not_continuous")
        else:
            sessions.add(native_id)
        records = (response.get("_ng_agent_observations") or {}).get("records") or []
        roots = [r for r in records if r.get("kind") == "agent_invocation" and not r.get("parent_invocation_id")]
        if not roots or any(r.get("activation_id") != turn["activation_id"] for r in roots):
            errors.append("missing_activation_observations")
    if len(sessions) != 1:
        errors.append("native_conversation_not_continuous")
    close = row.get("ng_agent_close") or {}
    if close.get("cleanup_confirmed") is not True:
        errors.append("unconfirmed_candidate_close")
    judge_close = row.get("judge_close") or {}
    if (
        not judge_close.get("agent_session_id")
        or not (judge_close.get("agent_observations") or {}).get("source")
        or judge_close.get("cleanup_confirmed") is not True
        or row.get("judge_close_error")
    ):
        errors.append("unconfirmed_judge_close")
    for role in ("candidate_sandbox", "judge_sandbox"):
        if ((row.get("cleanup_receipts") or {}).get(role) or {}).get("cleanup_confirmed") is not True:
            errors.append("unconfirmed_" + role)
    submission = row.get("submission") or {}
    if not submission.get("archive_sha256") or submission.get("transfer") != "shared_storage":
        errors.append("missing_submission_receipt")
    for key in ("tagger_error", "coverage_error", "interaction_metrics_error"):
        if row.get(key):
            errors.append(key)
    coverage = row.get("intent_coverage") or {}
    for key in ("coverage_rate", "weighted_coverage", "scope_precision", "overall_score"):
        value = coverage.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            errors.append("invalid_intent_coverage")
    correction = row.get("user_correction")
    if correction is not None and (
        not isinstance(correction, (int, float))
        or isinstance(correction, bool)
        or not math.isfinite(correction)
        or correction < 0
    ):
        errors.append("invalid_user_correction")
    if row.get("n_simulator_messages", 0) and row.get("user_correction") is None:
        errors.append("missing_user_correction")
    return sorted(set(errors))


def audit(path: Path, *, asset_root: Path, expected: list[str]) -> dict:
    """Stream one canonical run and require each expected task exactly once."""
    counts = Counter()
    failures = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            task = row.get("task_id")
            counts[task] += 1
            errors = inspect_row(row, asset_root=asset_root)
            if errors:
                failures[task] = errors
    missing = sorted(set(expected) - counts.keys())
    unexpected = sorted(counts.keys() - set(expected))
    duplicates = sorted(task for task, count in counts.items() if count != 1)
    return {
        "passed": not (failures or missing or unexpected or duplicates),
        "planned": len(expected),
        "returned": sum(counts.values()),
        "healthy": sum(task in expected and task not in failures and count == 1 for task, count in counts.items()),
        "missing": missing,
        "unexpected": unexpected,
        "duplicates": duplicates,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    expected = [r["task_id"] for r in json.loads(MANIFEST.read_text())["tasks"]]
    result = audit(args.input, asset_root=args.asset_root, expected=expected)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
