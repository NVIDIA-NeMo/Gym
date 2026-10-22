# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conservative action heuristics; no model judge and no transcript upload."""

from __future__ import annotations

from collections import Counter

from .common import finish, ratio
from .models import AnalysisResult, Options, Rollout, Tool


def category(tool: Tool) -> str:
    action = tool.action or tool.name.lower()
    if action in ("read", "explore", "read_file", "grep", "search", "list_directory"):
        return "exploration"
    if action in ("write", "implement", "write_file", "edit_file", "apply_patch"):
        return "implementation"
    if action in ("verify", "test", "run_tests", "lint"):
        return "verification"
    return "unknown"


def turn_classification(rollouts: list[Rollout], options: Options) -> AnalysisResult:
    counts, rows = Counter(), []
    for r in rollouts:
        if r.eligible:
            for tool in r.tools:
                phase = category(tool)
                counts[phase] += 1
                rows.append(
                    [
                        r.task_id,
                        r.repeat,
                        tool.id,
                        tool.invocation_id,
                        tool.name,
                        phase,
                        tool.status,
                    ]
                )
    known = sum(value for key, value in counts.items() if key != "unknown")
    return finish(
        "turn_classification",
        "Tool-action phase heuristics: exploration, implementation, verification",
        rollouts,
        rows=rows,
        columns=["task_id", "repeat", "tool_id", "invocation_id", "tool_name", "phase", "status"],
        metrics={f"{key}_count": counts[key] for key in ("exploration", "implementation", "verification", "unknown")},
        used=known,
        options=options,
        extra_coverage={"observations_missing": counts["unknown"]},
        reason="Need recognized tool names or explicit action labels",
        notes=[
            "Classification unit is a tool action; a model turn can contain several actions.",
            "Shell commands and opaque tools remain unknown unless the producer supplies an action label.",
        ],
    )


def wasted_turns(rollouts: list[Rollout], options: Options) -> AnalysisResult:
    rows = []
    measured = missing = failures = 0
    for r in rollouts:
        if r.eligible:
            seen = {}
            for tool in sorted(r.tools, key=lambda t: (t.index is None, t.index or 0, t.id)):
                failed = tool.status in ("failed", "error", "timeout", "cancelled")
                failures += failed
                observable = (
                    tool.invocation_id is not None
                    and tool.index is not None
                    and tool.target is not None
                    and tool.content_hash is not None
                    and tool.action in ("read", "write")
                    and tool.status == "completed"
                )
                if observable:
                    measured += 1
                    key = tool.invocation_id, tool.target
                    previous = seen.get(key)
                    flag = None
                    if previous is not None and previous[1] == tool.content_hash:
                        if tool.action == "read":
                            flag = (
                                "read_after_write_same_content"
                                if previous[0] == "write"
                                else "repeated_read_same_content"
                            )
                        elif previous[0] == "write":
                            flag = "repeated_write_same_content"
                    seen[key] = tool.action, tool.content_hash
                    if flag:
                        # Avoid propagating paths, hashes, commands or file contents into results.
                        rows.append([r.task_id, r.repeat, tool.id, flag])
                else:
                    missing += 1
    return finish(
        "wasted_turns",
        "Potential action reuse backed by explicit ordering and observed content hashes",
        rollouts,
        rows=rows,
        columns=["task_id", "repeat", "tool_id", "flag"],
        metrics={
            "flagged_actions": len(rows),
            "flagged_fraction_of_observable_actions": ratio(len(rows), measured),
            "failed_actions": failures,
        },
        used=measured,
        options=options,
        extra_coverage={"observations_missing": missing},
        reason="Need ordered read/write actions with invocation, target, completed status and content hash",
        notes=[
            "Flags are candidates for inspection, not proof of wasted reasoning or unnecessary actions.",
            "Failed actions are counted separately. No action redundancy is inferred from a failure alone.",
            "Requires distinct per-invocation action indices; each rollout resets the state.",
        ],
    )
