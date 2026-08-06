#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize a finance_agent_v2 run: scores with error bars, tool usage, tool errors.

A run's aggregate metrics are bare means over rollouts, which cannot separate two
models on 27 questions. This adds:

* Bootstrap CIs that resample *questions*, averaging repeats first. Repeats of one
  question are not independent draws, so resampling rollouts would treat 3 repeats
  of 27 questions as 81 samples and report a CI about sqrt(3) too narrow.
* Mean within-question spread across repeats: how much of a gap is just noise.
* Tool calls per question, and the error strings tools hand back to the model.

Tool errors are counted, never fixed: the error text is part of the observation the
agent is scored on, so changing it would void comparisons against Vals's leaderboard.
`^` gets its own counter because simpleeval reads it as XOR, so `2^10` returns 8.

Usage:
    python report_run.py RUN.jsonl [RUN2.jsonl ...] [--per-question]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence


_BOOTSTRAP_SAMPLES = 5000
# Fixed so a report is reproducible; the CI should not move when you rerun it.
_BOOTSTRAP_SEED = 20260805

_CALCULATOR_ERRORS = {
    "Error: invalid expression": "invalid expression (parse/type error)",
    "Error: division by zero": "division by zero",
    "Error: numerical overflow": "numerical overflow",
    "Error: expression must not be empty": "empty expression",
}

# The six tools Vals's harness counts; `submit_final_result` is the termination
# signal, so including it would put every number exactly 1.0 above the comparable one.
_VALS_TOOLS = (
    "edgar_search",
    "web_search",
    "parse_html_page",
    "retrieve_information",
    "calculator",
    "price_history",
)


def load(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def question_of(row: Dict[str, Any], index: int) -> str:
    """Group key for repeats of the same question.

    Prefers the harness task id. Question text is the fallback for hand-assembled
    files, whitespace-normalized because upstream reflowed 3 of the 27 public
    questions in Aug 2026 without rewording them.
    """
    task_index = row.get("_ng_task_index")
    if task_index is not None:
        return f"task_{task_index}"
    for message in row.get("responses_create_params", {}).get("input") or []:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return " ".join(message["content"].split())
    return f"__row_{index}"


def partial_credit_of(row: Dict[str, Any]) -> float:
    """Partial Credit, falling back to `reward` for runs that predate the field."""
    value = row.get("rubric_partial_credit")
    return float(value if value is not None else (row.get("reward") or 0.0))


def _tool_calls(row: Dict[str, Any]) -> tuple[Counter, Counter, Counter]:
    """Return (calls per tool, error counts, notable-expression counts) for one rollout."""
    calls: Counter = Counter()
    errors: Counter = Counter()
    notable: Counter = Counter()

    outputs = row.get("response", {}).get("output") or []
    name_by_call_id: Dict[str, str] = {}
    caret_call_ids = set()
    for item in outputs:
        if item.get("type") != "function_call":
            continue
        name = item.get("name") or "unknown"
        calls[name] += 1
        name_by_call_id[item.get("call_id")] = name
        if name == "calculator":
            try:
                expression = json.loads(item.get("arguments") or "{}").get("expression", "")
            except (json.JSONDecodeError, TypeError):
                expression = ""
            if "^" in expression:
                caret_call_ids.add(item.get("call_id"))

    for item in outputs:
        if item.get("type") != "function_call_output":
            continue
        call_id = item.get("call_id")
        tool = name_by_call_id.get(call_id, "unknown")
        text, raised = _tool_reply_text(item.get("output"))
        failed = raised or text.startswith("Error:")
        if raised:
            # The agent loop's own wrapper for a tool that threw.
            errors[f"{tool}: tool call raised"] += 1
        elif failed:
            matched = next((label for prefix, label in _CALCULATOR_ERRORS.items() if text.startswith(prefix)), None)
            errors[f"{tool}: {matched or 'other Error: reply'}"] += 1
        if call_id in caret_call_ids:
            caret_call_ids.discard(call_id)
            # An error at least tells the model to retry; a returned value means
            # simpleeval XOR'd two integers into a wrong number nothing can flag.
            notable["'^' in a calculator expression -> error"] += int(failed)
            notable["'^' in a calculator expression -> RETURNED A VALUE (silent XOR)"] += int(not failed)
    return calls, errors, notable


def _tool_reply_text(output: Any) -> tuple[str, bool]:
    """Unwrap what the tool said, plus whether the call raised.

    The server wraps replies as `{"results": ...}`; the agent loop emits
    `{"error": ...}` when a tool throws.
    """
    if not isinstance(output, str):
        return "", False
    try:
        payload = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return output, False
    if not isinstance(payload, dict):
        return output, False
    if "error" in payload:
        return str(payload.get("error") or ""), True
    return str(payload.get("results") or ""), False


def bootstrap_ci(clusters: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    """Percentile CI for the mean, resampling whole questions (clusters)."""
    if len(clusters) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(_BOOTSTRAP_SEED)
    n = len(clusters)
    means = []
    for _ in range(_BOOTSTRAP_SAMPLES):
        sample = [clusters[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    return (
        means[int(tail * _BOOTSTRAP_SAMPLES)],
        means[min(int((1 - tail) * _BOOTSTRAP_SAMPLES), _BOOTSTRAP_SAMPLES - 1)],
    )


def report(path: Path, show_per_question: bool) -> None:
    rows = load(path)
    if not rows:
        print(f"{path.name}: empty\n")
        return

    by_question: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_question[question_of(row, i)].append(row)

    ungrouped = sum(1 for key in by_question if key.startswith("__row_"))
    repeats = [len(v) for v in by_question.values()]

    print("=" * 100)
    print(f"{path.name}")
    print("=" * 100)
    print(
        f"rollouts: {len(rows)}   questions: {len(by_question)}   repeats/question: min={min(repeats)} max={max(repeats)}"
    )
    if min(repeats) != max(repeats):
        short = sum(1 for r in repeats if r < max(repeats))
        print(
            f"  NOTE: uneven repeats — {short} question(s) have fewer than {max(repeats)}. A run that "
            "stopped early is\n        biased toward whichever questions finished, usually the fast ones."
        )
    if ungrouped:
        print(f"  NOTE: {ungrouped} row(s) had no task index or question text and are treated as separate questions.")
    if len(by_question) < 5:
        print("  NOTE: too few questions for a meaningful confidence interval.")

    # --- scores -------------------------------------------------------------
    print("\nScores (mean over questions; repeats averaged first)")
    metrics = {
        "partial credit (reward)": lambda r: partial_credit_of(r),
        "all-pass": lambda r: 1.0 if r.get("rubric_all_pass") else 0.0,
        "unweighted fraction": lambda r: float(r.get("rubric_fraction") or 0.0),
    }
    for label, fn in metrics.items():
        per_question = [statistics.fmean([fn(r) for r in rs]) for rs in by_question.values()]
        mean = statistics.fmean(per_question)
        lo, hi = bootstrap_ci(per_question)
        # Spread between repeats of one question, i.e. how much rerunning alone moves a score.
        spreads = [statistics.stdev([fn(r) for r in rs]) for rs in by_question.values() if len(rs) > 1]
        spread = f"   repeat sd {statistics.fmean(spreads):.3f}" if spreads else ""
        ci = "" if lo != lo else f"  95% CI [{lo:.3f}, {hi:.3f}]"
        print(f"  {label:24s} {mean:.3f}{ci}{spread}")

    scored = [r for r in rows if r.get("rubric_judgements")]
    no_submission = len(rows) - len(scored)
    judge_errors = sum(1 for r in rows if r.get("judge_error"))
    tripped = sum(1 for r in scored if r.get("rubric_dealbreakers_failed"))
    print(f"\n  rollouts with no submission: {no_submission}   with a judge error: {judge_errors}")
    if any(r.get("rubric_dealbreakers_total") for r in scored):
        forfeited = sum(
            1 for r in scored if r.get("rubric_dealbreakers_failed") and (r.get("rubric_weighted_fraction") or 0) > 0
        )
        print(f"  dealbreaker tripped: {tripped}/{len(scored)} scored rollouts ({forfeited} forfeited earned credit)")
    else:
        print("  dealbreakers: none recorded (run predates weighted scoring — rescore with rescore_rubrics.py)")

    # --- why trajectories ended --------------------------------------------
    stop_reasons = Counter(
        (r.get("response", {}).get("metadata") or {}).get("stop_reason", "(not recorded)") for r in rows
    )
    print("\nStop reasons")
    for reason, count in stop_reasons.most_common():
        print(f"  {reason:24s} {count:5d}")

    # --- tools --------------------------------------------------------------
    calls_total: Counter = Counter()
    errors_total: Counter = Counter()
    notable_total: Counter = Counter()
    for row in rows:
        calls, errors, notable = _tool_calls(row)
        calls_total += calls
        errors_total += errors
        notable_total += notable

    n_rollouts = len(rows)
    # Per question, averaging repeats first, over the six tools Vals counts.
    per_question_calls = [
        statistics.fmean([sum(v for k, v in _tool_calls(r)[0].items() if k in _VALS_TOOLS) for r in rs])
        for rs in by_question.values()
    ]
    turns = [sum(1 for o in r.get("response", {}).get("output") or [] if o.get("type") == "reasoning") for r in rows]
    graded = sum(count for tool, count in calls_total.items() if tool in _VALS_TOOLS)

    lo, hi = bootstrap_ci(per_question_calls)
    ci = "" if lo != lo else f"  95% CI [{lo:.1f}, {hi:.1f}]"
    print("\nTool calls per question (six Vals tools; excludes submit_final_result)")
    print(
        f"  mean {statistics.fmean(per_question_calls):.2f}{ci}   range {min(per_question_calls):.1f}-{max(per_question_calls):.1f}"
    )
    if sum(turns):
        mean_turns = statistics.fmean(turns)
        print(
            f"  agent turns per question {mean_turns:.2f}, so {graded / n_rollouts / mean_turns:.2f} calls per turn (parallel fan-out)"
        )

    # Calls per question is the figure Vals plots, so it leads; share is a within-run
    # profile only (two models at different call volumes can show identical shares).
    print(f"\n  {'tool':24s} {'per question':>13s}  {'share':>7s}  {'total':>7s}")
    for tool, count in calls_total.most_common():
        share = f"{100 * count / graded:.1f}%" if tool in _VALS_TOOLS else "-"
        print(f"  {tool:24s} {count / n_rollouts:13.2f}  {share:>7s}  {count:7d}")

    print(f"\nTool errors returned to the model (total {sum(errors_total.values())})")
    if errors_total:
        for label, count in errors_total.most_common():
            print(f"  {label:60s} {count:5d}   {count / n_rollouts:5.2f} per rollout")
    else:
        print("  none")
    for label, count in sorted(notable_total.items()):
        if count:
            print(f"  {label:60s} {count:5d}")

    # --- per question -------------------------------------------------------
    if show_per_question:
        print("\nPer question (partial credit by repeat)")
        header = f"  {'question':10s} {'mean':>6s} {'repeats':>28s}  {'tools':>6s}  stop reasons"
        print(header)
        for key, rs in sorted(
            by_question.items(), key=lambda kv: statistics.fmean([partial_credit_of(r) for r in kv[1]])
        ):
            values = [partial_credit_of(r) for r in rs]
            tools = sum(sum(_tool_calls(r)[0].values()) for r in rs) / len(rs)
            reasons = ",".join(
                sorted({(r.get("response", {}).get("metadata") or {}).get("stop_reason", "?") for r in rs})
            )
            shown = " ".join(f"{v:5.2f}" for v in values)
            print(f"  {key[:10]:10s} {statistics.fmean(values):6.3f} {shown:>28s}  {tools:6.1f}  {reasons}")
    print()


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="rollout JSONL file(s)")
    parser.add_argument("--per-question", action="store_true", help="print the per-question breakdown")
    args = parser.parse_args(argv)

    for run in args.runs:
        report(run, args.per_question)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
