# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-checks between the timeline and numbers Gym and the harnesses record on their own.

Each check compares the spans the analysis will read against an independent record of the same quantity: Harbor's
own timers and usage list, Gym's capture aggregates and perf summary, OpenCode's export, the verifier's timings,
the engine's durations. They run per rollout, in seconds, and produce the trust line on the report page.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import orjson

from ng_latency.paths import locate, metrics_dir


TOLERANCE_S = 5e-3  # same-clock durations read by two timers around the same await; 1.25 ms seen at 48 rollouts
WALL_TOLERANCE_S = 0.05  # a monotonic duration against a wall-clock difference of the same interval


@dataclass
class CheckResult:
    name: str
    passed: bool
    samples: int = 0
    max_abs_diff: Optional[float] = None
    details: list[str] = field(default_factory=list)
    warning: bool = False


class _Check:
    def __init__(self, name: str) -> None:
        self.name = name
        self.samples = 0
        self.max_abs_diff: Optional[float] = None
        self.details: list[str] = []

    def compare(self, expected: Any, actual: Any, *, tolerance: float = 0.0, label: str = "") -> None:
        self.samples += 1
        if expected is None or actual is None:
            if expected is not actual:
                self.details.append(f"{label}: expected={expected!r} actual={actual!r}")
            return
        diff = abs(float(expected) - float(actual))
        self.max_abs_diff = diff if self.max_abs_diff is None else max(self.max_abs_diff, diff)
        if diff > tolerance:
            self.details.append(f"{label}: expected={expected!r} actual={actual!r} diff={diff:.6g}")

    def require(self, condition: bool, label: str) -> None:
        self.samples += 1
        if not condition:
            self.details.append(label)

    def result(self) -> CheckResult:
        return CheckResult(
            self.name,
            passed=not self.details,
            samples=self.samples,
            max_abs_diff=self.max_abs_diff,
            details=self.details[:20],
        )


def _duration_s(span: dict[str, Any]) -> Optional[float]:
    if span.get("duration_ns") is not None:
        return span["duration_ns"] / 1e9
    if span.get("end_wall_s") is not None:
        return span["end_wall_s"] - span["start_wall_s"]
    return None


def _by_name(spans: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for span in spans:
        grouped[span["name"]].append(span)
    for group in grouped.values():
        group.sort(key=lambda s: (s["start_wall_s"], s["span_id"]))
    return grouped


def _is_sandbox(span: dict[str, Any]) -> bool:
    return str(span.get("clock", {}).get("domain", "")).startswith("sandbox:")


def _check_terminus(
    record: dict[str, Any], by_name: dict[str, list[dict[str, Any]]], by_id: dict[str, dict[str, Any]]
) -> list[CheckResult]:
    results: list[CheckResult] = []
    calls = [c for c in by_name.get("model_call", []) if not _is_sandbox(c)]
    # Harbor appends to model_call_times after its attempt loop: every call that returned or timed out, not one
    # that raised out of the loop (cancelled, transport error).
    timed = [c for c in calls if c.get("status") in ("ok", "incomplete", "timeout")]
    if isinstance(record.get("model_call_times"), list):
        check = _Check("terminus_model_call_times")
        times = record["model_call_times"]
        check.require(len(times) == len(timed), f"count: model_call_times={len(times)} spans={len(timed)}")
        for i, (expected, span) in enumerate(zip(times, timed)):
            check.compare(expected, _duration_s(span), tolerance=TOLERANCE_S, label=f"call[{i}]")
        results.append(check.result())

    # Harbor's batch timer runs from the first command to the screen read after the last; the batch span spans
    # the commands and the next-iteration mark bounds the read, so Harbor's value lies between the two. Harbor
    # runs _execute_commands for every decision, with or without commands: a decision without commands has an
    # exec time (the screen read alone) and no batch span, so the two lists are walked in decision order.
    if isinstance(record.get("command_exec_times"), list):
        check = _Check("terminus_command_exec_times")
        batches = [b for b in by_name.get("tool_batch", []) if b.get("status") != "cancelled"]
        exec_times = record["command_exec_times"]
        decisions = [
            c
            for c in calls
            if (c.get("attrs") or {}).get("purpose") == "agent_step"
            and c.get("status") == "ok"
            and (c.get("attrs") or {}).get("parse_error") is not True  # after a parse error Harbor runs no batch
        ]
        empty = sum(1 for c in decisions if not (c.get("attrs") or {}).get("parsed_commands"))
        check.require(
            len(exec_times) == len(batches) + empty,
            f"count: command_exec_times={len(exec_times)} batches={len(batches)} empty decisions={empty}",
        )
        next_batch = 0
        for i, (expected, decision) in enumerate(zip(exec_times, decisions)):
            if not (decision.get("attrs") or {}).get("parsed_commands"):
                continue  # the screen read alone: nothing of ours to compare it with
            if next_batch >= len(batches):
                break
            batch = batches[next_batch]
            next_batch += 1
            span_s = _duration_s(batch)
            received = (batch.get("attrs") or {}).get("response_received_wall_s")
            upper = float(received) - batch["start_wall_s"] if received is not None else None
            if span_s is None:
                check.require(False, f"decision[{i}]: open batch span")
                continue
            check.require(expected >= span_s - TOLERANCE_S, f"decision[{i}]: harbor {expected} < commands {span_s}")
            if upper is not None:
                check.require(
                    expected <= upper + TOLERANCE_S, f"decision[{i}]: harbor {expected} > until next mark {upper}"
                )
        results.append(check.result())

    if record.get("terminus2_time_taken") is not None:
        check = _Check("terminus_time_taken")
        execute = by_name.get("agent_execute", [])
        agent = by_name.get("agent", [])
        check.require(len(execute) == 1, f"agent_execute spans={len(execute)}")
        if execute:
            check.compare(
                record["terminus2_time_taken"], _duration_s(execute[0]), tolerance=TOLERANCE_S, label="agent_execute"
            )
            if agent:
                inside = (
                    agent[0]["start_wall_s"] >= execute[0]["start_wall_s"] - TOLERANCE_S
                    and (agent[0].get("end_wall_s") or 0) <= (execute[0].get("end_wall_s") or 0) + TOLERANCE_S
                )
                check.require(inside, "agent not inside agent_execute")
        results.append(check.result())

    # Harbor's usages list has one entry per HTTP response it received, in order; the server span of each
    # answered call carries the same usage from the capture.
    if isinstance(record.get("usages"), list):
        check = _Check("terminus_usages")
        answered = [
            c for c in calls if c.get("status") in ("ok", "incomplete") and c["attrs"].get("server_span_id") in by_id
        ]
        usages = [u for u in record["usages"] if isinstance(u, dict)]
        check.require(len(usages) == len(answered), f"count: usages={len(usages)} answered calls={len(answered)}")
        for i, (usage, call) in enumerate(zip(usages, answered)):
            tokens = by_id[call["attrs"]["server_span_id"]]["attrs"].get("tokens") or {}
            check.compare(usage.get("input_tokens"), tokens.get("input"), label=f"call[{i}].input")
            check.compare(
                usage.get("output_tokens"), tokens.get("completion_total"), label=f"call[{i}].completion_total"
            )
            reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
            check.compare(reasoning, tokens.get("reasoning"), label=f"call[{i}].reasoning")
        results.append(check.result())

    if record.get("num_compactions") is not None:
        check = _Check("terminus_compaction_counts")
        overflow = sum(1 for c in calls if c["attrs"].get("context_overflow") is True)
        proactive_done = sum(
            1
            for c in by_name.get("compaction", [])
            if c["attrs"].get("reason") == "proactive" and c["attrs"].get("outcome") == "completed"
        )
        timeouts = sum(1 for a in by_name.get("model_call_attempt", []) if a.get("status") == "timeout")
        check.compare(record["num_compactions"], overflow, label="num_compactions")
        check.compare(record.get("num_proactive_compactions"), proactive_done, label="num_proactive_compactions")
        check.compare(record.get("model_calls_gt_10min"), timeouts, label="model_calls_gt_10min")
        results.append(check.result())
    return results


def _check_gym(record: dict[str, Any], by_name: dict[str, list[dict[str, Any]]]) -> list[CheckResult]:
    results: list[CheckResult] = []
    servers = by_name.get("model_call_server", [])

    # Gym aggregates the capture records on its own; the server spans must add up to the same numbers.
    capture = record.get("ng_model_call_capture")
    metrics = capture.get("metrics") if isinstance(capture, dict) else None
    if isinstance(metrics, dict):
        check = _Check("capture_metrics_sums")

        def _sum(key: str) -> Optional[int]:
            values = [
                s["attrs"].get("tokens", {}).get(key)
                for s in servers
                if s["attrs"].get("tokens", {}).get(key) is not None
            ]
            return sum(values) if values else None

        check.compare(metrics.get("num_calls"), len(servers), label="num_calls")
        check.compare(metrics.get("tokens_in"), _sum("input"), label="tokens_in")
        check.compare(metrics.get("tokens_out"), _sum("completion_total"), label="tokens_out")
        check.compare(metrics.get("tokens_reasoning"), _sum("reasoning"), label="tokens_reasoning")
        latencies = [s["duration_ns"] / 1e6 for s in servers if s.get("duration_ns") is not None]
        check.compare(
            metrics.get("latency_total_ms"),
            sum(latencies) if latencies else None,
            tolerance=0.01 * max(1, len(latencies)),
            label="latency_total_ms",
        )
        results.append(check.result())

    # The rollout span's monotonic duration against Gym's wall-clock latency of the same call.
    perf = record.get("ng_perf")
    rollout = by_name.get("rollout", [])
    if isinstance(perf, dict) and rollout:
        check = _Check("perf_total_latency")
        duration_s = _duration_s(rollout[0])
        check.compare(
            perf.get("total_latency_ms"),
            duration_s * 1000.0 if duration_s is not None else None,
            tolerance=WALL_TOLERANCE_S * 1000.0,
            label="total_latency_ms vs rollout duration",
        )
        results.append(check.result())
        if "opencode_finished" in record:
            check = _Check("perf_tool_calls")
            check.compare(perf.get("num_tool_calls"), len(by_name.get("tool_call", [])), label="num_tool_calls")
            results.append(check.result())
    return results


def _check_opencode(record: dict[str, Any], by_name: dict[str, list[dict[str, Any]]]) -> list[CheckResult]:
    if "opencode_finished" not in record:
        return []
    check = _Check("opencode_token_accounting")
    response = record.get("response")
    usage = response.get("usage") if isinstance(response, dict) else None
    servers = by_name.get("model_call_server", [])
    main_servers = [s for s in servers if not (s.get("ids") or {}).get("parent_invocation_id")]
    if usage and main_servers:
        # OpenCode's export sums its own input counts, which exclude cache reads; the captured prompt tokens
        # include them. The two sources are independent (OpenCode's database vs the model server's capture).
        cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
        usage_input = usage.get("input_tokens")
        check.compare(
            usage_input + cached if usage_input is not None else None,
            sum((s["attrs"].get("tokens") or {}).get("input") or 0 for s in main_servers),
            label="usage.input+cached vs capture",
        )
    for server in servers:
        tokens = server["attrs"].get("tokens") or {}
        if tokens.get("cached") is not None and tokens.get("input") is not None:
            check.require(tokens["cached"] <= tokens["input"], f"cached {tokens['cached']} > input {tokens['input']}")
    return [check.result()]


def _check_engine_and_verify(
    record: dict[str, Any], by_name: dict[str, list[dict[str, Any]]], by_id: dict[str, dict[str, Any]]
) -> list[CheckResult]:
    results: list[CheckResult] = []
    servers = by_name.get("model_call_server", [])
    if servers:
        check = _Check("engine_within_server_within_call")
        for server in servers:
            server_s = _duration_s(server)
            e2e = (server["attrs"].get("vllm") or {}).get("e2e_s")
            if e2e is not None and server_s is not None:
                check.require(e2e <= server_s + TOLERANCE_S, f"engine e2e {e2e} > server {server_s}")
            parent = by_id.get(server.get("parent_span_id") or "")
            if parent is not None and parent["name"] == "model_call" and not _is_sandbox(parent):
                harness_s = _duration_s(parent)
                if harness_s is not None and server_s is not None:
                    check.require(server_s <= harness_s + TOLERANCE_S, f"server {server_s} > harness call {harness_s}")
        results.append(check.result())

    verify = by_name.get("verify", [])
    verifier_total = None
    if record.get("verification_time_taken") is not None:
        verifier_total = record["verification_time_taken"]
    elif record.get("patch_verification_time_taken") is not None:
        verifier_total = (record.get("eval_sandbox_start_time_taken") or 0) + record["patch_verification_time_taken"]
    if verify and verifier_total is not None:
        check = _Check("verify_containment")
        span_s = _duration_s(verify[0])
        check.require(
            span_s is not None and verifier_total <= span_s + TOLERANCE_S,
            f"verifier {verifier_total} > verify span {span_s}",
        )
        results.append(check.result())
    return results


_TERMINAL_STATUSES = ("ok", "error", "timeout", "cancelled", "incomplete")


def _check_integrity(
    rollout_id: str, spans: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]
) -> list[CheckResult]:
    """Event integrity of the raw timeline: every closed span has start <= end, span ids are unique within the
    rollout, parent references resolve inside it and every span names it, model calls and their request attempts
    carry a completion, error, timeout, cancellation or incomplete status (never unknown or open), a tool call is
    closed unless its status says why not, and a child timed on its parent's clock lies inside it (an attempt
    inside its logical call, a Terminus command inside its batch)."""
    check = _Check("span_integrity")
    for span_id, uses in sorted(Counter(s["span_id"] for s in spans).items()):
        check.require(uses == 1, f"span id {span_id} used {uses} times")
    for span in spans:
        label = f"{span['name']} {span['span_id']}"
        start, end = span.get("start_wall_s"), span.get("end_wall_s")
        check.require(start is not None, f"{label}: no start")
        if start is not None and end is not None:
            check.require(end >= start, f"{label}: end {end} before start {start}")
        parent_id = span.get("parent_span_id")
        parent = by_id.get(parent_id) if parent_id else None
        check.require(parent_id is None or parent is not None, f"{label}: parent {parent_id} not in this rollout")
        span_rollout = (span.get("ids") or {}).get("rollout_id")
        check.require(span_rollout == rollout_id, f"{label}: rollout id {span_rollout} is not {rollout_id}")
        status = span.get("status")
        if span["name"] in ("model_call", "model_call_attempt"):
            check.require(status in _TERMINAL_STATUSES, f"{label}: status {status}")
            check.require(end is not None or status != "ok", f"{label}: open with status ok")
        if span["name"] == "tool_call":
            check.require(end is not None or status in _TERMINAL_STATUSES[1:], f"{label}: open with status {status}")
        if (
            parent is not None
            and start is not None
            and end is not None
            and parent.get("end_wall_s") is not None
            and (span["name"], parent["name"]) in (("model_call_attempt", "model_call"), ("tool_call", "tool_batch"))
            and span.get("clock", {}).get("domain") == parent.get("clock", {}).get("domain")
        ):
            inside = start >= parent["start_wall_s"] - TOLERANCE_S and end <= parent["end_wall_s"] + TOLERANCE_S
            check.require(inside, f"{label}: outside its {parent['name']}")
    return [check.result()]


def _check_lifecycle(by_name: dict[str, list[dict[str, Any]]]) -> list[CheckResult]:
    check = _Check("agent_span_count")
    agents = by_name.get("agent", [])
    check.require(len(agents) <= 1, f"{len(agents)} agent root spans in one rollout")
    return [check.result()]


def check_record(record: dict[str, Any], timeline: dict[str, Any]) -> list[CheckResult]:
    spans = timeline.get("spans") or []
    by_name = _by_name(spans)
    by_id = {s["span_id"]: s for s in spans}
    results = _check_lifecycle(by_name)
    results += _check_integrity(timeline.get("rollout_id") or "", spans, by_id)
    results += _check_terminus(record, by_name, by_id)
    results += _check_gym(record, by_name)
    results += _check_opencode(record, by_name)
    results += _check_engine_and_verify(record, by_name, by_id)
    return results


def check_coverage(counts: dict[str, Any], *, min_coverage: float = 0.99) -> list[CheckResult]:
    """Coverage ratios; below the threshold is a warning, never a failure."""
    denominator = counts.get("model_calls") or 0
    if not denominator:
        return [CheckResult("server_coverage", passed=True, samples=0, details=["no samples"])]
    ratio = (counts.get("model_calls_with_server") or 0) / denominator
    return [
        CheckResult(
            "server_coverage",
            passed=True,
            samples=denominator,
            max_abs_diff=1 - ratio,
            details=[f"coverage={ratio:.2f} threshold={min_coverage}"],
            warning=ratio < min_coverage,
        ),
        CheckResult(
            "unverified_joins",
            passed=True,
            samples=denominator,
            max_abs_diff=None,
            details=[f"rollouts with an unverified step pairing: {counts.get('join_unverified_rollouts', 0)}"],
            warning=bool(counts.get("join_unverified_rollouts")),
        ),
    ]


def _load_timelines(path: Path) -> dict[str, dict[str, Any]]:
    timelines: dict[str, dict[str, Any]] = {}
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = orjson.loads(line)
            if row.get("kind") == "rollout":
                timelines[row["rollout_id"]] = row
    return timelines


def run_checks(run_dir: Path, *, rollouts_name: str = "rollouts.jsonl", min_coverage: float = 0.99) -> dict[str, Any]:
    run_dir = Path(run_dir)
    timelines = _load_timelines(locate(run_dir, "rollouts_timeline.jsonl"))
    aggregated: dict[str, _Check] = {}
    counts: Counter = Counter()
    with (run_dir / rollouts_name).open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = orjson.loads(line)
            attempt = record.get("_ng_attempt_index")
            rollout_id = (
                record.get("_ng_rollout_id") or f"{record.get('_ng_task_index')}-{record.get('_ng_rollout_index')}"
            )
            if attempt:
                rollout_id = f"{rollout_id}-a{attempt}"
            timeline = timelines.get(rollout_id)
            if timeline is None:
                counts["rollouts_without_timeline"] += 1
                continue
            counts["rollouts"] += 1
            if "model_call_join_unverified" in (timeline.get("flags") or []):
                counts["join_unverified_rollouts"] += 1
            for span in timeline.get("spans") or []:
                if span["name"] == "model_call":
                    counts["model_calls"] += 1
                    counts["model_calls_with_server"] += "server_span_id" in (span.get("attrs") or {})
            for result in check_record(record, timeline):
                check = aggregated.setdefault(result.name, _Check(result.name))
                check.samples += result.samples
                if result.max_abs_diff is not None:
                    check.max_abs_diff = (
                        result.max_abs_diff
                        if check.max_abs_diff is None
                        else max(check.max_abs_diff, result.max_abs_diff)
                    )
                check.details.extend(f"{rollout_id}: {d}" for d in result.details)
    results = [c.result() for c in aggregated.values()] + check_coverage(counts, min_coverage=min_coverage)
    report = {
        "passed": all(r.passed for r in results),
        "failed_checks": [r.name for r in results if not r.passed],
        "warnings": [r.name for r in results if r.warning],
        "counts": dict(counts),
        "checks": [asdict(r) for r in results],
    }
    (metrics_dir(run_dir) / "timeline_check.json").write_text(json.dumps(report, indent=2))
    return report
