# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark-level summary of one run: ``metrics/benchmark_summary.json`` and ``metrics/report.html``.

Per-trial numbers are read from the files the pipeline already wrote (derived metrics, native rows, the check
and build summaries, Gym's aggregate metrics, the launcher's ``run_info.json``). This module adds run-level
aggregation, task groups and one streaming pass over the timelines for agent statuses and occupancy bins,
and nothing else.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import orjson

from ng_latency.analyze import TIMELINE_NAME, TIMELINES_PAGE, run_analysis
from ng_latency.build import _identity, count_failures
from ng_latency.derive import _TRIVIAL_COMMANDS, _TRIVIAL_WAIT_S, FLAGS, GENERATION_TAIL_TOKENS
from ng_latency.limitations import LIMITATIONS
from ng_latency.paths import METRICS_DIR_NAME, locate, metrics_dir
from ng_latency.report_html import render_report_html
from ng_latency.summarize import stats, summarize


REPORT_SCHEMA_VERSION = 1
REPORT_DEFINITIONS = {
    "planned": "rollouts the run was asked to produce: rows of rollouts_materialized_inputs.jsonl over all replicas "
    "(null when a replica lacks the file)",
    "evaluated": "finished rollouts whose row says evaluation_completed (null when no row carries the field)",
    "e2e_agent_time_completed": "e2e_agent_time over completed rollouts: agent status ok, so no timeout and no error",
    "successful": "recorded rollouts whose agent finished on its own (agent status ok); says nothing about the reward",
    "with_agent_data": "recorded rollouts with agent-side spans (an e2e_agent_time); agent-side metrics are over these",
    "benchmark_run_time_s": "first rollout start to last rollout end over all replicas, on the collectors' clocks",
    "dispatched": "rollouts the collector sent to /run: recorded rows plus rows of the failures sidecar",
    "scoring_denominator": "recorded rollouts that carry a reward; the score and every aggregate are over these",
}
PROGRESS_BINS = 20  # state bands: each rollout's agent window split into this many equal parts
BIN_S = 60.0
MIN_COVERAGE = 0.99
TAIL_SLOWEST = 20
TAIL_OTHER = 10
TOP_OPERATIONS = 15
STAT_KEYS = ("count", "sum", "mean", "median", "p90", "p99", "min", "max")
SUMMARY_KEYS = (
    "schema_version",
    "generated_at",
    "definitions",
    "run",
    "validation",
    "outcomes",
    "native_aggregate",
    "time",
    "tokens",
    "where_time_goes",
    "repeats",
    "tail",
    "per_task",
    "tasks",
    "occupancy",
    "data_quality",
)
HANG_STALL_S = 300.0  # a timed-out agent silent this long before the kill hung rather than ran out of time
STEP_BUCKET = 10
MIN_ROLLOUTS_PER_BUCKET = 5  # a per-step median drawn from fewer rollouts is one rollout's story, not the policy's
STEP_CAPS = (40, 60, 80, 100, 120)  # what-if caps on main-agent steps
GENERATION_CAPS = (16_000, 32_000, 64_000, 128_000)  # what-if caps on generated tokens per model call
STALL_QUANTILE = 0.95  # a step in the top 5 % of reasoning length is a stall
TOP_ACTIONS = 10
SPREAD_EDGES = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 8.0)  # slowest / fastest repeat; the last bucket is open
EXCESS_EDGES = tuple(range(0, 84, 4))  # extra tool calls over the leanest repeat; the last bucket is open
CONTEXT_BUCKETS = (
    (0, 8_000, "<8k"),
    (8_000, 16_000, "8–16k"),
    (16_000, 32_000, "16–32k"),
    (32_000, 64_000, "32–64k"),
    (64_000, 128_000, "64–128k"),
    (128_000, float("inf"), "≥128k"),
)
COMPARED_QUANTITIES = (
    "agent_time",
    "steps",
    "tool_calls",
    "reasoning_tokens",
    "output_tokens",
    "input_tokens",
    "subagents",
    "longest_model_call",
    "model_time",
    "tool_time",
    "other_time",
)
_SANDBOX_ENDED = re.compile(r"has ended \((\w+)\)")
BEHAVIOUR_KEYS = (
    "e2e_agent_time",
    "model_calls",
    "tool_calls",
    "repeated_actions",
    "repeated_reads",
    "interrupts_sent",
    "reasoning_per_step_drift",
    "generation_tail_calls",
    "length_stop_calls",
    "max_context",
    "cumulative_input_tokens",
    "stall_before_end_s",
    "tool_wait_share",
)


class ReportError(Exception):
    """A usage error or unreadable input; the CLI prints the message and exits 2."""


@dataclass
class ReplicaInputs:
    name: str
    path: Path
    run_info: Optional[dict[str, Any]] = None
    timeline: Optional[Path] = None
    derived: Optional[Path] = None
    check: Optional[dict[str, Any]] = None
    build_summary: Optional[dict[str, Any]] = None
    rollouts: Optional[Path] = None
    failures: dict[str, Any] = field(default_factory=dict)  # count_failures(): path, present, rows, by_class
    planned: Optional[int] = None  # rows of rollouts_materialized_inputs.jsonl
    native_aggregate: Optional[Any] = None  # Gym writes a list with one entry per agent
    timelines_page: Optional[Path] = None


@dataclass
class RolloutTrace:
    rollout_id: str
    replica: str
    start: Optional[float] = None
    end: Optional[float] = None
    status: Optional[str] = None
    error_type: Optional[str] = None
    model: list[tuple[float, float]] = field(default_factory=list)
    tool: list[tuple[float, float]] = field(default_factory=list)
    exec_status: Optional[str] = None  # the agent_sandbox_exec span, when the record has one
    exec_error: Optional[str] = None
    gap_detail: Optional[str] = None  # why the agent artifact is missing, when it is
    calls: list[dict[str, Any]] = field(default_factory=list)  # per model call: agent, server and engine seconds


def resolve_job_dir(experiment_dir: Path) -> Path:
    """A directory holding ``replica_*`` (an experiment directory, or an old ``slurm_job_id_<id>`` directory), or an
    experiment root holding exactly one such job directory."""
    path = Path(experiment_dir)
    if not path.is_dir():
        raise ReportError(f"{path} is not a directory")
    if any(p.is_dir() and p.name.startswith("replica_") for p in path.iterdir()):
        return path
    jobs = sorted(p for p in path.iterdir() if p.is_dir() and p.name.startswith("slurm_job_id_"))
    if len(jobs) == 1:
        return jobs[0]
    if not jobs:
        raise ReportError(f"{path} has neither replica_* nor slurm_job_id_* directories")
    raise ReportError(f"{path} holds {len(jobs)} jobs; pass one of: " + ", ".join(str(j) for j in jobs))


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ReportError(f"{path}: invalid JSON ({exc})") from exc


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Rows of a JSONL file; a line that does not parse is a truncated file and stops the report."""
    with Path(path).open("rb") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                raise ReportError(f"{path}:{number}: invalid JSON ({exc})") from exc


def discover_replicas(job_dir: Path, *, rollouts_name: str = "rollouts.jsonl") -> list[ReplicaInputs]:
    """Every ``replica_*`` directory with its inputs; derives metrics for a replica that has a timeline but no
    derived file yet. Never runs the build step."""
    replicas: list[ReplicaInputs] = []
    for path in sorted(p for p in Path(job_dir).iterdir() if p.is_dir() and p.name.startswith("replica_")):
        rep = ReplicaInputs(name=path.name, path=path)
        rep.run_info = _read_json(locate(path, "run_info.json"))
        timeline = locate(path, TIMELINE_NAME)
        rep.timeline = timeline if timeline.exists() else None
        derived = locate(path, "derived_metrics.jsonl")
        if rep.timeline is not None and not derived.exists():
            for source in (rep.timeline, path / rollouts_name):
                if source.exists():
                    for _ in read_jsonl(source):  # a truncated line must stop with file:line, not a traceback
                        pass
            run_analysis(path, timelines=False, rollouts_name=rollouts_name)
            derived = locate(path, "derived_metrics.jsonl")
        rep.derived = derived if derived.exists() else None
        rep.check = _read_json(locate(path, "timeline_check.json"))
        rep.build_summary = _read_json(locate(path, "timeline_summary.json"))
        rollouts = path / rollouts_name
        rep.rollouts = rollouts if rollouts.exists() else None
        rep.failures = count_failures(path, rollouts_name)
        planned = path / f"{Path(rollouts_name).stem}_materialized_inputs.jsonl"
        rep.planned = sum(1 for _ in read_jsonl(planned)) if planned.exists() else None
        rep.native_aggregate = _read_json(path / "rollouts_aggregate_metrics.json")
        page = locate(path, TIMELINES_PAGE)
        rep.timelines_page = page if page.exists() else None
        replicas.append(rep)
    if not replicas:
        raise ReportError(f"no replica_* directories under {job_dir}")
    return replicas


def _timed(start: Any, end: Any) -> bool:
    return isinstance(start, (int, float)) and isinstance(end, (int, float))


def scan_timeline(path: Path, replica: str) -> list[RolloutTrace]:
    """One trace per rollout record: its window, the agent status and the model and tool intervals."""
    traces: list[RolloutTrace] = []
    for row in read_jsonl(path):
        if row.get("kind") != "rollout":
            continue
        trace = RolloutTrace(rollout_id=str(row.get("rollout_id")), replica=replica)
        for span in row.get("spans") or []:
            name, start, end = span.get("name"), span.get("start_wall_s"), span.get("end_wall_s")
            if name == "rollout":
                trace.start = float(start) if isinstance(start, (int, float)) else None
                trace.end = float(end) if isinstance(end, (int, float)) else None
            elif name == "agent":
                trace.status = span.get("status")
                attrs = span.get("attrs") or {}
                outcome = attrs.get("exec_outcome") if trace.status != "ok" else None
                trace.error_type = attrs.get("error_type") or outcome
            elif name == "model_call" and _timed(start, end):
                trace.model.append((float(start), float(end)))
                trace.calls.append(
                    {"span_id": span.get("span_id"), "agent_s": float(end) - float(start), "generated": 0}
                )
            elif name in ("tool_call", "tool_batch") and _timed(start, end):
                trace.tool.append((float(start), float(end)))
            elif name in ("agent_sandbox_exec", "agent_execute"):
                trace.exec_status = span.get("status")
                trace.exec_error = (span.get("attrs") or {}).get("error")
            elif name == "model_call_server" and _timed(start, end):
                parent = span.get("parent_span_id")
                call = next((c for c in trace.calls if c["span_id"] == parent), None)
                if call is not None:
                    vllm = (span.get("attrs") or {}).get("vllm") or {}
                    # tokens live on the server record; generated = completion (output + reasoning)
                    tokens = (span.get("attrs") or {}).get("tokens") or {}
                    generated = tokens.get("completion_total")
                    if not isinstance(generated, (int, float)):
                        generated = sum(int(tokens.get(k) or 0) for k in ("output", "reasoning"))
                    call["generated"] = int(generated or 0)
                    call["server_s"] = float(end) - float(start)
                    call["engine_e2e_s"] = vllm.get("e2e_s") if _is_number(vllm.get("e2e_s")) else None
                    call["ttft_s"] = vllm.get("ttft_s") if _is_number(vllm.get("ttft_s")) else None
        for gap in row.get("gaps") or []:
            if isinstance(gap, dict) and gap.get("code") == "agent_artifact_unavailable":
                trace.gap_detail = gap.get("detail")
        traces.append(trace)
    return traces


def trace_status(trace: Optional[RolloutTrace]) -> str:
    """The agent span's status. A rollout without an agent span whose sandbox exec failed or timed out takes the
    exec's status (the sandbox died under the agent); anything else is unknown."""
    if trace is None:
        return "unknown"
    if trace.status:
        return trace.status
    if trace.exec_status in ("error", "timeout"):
        return trace.exec_status
    return "unknown"


def error_label(trace: RolloutTrace, row: Optional[dict[str, Any]]) -> str:
    """What went wrong, as far as the records say: the sandbox's end reason when the agent span is missing, else
    the agent's error type followed by the process's last words when the native row kept them."""
    if not trace.status:
        match = _SANDBOX_ENDED.search(trace.gap_detail or "")
        return f"sandbox ended: {match.group(1)}" if match else f"sandbox error: {trace.exec_error or 'unknown'}"
    base = str(trace.error_type or "unknown")
    tail = (row or {}).get("stderr_tail")
    return f"{base}: {tail}" if tail else base


def occupancy(traces: list[RolloutTrace], *, bin_s: float = BIN_S) -> Optional[dict[str, Any]]:
    """Per-rollout windows relative to the first rollout start, and bins over the run counting rollouts active, in
    a model call, in a tool, or elsewhere at the bin's midpoint. Rollouts without a timed window are left out."""
    timed = [t for t in traces if t.start is not None and t.end is not None]
    if not timed:
        return None
    origin, last = min(t.start for t in timed), max(t.end for t in timed)
    rollouts = sorted(
        [
            [t.replica, t.rollout_id, round(t.start - origin, 3), round(t.end - origin, 3), t.status or "unknown"]
            for t in timed
        ],
        key=lambda r: (r[2], r[0], r[1]),
    )
    bins: list[dict[str, Any]] = []
    for i in range(int((last - origin) // bin_s) + 1):
        mid = origin + (i + 0.5) * bin_s
        active = in_model = in_tool = 0
        for t in timed:
            if not (t.start <= mid < t.end):
                continue
            active += 1
            if any(s <= mid < e for s, e in t.model):
                in_model += 1
            elif any(s <= mid < e for s, e in t.tool):
                in_tool += 1
        bins.append(
            {
                "t": round(i * bin_s, 3),
                "active": active,
                "in_model": in_model,
                "in_tool": in_tool,
                "elsewhere": active - in_model - in_tool,
            }
        )
    return {"origin_wall_s": origin, "bin_s": bin_s, "rollouts": rollouts, "bins": bins, "breaks": []}


def state_bands(traces: list[RolloutTrace], *, bins: int = PROGRESS_BINS) -> Optional[dict[str, Any]]:
    """What rollouts are doing as they progress: each rollout's window is split into ``bins`` equal parts and the
    share of rollouts in a model call, in a tool or elsewhere at each part's midpoint is averaged over rollouts.
    Short and long rollouts weigh the same, so this is the shape of a typical rollout, not of the run."""
    timed = [t for t in traces if t.start is not None and t.end is not None and t.end > t.start]
    if not timed:
        return None
    out: list[dict[str, Any]] = []
    for i in range(bins):
        model = tool = 0
        for t in timed:
            mid = t.start + (i + 0.5) / bins * (t.end - t.start)
            if any(s <= mid < e for s, e in t.model):
                model += 1
            elif any(s <= mid < e for s, e in t.tool):
                tool += 1
        n = len(timed)
        out.append(
            {
                "progress": round((i + 0.5) / bins, 4),
                "model": round(model / n, 4),
                "tool": round(tool / n, 4),
                "other": round((n - model - tool) / n, 4),
            }
        )
    return {"rollouts": len(timed), "bins": out}


_RUN_KEYS = (
    "benchmark",
    "instance",
    "agent",
    "model",
    "model_name",
    "vllm_config",
    "container",
    "concurrency",
    "num_replicas",
    "sandbox_timeout_s",
    "walltime",
    "max_num_seqs",
    "experiment_name",
    "slurm_job_id",
    "code_commit",
    "latency_metrics",
)
_TOKEN_KEYS = ("input", "output", "reasoning", "cached")
_TIME_KEYS = ("e2e_rollout_time", "e2e_agent_time", "pre_agent_time", "post_agent_time")
_TOKEN_METRIC_KEYS = (
    "prompt_cache_hit_share",
    "max_context",
    "context_growth",
    "cumulative_input_tokens",
    "reasoning_share_of_generated_tokens",
)
_WHERE_KEYS = (
    "critical_path_duration",
    "cp_model_time",
    "cp_tool_time",
    "cp_other_time",
    "cp_model_fraction",
    "cp_tool_fraction",
    "cp_other_fraction",
    "model_call_time",
    "model_engine_queue_time",
    "model_engine_prefill_time",
    "model_engine_decode_time",
    "model_engine_e2e_time",
    "gym_hop_overhead_time",
    "tool_execution_time",
    "tool_dispatch_delay",
    "tool_observation_delay",
    "compactions",
    "compaction_time",
    "compaction_reasons",
    "subagent_invocations",
    "subagent_time",
    "retry_rate",
    "model_retry_overhead_time",
    "failed_model_attempts",
    "unexplained_no_model_tool_time",
    "unexplained_fraction",
    "lifecycle_residual_time",
    "parallel_execution_time",
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _mean(values: list[float]) -> Optional[float]:
    return round(sum(values) / len(values), 6) if values else None


def _count_true(rows: list[dict[str, Any]], key: str) -> Optional[int]:
    present = [r for r in rows if r.get(key) is not None]
    return sum(1 for r in present if r[key] is True) if present else None


def task_group(task_id: Any) -> Optional[str]:
    """SWE-bench ids are ``<repo>__<repo>-<n>``; the part before ``__`` is the group. Others have none."""
    text = str(task_id or "")
    return text.split("__", 1)[0] if "__" in text else None


def load_trials(rep: ReplicaInputs) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    """The derived-metrics header and trial rows of one replica, relabelled with the replica name."""
    if rep.derived is None:
        return None, []
    header: Optional[dict[str, Any]] = None
    trials: list[dict[str, Any]] = []
    for row in read_jsonl(rep.derived):
        if "rollout_id" in row:
            row["replica"] = rep.name  # the per-replica analyze labels its own directory "."
            trials.append(row)
        elif row.get("kind") == "header":
            header = row
    return header, trials


def load_native_rows(rep: ReplicaInputs) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if rep.rollouts is None:
        return rows
    for row in read_jsonl(rep.rollouts):
        ident = _identity(row)
        rows[str(ident["rollout_id"])] = {
            "task_id": ident["task_id"],
            "reward": row.get("reward"),
            "resolved": row.get("resolved"),
            "completed": row.get("terminus2_completed"),
            "evaluated": row.get("evaluation_completed"),
            "stderr_tail": _stderr_tail(row.get("opencode_run_stderr")),
        }
    return rows


def _stderr_tail(text: Any) -> Optional[str]:
    """The last non-empty line of the agent process's stderr, trimmed, or None."""
    if not isinstance(text, str):
        return None
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1][-80:] if lines else None


def _trial_tokens(trial: dict[str, Any]) -> dict[str, int]:
    totals = {k: 0 for k in _TOKEN_KEYS}
    for data in (trial.get("by_purpose") or {}).values():
        for key in _TOKEN_KEYS:
            totals[key] += int((data.get("tokens") or {}).get(key) or 0)
    return totals


def agent_time_completed_only(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """``e2e_agent_time`` over completed rollouts (agent status ok): a timed-out agent's duration measures the
    limit, not the task, and a failed agent's duration measures when it died."""
    values = [(t.get("metrics") or {}).get("e2e_agent_time") for t in trials if t.get("agent_status") == "ok"]
    return stats(values)


def benchmark_run_time(traces: list[RolloutTrace]) -> Optional[float]:
    """First rollout start to last rollout end over all replicas: what the collectors' clocks saw of the run."""
    timed = [t for t in traces if t.start is not None and t.end is not None]
    if not timed:
        return None
    return round(max(t.end for t in timed) - min(t.start for t in timed), 3)


def tokens_total(tokens: dict[str, Any]) -> int:
    """Tokens processed: input + output + reasoning. The captured input (prompt_tokens) already includes cache
    reads, so ``cached`` is a subset of input and is reported separately, never added."""
    return sum(int(tokens.get(key) or 0) for key in ("input", "output", "reasoning"))


def _run_section(replicas: list[ReplicaInputs], traces: list[RolloutTrace], layout: str) -> dict[str, Any]:
    infos = [r.run_info for r in replicas if r.run_info]
    first = infos[0] if infos else {}
    run: dict[str, Any] = {key: first.get(key, "unknown") for key in _RUN_KEYS}
    run["replicas"] = [
        {
            "name": r.name,
            "node": (r.run_info or {}).get("node", "unknown"),
            "started_at": (r.run_info or {}).get("started_at", "unknown"),
            "shard_config": (r.run_info or {}).get("shard_config", "unknown"),
        }
        for r in replicas
    ]
    ends = [t.end for t in traces if t.end is not None]
    run["ended_at_wall_s"] = max(ends) if ends else None
    run["layout"] = layout
    return run


def _validation_section(
    replicas: list[ReplicaInputs],
    trials: list[dict[str, Any]],
    rows_by_replica: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    per: list[dict[str, Any]] = []
    trusted = True
    for rep in replicas:
        build, check = rep.build_summary or {}, rep.check or {}
        coverage = build.get("server_coverage")
        per.append(
            {
                "replica": rep.name,
                "rollouts": len(rows_by_replica[rep.name]),
                "rollouts_with_timeline": build.get("rollouts_with_timeline"),
                "server_coverage": coverage,
                "check_passed": check.get("passed"),
                "failed_checks": list(check.get("failed_checks") or []),
                "failed_details": {
                    str(c.get("name")): list(c.get("details") or [])[:3]
                    for c in check.get("checks") or []
                    if isinstance(c, dict) and c.get("passed") is False
                },
                "warnings": list(check.get("warnings") or []),
                "latency_data": rep.timeline is not None,
            }
        )
        low_coverage = _is_number(coverage) and coverage < MIN_COVERAGE
        if rep.timeline is None or check.get("passed") is not True or low_coverage:
            trusted = False
    excluded: Counter = Counter()
    for trial in trials:
        for gap in trial.get("gaps") or []:
            excluded[str(gap.get("code"))] += 1
    for rep in replicas:
        without = len(rows_by_replica[rep.name]) - sum(1 for t in trials if t["replica"] == rep.name)
        if without > 0:
            excluded["no_timeline"] += without
        if rep.failures.get("rows"):
            excluded["collector_failure"] += rep.failures["rows"]
    return {"replicas": per, "trusted": trusted, "excluded_trials": dict(sorted(excluded.items()))}


def _outcomes_section(
    rows_by_replica: dict[str, dict[str, Any]],
    traces: list[RolloutTrace],
    replicas: list[ReplicaInputs],
    traces_trials: list[dict[str, Any]] = (),
) -> dict[str, Any]:
    flat = [(rep, rid, row) for rep, rows in rows_by_replica.items() for rid, row in rows.items()]
    rows = [row for _, _, row in flat]
    rewards = [r["reward"] for r in rows if _is_number(r["reward"])]
    success = sum(1 for v in rewards if v > 0)
    by_key = {(t.replica, t.rollout_id): t for t in traces}
    status: Counter = Counter()
    errors: Counter = Counter()
    for rep, rid, row in flat:
        trace = by_key.get((rep, rid))
        name = trace_status(trace)
        status[name] += 1
        if name == "error" and trace is not None:
            errors[error_label(trace, row)] += 1
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        group = task_group(row["task_id"])
        if group is None:
            continue
        entry = groups.setdefault(group, {"count": 0, "rewards": []})
        entry["count"] += 1
        if _is_number(row["reward"]):
            entry["rewards"].append(row["reward"])
    failures_by_class: Counter = Counter()
    for rep in replicas:
        failures_by_class.update(rep.failures.get("by_class") or {})
    collector_failures = sum(int(rep.failures.get("rows") or 0) for rep in replicas)
    return {
        "rollouts": len(flat),
        "dispatched": len(flat) + collector_failures,
        "scoring_denominator": len(rewards),
        "success": success,
        "failure": len(rewards) - success,
        "reward_mean": _mean(rewards),
        "resolved": _count_true(rows, "resolved"),
        "completed": _count_true(rows, "completed"),
        "planned": None if any(r.planned is None for r in replicas) else sum(r.planned for r in replicas),
        "evaluated": _count_true(rows, "evaluated"),
        "successful": status.get("ok", 0),
        "with_agent_data": sum(1 for t in traces_trials if (t.get("metrics") or {}).get("e2e_agent_time") is not None),
        "agent_status": dict(sorted(status.items())),
        "error_types": dict(sorted(errors.items())),
        "collector_failures": collector_failures,
        "collector_failures_by_class": dict(sorted(failures_by_class.items())),
        "failures_path": "; ".join(str(rep.failures.get("path")) for rep in replicas if rep.failures.get("present"))
        or None,
        "by_group": {g: {"count": e["count"], "reward_mean": _mean(e["rewards"])} for g, e in sorted(groups.items())},
        "timeouts": timeouts_view(traces_trials),
    }


def _time_section(
    summary_all: dict[str, Any],
    traces: list[RolloutTrace],
    occ: Optional[dict[str, Any]],
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    """Wall clock is first rollout start to last rollout end: per replica over its own rollouts, overall over all."""
    timed = [t for t in traces if t.start is not None and t.end is not None]
    wall: dict[str, Any] = {}
    for name in sorted({t.replica for t in timed}):
        own = [t for t in timed if t.replica == name]
        wall[name] = round(max(t.end for t in own) - min(t.start for t in own), 3)
    overall = benchmark_run_time(traces)
    throughput = round(len(timed) / (overall / 3600.0), 3) if overall else None
    metrics = summary_all["metrics"]
    return {
        "wall_clock_s": {"overall": overall, **wall},
        "throughput_rollouts_per_hour": throughput,
        "achieved_concurrency": stats([b["active"] for b in occ["bins"]]) if occ else None,
        **{key: metrics.get(key) for key in _TIME_KEYS},
        "e2e_agent_time_completed": agent_time_completed_only(trials),
        "benchmark_run_time_s": overall,
    }


def _tokens_section(summary_all: dict[str, Any], trials: list[dict[str, Any]]) -> dict[str, Any]:
    by_purpose = summary_all["by_purpose"]
    per_trial = [_trial_tokens(t) for t in trials]
    metrics = summary_all["metrics"]
    totals = {k: sum(int((p.get("tokens") or {}).get(k) or 0) for p in by_purpose.values()) for k in _TOKEN_KEYS}
    return {
        "totals": totals,
        "total": tokens_total(totals),
        "per_rollout": {k: stats([t[k] for t in per_trial]) for k in _TOKEN_KEYS},
        "by_purpose": {p: dict(d.get("tokens") or {}) for p, d in by_purpose.items()},
        **{key: metrics.get(key) for key in _TOKEN_METRIC_KEYS},
    }


def _median(values: list[Any]) -> Optional[float]:
    return stats([v for v in values if _is_number(v)])["median"]


def task_consistency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per task over its repeats: verdict (always pass / always fail / mixed / single repeat) and the
    means behind it, sorted by mean reward then task id."""
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row.get("task_id")), []).append(row)
    out = []
    for task_id, group in by_task.items():
        rewards = [float(r.get("reward") or 0) for r in group]
        if len(group) < 2:
            verdict = "single repeat"
        elif min(rewards) > 0:
            verdict = "always pass"
        elif max(rewards) == 0:
            verdict = "always fail"
        else:
            verdict = "mixed"
        out.append(
            {
                "task_id": task_id,
                "repeats": len(group),
                "reward_mean": _mean(rewards),
                "verdict": verdict,
                "agent_time_mean": _mean([r["e2e_agent_time"] for r in group if _is_number(r.get("e2e_agent_time"))]),
                "model_calls_mean": _mean([r["model_calls"] for r in group if _is_number(r.get("model_calls"))]),
                "repeated_actions_mean": _mean(
                    [r["repeated_actions"] for r in group if _is_number(r.get("repeated_actions"))]
                ),
                "timeouts": sum(1 for r in group if r.get("status") == "timeout"),
                "agent_time_pass_median": _median([r.get("e2e_agent_time") for r in group if _passed(r)]),
                "agent_time_fail_median": _median([r.get("e2e_agent_time") for r in group if not _passed(r)]),
                "steps_median": _median([r.get("agent_steps") for r in group]),
            }
        )
    return sorted(out, key=lambda t: (t["reward_mean"] if t["reward_mean"] is not None else -1.0, t["task_id"]))


def _passed(row: dict[str, Any]) -> bool:
    return _is_number(row.get("reward")) and row["reward"] > 0


def always_solved(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Task id -> its rollout rows, for tasks with at least two repeats, every repeat resolved and every repeat
    carrying an agent time; repeats are ordered fastest first. Variation inside such a task is the policy's, not
    the task's."""
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row.get("task_id")), []).append(row)
    out: dict[str, list[dict[str, Any]]] = {}
    for task_id, group in by_task.items():
        if len(group) < 2 or not all(_passed(r) and _is_number(r.get("e2e_agent_time")) for r in group):
            continue
        out[task_id] = sorted(group, key=lambda r: r["e2e_agent_time"])
    return out


def _bucket_counts(values: list[float], edges: tuple[float, ...]) -> dict[str, Any]:
    """Counts per [edges[i], edges[i+1]) plus an open last bucket from edges[-1]."""
    counts = [0] * len(edges)
    for value in values:
        index = len(edges) - 1
        for i in range(len(edges) - 1):
            if edges[i] <= value < edges[i + 1]:
                index = i
                break
        counts[index] += 1
    return {"edges": list(edges), "counts": counts}


def repeatability_view(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Wall-time repeatability on always-solved tasks: every repeat's agent time per task, the task median, the
    spread (slowest / fastest) and the coefficient of variation, with run-level medians and a spread histogram."""
    tasks = []
    for task_id, group in always_solved(rows).items():
        times = [float(r["e2e_agent_time"]) for r in group]
        mean = sum(times) / len(times)
        deviation = (sum((v - mean) ** 2 for v in times) / len(times)) ** 0.5
        tasks.append(
            {
                "task_id": task_id,
                "repeats": len(times),
                "agent_times": times,
                "median": stats(times)["median"],
                "spread": round(times[-1] / times[0], 3) if times[0] > 0 else None,
                "cv": round(deviation / mean, 6) if mean > 0 else None,
            }
        )
    tasks.sort(key=lambda t: (t["median"], t["task_id"]))
    spreads = [t["spread"] for t in tasks if _is_number(t["spread"])]
    return {
        "tasks": tasks,
        "n_tasks": len(tasks),
        "spread_median": _median(spreads),
        "cv_median": _median([t["cv"] for t in tasks]),
        "spread_histogram": _bucket_counts(spreads, SPREAD_EDGES),
    }


def excess_tool_calls_view(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Tool calls above the leanest successful repeat of the same task (always-solved tasks), and tool calls per
    step by outcome over every rollout with steps."""
    values: list[int] = []
    for group in always_solved(rows).values():
        counts = [r.get("tool_calls") for r in group]
        if not all(_is_number(c) for c in counts):
            continue
        floor = min(counts)
        values.extend(int(c - floor) for c in counts)
    per_step: dict[str, list[float]] = {"success": [], "failure": []}
    for row in rows:
        steps, tools = row.get("agent_steps"), row.get("tool_calls")
        if _is_number(steps) and steps > 0 and _is_number(tools) and _is_number(row.get("reward")):
            per_step["success" if _passed(row) else "failure"].append(tools / steps)
    st = stats(values)
    return {
        "rollouts": len(values),
        "values": values,
        "median": st["median"],
        "p90": st["p90"],
        "histogram": _bucket_counts([float(v) for v in values], EXCESS_EDGES),
        "tools_per_step": {k: _median(v) for k, v in per_step.items()},
    }


def _quantities(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent_time": row.get("e2e_agent_time"),
        "steps": row.get("agent_steps"),
        "tool_calls": row.get("tool_calls"),
        "reasoning_tokens": row.get("tokens_reasoning"),
        "output_tokens": row.get("tokens_output"),
        "input_tokens": row.get("tokens_input"),
        "subagents": row.get("subagents"),
        "longest_model_call": row.get("longest_model_call"),
        "model_time": row.get("model_time"),
        "tool_time": row.get("tool_time"),
        "other_time": row.get("other_time"),
    }


def fast_vs_slow_view(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """For every always-solved task, the fastest against the slowest repeat: each quantity on both sides, their
    slow / fast ratio, and where the slow repeat's extra seconds went (model, tool, other); plus the median ratio of
    each quantity across tasks. Tasks are ordered by spread, most variable first."""
    tasks = []
    for task_id, group in always_solved(rows).items():
        fast, slow = _quantities(group[0]), _quantities(group[-1])
        ratios = {
            key: round(slow[key] / fast[key], 3)
            if _is_number(fast[key]) and fast[key] > 0 and _is_number(slow[key])
            else None
            for key in COMPARED_QUANTITIES
        }
        extra = {
            part: round(slow[f"{part}_time"] - fast[f"{part}_time"], 3)
            if _is_number(slow[f"{part}_time"]) and _is_number(fast[f"{part}_time"])
            else None
            for part in ("model", "tool", "other")
        }
        tasks.append(
            {
                "task_id": task_id,
                "repeats": len(group),
                "spread": ratios["agent_time"],
                "fast": {**fast, "replica": group[0].get("replica"), "rollout_id": group[0].get("rollout_id")},
                "slow": {**slow, "replica": group[-1].get("replica"), "rollout_id": group[-1].get("rollout_id")},
                "ratios": ratios,
                "extra": extra,
            }
        )
    tasks.sort(key=lambda t: (-(t["spread"] or 0), t["task_id"]))
    return {
        "tasks": tasks,
        "n_tasks": len(tasks),
        "ratio_medians": {key: _median([t["ratios"][key] for t in tasks]) for key in COMPARED_QUANTITIES},
    }


def _group_entries(steps: list[tuple[str, int]], threshold: float, order: Optional[list[str]] = None) -> list[dict]:
    """Per label: step count, share of stalls (reasoning at or above the threshold), median and p90 reasoning."""
    groups: dict[str, list[int]] = {}
    for label, reasoning in steps:
        groups.setdefault(label, []).append(reasoning)
    labels = order if order is not None else sorted(groups, key=lambda k: -len(groups[k]))
    out = []
    for label in labels:
        values = groups.get(label)
        if not values:
            continue
        st = stats(values)
        out.append(
            {
                "label": label,
                "steps": len(values),
                "stall_rate": round(sum(1 for v in values if v >= threshold) / len(values), 4),
                "median": st["median"],
                "p90": st["p90"],
            }
        )
    return out


def reasoning_clusters_view(trials: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Where long reasoning clusters: every main-agent step tagged with its action, its position decile, the
    context it saw and what preceded it; a stall is a step in the top 5 % of reasoning length run-wide, so the
    baseline stall rate is 5 % and a group above it stands out."""
    steps: list[dict[str, Any]] = []
    for trial in trials:
        series = trial.get("series") or {}
        reasoning = series.get("reasoning_tokens_per_step") or []
        actions, contexts, preceded = (
            series.get("step_action") or [],
            series.get("step_context") or [],
            series.get("step_preceded_by") or [],
        )
        success = (trial.get("outcome") or {}).get("success")
        n = len(reasoning)
        for i, value in enumerate(reasoning):
            if not _is_number(value):
                continue
            context = contexts[i] if i < len(contexts) else None
            bucket = next(
                (label for lo, hi, label in CONTEXT_BUCKETS if _is_number(context) and lo <= context < hi), None
            )
            steps.append(
                {
                    "reasoning": int(value),
                    "action": str(actions[i]) if i < len(actions) else "unknown",
                    "position": f"{min(9, 10 * i // n) * 10}–{min(9, 10 * i // n) * 10 + 10}%",
                    "context": bucket,
                    "preceded": str(preceded[i]) if i < len(preceded) else "unknown",
                    "outcome": None
                    if success is None
                    else ("in a rollout that succeeded" if success else "in a rollout that eventually failed"),
                }
            )
    if not steps:
        return None
    values = sorted(s["reasoning"] for s in steps)
    threshold = values[min(len(values) - 1, int(STALL_QUANTILE * len(values)))]
    st = stats(values)
    by_action = _group_entries([(s["action"], s["reasoning"]) for s in steps], threshold)[:TOP_ACTIONS]
    positions = [f"{d * 10}–{d * 10 + 10}%" for d in range(10)]
    by_position = _group_entries([(s["position"], s["reasoning"]) for s in steps], threshold, positions)
    by_context = _group_entries(
        [(s["context"], s["reasoning"]) for s in steps if s["context"]], threshold, [b[2] for b in CONTEXT_BUCKETS]
    )
    preceded = [(s["preceded"], s["reasoning"]) for s in steps]
    preceded += [(s["outcome"], s["reasoning"]) for s in steps if s["outcome"]]
    return {
        "steps": len(steps),
        "threshold": threshold,
        "baseline": round(1 - STALL_QUANTILE, 4),
        "median": st["median"],
        "p90": st["p90"],
        "p99": st["p99"],
        "by_action": by_action,
        "by_position": by_position,
        "by_context": by_context,
        "by_preceded": _group_entries(preceded, threshold),
    }


def budgets_view(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """What-if caps on the recorded data. Step caps: a resolved rollout with more main-agent steps than the cap is a
    lost pass; a rollout over the cap saves (1 - cap / steps) of its agent time, time spread evenly over steps.
    Generation caps: a resolved rollout with a call above the cap is a lost pass; the tokens above the cap are the
    tokens saved. Upper bounds on savings, lower bounds on losses: a capped agent would behave differently."""
    rows = []
    for trial in trials:
        metrics = trial.get("metrics") or {}
        rows.append(
            {
                "success": bool((trial.get("outcome") or {}).get("success")),
                "steps": metrics.get("agent_steps"),
                "agent_time": metrics.get("e2e_agent_time"),
                "generated": [
                    v for v in (trial.get("series") or {}).get("generated_tokens_per_call") or [] if _is_number(v)
                ],
            }
        )
    passes = sum(1 for r in rows if r["success"])
    timed = [r for r in rows if _is_number(r["steps"]) and r["steps"] > 0 and _is_number(r["agent_time"])]
    time_total = round(sum(r["agent_time"] for r in timed), 3)
    step_caps = []
    for cap in STEP_CAPS:
        lost = sum(1 for r in timed if r["success"] and r["steps"] > cap)
        saved = round(sum(r["agent_time"] * (1 - cap / r["steps"]) for r in timed if r["steps"] > cap), 3)
        step_caps.append(
            {
                "cap": cap,
                "passes": passes,
                "passes_lost": lost,
                "passes_lost_share": round(lost / passes, 4) if passes else None,
                "time_saved_s": saved,
                "time_total_s": time_total,
                "time_saved_share": round(saved / time_total, 4) if time_total else None,
            }
        )
    tokens_total = sum(v for r in rows for v in r["generated"])
    generation_caps = []
    for cap in GENERATION_CAPS:
        lost = sum(1 for r in rows if r["success"] and r["generated"] and max(r["generated"]) > cap)
        saved = sum(max(0, v - cap) for r in rows for v in r["generated"])
        generation_caps.append(
            {
                "cap": cap,
                "passes": passes,
                "passes_lost": lost,
                "passes_lost_share": round(lost / passes, 4) if passes else None,
                "tokens_saved": saved,
                "tokens_total": tokens_total,
                "tokens_saved_share": round(saved / tokens_total, 4) if tokens_total else None,
            }
        )
    return {"step_caps": step_caps, "generation_caps": generation_caps}


def _repeats_section(per_task: list[dict[str, Any]], trials: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "repeatability": repeatability_view(per_task),
        "excess_tool_calls": excess_tool_calls_view(per_task),
        "fast_vs_slow": fast_vs_slow_view(per_task),
        "budgets": budgets_view(trials),
    }


def _last_call(last: dict[str, Any]) -> dict[str, Any]:
    """The model request a killed agent was waiting on, from the server's record of it when there is one. A request
    the server never answered has no record (see the engine-durations limitation) and leaves this empty."""
    flight = last.get("in_flight") if last.get("waiting_on") == "model_call" else None
    if not flight:
        return {}
    seconds = flight.get("vllm_s") if _is_number(flight.get("vllm_s")) else flight.get("server_s")
    return {
        "completion_tokens": flight.get("completion_tokens"),
        "stop_reason": flight.get("stop_reason"),
        "seconds": seconds,
    }


def timeouts_view(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every timed-out rollout with how long it was silent before the kill, what it was last doing and, when it
    was silent, what it was waiting on."""
    out = []
    for trial in trials:
        if trial.get("agent_status") != "timeout":
            continue
        metrics = trial.get("metrics") or {}
        last = trial.get("last_action") or {}
        stall = metrics.get("stall_before_end_s")
        hang = bool(_is_number(stall) and stall >= HANG_STALL_S)
        waiting = last.get("waiting_on")
        call = _last_call(last) if hang else {}
        if not hang:
            verdict = "ran to the cap"
        elif waiting == "model_call":
            verdict = (
                "model call ran to the length cap"
                if call.get("stop_reason") == "length"
                else "waiting on a model call"
            )
        elif waiting == "tool":
            verdict = "waiting on a tool"
        else:
            verdict = "silent, cause unknown"
        out.append(
            {
                "replica": trial.get("replica"),
                "rollout_id": trial.get("rollout_id"),
                "task_id": trial.get("task_id"),
                "agent_time": metrics.get("e2e_agent_time"),
                "stall_s": stall,
                "model_calls": metrics.get("model_calls"),
                "last_action": f"{last['name']} ({last['kind']})" if last.get("name") else "unknown",
                "hang": hang,
                "waiting_on": waiting,
                "verdict": verdict,
                "last_call_tokens": call.get("completion_tokens"),
                "last_call_s": call.get("seconds"),
            }
        )
    return out


def series_by_step(
    trials: list[dict[str, Any]], name: str, *, bucket: int = STEP_BUCKET, min_rollouts: int = MIN_ROLLOUTS_PER_BUCKET
) -> dict[str, Any]:
    """Medians of a per-step series by step bucket, for successful and failed rollouts separately. A bucket fewer
    than ``min_rollouts`` rollouts reach has no median (null); ``rollouts`` says how many reached each bucket."""
    values: dict[str, dict[int, list[float]]] = {"success": {}, "failure": {}}
    reached: dict[str, dict[int, int]] = {"success": {}, "failure": {}}
    for trial in trials:
        success = (trial.get("outcome") or {}).get("success")
        if success is None:
            continue
        key = "success" if success else "failure"
        group = values[key]
        touched: set[int] = set()
        for index, value in enumerate((trial.get("series") or {}).get(name) or []):
            if _is_number(value):
                group.setdefault(index // bucket, []).append(float(value))
                touched.add(index // bucket)
        for b in touched:
            reached[key][b] = reached[key].get(b, 0) + 1
    top = max([b for g in values.values() for b in g] or [-1])
    buckets = list(range(top + 1))
    return {
        "bucket": bucket,
        "min_rollouts": min_rollouts,
        "buckets": [b * bucket for b in buckets],
        **{
            grp: [stats(values[grp][b])["median"] if reached[grp].get(b, 0) >= min_rollouts else None for b in buckets]
            for grp in ("success", "failure")
        },
        "counts": {grp: [len(values[grp].get(b, [])) for b in buckets] for grp in ("success", "failure")},
        "rollouts": {grp: [reached[grp].get(b, 0) for b in buckets] for grp in ("success", "failure")},
    }


def log_histogram(values: list[Any]) -> dict[str, Any]:
    """Counts per decade: [0,10), [10,100), ... up to the decade holding the maximum."""
    numbers = [float(v) for v in values if _is_number(v)]
    top = 10
    while numbers and top <= max(numbers):
        top *= 10
    edges = [0] + [10**k for k in range(1, len(str(top)))]
    counts = [0] * (len(edges) - 1)
    for value in numbers:
        for i in range(len(edges) - 1):
            if edges[i] <= value < edges[i + 1]:
                counts[i] += 1
                break
    return {"edges": edges, "counts": counts}


def _behaviour_by_outcome(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in BEHAVIOUR_KEYS:
        st = metrics.get(key)
        if not isinstance(st, dict) or "all" not in st:
            continue
        success = (st.get("success") or {}).get("mean")
        failure = (st.get("failure") or {}).get("mean")
        both = _is_number(success) and _is_number(failure)
        # A ratio of signed values (reasoning drift) says nothing; the difference does.
        ratio = round(failure / success, 3) if both and success > 0 and failure >= 0 else None
        out[key] = {
            "success": success,
            "failure": failure,
            "ratio": ratio,
            "difference": round(failure - success, 3) if both else None,
        }
    return out


def _generation_tail(summary_all: dict[str, Any], trials: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = summary_all["metrics"]
    total = lambda name: ((metrics.get(name) or {}).get("all") or {}).get("sum") or 0  # noqa: E731
    generated = sum(
        int((p.get("tokens") or {}).get("reasoning") or 0) + int((p.get("tokens") or {}).get("output") or 0)
        for p in summary_all["by_purpose"].values()
    )
    decode_total = ((metrics.get("model_engine_decode_time") or {}).get("per_trial_sum") or {}).get("sum") or 0
    tokens = total("generation_tail_tokens")
    decode = total("generation_tail_decode_time")
    return {
        "threshold_tokens": GENERATION_TAIL_TOKENS,
        "calls": int(total("generation_tail_calls")),
        "calls_total": int(total("model_calls")),
        "rollouts": sum(1 for t in trials if ((t.get("metrics") or {}).get("generation_tail_calls") or 0) > 0),
        "tokens": int(tokens),
        "generated_tokens_total": generated,
        "token_share": round(tokens / generated, 4) if generated else 0.0,
        "decode_time": round(decode, 3),
        "decode_time_total": round(decode_total, 3),
        "decode_share": round(decode / decode_total, 4) if decode_total else 0.0,
        "length_stops": int(total("length_stop_calls")),
    }


def _subagent_effect(trials: list[dict[str, Any]]) -> dict[str, Any]:
    def view(group: list[dict[str, Any]]) -> dict[str, Any]:
        rewards = [(t.get("outcome") or {}).get("reward") for t in group]
        times = [(t.get("metrics") or {}).get("e2e_agent_time") for t in group]
        return {
            "rollouts": len(group),
            "reward_mean": _mean([r for r in rewards if _is_number(r)]),
            "agent_time_mean": _mean([x for x in times if _is_number(x)]),
        }

    def uses(trial: dict[str, Any]) -> bool:
        sub = (trial.get("metrics") or {}).get("subagent_invocations")
        return bool((sub.get("total") if isinstance(sub, dict) else sub) or 0)

    with_, without = [t for t in trials if uses(t)], [t for t in trials if not uses(t)]
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for trial in trials:
        group = task_group(trial.get("task_id"))
        if group is not None:
            groups.setdefault(group, {"with": [], "without": []})["with" if uses(trial) else "without"].append(trial)
    return {
        "with": view(with_),
        "without": view(without),
        "by_group": {g: {k: view(v) for k, v in parts.items()} for g, parts in sorted(groups.items())},
    }


def _waits(summary_all: dict[str, Any]) -> Optional[dict[str, Any]]:
    metrics = summary_all["metrics"]
    merged = metrics.get("wait_analysis")
    if not isinstance(merged, dict) or not merged:
        return None
    share = metrics.get("tool_wait_share") or {}
    return {
        **merged,
        "trivial_commands": list(_TRIVIAL_COMMANDS),
        "trivial_wait_s": _TRIVIAL_WAIT_S,
        "wait_share": {grp: (share.get(grp) or {}).get("mean") for grp in ("all", "success", "failure")},
        "bare_wait_operation": summary_all["by_operation"].get("wait"),
    }


def per_step_tax(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Seconds per main-agent step, per rollout: how much of a step is model, tool, known harness work and
    unexplained. A policy that takes fewer steps pays the per-step tax fewer times; this is the tax."""
    per: dict[str, list[float]] = {
        "model_per_step": [],
        "tool_per_step": [],
        "known_per_step": [],
        "unexplained_per_step": [],
    }
    steps = 0
    for trial in trials:
        m = trial.get("metrics") or {}
        n = m.get("agent_steps")
        if not _is_number(n) or n <= 0:
            continue
        steps += int(n)
        known = sum(v for v in (m.get("known_no_model_tool_time_by_event") or {}).values() if _is_number(v))
        for key, value in (
            ("model_per_step", m.get("model_union_time")),
            ("tool_per_step", m.get("tool_union_time")),
            ("known_per_step", known),
            ("unexplained_per_step", m.get("unexplained_no_model_tool_time")),
        ):
            if _is_number(value):
                per[key].append(float(value) / n)
    return {
        "steps": steps,
        "rollouts": sum(1 for t in trials if _is_number((t.get("metrics") or {}).get("agent_steps"))),
        **{k: stats(v) for k, v in per.items()},
    }


_GENERATED_EDGES = ((0, 100, "<100"), (100, 1_000, "100-1k"), (1_000, 10_000, "1k-10k"), (10_000, None, "10k+"))


def engine_vs_agent(traces: list[RolloutTrace]) -> dict[str, Any]:
    """Model-call time as the agent saw it against the model server's and the engine's time for the same calls, by
    generated-token bucket: the gap between the agent's and the engine's medians is network, proxy and capture."""
    calls = [c for t in traces for c in t.calls if _is_number(c.get("engine_e2e_s"))]
    buckets = []
    for low, high, label in _GENERATED_EDGES:
        chosen = [c for c in calls if c["generated"] >= low and (high is None or c["generated"] < high)]
        if not chosen:
            continue
        buckets.append(
            {
                "bucket": label,
                "calls": len(chosen),
                "agent_s": stats([c["agent_s"] for c in chosen]),
                "server_s": stats([c["server_s"] for c in chosen if _is_number(c.get("server_s"))]),
                "engine_e2e_s": stats([c["engine_e2e_s"] for c in chosen]),
                "ttft_s": stats([c["ttft_s"] for c in chosen if _is_number(c.get("ttft_s"))]),
            }
        )
    return {"calls": len(calls), "buckets": buckets}


def _ecdf(values: list[float]) -> list[list[float]]:
    items = sorted(values)
    n = len(items)
    return [[round(v, 6), round((i + 1) / n, 6)] for i, v in enumerate(items)]


def distributions(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-rollout values by outcome for the box plots, and pooled per-call values with their ECDF."""
    out: dict[str, Any] = {}
    for key in ("e2e_agent_time", "unexplained_no_model_tool_time", "model_calls", "tool_calls"):
        groups: dict[str, list[float]] = {}
        for trial in trials:
            value = (trial.get("metrics") or {}).get(key)
            if _is_number(value):
                outcome = "success" if (trial.get("outcome") or {}).get("success") else "failure"
                groups.setdefault(outcome, []).append(float(value))
        out[key] = {g: {**stats(v), "values": sorted(v)} for g, v in sorted(groups.items())}
    for key in ("model_call_time", "tool_execution_time"):
        pooled = [float(v) for t in trials for v in ((t.get("series") or {}).get(key) or []) if _is_number(v)]
        out[key] = {"all": {**stats(pooled), "ecdf": _ecdf(pooled)}} if pooled else {}
    return out


def _where_time_goes_section(
    summary_all: dict[str, Any], trials: list[dict[str, Any]], traces: list[RolloutTrace]
) -> dict[str, Any]:
    metrics = summary_all["metrics"]
    operations = sorted(summary_all["by_operation"].items(), key=lambda kv: -(kv[1].get("total_time") or 0.0))
    durations: dict[str, list[float]] = {}
    for trial in trials:
        series = trial.get("series") or {}
        for operation, duration in zip(series.get("tool_operation") or [], series.get("tool_execution_time") or []):
            if _is_number(duration):
                durations.setdefault(str(operation), []).append(float(duration))
    by_operation = {}
    for operation, data in operations[:TOP_OPERATIONS]:
        count, total = data.get("count") or 0, data.get("total_time")
        values = durations.get(operation)
        by_operation[operation] = {
            **data,
            "mean": round(total / count, 6) if count and _is_number(total) else None,
            "median": stats(values)["median"] if values else None,
        }
    generated = [v for t in trials for v in ((t.get("series") or {}).get("generated_tokens_per_call") or [])]
    return {
        **{key: metrics.get(key) for key in _WHERE_KEYS},
        "per_step_tax": per_step_tax(trials),
        "engine_vs_agent": engine_vs_agent(traces),
        "tool_time_by_operation": by_operation,
        "by_purpose": summary_all["by_purpose"],
        "behaviour_by_outcome": _behaviour_by_outcome(metrics),
        "generation_tail": _generation_tail(summary_all, trials),
        "generated_tokens_histogram": log_histogram(generated),
        "reasoning_by_step": series_by_step(trials, "reasoning_tokens_per_step"),
        "context_by_step": series_by_step(trials, "context_length_at_call"),
        "waits": _waits(summary_all),
        "subagent_effect": _subagent_effect(trials),
        "reasoning_clusters": reasoning_clusters_view(trials),
    }



# Per-rollout metrics compared between successful and non-successful rollouts: what a failure costs, and whether
# the extra is route length (steps, tools) or thinking per step (reasoning). Each is the median over its group.
SIDE_BY_SIDE_KEYS = (
    ("e2e_agent_time", "agent time (s)"),
    ("agent_steps", "agent steps"),
    ("model_calls", "model calls"),
    ("tool_calls", "tool calls"),
    ("cumulative_input_tokens", "input tokens, cumulative"),
    ("reasoning_tokens_total", "reasoning tokens"),
    ("output_tokens_total", "output tokens"),
    ("max_context", "peak context (tokens)"),
    ("compactions", "compactions"),
    ("repeated_actions", "repeated identical tool calls"),
    ("unexplained_fraction", "unexplained share of agent time"),
)


def failed_vs_successful(metrics: dict[str, Any]) -> dict[str, Any]:
    """Median of each SIDE_BY_SIDE metric over successful and over non-successful rollouts, with the ratio.

    Reads the summary's per-outcome split, so both columns are over the same rollouts as every other table.
    A ratio is given only when both medians exist and the success median is positive.
    """
    rows = []
    for key, label in SIDE_BY_SIDE_KEYS:
        st = metrics.get(key)
        if not isinstance(st, dict):
            continue
        success = (st.get("success") or {}).get("median")
        failure = (st.get("failure") or {}).get("median")
        n_s, n_f = (st.get("success") or {}).get("count"), (st.get("failure") or {}).get("count")
        if not (_is_number(success) or _is_number(failure)):
            continue
        ratio = round(failure / success, 3) if _is_number(success) and _is_number(failure) and success > 0 else None
        rows.append({"metric": key, "label": label, "success": success, "failure": failure, "ratio": ratio,
                     "n_success": n_s, "n_failure": n_f})
    return {"rows": rows}


def _page_links(
    replicas: list[ReplicaInputs], trials: list[dict[str, Any]], report_dir: Path
) -> dict[tuple[str, str], str]:
    links: dict[tuple[str, str], str] = {}
    for rep in replicas:
        if rep.timelines_page is None:
            continue
        page = os.path.relpath(rep.timelines_page, report_dir)
        for trial in trials:
            if trial["replica"] == rep.name:
                # the per-replica page keys its rollouts "./<id>"; the report names the replica
                links[(rep.name, str(trial["rollout_id"]))] = f"{page}#{rep.name}/{trial['rollout_id']}"
    return links


def _tail_entry(trial: dict[str, Any], value: float, links: dict[tuple[str, str], str]) -> dict[str, Any]:
    return {
        "replica": trial["replica"],
        "rollout_id": trial["rollout_id"],
        "task_id": trial.get("task_id"),
        "reward": (trial.get("outcome") or {}).get("reward"),
        "status": trial.get("agent_status"),
        "value": value,
        "page": links.get((trial["replica"], str(trial["rollout_id"]))),
    }


def _top(trials: list[dict[str, Any]], metric: str, n: int, links: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    scored = [(float(v), t) for t in trials if _is_number(v := (t.get("metrics") or {}).get(metric))]
    scored.sort(key=lambda vt: (-vt[0], str(vt[1]["replica"]), str(vt[1]["rollout_id"])))
    return [_tail_entry(t, v, links) for v, t in scored[:n]]


def _tail_section(trials: list[dict[str, Any]], links: dict[tuple[str, str], str]) -> dict[str, Any]:
    return {
        "slowest": _top(trials, "e2e_agent_time", TAIL_SLOWEST, links),
        "most_unexplained": _top(trials, "unexplained_no_model_tool_time", TAIL_OTHER, links),
        "most_calls": _top(trials, "model_calls", TAIL_OTHER, links),
    }


def _per_task_rows(
    rows_by_replica: dict[str, dict[str, Any]], trials: list[dict[str, Any]], links: dict[tuple[str, str], str]
) -> list[dict[str, Any]]:
    by_key = {(t["replica"], str(t["rollout_id"])): t for t in trials}
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []

    def row_for(rep: str, rid: str, task_id: Any, reward: Any) -> dict[str, Any]:
        trial = by_key.get((rep, rid)) or {}
        metrics = trial.get("metrics") or {}
        sub = metrics.get("subagent_invocations")
        tokens = _trial_tokens(trial) if trial else None
        return {
            "replica": rep,
            "rollout_id": rid,
            "task_id": task_id,
            "reward": reward,
            "status": trial.get("agent_status") or "unknown",
            "e2e_agent_time": metrics.get("e2e_agent_time"),
            "model_calls": metrics.get("model_calls"),
            "tool_calls": metrics.get("tool_calls"),
            "tokens_total": tokens_total(tokens) if tokens is not None else None,
            **{f"tokens_{k}": (tokens[k] if tokens is not None else None) for k in _TOKEN_KEYS},
            "max_context": metrics.get("max_context"),
            "context_growth": metrics.get("context_growth"),
            "cumulative_input_tokens": metrics.get("cumulative_input_tokens"),
            "repeated_actions": metrics.get("repeated_actions"),
            "stall_s": metrics.get("stall_before_end_s"),
            "agent_steps": metrics.get("agent_steps"),
            "longest_model_call": max(
                [v for v in (trial.get("series") or {}).get("model_call_time", []) if _is_number(v)], default=None
            ),
            "model_time": metrics.get("model_union_time"),
            "tool_time": metrics.get("tool_union_time"),
            "other_time": metrics.get("non_model_tool_time"),
            "compactions": metrics.get("compactions"),
            "subagents": sub.get("total") if isinstance(sub, dict) else sub,
            "flags": list(trial.get("flags") or []),
            "page": links.get((rep, rid)),
        }

    for rep in sorted(rows_by_replica):
        for rid in sorted(rows_by_replica[rep], key=str):
            native = rows_by_replica[rep][rid]
            out.append(row_for(rep, rid, native["task_id"], native["reward"]))
            seen.add((rep, rid))
    for (rep, rid), trial in sorted(by_key.items()):
        if (rep, rid) not in seen:  # a timeline record whose native row is missing
            out.append(row_for(rep, rid, trial.get("task_id"), (trial.get("outcome") or {}).get("reward")))
    return out


def _data_quality_section(summary_all: dict[str, Any], notes: list[str], without_timeline: int) -> dict[str, Any]:
    return {
        "gaps": summary_all["gaps"],
        "flags": {code: {"count": n, "text": FLAGS.get(code, "")} for code, n in summary_all["flags"].items()},
        "trials_without_timeline": without_timeline,
        "notes": notes,
    }


def build_summary(
    job_dir: Path, *, rollouts_name: str = "rollouts.jsonl", with_occupancy: bool = True
) -> dict[str, Any]:
    """The benchmark summary of one job directory (all its replicas)."""
    job_dir = Path(job_dir)
    replicas = discover_replicas(job_dir, rollouts_name=rollouts_name)
    report_dir = metrics_dir(job_dir, create=False)
    trials: list[dict[str, Any]] = []
    traces: list[RolloutTrace] = []
    notes: list[str] = []
    rows_by_replica: dict[str, dict[str, Any]] = {}
    definitions: dict[str, Any] = {}
    for rep in replicas:
        header, rep_trials = load_trials(rep)
        if header and not definitions:
            definitions = dict(header.get("definitions") or {})
        trials.extend(rep_trials)
        rows_by_replica[rep.name] = load_native_rows(rep)
        if rep.timeline is not None:
            traces.extend(scan_timeline(rep.timeline, rep.name))
        else:
            notes.append(f"latency data missing for {rep.name}: no {TIMELINE_NAME}")
        if rep.run_info is None:
            notes.append(f"no run_info.json for {rep.name}: run facts unknown")
    definitions.update(REPORT_DEFINITIONS)
    with_timeline = [r for r in replicas if r.timeline is not None]
    if not with_timeline:
        layout = "unknown"
    elif all(r.timeline.parent.name == METRICS_DIR_NAME for r in with_timeline):
        layout = "metrics"
    else:
        layout = "legacy"
        notes.append("old layout: files found at the replica root, not under metrics/")
    by_key = {(t.replica, t.rollout_id): t for t in traces}
    for trial in trials:
        trial["agent_status"] = trace_status(by_key.get((trial["replica"], str(trial["rollout_id"]))))
    summary_all = summarize(trials)
    occ = occupancy(traces) if with_occupancy else None
    links = _page_links(replicas, trials, report_dir)
    per_task = _per_task_rows(rows_by_replica, trials, links)
    validation = _validation_section(replicas, trials, rows_by_replica)
    without_timeline = validation["excluded_trials"].get("no_timeline", 0)
    summary = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "definitions": definitions,
        "limitations": [{"title": t, "what": w, "implication": i} for t, w, i in LIMITATIONS],
        "run": _run_section(replicas, traces, layout),
        "validation": validation,
        "outcomes": _outcomes_section(rows_by_replica, traces, replicas, trials),
        "native_aggregate": {r.name: r.native_aggregate for r in replicas},
        "time": _time_section(summary_all, traces, occ, trials),
        "tokens": _tokens_section(summary_all, trials),
        "where_time_goes": _where_time_goes_section(summary_all, trials, traces),
        "distributions": distributions(trials),
        "tail": _tail_section(trials, links),
        "per_task": per_task,
        "tasks": task_consistency(per_task),
        "repeats": _repeats_section(per_task, trials),
        "occupancy": occ,
        "state_bands": state_bands(traces) if with_occupancy else None,
        "data_quality": _data_quality_section(summary_all, notes, without_timeline),
    }
    summary["failed_vs_successful"] = failed_vs_successful(summary_all["metrics"])
    return summary


def write_report(
    experiment_dir: Path, *, rollouts_name: str = "rollouts.jsonl", with_occupancy: bool = True
) -> dict[str, Any]:
    """Write ``metrics/benchmark_summary.json`` and ``metrics/report.html`` for one job."""
    job_dir = resolve_job_dir(experiment_dir)
    summary = build_summary(job_dir, rollouts_name=rollouts_name, with_occupancy=with_occupancy)
    out_dir = metrics_dir(job_dir)
    summary_path = out_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    page_path = out_dir / "report.html"
    page_path.write_text(render_report_html(summary), encoding="utf-8")
    return {
        "job_dir": str(job_dir),
        "rollouts": summary["outcomes"]["rollouts"],
        "trusted": summary["validation"]["trusted"],
        "outputs": [str(summary_path), str(page_path)],
    }
