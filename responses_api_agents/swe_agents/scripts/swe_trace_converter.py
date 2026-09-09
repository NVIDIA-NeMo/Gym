#!/usr/bin/env python3
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

"""
Convert SWE-agent RL training logs to Chrome Trace Event Format for visualization
in Perfetto (https://ui.perfetto.dev/) or chrome://tracing.

This tool parses each rollout's nemo_gym_metrics.json and reconstructs the
execution timeline from absolute event timestamps. The timeline shows:
  - LLM Generation (GPU work) - green
  - Tool Execution (CPU work) - blue
  - Evaluation (CPU work) - red
  - Framework Overhead (time between measured agent events) - red
  - Agent Startup (not instrumented) - gray
  - Agent Finalization (not instrumented) - gray
  - Agent Init (sum of measured container/runtime startup phases) - yellow

Multiple parallel agent rollouts are shown simultaneously, grouped by instance ID,
so you can understand system utilization and CPU overhead in the RL training loop.

Usage:
    python swe_trace_converter.py --log-dir /path/to/results --output trace.json

    Then open the output JSON in https://ui.perfetto.dev/
"""

import argparse
import json
import os
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timezone


AGENT_INIT_METRICS = (
    "generation_apptainer_spinup_time",
    "create_runtime_time",
    "connect_to_runtime_time",
    "initialize_runtime_time",
)

PER_TURN_METRICS = (
    "response_latencies",
    "action_execution_latencies",
    "token_usages",
)


def parse_iso_timestamp(ts_str, naive_offset_seconds=0):
    """Parse an ISO timestamp, treating naive values as UTC plus an offset."""
    if ts_str.endswith("Z"):
        ts_str = f"{ts_str[:-1]}+00:00"
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp() + naive_offset_seconds
    return dt.timestamp()


def is_naive_iso_timestamp(ts_str):
    """Return whether an ISO timestamp omits its UTC offset."""
    if ts_str.endswith("Z"):
        return False
    return datetime.fromisoformat(ts_str).tzinfo is None


def record_span(record, naive_offset_seconds=0):
    """Return a record's absolute start and end timestamps."""
    end = parse_iso_timestamp(record["timestamp"], naive_offset_seconds)
    start = (
        parse_iso_timestamp(record["start_timestamp"], naive_offset_seconds)
        if record.get("start_timestamp")
        else end - record["latency"]
    )
    return start, end, record


def gaps_within(container_start, container_end, spans):
    """Yield uncovered intervals inside a container span."""
    previous_end = container_start
    for start, end in sorted(spans):
        start = max(start, container_start)
        end = min(end, container_end)
        if start >= container_end or end <= container_start:
            continue
        if start > previous_end:
            yield previous_end, start
        previous_end = max(previous_end, end)
    if previous_end < container_end:
        yield previous_end, container_end


def infer_action_timestamp_offset(actions, response_spans, generation_start, generation_end):
    """Infer a whole-hour UTC correction for legacy naive action timestamps.

    OpenHands historically stamped events with the sandbox's local wall clock
    and omitted the offset. Most SWE images use UTC, but an image with another
    timezone can otherwise place its tool spans hours outside the rollout.
    """
    if not any(is_naive_iso_timestamp(action["timestamp"]) for action in actions):
        return 0

    response_ends = sorted(end for _, end, _ in response_spans)

    def nearest_response_distance(timestamp):
        index = bisect_left(response_ends, timestamp)
        distances = []
        if index < len(response_ends):
            distances.append(abs(timestamp - response_ends[index]))
        if index:
            distances.append(abs(timestamp - response_ends[index - 1]))
        return min(distances)

    def candidate_score(offset_seconds):
        action_spans = []
        for action in actions:
            end = parse_iso_timestamp(action["timestamp"], offset_seconds)
            start = (
                parse_iso_timestamp(action["start_timestamp"], offset_seconds)
                if action.get("start_timestamp")
                else end - action["latency"]
            )
            action_spans.append((start, end))

        outside_seconds = sum(
            max(generation_start - start, 0) + max(end - generation_end, 0) for start, end in action_spans
        )
        if response_ends:
            nearest_response_seconds = sum(nearest_response_distance(start) for start, _ in action_spans) / len(
                action_spans
            )
            return outside_seconds, nearest_response_seconds, abs(offset_seconds)
        return outside_seconds, abs(offset_seconds)

    candidates = (hours * 3600 for hours in range(-14, 15))
    return min(candidates, key=candidate_score)


def to_us(seconds):
    """Convert seconds to microseconds for Chrome trace format."""
    return round(seconds * 1_000_000)


def extract_instance_id(dirname):
    """Extract instance ID from directory name.
    Format: <instance_id>_<timestamp_ms>_<hash>
    """
    return dirname.rsplit("_", 2)[0]


def has_per_turn_metrics(nm):
    """Return whether all data needed for a detailed rollout timeline exists."""
    ptm = nm.get("per_turn_metrics")
    return isinstance(ptm, dict) and all(isinstance(ptm.get(field), list) for field in PER_TURN_METRICS)


def validate_precise_metrics(nm, dir_name):
    """Require complete source data for every emitted event."""
    errors = []

    if not nm.get("generation_start_timestamp"):
        errors.append("generation_start_timestamp is missing")
    openhands_run_time = nm.get("openhands_run_time")
    if not isinstance(openhands_run_time, (int, float)) or openhands_run_time < 0:
        errors.append("openhands_run_time is invalid")
    if "ray_queue_time" not in nm:
        errors.append("ray_queue_time is missing")
    elif not isinstance(nm["ray_queue_time"], (int, float)) or nm["ray_queue_time"] < 0:
        errors.append("ray_queue_time is invalid")
    if not isinstance(nm.get("resolved"), bool):
        errors.append("resolved is missing or invalid")
    eval_start = nm.get("evaluation_start_timestamp")
    eval_time = nm.get("final_eval_time")
    if eval_start and (not isinstance(eval_time, (int, float)) or eval_time < 0):
        errors.append("final_eval_time is invalid for a started evaluation")
    if (eval_time or 0) > 0 and not eval_start:
        errors.append("evaluation_start_timestamp is missing for a completed evaluation")

    if not has_per_turn_metrics(nm):
        if errors:
            raise ValueError(f"Incomplete metrics for {dir_name}: " + "; ".join(errors))
        return

    ptm = nm["per_turn_metrics"]

    for field in AGENT_INIT_METRICS:
        if field not in nm:
            continue
        duration = nm.get(field)
        if not isinstance(duration, (int, float)) or duration < 0:
            errors.append(f"{field} is invalid")

    responses = ptm["response_latencies"]
    actions = ptm["action_execution_latencies"]
    token_usages = ptm["token_usages"]

    for label, records in (("response", responses), ("action", actions)):
        for index, record in enumerate(records):
            if not record.get("timestamp"):
                errors.append(f"{label} {index} timestamp is missing")
            if "start_timestamp" in record and not record.get("start_timestamp"):
                errors.append(f"{label} {index} start_timestamp is invalid")
            latency = record.get("latency")
            if not isinstance(latency, (int, float)) or latency < 0:
                errors.append(f"{label} {index} latency is invalid")
            elif record.get("start_timestamp") and record.get("timestamp"):
                measured = parse_iso_timestamp(record["timestamp"]) - parse_iso_timestamp(record["start_timestamp"])
                if abs(measured - latency) > 0.001:
                    errors.append(f"{label} {index} timestamps and latency are inconsistent")

    response_ids = [record.get("response_id") for record in responses]
    token_ids = [record.get("response_id") for record in token_usages]
    if any(not response_id for response_id in response_ids):
        errors.append("a response_id is missing")
    if Counter(response_ids) != Counter(token_ids):
        errors.append("response_latencies and token_usages response_ids do not match")

    if errors:
        raise ValueError(f"Incomplete precise metrics for {dir_name}: " + "; ".join(errors))


def reconstruct_rollout_events(nm):
    """Place events from their absolute UTC timestamps in nemo_gym_metrics.json."""
    gen_start = parse_iso_timestamp(nm["generation_start_timestamp"])
    eval_start = (
        parse_iso_timestamp(nm["evaluation_start_timestamp"]) if nm.get("evaluation_start_timestamp") else None
    )
    ray_queue_time = nm["ray_queue_time"]
    eval_time = nm["final_eval_time"] if eval_start is not None else 0

    events = []
    if not has_per_turn_metrics(nm):
        if ray_queue_time > 0:
            events.append(("queue_wait", gen_start - ray_queue_time, ray_queue_time, {}))
        events.append(("agent_rollout", gen_start, nm["openhands_run_time"], {}))
        if eval_start is not None and eval_time > 0:
            events.append(
                (
                    "evaluation",
                    eval_start,
                    eval_time,
                    {"resolved": nm["resolved"]},
                )
            )
        return events

    ptm = nm["per_turn_metrics"]
    responses = ptm["response_latencies"]
    actions = ptm["action_execution_latencies"]
    token_by_rid = {usage["response_id"]: usage for usage in ptm["token_usages"]}
    root_session_ids = {
        response["session_id"]
        for response in responses
        if response.get("session_id") and response.get("parent_session_id") is None
    }

    response_spans = [record_span(response) for response in responses]
    generation_end = gen_start + nm["openhands_run_time"]
    action_timestamp_offset = infer_action_timestamp_offset(
        actions,
        response_spans,
        gen_start,
        generation_end,
    )
    action_spans = [record_span(action, action_timestamp_offset) for action in actions]

    # Parallel OpenCode subagents can complete out of order. Number turns by
    # their recorded request starts so the trace reflects launch order.
    response_spans.sort(key=lambda span: (span[0], span[1], span[2]["response_id"]))

    llm_starts = []
    for turn, (start, end, response) in enumerate(response_spans, start=1):
        latency = response["latency"]
        duration = end - start
        response_id = response["response_id"]
        token_usage = token_by_rid[response_id]
        request_kind = response.get("request_kind", "agent")
        metadata = {
            "response_id": response_id,
            "recorded_latency": latency,
            "turn": turn,
        }
        for field in ("session_id", "parent_session_id", "session_turn", "request_kind"):
            if field in response:
                metadata[field] = response[field]
        if request_kind == "title":
            metadata["_trace_name"] = "Session Title Generation (GPU)"
        elif request_kind == "subagent":
            metadata["_trace_name"] = "Subagent LLM Generation (GPU)"
        metadata["prompt_tokens"] = token_usage["prompt_tokens"]
        metadata["completion_tokens"] = token_usage["completion_tokens"]
        if "reasoning_tokens" in token_usage:
            metadata["reasoning_tokens"] = token_usage["reasoning_tokens"]
        events.append(("llm_generation", start, duration, metadata))
        llm_starts.append(start)

    for start, end, action in action_spans:
        metadata = {
            "observation_type": action["observation_type"],
            "observation_id": action["observation_id"],
            "message": action["message"],
        }
        for field in ("session_id", "child_session_id", "input", "output"):
            if field in action:
                metadata[field] = action[field]
        if action["observation_type"] == "task":
            metadata["_trace_name"] = "Subagent Task"
            metadata["_nested"] = True
        elif action.get("session_id") and action["session_id"] not in root_session_ids:
            metadata["_trace_name"] = "Subagent Tool Execution (CPU)"
        events.append(
            (
                "tool_execution",
                start,
                end - start,
                metadata,
            )
        )

    # A Subagent Task is a parent tool span containing the child session's LLM
    # and tool activity. The task record's child_session_id provides the exact
    # relationship, including when sibling subagents overlap. Show the exact
    # complement as nested Framework Overhead.
    child_parent = {
        response["session_id"]: response["parent_session_id"]
        for _, _, response in response_spans
        if response.get("session_id") and response.get("parent_session_id")
    }
    measured_by_session = defaultdict(list)
    for start, end, response in response_spans:
        session_id = response.get("session_id")
        if session_id in child_parent:
            measured_by_session[session_id].append((start, end))
    for start, end, action in action_spans:
        session_id = action.get("session_id")
        if session_id in child_parent and action.get("observation_type") != "task":
            measured_by_session[session_id].append((start, end))

    for task_start, task_end, task in action_spans:
        if task.get("observation_type") != "task":
            continue
        child_session_id = task.get("child_session_id")
        if not child_session_id:
            continue

        measured = [
            (start, end)
            for start, end in measured_by_session.get(child_session_id, [])
            if start < task_end and end > task_start
        ]
        if measured:
            for gap_start, gap_end in gaps_within(task_start, task_end, measured):
                events.append(
                    (
                        "framework_overhead",
                        gap_start,
                        gap_end - gap_start,
                        {
                            "scope": "subagent",
                            "session_id": child_session_id,
                            "parent_session_id": task.get("session_id"),
                            "_nested": True,
                        },
                    )
                )

    if ray_queue_time > 0:
        events.append(("queue_wait", gen_start - ray_queue_time, ray_queue_time, {}))

    init_components = {field: nm[field] for field in AGENT_INIT_METRICS if field in nm}
    init_duration = sum(init_components.values())
    if init_duration > 0:
        events.append(("agent_init", gen_start, init_duration, init_components))

    # Agent Startup (not instrumented) is the exact interval after the measured
    # container/runtime init phases finish and before the first LLM request
    # begins. The agent backends emit no finer-grained timestamps inside this
    # interval, so keep it separate from both Agent Init and Framework Overhead.
    init_end = gen_start + init_duration
    if init_duration > 0 and llm_starts:
        first_llm_start = min(llm_starts)
        has_measured_activity_before_first_llm = any(
            category in ("llm_generation", "tool_execution")
            and start < first_llm_start
            and start + duration > init_end
            for category, start, duration, _ in events
        )
        if first_llm_start > init_end and not has_measured_activity_before_first_llm:
            events.append(
                (
                    "agent_startup_uninstrumented",
                    init_end,
                    first_llm_start - init_end,
                    {},
                )
            )

    if eval_start is not None and eval_time > 0:
        events.append(
            (
                "evaluation",
                eval_start,
                eval_time,
                {"resolved": nm["resolved"]},
            )
        )

    spans = sorted(
        (start, start + duration)
        for category, start, duration, _ in events
        if category
        in (
            "agent_init",
            "agent_startup_uninstrumented",
            "llm_generation",
            "tool_execution",
        )
    )
    if spans:
        for gap_start, gap_end in gaps_within(gen_start, generation_end, spans):
            category = "agent_finalization_uninstrumented" if gap_end == generation_end else "framework_overhead"
            events.append((category, gap_start, gap_end - gap_start, {}))

    return events


# Chrome trace cname color palette
CATEGORY_COLORS = {
    "agent_rollout": "grey",  # neutral gray
    "llm_generation": "good",  # green
    "tool_execution": "vsync_highlight_color",  # blue/teal
    "evaluation": "terrible",  # dark red
    "framework_overhead": "terrible",  # dark red
    "agent_startup_uninstrumented": "grey",  # neutral gray
    "agent_finalization_uninstrumented": "grey",  # neutral gray
    "agent_init": "yellow",  # yellow
    "queue_wait": "thread_state_sleeping",  # light purple
}

CATEGORY_DISPLAY = {
    "agent_rollout": "Agent Rollout",
    "llm_generation": "LLM Generation (GPU)",
    "tool_execution": "Tool Execution (CPU)",
    "evaluation": "Evaluation (CPU)",
    "framework_overhead": "Framework Overhead",
    "agent_startup_uninstrumented": "Agent Startup (not instrumented)",
    "agent_finalization_uninstrumented": "Agent Finalization (not instrumented)",
    "agent_init": "Agent Init",
    "queue_wait": "Ray Queue Wait",
}

# Perfetto's current Chrome JSON viewer ignores cname and hashes the slice name
# for color. This non-rendering suffix maps Framework Overhead to red in both
# palettes while preserving the visible label. Legacy viewers use CATEGORY_COLORS.
PERFETTO_NAME_SUFFIX = {
    "framework_overhead": "\ufe01\ufeff",
}


def build_chrome_trace(log_dir):
    """Build Chrome Trace Event Format JSON from all rollouts.

    Args:
        log_dir: Path to swebench_results directory.

    Returns:
        dict: Chrome Trace Event Format data.
    """
    trace_events = []

    # Collect all entry directories
    entries = []
    entry_events = {}
    entry_start_times = {}
    skipped_entries = 0
    for name in sorted(os.listdir(log_dir)):
        full_path = os.path.join(log_dir, name)
        if not os.path.isdir(full_path):
            continue
        if name == "venv":
            continue
        metrics_file = os.path.join(full_path, "nemo_gym_metrics.json")
        if not os.path.exists(metrics_file):
            continue
        try:
            with open(metrics_file, "r") as f:
                data = json.load(f)
            validate_precise_metrics(data, name)
            start_time = parse_iso_timestamp(data["generation_start_timestamp"])
            events = reconstruct_rollout_events(data)
        except (OSError, KeyError, TypeError, ValueError):
            skipped_entries += 1
            continue
        entries.append((name, data))
        entry_events[name] = events
        entry_start_times[name] = min((event[1] for event in events), default=start_time)

    print(f"Processing {len(entries)} rollout entries...")
    if skipped_entries:
        print(f"Skipped {skipped_entries} incomplete rollout entries")

    # Group by instance ID
    instance_groups = defaultdict(list)
    for name, _ in entries:
        iid = extract_instance_id(name)
        instance_groups[iid].append(name)
    for group in instance_groups.values():
        group.sort(key=entry_start_times.__getitem__)

    # Assign pid per instance, sorted by earliest start time
    instance_to_pid = {}

    def _earliest_ts(iid):
        return min(entry_start_times[dirname] for dirname in instance_groups[iid])

    for i, iid in enumerate(sorted(instance_groups.keys(), key=_earliest_ts)):
        instance_to_pid[iid] = i + 1

    # Process metadata events (process names)
    for iid, pid in instance_to_pid.items():
        n_rollouts = len(instance_groups[iid])
        trace_events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "args": {"name": f"{iid} ({n_rollouts} rollouts)"},
            }
        )
        trace_events.append(
            {
                "name": "process_sort_index",
                "ph": "M",
                "pid": pid,
                "args": {"sort_index": pid},
            }
        )

    # --- Process each entry ---
    rollout_count = 0
    stats = {
        "total_fallback_rollout_time": 0.0,
        "total_llm_time": 0.0,
        "total_tool_time": 0.0,
        "total_eval_time": 0.0,
        "total_init_time": 0.0,
        "total_startup_time": 0.0,
        "total_finalization_time": 0.0,
        "total_framework_overhead_time": 0.0,
        "resolved_count": 0,
        "total_count": 0,
        "detailed_count": 0,
        "fallback_count": 0,
    }

    for dir_name, data in entries:
        iid = extract_instance_id(dir_name)
        pid = instance_to_pid[iid]
        rollout_number = instance_groups[iid].index(dir_name) + 1
        tid = rollout_number

        # Thread metadata
        hash_suffix = dir_name.rsplit("_", 2)[-1][:8]
        resolved = data["resolved"]
        status = "PASS" if resolved else "FAIL"
        gen_time = data["openhands_run_time"]
        eval_time = data["final_eval_time"] if data.get("evaluation_start_timestamp") else 0

        # Reconstruct events first to compute per-rollout sums
        events = entry_events[dir_name]
        is_detailed = has_per_turn_metrics(data)

        rollout_llm_time = sum(dur for cat, _, dur, _ in events if cat == "llm_generation")
        rollout_tool_time = sum(
            duration
            for category, _, duration, metadata in events
            if category == "tool_execution" and not metadata.get("_nested")
        )

        trace_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "args": {
                    "name": f"R{rollout_number} [{status}] gen={gen_time:.0f}s eval={eval_time:.0f}s llm={rollout_llm_time:.0f}s tool={rollout_tool_time:.0f}s ({hash_suffix})"
                },
            }
        )
        trace_events.append(
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "args": {"sort_index": tid},
            }
        )
        for cat, start_s, dur_s, meta in events:
            ts_us = to_us(start_s)
            end_us = to_us(start_s + dur_s)
            dur_us = max(0, end_us - ts_us)
            event_args = dict(meta)
            event_name = event_args.pop("_trace_name", CATEGORY_DISPLAY[cat])
            nested = event_args.pop("_nested", False)
            event = {
                "name": event_name + PERFETTO_NAME_SUFFIX.get(cat, ""),
                "cat": cat,
                "ph": "X",
                "ts": ts_us,
                "dur": dur_us,
                "pid": pid,
                "tid": tid,
                "args": event_args,
            }
            event["cname"] = CATEGORY_COLORS[cat]

            trace_events.append(event)

            # Accumulate stats
            if cat == "agent_rollout":
                stats["total_fallback_rollout_time"] += dur_s
            elif not is_detailed:
                continue
            elif cat == "llm_generation":
                stats["total_llm_time"] += dur_s
            elif cat == "tool_execution" and not nested:
                stats["total_tool_time"] += dur_s
            elif cat == "evaluation":
                stats["total_eval_time"] += dur_s
            elif cat == "agent_init":
                stats["total_init_time"] += dur_s
            elif cat == "agent_startup_uninstrumented":
                stats["total_startup_time"] += dur_s
            elif cat == "agent_finalization_uninstrumented":
                stats["total_finalization_time"] += dur_s
            elif cat == "framework_overhead" and not nested:
                stats["total_framework_overhead_time"] += dur_s

        stats["total_count"] += 1
        if is_detailed:
            stats["detailed_count"] += 1
        else:
            stats["fallback_count"] += 1
        if resolved:
            stats["resolved_count"] += 1

        rollout_count += 1
        if rollout_count % 200 == 0:
            print(f"  Processed {rollout_count}/{len(entries)} entries...")

    print(f"Processed {rollout_count} rollouts")
    print(f"Generated {len(trace_events)} trace events")

    print("\n[INFO] Category descriptions:")
    print("Agent Rollout: Full measured generation span when per-turn details are unavailable")
    print("LLM Generation (GPU): Time spent on LLM inference API calls (GPU-bound)")
    print("Tool Execution (CPU): Bash commands, file edits, etc. (CPU-bound)")
    print("Evaluation (CPU): SWE-bench test suite execution after agent completes (CPU-bound)")
    print("Agent Init: Sum of measured Apptainer spinup, runtime creation,")
    print("      runtime connection, and runtime initialization durations")
    print("Agent Startup (not instrumented): Uninstrumented interval from the end")
    print("      of measured Agent Init to the first LLM generation")
    print("Agent Finalization (not instrumented): Uninstrumented interval from the")
    print("      final measured agent event to generation process completion")
    print("Framework Overhead: Time between measured agent events that is not")
    print("      LLM generation or tool execution")

    # Print summary statistics
    print("\n--- Summary Statistics (aggregated across all rollouts) ---")
    print(f"  Total rollouts: {stats['total_count']}")
    print(f"  Detailed rollouts: {stats['detailed_count']}")
    print(f"  Fallback-only rollouts: {stats['fallback_count']}")
    print(
        f"  Resolved: {stats['resolved_count']}/{stats['total_count']} "
        f"({100 * stats['resolved_count'] / max(stats['total_count'], 1):.1f}%)"
    )
    if stats["fallback_count"] > 0:
        print(f"  Avg fallback duration:  {stats['total_fallback_rollout_time'] / stats['fallback_count']:>10.1f}s")

    detailed_total_time = (
        stats["total_llm_time"]
        + stats["total_tool_time"]
        + stats["total_eval_time"]
        + stats["total_init_time"]
        + stats["total_startup_time"]
        + stats["total_finalization_time"]
        + stats["total_framework_overhead_time"]
    )

    n = max(stats["detailed_count"], 1)
    if detailed_total_time > 0:
        avg_time = detailed_total_time / n
        print("  Detailed timing (fallback-only rollouts excluded):")
        print(f"  Avg per detailed rollout:{avg_time:>9.1f}s")
        print(
            f"  LLM Generation (GPU):   {stats['total_llm_time'] / n:>10.1f}s  "
            f"({100 * stats['total_llm_time'] / detailed_total_time:.1f}%)"
        )
        print(
            f"  Tool Execution (CPU):   {stats['total_tool_time'] / n:>10.1f}s  "
            f"({100 * stats['total_tool_time'] / detailed_total_time:.1f}%)"
        )
        print(
            f"  Evaluation (CPU):       {stats['total_eval_time'] / n:>10.1f}s  "
            f"({100 * stats['total_eval_time'] / detailed_total_time:.1f}%)"
        )
        print(
            f"  Agent Init:             {stats['total_init_time'] / n:>10.1f}s  "
            f"({100 * stats['total_init_time'] / detailed_total_time:.1f}%)"
        )
        print(
            f"  Agent Startup:          {stats['total_startup_time'] / n:>10.1f}s  "
            f"({100 * stats['total_startup_time'] / detailed_total_time:.1f}%)"
        )
        print(
            f"  Agent Finalization:     {stats['total_finalization_time'] / n:>10.1f}s  "
            f"({100 * stats['total_finalization_time'] / detailed_total_time:.1f}%)"
        )
        print(
            f"  Framework Overhead:     "
            f"{stats['total_framework_overhead_time'] / n:>10.1f}s  "
            f"({100 * stats['total_framework_overhead_time'] / detailed_total_time:.1f}%)"
        )
        cpu_core_work_time = stats["total_tool_time"] + stats["total_eval_time"]
        framework_inefficiency_time = (
            stats["total_init_time"]
            + stats["total_startup_time"]
            + stats["total_finalization_time"]
            + stats["total_framework_overhead_time"]
        )
        print("  ---")
        print(
            f"  Total CPU core work (Tool call + Eval): {cpu_core_work_time / n:>10.1f}s  "
            f"({100 * cpu_core_work_time / detailed_total_time:.1f}%)"
        )
        print(
            f"  Total Framework inefficiency: {framework_inefficiency_time / n:>10.1f}s  "
            f"({100 * framework_inefficiency_time / detailed_total_time:.1f}%)"
        )
        print(
            f"  Total GPU (LLM) time:   {stats['total_llm_time'] / n:>10.1f}s  "
            f"({100 * stats['total_llm_time'] / detailed_total_time:.1f}%)"
        )

    return {"traceEvents": trace_events}


def main():
    parser = argparse.ArgumentParser(
        description="Convert SWE-agent RL training logs to Chrome Trace Event Format "
        "for visualization in Perfetto (https://ui.perfetto.dev/)"
    )
    parser.add_argument("--log-dir", required=True, help="Path to the SWE-agent rollout directory")
    parser.add_argument("--output", required=True, help="Output Chrome trace JSON file path")
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        parser.error(f"log directory not found: {args.log_dir}")

    print(f"Log directory: {args.log_dir}")
    print(f"Output: {args.output}")

    trace = build_chrome_trace(args.log_dir)

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f"\nWriting trace to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(trace, f)

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Trace written: {args.output} ({file_size_mb:.1f} MB)")
    print("\nOpen in https://ui.perfetto.dev/ to visualize the timeline.")


if __name__ == "__main__":
    main()
