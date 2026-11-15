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


def parse_iso_timestamp(ts_str):
    """Parse ISO timestamp string to epoch seconds (float).
    Rollout timestamps are UTC; OpenHands action timestamps may omit the offset.
    """
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


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
            latency = record.get("latency")
            if not isinstance(latency, (int, float)) or latency < 0:
                errors.append(f"{label} {index} latency is invalid")

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

    llm_starts = []
    for turn, response in enumerate(responses, start=1):
        timestamp = response["timestamp"]
        end = parse_iso_timestamp(timestamp)
        latency = response["latency"]
        start = end - latency
        response_id = response["response_id"]
        token_usage = token_by_rid[response_id]
        metadata = {
            "response_id": response_id,
            "recorded_latency": latency,
            "turn": turn,
        }
        metadata["prompt_tokens"] = token_usage["prompt_tokens"]
        metadata["completion_tokens"] = token_usage["completion_tokens"]
        events.append(("llm_generation", start, latency, metadata))
        llm_starts.append(start)

    for action in actions:
        timestamp = action["timestamp"]
        end = parse_iso_timestamp(timestamp)
        latency = action["latency"]
        events.append(
            (
                "tool_execution",
                end - latency,
                latency,
                {
                    "observation_type": action["observation_type"],
                    "observation_id": action["observation_id"],
                    "message": action["message"],
                },
            )
        )

    if ray_queue_time > 0:
        events.append(("queue_wait", gen_start - ray_queue_time, ray_queue_time, {}))

    init_components = {field: nm[field] for field in AGENT_INIT_METRICS}
    init_duration = sum(init_components.values())
    if init_duration > 0:
        events.append(("agent_init", gen_start, init_duration, init_components))

    # Agent Startup (not instrumented) is the exact interval after the measured
    # container/runtime init phases finish and before the first LLM request
    # begins. OpenHands emits no finer-grained timestamps inside this interval,
    # so keep it separate from both Agent Init and Framework Overhead.
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
        previous_end = gen_start
        for next_start, next_end in spans:
            gap = next_start - previous_end
            if gap > 0:
                events.append(("framework_overhead", previous_end, gap, {}))
            previous_end = max(previous_end, next_end)

        generation_end = eval_start if eval_start is not None else gen_start + nm["openhands_run_time"]
        trailing_gap = generation_end - previous_end
        if trailing_gap > 0:
            events.append(("framework_overhead", previous_end, trailing_gap, {}))

    return events


# Chrome trace cname color palette
CATEGORY_COLORS = {
    "agent_rollout": "grey",  # neutral gray
    "llm_generation": "good",  # green
    "tool_execution": "vsync_highlight_color",  # blue/teal
    "evaluation": "terrible",  # dark red
    "framework_overhead": "terrible",  # dark red
    "agent_startup_uninstrumented": "grey",  # neutral gray
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
        "total_agent_rollout_time": 0.0,
        "total_llm_time": 0.0,
        "total_tool_time": 0.0,
        "total_eval_time": 0.0,
        "total_init_time": 0.0,
        "total_startup_time": 0.0,
        "total_framework_overhead_time": 0.0,
        "resolved_count": 0,
        "total_count": 0,
    }

    for dir_name, data in entries:
        iid = extract_instance_id(dir_name)
        pid = instance_to_pid[iid]
        # tid = rollout index within this instance (1-based)
        tid = instance_groups[iid].index(dir_name) + 1

        # Thread metadata
        hash_suffix = dir_name.rsplit("_", 2)[-1][:8]
        resolved = data["resolved"]
        status = "PASS" if resolved else "FAIL"
        gen_time = data["openhands_run_time"]
        eval_time = data["final_eval_time"] if data.get("evaluation_start_timestamp") else 0

        # Reconstruct events first to compute per-rollout sums
        events = reconstruct_rollout_events(data)

        rollout_llm_time = sum(dur for cat, _, dur, _ in events if cat == "llm_generation")
        rollout_tool_time = sum(dur for cat, _, dur, _ in events if cat == "tool_execution")

        trace_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "args": {
                    "name": f"R{tid} [{status}] gen={gen_time:.0f}s eval={eval_time:.0f}s llm={rollout_llm_time:.0f}s tool={rollout_tool_time:.0f}s ({hash_suffix})"
                },
            }
        )

        for cat, start_s, dur_s, meta in events:
            ts_us = to_us(start_s)
            end_us = to_us(start_s + dur_s)
            dur_us = max(0, end_us - ts_us)

            event = {
                "name": CATEGORY_DISPLAY[cat] + PERFETTO_NAME_SUFFIX.get(cat, ""),
                "cat": cat,
                "ph": "X",
                "ts": ts_us,
                "dur": dur_us,
                "pid": pid,
                "tid": tid,
                "args": meta,
            }
            event["cname"] = CATEGORY_COLORS[cat]

            trace_events.append(event)

            # Accumulate stats
            if cat == "agent_rollout":
                stats["total_agent_rollout_time"] += dur_s
            elif cat == "llm_generation":
                stats["total_llm_time"] += dur_s
            elif cat == "tool_execution":
                stats["total_tool_time"] += dur_s
            elif cat == "evaluation":
                stats["total_eval_time"] += dur_s
            elif cat == "agent_init":
                stats["total_init_time"] += dur_s
            elif cat == "agent_startup_uninstrumented":
                stats["total_startup_time"] += dur_s
            elif cat == "framework_overhead":
                stats["total_framework_overhead_time"] += dur_s

        stats["total_count"] += 1
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
    print("Framework Overhead: Time between measured agent events that is not")
    print("      LLM generation or tool execution")

    # Print summary statistics
    print("\n--- Summary Statistics (aggregated across all rollouts) ---")
    print(f"  Total rollouts: {stats['total_count']}")
    print(
        f"  Resolved: {stats['resolved_count']}/{stats['total_count']} "
        f"({100 * stats['resolved_count'] / max(stats['total_count'], 1):.1f}%)"
    )
    total_time = (
        stats["total_agent_rollout_time"]
        + stats["total_llm_time"]
        + stats["total_tool_time"]
        + stats["total_eval_time"]
        + stats["total_init_time"]
        + stats["total_startup_time"]
        + stats["total_framework_overhead_time"]
    )

    n = max(stats["total_count"], 1)
    if total_time > 0:
        avg_time = total_time / n
        print(f"  Avg per rollout:        {avg_time:>10.1f}s")
        if stats["total_agent_rollout_time"] > 0:
            print(
                f"  Agent Rollout:          {stats['total_agent_rollout_time'] / n:>10.1f}s  "
                f"({100 * stats['total_agent_rollout_time'] / total_time:.1f}%)"
            )
        print(
            f"  LLM Generation (GPU):   {stats['total_llm_time'] / n:>10.1f}s  "
            f"({100 * stats['total_llm_time'] / total_time:.1f}%)"
        )
        print(
            f"  Tool Execution (CPU):   {stats['total_tool_time'] / n:>10.1f}s  "
            f"({100 * stats['total_tool_time'] / total_time:.1f}%)"
        )
        print(
            f"  Evaluation (CPU):       {stats['total_eval_time'] / n:>10.1f}s  "
            f"({100 * stats['total_eval_time'] / total_time:.1f}%)"
        )
        print(
            f"  Agent Init:             {stats['total_init_time'] / n:>10.1f}s  "
            f"({100 * stats['total_init_time'] / total_time:.1f}%)"
        )
        print(
            f"  Agent Startup:          {stats['total_startup_time'] / n:>10.1f}s  "
            f"({100 * stats['total_startup_time'] / total_time:.1f}%)"
        )
        print(
            f"  Framework Overhead:     "
            f"{stats['total_framework_overhead_time'] / n:>10.1f}s  "
            f"({100 * stats['total_framework_overhead_time'] / total_time:.1f}%)"
        )
        cpu_time = stats["total_tool_time"] + stats["total_eval_time"]
        print("  ---")
        print(f"  Total CPU overhead:     {cpu_time / n:>10.1f}s  ({100 * cpu_time / total_time:.1f}%)")
        print(
            f"  Total GPU (LLM) time:   {stats['total_llm_time'] / n:>10.1f}s  "
            f"({100 * stats['total_llm_time'] / total_time:.1f}%)"
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
