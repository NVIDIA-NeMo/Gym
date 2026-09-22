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
"""Resource-efficiency / Pareto analysis: joins experiment telemetry (rollout JSONL --
reward, tokens, cost) with system telemetry (OTel traces -- CPU-seconds, latency) by
rollout id, so questions like "CPU-seconds per successful task" or "latency vs accuracy"
can be answered without a live dashboard query.

Lives here, not inside ``nemo_gym.telemetry``, on purpose -- it reads reward/accuracy
(experiment telemetry, which by design never enters OTel, see
``nemo_gym.telemetry.README.md``) and joins it against trace data pulled from a Tempo
backend after the fact. That is a fundamentally different shape of code from the
inline, span-group-gated instrumentation the ``telemetry`` package holds, and living next
to ``reward_profile.py`` (which owns the experiment-telemetry side of the same rollout
JSONL) keeps the application/experiment split the rest of the codebase already maintains.

Why a Tempo trace query and not a Prometheus metric query: Gym's OTel *metrics*
deliberately never carry ``rollout_id`` as an attribute (`nemo_gym.telemetry.endpoints`'s
module docstring on Gap-1 correlation -- "unique IDs on traces and trajectory records,
not metric labels, to avoid excessive cardinality"). Only *spans* carry it
(``nemo.gym.rollout.id``). So a per-rollout CPU-seconds figure has to come from summing
span-attached CPU readings across one rollout's trace, not from a Prometheus histogram.

GPU-seconds per task is computed the same way as CPU-seconds, from a
``nemo.gym.gpu.utilization_percent`` span attribute -- but it is a coarser
approximation than the CPU figure, worth knowing before trusting it: GPU readings come
from a background thread polling ``nvidia-smi`` every ``gpu_sample_interval_s`` (10s by
default), not a fresh read at each span's close the way CPU/memory are. A span attaches
whatever the sampler's *last* tick produced, so a rollout much shorter than that interval
may see a stale or entirely absent reading, and on a node sharing GPUs across more than
one process, the figure is not exclusively "this rollout's" usage -- see
`nemo_gym.telemetry.gpu.last_gpu_utilization_percent` for the full caveat.

Example usage:

    python -m nemo_gym.pareto_analysis \\
        --input results/swebench_verified/rollouts.jsonl \\
        --tempo-url http://localhost:3200 \\
        --output-csv results/swebench_verified/efficiency.csv
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

import orjson
from pandas import DataFrame

from nemo_gym.global_config import AGENT_REF_KEY_NAME


logger = logging.getLogger(__name__)

#: Trace tag Tempo indexes rollout id under -- the same span attribute
#: `nemo_gym.telemetry.endpoints.ROLLOUT_ID_ATTRIBUTE` sets.
ROLLOUT_ID_TRACE_TAG = "nemo.gym.rollout.id"

#: Span attributes carrying a CPU reading, preferring the process-tree figure (the
#: "actual job workload") over the Gym-PID-only one when both are present on a span.
_CPU_PERCENT_SPAN_ATTRS = ("nemo.gym.process_tree.cpu.percent", "nemo.gym.cpu.percent")

#: Span attribute carrying a GPU reading -- see `nemo_gym.telemetry.gpu` for why there is
#: only one, host-shared-tenant, source for this (unlike CPU's PID-vs-tree choice).
_GPU_PERCENT_SPAN_ATTR = "nemo.gym.gpu.utilization_percent"


def _extract_rollout_summary(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Pull the experiment-telemetry fields this analysis needs out of one rollout's
    JSONL row. Returns ``None`` for a row with no rollout id at all -- there is nothing
    to join it against on the trace side."""
    trajectory = row.get("ng_trajectory") or {}
    rollout_id = trajectory.get("rollout_id") or row.get("_ng_rollout_id")
    if not rollout_id:
        return None
    ng_perf = row.get("ng_perf") or {}
    return {
        "rollout_id": rollout_id,
        "run_id": trajectory.get("run_id"),
        "benchmark": trajectory.get("benchmark"),
        "agent_name": (row.get(AGENT_REF_KEY_NAME) or {}).get("name"),
        "reward": row.get("reward"),
        "num_tool_calls": ng_perf.get("num_tool_calls"),
        "num_turns": ng_perf.get("num_turns"),
        "latency_ms": ng_perf.get("total_latency_ms"),
        "prompt_tokens": ng_perf.get("prompt_tokens"),
        "completion_tokens": ng_perf.get("completion_tokens"),
    }


def load_rollout_summaries(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """Read a rollout-collection output JSONL and extract one summary row per rollout
    that carries a joinable rollout id. Rows a benchmark's verifier failed to score, or
    that predate Gap-1's correlation work (no ``ng_trajectory.rollout_id``), are skipped
    rather than raising -- a partial join is still useful."""
    summaries: list[dict[str, Any]] = []
    with open(jsonl_path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in %s", jsonl_path)
                continue
            summary = _extract_rollout_summary(row)
            if summary is not None:
                summaries.append(summary)
    return summaries


def _span_attribute_value(attributes: list[dict[str, Any]], key: str) -> Optional[float]:
    """Decode one OTLP-JSON span attribute's numeric value, or ``None`` if absent."""
    for attribute in attributes:
        if attribute.get("key") != key:
            continue
        value = attribute.get("value", {})
        for numeric_field in ("doubleValue", "intValue"):
            if numeric_field in value:
                return float(value[numeric_field])
        return None
    return None


def _resource_seconds_from_trace(trace: dict[str, Any]) -> dict[str, Optional[float]]:
    """Sum ``reading / 100 * span_duration_s`` across every span in one OTLP-JSON trace
    that carries a CPU or GPU reading -- an integral approximation of resource-seconds
    consumed during that rollout's trace, not an exact accounting (a span's reading is
    one instantaneous sample taken at span-close, not a continuous measurement over the
    span's lifetime; see `nemo_gym.telemetry.cpu`'s module docstring on why sampling is
    inline-at-span-boundary rather than continuous, and `nemo_gym.telemetry.gpu`'s for
    why the GPU figure is coarser still). Either key is ``None`` if no span in this trace
    carried that reading."""
    totals = {"cpu_seconds": 0.0, "gpu_seconds": 0.0}
    found = {"cpu_seconds": False, "gpu_seconds": False}
    for batch in trace.get("batches", []):
        for scope_spans in batch.get("scopeSpans", []):
            for span in scope_spans.get("spans", []):
                attributes = span.get("attributes", [])
                start_ns = int(span.get("startTimeUnixNano", 0))
                end_ns = int(span.get("endTimeUnixNano", 0))
                duration_s = max(0.0, (end_ns - start_ns) / 1_000_000_000.0)

                cpu_percent = None
                for attr_name in _CPU_PERCENT_SPAN_ATTRS:
                    cpu_percent = _span_attribute_value(attributes, attr_name)
                    if cpu_percent is not None:
                        break
                if cpu_percent is not None:
                    totals["cpu_seconds"] += (cpu_percent / 100.0) * duration_s
                    found["cpu_seconds"] = True

                gpu_percent = _span_attribute_value(attributes, _GPU_PERCENT_SPAN_ATTR)
                if gpu_percent is not None:
                    totals["gpu_seconds"] += (gpu_percent / 100.0) * duration_s
                    found["gpu_seconds"] = True
    return {key: (totals[key] if found[key] else None) for key in totals}


async def _fetch_resource_seconds(session: Any, tempo_url: str, rollout_id: str) -> dict[str, Optional[float]]:
    """Look up one rollout's trace(s) by the ``nemo.gym.rollout.id`` tag and sum
    CPU-seconds/GPU-seconds across them. Both values are ``None`` when Tempo has no
    trace for this rollout id (e.g. tracing was disabled for that run, or the export
    window has not flushed yet -- not necessarily an error)."""
    search_url = f"{tempo_url.rstrip('/')}/api/search"
    params = {"tags": f"{ROLLOUT_ID_TRACE_TAG}={rollout_id}", "limit": "20"}
    async with session.get(search_url, params=params) as response:
        if response.status != 200:
            logger.debug("Tempo search failed for rollout_id=%s: status=%s", rollout_id, response.status)
            return {"cpu_seconds": None, "gpu_seconds": None}
        search_result = await response.json()

    trace_ids = [t["traceID"] for t in search_result.get("traces", []) if t.get("traceID")]
    if not trace_ids:
        return {"cpu_seconds": None, "gpu_seconds": None}

    totals = {"cpu_seconds": 0.0, "gpu_seconds": 0.0}
    found = {"cpu_seconds": False, "gpu_seconds": False}
    for trace_id in trace_ids:
        async with session.get(f"{tempo_url.rstrip('/')}/api/traces/{trace_id}") as response:
            if response.status != 200:
                continue
            trace = await response.json()
        per_trace = _resource_seconds_from_trace(trace)
        for key, value in per_trace.items():
            if value is not None:
                totals[key] += value
                found[key] = True
    return {key: (totals[key] if found[key] else None) for key in totals}


async def _join_cpu_seconds(
    summaries: list[dict[str, Any]], tempo_url: str, *, concurrency: int = 8
) -> list[dict[str, Any]]:
    """Attach ``cpu_seconds``/``gpu_seconds`` fields to each summary by querying Tempo,
    bounded by ``concurrency`` concurrent requests so a large rollout file does not open
    hundreds of connections against Tempo at once."""
    import aiohttp

    semaphore = asyncio.Semaphore(concurrency)

    async def _one(session: Any, summary: dict[str, Any]) -> None:
        async with semaphore:
            summary.update(await _fetch_resource_seconds(session, tempo_url, summary["rollout_id"]))

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(_one(session, summary) for summary in summaries))
    return summaries


def compute_efficiency_table(jsonl_path: str | Path, tempo_url: str) -> DataFrame:
    """The main entry point: read a rollout JSONL, join each row's CPU-seconds and
    GPU-seconds from Tempo, return one :class:`pandas.DataFrame` row per rollout with
    reward, latency, token usage, and both resource figures side by side."""
    summaries = load_rollout_summaries(jsonl_path)
    if not summaries:
        return DataFrame(
            columns=[
                "rollout_id",
                "run_id",
                "benchmark",
                "agent_name",
                "reward",
                "num_tool_calls",
                "num_turns",
                "latency_ms",
                "prompt_tokens",
                "completion_tokens",
                "cpu_seconds",
                "gpu_seconds",
            ]
        )
    summaries = asyncio.run(_join_cpu_seconds(summaries, tempo_url))
    return DataFrame(summaries)


def tool_calls_per_successful_task(table: DataFrame, *, success_threshold: float = 1.0) -> Optional[float]:
    """Mean ``num_tool_calls`` across rollouts whose ``reward >= success_threshold``.
    Pure JSONL, no Tempo join needed -- both fields already live in the same row."""
    successful = table[table["reward"] >= success_threshold]
    if successful.empty or "num_tool_calls" not in successful:
        return None
    values = successful["num_tool_calls"].dropna()
    return float(values.mean()) if not values.empty else None


def pareto_efficient_configs(
    table: DataFrame, *, group_by: str = "agent_name", quality_col: str = "reward", cost_col: str = "cpu_seconds"
) -> DataFrame:
    """Group rows by ``group_by`` (one row per configuration, e.g. agent/model variant),
    compute mean quality and mean cost per group, and flag which groups are
    Pareto-efficient: no other group achieves both higher quality AND lower cost.

    This is the "did the extra compute actually help" question from the MVP doc, reduced
    to its simplest form -- one point per configuration, not per rollout. A configuration
    with no `cost_col` data (e.g. Tempo had no trace for any of its rollouts) is dropped
    rather than silently treated as free.
    """
    grouped = (
        table.dropna(subset=[quality_col, cost_col])
        .groupby(group_by)
        .agg(mean_quality=(quality_col, "mean"), mean_cost=(cost_col, "mean"), n_rollouts=(quality_col, "count"))
        .reset_index()
    )

    def _is_efficient(row) -> bool:
        dominated_by = grouped[
            (grouped["mean_quality"] >= row["mean_quality"])
            & (grouped["mean_cost"] <= row["mean_cost"])
            & ((grouped["mean_quality"] > row["mean_quality"]) | (grouped["mean_cost"] < row["mean_cost"]))
        ]
        return dominated_by.empty

    grouped["pareto_efficient"] = grouped.apply(_is_efficient, axis=1)
    return grouped.sort_values("mean_cost")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Rollout-collection output JSONL path.")
    parser.add_argument("--tempo-url", required=True, help="Tempo base URL, e.g. http://localhost:3200.")
    parser.add_argument("--output-csv", default=None, help="Optional path to write the per-rollout table as CSV.")
    parser.add_argument(
        "--group-by",
        default="agent_name",
        help="Column identifying a 'configuration' for the Pareto comparison (default: agent_name).",
    )
    parser.add_argument(
        "--cost-col",
        default="cpu_seconds",
        choices=["cpu_seconds", "gpu_seconds"],
        help="Resource axis for the Pareto comparison (default: cpu_seconds). "
        "gpu_seconds is a coarser, up-to-gpu_sample_interval_s-stale approximation -- see the module docstring.",
    )
    args = parser.parse_args()

    table = compute_efficiency_table(args.input, args.tempo_url)
    print(f"Loaded {len(table)} rollouts with a joinable rollout id.")
    for col in ("cpu_seconds", "gpu_seconds"):
        joined = table[col].notna().sum() if col in table else 0
        print(f"Found a Tempo trace with {col} data for {joined}/{len(table)} rollouts.")

    if args.output_csv:
        table.to_csv(args.output_csv, index=False)
        print(f"Wrote per-rollout table to {args.output_csv}")

    mean_tool_calls = tool_calls_per_successful_task(table)
    if mean_tool_calls is not None:
        print(f"Mean tool calls per successful task (reward >= 1.0): {mean_tool_calls:.2f}")

    joined = table[args.cost_col].notna().sum() if args.cost_col in table else 0
    if joined:
        efficiency = pareto_efficient_configs(table, group_by=args.group_by, cost_col=args.cost_col)
        print(f"\nPareto comparison by {args.group_by} (accuracy vs. {args.cost_col}):")
        print(efficiency.to_string(index=False))
    else:
        print(
            f"\nNo Tempo {args.cost_col} data joined — skipping the Pareto comparison. "
            "Check that --tempo-url is reachable and that the run had telemetry.span_groups "
            "including 'sandbox'/'rollout' with cpu_sampling_enabled=true (or gpu_sampling_enabled=true)."
        )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _main()
