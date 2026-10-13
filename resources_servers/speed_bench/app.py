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
"""SPEED-Bench speculative decoding resources server.

Ported from
https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/eval/specdec.py.

Speculative-decoding throughput is **not** a per-task correctness metric. It's a
property of the model server's Prometheus counters during the benchmark window.
This server reproduces Skills' approach:

1. Snapshot the vLLM server's `vllm:spec_decode_*` counters at the start of the
   benchmark window (lazily on the first `verify()` call).
2. On each `verify()`, snapshot the counters again, compute the running delta
   vs the start, and stamp the running aggregate (acceptance length, acceptance
   rate, per-position rates) onto that task's response.
3. In `compute_metrics()`, take the latest (largest-delta) running aggregate
   across all tasks as the headline `spec_acceptance_length` /
   `spec_acceptance_rate`. This matches Skills' final value, since Skills also
   computes a single before/after delta over the whole run.

SGLang is supported via Prometheus delta on
`sglang:spec_accept_length` / `sglang:spec_accept_rate` (running-average
gauges) combined with `sglang:num_requests_total` /
`sglang:generation_tokens_total` counters; the benchmark-only average is
recovered with a weighted delta. Skills also supports an SGLang per-request
metrics-file fallback (`--export-metrics-to-file`) — that path is not
ported here; the Prometheus delta is sufficient for parity comparisons.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import request as global_request


LOG = logging.getLogger(__name__)


@dataclass
class SpecDecodeMetricsSnapshot:
    """Cumulative spec-decode counters scraped from `/metrics`.

    vLLM exposes raw counters (drafts, draft_tokens, accepted_tokens, plus
    per-position breakdowns); SGLang exposes acceptance length and rate as
    *running-average gauges* alongside request and generation_tokens
    counters. Both worlds populate this struct; consumers downstream pick
    the fields relevant to the backend they're scraping.
    """

    # vLLM counters
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    accepted_per_pos: Dict[int, int] = field(default_factory=dict)

    # SGLang gauges + counters
    spec_accept_length: float = 0.0
    spec_accept_rate: float = 0.0
    num_requests: int = 0
    generation_tokens: int = 0


def _parse_vllm_metrics(text: str) -> SpecDecodeMetricsSnapshot:
    """Parse vLLM's Prometheus text exposition for `vllm:spec_decode_*`.

    Mirrors the line-by-line parsing in
    `nemo_skills/inference/eval/specdec.py::fetch_vllm_spec_decode_metrics`.
    """
    snapshot = SpecDecodeMetricsSnapshot()
    found_spec_decode = False
    pos_label = 'position="'

    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("vllm:spec_decode"):
            continue

        found_spec_decode = True
        if "_created" in line:
            continue

        parts = line.split()
        if not parts:
            continue

        with contextlib.suppress(ValueError):
            value = int(float(parts[-1]))
            if "num_drafts" in line:
                snapshot.num_drafts += value
            elif "num_draft_tokens" in line:
                snapshot.num_draft_tokens += value
            elif "num_accepted_tokens_per_pos" in line:
                if pos_label in line:
                    start = line.index(pos_label) + len(pos_label)
                    end = line.index('"', start)
                    pos = int(line[start:end])
                    snapshot.accepted_per_pos[pos] = snapshot.accepted_per_pos.get(pos, 0) + value
            elif "num_accepted_tokens" in line:
                snapshot.num_accepted_tokens += value

    if not found_spec_decode:
        # Distinguishable from "spec decode disabled but server is up" — the
        # caller can decide whether to treat as fatal or just zero out metrics.
        raise SpecDecodeMetricsUnavailable(
            "No vllm:spec_decode_* metrics found on the server (speculative decoding may not be enabled)."
        )

    return snapshot


def _parse_sglang_metrics(text: str) -> SpecDecodeMetricsSnapshot:
    """Parse SGLang's Prometheus exposition for `sglang:spec_accept_*` + counters.

    Mirrors `fetch_sglang_spec_decode_metrics` in Skills' specdec.py. SGLang
    treats acceptance length / rate as *running-average gauges* — combined
    with `num_requests_total` and `generation_tokens_total` counters we can
    back out the benchmark-only average via a weighted delta (see
    `_compute_sglang_running_delta`).
    """
    snapshot = SpecDecodeMetricsSnapshot()
    found_spec = False

    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue

        with contextlib.suppress(ValueError):
            if "sglang:spec_accept_length{" in line or line.startswith("sglang:spec_accept_length "):
                snapshot.spec_accept_length = float(parts[-1])
                found_spec = True
            elif "sglang:spec_accept_rate{" in line or line.startswith("sglang:spec_accept_rate "):
                snapshot.spec_accept_rate = float(parts[-1])
                found_spec = True
            elif "sglang:num_requests_total{" in line or line.startswith("sglang:num_requests_total "):
                snapshot.num_requests = int(float(parts[-1]))
            elif "sglang:generation_tokens_total{" in line or line.startswith("sglang:generation_tokens_total "):
                snapshot.generation_tokens = int(float(parts[-1]))

    if not found_spec:
        raise SpecDecodeMetricsUnavailable(
            "No sglang:spec_accept_* metrics found on the server (speculative decoding may not be enabled)."
        )
    return snapshot


def _weighted_delta_average(
    before_avg: float, after_avg: float, before_count: int, after_count: int
) -> Optional[float]:
    """Recover the benchmark-only average from two running-average gauges.

        weighted_after  = after_avg  * after_count
        weighted_before = before_avg * before_count
        benchmark_avg   = (weighted_after - weighted_before) / (after_count - before_count)

    Returns None if the request counter didn't advance (no benchmark traffic).
    """
    delta_count = after_count - before_count
    if delta_count <= 0:
        return None
    if before_count == 0:
        return after_avg
    return (after_avg * after_count - before_avg * before_count) / delta_count


def _compute_sglang_running_delta(
    before: SpecDecodeMetricsSnapshot, after: SpecDecodeMetricsSnapshot
) -> Optional[Dict[str, Any]]:
    """SGLang equivalent of `_compute_running_delta`.

    Mirrors Skills' `compute_sglang_spec_decode_delta`. Uses weighted-delta
    arithmetic on the running-average gauges to recover benchmark-window
    acceptance length and rate.
    """
    delta_requests = after.num_requests - before.num_requests
    delta_gen_tokens = after.generation_tokens - before.generation_tokens
    if delta_requests <= 0:
        return None

    al = _weighted_delta_average(
        before.spec_accept_length, after.spec_accept_length, before.num_requests, after.num_requests
    )
    ar_fraction = _weighted_delta_average(
        before.spec_accept_rate, after.spec_accept_rate, before.num_requests, after.num_requests
    )
    if al is None or ar_fraction is None:
        return None

    return {
        # SGLang doesn't expose draft counts directly — Skills approximates
        # `num_drafts ≈ delta_requests` (one round of drafting per request)
        # and `accepted_tokens ≈ delta_gen_tokens * ar_fraction`. Per-position
        # rates aren't available from SGLang's gauges.
        "num_drafts": delta_requests,
        "draft_tokens": delta_gen_tokens,
        "accepted_tokens": int(delta_gen_tokens * ar_fraction) if delta_gen_tokens > 0 else 0,
        "acceptance_rate": ar_fraction * 100,
        "acceptance_length": al,
        "per_position_acceptance_rates": [],
    }


def _compute_running_delta(
    before: SpecDecodeMetricsSnapshot, after: SpecDecodeMetricsSnapshot
) -> Optional[Dict[str, Any]]:
    """Compute Skills-equivalent acceptance metrics from before/after counter snapshots.

    Mirrors `compute_vllm_spec_decode_delta` in Skills.
    Returns `None` if no spec-decode activity has happened yet (delta ≤ 0).
    """
    delta_drafts = after.num_drafts - before.num_drafts
    delta_draft_tokens = after.num_draft_tokens - before.num_draft_tokens
    delta_accepted = after.num_accepted_tokens - before.num_accepted_tokens

    per_pos_rates: List[float] = []
    if delta_drafts > 0:
        positions = sorted(set(before.accepted_per_pos.keys()) | set(after.accepted_per_pos.keys()))
        for pos in positions:
            delta_pos = after.accepted_per_pos.get(pos, 0) - before.accepted_per_pos.get(pos, 0)
            per_pos_rates.append(delta_pos / delta_drafts)

    if delta_draft_tokens <= 0:
        return None

    acceptance_rate = (delta_accepted / delta_draft_tokens) * 100
    acceptance_length = 1 + delta_accepted / delta_drafts if delta_drafts > 0 else 0.0

    return {
        "num_drafts": delta_drafts,
        "draft_tokens": delta_draft_tokens,
        "accepted_tokens": delta_accepted,
        "acceptance_rate": acceptance_rate,
        "acceptance_length": acceptance_length,
        "per_position_acceptance_rates": per_pos_rates,
    }


class SpecDecodeMetricsUnavailable(RuntimeError):
    """Raised when the server's `/metrics` endpoint has no `vllm:spec_decode_*` lines."""


class SpeedBenchResourcesServerConfig(BaseResourcesServerConfig):
    """Config for the SPEED-Bench resources server.

    Attributes:
        vllm_metrics_url: Full URL to the model server's Prometheus `/metrics`
            endpoint (e.g. `http://hostname:8000/metrics`). When set, this
            takes precedence. Default `None`, in which case the server uses
            `vllm_base_url` to derive `<base>/metrics`.
        vllm_base_url: The model server's OpenAI-compatible base URL
            (e.g. `http://hostname:8000/v1`). The `/v1` suffix is stripped to
            derive the metrics URL. Defaulted from Hydra's `${policy_base_url}`
            interpolation in `configs/speed_bench.yaml`.
        server_type_for_metrics: `vllm` (default) or `sglang`. SGLang scrape is
            currently a stub and will raise NotImplementedError.
        metrics_request_timeout_s: Per-request timeout for `/metrics` scrapes.
        snapshot_at_init: If True, take the "before" snapshot at server init
            (matches a freshly-started vLLM server with no prior traffic). If
            False (default), take the snapshot lazily on the first `verify()`
            call to better isolate the benchmark window when the server has
            been used for warmup. Skills' approach is closer to the False path
            (it snapshots in `wait_for_server` after the server is up but
            before any benchmark traffic).
    """

    vllm_metrics_url: Optional[str] = None
    vllm_base_url: Optional[str] = None
    server_type_for_metrics: Literal["vllm", "sglang"] = "vllm"
    metrics_request_timeout_s: float = 30.0
    snapshot_at_init: bool = False


class SpeedBenchVerifyRequest(BaseVerifyRequest):
    """Speed-bench verify request.

    `verifier_metadata` carries the per-row fields that prepare.py emits
    (`src_id`, `source`, `speed_config`, `num_turns`, `sub_category`).
    Pydantic strips unknown top-level fields by default, so we declare it
    explicitly here to keep it in the rollout JSONL output — the
    cross-pipeline diff in `debug_compare_specdec.py` matches Skills↔Gym
    rollouts on `verifier_metadata.src_id`. Speed-bench has no
    expected_answer / correctness; the only "verification" we do is
    record per-task token counts.
    """

    verifier_metadata: Optional[Dict[str, Any]] = None


class SpeedBenchVerifyResponse(BaseVerifyResponse):
    """Per-task verify response.

    Mirrors the per-row payload Skills' `eval_specdec` stamps onto its
    output JSONL. Spec-decode fields are *running* aggregates — the values at
    the moment this task's verify() ran, computed against the start-of-window
    snapshot. Headline aggregate is the max running-delta across all tasks
    (computed in compute_metrics).

    Attributes:
        num_generated_tokens: Total tokens emitted by the model for this task
            (sum across multi-turn replies).
        gen_seconds: Wall-clock seconds elapsed in the model server between
            session start and this task's verify(). Skills' equivalent is the
            generation step's duration; we approximate by recording the
            elapsed time since the resources server first scraped /metrics.
        acceptance_length, acceptance_rate, num_drafts, draft_tokens,
        accepted_tokens, per_position_acceptance_rates: spec-decode running
            aggregates. None if scrape failed or counters are still zero.
        spec_decode_unavailable: True if the server reports no spec-decode
            counters at all (e.g. spec decoding disabled on the server).
    """

    num_generated_tokens: int = 0
    gen_seconds: float = 0.0
    acceptance_length: Optional[float] = None
    acceptance_rate: Optional[float] = None
    num_drafts: Optional[int] = None
    draft_tokens: Optional[int] = None
    accepted_tokens: Optional[int] = None
    per_position_acceptance_rates: Optional[List[float]] = None
    spec_decode_unavailable: bool = False


# Score keys we compute multi-seed variance over. Mirrors Skills'
# `SpecdecMetrics._get_score_dict()` exactly.
_SPEC_SCORE_KEYS = (
    "acceptance_length",
    "acceptance_rate",
    "num_drafts",
    "draft_tokens",
    "accepted_tokens",
)


def _compute_std_metrics(tasks: List[List[Dict[str, Any]]], score_keys: tuple) -> Dict[str, Any]:
    """Multi-seed variance metrics — mirrors Skills' `_add_std_metrics`.

    For each score key, with `k = max_k` rollouts/task:

    - `spec_<key>_avg` — overall mean across all (task, rollout) pairs.
    - `spec_<key>_std_dev_across_runs` — std-dev of per-run averages,
      where run i is "take rollout i from each task and average".
    - `spec_<key>_std_err_across_runs` — `std_dev_across_runs / sqrt(k)`.
    - `spec_<key>_avg_sample_std_dev` — mean of per-task std-devs across
      that task's k rollouts.

    Skipped entirely when every task has only 1 rollout (max_k == 1) or
    when score values are missing (all None). Tasks with mismatched
    rollout counts are ignored — uses the minimum k present so the matrix
    is rectangular (Skills enforces a strict equality; we relax for
    robustness against partial outputs).
    """
    if not tasks:
        return {}
    max_k = min((len(t) for t in tasks if t), default=0)
    if max_k < 2:
        return {}

    out: Dict[str, Any] = {}
    for key in score_keys:
        # Build a rectangular [num_tasks × max_k] matrix of float values.
        # Drop tasks whose rollouts don't all have a numeric value for this
        # score (running aggregates can be None on the very first task).
        matrix: List[List[float]] = []
        for task_rollouts in tasks:
            if len(task_rollouts) < max_k:
                continue
            row = [task_rollouts[i].get(key) for i in range(max_k)]
            if any(v is None for v in row):
                continue
            matrix.append([float(v) for v in row])

        if not matrix:
            continue

        # avg across all (task, rollout) pairs.
        all_values = [v for row in matrix for v in row]
        out[f"spec_{key}_avg"] = sum(all_values) / len(all_values)

        # std_dev_across_runs: transpose → run i averages → std-dev.
        run_averages = [sum(row[i] for row in matrix) / len(matrix) for i in range(max_k)]
        if len(run_averages) >= 2 and not all(v == run_averages[0] for v in run_averages):
            std_dev_across_runs = statistics.stdev(run_averages)
        else:
            # All runs identical (or only one) → variance is 0 (Skills' np.std with ddof=1
            # returns NaN for n=1; we report 0 so downstream JSON serialization stays clean).
            std_dev_across_runs = 0.0
        out[f"spec_{key}_std_dev_across_runs"] = std_dev_across_runs
        out[f"spec_{key}_std_err_across_runs"] = std_dev_across_runs / math.sqrt(max_k)

        # avg_sample_std_dev: per-task std-dev across rollouts → mean.
        per_sample_std = []
        for row in matrix:
            if len(row) >= 2 and not all(v == row[0] for v in row):
                per_sample_std.append(statistics.stdev(row))
            else:
                per_sample_std.append(0.0)
        out[f"spec_{key}_avg_sample_std_dev"] = sum(per_sample_std) / len(per_sample_std)

    return out


class SpeedBenchResourcesServer(SimpleResourcesServer):
    config: SpeedBenchResourcesServerConfig

    # State populated lazily on the first verify() call so the benchmark window
    # is bounded by the first model traffic our server sees, not by server
    # init time. Wrapped in an asyncio.Lock to avoid two concurrent first calls
    # both triggering the snapshot.
    _before_snapshot: Optional[SpecDecodeMetricsSnapshot] = None
    _before_snapshot_time: Optional[float] = None
    _spec_decode_unavailable: bool = False
    _init_lock: Optional[asyncio.Lock] = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._init_lock = asyncio.Lock()
        if self.config.snapshot_at_init:
            # Best-effort eager snapshot — failure is fine, we'll retry lazily.
            try:
                asyncio.run(self._take_before_snapshot())
            except Exception as exc:
                LOG.warning("Eager /metrics snapshot at init failed: %s", exc)

    # ──────────────────────────────────────────────────────────
    # Metrics scraping
    # ──────────────────────────────────────────────────────────

    def _resolve_metrics_url(self) -> str:
        if self.config.vllm_metrics_url:
            return self.config.vllm_metrics_url
        if not self.config.vllm_base_url:
            raise RuntimeError(
                "speed_bench resources server: neither vllm_metrics_url nor vllm_base_url is set. "
                "Set one in configs/speed_bench.yaml (default: vllm_base_url=${policy_base_url})."
            )
        base = self.config.vllm_base_url.rstrip("/")
        # Drop a trailing /v1 so we land on <base>/metrics, not <base>/v1/metrics.
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return f"{base}/metrics"

    async def _scrape_metrics(self) -> SpecDecodeMetricsSnapshot:
        url = self._resolve_metrics_url()
        async with await global_request(
            method="GET",
            url=url,
            timeout=self.config.metrics_request_timeout_s,
        ) as response:
            if response.status != 200:
                raise SpecDecodeMetricsUnavailable(f"GET {url} returned status {response.status}")
            text = await response.text()

        if self.config.server_type_for_metrics == "sglang":
            return _parse_sglang_metrics(text)
        return _parse_vllm_metrics(text)

    async def _take_before_snapshot(self) -> None:
        """Scrape /metrics once and remember the result as the start-of-window."""
        try:
            self._before_snapshot = await self._scrape_metrics()
            self._before_snapshot_time = time.time()
            LOG.info(
                "speed_bench: BEFORE snapshot — drafts=%d, draft_tokens=%d, accepted=%d",
                self._before_snapshot.num_drafts,
                self._before_snapshot.num_draft_tokens,
                self._before_snapshot.num_accepted_tokens,
            )
        except SpecDecodeMetricsUnavailable as exc:
            LOG.warning("speed_bench: spec-decode metrics unavailable on the model server: %s", exc)
            self._spec_decode_unavailable = True
            self._before_snapshot = SpecDecodeMetricsSnapshot()
            self._before_snapshot_time = time.time()

    async def _ensure_before_snapshot(self) -> None:
        if self._before_snapshot is not None:
            return
        async with self._init_lock:
            if self._before_snapshot is None:
                await self._take_before_snapshot()

    # ──────────────────────────────────────────────────────────
    # /verify
    # ──────────────────────────────────────────────────────────

    async def verify(self, body: SpeedBenchVerifyRequest) -> SpeedBenchVerifyResponse:
        await self._ensure_before_snapshot()

        # Tokens emitted by the model for this task. The agent's response.usage
        # is the per-task accumulation across all multi-turn calls.
        num_generated_tokens = 0
        if body.response is not None and body.response.usage is not None:
            num_generated_tokens = int(body.response.usage.output_tokens or 0)

        running: Optional[Dict[str, Any]] = None
        if not self._spec_decode_unavailable:
            try:
                after = await self._scrape_metrics()
                if self.config.server_type_for_metrics == "sglang":
                    running = _compute_sglang_running_delta(self._before_snapshot, after)
                else:
                    running = _compute_running_delta(self._before_snapshot, after)
            except SpecDecodeMetricsUnavailable as exc:
                # Treat as a one-time transition: spec decoding is off on the
                # server. Don't keep retrying every task.
                LOG.warning("speed_bench: spec-decode counters disappeared mid-run: %s", exc)
                self._spec_decode_unavailable = True

        elapsed = (time.time() - (self._before_snapshot_time or time.time())) if self._before_snapshot_time else 0.0
        running = running or {}
        return SpeedBenchVerifyResponse(
            **body.model_dump(),
            reward=0.0,  # Speed-bench has no notion of correctness. Placeholder.
            num_generated_tokens=num_generated_tokens,
            gen_seconds=elapsed,
            acceptance_length=running.get("acceptance_length"),
            acceptance_rate=running.get("acceptance_rate"),
            num_drafts=running.get("num_drafts"),
            draft_tokens=running.get("draft_tokens"),
            accepted_tokens=running.get("accepted_tokens"),
            per_position_acceptance_rates=running.get("per_position_acceptance_rates"),
            spec_decode_unavailable=self._spec_decode_unavailable,
        )

    # ──────────────────────────────────────────────────────────
    # /aggregate_metrics
    # ──────────────────────────────────────────────────────────

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Aggregate per-task spec-decode running values into headline metrics.

        - `num_entries`: total number of tasks (rows) seen.
        - `avg_tokens`: mean num_generated_tokens across tasks.
        - `gen_seconds`: max gen_seconds across tasks (= total benchmark window).
        - `spec_acceptance_length` / `spec_acceptance_rate`: the running
          aggregate from the *task with the largest cumulative draft tokens*.
          That task ran latest in the benchmark window and so its running
          delta most closely approximates Skills' overall before/after delta.
        - `spec_per_position_acceptance_rates`: the per-position rates from
          the same task.

        These keys mirror Skills' `SpecdecMetrics.metrics_to_print` so the
        comparison table in COMPARISON_RESULTS.md is straightforward to
        line up.

        When each task has >1 rollouts (e.g. `+num_repeats=N` on the rollout
        CLI), `_compute_std_metrics` adds multi-seed variance estimators
        (`spec_<key>_avg`, `spec_<key>_std_dev_across_runs`,
        `spec_<key>_std_err_across_runs`, `spec_<key>_avg_sample_std_dev`)
        that mirror Skills' `BaseMetrics._add_std_metrics`. Single-seed
        runs skip these.
        """
        flat: List[Dict[str, Any]] = [r for task_rollouts in tasks for r in task_rollouts]
        n = len(flat)
        out: Dict[str, Any] = {"num_entries": n}
        if n == 0:
            return out

        out["avg_tokens"] = sum(r.get("num_generated_tokens", 0) for r in flat) / n
        out["gen_seconds"] = max(r.get("gen_seconds", 0.0) for r in flat)

        # Pick the task with the most accumulated draft tokens — that's the
        # one whose running aggregate most closely equals Skills' end-of-run
        # delta. Skip tasks where draft_tokens is None or 0.
        with_drafts = [r for r in flat if r.get("draft_tokens")]
        if with_drafts:
            best = max(with_drafts, key=lambda r: r["draft_tokens"])
            out["spec_acceptance_length"] = best.get("acceptance_length")
            out["spec_acceptance_rate"] = best.get("acceptance_rate")
            out["spec_num_drafts"] = best.get("num_drafts")
            out["spec_draft_tokens"] = best.get("draft_tokens")
            out["spec_accepted_tokens"] = best.get("accepted_tokens")
            out["spec_per_position_acceptance_rates"] = best.get("per_position_acceptance_rates")
        else:
            out["spec_acceptance_length"] = None
            out["spec_acceptance_rate"] = None
            out["spec_num_drafts"] = 0
            out["spec_draft_tokens"] = 0
            out["spec_accepted_tokens"] = 0
            out["spec_per_position_acceptance_rates"] = []
        out["spec_decode_unavailable"] = any(r.get("spec_decode_unavailable") for r in flat)

        # Multi-seed variance (mirrors Skills' BaseMetrics._add_std_metrics
        # for SpecdecMetrics' score keys). Only meaningful when each task has
        # >1 rollouts.
        out.update(_compute_std_metrics(tasks, _SPEC_SCORE_KEYS))
        return out

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Surface the headline numbers the comparison cares about.

        Always includes the 5 keys Skills' `metrics_to_print` exposes. When
        a multi-seed run produced std-metrics, also surface
        `spec_acceptance_length_std_err_across_runs` and
        `spec_acceptance_rate_std_err_across_runs` so a reader can tell at a
        glance whether a Skills↔Gym Δ is within sampling noise.
        """
        always = (
            "num_entries",
            "avg_tokens",
            "gen_seconds",
            "spec_acceptance_length",
            "spec_acceptance_rate",
        )
        optional = (
            "spec_acceptance_length_std_err_across_runs",
            "spec_acceptance_rate_std_err_across_runs",
        )
        keys = always + optional
        return {k: agent_metrics[k] for k in keys if k in agent_metrics}


if __name__ == "__main__":
    SpeedBenchResourcesServer.run_webserver()
