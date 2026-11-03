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

By default, aggregate vLLM per-request acceptance counters carried by the
speed_bench_agent. Each row contains only its own generation turns; headline
metrics give each model request equal weight across all rows. No model URL or metrics
scraping is needed. The explicit ``prometheus`` mode retains the legacy vLLM
and SGLang before/after-counter implementation for existing deployments.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, JsonValue, model_validator

from nemo_gym.client import request as global_request
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


LOG = logging.getLogger(__name__)


class SpecDecodeRequestMetrics(BaseModel):
    """vLLM's summary counters, excluding the bonus token from accepted counts."""

    num_spec_steps: int = Field(ge=0, strict=True)
    num_draft_tokens: int = Field(ge=0, strict=True)
    num_accepted_draft_tokens: int = Field(ge=0, strict=True)
    num_spec_tokens: int = Field(ge=0, strict=True)
    acceptance_histogram: list[Annotated[int, Field(ge=0, strict=True)]]

    @model_validator(mode="after")
    def validate_counts(self) -> SpecDecodeRequestMetrics:
        histogram = self.acceptance_histogram
        if len(histogram) != self.num_spec_tokens + 1 or sum(histogram) != self.num_spec_steps:
            raise ValueError("Acceptance histogram must match num_spec_tokens and num_spec_steps")
        if sum(i * count for i, count in enumerate(histogram)) != self.num_accepted_draft_tokens:
            raise ValueError("Acceptance histogram must match num_accepted_draft_tokens")
        if not self.num_accepted_draft_tokens <= self.num_draft_tokens <= self.num_spec_steps * self.num_spec_tokens:
            raise ValueError("Draft token counts are inconsistent with verification steps")
        return self


def _acceptance_metrics(requests: list[SpecDecodeRequestMetrics]) -> dict[str, JsonValue]:
    """Average request metrics, normalizing each histogram before averaging."""
    active = [r for r in requests if r.num_spec_steps]
    drafted = [r for r in requests if r.num_draft_tokens]
    histogram = (
        [
            statistics.mean(
                r.acceptance_histogram[i] / r.num_spec_steps if i < len(r.acceptance_histogram) else 0.0
                for r in active
            )
            for i in range(max(len(r.acceptance_histogram) for r in active))
        ]
        if active
        else None
    )
    return {
        "num_requests": len(requests),
        "num_drafts": statistics.mean(r.num_spec_steps for r in requests),
        "draft_tokens": statistics.mean(r.num_draft_tokens for r in requests),
        "accepted_tokens": statistics.mean(r.num_accepted_draft_tokens for r in requests),
        "acceptance_histogram": histogram,
        "acceptance_length": statistics.mean(1 + r.num_accepted_draft_tokens / r.num_spec_steps for r in active)
        if active
        else None,
        "acceptance_rate": statistics.mean(100 * r.num_accepted_draft_tokens / r.num_draft_tokens for r in drafted)
        if drafted
        else None,
        "per_position_acceptance_rates": [sum(histogram[i:]) for i in range(1, len(histogram))]
        if histogram is not None
        else None,
    }


def _parse_request_metrics(metrics: dict[str, JsonValue] | None) -> list[SpecDecodeRequestMetrics] | None:
    """Require valid provider counters for every model call, including zero-step calls."""
    if not metrics:
        return None
    requests = metrics.get("requests", [metrics])
    if not isinstance(requests, list) or not requests:
        return None
    summaries = []
    for request in requests:
        spec = request.get("speculative_decoding") if isinstance(request, dict) else None
        if spec is None:
            return None
        summaries.append(SpecDecodeRequestMetrics.model_validate(spec))
    return summaries


def _request_acceptance_metrics(metrics: dict[str, JsonValue] | None) -> dict[str, JsonValue] | None:
    summaries = _parse_request_metrics(metrics)
    return _acceptance_metrics(summaries) if summaries else None


def _row_request_metrics(row: Dict[str, Any]) -> list[SpecDecodeRequestMetrics] | None:
    # Read original provider counters so multi-turn tasks do not become one vote,
    # and older response-mode rollouts can be reaggregated with the new weighting.
    return _parse_request_metrics((row.get("response") or {}).get("metrics"))


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
    """Use response metrics by default, or explicitly select legacy Prometheus scraping.

    ``vllm_base_url``, ``vllm_metrics_url``, ``server_type_for_metrics``,
    ``metrics_request_timeout_s`` and ``snapshot_at_init`` apply only in
    ``prometheus`` mode. That mode requires an explicit backend URL.
    """

    metrics_source: Literal["response", "prometheus"] = "response"
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
    """Per-task token counts, generation duration, and request-averaged acceptance metrics.

    In response mode these cover all turns of this task. Missing metrics on any
    turn make acceptance metrics unavailable; a reported zero-step histogram is
    available but has undefined rates and histogram probabilities. Acceptance rate is a percentage.
    Legacy Prometheus mode retains cumulative benchmark-window semantics.
    """

    num_generated_tokens: int = 0
    gen_seconds: float = 0.0
    acceptance_length: Optional[float] = None
    acceptance_rate: Optional[float] = None
    num_requests: Optional[int] = None
    num_drafts: Optional[int | float] = None
    draft_tokens: Optional[int | float] = None
    accepted_tokens: Optional[int | float] = None
    per_position_acceptance_rates: Optional[List[float]] = None
    acceptance_histogram: Optional[List[float]] = None
    generation_started_at: Optional[float] = None
    generation_finished_at: Optional[float] = None
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


def _compute_std_metrics(
    tasks: List[List[Dict[str, Any]]], score_keys: tuple, *, request_weighted: bool = False
) -> Dict[str, Any]:
    """Multi-seed variance metrics, optionally giving each model request one vote.

    For each score key, with `k = max_k` rollouts/task:

    - `spec_<key>_avg` — overall mean across requests in response mode,
      or (task, rollout) pairs in legacy mode.
    - `spec_<key>_std_dev_across_runs` — std-dev of per-run averages,
      where run i uses rollout i from each task, averaging its individual
      requests in response mode or its row values in legacy mode.
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
        # Each task/run cell contains its individual request values (one row
        # value in legacy mode). Preserve these until computing run means.
        # Drop tasks whose rollouts don't all have a numeric value for this
        # score (running aggregates can be None on the very first task).
        matrix: List[List[List[float]]] = []
        for task_rollouts in tasks:
            if len(task_rollouts) < max_k:
                continue
            if request_weighted:
                requests_by_run = [_row_request_metrics(task_rollouts[i]) for i in range(max_k)]
                if any(requests is None for requests in requests_by_run):
                    continue
                row = [
                    [value for r in requests if (value := _acceptance_metrics([r])[key]) is not None]
                    for requests in requests_by_run
                ]
            else:
                row = [[task_rollouts[i].get(key)] for i in range(max_k)]
            if any(not values or any(v is None for v in values) for values in row):
                continue
            matrix.append([[float(v) for v in values] for values in row])

        if not matrix:
            continue

        # Average all included requests (or rows in legacy mode).
        all_values = [v for row in matrix for values in row for v in values]
        out[f"spec_{key}_avg"] = sum(all_values) / len(all_values)

        # std_dev_across_runs: transpose → run i averages → std-dev.
        run_averages = [statistics.mean(v for row in matrix for v in row[i]) for i in range(max_k)]
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
        for values_by_run in matrix:
            row = [statistics.mean(values) for values in values_by_run]
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
        if self.config.metrics_source == "prometheus" and self.config.snapshot_at_init:
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
                "Set one explicitly when using metrics_source=prometheus."
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
        if self.config.metrics_source == "response":
            metrics = body.response.metrics if body.response is not None else None
            acceptance = _request_acceptance_metrics(metrics)
            timing = metrics or {}
            usage = body.response.usage if body.response is not None else None
            return SpeedBenchVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                num_generated_tokens=usage.output_tokens if usage is not None else 0,
                gen_seconds=timing.get("generation_duration", 0.0),
                generation_started_at=timing.get("generation_started_at"),
                generation_finished_at=timing.get("generation_finished_at"),
                spec_decode_unavailable=acceptance is None,
                **(acceptance or {}),
            )
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
        """Average individual request metrics across tasks in response mode.

        The following cumulative behavior applies only to legacy Prometheus mode:

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

        if self.config.metrics_source == "response":
            # Requests can overlap. Wall time is the enclosing generation window,
            # not the sum of request latencies. Agent hosts must have synced clocks.
            if all(
                r.get("generation_started_at") is not None and r.get("generation_finished_at") is not None
                for r in flat
            ):
                out["gen_seconds"] = max(r["generation_finished_at"] for r in flat) - min(
                    r["generation_started_at"] for r in flat
                )
            requests_by_row = [_row_request_metrics(row) for row in flat]
            unavailable = any(requests is None for requests in requests_by_row)
            if unavailable:
                out["avg_tokens"] = None
                # Do not present rates for a subset as complete benchmark metrics.
                summary = dict.fromkeys(
                    (*_SPEC_SCORE_KEYS, "num_requests", "per_position_acceptance_rates", "acceptance_histogram")
                )
            else:
                requests = [request for row in requests_by_row for request in row]
                summary = _acceptance_metrics(requests)
                out["avg_tokens"] = sum(r.get("num_generated_tokens", 0) for r in flat) / len(requests)
            out.update({f"spec_{key}": value for key, value in summary.items()})
            out["spec_decode_unavailable"] = unavailable
            out.update(_compute_std_metrics(tasks, _SPEC_SCORE_KEYS, request_weighted=True))
            return out

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
