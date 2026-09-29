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
"""Unit tests for the speed_bench resources server.

Covers the pure-function paths (Prometheus parser, before/after delta,
metrics URL resolution, compute_metrics aggregation). The async verify()
call is exercised via a fake metrics-scrape stub.
"""

import asyncio
import math
from unittest.mock import MagicMock

import pytest
from app import (
    _SPEC_SCORE_KEYS,
    SpecDecodeMetricsSnapshot,
    SpecDecodeMetricsUnavailable,
    SpeedBenchResourcesServer,
    SpeedBenchResourcesServerConfig,
    _compute_running_delta,
    _compute_sglang_running_delta,
    _compute_std_metrics,
    _parse_sglang_metrics,
    _parse_vllm_metrics,
    _weighted_delta_average,
)

from nemo_gym.server_utils import ServerClient


# ──────────────────────────────────────────────────────────────────────────────
# Prometheus parser
# ──────────────────────────────────────────────────────────────────────────────

VLLM_METRICS_TEXT = """\
# HELP vllm:spec_decode_num_drafts_total Total number of drafts.
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{model_name="m"} 100
vllm:spec_decode_num_draft_tokens_total{model_name="m"} 300
vllm:spec_decode_num_accepted_tokens_total{model_name="m"} 240
vllm:spec_decode_num_accepted_tokens_per_pos{position="0",model_name="m"} 90
vllm:spec_decode_num_accepted_tokens_per_pos{position="1",model_name="m"} 80
vllm:spec_decode_num_accepted_tokens_per_pos{position="2",model_name="m"} 70
"""


def test_parse_vllm_metrics_basic():
    snap = _parse_vllm_metrics(VLLM_METRICS_TEXT)
    assert snap.num_drafts == 100
    assert snap.num_draft_tokens == 300
    assert snap.num_accepted_tokens == 240
    assert snap.accepted_per_pos == {0: 90, 1: 80, 2: 70}


def test_parse_vllm_metrics_skips_created_lines():
    text = VLLM_METRICS_TEXT + 'vllm:spec_decode_num_drafts_created{model_name="m"} 1234\n'
    snap = _parse_vllm_metrics(text)
    # _created lines are ignored
    assert snap.num_drafts == 100


def test_parse_vllm_metrics_no_spec_decode_raises():
    with pytest.raises(SpecDecodeMetricsUnavailable):
        _parse_vllm_metrics("# nothing here\nvllm:request_count_total 1\n")


def test_parse_vllm_metrics_skips_blank_and_comment_lines():
    text = "\n   \n# comment\n" + VLLM_METRICS_TEXT
    snap = _parse_vllm_metrics(text)
    assert snap.num_drafts == 100


# ──────────────────────────────────────────────────────────────────────────────
# Running delta
# ──────────────────────────────────────────────────────────────────────────────


def test_compute_running_delta_basic():
    before = SpecDecodeMetricsSnapshot(
        num_drafts=10, num_draft_tokens=30, num_accepted_tokens=20, accepted_per_pos={0: 10, 1: 5, 2: 5}
    )
    after = SpecDecodeMetricsSnapshot(
        num_drafts=110,
        num_draft_tokens=330,
        num_accepted_tokens=260,
        accepted_per_pos={0: 100, 1: 85, 2: 75},
    )
    d = _compute_running_delta(before, after)
    assert d is not None
    assert d["num_drafts"] == 100
    assert d["draft_tokens"] == 300
    assert d["accepted_tokens"] == 240
    # acceptance_rate = 240/300 * 100
    assert d["acceptance_rate"] == pytest.approx(80.0)
    # acceptance_length = 1 + 240/100 = 3.4
    assert d["acceptance_length"] == pytest.approx(3.4)
    # per-position rates: (100-10)/100, (85-5)/100, (75-5)/100
    assert d["per_position_acceptance_rates"] == pytest.approx([0.9, 0.8, 0.7])


def test_compute_running_delta_zero_activity_returns_none():
    before = SpecDecodeMetricsSnapshot()
    after = SpecDecodeMetricsSnapshot()
    assert _compute_running_delta(before, after) is None


def test_compute_running_delta_zero_drafts_yields_zero_acceptance_length():
    # Edge case: tokens but no drafts (shouldn't happen in practice, but guard).
    before = SpecDecodeMetricsSnapshot()
    after = SpecDecodeMetricsSnapshot(num_drafts=0, num_draft_tokens=10, num_accepted_tokens=5)
    d = _compute_running_delta(before, after)
    assert d is not None
    assert d["acceptance_length"] == 0.0
    assert d["per_position_acceptance_rates"] == []


# ──────────────────────────────────────────────────────────────────────────────
# Metrics URL resolution
# ──────────────────────────────────────────────────────────────────────────────


def _make_server(**kwargs):
    config = SpeedBenchResourcesServerConfig(
        type="resources_servers",
        name="speed_bench",
        host="127.0.0.1",
        port=12345,
        entrypoint="app.py",
        **kwargs,
    )
    server = SpeedBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    return server


def test_resolve_metrics_url_strips_v1_suffix():
    s = _make_server(vllm_base_url="http://host:8000/v1")
    assert s._resolve_metrics_url() == "http://host:8000/metrics"


def test_resolve_metrics_url_no_v1_suffix():
    s = _make_server(vllm_base_url="http://host:8000")
    assert s._resolve_metrics_url() == "http://host:8000/metrics"


def test_resolve_metrics_url_explicit_overrides_base():
    s = _make_server(vllm_base_url="http://host:8000/v1", vllm_metrics_url="http://other:9000/metrics")
    assert s._resolve_metrics_url() == "http://other:9000/metrics"


def test_resolve_metrics_url_unset_raises():
    s = _make_server()
    with pytest.raises(RuntimeError, match="neither vllm_metrics_url nor vllm_base_url"):
        s._resolve_metrics_url()


# ──────────────────────────────────────────────────────────────────────────────
# compute_metrics aggregation
# ──────────────────────────────────────────────────────────────────────────────


def test_compute_metrics_picks_largest_draft_tokens():
    s = _make_server(vllm_base_url="http://host:8000/v1")
    tasks = [
        [
            {
                "num_generated_tokens": 100,
                "gen_seconds": 5.0,
                "draft_tokens": 50,
                "num_drafts": 10,
                "accepted_tokens": 30,
                "acceptance_length": 4.0,
                "acceptance_rate": 60.0,
                "per_position_acceptance_rates": [0.6, 0.5, 0.4],
            }
        ],
        [
            {
                "num_generated_tokens": 200,
                "gen_seconds": 10.0,
                "draft_tokens": 500,  # this task's running aggregate is the headline
                "num_drafts": 100,
                "accepted_tokens": 380,
                "acceptance_length": 4.8,
                "acceptance_rate": 76.0,
                "per_position_acceptance_rates": [0.9, 0.85, 0.6],
            }
        ],
    ]
    m = s.compute_metrics(tasks)
    assert m["num_entries"] == 2
    assert m["avg_tokens"] == 150
    assert m["gen_seconds"] == 10.0
    assert m["spec_acceptance_length"] == 4.8
    assert m["spec_acceptance_rate"] == 76.0
    assert m["spec_draft_tokens"] == 500
    assert m["spec_decode_unavailable"] is False


def test_compute_metrics_empty_tasks():
    s = _make_server(vllm_base_url="http://host:8000/v1")
    assert s.compute_metrics([]) == {"num_entries": 0}


def test_compute_metrics_no_drafts_yields_none_headlines():
    s = _make_server(vllm_base_url="http://host:8000/v1")
    tasks = [[{"num_generated_tokens": 50, "gen_seconds": 1.0}]]
    m = s.compute_metrics(tasks)
    assert m["num_entries"] == 1
    assert m["avg_tokens"] == 50
    assert m["spec_acceptance_length"] is None
    assert m["spec_acceptance_rate"] is None
    assert m["spec_draft_tokens"] == 0


def test_compute_metrics_propagates_unavailable_flag():
    s = _make_server(vllm_base_url="http://host:8000/v1")
    tasks = [[{"num_generated_tokens": 50, "gen_seconds": 1.0, "spec_decode_unavailable": True}]]
    m = s.compute_metrics(tasks)
    assert m["spec_decode_unavailable"] is True


def test_get_key_metrics_filters_to_headline_keys():
    s = _make_server(vllm_base_url="http://host:8000/v1")
    full = {
        "num_entries": 880,
        "avg_tokens": 463,
        "gen_seconds": 104.0,
        "spec_acceptance_length": 2.37,
        "spec_acceptance_rate": 45.52,
        "spec_draft_tokens": 1234,  # excluded
        "irrelevant": 42,
    }
    key = s.get_key_metrics(full)
    assert set(key) == {"num_entries", "avg_tokens", "gen_seconds", "spec_acceptance_length", "spec_acceptance_rate"}
    assert key["spec_acceptance_length"] == 2.37


# ──────────────────────────────────────────────────────────────────────────────
# verify() — async path with stubbed scrape
# ──────────────────────────────────────────────────────────────────────────────


def _make_fake_body(output_tokens: int | None):
    """Build a duck-typed verify request stub that bypasses pydantic validation.

    The real `SpeedBenchVerifyRequest` requires a fully-formed
    NeMoGymResponseCreateParamsNonStreaming + NeMoGymResponse which is heavy
    to mock. The server's `verify()` only reads
    `body.response.usage.output_tokens` and `body.model_dump()` (used to
    forward fields to the response), so a duck-typed stub suffices.
    """

    class _Usage:
        def __init__(self, n):
            self.output_tokens = n

    class _Response:
        def __init__(self, n):
            self.usage = _Usage(n) if n is not None else None

    class _FakeBody:
        def __init__(self, n):
            self.response = _Response(n) if n is not None else None

        def model_dump(self):
            return {"responses_create_params": {"input": []}, "response": None}

    return _FakeBody(output_tokens)


@pytest.mark.asyncio
async def test_verify_records_running_aggregate(monkeypatch):
    s = _make_server(vllm_base_url="http://host:8000/v1")
    s._init_lock = asyncio.Lock()

    snapshots = iter(
        [
            SpecDecodeMetricsSnapshot(num_drafts=0, num_draft_tokens=0, num_accepted_tokens=0),
            SpecDecodeMetricsSnapshot(num_drafts=5, num_draft_tokens=15, num_accepted_tokens=12),
        ]
    )

    async def fake_scrape():
        return next(snapshots)

    monkeypatch.setattr(s, "_scrape_metrics", fake_scrape)

    # Bypass strict pydantic validation by constructing the response directly.
    # We patch `SpeedBenchVerifyResponse` instantiation through the verify
    # method by constructing the response with model_construct.
    from app import SpeedBenchVerifyResponse

    monkeypatch.setattr(
        "app.SpeedBenchVerifyResponse",
        lambda **kw: SpeedBenchVerifyResponse.model_construct(**kw),
    )

    out = await s.verify(_make_fake_body(42))
    assert out.num_generated_tokens == 42
    assert out.num_drafts == 5
    assert out.draft_tokens == 15
    assert out.accepted_tokens == 12
    assert out.acceptance_rate == pytest.approx(80.0)
    assert out.acceptance_length == pytest.approx(3.4)
    assert out.spec_decode_unavailable is False


@pytest.mark.asyncio
async def test_verify_marks_unavailable_when_scrape_fails(monkeypatch):
    s = _make_server(vllm_base_url="http://host:8000/v1")
    s._init_lock = asyncio.Lock()

    async def fake_scrape():
        raise SpecDecodeMetricsUnavailable("disabled")

    monkeypatch.setattr(s, "_scrape_metrics", fake_scrape)

    from app import SpeedBenchVerifyResponse

    monkeypatch.setattr(
        "app.SpeedBenchVerifyResponse",
        lambda **kw: SpeedBenchVerifyResponse.model_construct(**kw),
    )

    out = await s.verify(_make_fake_body(None))
    assert out.spec_decode_unavailable is True
    assert out.acceptance_length is None
    assert out.acceptance_rate is None


# ──────────────────────────────────────────────────────────────────────────────
# SGLang Prometheus path
# ──────────────────────────────────────────────────────────────────────────────

SGLANG_METRICS_TEXT = """\
# HELP sglang:spec_accept_length Acceptance length running average.
# TYPE sglang:spec_accept_length gauge
sglang:spec_accept_length{model="m"} 2.65
sglang:spec_accept_rate{model="m"} 0.55
sglang:num_requests_total{model="m"} 250
sglang:generation_tokens_total{model="m"} 75000
"""


def test_parse_sglang_metrics_basic():
    snap = _parse_sglang_metrics(SGLANG_METRICS_TEXT)
    assert snap.spec_accept_length == pytest.approx(2.65)
    assert snap.spec_accept_rate == pytest.approx(0.55)
    assert snap.num_requests == 250
    assert snap.generation_tokens == 75000


def test_parse_sglang_metrics_no_spec_accept_raises():
    text = '# nothing here\nsglang:num_requests_total{m="a"} 1\n'
    with pytest.raises(SpecDecodeMetricsUnavailable):
        _parse_sglang_metrics(text)


def test_weighted_delta_average_basic():
    # Before: 100 requests at AL=2.0 (weighted=200)
    # After:  300 requests at AL=2.5 (weighted=750)
    # Benchmark window: 200 requests, weighted=550 → AL=2.75
    assert _weighted_delta_average(2.0, 2.5, 100, 300) == pytest.approx(2.75)


def test_weighted_delta_average_first_window_returns_after():
    # Before count=0 → can't weight; just return the after average.
    assert _weighted_delta_average(0.0, 3.1, 0, 50) == pytest.approx(3.1)


def test_weighted_delta_average_no_new_requests_returns_none():
    assert _weighted_delta_average(2.0, 2.5, 100, 100) is None
    assert _weighted_delta_average(2.0, 2.5, 100, 50) is None


def test_compute_sglang_running_delta_basic():
    before = SpecDecodeMetricsSnapshot(
        spec_accept_length=2.0, spec_accept_rate=0.5, num_requests=100, generation_tokens=10000
    )
    after = SpecDecodeMetricsSnapshot(
        spec_accept_length=2.5, spec_accept_rate=0.6, num_requests=300, generation_tokens=40000
    )
    d = _compute_sglang_running_delta(before, after)
    assert d is not None
    assert d["num_drafts"] == 200  # delta_requests
    assert d["draft_tokens"] == 30000  # delta_gen_tokens
    # weighted AR: (0.6 * 300 - 0.5 * 100) / 200 = (180 - 50) / 200 = 0.65
    assert d["acceptance_rate"] == pytest.approx(65.0)
    # weighted AL: (2.5 * 300 - 2.0 * 100) / 200 = (750 - 200) / 200 = 2.75
    assert d["acceptance_length"] == pytest.approx(2.75)
    # accepted_tokens ≈ delta_gen_tokens * ar_fraction = 30000 * 0.65 = 19500
    assert d["accepted_tokens"] == 19500
    assert d["per_position_acceptance_rates"] == []


def test_compute_sglang_running_delta_no_traffic_returns_none():
    before = SpecDecodeMetricsSnapshot(num_requests=100)
    after = SpecDecodeMetricsSnapshot(num_requests=100)
    assert _compute_sglang_running_delta(before, after) is None


@pytest.mark.asyncio
async def test_verify_sglang_path_uses_sglang_delta(monkeypatch):
    """End-to-end: SGLang config + sglang snapshots → SGLang weighted-delta arithmetic."""
    s = _make_server(vllm_base_url="http://host:8000", server_type_for_metrics="sglang")
    s._init_lock = asyncio.Lock()

    snapshots = iter(
        [
            SpecDecodeMetricsSnapshot(
                spec_accept_length=2.0, spec_accept_rate=0.5, num_requests=10, generation_tokens=1000
            ),
            SpecDecodeMetricsSnapshot(
                spec_accept_length=2.5, spec_accept_rate=0.6, num_requests=30, generation_tokens=4000
            ),
        ]
    )

    async def fake_scrape():
        return next(snapshots)

    monkeypatch.setattr(s, "_scrape_metrics", fake_scrape)
    from app import SpeedBenchVerifyResponse

    monkeypatch.setattr(
        "app.SpeedBenchVerifyResponse",
        lambda **kw: SpeedBenchVerifyResponse.model_construct(**kw),
    )

    out = await s.verify(_make_fake_body(42))
    # weighted AL: (2.5*30 - 2.0*10)/20 = (75-20)/20 = 2.75
    assert out.acceptance_length == pytest.approx(2.75)
    # weighted AR fraction: (0.6*30 - 0.5*10)/20 = (18-5)/20 = 0.65 → 65%
    assert out.acceptance_rate == pytest.approx(65.0)
    assert out.num_drafts == 20  # delta_requests
    assert out.draft_tokens == 3000  # delta_gen_tokens
    assert out.spec_decode_unavailable is False


# ──────────────────────────────────────────────────────────────────────────────
# Multi-seed variance (mirrors Skills' _add_std_metrics)
# ──────────────────────────────────────────────────────────────────────────────


def test_compute_std_metrics_single_rollout_returns_empty():
    """max_k == 1: no variance to compute, skip the whole block."""
    tasks = [[{"acceptance_length": 1.5}], [{"acceptance_length": 1.7}]]
    assert _compute_std_metrics(tasks, _SPEC_SCORE_KEYS) == {}


def test_compute_std_metrics_empty_tasks_returns_empty():
    assert _compute_std_metrics([], _SPEC_SCORE_KEYS) == {}


def test_compute_std_metrics_basic_two_rollouts():
    """Two tasks × two rollouts: hand-computed values for acceptance_length."""
    tasks = [
        [
            {
                "acceptance_length": 2.0,
                "acceptance_rate": 50.0,
                "num_drafts": 10,
                "draft_tokens": 30,
                "accepted_tokens": 15,
            },
            {
                "acceptance_length": 2.4,
                "acceptance_rate": 60.0,
                "num_drafts": 20,
                "draft_tokens": 60,
                "accepted_tokens": 36,
            },
        ],
        [
            {
                "acceptance_length": 2.2,
                "acceptance_rate": 55.0,
                "num_drafts": 12,
                "draft_tokens": 36,
                "accepted_tokens": 19,
            },
            {
                "acceptance_length": 2.6,
                "acceptance_rate": 65.0,
                "num_drafts": 22,
                "draft_tokens": 66,
                "accepted_tokens": 42,
            },
        ],
    ]
    out = _compute_std_metrics(tasks, _SPEC_SCORE_KEYS)

    # acceptance_length avg = mean of [2.0, 2.4, 2.2, 2.6] = 2.3
    assert out["spec_acceptance_length_avg"] == pytest.approx(2.3)

    # Run-1 avg = (2.0 + 2.2)/2 = 2.1; Run-2 avg = (2.4 + 2.6)/2 = 2.5
    # std_dev_across_runs = stdev([2.1, 2.5]) with ddof=1 = sqrt(((2.1-2.3)^2 + (2.5-2.3)^2)/1) = sqrt(0.08) ≈ 0.2828
    assert out["spec_acceptance_length_std_dev_across_runs"] == pytest.approx(0.282842712474619)

    # std_err = std_dev / sqrt(2)
    assert out["spec_acceptance_length_std_err_across_runs"] == pytest.approx(0.282842712474619 / math.sqrt(2))

    # Per-task std-devs: stdev([2.0,2.4]) = 0.2828..., stdev([2.2,2.6]) = 0.2828...
    # avg = 0.2828...
    assert out["spec_acceptance_length_avg_sample_std_dev"] == pytest.approx(0.282842712474619)

    # Sanity: every score key emitted with all 4 fields.
    for key in _SPEC_SCORE_KEYS:
        for suffix in ("_avg", "_std_dev_across_runs", "_std_err_across_runs", "_avg_sample_std_dev"):
            assert f"spec_{key}{suffix}" in out


def test_compute_std_metrics_skips_tasks_with_none_values():
    """Tasks where any of the k rollouts has None for a key are dropped for that key."""
    tasks = [
        [
            {"acceptance_length": 2.0, "acceptance_rate": None},
            {"acceptance_length": 2.4, "acceptance_rate": 60.0},
        ],
        [
            {"acceptance_length": 2.2, "acceptance_rate": 55.0},
            {"acceptance_length": 2.6, "acceptance_rate": 65.0},
        ],
    ]
    out = _compute_std_metrics(tasks, ("acceptance_length", "acceptance_rate"))
    # acceptance_length: both tasks usable → variance computed.
    assert "spec_acceptance_length_avg" in out
    # acceptance_rate: first task has a None value in row → drop, only one task left.
    # avg from second task only: (55+65)/2 = 60
    assert out["spec_acceptance_rate_avg"] == pytest.approx(60.0)


def test_compute_std_metrics_all_runs_identical_yields_zero_std():
    """Edge case: every rollout has the same value (Skills stamps the same global delta)."""
    tasks = [
        [{"acceptance_length": 1.5}, {"acceptance_length": 1.5}],
        [{"acceptance_length": 1.5}, {"acceptance_length": 1.5}],
    ]
    out = _compute_std_metrics(tasks, ("acceptance_length",))
    assert out["spec_acceptance_length_avg"] == pytest.approx(1.5)
    assert out["spec_acceptance_length_std_dev_across_runs"] == 0.0
    assert out["spec_acceptance_length_std_err_across_runs"] == 0.0
    assert out["spec_acceptance_length_avg_sample_std_dev"] == 0.0


def test_compute_std_metrics_uneven_rollout_counts_uses_min_k():
    """Skills enforces equal k; we relax to min(k) to be robust against partial outputs."""
    tasks = [
        [{"acceptance_length": 2.0}, {"acceptance_length": 2.4}, {"acceptance_length": 3.0}],  # k=3
        [{"acceptance_length": 2.2}, {"acceptance_length": 2.6}],  # k=2
    ]
    out = _compute_std_metrics(tasks, ("acceptance_length",))
    # Both tasks contribute their first 2 rollouts (min_k = 2).
    # avg over (2.0, 2.4, 2.2, 2.6) = 2.3
    assert out["spec_acceptance_length_avg"] == pytest.approx(2.3)


def test_compute_metrics_emits_std_metrics_when_multi_seed():
    """End-to-end: compute_metrics(tasks_with_2_rollouts) includes std metrics."""
    s = _make_server(vllm_base_url="http://h:8000")
    tasks = [
        [
            {
                "num_generated_tokens": 100,
                "gen_seconds": 5.0,
                "draft_tokens": 50,
                "num_drafts": 10,
                "accepted_tokens": 30,
                "acceptance_length": 4.0,
                "acceptance_rate": 60.0,
            },
            {
                "num_generated_tokens": 200,
                "gen_seconds": 10.0,
                "draft_tokens": 100,
                "num_drafts": 20,
                "accepted_tokens": 70,
                "acceptance_length": 4.5,
                "acceptance_rate": 70.0,
            },
        ],
        [
            {
                "num_generated_tokens": 150,
                "gen_seconds": 7.0,
                "draft_tokens": 80,
                "num_drafts": 15,
                "accepted_tokens": 50,
                "acceptance_length": 4.2,
                "acceptance_rate": 62.0,
            },
            {
                "num_generated_tokens": 250,
                "gen_seconds": 12.0,
                "draft_tokens": 120,
                "num_drafts": 25,
                "accepted_tokens": 90,
                "acceptance_length": 4.6,
                "acceptance_rate": 75.0,
            },
        ],
    ]
    m = s.compute_metrics(tasks)
    # Headlines still present.
    assert m["num_entries"] == 4
    # Std-metrics section present for every key.
    for key in _SPEC_SCORE_KEYS:
        assert f"spec_{key}_avg" in m
        assert f"spec_{key}_std_dev_across_runs" in m


def test_compute_metrics_omits_std_metrics_when_single_seed():
    """Single-rollout tasks (the typical speed-bench shape) skip std-metrics block."""
    s = _make_server(vllm_base_url="http://h:8000")
    tasks = [
        [
            {
                "num_generated_tokens": 100,
                "gen_seconds": 5.0,
                "draft_tokens": 50,
                "num_drafts": 10,
                "accepted_tokens": 30,
                "acceptance_length": 4.0,
                "acceptance_rate": 60.0,
            }
        ],
    ]
    m = s.compute_metrics(tasks)
    assert "spec_acceptance_length_avg" not in m
    assert "spec_acceptance_length_std_dev_across_runs" not in m
