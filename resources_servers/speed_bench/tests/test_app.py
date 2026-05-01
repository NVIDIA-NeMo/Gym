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
from unittest.mock import MagicMock

import pytest
from app import (
    SpecDecodeMetricsSnapshot,
    SpecDecodeMetricsUnavailable,
    SpeedBenchResourcesServer,
    SpeedBenchResourcesServerConfig,
    _compute_running_delta,
    _parse_vllm_metrics,
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


@pytest.mark.asyncio
async def test_sglang_scrape_is_stubbed():
    s = _make_server(vllm_base_url="http://host:8000/v1", server_type_for_metrics="sglang")
    with pytest.raises(NotImplementedError, match="SGLang"):
        await s._scrape_metrics()
