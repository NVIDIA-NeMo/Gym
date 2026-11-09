# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in regressions against the exact external evaluators, without model compute."""

import os
from pathlib import Path

import pytest

from resources_servers.spatialclaw.app import SpatialClawVerifyRequest
from resources_servers.spatialclaw.tests.test_app import _response, _server
from responses_api_agents.spatialclaw_agent.app import SPATIALCLAW_COMMIT, _validate_spatialclaw_checkout


pytestmark = pytest.mark.skipif(not os.environ.get("SPATIALCLAW_ROOT"), reason="requires pinned SPATIALCLAW_ROOT")


@pytest.fixture
def native_source(monkeypatch):
    root = _validate_spatialclaw_checkout(Path(os.environ["SPATIALCLAW_ROOT"]), SPATIALCLAW_COMMIT)
    monkeypatch.syspath_prepend(str(root))
    return root


@pytest.mark.parametrize("missing", [0, 3, 7])
def test_v2_mask_filter_drops_whole_native_group(native_source, missing):
    from spatial_agent.evals.videommev2 import VideoMMEv2Bench, VideoMMEv2Sample

    benchmark = VideoMMEv2Bench.__new__(VideoMMEv2Bench)
    benchmark.data = [
        VideoMMEv2Sample(
            sample_id=str(i),
            question="q",
            question_type="logic",
            images=[],
            answer="A",
            group_type="logic",
            group_structure="[1,[2,3],4]",
            level="1",
        )
        for i in range(8)
    ]
    server, _, _ = _server()
    server._benchmark_instance = benchmark
    server._sample_by_id = {sample.sample_id: sample for sample in benchmark.data}
    original = benchmark.data
    assert benchmark.evaluate({str(i): "A" for i in range(8)})["overall_accuracy"] == 1.0
    # Exactly what reward_profile passes after select_measured: no masked row.
    tasks = [
        [
            {
                "sample_id": str(i),
                "prediction": "A",
                "native_score": 1.0,
                "reward": 1.0,
                "extracted_answer": "A",
            }
        ]
        for i in reversed(range(8))
        if i != missing
    ]
    metrics = server.compute_metrics(tasks)
    assert metrics["spatialclaw/final_rating/total"] == 100
    assert metrics["spatialclaw/overall_accuracy"] == 1
    assert metrics["spatialclaw/num_evaluated_groups"] == 1
    assert metrics["spatialclaw/num_dropped_groups"] == 1
    assert "spatialclaw/final_rating/total" in server.get_key_metrics(metrics)
    assert benchmark.data is original
    empty = server.compute_metrics(tasks[:1])
    assert empty["spatialclaw/num_evaluated_groups"] == 0
    assert empty["spatialclaw/num_dropped_groups"] == 2
    assert "spatialclaw/final_rating/total" not in empty


async def test_spatialtree_native_unscoreable_row(native_source):
    from spatial_agent.evals.spatialtree import SpatialTreeBench, SpatialTreeSample

    sample = SpatialTreeSample(
        sample_id="unscoreable", question="q", question_type="open", images=[], answer="A", metricfunc="gpteval"
    )
    benchmark = SpatialTreeBench.__new__(SpatialTreeBench)
    benchmark.data = [sample]
    server, _, _ = _server()
    server._benchmark_instance = benchmark
    server._sample_by_id = {sample.sample_id: sample}
    result = await server.verify(
        SpatialClawVerifyRequest(
            sample_id=sample.sample_id, responses_create_params={"input": "q"}, response=_response("")
        )
    )
    assert result.native_score is None
    assert result.scored is False
    assert result.evaluation_completed is True
    assert result.extracted_answer is None
    assert result.reward == 0
    assert result.mask_sample is False
