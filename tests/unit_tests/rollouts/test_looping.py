# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from concurrent.futures import Future
from random import Random
from unittest.mock import Mock

import pytest

from nemo_gym.core.responses_api_types import NeMoGymResponseOutputMessage
from nemo_gym.eval.comparison.loading import LoadedRun
from nemo_gym.eval.comparison.pooling import pool_loaded_runs
from nemo_gym.metrics.looping import LoopingAssessment, LoopingLimits, assess_text, collect_response_text
from nemo_gym.metrics.reward_profile import compute_aggregate_metrics
from nemo_gym.rollouts.looping import LoopingAssessor


@pytest.mark.parametrize("period", [1, 73, 4096])
def test_periodic_span_at_an_unaligned_boundary(period):
    unit = "".join(Random(period).choices("abcdefghijklmnopqrstuvwxyz0123456789", k=period))
    assert assess_text(("prefix " * 9 + unit * max(16, (2048 + period - 1) // period) + "partial",)).is_looping is True


def test_paragraph_rules_and_work_exhaustion():
    paragraph = "".join(Random(17).choices("abcdefghijklmnopqrstuvwxyz ", k=2100))
    assert assess_text(("\n\n".join([paragraph] * 5),)).is_looping is True
    assert assess_text(("\n\n".join([paragraph] * 4),)).is_looping is False
    negative = (paragraph[:256] + paragraph[:256].upper()) * 8
    assert assess_text((negative,)).is_looping is False
    assert assess_text((negative,), LoopingLimits(search_work=2)).is_looping is None


def test_content_defaults_selection_and_unreadable_text():
    item = NeMoGymResponseOutputMessage(id="m", content=[{"text": " a "}, {"refusal": "b"}], status="incomplete")
    for message in (item, item.model_dump(exclude_defaults=True)):
        response = {"output": [message, {"type": "reasoning", "content": [{"text": "think"}], "summary": []}]}
        assert collect_response_text(response, LoopingLimits()) == (" a b", "think")
    assert (
        collect_response_text(
            {"output": [{"role": "user", "content": "x"}, {"type": "function_call", "arguments": "x"}]},
            LoopingLimits(),
        )
        == ()
    )
    assert collect_response_text({"output": [{"content": [{"text": 3}]}]}, LoopingLimits()) is None
    assert collect_response_text({"output": [{"content": "long"}]}, LoopingLimits(max_text_chars=2)) is None


@pytest.mark.asyncio
async def test_spawned_assessment_and_observed_empty_output():
    assessor = LoopingAssessor()
    try:
        assert (await assessor.assess({"output": [{"content": "loop " * 600}]})).is_looping is True
        assert (await assessor.assess({"output": []})).is_looping is False
        assert (await assessor.assess({})).is_looping is None
    finally:
        assessor.close()


@pytest.mark.asyncio
async def test_cancellation_reserves_capacity_until_submitted_work_finishes(monkeypatch):
    import nemo_gym.rollouts.looping as module

    future = Future()
    future.set_running_or_notify_cancel()
    pool = Mock()
    pool.submit.return_value = future
    monkeypatch.setattr(module, "ProcessPoolExecutor", Mock(return_value=pool))
    reader = Mock(wraps=module.collect_response_text)
    monkeypatch.setattr(module, "collect_response_text", reader)
    assessor = LoopingAssessor()
    task = asyncio.create_task(assessor.assess({"output": []}))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    second = asyncio.create_task(assessor.assess({"output": []}))
    await asyncio.sleep(0)
    assert reader.call_count == 1
    assert pool.submit.call_args.args[1] == ()
    future.set_result(LoopingAssessment(False))
    assert (await second).is_looping is False
    assessor.close()


def test_metrics_include_masked_outputs_and_exclude_imputed_zeros():
    rows = [
        {"reward": 1, "is_looping": True, "mask_sample": True},
        {"reward": 1, "is_looping": False},
        {"reward": 1},
        {"reward": 0, "_ng_failure_counts_as_zero": True},
    ]
    rows = [row | {"_ng_rollout_index": index} for index, row in enumerate(rows)]
    metrics = compute_aggregate_metrics(rows).agent_metrics
    assert metrics["looping/rate"] == 0.5
    assert metrics["looping/coverage"] == pytest.approx(2 / 3)
    assert "mean/is_looping" not in metrics
    runs = []
    for labels in ([True, False], [False] * 8):
        data = [dict(reward=1, is_looping=flag, _ng_rollout_index=index) for index, flag in enumerate(labels)]
        aggregate = compute_aggregate_metrics(data)
        repeat_rows = [row | {"looping/rate": 0.5} for row in aggregate.repeat_level_metrics]
        runs.append(LoadedRun("agent", aggregate.agent_metrics, aggregate.key_metrics, [], repeat_rows))
    pooled = pool_loaded_runs(runs, source_labels=["a", "b"]).agent_metrics
    assert pooled["looping/rate"] == 0.1
    assert "mean_across_repeats/looping/rate" not in pooled
