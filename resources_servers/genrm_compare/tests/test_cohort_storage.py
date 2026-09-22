# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reward parity, response ownership, and retention behavior for compact cohorts."""

import asyncio
import gc
import weakref
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

import resources_servers.genrm_compare.app as genrm
from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.genrm_compare.tests.test_cohort_lifecycle import member
from resources_servers.genrm_compare.utils import extract_from_response_obj


def training_member(index, *, group="group", attempt=0):
    request = member(index, group=group, attempt=attempt)
    response = request.response.model_dump()
    response["output"] = [
        {
            "id": f"r{index}",
            "type": "reasoning",
            "summary": [
                {"type": "summary_text", "text": "reason " * (index + 1)},
            ],
            "prompt_token_ids": [10, 11],
            "generation_token_ids": [20, 21],
            "generation_log_probs": [-0.1, -0.2],
        },
        {
            "id": f"m{index}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "# Answer\n" + "**yes** " * (index + 1), "annotations": []},
                {"type": "output_text", "text": " another part", "annotations": []},
                {"type": "refusal", "refusal": "ignored by scoring"},
            ],
            "prompt_token_ids": [30, 31],
            "generation_token_ids": list(range(1024)),
            "generation_log_probs": [-0.3] * 1024,
        },
        {
            "id": f"call{index}",
            "type": "function_call",
            "call_id": f"c{index}",
            "name": "ignored_tool",
            "arguments": '{"x": 1}',
        },
    ]
    request.response = NeMoGymResponse.model_validate(response)
    request.principle = "Prefer concise, correct answers."
    return request


@pytest.mark.parametrize("strategy", ["circular", "all_pairs"])
@pytest.mark.parametrize("judges", [1, 3])
@pytest.mark.parametrize(
    "adjustment",
    [
        None,
        "reasoning_bonus",
        "answer_bonus",
        "group_reasoning_length_penalty_coeff",
        "group_answer_length_penalty_coeff",
        "group_style_penalty_coeff",
        "all",
    ],
)
async def test_compact_cohort_preserves_all_scoring_inputs_and_response_echo(server, strategy, judges, adjustment):
    server.config.num_rollouts_per_prompt = 16
    server.config.comparison_strategy = strategy
    server.config.num_judges_per_comparison = judges
    server.config.use_principle = True
    fields = [
        "reasoning_bonus",
        "answer_bonus",
        "group_reasoning_length_penalty_coeff",
        "group_answer_length_penalty_coeff",
        "group_style_penalty_coeff",
    ]
    for field in fields:
        setattr(server.config, field, 0.4 if adjustment in (field, "all") else 0)
    requests = [training_member(i) for i in range(16)]
    raw = [body.response.model_dump() for body in requests]
    server._run_single_comparison = AsyncMock(return_value=(3.0, 3.0, 3.5))
    reference = await server._run_compare([{"role": "user", "content": "2+2?"}], raw, principle=requests[0].principle)
    if adjustment is not None:
        assert reference[0] != [3.0] * 16
    actual_compare = server._run_compare
    compared = []

    async def compare(**kwargs):
        compact = kwargs["response_objs"]
        assert kwargs["principle"] == requests[0].principle
        assert [extract_from_response_obj(obj) for obj in compact] == [extract_from_response_obj(obj) for obj in raw]
        assert "generation_token_ids" not in str(compact)
        result = await actual_compare(**kwargs)
        compared.append(result)
        return result

    server._run_compare = compare
    tasks = {i: asyncio.create_task(server.verify(requests[i])) for i in reversed(range(16))}
    results = await asyncio.gather(*(tasks[i] for i in range(16)))
    assert compared == [reference]
    assert [r.reward for r in results] == reference[0]
    assert [r.response.model_dump() for r in results] == raw
    assert all(c.conversation_history == [] and c.principle is None for c in server._verify_cohorts.values())


@pytest.mark.parametrize("size", [2, 16])
async def test_each_unique_answer_is_compacted_once_without_a_full_comparison_dump(server, monkeypatch, size):
    server.config.num_rollouts_per_prompt = size
    requests = [training_member(i) for i in range(size)]
    started, release = asyncio.Event(), asyncio.Event()
    dump_modes, compact_ids = [], []
    original_dump = NeMoGymResponse.model_dump
    original_compact = getattr(server, "_comparison_response", None)

    def dump(self, **kwargs):
        dump_modes.append(kwargs.get("mode", "python"))
        return original_dump(self, **kwargs)

    def compact(response):
        compact_ids.append(response.id)
        return original_compact(response)

    async def judge(*args, **kwargs):
        started.set()
        await release.wait()
        return 3.0, 3.0, 3.5

    monkeypatch.setattr(NeMoGymResponse, "model_dump", dump)
    if original_compact is not None:
        monkeypatch.setattr(server, "_comparison_response", compact)
    server._run_single_comparison = judge
    tasks = [asyncio.create_task(server.verify(body)) for body in requests]
    await asyncio.wait_for(started.wait(), 1)
    duplicate = asyncio.create_task(server.verify(requests[0]))
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(*tasks, duplicate)
    await server.verify(requests[0])
    assert dump_modes == ["json"] * (size + 2)
    assert compact_ids == [r.response.id for r in requests]


async def test_training_token_change_still_conflicts_with_identical_text(server):
    requests = [training_member(i) for i in range(2)]
    server._run_single_comparison = AsyncMock(return_value=(3.0, 3.0, 3.5))
    await asyncio.gather(*(server.verify(body) for body in requests))
    changed = requests[0].model_copy(deep=True)
    changed.response.output[1].generation_token_ids[0] = 999
    with pytest.raises(HTTPException) as error:
        await server.verify(changed)
    assert error.value.status_code == 409
    assert (await server.verify(requests[0])).reward == 3.0


async def test_disconnected_request_can_be_freed_while_its_compact_answer_waits(server):
    request = training_member(0)
    body_ref, response_ref = weakref.ref(request), weakref.ref(request.response)
    task = asyncio.create_task(server.verify(request))
    await asyncio.sleep(0)
    cohort = next(iter(server._verify_cohorts.values()))
    assert len(cohort.members) == 1
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    del task, request
    await asyncio.sleep(0)
    gc.collect()
    assert body_ref() is None and response_ref() is None
    assert cohort.members[0].response_obj is not None
    server._run_single_comparison = AsyncMock(return_value=(3.0, 3.0, 3.5))
    results = await asyncio.gather(server.verify(training_member(0)), server.verify(training_member(1)))
    assert [r.reward for r in results] == [3.0, 3.0]


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(genrm, "time", SimpleNamespace(monotonic=lambda: now[0]))
    return now


async def complete(server, group):
    return await asyncio.gather(*(server.verify(member(i, group=group)) for i in range(2)))


async def test_replay_refreshes_watermark_but_not_terminal_expiry(server, clock):
    server.config.cohort_result_ttl_s = 10
    server._run_single_comparison = AsyncMock(return_value=(3.0, 3.0, 3.5))
    await complete(server, "a")
    clock[0] = 101
    await complete(server, "b")
    clock[0] = 109
    assert (await server.verify(member(0, group="a"))).reward == 3
    clock[0] = 110.5
    server._prune_terminal_cohorts()
    assert [c.group_id for c in server._verify_cohorts.values()] == ["b"]
    assert list(server._latest_group_attempts) == ["b", "a"]
    clock[0] = 111.5
    server._prune_terminal_cohorts()
    assert not server._verify_cohorts
    assert list(server._latest_group_attempts) == ["a"]


async def test_count_eviction_uses_completion_order_and_protects_active_attempts(server, clock):
    server.config.cohort_result_ttl_s = None
    server.config.max_terminal_cohorts = 1
    server._run_single_comparison = AsyncMock(return_value=(3.0, 3.0, 3.5))
    a = asyncio.create_task(server.verify(member(0, group="a")))
    await asyncio.sleep(0)
    clock[0] = 101
    await complete(server, "b")
    clock[0] = 102
    server._prune_terminal_cohorts()
    assert "a" in server._latest_group_attempts
    assert "b" not in server._latest_group_attempts
    await server.verify(member(1, group="a"))
    await a
    server._prune_terminal_cohorts()
    assert [c.group_id for c in server._verify_cohorts.values()] == ["a"]
    assert not server._active_group_cohorts


async def test_expired_active_watermark_does_not_block_expiry_of_later_completed_group(server, clock):
    server.config.cohort_result_ttl_s = 10
    server._run_single_comparison = AsyncMock(return_value=(3.0, 3.0, 3.5))
    active = asyncio.create_task(server.verify(member(0, group="active")))
    await asyncio.sleep(0)
    clock[0] = 101
    await complete(server, "done")
    clock[0] = 112
    server._prune_terminal_cohorts()
    assert list(server._latest_group_attempts) == ["active"]
    assert [c.group_id for c in server._verify_cohorts.values()] == ["active"]
    active.cancel()
    await asyncio.gather(active, return_exceptions=True)


def test_pruning_retained_records_does_not_scan_the_full_registry(server, clock):
    class NoScanDict(dict):
        def items(self):
            raise AssertionError("scanned all cohorts")

        def values(self):
            raise AssertionError("scanned all cohorts")

    class CountedOrder(OrderedDict):
        visited = 0

        def items(self):
            for item in super().items():
                self.visited += 1
                yield item

    for i in range(1024):
        key = str(i)
        cohort = genrm._CohortState(prompt_digest="prompt", key=key, group_id=key, phase="completed")
        server._verify_cohorts[key] = cohort
        if hasattr(server, "_record_terminal_cohort"):
            server._record_terminal_cohort(cohort)
        else:
            cohort.terminal_at = clock[0]
        server._latest_group_attempts[key] = genrm._GroupAttemptWatermark(0, "prompt", clock[0])
    original = server._verify_cohorts
    server._verify_cohorts = NoScanDict(original)
    watermarks = CountedOrder(server._latest_group_attempts)
    server._latest_group_attempts = watermarks
    try:
        server._prune_terminal_cohorts()
        assert len(server._verify_cohorts) == 1024
        assert watermarks.visited == 1
    finally:
        server._verify_cohorts = original


@pytest.mark.parametrize("strategy", ["circular", "all_pairs"])
@pytest.mark.parametrize("judges", [1, 3])
async def test_every_required_comparison_finishes_before_publish_without_blocking_other_groups(
    server, strategy, judges
):
    server.config.num_rollouts_per_prompt = 4
    server.config.comparison_strategy = strategy
    server.config.num_judges_per_comparison = judges
    expected = (4 if strategy == "circular" else 6) * judges
    completed = {"slow": 0, "fast": 0}
    release, ready = asyncio.Event(), asyncio.Event()

    async def judge(history, first, second, pair_idx, **kwargs):
        group = first["id"].split("-")[0]
        if group == "slow" and pair_idx == (0, 1):
            await release.wait()
        completed[group] += 1
        if completed["slow"] == expected - judges:
            ready.set()
        return 3.0, 3.0, 3.5

    def request(i, group):
        return member(i, group=group, response_id=f"{group}-{i}")

    server._run_single_comparison = judge
    slow = [asyncio.create_task(server.verify(request(i, "slow"))) for i in range(4)]
    try:
        await asyncio.wait_for(ready.wait(), 1)
        assert all(not t.done() for t in slow)
        fast = await asyncio.wait_for(asyncio.gather(*(server.verify(request(i, "fast")) for i in range(4))), 1)
        assert [r.reward for r in fast] == [3.0] * 4
        assert all(not t.done() for t in slow)
        release.set()
        result = await asyncio.wait_for(asyncio.gather(*slow), 1)
        assert [r.reward for r in result] == [3.0] * 4
        assert completed == {"slow": expected, "fast": expected}
    finally:
        release.set()
        await asyncio.gather(*slow, return_exceptions=True)
