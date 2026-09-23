# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compaction preserves scoring text across validated and direct Python inputs."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import resources_servers.genrm_compare.app as genrm
from resources_servers.genrm_compare.tests.test_cohort_lifecycle import member
from resources_servers.genrm_compare.tests.test_cohort_storage import assert_cohort_indices_match, training_member
from resources_servers.genrm_compare.utils import extract_from_response_obj


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
@pytest.mark.parametrize("shape", ["validated", "dict_response", "dict_output", "dict_content", "dict_summary"])
async def test_scoring_text_survives_dictionary_shaped_python_inputs(server, shape):
    requests = [training_member(i) for i in range(2)]
    raw = [body.response.model_dump() for body in requests]
    expected_texts = [extract_from_response_obj(response) for response in raw]
    assert all(reasoning and answer for reasoning, answer in expected_texts)
    for body, response in zip(requests, raw):
        if shape == "dict_response":
            body.response = response
        elif shape == "dict_output":
            body.response.output = response["output"]
        elif shape == "dict_content":
            body.response.output[1].content = response["output"][1]["content"]
        elif shape == "dict_summary":
            body.response.output[0].summary = response["output"][0]["summary"]

    compared = []

    async def compare(*, response_objs, **kwargs):
        compared.append([extract_from_response_obj(obj) for obj in response_objs])
        return [1.0, 2.0], {}, [], []

    server._run_compare = compare
    results = await asyncio.gather(*(server.verify(body) for body in requests))
    assert compared == [expected_texts]
    assert [result.reward for result in results] == [1.0, 2.0]
    assert [result.response.model_dump() for result in results] == raw


async def test_rejected_member_is_not_converted_and_cannot_fail_existing_group(server, monkeypatch):
    first = asyncio.create_task(server.verify(member(0)))
    await asyncio.sleep(0)
    cohort = next(iter(server._verify_cohorts.values()))

    def reject_conversion(*args):
        raise AssertionError("rejected request reached conversion")

    monkeypatch.setattr(server, "_comparison_response", reject_conversion)
    with pytest.raises(HTTPException) as error:
        await server.verify(member(0, response_id="conflicting-response"))
    assert error.value.status_code == 409
    assert cohort.phase == "collecting" and not first.done()
    assert_cohort_indices_match(server)
    await server.aclose()
    with pytest.raises(HTTPException):
        await first


async def test_active_judging_keeps_attempt_watermark_after_other_records_expire(server, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(genrm, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    server.config.cohort_result_ttl_s = 10
    server.config.max_terminal_cohorts = 1
    started = asyncio.Queue()
    release = asyncio.Event()

    async def compare(*, response_objs, **kwargs):
        started.put_nowait(response_objs[0]["id"])
        await release.wait()
        return [3.0, 3.0], {}, [], []

    server._run_compare = compare
    old = [asyncio.create_task(server.verify(member(i))) for i in range(2)]
    assert await asyncio.wait_for(started.get(), 1) == "answer-0"
    latest = [asyncio.create_task(server.verify(member(i, attempt=1, response_id=f"new-{i}"))) for i in range(2)]
    try:
        assert await asyncio.wait_for(started.get(), 1) == "new-0"
        old_results = await asyncio.gather(*old, return_exceptions=True)
        assert all(isinstance(r, HTTPException) and r.status_code == 503 for r in old_results)
        clock[0] += 11
        server._prune_terminal_cohorts()
        assert not server._terminal_cohorts
        watermark = server._latest_group_attempts["group"]
        assert watermark.latest_attempt == 1 and watermark.active_cohort.phase == "evaluating"
        assert all(not task.done() for task in latest)
        with pytest.raises(HTTPException) as error:
            await server.verify(member(0))
        assert error.value.status_code == 409
        assert_cohort_indices_match(server)
        release.set()
        assert [r.reward for r in await asyncio.gather(*latest)] == [3.0, 3.0]
        assert_cohort_indices_match(server)
    finally:
        release.set()
        await server.aclose()
        await asyncio.gather(*old, *latest, return_exceptions=True)
