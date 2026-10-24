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
"""Shared request accounting under concurrency, retries, and cancellation."""

import asyncio

import pytest

from environments.skill_trigger.budget import BudgetReached, TrialBudget
from environments.skill_trigger.scoring import EvidenceError


def budget(*, preserve_payload=False, clock=lambda: 1.0):
    return TrialBudget(model="test-model", root_session_id="root", preserve_payload=preserve_payload, clock=clock)


def forward(trial, *, session="root", endpoint="/responses", payload=None):
    return trial.forward(
        session_id=session, endpoint=endpoint, payload={"model": "test-model"} if payload is None else payload
    )


async def test_setup_time_is_not_trial_time_but_first_request_cannot_reset_clock():
    now = [50.0]
    trial = budget(clock=lambda: now[0])
    with pytest.raises(EvidenceError, match="before prompt release"):
        async with forward(trial):
            pytest.fail("not released")
    now[0] = 1000.0
    trial.release()
    now[0] = 1120.0
    with pytest.raises(BudgetReached, match="timeout"):
        async with forward(trial):
            pytest.fail("deadline elapsed before first request")
    assert trial.requests == []


async def test_concurrent_children_and_retries_share_three_requests():
    trial = budget()
    trial.release()
    trial.add_session(session_id="child", parent_id="root")
    entered = []

    async def attempt(index):
        try:
            async with forward(trial, session="root" if index % 2 else "child"):
                entered.append(index)
                await asyncio.sleep(0.01)
            return "forwarded"
        except BudgetReached as error:
            return error.outcome

    results = await asyncio.gather(*(attempt(i) for i in range(12)))
    assert len(entered) == 3
    assert results.count("forwarded") == 3
    assert results.count("routing_horizon") == 9
    assert len(trial.requests) == 3
    assert {r.session_id for r in trial.requests} == {"root", "child"}
    await asyncio.wait_for(trial.wait_settled(), 1)


async def test_connection_errors_are_counted_and_do_not_become_budget_outcomes():
    trial = budget()
    trial.release()
    for _ in range(3):
        with pytest.raises(ConnectionError):
            async with forward(trial):
                raise ConnectionError("provider refused connection")
    with pytest.raises(BudgetReached, match="routing_horizon"):
        async with forward(trial):
            pytest.fail("retry exceeded shared allowance")


async def test_provider_timeout_is_not_misreported_as_trial_deadline():
    trial = budget()
    trial.release()
    with pytest.raises(TimeoutError, match="provider"):
        async with forward(trial):
            raise TimeoutError("provider connection timeout")


async def test_absolute_deadline_cancels_stream_and_runs_transport_cleanup():
    now = [0.0]
    trial = budget(clock=lambda: now[0])
    trial.release()
    now[0] = 119.97
    cleaned = False
    with pytest.raises(BudgetReached, match="timeout"):
        async with forward(trial):
            try:
                while True:
                    await asyncio.sleep(0.001)  # a stream can keep producing bytes
            finally:
                cleaned = True
    assert cleaned
    await asyncio.wait_for(trial.wait_settled(), 1)


async def test_external_cancellation_remains_cancellation_and_settles():
    trial = budget()
    trial.release()
    started = asyncio.Event()

    async def transport():
        async with forward(trial):
            started.set()
            await asyncio.Future()

    task = asyncio.create_task(transport())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(trial.requests) == 1
    await asyncio.wait_for(trial.wait_settled(), 1)


@pytest.mark.parametrize("endpoint", ["/responses", "/chat/completions"])
async def test_budget_overrides_preserve_instructions_tools_and_caller_payload(endpoint):
    trial = budget()
    trial.release()
    payload = {
        "model": "test-model",
        "input": [{"role": "user", "content": "original prompt"}],
        "tools": [{"type": "function", "name": "read"}],
        "reasoning": {"effort": "low"},
        "max_completion_tokens": 9000,
    }
    async with forward(trial, endpoint=endpoint, payload=payload) as sent:
        assert sent["input"] == payload["input"]
        assert sent["tools"] == payload["tools"]
        assert sent["max_output_tokens" if endpoint == "/responses" else "max_tokens"] == 8192
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["max_completion_tokens"] == 9000


async def test_pi_payload_is_preserved_exactly_and_cap_is_validated():
    trial = budget(preserve_payload=True)
    trial.release()
    payload = {"model": "test-model", "max_tokens": 8192, "reasoning_effort": "medium", "stream": True}
    async with forward(trial, endpoint="/chat/completions", payload=payload) as sent:
        assert sent == payload
    for cap in (None, 8193, "8192", True):
        with pytest.raises(EvidenceError, match="cap"):
            async with forward(trial, payload={"model": "test-model", "max_output_tokens": cap}):
                pytest.fail("invalid cap")
    assert len(trial.requests) == 1


async def test_unproven_session_model_substitution_and_stopped_trial_cannot_forward():
    trial = budget()
    trial.release()
    with pytest.raises(EvidenceError):
        async with forward(trial, session="unregistered-child"):
            pytest.fail("unproven child")
    with pytest.raises(EvidenceError):
        async with forward(trial, payload={"model": "different-model"}):
            pytest.fail("model substitution")
    trial.close()
    with pytest.raises(EvidenceError):
        async with forward(trial):
            pytest.fail("closed trial")
    with pytest.raises(EvidenceError):
        trial.release()
    assert trial.requests == []
