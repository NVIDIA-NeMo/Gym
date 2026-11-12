# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientResponseError

from environment_servers.swe_together import model_client
from environment_servers.swe_together.model_client import AuxiliaryModel, error_evidence
from environment_servers.swe_together.simulator import UserAgent


GATEWAY_TIMEOUT = b'{"error":{"message":"litellm.Timeout: Connection timed out. Timeout passed=360.0"}}'


def http_error(status=408, content=GATEWAY_TIMEOUT):
    url = "http://127.0.0.1:1/ng-rollout/one/v1/chat/completions"
    request_info = SimpleNamespace(url=url, real_url=url, method="POST", headers={})
    error = ClientResponseError(request_info=request_info, history=(), status=status, message="Request Timeout")
    error.response_content = content
    return error


async def test_failed_call_keeps_model_server_status_body_and_timing(monkeypatch):
    monkeypatch.setattr(model_client, "raise_for_status", AsyncMock(side_effect=http_error()))
    model = AuxiliaryModel(SimpleNamespace(post=AsyncMock()), "sim", cookies={}, temperature=0.5)
    with pytest.raises(ClientResponseError):
        await model.call("prompt")
    record = model.calls[-1]
    assert record["status_code"] == 408
    assert "Timeout passed=360.0" in record["body"]
    assert record["elapsed_seconds"] >= 0 and record["started_at_unix_seconds"] > 0


async def test_cancelled_call_keeps_its_duration():
    started = asyncio.Event()

    async def hang(*args, **kwargs):
        started.set()
        await asyncio.sleep(60)

    model = AuxiliaryModel(SimpleNamespace(post=hang), "sim", cookies={}, temperature=0.5)
    task = asyncio.create_task(model.call("prompt"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert model.calls[-1]["cancelled"] is True and model.calls[-1]["elapsed_seconds"] >= 0


async def test_simulator_wrapper_keeps_the_upstream_cause():
    simulator = UserAgent(SimpleNamespace(call=AsyncMock(side_effect=http_error())))
    with pytest.raises(RuntimeError) as raised:
        await simulator.process("task", "activity", "report", None, 1, True)
    evidence = error_evidence(raised.value)
    assert evidence["message"] == "User simulator request failed"
    assert evidence["cause"].startswith("ClientResponseError: 408")
    assert evidence["status_code"] == 408 and "litellm.Timeout" in evidence["body"]


def test_error_without_http_cause_has_no_status():
    assert error_evidence(ValueError("asset digest mismatch")) == {
        "type": "ValueError",
        "message": "asset digest mismatch",
    }
