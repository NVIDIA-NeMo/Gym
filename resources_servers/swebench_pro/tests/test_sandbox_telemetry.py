# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SWE-bench Pro labels survive the resource owner's complete lifecycle."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pytest import MonkeyPatch

from nemo_gym.sandbox import api, resource_telemetry
from nemo_gym.sandbox.providers.base import SandboxHandle
from resources_servers.swebench_pro.app import SWEBenchProInstanceRequest
from resources_servers.swebench_pro.tests.test_app import make_server, request_body


@pytest.mark.asyncio
@pytest.mark.parametrize("task,role", [(True, "agent"), (False, "verifier")])
async def test_names_follow_resource_samples_and_cleanup(monkeypatch: MonkeyPatch, task: bool, role: str) -> None:
    records: list[tuple[str, dict]] = []

    @contextmanager
    def span(group, name, **attributes):
        records.append((name, attributes))
        yield attributes

    for module in (api, resource_telemetry):
        monkeypatch.setattr(module, "managed_span", span)
        monkeypatch.setattr(module, "safe_set_span_attributes", lambda target, attrs: target.update(attrs))
    monkeypatch.setattr(api, "is_span_group_enabled", lambda group: True)
    monkeypatch.setattr(api, "sampling_interval", lambda: 100)
    current_rollout = ["original-rollout"]
    monkeypatch.setattr(
        api,
        "default_span_attributes",
        lambda explicit=None: {"nemo.gym.rollout.id": current_rollout[0], **(explicit or {})},
    )
    provider = SimpleNamespace(
        name="fake",
        create=AsyncMock(return_value=SandboxHandle("box", "fake", None)),
        close=AsyncMock(),
        aclose=AsyncMock(),
        resource_allocation=AsyncMock(return_value=resource_telemetry.ResourceAllocation(source="test")),
        resource_sample=AsyncMock(return_value={"nemo.gym.sandbox.sample.memory_used_mib": 3.0}),
    )
    monkeypatch.setattr(api, "create_provider", lambda config: provider)
    monkeypatch.setattr("resources_servers.swebench_pro.app.get_global_config_dict", lambda: {"test": {"fake": {}}})
    server = make_server(golden=False, apply_anti_cheating=False)
    sandbox = await server._create_sandbox(SWEBenchProInstanceRequest.model_validate(request_body()), task=task)
    try:
        await asyncio.sleep(0)
        current_rollout[0] = "different-cleanup-context"
    finally:
        await sandbox.stop()

    by_name = {name for name, _ in records}
    assert by_name == {
        "gym.sandbox.request",
        "gym.sandbox.start",
        "gym.sandbox.resource.allocation",
        "gym.sandbox.resource.sample",
        "gym.sandbox.stop",
    }
    for _, attrs in records:
        assert attrs["nemo.gym.sandbox.name"] == f"{server.config.name}/{role}"
        assert attrs["nemo.gym.sandbox.role"] == role
        assert attrs["nemo.gym.rollout.id"] == "original-rollout"
    assert len({attrs["nemo.gym.sandbox.request.id"] for _, attrs in records}) == 1
    samples = [attrs for name, attrs in records if name == "gym.sandbox.resource.sample"]
    assert len(samples) == 2
    assert samples[-1]["nemo.gym.sandbox.resource.final"] is True
    assert samples[-1]["nemo.gym.sandbox.resource.status"] == "ok"
    provider.close.assert_awaited_once()
    provider.aclose.assert_awaited_once()
