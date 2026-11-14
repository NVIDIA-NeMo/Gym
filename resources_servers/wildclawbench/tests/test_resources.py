# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource cleanup retains ownership until the native container is removed."""

import sys

import pytest

from responses_api_agents.wildclawbench_agent.execution import NativeProcess
from responses_api_agents.wildclawbench_agent.tests.helpers import make_pipeline


async def test_failed_cleanup_keeps_owner_and_unknown_close_is_idempotent(tmp_path, monkeypatch):
    _, _, resources, _ = make_pipeline(tmp_path, sys.executable, monkeypatch)
    process = NativeProcess(resources.config.execution, tmp_path, "sha256:" + "a" * 64)
    resources._owned["session"] = process

    async def failed(self, stage, timeout):
        raise RuntimeError("container runtime unavailable")

    monkeypatch.setattr(NativeProcess, "invoke", failed)
    with pytest.raises(RuntimeError):
        await resources.release_session_state("session")
    assert resources.has_session_state("session")
    calls = []

    async def succeeded(self, stage, timeout):
        calls.append(stage)
        return {}

    monkeypatch.setattr(NativeProcess, "invoke", succeeded)
    await resources.release_session_state("session")
    await resources.release_session_state("session")
    await resources.release_session_state("unknown")
    assert calls == ["cleanup"]
    assert not resources.has_session_state("session")
