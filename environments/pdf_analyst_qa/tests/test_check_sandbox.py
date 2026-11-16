# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from environments.pdf_analyst_qa import check_sandbox


@pytest.mark.asyncio
@pytest.mark.parametrize("read_only", [True, False])
async def test_deployment_probe_switches_task_user_and_checks_mount_as_root(monkeypatch, read_only):
    sandbox = MagicMock(default_user=0)
    sandbox.start = AsyncMock()
    sandbox.stop = AsyncMock()
    sandbox.set_default_user.side_effect = lambda user: setattr(sandbox, "default_user", user)
    calls = []

    async def execute(command, *, user=None, timeout_s):
        uid = sandbox.default_user if user is None else user
        calls.append((command, uid))
        return SimpleNamespace(
            return_code=int(not read_only) if "touch" in command else 0,
            error_type=None,
            stdout=str(uid) if command == "id -u" else "",
            stderr="",
        )

    sandbox.exec = execute
    monkeypatch.setattr(check_sandbox, "AsyncSandbox", lambda provider: sandbox)
    monkeypatch.setattr(check_sandbox, "create_provider", lambda config: None)
    monkeypatch.setattr(check_sandbox, "resolve_provider_config", lambda *args: None)
    report = await check_sandbox.check(
        "test-image", "environments/pdf_analyst_qa/example.yaml", "http://model.example:8000", "/tmp/cache"
    )
    assert report["root"]["stdout"] == "0"
    assert report["default"]["stdout"] == "65534"
    assert next(uid for command, uid in calls if "network_probe.py" in command) == 65534
    assert next(uid for command, uid in calls if "touch" in command) == 0
    assert report["missing_view_mount"]["return_code"] == int(not read_only)
    sandbox.stop.assert_awaited_once()
