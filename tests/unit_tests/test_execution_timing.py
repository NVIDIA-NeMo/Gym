# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from nemo_gym.execution_timing import ExecutionTiming
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle
from nemo_gym.sandbox.providers.opensandbox import provider as module


@pytest.mark.sandbox
@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("exit_code,error", [(0, None), (7, None), (None, "deadline exceeded")])
async def test_provider_uses_runtime_clocks_not_dispatch_or_poll_time(monkeypatch, background, exit_code, error):
    start = datetime(2026, 10, 23, tzinfo=timezone.utc)
    status = SimpleNamespace(
        running=False, exit_code=exit_code, error=error, started_at=start, finished_at=start + timedelta(seconds=2)
    )

    class Commands:
        async def run(self, command, opts):
            return SimpleNamespace(
                id="exec-1",
                logs=SimpleNamespace(stdout=[], stderr=[]),
                exit_code=exit_code,
                error=None if error is None else SimpleNamespace(name="Timeout", value=error),
                complete=SimpleNamespace(execution_time_in_millis=1900),
            )

        async def get_command_status(self, execution_id):
            assert execution_id == "exec-1"
            return status

        async def get_background_command_logs(self, execution_id):
            return SimpleNamespace(content="result")

    monkeypatch.setattr(module, "_require_opensandbox_sdk", lambda: (object, object, lambda **kw: kw, object, object))
    provider = module.OpenSandboxProvider(
        operations={"background_exec": background}, connection={"request_timeout_s": 1}
    )
    result = await provider.exec(
        SandboxHandle("sb", "opensandbox", SimpleNamespace(commands=Commands())), "echo result"
    )
    timing = result.execution_timing
    assert timing.started_at == start.timestamp()
    assert timing.completed_at == start.timestamp() + 2
    assert result.duration_ms == (2000 if background else 1900)
    assert timing.clock_domain == "opensandbox:sb:unix"
    assert timing.execution_id == "exec-1"
    assert timing.duration_source.endswith("wall_clock_delta" if background else "execution_complete")
    assert result.return_code == (125 if exit_code is None else exit_code)


@pytest.mark.parametrize("start,end", [(None, None), ("2026-10-23", None), (datetime(2026, 10, 23), None)])
def test_duration_alone_never_invents_an_endpoint(start, end):
    timing = module._execution_timing(
        SandboxHandle("sb", "opensandbox", None),
        "id",
        SimpleNamespace(started_at=start, finished_at=end),
        SimpleNamespace(execution_time_in_millis=500),
    )
    assert timing.started_at is None and timing.completed_at is None
    assert timing.duration_ms == 500


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True, "500"])
def test_invalid_optional_provider_duration_is_unknown(bad):
    timing = module._execution_timing(
        SandboxHandle("sb", "opensandbox", None), None, None, SimpleNamespace(execution_time_in_millis=bad)
    )
    assert timing.duration_ms is None
    assert SandboxExecResult("", "", 0).duration_ms is None


def test_clock_validation_and_reversed_runtime_interval():
    start = datetime(2026, 10, 23, tzinfo=timezone.utc)
    timing = module._execution_timing(
        SandboxHandle("sb", "opensandbox", None),
        None,
        SimpleNamespace(started_at=start, finished_at=start - timedelta(seconds=1)),
    )
    assert timing.started_at is None and timing.completed_at is None and timing.duration_ms is None
    for extra in ({"duration_ms": -1}, {"started_at": float("nan")}, {"started_at": 2, "completed_at": 1}):
        with pytest.raises(ValidationError):
            ExecutionTiming(clock_domain="clock", timing_source="runtime", **extra)


@pytest.mark.sandbox
async def test_foreground_timing_status_failure_keeps_command_outcome(monkeypatch):
    runs = []

    class Commands:
        async def run(self, command, opts):
            runs.append(command)
            return SimpleNamespace(
                id="exec",
                logs=SimpleNamespace(stdout=[], stderr=[]),
                exit_code=7,
                error=None,
                complete=SimpleNamespace(execution_time_in_millis=12),
            )

        async def get_command_status(self, execution_id):
            raise ConnectionError("status endpoint unavailable")

    monkeypatch.setattr(module, "_require_opensandbox_sdk", lambda: (object, object, lambda **kw: kw, object, object))
    result = await module.OpenSandboxProvider().exec(
        SandboxHandle("sb", "opensandbox", SimpleNamespace(commands=Commands())), "exit 7"
    )
    assert runs == ["exit 7"]
    assert result.return_code == 7 and result.duration_ms == 12
    assert result.execution_timing.started_at is None and result.execution_timing.completed_at is None


def test_partial_endpoint_is_preserved_without_duration():
    start = datetime(2026, 10, 23, tzinfo=timezone.utc)
    timing = module._execution_timing(
        SandboxHandle("sb", "opensandbox", None), "id", SimpleNamespace(started_at=start)
    )
    assert timing.started_at == start.timestamp()
    assert timing.completed_at is None and timing.duration_ms is None
