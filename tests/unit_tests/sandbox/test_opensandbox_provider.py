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

import asyncio
import builtins
import gc
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from nemo_gym.sandbox.providers.base import SandboxResources, SandboxSpec, SandboxStatus


pytestmark = pytest.mark.sandbox


pytest.importorskip("tenacity", reason="tenacity optional sandbox dependency is not installed")

from nemo_gym.core.failure_kinds import (
    PROVIDER_COMMAND_OUTCOME_UNKNOWN,
    PROVIDER_CREATE_PENDING,
    PROVIDER_CREATE_TIMEOUT,
    PROVIDER_IO_TIMEOUT,
    PROVIDER_OOM_KILLED,
    PROVIDER_REQUEST_FAILED,
    PROVIDER_STATUS_UNAVAILABLE,
    PROVIDER_UNAVAILABLE,
)
from nemo_gym.core.failure_transport import RequestFailureLog
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.sandbox.failures import sandbox_failure
from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider


@contextmanager
def raises_failure(kind, **kwargs):
    with pytest.raises(NgFailureEvent, **kwargs) as caught:
        yield caught
    assert caught.value.failure_kind == kind


TEST_REGISTRY_PASSWORD = "secret"  # pragma: allowlist secret


@pytest.mark.parametrize(
    "outcome", ["success", "exit-124", "timeout", "cancel", "timeout-stream-error", "cancel-stream-error"]
)
async def test_session_execution_releases_session_before_cancelling(outcome: str, monkeypatch) -> None:
    unhandled_errors = []
    monkeypatch.setattr(asyncio.get_running_loop(), "call_exception_handler", unhandled_errors.append)
    started = asyncio.Event()
    stopped = asyncio.Event()
    calls = []

    class Commands:
        async def create_session(self, **kwargs):
            calls.append("create")
            assert kwargs == {"working_directory": "/work"}
            return "session-1"

        async def run_in_session(self, session_id, command, **kwargs):
            assert session_id == "session-1"
            assert "solve.sh" in command
            started.set()
            if outcome.startswith(("timeout", "cancel")):
                try:
                    await stopped.wait()
                except asyncio.CancelledError:
                    assert stopped.is_set(), "cancelled the client before stopping the remote process group"
                    raise
                if outcome.endswith("stream-error"):
                    raise RuntimeError("output stream closed")
            calls.append("finished")
            return SimpleNamespace(
                logs=SimpleNamespace(stdout=[SimpleNamespace(text="solution output")], stderr=[]),
                exit_code=124 if outcome == "exit-124" else 0,
                error=None,
            )

        async def delete_session(self, session_id):
            assert session_id == "session-1"
            calls.append("delete")
            stopped.set()

    provider = opensandbox_provider.OpenSandboxProvider(connection={"request_timeout_s": 1})
    handle = opensandbox_provider.SandboxHandle("sandbox-1", "opensandbox", SimpleNamespace(commands=Commands()))
    task = asyncio.create_task(
        provider.exec_with_background_services(handle, "bash solve.sh", cwd="/work", timeout_s=0.05)
    )
    if outcome.startswith("cancel"):
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        result = await task
        assert result.stdout == (None if outcome.endswith("stream-error") else "solution output")
        assert result.return_code == (124 if outcome.startswith("timeout") or outcome == "exit-124" else 0)
        assert result.error_type == ("timeout" if outcome.startswith("timeout") else None)
    assert calls.count("delete") == 1
    if outcome in ("timeout", "cancel"):
        assert calls.index("delete") < calls.index("finished")
    del task
    # Let completed-task callbacks release their references before checking unhandled errors.
    await asyncio.sleep(0)
    gc.collect()
    assert unhandled_errors == []


@pytest.mark.parametrize("failure_phase", ["create", "run", "delete"])
async def test_session_provider_request_timeout_preserves_unknown_outcome(failure_phase: str) -> None:
    calls: list[str] = []

    class Commands:
        async def create_session(self, **kwargs: Any) -> str:
            calls.append("create")
            if failure_phase == "create":
                raise TimeoutError("session create request expired")
            return "session-1"

        async def run_in_session(self, session_id: str, command: str, **kwargs: Any) -> Any:
            calls.append("run")
            if failure_phase == "run":
                raise TimeoutError("session run request expired")
            return SimpleNamespace(logs=SimpleNamespace(stdout=[], stderr=[]), exit_code=0, error=None)

        async def delete_session(self, session_id: str) -> None:
            calls.append("delete")
            if failure_phase == "delete":
                raise TimeoutError("session delete request expired")

    provider = opensandbox_provider.OpenSandboxProvider(connection={"request_timeout_s": 1})
    handle = opensandbox_provider.SandboxHandle("session-timeout", "opensandbox", SimpleNamespace(commands=Commands()))

    with raises_failure(PROVIDER_IO_TIMEOUT, match="failed") as caught:
        await provider.exec_with_background_services(handle, "echo completed", timeout_s=5)

    assert not isinstance(caught.value, TimeoutError)
    assert calls == (["create"] if failure_phase == "create" else ["create", "run", "delete"])


@pytest.mark.parametrize("user", [None, "root", "runner 'quoted'", 1000])
async def test_session_execution_preserves_env_and_user(tmp_path, user):
    import os
    import shlex

    from nemo_gym.sandbox.api import AsyncSandbox

    value = "spaces 'quotes' $HOME $(touch should-not-exist)\nsecond line"
    calls = []

    class Commands:
        async def create_session(self, **kwargs):
            assert kwargs == {"working_directory": str(tmp_path)}
            return "session-1"

        async def run_in_session(self, session_id, command, **kwargs):
            calls.append("run")
            assert session_id == "session-1"
            assert kwargs == {"timeout": timedelta(0)}
            shell, flag, inner = shlex.split(command)
            assert (shell, flag) == ("bash", "-c")
            if user is not None:
                # Check the account-switch boundary, then execute its payload as
                # the test user; this test does not require host root privileges.
                su, shell_flag, login_shell, command_flag, inner, separator, account = shlex.split(inner)
                assert (su, shell_flag, login_shell, command_flag, separator) == ("su", "-s", "/bin/sh", "-c", "--")
                assert account == ("$(id -nu -- 1000)" if user == 1000 else user)
            proc = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                inner,
                cwd=tmp_path,
                env={**os.environ, "GREETING": "wrong default"},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            return SimpleNamespace(
                logs=SimpleNamespace(
                    stdout=[SimpleNamespace(text=stdout.decode())], stderr=[SimpleNamespace(text=stderr.decode())]
                ),
                exit_code=proc.returncode,
                error=None,
            )

        async def delete_session(self, session_id):
            assert session_id == "session-1"
            calls.append("delete")

    provider = opensandbox_provider.OpenSandboxProvider(connection={"request_timeout_s": 1})
    sandbox = AsyncSandbox(provider)
    sandbox._handle = opensandbox_provider.SandboxHandle(
        "sandbox-1", "opensandbox", SimpleNamespace(commands=Commands())
    )
    sandbox._stopped = False
    result = await sandbox.exec(
        'printf "%s" "$GREETING"',
        cwd=str(tmp_path),
        env={"GREETING": value},
        user=user,
        timeout_s=10,
        preserve_background_services=True,
    )
    assert result.return_code == 0 and result.stdout == value
    assert not (tmp_path / "should-not-exist").exists()
    assert calls == ["run", "delete"]


@pytest.mark.parametrize("recovers", [True, False])
@pytest.mark.parametrize("timeout_kind", ["client", "server", "readiness"])
async def test_tb4_create_timeout_retries_with_jitter(monkeypatch, recovers, timeout_kind):
    import tenacity
    import tenacity.wait
    import yaml
    from opensandbox.exceptions import SandboxApiException, SandboxReadyTimeoutException

    config = yaml.safe_load(Path("benchmarks/terminal_bench_4/resources.yaml").read_text())
    environment = config["terminal_bench_4"]["resources_servers"]["terminal_bench_4"]["environment"]
    provider = opensandbox_provider.OpenSandboxProvider(
        create=environment["sandbox_provider"]["opensandbox"]["create"]
    )
    errors = {
        "client": sandbox_failure(operation="create", failure_kind="provider_create_timeout", reason="create timeout"),
        "server": SandboxApiException("POD_READY_TIMEOUT: BATCHSANDBOX_PENDING", status_code=504),
        "readiness": SandboxReadyTimeoutException("readiness timeout"),
    }
    handle = opensandbox_provider.SandboxHandle(sandbox_id="ready", provider_name="opensandbox", raw=None)
    provider._create_once = AsyncMock(
        side_effect=[errors[timeout_kind]] * 5 + [handle if recovers else errors[timeout_kind]]
    )
    sleep = AsyncMock()
    monkeypatch.setattr(
        opensandbox_provider,
        "_require_tenacity",
        lambda: (
            lambda **kwargs: tenacity.AsyncRetrying(sleep=sleep, **kwargs),
            tenacity.retry_if_exception,
            tenacity.stop_after_attempt,
            tenacity.wait_random_exponential,
        ),
    )
    bounds = []

    def midpoint(low, high):
        bounds.append((low, high))
        return (low + high) / 2

    monkeypatch.setattr(tenacity.wait.random, "uniform", midpoint)
    if recovers:
        assert await provider.create(SandboxSpec(image="task")) is handle
    else:
        with pytest.raises(NgFailureEvent) as caught:
            await provider.create(SandboxSpec(image="task"))
        assert caught.value.failure_kind == (
            "provider_unavailable" if timeout_kind == "server" else "provider_create_timeout"
        )
    assert provider._create_once.await_count == 6
    assert [call.args[0] for call in sleep.await_args_list] == [2.5, 5, 10, 20, 30]
    assert bounds[:5] == [(0, 5), (0, 10), (0, 20), (0, 40), (0, 60)]


@pytest.mark.parametrize("abandon", ["timeout", "cancel"])
async def test_create_deletes_the_sandbox_of_an_abandoned_request(
    monkeypatch: pytest.MonkeyPatch, abandon: str
) -> None:
    """A create the caller gave up on before the server answered still finishes, and its sandbox is deleted."""
    server_answers = asyncio.Event()
    killed: list[str] = []

    class SlowSandbox(FakeSandbox):
        @classmethod
        async def create(cls, **_kwargs: Any) -> "FakeSandbox":
            await server_answers.wait()
            return cls("late-sandbox")

        async def kill(self) -> None:
            killed.append(self.id)

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (SlowSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    create = {"retries": 0}
    if abandon == "timeout":
        create["timeout_s"] = 0.01
    provider = opensandbox_provider.OpenSandboxProvider(create=create, probe={"command": None})
    sweeps: list[dict[str, str]] = []

    async def server_has_not_listed_it(metadata: dict[str, str], _attempt_key: str) -> list[dict[str, Any]]:
        sweeps.append(metadata)
        return []

    # The attempt-key sweep runs first and finds nothing: the server has not recorded the sandbox yet.
    monkeypatch.setattr(provider, "_reap_orphaned_create", server_has_not_listed_it)

    if abandon == "timeout":
        with raises_failure(PROVIDER_CREATE_TIMEOUT):
            await provider.create(SandboxSpec(image="image:tag"))
    else:
        task = asyncio.create_task(provider.create(SandboxSpec(image="image:tag")))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    # The caller is released before the server answers; nothing is deleted yet.
    assert len(sweeps) == 1
    assert killed == []
    assert len(provider._abandoned_creates) == 1

    server_answers.set()
    await provider.aclose()
    assert killed == ["late-sandbox"]
    assert provider._abandoned_creates == set()


@pytest.mark.parametrize("abandon", ["timeout", "cancel"])
async def test_create_stops_waiting_once_the_sweep_deleted_its_sandbox(
    monkeypatch: pytest.MonkeyPatch, abandon: str
) -> None:
    """The server never answers a create whose Pending sandbox was deleted, so its request is cancelled."""
    request_cancelled = asyncio.Event()

    class HungSandbox(FakeSandbox):
        @classmethod
        async def create(cls, **_kwargs: Any) -> "FakeSandbox":
            try:
                await asyncio.get_running_loop().create_future()
            except asyncio.CancelledError:
                request_cancelled.set()
                raise
            return cls()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (HungSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    create = {"retries": 0}
    if abandon == "timeout":
        create["timeout_s"] = 0.01
    provider = opensandbox_provider.OpenSandboxProvider(create=create, probe={"command": None})

    async def sweep_deletes_it(_metadata: dict[str, str], _attempt_key: str) -> list[dict[str, Any]]:
        return [{"id": "pending-sandbox", "state": "Pending", "reason": "CREATING", "message": None}]

    monkeypatch.setattr(provider, "_reap_orphaned_create", sweep_deletes_it)

    if abandon == "timeout":
        with raises_failure(PROVIDER_CREATE_PENDING):
            await provider.create(SandboxSpec(image="image:tag"))
    else:
        task = asyncio.create_task(provider.create(SandboxSpec(image="image:tag")))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    await asyncio.wait_for(request_cancelled.wait(), timeout=1)
    await asyncio.wait_for(asyncio.gather(*provider._abandoned_creates), timeout=1)
    assert provider._abandoned_creates == set()


async def test_aclose_cancels_abandoned_creates_the_server_never_answers(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cancelled: list[bool] = []

    class HungSandbox(FakeSandbox):
        @classmethod
        async def create(cls, **_kwargs: Any) -> "FakeSandbox":
            try:
                await asyncio.get_running_loop().create_future()
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
            return cls()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (HungSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"timeout_s": 0.01, "retries": 0},
        operations={"close_timeout_s": 0.05},
        probe={"command": None},
    )
    with raises_failure(PROVIDER_CREATE_TIMEOUT):
        await provider.create(SandboxSpec(image="image:tag"))

    with caplog.at_level(logging.WARNING):
        await asyncio.wait_for(provider.aclose(), timeout=5)
    assert cancelled == [True]
    assert provider._abandoned_creates == set()
    assert "Cancelling 1 abandoned OpenSandbox creates" in caplog.text


async def test_create_does_not_retry_rejected_credentials():
    from opensandbox.exceptions import SandboxApiException

    provider = opensandbox_provider.OpenSandboxProvider(create={"retries": 5})
    provider._create_once = AsyncMock(side_effect=SandboxApiException("unauthorized", status_code=401))
    with pytest.raises(NgFailureEvent) as caught:
        await provider.create(SandboxSpec(image="task"))
    assert caught.value.failure_kind == "provider_credentials_rejected"
    assert caught.value.http_status == 401
    provider._create_once.assert_awaited_once()


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_renewal_rejects_invalid_intervals(interval: float) -> None:
    with pytest.raises(ValueError, match="renew_interval_s"):
        opensandbox_provider.OpenSandboxCreateConfig(renew_interval_s=interval)


@pytest.mark.parametrize("ttl", [None, 1, 2])
async def test_renewal_requires_ttl_longer_than_interval(ttl: int | None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(create={"renew_interval_s": 2, "retries": 0})
    with pytest.raises(ValueError, match="longer, explicit"):
        await provider.create(SandboxSpec(image="example", ttl_s=ttl))


async def test_renewal_refreshes_ttl_and_stops_before_termination(fake_opensandbox_sdk: None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(create={"renew_interval_s": 0.01}, probe={"command": None})
    handle = await provider.create(SandboxSpec(image="example", ttl_s=28800))
    renewed = asyncio.Event()
    durations = []

    async def renew(duration: timedelta) -> None:
        durations.append(duration)
        renewed.set()

    async def kill() -> None:
        assert handle.sandbox_id not in provider._renewals

    handle.raw.renew = renew
    handle.raw.kill = AsyncMock(side_effect=kill)
    handle.raw.close = AsyncMock()
    await asyncio.wait_for(renewed.wait(), timeout=1)
    task = provider._renewals[handle.sandbox_id]
    await provider.close(handle)
    assert durations and all(duration == timedelta(seconds=28800) for duration in durations)
    assert task.cancelled()
    handle.raw.kill.assert_awaited_once()
    handle.raw.close.assert_awaited_once()
    await provider.aclose()


async def test_renewal_failure_surfaces_and_still_allows_cleanup(fake_opensandbox_sdk: None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"renew_interval_s": 0.01}, probe={"command": None}, operations={"retries": 0}
    )
    handle = await provider.create(SandboxSpec(image="example", ttl_s=28800))
    handle.raw.renew = AsyncMock(side_effect=RuntimeError("renewal rejected"))
    handle.raw.kill = AsyncMock()
    handle.raw.close = AsyncMock()
    task = provider._renewals[handle.sandbox_id]
    with pytest.raises(NgFailureEvent) as observed_failure:
        await asyncio.wait_for(asyncio.shield(task), timeout=1)
    assert observed_failure.value.failure_kind == "provider_request_failed"
    operation = AsyncMock()
    with pytest.raises(NgFailureEvent, match="lifetime renewal failed") as observed_failure:
        await provider._await_sdk_operation(operation, operation="get_info", sandbox_id=handle.sandbox_id, timeout_s=1)
    assert observed_failure.value.failure_kind == "provider_unavailable"
    operation.assert_not_awaited()
    with pytest.raises(NgFailureEvent, match="lifetime renewal failed") as observed_failure:
        await provider.close(handle)
    assert observed_failure.value.failure_kind == "provider_unavailable"
    handle.raw.kill.assert_awaited_once()
    handle.raw.close.assert_awaited_once()
    await provider.aclose()


async def test_provider_shutdown_cancels_renewal_without_terminating_sandbox(fake_opensandbox_sdk: None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(create={"renew_interval_s": 10}, probe={"command": None})
    handle = await provider.create(SandboxSpec(image="example", ttl_s=28800))
    handle.raw.kill = AsyncMock()
    task = provider._renewals[handle.sandbox_id]
    await provider.aclose()
    assert task.cancelled()
    assert not provider._renewals
    handle.raw.kill.assert_not_awaited()


@pytest.mark.parametrize(
    ("create", "match"),
    [
        ({"renew_ttl_s": 1200}, "renew_ttl_s requires a shorter"),
        ({"renew_ttl_s": 300, "renew_interval_s": 300}, "renew_ttl_s requires a shorter"),
        ({"renew_ttl_s": 0, "renew_interval_s": 1}, "renew_ttl_s must be finite"),
        ({"renew_ttl_s": float("inf"), "renew_interval_s": 1}, "renew_ttl_s must be finite"),
        ({"renew_timeout_s": 0}, "renew_timeout_s must be finite"),
        ({"renew_timeout_s": float("nan")}, "renew_timeout_s must be finite"),
    ],
)
def test_lease_settings_are_validated(create: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        opensandbox_provider.OpenSandboxCreateConfig(**create)


async def test_a_lease_caps_the_spec_ttl_at_create_and_in_every_renewal(
    fake_opensandbox_sdk: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"renew_ttl_s": 1200, "renew_interval_s": 0.01, "renew_timeout_s": 7}, probe={"command": None}
    )
    budgets: list[float | None] = []
    await_operation = provider._await_sdk_operation

    async def recording(factory: Any, **kwargs: Any) -> Any:
        if kwargs["operation"] == "renew":
            budgets.append(kwargs["timeout_s"])
        return await await_operation(factory, **kwargs)

    monkeypatch.setattr(provider, "_await_sdk_operation", recording)
    # Without a spec TTL the lease is the only lifetime; with a longer one the lease still caps each extension.
    for ttl_s in (None, 18000):
        handle = await provider.create(SandboxSpec(image="example", ttl_s=ttl_s))
        assert FakeSandbox.created_kwargs["timeout"] == timedelta(seconds=1200)
        durations: list[timedelta] = []
        renewed = asyncio.Event()

        async def renew(duration: timedelta) -> None:
            durations.append(duration)
            if len(durations) >= 2:
                renewed.set()

        handle.raw.renew = renew
        handle.raw.kill = AsyncMock()
        handle.raw.close = AsyncMock()
        await asyncio.wait_for(renewed.wait(), timeout=1)
        await provider.close(handle)
        assert set(durations) == {timedelta(seconds=1200)}
    assert budgets and set(budgets) == {7}
    await provider.aclose()


async def test_a_spec_ttl_within_the_lease_is_used_as_is_and_never_renewed(fake_opensandbox_sdk: None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"renew_ttl_s": 1200, "renew_interval_s": 0.01}, probe={"command": None}
    )
    handle = await provider.create(SandboxSpec(image="example", ttl_s=600))
    assert FakeSandbox.created_kwargs["timeout"] == timedelta(seconds=600)
    assert handle.sandbox_id not in provider._renewals
    await provider.aclose()


async def test_a_lease_is_never_renewed_past_the_spec_lifetime(fake_opensandbox_sdk: None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"renew_ttl_s": 0.2, "renew_interval_s": 0.05}, probe={"command": None}
    )
    handle = await provider.create(SandboxSpec(image="example", ttl_s=0.5))
    durations: list[float] = []

    async def renew(duration: timedelta) -> None:
        durations.append(duration.total_seconds())

    handle.raw.renew = renew
    task = provider._renewals[handle.sandbox_id]
    # The loop ends by itself once the spec's lifetime is used up; the service then ends the sandbox at its TTL.
    await asyncio.wait_for(asyncio.shield(task), timeout=2)
    assert durations and all(0 < duration <= 0.2 for duration in durations)
    assert min(durations) < 0.2, "the last extensions are cut to what is left of the spec's lifetime"
    await provider.aclose()


async def test_a_failed_lease_renewal_is_logged_and_the_next_one_tries_again(
    fake_opensandbox_sdk: None, caplog: pytest.LogCaptureFixture
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"renew_ttl_s": 1200, "renew_interval_s": 0.01}, probe={"command": None}, operations={"retries": 0}
    )
    handle = await provider.create(SandboxSpec(image="example", ttl_s=28800))
    attempts: list[int] = []
    renewed = asyncio.Event()

    async def flaky_renew(duration: timedelta) -> None:
        attempts.append(len(attempts))
        if len(attempts) <= 2:
            raise RuntimeError("renewal rejected")
        renewed.set()

    handle.raw.renew = flaky_renew
    handle.raw.kill = AsyncMock()
    handle.raw.close = AsyncMock()
    task = provider._renewals[handle.sandbox_id]
    with caplog.at_level(logging.WARNING):
        await asyncio.wait_for(renewed.wait(), timeout=1)
    assert not task.done(), "the loop survives failed renewals"
    assert "lifetime renewal failed" in caplog.text
    # Other operations and the close are not held hostage by earlier renewal failures.
    operation = AsyncMock(return_value="ok")
    assert (
        await provider._await_sdk_operation(operation, operation="get_info", sandbox_id=handle.sandbox_id, timeout_s=1)
        == "ok"
    )
    await provider.close(handle)
    assert task.cancelled()
    handle.raw.kill.assert_awaited_once()
    await provider.aclose()


def _skip_sleeps(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Make every sleep move the event loop's clock forward instead of waiting; returns the real sleep."""
    loop = asyncio.get_running_loop()
    real_time, real_sleep = loop.time, asyncio.sleep
    skipped = 0.0

    async def skipping_sleep(delay: float) -> None:
        nonlocal skipped
        skipped += delay
        await real_sleep(0)

    monkeypatch.setattr(loop, "time", lambda: real_time() + skipped)
    monkeypatch.setattr(asyncio, "sleep", skipping_sleep)
    return real_sleep


async def test_the_spec_lifetime_is_counted_from_the_create_request_not_from_when_renewal_starts(
    fake_opensandbox_sdk: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease_s, interval_s, lifetime_s, startup_s = 200, 50, 1000, 120
    monkeypatch.setattr(opensandbox_provider.random, "uniform", lambda low, high: high)
    real_sleep = _skip_sleeps(monkeypatch)
    loop = asyncio.get_running_loop()
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"renew_ttl_s": lease_s, "renew_interval_s": interval_s}, probe={"command": None}
    )
    sdk_create = FakeSandbox.create.__func__

    async def slow_create(cls: Any, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(startup_s)
        return await sdk_create(cls, *args, **kwargs)

    monkeypatch.setattr(FakeSandbox, "create", classmethod(slow_create))
    ends_at: list[float] = []

    async def renew(self: Any, duration: timedelta) -> None:
        ends_at.append(loop.time() - created_at + duration.total_seconds())

    monkeypatch.setattr(FakeSandbox, "renew", renew, raising=False)

    created_at = loop.time()
    handle = await provider.create(SandboxSpec(image="example", ttl_s=lifetime_s))
    task = provider._renewals[handle.sandbox_id]
    for _ in range(5000):
        if task.done():
            break
        await real_sleep(0)
    assert task.done() and task.exception() is None, "the loop ends once the spec's lifetime is used up"
    await provider.aclose()

    # No renewal extends the sandbox past the spec's ttl_s from its create request, whatever the startup took.
    assert max(ends_at) == pytest.approx(lifetime_s, abs=5)
    assert all(end <= lifetime_s + 1 for end in ends_at)


async def test_a_failed_lease_renewal_is_retried_before_the_lease_runs_out(
    fake_opensandbox_sdk: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every try times out while the API is down from just before the first renewal until 150 s before the lease
    ends. A full interval's wait after each failure would put the next try after the expiry."""
    lease_s, interval_s, timeout_s, outage_ends_s = 1200, 300, 30, 1050
    monkeypatch.setattr(opensandbox_provider.random, "uniform", lambda low, high: high)
    real_sleep = _skip_sleeps(monkeypatch)
    loop = asyncio.get_running_loop()
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"renew_ttl_s": lease_s, "renew_interval_s": interval_s, "renew_timeout_s": timeout_s},
        probe={"command": None},
        operations={"retries": 0},
    )
    tried_at: list[float] = []
    renewed_at: list[float] = []

    async def renew(self: Any, duration: timedelta) -> None:
        tried_at.append(loop.time() - created_at)
        if tried_at[-1] < outage_ends_s:
            await asyncio.sleep(timeout_s)
            raise RuntimeError("renew timed out")
        renewed_at.append(tried_at[-1])

    monkeypatch.setattr(FakeSandbox, "renew", renew, raising=False)

    created_at = loop.time()
    await provider.create(SandboxSpec(image="example"))
    for _ in range(5000):
        if len(renewed_at) >= 2:
            break
        await real_sleep(0)
    await provider.aclose()

    assert tried_at[0] == pytest.approx(interval_s, abs=5)
    assert len([at for at in tried_at if at < outage_ends_s]) >= 3
    assert outage_ends_s <= renewed_at[0] < lease_s, "renewed before the lease taken at create would expire"
    gaps = [later - earlier for earlier, later in zip(tried_at, tried_at[1:])]
    assert max(gaps) <= interval_s + timeout_s + 5, "a retry never waits longer than the interval"
    assert renewed_at[1] - renewed_at[0] == pytest.approx(interval_s, abs=5), "then the interval is kept again"


@pytest.mark.parametrize(
    ("remaining_s", "expected_s"),
    [(870, 300), (540, 255), (255, 112.5), (31, 1.0), (10, 1.0), (0, 300), (-60, 300)],
    ids=["far", "nearer", "near", "floor", "inside-one-timeout", "expired", "long-expired"],
)
def test_a_renewal_retry_waits_half_of_what_is_left_of_the_lease_and_never_past_the_interval(
    remaining_s: float, expected_s: float
) -> None:
    assert opensandbox_provider._renewal_retry_delay(remaining_s, interval_s=300, timeout_s=30) == expected_s


def _sandbox_gone() -> RuntimeError:
    gone = RuntimeError("Renew sandbox failed: [KUBERNETES::SANDBOX_NOT_FOUND]")
    gone.status_code = 404  # type: ignore[attr-defined]
    return gone


async def test_a_lease_renewal_that_finds_the_sandbox_gone_ends_quietly(
    fake_opensandbox_sdk: None, caplog: pytest.LogCaptureFixture
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"renew_ttl_s": 1200, "renew_interval_s": 0.01}, probe={"command": None}, operations={"retries": 0}
    )
    handle = await provider.create(SandboxSpec(image="example", ttl_s=28800))
    handle.raw.renew = AsyncMock(side_effect=_sandbox_gone())
    handle.raw.kill = AsyncMock()
    handle.raw.close = AsyncMock()
    task = provider._renewals[handle.sandbox_id]

    with caplog.at_level(logging.WARNING, logger=opensandbox_provider.__name__):
        await asyncio.wait_for(asyncio.shield(task), timeout=1)
        await asyncio.sleep(0)  # let the task's done callback run

    assert task.exception() is None and handle.raw.renew.await_count == 1
    assert "lifetime renewal failed" not in caplog.text
    # A stopped or deleted sandbox is not a renewal failure: later operations and the close go ahead.
    await provider.close(handle)
    await provider.aclose()


async def test_a_renewal_without_a_lease_that_finds_the_sandbox_gone_fails_the_sandbox(
    fake_opensandbox_sdk: None, caplog: pytest.LogCaptureFixture
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"renew_interval_s": 0.01}, probe={"command": None}, operations={"retries": 0}
    )
    handle = await provider.create(SandboxSpec(image="example", ttl_s=28800))
    gone = _sandbox_gone()
    handle.raw.renew = AsyncMock(side_effect=gone)
    task = provider._renewals[handle.sandbox_id]

    # Without a lease nothing but this loop keeps the sandbox alive, so losing it is reported, not passed over.
    with caplog.at_level(logging.ERROR, logger=opensandbox_provider.__name__):
        with pytest.raises(NgFailureEvent) as ended:
            await asyncio.wait_for(asyncio.shield(task), timeout=1)
        await asyncio.sleep(0)  # let the task's done callback run

    assert ended.value.__cause__ is gone and handle.raw.renew.await_count == 1
    [record] = [record for record in caplog.records if "lifetime renewal failed" in record.getMessage()]
    assert record.levelno == logging.ERROR and repr(handle.sandbox_id) in record.getMessage()
    operation = AsyncMock()
    with pytest.raises(NgFailureEvent, match="lifetime renewal failed"):
        await provider._await_sdk_operation(operation, operation="get_info", sandbox_id=handle.sandbox_id, timeout_s=1)
    operation.assert_not_awaited()
    await provider.aclose()


@pytest.mark.parametrize(
    ("request_s", "setup_s", "first_renewal_s"),
    [(0, 250, 300), (0, 700, 700), (250, 0, 300), (700, 0, 700), (200, 200, 400)],
    ids=[
        "setup-within-one-interval",
        "setup-longer-than-one-interval",
        "create-request-within-one-interval",
        "create-request-longer-than-one-interval",
        "create-request-and-setup-longer-than-one-interval",
    ],
)
async def test_the_first_renewal_counts_the_time_since_the_create_request_was_sent(
    fake_opensandbox_sdk: None,
    monkeypatch: pytest.MonkeyPatch,
    request_s: float,
    setup_s: float,
    first_renewal_s: float,
) -> None:
    """The server starts the TTL when it receives the create request and answers once the sandbox is ready, so
    the time the request stays open (placement, image pull) is spent from the lease like the setup after it."""
    lease_s, interval_s = 320, 300
    monkeypatch.setattr(opensandbox_provider.random, "uniform", lambda low, high: high)
    # The event loop's clock, which every sleep moves forward instead of waiting.
    loop = asyncio.get_running_loop()
    real_time, real_sleep = loop.time, asyncio.sleep
    skipped = 0.0

    async def skipping_sleep(delay: float) -> None:
        nonlocal skipped
        skipped += delay
        await real_sleep(0)

    monkeypatch.setattr(loop, "time", lambda: real_time() + skipped)
    monkeypatch.setattr(asyncio, "sleep", skipping_sleep)
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"renew_ttl_s": lease_s, "renew_interval_s": interval_s}, probe={"command": None}
    )

    async def slow_setup(handle: Any) -> None:
        await asyncio.sleep(setup_s)

    monkeypatch.setattr(provider, "_run_setup_commands", slow_setup)
    sdk_create = FakeSandbox.create.__func__

    async def slow_create(cls: Any, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(request_s)
        return await sdk_create(cls, *args, **kwargs)

    monkeypatch.setattr(FakeSandbox, "create", classmethod(slow_create))
    renewed_at: list[float] = []

    async def renew(self: Any, duration: timedelta) -> None:
        renewed_at.append(loop.time() - created_at)

    monkeypatch.setattr(FakeSandbox, "renew", renew, raising=False)

    created_at = loop.time()
    await provider.create(SandboxSpec(image="example"))
    for _ in range(1000):
        if len(renewed_at) >= 2:
            break
        await real_sleep(0)
    await provider.aclose()

    # The lease has been running since the create request was sent, not since it returned or the setup ended.
    assert renewed_at[0] == pytest.approx(first_renewal_s, abs=5)
    if request_s + setup_s < interval_s:
        assert renewed_at[0] < lease_s, "renewed before the lease taken at create would expire"
    assert renewed_at[1] - renewed_at[0] == pytest.approx(interval_s, abs=5), "later renewals keep the interval"


@pytest.mark.parametrize(
    ("create", "ttl_s", "interval_s"),
    [
        ({"renew_ttl_s": 1200, "renew_interval_s": 300}, None, 300),
        # A TTL barely above the interval: a renewal later than the interval could miss it.
        ({"renew_interval_s": 95}, 100, 95),
    ],
    ids=["lease", "spec-ttl-just-above-the-interval"],
)
async def test_the_renewal_interval_is_jittered_by_up_to_ten_percent_earlier_and_never_later(
    fake_opensandbox_sdk: None,
    monkeypatch: pytest.MonkeyPatch,
    create: dict[str, Any],
    ttl_s: int | None,
    interval_s: int,
) -> None:
    bounds: list[tuple[float, float]] = []
    monkeypatch.setattr(opensandbox_provider.random, "uniform", lambda low, high: bounds.append((low, high)) or high)
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def recording_sleep(delay: float) -> None:
        sleeps.append(delay)
        await real_sleep(0)

    # Patched before create, so the renewal loop's first wait is recorded too, and returns at once.
    monkeypatch.setattr(asyncio, "sleep", recording_sleep)
    provider = opensandbox_provider.OpenSandboxProvider(create=create, probe={"command": None})
    handle = await provider.create(SandboxSpec(image="example", ttl_s=ttl_s))
    handle.raw.renew = AsyncMock()
    handle.raw.kill = AsyncMock()
    handle.raw.close = AsyncMock()
    async with asyncio.timeout(1):
        while handle.raw.renew.await_count < 3:
            await real_sleep(0)
    await provider.close(handle)
    await provider.aclose()

    assert bounds and set(bounds) == {(0.9, 1.0)}
    # The longest wait the jitter allows is the interval itself; the first is shorter by what the create took.
    assert sleeps.count(interval_s) >= 2
    assert max(sleeps) == interval_s


@dataclass(frozen=True)
class FakePlatformSpec:
    os: str
    arch: str


class FakeConnectionConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def with_transport_if_missing(self) -> "FakeConnectionConfig":
        return self

    async def close_transport_if_owned(self) -> None:
        return None


@dataclass(frozen=True)
class FakeVolume:
    name: str


class FakeSandbox:
    created_kwargs: dict[str, Any] = {}
    connected_args: tuple[Any, ...] = ()
    connected_state = "RUNNING"
    ready_calls = 0
    resumed_args: tuple[Any, ...] = ()
    resumed_kwargs: dict[str, Any] = {}

    def __init__(self, sandbox_id: str = "sandbox-1") -> None:
        self.id = sandbox_id

    @classmethod
    async def create(cls, *_args: Any, **kwargs: Any) -> "FakeSandbox":
        cls.created_kwargs = kwargs
        return cls()

    @classmethod
    async def connect(cls, *args: Any, **kwargs: Any) -> "FakeSandbox":
        cls.connected_args = args
        sandbox = cls(str(args[0]))
        if not kwargs.get("skip_health_check", False):
            await sandbox.check_ready(
                kwargs["connect_timeout"],
                kwargs.get("health_check_polling_interval", timedelta(milliseconds=200)),
            )
        return sandbox

    @classmethod
    async def resume(cls, *args: Any, **kwargs: Any) -> "FakeSandbox":
        cls.resumed_args = args
        cls.resumed_kwargs = kwargs
        return cls(str(args[0]))

    async def get_info(self) -> Any:
        return SimpleNamespace(status=SimpleNamespace(state=type(self).connected_state))

    async def check_ready(self, _timeout: timedelta, _polling_interval: timedelta) -> None:
        if type(self).connected_state == "PAUSED":
            raise RuntimeError("a paused sandbox has no exec daemon")
        type(self).ready_calls += 1

    async def close(self) -> None:
        return None


@pytest.fixture
def fake_opensandbox_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSandbox.connected_state = "RUNNING"
    FakeSandbox.ready_calls = 0
    FakeSandbox.resumed_args = ()
    FakeSandbox.resumed_kwargs = {}

    def require_sdk() -> tuple[Any, Any, Any, Any, Any]:
        return FakeSandbox, FakeConnectionConfig, object, FakePlatformSpec, object

    monkeypatch.setattr(opensandbox_provider, "_require_opensandbox_sdk", require_sdk)


def test_sdk_info_logs_are_silenced(caplog: pytest.LogCaptureFixture) -> None:
    sdk_logger = logging.getLogger("opensandbox.sandbox")

    with caplog.at_level(logging.INFO):
        sdk_logger.info("SDK info")
        sdk_logger.warning("SDK warning")

    sdk_messages = [record.message for record in caplog.records if record.name == sdk_logger.name]
    assert sdk_messages == ["SDK warning"]


def test_sdk_missing_sandbox_terminate_warning_is_silenced(caplog: pytest.LogCaptureFixture) -> None:
    sdk_logger = logging.getLogger("opensandbox.adapters.sandboxes_adapter")

    with caplog.at_level(logging.WARNING):
        sdk_logger.warning(
            "Failed to terminate sandbox sandbox-1: Kill sandbox sandbox-1 failed: [KUBERNETES::SANDBOX_NOT_FOUND]"
        )
        sdk_logger.warning("Failed to terminate sandbox sandbox-2: backend unavailable")
        sdk_logger.warning("Sandbox sandbox-3 not found | [KUBERNETES::SANDBOX_NOT_FOUND]")

    sdk_messages = [record.message for record in caplog.records if record.name == sdk_logger.name]
    assert sdk_messages == [
        "Failed to terminate sandbox sandbox-2: backend unavailable",
        "Sandbox sandbox-3 not found | [KUBERNETES::SANDBOX_NOT_FOUND]",
    ]


def test_sdk_import_helpers_and_retry_classification() -> None:
    assert len(opensandbox_provider._require_opensandbox_sdk()) == 5
    assert len(opensandbox_provider._require_tenacity()) == 4

    class StatusCodeError(Exception):
        status_code = 429

    assert opensandbox_provider._exception_status_code(StatusCodeError("rate limited")) == 429
    assert opensandbox_provider._is_retryable_create_error(
        sandbox_failure(operation="create", failure_kind="provider_internal_error", reason="create failed")
    )

    from opensandbox.exceptions import (  # noqa: PLC0415
        InvalidArgumentException,
        SandboxApiException,
        SandboxException,
        SandboxInternalException,
    )

    assert opensandbox_provider._is_retryable_create_error(InvalidArgumentException("bad input")) is False
    assert opensandbox_provider._is_retryable_create_error(SandboxInternalException("server failed")) is True

    retryable_api_error = SandboxApiException("busy")
    retryable_api_error.status_code = 503
    assert opensandbox_provider._is_retryable_create_error(retryable_api_error) is True

    nonretryable_api_error = SandboxApiException("not found")
    nonretryable_api_error.status_code = 404
    assert opensandbox_provider._is_retryable_create_error(nonretryable_api_error) is False
    assert opensandbox_provider._is_retryable_create_error(SandboxException("gateway timeout")) is False

    status_only_not_found = SandboxApiException("gone")
    status_only_not_found.status_code = 404
    assert opensandbox_provider._is_missing_sandbox_delete_error(status_only_not_found) is True
    assert (
        opensandbox_provider._is_missing_sandbox_delete_error(RuntimeError("[KUBERNETES::SANDBOX_NOT_FOUND]")) is True
    )
    assert opensandbox_provider._is_missing_sandbox_delete_error(RuntimeError("sandbox sandbox-1 not found")) is True
    assert opensandbox_provider._is_missing_sandbox_delete_error(RuntimeError("http 500 boom")) is False

    retry_state = SimpleNamespace(
        outcome=SimpleNamespace(exception=lambda: RuntimeError("temporary")),
        next_action=SimpleNamespace(sleep=0.5),
        attempt_number=2,
    )
    opensandbox_provider._log_create_retry(retry_state)


def test_missing_optional_dependency_import_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def block_imports(*blocked_names: str) -> None:
        def fake_import(
            name: str,
            globals_: dict[str, Any] | None = None,
            locals_: dict[str, Any] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> Any:
            if any(name == blocked or name.startswith(f"{blocked}.") for blocked in blocked_names):
                raise ModuleNotFoundError(name)
            return real_import(name, globals_, locals_, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

    block_imports("opensandbox")
    with pytest.raises(ModuleNotFoundError, match="OpenSandbox SDK is required"):
        opensandbox_provider._require_opensandbox_sdk()

    block_imports("tenacity")
    with pytest.raises(ModuleNotFoundError, match="tenacity is required"):
        opensandbox_provider._require_tenacity()

    block_imports("opensandbox.exceptions")
    assert opensandbox_provider._is_retryable_create_error(RuntimeError("gateway timeout")) is False


async def test_provider_conversion_helpers(
    fake_opensandbox_sdk: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection_config = opensandbox_provider.OpenSandboxConnectionConfig(domain="sandbox.example")
    assert (
        opensandbox_provider._coerce_config(connection_config, opensandbox_provider.OpenSandboxConnectionConfig)
        is connection_config
    )

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, object, FakePlatformSpec, FakeVolume),
    )
    assert opensandbox_provider._to_volumes([{"name": "workspace"}]) == [FakeVolume(name="workspace")]


async def test_direct_create_passes_platform_to_sdk_create(
    fake_opensandbox_sdk: None,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 10},
        probe={"command": None},
    )

    handle = await provider.create(
        SandboxSpec(
            image="mirror.gcr.io/astral/uv:python3.12-bookworm-slim",
            provider_options={"platform": {"os": "linux", "arch": "amd64"}},
        ),
    )

    assert handle.sandbox_id == "sandbox-1"
    assert FakeSandbox.created_kwargs["platform"] == FakePlatformSpec(
        os="linux",
        arch="amd64",
    )
    assert "network_policy" not in FakeSandbox.created_kwargs


async def test_direct_create_passes_network_policy_to_sdk_create(fake_opensandbox_sdk: None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    policy = {
        "defaultAction": "deny",
        "egress": [{"action": "allow", "target": "pypi.org"}],
    }

    await provider.create(SandboxSpec(image="image:tag", provider_options={"network_policy": policy}))

    assert FakeSandbox.created_kwargs["network_policy"].model_dump(by_alias=True, exclude_none=True) == policy


async def test_direct_create_passes_resource_requests_to_sdk_create(
    fake_opensandbox_sdk: None,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})

    await provider.create(
        SandboxSpec(
            image="mirror.gcr.io/astral/uv:python3.12-bookworm-slim",
            resources={"cpu": 1, "memory_mib": 8192, "disk_gib": 30},
            provider_options={"resource_requests": {"cpu": 0.5, "memory_mib": 2048, "disk_gib": 30}},
        ),
    )

    assert FakeSandbox.created_kwargs["resource"] == {"cpu": "1", "memory": "8192Mi", "ephemeral-storage": "30Gi"}
    assert FakeSandbox.created_kwargs["resource_requests"] == {
        "cpu": "0.5",
        "memory": "2048Mi",
        "ephemeral-storage": "30Gi",
    }

    with pytest.raises(TypeError, match="'resource_requests' must be a mapping"):
        opensandbox_provider.OpenSandboxProviderOptions.from_mapping({"resource_requests": "big"})

    with pytest.raises(ValueError, match="Unknown sandbox resource keys"):
        await provider.create(
            SandboxSpec(
                image="mirror.gcr.io/astral/uv:python3.12-bookworm-slim",
                provider_options={"resource_requests": {"memory_gib": 2}},
            ),
        )


@pytest.mark.parametrize(
    ("resources", "expected"),
    [
        (
            {"cpu": 4, "memory_mib": 4096, "disk_gib": 10},
            {"cpu": "4", "memory": "4096Mi", "ephemeral-storage": "10Gi"},
        ),
        ({"cpu": 0.5, "memory_mib": 512}, {"cpu": "0.5", "memory": "512Mi"}),
        (
            {"cpu": 4, "memory_mib": 16384, "gpu": 1, "gpu_type": "H100"},
            {"cpu": "4", "memory": "16384Mi", "gpu": "1", "gpu_type": "H100"},
        ),
        ({}, {"cpu": "1", "memory": "2048Mi"}),
    ],
)
async def test_explicit_requests_match_each_sandbox_limits(fake_opensandbox_sdk, resources, expected):
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    await provider.create(
        SandboxSpec(image="image:tag", resources=resources, provider_options={"resource_requests": "limits"})
    )
    assert FakeSandbox.created_kwargs["resource"] == expected
    assert FakeSandbox.created_kwargs["resource_requests"] == expected
    assert FakeSandbox.created_kwargs["resource_requests"] is not FakeSandbox.created_kwargs["resource"]


async def test_direct_create_passes_image_auth_to_sdk_create(
    fake_opensandbox_sdk: None,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})

    await provider.create(
        SandboxSpec(
            image="registry.example/repo:tag",
            provider_options={"image_auth": {"username": "user", "password": TEST_REGISTRY_PASSWORD}},
        )
    )

    image = FakeSandbox.created_kwargs["image"]
    assert image.image == "registry.example/repo:tag"
    assert image.auth.username == "user"
    assert image.auth.password == TEST_REGISTRY_PASSWORD


async def test_pool_create_uses_sdk_compatibility_image_and_proxy_auth(
    fake_opensandbox_sdk: None,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={
            "domain": "http://sandbox.example/",
            "api_key": "pool-api-key",  # pragma: allowlist secret
            "request_timeout_s": 30,
            "use_server_proxy": True,
        },
        create={
            "request_timeout_s": 120,
            "timeout_s": 30,
        },
        probe={"command": None},
    )
    handle = await provider.create(
        SandboxSpec(
            image="busybox:1.36",
            ttl_s=1800,
            metadata={"purpose": "osworld"},
            provider_options={
                "skip_health_check": True,
                "extensions": {"poolRef": "osworld-kvm"},
            },
        )
    )

    assert handle.sandbox_id == "sandbox-1"
    assert FakeSandbox.created_kwargs["image"] == "busybox:1.36"
    assert FakeSandbox.created_kwargs["timeout"] == timedelta(seconds=1800)
    assert FakeSandbox.created_kwargs["extensions"]["poolRef"] == "osworld-kvm"
    assert FakeSandbox.created_kwargs["metadata"]["purpose"] == "osworld"
    assert FakeSandbox.created_kwargs["skip_health_check"] is True
    create_connection = FakeSandbox.created_kwargs["connection_config"]
    # A scheme in the configured domain is split into protocol + bare host before it reaches the SDK.
    assert create_connection.kwargs["domain"] == "sandbox.example"
    assert create_connection.kwargs["protocol"] == "http"
    assert create_connection.kwargs["headers"] == {
        "OPEN-SANDBOX-API-KEY": "pool-api-key"  # pragma: allowlist secret
    }
    assert FakeSandbox.connected_args == ()


async def test_endpoint_normalizes_missing_scheme_and_merges_sdk_headers() -> None:
    class FakeRaw:
        connection_config = SimpleNamespace(
            get_base_url=lambda: "https://sandbox.example/v1",
            headers={
                "OPEN-SANDBOX-API-KEY": "pool-api-key",  # pragma: allowlist secret
                "X-Shared": "connection",
            },
        )

        async def get_endpoint(self, port: int) -> Any:
            assert port == 5000
            return SimpleNamespace(
                endpoint="10.0.0.22:5000",
                headers={"X-Route": "sandbox", "X-Shared": "endpoint"},
            )

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={
            "domain": "https://sandbox.example/",
            "api_key": "pool-api-key",  # pragma: allowlist secret
            "use_server_proxy": True,
        },
        operations={"retries": 0},
        probe={"command": None},
    )
    resolved = await provider.endpoint(
        opensandbox_provider.SandboxHandle(
            sandbox_id="sandbox-1",
            provider_name="opensandbox",
            raw=FakeRaw(),
        ),
        5000,
    )

    assert resolved.endpoint == "https://10.0.0.22:5000"
    assert resolved.headers == {
        "OPEN-SANDBOX-API-KEY": "pool-api-key",  # pragma: allowlist secret
        "X-Shared": "endpoint",
        "X-Route": "sandbox",
    }


async def test_endpoint_uses_effective_sdk_scheme_when_provider_input_is_unset() -> None:
    class FakeRaw:
        connection_config = SimpleNamespace(
            get_base_url=lambda: "https://gateway.example/v1",
            headers={},
        )

        async def get_endpoint(self, _port: int) -> Any:
            return SimpleNamespace(endpoint="sandbox.example:5000", headers={})

    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"retries": 0},
        probe={"command": None},
    )
    resolved = await provider.endpoint(
        opensandbox_provider.SandboxHandle(
            sandbox_id="sandbox-1",
            provider_name="opensandbox",
            raw=FakeRaw(),
        ),
        5000,
    )

    assert resolved.endpoint == "https://sandbox.example:5000"


async def test_direct_endpoint_never_receives_management_api_key() -> None:
    class FakeRaw:
        connection_config = SimpleNamespace(headers={})

        async def get_endpoint(self, _port: int) -> Any:
            return SimpleNamespace(endpoint="http://10.0.0.22:5000", headers={})

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={
            "api_key": "pool-api-key",  # pragma: allowlist secret
            "use_server_proxy": False,
        },
        operations={"retries": 0},
        probe={"command": None},
    )
    resolved = await provider.endpoint(
        opensandbox_provider.SandboxHandle(
            sandbox_id="sandbox-1",
            provider_name="opensandbox",
            raw=FakeRaw(),
        ),
        5000,
    )

    assert resolved.headers == {}


async def test_endpoint_requires_sdk_get_endpoint() -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"retries": 0},
        probe={"command": None},
    )
    with pytest.raises(NotImplementedError, match="opensandbox>=0.1.15"):
        await provider.endpoint(
            opensandbox_provider.SandboxHandle(
                sandbox_id="sandbox-1",
                provider_name="opensandbox",
                raw=object(),
            ),
            5000,
        )


def test_provider_validation_and_retry_helpers() -> None:
    with pytest.raises(ValueError, match="image_pull_policy"):
        opensandbox_provider.validate_image_pull_policy("Sometimes")
    with pytest.raises(TypeError, match="extensions"):
        opensandbox_provider.OpenSandboxProviderOptions.from_mapping({"extensions": ["not", "a", "mapping"]})
    with pytest.raises(TypeError, match="must be a bool"):
        opensandbox_provider.OpenSandboxProviderOptions.from_mapping({"skip_health_check": "true"})

    assert opensandbox_provider._resource_map(SandboxResources(cpu=2.0))["cpu"] == "2"
    assert opensandbox_provider._to_sandbox_status("starting") == SandboxStatus.STARTING
    assert opensandbox_provider._to_sandbox_status("pausing") == SandboxStatus.STARTING
    assert opensandbox_provider._to_sandbox_status("paused") == SandboxStatus.PAUSED
    assert opensandbox_provider._to_sandbox_status("terminated") == SandboxStatus.STOPPED
    assert opensandbox_provider._to_sandbox_status("failed") == SandboxStatus.ERROR
    assert opensandbox_provider._to_sandbox_status(None) == SandboxStatus.UNKNOWN

    invalid_kwargs = [
        {"create": {"timeout_s": 0}},
        {"probe": {"timeout_s": 0}},
        {"probe": {"deadline_s": 0}},
        {"probe": {"stable_count": 0}},
        {"probe": {"stable_delay_s": -1}},
        {"create": {"retries": -1}},
        {"create": {"retry_delay_s": -1}},
        {"create": {"retry_max_delay_s": -1}},
        {"operations": {"retries": -1}},
        {"operations": {"retry_delay_s": -1}},
        {"operations": {"retry_max_delay_s": -1}},
        {"operations": {"command_retries": -1}},
        {"operations": {"close_timeout_s": 0}},
        {"operations": {"status_poll_timeout_s": 0}},
        {"operations": {"pause_resume_timeout_s": 0}},
        {"create": {"connect_attempt_timeout_s": 0}},
        {"create": {"connect_poll_s": 0}},
        {"create": {"image_pull_policy": "Sometimes"}},
    ]
    for kwargs in invalid_kwargs:
        with pytest.raises(ValueError):
            opensandbox_provider.OpenSandboxProvider(**kwargs)
    with pytest.raises(TypeError):
        opensandbox_provider.OpenSandboxProvider(**{"batch_" + "create_retries": 1})
    with pytest.raises(TypeError):
        opensandbox_provider.OpenSandboxProvider(connection=object())
    with pytest.raises(TypeError, match="diagnostics.inspect_oom must be a bool"):
        opensandbox_provider.OpenSandboxProvider(diagnostics={"inspect_oom": "true"})

    assert opensandbox_provider._exception_status_code(RuntimeError("HTTP status code: 503")) == 503
    assert opensandbox_provider._exception_status_code(RuntimeError("plain error")) is None
    attrs = opensandbox_provider._sdk_error_attributes(
        RuntimeError("HTTP 502 bad gateway"),
        operation="exec",
        sandbox_id="sandbox-1",
        attempt_number=2,
        max_attempts=3,
        sleep_s=0.5,
    )
    assert attrs["status_code"] == 502
    assert attrs["attempt_number"] == 2
    assert attrs["next_sleep_s"] == 0.5


def test_provider_options_from_mapping() -> None:
    options_cls = opensandbox_provider.OpenSandboxProviderOptions

    assert options_cls.from_mapping(None) == options_cls()

    parsed = options_cls.from_mapping(
        {
            "image_auth": {"username": "user", "password": TEST_REGISTRY_PASSWORD},
            "network_policy": {"defaultAction": "allow", "egress": []},
            "platform": {"os": "linux", "arch": "amd64"},
            "snapshot_id": "snap-1",
            "volumes": [{"name": "workspace"}],
            "skip_health_check": True,
            "extensions": {"imagePullPolicy": "Never"},
        }
    )
    assert parsed.image_auth == {"username": "user", "password": TEST_REGISTRY_PASSWORD}
    assert parsed.network_policy == {"defaultAction": "allow", "egress": []}
    assert parsed.platform == {"os": "linux", "arch": "amd64"}
    assert parsed.snapshot_id == "snap-1"
    assert parsed.volumes == ({"name": "workspace"},)
    assert parsed.skip_health_check is True
    assert parsed.extensions == {"imagePullPolicy": "Never"}

    with pytest.raises(ValueError, match="Unknown OpenSandbox provider option"):
        options_cls.from_mapping({"bogus": 1})
    with pytest.raises(TypeError, match="provider_options must be a mapping"):
        options_cls.from_mapping(["not", "a", "mapping"])
    with pytest.raises(TypeError, match="'platform' must be a mapping"):
        options_cls.from_mapping({"platform": "linux/amd64"})
    with pytest.raises(TypeError, match="'image_auth' must be a mapping"):
        options_cls.from_mapping({"image_auth": "not-a-mapping"})
    with pytest.raises(TypeError, match="'network_policy' must be a mapping"):
        options_cls.from_mapping({"network_policy": "allow"})
    with pytest.raises(TypeError, match="'snapshot_id' must be a string"):
        options_cls.from_mapping({"snapshot_id": 123})
    with pytest.raises(TypeError, match="'volumes' must be a list of mappings"):
        options_cls.from_mapping({"volumes": ["workspace"]})


def test_connection_config_and_image_policy(fake_opensandbox_sdk: None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={
            "domain": "sandbox.example/",
            "api_key": "key",  # pragma: allowlist secret
            "protocol": "https",
            "request_timeout_s": 10,
            "use_server_proxy": True,
        }
    )

    config = provider._connection_config()
    transport = config.kwargs.pop("transport")
    assert isinstance(transport, httpx.AsyncBaseTransport)
    assert config.kwargs == {
        "domain": "sandbox.example",
        "api_key": "key",  # pragma: allowlist secret
        "protocol": "https",
        "request_timeout": timedelta(seconds=10),
        "use_server_proxy": True,
        # The API key must also travel as a header: the SDK's execd clients
        # (health ping, commands, files) send only ConnectionConfig.headers,
        # and proxied /proxy/* routes may enforce auth.
        "headers": {"OPEN-SANDBOX-API-KEY": "key"},  # pragma: allowlist secret
    }
    short_timeout_config = provider._connection_config(request_timeout_s=3)
    assert short_timeout_config.kwargs["request_timeout"] == timedelta(seconds=3)

    # Direct-endpoint mode must NOT carry the key: the sandbox runs untrusted
    # code and would be able to read it.
    direct = opensandbox_provider.OpenSandboxProvider(
        connection={"domain": "sandbox.example", "api_key": "key"}  # pragma: allowlist secret
    )
    assert "headers" not in direct._connection_config().kwargs


def test_connection_transport_backends(fake_opensandbox_sdk: None) -> None:
    from nemo_gym.sandbox.providers._http_transport import GymAiohttpTransport

    # The default backend borrows Gym's global aiohttp client.
    provider = opensandbox_provider.OpenSandboxProvider()
    transport = provider._build_transport()
    assert isinstance(transport, GymAiohttpTransport)

    # Custom pool settings still produce an httpx transport.
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={
            "transport_backend": "httpx",
            "keepalive_expiry_s": 2.5,
            "max_connections": 7,
            "max_keepalive_connections": 3,
            "connect_retries": 1,
        }
    )
    transport = provider._build_transport()
    assert isinstance(transport, httpx.AsyncHTTPTransport)
    # connect_retries reaches the pool rather than silently falling back.
    assert transport._pool._retries == 1

    # SDK transport defaults are sufficient when certificate verification is enabled.
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"transport_backend": "httpx", "keepalive_expiry_s": None, "tls_verify": True}
    )
    config = provider._connection_config()
    assert "transport" not in config.kwargs

    # max_connections=null uncaps the pool; max_keepalive_connections=0 disables reuse.
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"transport_backend": "httpx", "max_connections": None, "max_keepalive_connections": 0}
    )
    transport = provider._build_transport()
    assert isinstance(transport, httpx.AsyncHTTPTransport)
    assert transport._pool._max_connections > 2**32
    assert transport._pool._max_keepalive_connections == 0


async def test_connection_transport_is_shared_and_closed_by_provider(fake_opensandbox_sdk: None) -> None:
    # The SDK never closes a transport it did not create, so the provider owns
    # one: built on first use, reused by every ConnectionConfig rather than
    # leaking a pool per call, and closed in aclose().
    class FakeTransport:
        def __init__(self) -> None:
            self.aclosed = False

        async def aclose(self) -> None:
            self.aclosed = True

    provider = opensandbox_provider.OpenSandboxProvider()
    provider._build_transport = FakeTransport

    transport = provider._connection_config().kwargs["transport"]
    assert provider._connection_config().kwargs["transport"] is transport

    await provider.aclose()
    assert transport._transport.aclosed
    assert provider._transport is None


@pytest.mark.parametrize("backend", ["httpx", "aiohttp"])
@pytest.mark.parametrize("verify", [True, False])
@pytest.mark.parametrize(
    "keepalive_expiry_s,disable_pooling", [(None, False), (3.0, False), (None, True), (3.0, True)]
)
async def test_connection_tls_is_independent_of_pool_settings(
    fake_opensandbox_sdk: None,
    backend: str,
    verify: bool,
    keepalive_expiry_s: float | None,
    disable_pooling: bool,
) -> None:
    import ssl

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={
            "transport_backend": backend,
            "tls_verify": verify,
            "keepalive_expiry_s": keepalive_expiry_s,
            "disable_connection_pooling": disable_pooling,
        }
    )
    try:
        config = provider._connection_config()
        if backend == "httpx" and verify and keepalive_expiry_s is None and not disable_pooling:
            # The SDK verifies certificates by default; no custom transport is needed.
            assert "transport" not in config.kwargs
            assert provider._transport is None
        else:
            transport = config.kwargs["transport"]
            if backend == "aiohttp":
                assert transport._transport.verify is verify
            else:
                context = transport._transport._pool._ssl_context
                assert context.verify_mode == (ssl.CERT_REQUIRED if verify else ssl.CERT_NONE)
                assert context.check_hostname is verify
            assert provider._connection_config().kwargs["transport"] is transport
    finally:
        await provider.aclose()
    assert provider._transport is None


def test_connection_transport_backend_aiohttp_opt_in(fake_opensandbox_sdk: None) -> None:
    from nemo_gym.sandbox.providers._http_transport import GymAiohttpTransport

    provider = opensandbox_provider.OpenSandboxProvider(connection={"transport_backend": "aiohttp"})
    transport = provider._build_transport()
    assert isinstance(transport, GymAiohttpTransport)

    extensions = provider._resolve_extensions({"imagePullPolicy": "Never"})
    assert extensions["imagePullPolicy"] == "Never"
    assert extensions["opensandbox.extensions.image-pull-policy"] == "Never"

    no_policy_provider = opensandbox_provider.OpenSandboxProvider(create={"image_pull_policy": None})
    assert no_policy_provider._resolve_extensions({"imagePullPolicy": "Never"}) == {"imagePullPolicy": "Never"}


def test_connection_config_disable_pooling_sets_fresh_transport(fake_opensandbox_sdk: None) -> None:
    import httpx

    # Default: a keepalive-bounded transport with connection reuse enabled.
    pooled = opensandbox_provider.OpenSandboxProvider(
        connection={"transport_backend": "httpx", "domain": "sandbox.example"}
    )
    pooled_transport = pooled._connection_config().kwargs["transport"]
    assert isinstance(pooled_transport._transport, httpx.AsyncHTTPTransport)
    assert pooled_transport._transport._pool._max_keepalive_connections > 0

    # disable_connection_pooling -> same transport plumbing, but no reuse.
    fresh = opensandbox_provider.OpenSandboxProvider(
        connection={"transport_backend": "httpx", "domain": "sandbox.example", "disable_connection_pooling": True}
    )
    transport = fresh._connection_config().kwargs.get("transport")
    assert isinstance(transport._transport, httpx.AsyncHTTPTransport)
    assert transport._transport._pool._max_keepalive_connections == 0


@pytest.mark.parametrize(
    ("execution", "expected"),
    [
        (SimpleNamespace(exit_code=3, error=None, complete=None), (3, None, None)),
        (SimpleNamespace(exit_code=3, error=SimpleNamespace(), complete=None), (3, None, None)),
        (SimpleNamespace(exit_code=None, error=None, complete=SimpleNamespace()), (0, None, None)),
    ],
    ids=["exit-code", "explicit-nonzero-with-error", "completed"],
)
def test_execution_outcome_maps_sdk_results(execution: Any, expected: tuple[int, str | None, str | None]) -> None:
    assert opensandbox_provider._execution_outcome(execution, operation="exec", sandbox_id="sb") == expected


@pytest.mark.parametrize(
    ("execution", "message"),
    [
        (SimpleNamespace(exit_code=None, error=SimpleNamespace(), complete=None), "error without an exit code"),
        (SimpleNamespace(exit_code=0, error=SimpleNamespace(), complete=None), "exit code 0 and an error"),
        (SimpleNamespace(exit_code=None, error=None, complete=None), "without an exit code"),
    ],
)
def test_execution_outcome_rejects_unknown_or_contradictory_status(execution: Any, message: str) -> None:
    with pytest.raises(NgFailureEvent, match=message) as exc_info:
        opensandbox_provider._execution_outcome(execution, operation="exec", sandbox_id="sb")
    assert exc_info.value.failure_kind == PROVIDER_COMMAND_OUTCOME_UNKNOWN
    assert exc_info.value.boundary == "sandbox"
    assert exc_info.value.operation == "exec"
    assert exc_info.value.details.sandbox_id == "sb"


@pytest.mark.parametrize(
    ("exit_code", "error", "message"),
    [
        (None, None, "without an exit code"),
        (None, SimpleNamespace(name="CommandError", value="lost"), "error without an exit code"),
        (0, SimpleNamespace(name="CommandError", value="lost"), "exit code 0 and an error"),
    ],
)
async def test_exec_rejects_unknown_or_contradictory_stream_status(
    monkeypatch: pytest.MonkeyPatch, exit_code: int | None, error: Any, message: str
) -> None:
    """Partial output and inconsistent SDK fields cannot turn provider loss into a scored command result."""

    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    async def run(_command: str, *, opts: FakeRunCommandOpts, handlers: Any = None) -> Any:
        return SimpleNamespace(
            logs=SimpleNamespace(stdout=[SimpleNamespace(text="partial")], stderr=[]),
            error=error,
            exit_code=exit_code,
            complete=None,
        )

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(connection={"request_timeout_s": 5}, probe={"command": None})
    raw = SimpleNamespace(commands=SimpleNamespace(run=run))
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sandbox-1", provider_name="opensandbox", raw=raw)

    with raises_failure(PROVIDER_COMMAND_OUTCOME_UNKNOWN, match=message):
        await provider.exec(handle, "pytest", timeout_s=30)


async def test_exec_file_operations_and_reference_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeLog:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeCommands:
        def __init__(self) -> None:
            self.calls: list[tuple[str, FakeRunCommandOpts]] = []

        async def run(self, command: str, *, opts: FakeRunCommandOpts, handlers: Any = None) -> Any:
            self.calls.append((command, opts))
            if "fail" in command:
                return SimpleNamespace(
                    logs=SimpleNamespace(stdout=[], stderr=[FakeLog("stderr")]),
                    error=SimpleNamespace(name="CommandError", value="failed"),
                    exit_code=3,
                )
            return SimpleNamespace(
                logs=SimpleNamespace(stdout=[FakeLog("stdout")], stderr=[]),
                error=None,
                exit_code=None,
                complete=SimpleNamespace(execution_time_in_millis=1),
            )

    class FakeFiles:
        def __init__(self) -> None:
            self.writes: list[tuple[str, str | bytes]] = []

        async def write_file(self, target_path: str, data: str | bytes) -> None:
            self.writes.append((target_path, data))

        async def read_bytes_stream(self, source_path: str) -> Any:
            async def chunks() -> Any:
                yield b"bytes:"
                yield source_path.encode()

            return chunks()

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()
            self.files = FakeFiles()

        async def get_info(self) -> Any:
            return SimpleNamespace(status=SimpleNamespace(state="RUNNING"))

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
    )
    raw = FakeRaw()
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sandbox-1", provider_name="opensandbox", raw=raw)

    result = await provider.exec(
        handle,
        "echo hello",
        cwd="/repo",
        env={"A": "B"},
        timeout_s=2,
        user=1000,
    )
    assert result == opensandbox_provider.SandboxExecResult(stdout="stdout", stderr=None, return_code=0)
    command, opts = raw.commands.calls[0]
    assert command == "echo hello"
    assert opts.kwargs == {
        "working_directory": "/repo",
        "envs": {"A": "B"},
        "timeout": timedelta(seconds=2),
        "uid": 1000,
    }

    result = await provider.exec(handle, "fail", user="agent")
    assert result.return_code == 3
    assert result.error_type is None
    assert result.stderr == "stderr\nCommandError: failed"
    assert raw.commands.calls[1][0] == "su -s /bin/sh -c fail agent"

    upload_path = tmp_path / "upload.txt"
    upload_path.write_text("upload", encoding="utf-8")
    await provider.upload_file(handle, upload_path, "/remote/upload.txt")
    download_path = tmp_path / "nested" / "download.txt"
    await provider.download_file(handle, "/remote/download.txt", download_path)
    assert raw.files.writes == [("/remote/upload.txt", b"upload")]
    assert download_path.read_bytes() == b"bytes:/remote/download.txt"
    assert await provider.status(handle) == SandboxStatus.RUNNING
    bare_handle = opensandbox_provider.SandboxHandle(sandbox_id="sandbox-2", provider_name="opensandbox", raw=object())
    assert await provider.status(bare_handle) == SandboxStatus.UNKNOWN


def _download_provider(monkeypatch: pytest.MonkeyPatch, attempts: list[list[Any]]) -> tuple[Any, Any]:
    """A provider whose sandbox streams the given chunk lists in turn; an exception in a list is raised there."""
    served = iter(attempts)

    class FakeFiles:
        async def read_bytes_stream(self, source_path: str) -> Any:
            chunks = next(served)

            async def stream() -> Any:
                for chunk in chunks:
                    if isinstance(chunk, BaseException):
                        raise chunk
                    yield chunk

            return stream()

    monkeypatch.setattr(
        opensandbox_provider, "_require_opensandbox_sdk", lambda: (object, object, object, object, object)
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    provider = opensandbox_provider.OpenSandboxProvider(connection={"request_timeout_s": 5}, probe={"command": None})
    handle = opensandbox_provider.SandboxHandle(
        sandbox_id="sandbox-1", provider_name="opensandbox", raw=SimpleNamespace(files=FakeFiles())
    )
    return provider, handle


async def test_download_streams_a_large_file_to_disk_in_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    chunks = [bytes([i]) * 1024 for i in range(8)]
    provider, handle = _download_provider(monkeypatch, [chunks])
    target = tmp_path / "out" / "big.bin"

    await provider.download_file(handle, "/root/big.bin", target)

    assert target.read_bytes() == b"".join(chunks)


async def test_a_retried_download_starts_the_file_again(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider, handle = _download_provider(monkeypatch, [[b"partial", ConnectionError("reset")], [b"whole ", b"file"]])
    target = tmp_path / "report.docx"

    await provider.download_file(handle, "/root/report.docx", target)

    assert target.read_bytes() == b"whole file"


async def _no_sleep(_seconds: float) -> None:
    return None


@pytest.mark.asyncio
async def test_exec_background_polls_status_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """background_exec submits, polls status until finished, then reads logs."""

    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeCommands:
        def __init__(self) -> None:
            self.run_calls: list[tuple[str, FakeRunCommandOpts]] = []
            self.status_calls: list[str] = []
            self.log_calls: list[str] = []
            self._status_sequence = [
                SimpleNamespace(running=True, exit_code=None, error=None),
                SimpleNamespace(running=False, exit_code=7, error=None),
            ]

        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> Any:
            self.run_calls.append((command, opts))
            return SimpleNamespace(id="exec-42")

        async def get_command_status(self, execution_id: str) -> Any:
            self.status_calls.append(execution_id)
            return self._status_sequence[min(len(self.status_calls) - 1, len(self._status_sequence) - 1)]

        async def get_background_command_logs(self, execution_id: str) -> Any:
            self.log_calls.append(execution_id)
            return SimpleNamespace(content="combined output", cursor=None)

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
        operations={"background_exec": True, "background_poll_interval_s": 0.01},
    )
    raw = FakeRaw()
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sandbox-bg", provider_name="opensandbox", raw=raw)

    result = await provider.exec(handle, "make build", cwd="/repo", timeout_s=30)

    assert result == opensandbox_provider.SandboxExecResult(
        stdout="combined output", stderr=None, return_code=7, error_type=None
    )
    # Submitted once with background=True; polled twice (running -> finished); read logs once.
    assert len(raw.commands.run_calls) == 1
    assert raw.commands.run_calls[0][1].kwargs["background"] is True
    assert raw.commands.status_calls == ["exec-42", "exec-42"]
    assert raw.commands.log_calls == ["exec-42"]


# execd reports the kill at a command's timeout like any signal death: exit code -1, "signal: killed".
@pytest.mark.parametrize(
    "exit_code,ran_s,timeout_s,expected",
    [
        (-1, 30.0, 30, (-1, None)),  # killed at its timeout
        (-1, 1.0, 30, (-1, None)),  # killed before its deadline, e.g. by the OOM killer
        (7, 30.0, 30, (7, None)),  # exited on its own at the deadline
        (-1, 30.0, 0, (-1, None)),  # a zero timeout enforces none
    ],
)
async def test_exec_background_does_not_infer_timeout_from_signal_and_elapsed_time(
    monkeypatch: pytest.MonkeyPatch, exit_code: int, ran_s: float, timeout_s: int, expected: tuple
) -> None:
    started_at = datetime(2026, 10, 23, 17, 9, 8, 950131, tzinfo=timezone.utc)
    error = "signal: killed" if exit_code == -1 else f"exit status {exit_code}"

    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeCommands:
        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> Any:
            return SimpleNamespace(id="exec-42")

        async def get_command_status(self, execution_id: str) -> Any:
            return SimpleNamespace(
                running=False,
                exit_code=exit_code,
                error=error,
                started_at=started_at,
                finished_at=started_at + timedelta(seconds=ran_s, microseconds=646),
            )

        async def get_background_command_logs(self, execution_id: str) -> Any:
            return SimpleNamespace(content="partial output", cursor=None)

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5}, probe={"command": None}, operations={"background_exec": True}
    )
    handle = opensandbox_provider.SandboxHandle("sandbox-bg", "opensandbox", SimpleNamespace(commands=FakeCommands()))

    result = await provider.exec(handle, "bash solve.sh", timeout_s=timeout_s)

    assert (result.return_code, result.error_type) == expected
    assert result.stdout == "partial output"
    if expected[1] == "timeout":
        assert result.stderr == "signal: killed\nCommand timed out after 30s; OpenSandbox execd killed it"
    else:
        assert result.stderr == error


def _background_provider(
    monkeypatch: pytest.MonkeyPatch, commands: Any, **operations: Any
) -> tuple[opensandbox_provider.OpenSandboxProvider, opensandbox_provider.SandboxHandle]:
    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
        operations={
            "background_exec": True,
            "background_poll_initial_s": 0.01,
            "background_poll_interval_s": 0.02,
            **operations,
        },
    )
    return provider, opensandbox_provider.SandboxHandle(
        "sandbox-bg", "opensandbox", SimpleNamespace(commands=commands)
    )


# An uncertain submission must not be repeated, regardless of eventual command timing.
async def test_background_submit_failure_is_not_replayed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Backend502Error(Exception):
        status_code = 502

    class FakeCommands:
        submits = 0

        async def run(self, command: str, *, opts: Any) -> Any:
            self.submits += 1
            if self.submits < 3:
                await asyncio.sleep(0.15)
                raise Backend502Error("Failed to run command. Status code: 502")
            return SimpleNamespace(id="exec-42")

        async def get_command_status(self, execution_id: str) -> Any:
            return SimpleNamespace(running=False, exit_code=-1, error="signal: killed")

        async def get_background_command_logs(self, execution_id: str) -> Any:
            await asyncio.sleep(0.3)
            return SimpleNamespace(content="partial output", cursor=None)

    commands = FakeCommands()
    provider, handle = _background_provider(monkeypatch, commands, retries=3, retry_delay_s=0.01)

    with raises_failure(PROVIDER_UNAVAILABLE):
        await provider.exec(handle, "bash solve.sh", timeout_s=0.25)
    assert commands.submits == 1


def _sdk_transport_failure(*chain: BaseException) -> Exception:
    """``chain`` (outermost first) wrapped the way the OpenSandbox SDK wraps its HTTP client's error."""
    from opensandbox.exceptions import SandboxInternalException

    for outer, inner in zip(chain, chain[1:]):
        outer.__cause__ = inner
    return SandboxInternalException("Network connectivity error", cause=chain[0])


def _aiohttp_connect_timeout() -> list[BaseException]:
    import aiohttp

    os_error = TimeoutError(110, "Connection timed out")
    connector = aiohttp.ClientConnectorError(SimpleNamespace(host="sandbox.invalid", port=80, ssl="default"), os_error)
    return [httpx.ConnectError(str(connector)), connector, os_error]


def _aiohttp_reset_after_send() -> list[BaseException]:
    import aiohttp

    client_error = aiohttp.ClientOSError(104, "Connection reset by peer")
    return [httpx.ConnectError(str(client_error)), client_error, ConnectionResetError(104, "Connection reset by peer")]


@pytest.mark.parametrize(
    ("chain", "resent"),
    [
        pytest.param(lambda: [httpx.ConnectError("[Errno 110] Connection timed out")], True, id="connect"),
        pytest.param(lambda: [httpx.ConnectTimeout("connect")], True, id="connect-timeout"),
        pytest.param(lambda: [httpx.PoolTimeout("pool")], True, id="pool-timeout"),
        pytest.param(_aiohttp_connect_timeout, True, id="aiohttp-connect"),
        pytest.param(_aiohttp_reset_after_send, False, id="aiohttp-reset-after-send"),
        pytest.param(lambda: [httpx.ReadTimeout("read")], False, id="read-timeout"),
        pytest.param(lambda: [httpx.RemoteProtocolError("disconnected")], False, id="dropped"),
    ],
)
async def test_background_submit_is_sent_again_only_when_it_never_left_the_client(
    monkeypatch: pytest.MonkeyPatch, chain: Any, resent: bool
) -> None:
    class FakeCommands:
        submits = 0

        async def run(self, command: str, *, opts: Any) -> Any:
            self.submits += 1
            if self.submits == 1:
                raise _sdk_transport_failure(*chain())
            return SimpleNamespace(id="exec-42")

        async def get_command_status(self, execution_id: str) -> Any:
            return SimpleNamespace(running=False, exit_code=0, error=None)

        async def get_background_command_logs(self, execution_id: str) -> Any:
            return SimpleNamespace(content="done", cursor=None)

    commands = FakeCommands()
    provider, handle = _background_provider(monkeypatch, commands, retries=3, retry_delay_s=0, retry_max_delay_s=0)

    if resent:
        result = await provider.exec(handle, "bash solve.sh", timeout_s=5)
        assert result.return_code == 0
        assert commands.submits == 2
    else:
        with pytest.raises(NgFailureEvent):
            await provider.exec(handle, "bash solve.sh", timeout_s=5)
        assert commands.submits == 1


async def test_unsent_submit_stops_at_the_operation_retry_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCommands:
        submits = 0

        async def run(self, command: str, *, opts: Any) -> Any:
            self.submits += 1
            raise _sdk_transport_failure(httpx.ConnectError("[Errno 110] Connection timed out"))

    commands = FakeCommands()
    provider, handle = _background_provider(monkeypatch, commands, retries=2, retry_delay_s=0, retry_max_delay_s=0)

    with raises_failure(PROVIDER_UNAVAILABLE):
        await provider.exec(handle, "bash solve.sh", timeout_s=5)
    assert commands.submits == 3


async def test_background_signal_exit_near_deadline_keeps_unknown_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCommands:
        async def run(self, command: str, *, opts: Any) -> Any:
            self.submitted = asyncio.get_running_loop().time()
            return SimpleNamespace(id="exec-42")

        async def get_command_status(self, execution_id: str) -> Any:
            running = asyncio.get_running_loop().time() - self.submitted < 0.3
            return SimpleNamespace(running=running, exit_code=None if running else -1, error="signal: killed")

        async def get_background_command_logs(self, execution_id: str) -> Any:
            return SimpleNamespace(content="partial output", cursor=None)

    provider, handle = _background_provider(monkeypatch, FakeCommands())

    result = await provider.exec(handle, "bash solve.sh", timeout_s=0.25)

    assert (result.return_code, result.error_type) == (-1, None)
    assert result.stderr == "signal: killed"


class _DeadlineCommands:
    """A background command that runs until it is interrupted, or for ``runs_s`` on its own."""

    def __init__(self, *, runs_s: float = 60.0, interrupt_error: Exception | None = None) -> None:
        self.runs_s = runs_s
        self.interrupt_error = interrupt_error
        self.interrupts: list[str] = []
        self.opts: Any = None
        self.killed = False

    async def run(self, command: str, *, opts: Any) -> Any:
        self.opts = opts
        self.submitted = asyncio.get_running_loop().time()
        return SimpleNamespace(id="exec-42")

    async def interrupt(self, execution_id: str) -> None:
        self.interrupts.append(execution_id)
        if self.interrupt_error is not None:
            raise self.interrupt_error
        self.killed = True

    async def get_command_status(self, execution_id: str) -> Any:
        if self.killed:
            return SimpleNamespace(running=False, exit_code=-1, error="signal: killed")
        if asyncio.get_running_loop().time() - self.submitted < self.runs_s:
            return SimpleNamespace(running=True, exit_code=None, error=None)
        return SimpleNamespace(running=False, exit_code=-1, error="signal: killed")

    async def get_background_command_logs(self, execution_id: str) -> Any:
        return SimpleNamespace(content="partial output", cursor=None)


# execd reports its own timeout kill only as a signal death, so Gym enforces the deadline and owns the outcome.
async def test_background_command_running_at_gym_deadline_is_interrupted_and_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = _DeadlineCommands()
    provider, handle = _background_provider(monkeypatch, commands, command_timeout_headroom_s=2.0)

    result = await provider.exec(handle, "bash solve.sh", timeout_s=0.2)

    assert commands.interrupts == ["exec-42"]
    # execd's own timer is a backstop past Gym's deadline.
    assert commands.opts.kwargs["timeout"] == timedelta(seconds=1.2)
    assert (result.return_code, result.error_type) == (124, "timeout")
    assert result.stdout == "partial output"
    assert result.stderr == "signal: killed\nCommand timed out after 0.2s; Gym interrupted it"


async def test_background_command_finishing_before_gym_deadline_is_not_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = _DeadlineCommands(runs_s=0.05)
    provider, handle = _background_provider(monkeypatch, commands)

    result = await provider.exec(handle, "bash solve.sh", timeout_s=5)

    assert commands.interrupts == []
    assert (result.return_code, result.error_type) == (-1, None)
    assert result.stderr == "signal: killed"


# A status outage that spans Gym's deadline keeps the polling cadence and backoff; it does not busy-poll.
async def test_background_status_outage_across_gym_deadline_keeps_polling_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Backend503Error(Exception):
        status_code = 503

    class OutageCommands(_DeadlineCommands):
        status_calls = 0

        async def get_command_status(self, execution_id: str) -> Any:
            self.status_calls += 1
            if asyncio.get_running_loop().time() - self.submitted >= 0.08:
                raise Backend503Error("Get command status failed. Status code: 503")
            return await super().get_command_status(execution_id)

    commands = OutageCommands()
    provider, handle = _background_provider(
        monkeypatch,
        commands,
        command_timeout_headroom_s=0.5,
        status_recovery_timeout_s=0.2,
        retry_delay_s=0.01,
    )

    with pytest.raises(NgFailureEvent):
        await provider.exec(handle, "bash solve.sh", timeout_s=0.1)

    assert commands.interrupts == []
    # ~0.2 s of outage at the 0.01-0.02 s cadence is a few dozen reads at most, not thousands.
    assert commands.status_calls < 60


# A refused interrupt (e.g. the command finished just before it) proves nothing; the reported exit is kept.
async def test_background_refused_interrupt_keeps_the_reported_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _DeadlineCommands(runs_s=0.4, interrupt_error=RuntimeError("Interrupt command failed: 404"))
    provider, handle = _background_provider(monkeypatch, commands, command_timeout_headroom_s=2.0)

    result = await provider.exec(handle, "bash solve.sh", timeout_s=0.2)

    assert commands.interrupts == ["exec-42"]
    assert (result.return_code, result.error_type) == (-1, None)
    assert result.stderr == "signal: killed"


# execd's init and error stamps (ms) for commands run with `timeout_s` on an OpenSandbox cell (2026-10-28).
LIVE_INIT_MS = 1793227333925


@pytest.mark.parametrize(
    "command,exit_code,ran_ms,timeout_s,expected",
    [
        ("sleep 30", -1, 2999, 3, (-1, None)),
        ("kill -9 $$", -1, 0, 3, (-1, None)),
        ("sleep 1; kill -9 $$", -1, 1001, 3, (-1, None)),
        ("exit 7", 7, 1, 3, (7, None)),
        ("sleep 30", -1, 499, 0.5, (-1, None)),
        ("kill -9 $$", -1, 0, 0.5, (-1, None)),
        ("sleep 30", -1, 201, 0.2, (-1, None)),
        ("kill -9 $$", -1, 1, 0.2, (-1, None)),
        ("sleep 30", -1, 2999, None, (-1, None)),  # no timeout, so execd killed it for another reason
    ],
)
@pytest.mark.parametrize("submit_delay_s", [0.0, 0.35])  # e.g. the transport waiting out a 429's Retry-After
async def test_exec_times_a_command_with_execd_stamps(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    exit_code: int,
    ran_ms: int,
    timeout_s: float | None,
    expected: tuple,
    submit_delay_s: float,
) -> None:
    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeCommands:
        async def run(self, command: str, *, opts: FakeRunCommandOpts, handlers: Any = None) -> Any:
            await asyncio.sleep(submit_delay_s)
            error = SimpleNamespace(name="CommandExecError", value=str(exit_code), timestamp=LIVE_INIT_MS + ran_ms)
            return SimpleNamespace(
                logs=SimpleNamespace(stdout=[SimpleNamespace(text="partial output")], stderr=[]),
                error=error,
                exit_code=exit_code,
                complete=None,
            )

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(connection={"request_timeout_s": 5}, probe={"command": None})
    handle = opensandbox_provider.SandboxHandle("sandbox-1", "opensandbox", SimpleNamespace(commands=FakeCommands()))

    result = await provider.exec(handle, command, timeout_s=timeout_s)

    assert (result.return_code, result.error_type) == expected
    assert result.stdout == "partial output"
    note = f"\nCommand timed out after {timeout_s:g}s; OpenSandbox execd killed it" if expected[1] == "timeout" else ""
    assert result.stderr == f"CommandExecError: {exit_code}{note}"


@pytest.mark.parametrize("ran_s,expected", [(0.25, (-1, None)), (0.0, (-1, None))])
async def test_exec_without_execd_stamps_times_the_command_by_its_events(
    monkeypatch: pytest.MonkeyPatch, ran_s: float, expected: tuple
) -> None:
    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeCommands:
        async def run(self, command: str, *, opts: FakeRunCommandOpts, handlers: Any = None) -> Any:
            await asyncio.sleep(0.3)  # before execd started it
            await asyncio.sleep(ran_s)
            error = SimpleNamespace(name="CommandExecError", value="-1")
            return SimpleNamespace(
                logs=SimpleNamespace(stdout=[], stderr=[]), error=error, exit_code=-1, complete=None
            )

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(connection={"request_timeout_s": 5}, probe={"command": None})
    handle = opensandbox_provider.SandboxHandle("sandbox-1", "opensandbox", SimpleNamespace(commands=FakeCommands()))

    result = await provider.exec(handle, "bash solve.sh", timeout_s=0.25)

    assert (result.return_code, result.error_type) == expected


DELETED_OUTPUT_LOG_ERROR = (
    "Get command logs failed: error open combined output file for command exec-42: no such file or directory"
)


@pytest.mark.parametrize(
    "log_error,exit_code,kept",
    [
        (DELETED_OUTPUT_LOG_ERROR, 3, True),
        # Without an exit code there is no evidence of how the command ended.
        (DELETED_OUTPUT_LOG_ERROR, None, False),
        ("Get command logs failed: HTTP 500", 3, False),
    ],
)
@pytest.mark.parametrize("status_error", [None, "killed by signal"])
@pytest.mark.asyncio
async def test_exec_background_keeps_exit_status_when_the_command_deleted_its_output(
    monkeypatch: pytest.MonkeyPatch, log_error: str, exit_code: int | None, kept: bool, status_error: str | None
) -> None:
    from opensandbox.exceptions import SandboxApiException

    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeCommands:
        log_calls = 0

        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> Any:
            return SimpleNamespace(id="exec-42")

        async def get_command_status(self, execution_id: str) -> Any:
            return SimpleNamespace(running=False, exit_code=exit_code, error=status_error)

        async def get_background_command_logs(self, execution_id: str) -> Any:
            self.log_calls += 1
            # As the SDK raises it: a 500 is otherwise retryable.
            raise SandboxApiException(log_error, status_code=500)

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
        operations={"background_exec": True, "background_poll_interval_s": 0.01, "retries": 2},
    )
    commands = FakeCommands()
    handle = opensandbox_provider.SandboxHandle(
        sandbox_id="sandbox-bg", provider_name="opensandbox", raw=SimpleNamespace(commands=commands)
    )

    if kept:
        result = await provider.exec(handle, "/tests/test.sh", timeout_s=30)
        # The exit status stands, and the empty output says why instead of looking like silence.
        note = opensandbox_provider.DELETED_OUTPUT_NOTE
        assert result == opensandbox_provider.SandboxExecResult(
            stdout=None,
            stderr=f"{status_error}\n{note}" if status_error else note,
            return_code=3,
            error_type=None,
        )
    else:
        with raises_failure(PROVIDER_REQUEST_FAILED) as caught:
            await provider.exec(handle, "/tests/test.sh", timeout_s=30)
        assert isinstance(caught.value.__cause__, SandboxApiException)
        assert caught.value.http_status == 500
    # A deleted output file cannot come back, so it is read once; other 500s keep their retries.
    assert commands.log_calls == (1 if log_error == DELETED_OUTPUT_LOG_ERROR else 3)


@pytest.mark.asyncio
async def test_exec_background_preserves_unattributed_oom_reason_after_502(monkeypatch: pytest.MonkeyPatch) -> None:
    class Backend502Error(Exception):
        status_code = 502

    class FakeCommands:
        async def run(self, command: str, *, opts: Any) -> Any:
            return SimpleNamespace(id="exec-oom")

        async def get_command_status(self, execution_id: str) -> Any:
            raise Backend502Error("Get command status failed: HTTP 502")

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()
            self.statuses = [
                SimpleNamespace(state="Running", reason="OOMKilled", message="not terminal yet"),
                SimpleNamespace(
                    state="Failed",
                    reason="OOMKilled",
                    message="container sandbox terminated with OOMKilled (exit code 137); " + "x" * 1000,
                ),
            ]
            self.info_calls = 0

        async def get_info(self) -> Any:
            status = self.statuses[min(self.info_calls, len(self.statuses) - 1)]
            self.info_calls += 1
            return SimpleNamespace(status=status)

    monkeypatch.setattr(
        opensandbox_provider, "_require_opensandbox_sdk", lambda: (object, object, dict, object, object)
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
        operations={"background_exec": True, "retries": 0},
    )
    raw = FakeRaw()
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sandbox-oom", provider_name="opensandbox", raw=raw)

    with raises_failure(PROVIDER_UNAVAILABLE) as exc_info:
        await provider.exec(handle, "allocate memory", timeout_s=30)

    assert exc_info.value.details.sandbox_id == "sandbox-oom"
    message = str(exc_info.value)
    assert "is unavailable" in message
    assert "SandboxResources.memory_mib" not in message
    assert "reason='OOMKilled'" in message
    assert len(message) < 800
    assert isinstance(exc_info.value.__cause__.__cause__, Backend502Error)
    assert raw.info_calls == 2


@pytest.mark.parametrize(
    ("inspect", "oom_confirmed"),
    [
        (
            "Pod Name: sandbox-pod\nContainers:\n  sandbox:\n"
            "    State:          Terminated (exit=137, reason=OOMKilled)\n",
            True,
        ),
        (
            "Pod Name: sandbox-pod\nContainers:\n  sandbox:\n"
            "    State:          Terminated (exit=137, reason=Error)\n"
            "  egress:\n    State:          Terminated (exit=137, reason=OOMKilled)\n",
            False,
        ),
    ],
)
@pytest.mark.parametrize("control_status", ["unavailable", "container_failed"])
async def test_exec_background_retries_timed_out_status_then_checks_ended_sandbox(
    monkeypatch: pytest.MonkeyPatch, inspect: str, oom_confirmed: bool, control_status: str
) -> None:
    """A read timeout keeps the submitted command alive; a later 410 needs current-container OOM proof."""
    from nemo_gym.client import http_client

    class Backend410Error(Exception):
        status_code = 410

    class Backend502Error(Exception):
        status_code = 502

    class FakeCommands:
        def __init__(self) -> None:
            self.run_calls = 0
            self.status_calls: list[str] = []
            self.log_calls = 0

        async def run(self, command: str, *, opts: Any) -> Any:
            self.run_calls += 1
            return SimpleNamespace(id="exec-oom")

        async def get_command_status(self, execution_id: str) -> Any:
            self.status_calls.append(execution_id)
            if len(self.status_calls) == 1:
                raise TimeoutError("Timed out during OpenSandbox command status after 10s")
            if len(self.status_calls) == 2:
                raise Backend502Error("HTTP 502 [BACKEND_CONNECTION_FAILED] backend connection failed")
            raise Backend410Error("HTTP 410 [SANDBOX::ENDED] Sandbox sandbox-oom has ended (OOMKilled)")

        async def get_background_command_logs(self, execution_id: str) -> Any:
            self.log_calls += 1
            return SimpleNamespace(content="unexpected", cursor=None)

    class FakeConnectionConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def get_base_url(self) -> str:
            return "https://sandbox.example/v1"

        def get_api_key(self) -> str:
            return "key"

    class FakeResponse:
        def __init__(self, status: int, body: str) -> None:
            self.status = status
            self.content = SimpleNamespace(read=AsyncMock(return_value=body.encode()))

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
            return None

    no_oom = FakeResponse(200, "Pod Name: sandbox-pod\nContainers:\n  sandbox:\n    State: Running\n")
    request = AsyncMock(side_effect=[no_oom, no_oom, FakeResponse(200, inspect)])
    monkeypatch.setattr(http_client, "request", request)
    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, FakeConnectionConfig, dict, object, object),
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    provider = opensandbox_provider.OpenSandboxProvider(
        diagnostics={"inspect_oom": True},
        connection={"domain": "sandbox.example", "api_key": "key", "request_timeout_s": 5},
        probe={"command": None},
        operations={"background_exec": True, "retries": 0},
    )
    commands = FakeCommands()
    get_info = (
        AsyncMock(side_effect=RuntimeError("status unavailable"))
        if control_status == "unavailable"
        else AsyncMock(
            side_effect=[
                SimpleNamespace(status=SimpleNamespace(state="Running", reason=None)),
                SimpleNamespace(status=SimpleNamespace(state="Running", reason=None)),
                SimpleNamespace(status=SimpleNamespace(state="Failed", reason="ContainerFailed", message="ended")),
            ]
        )
    )
    raw = SimpleNamespace(commands=commands, get_info=get_info)
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sandbox-oom", provider_name="opensandbox", raw=raw)

    if oom_confirmed:
        with raises_failure(PROVIDER_OOM_KILLED, match="inspect confirms") as exc_info:
            await provider.exec(handle, "allocate memory", timeout_s=30)
        assert (
            isinstance(exc_info.value.__cause__.__cause__, Backend410Error)
            if oom_confirmed
            else isinstance(exc_info.value.__cause__, Backend410Error)
        )
        assert exc_info.value.details.sandbox_id == "sandbox-oom"
        assert exc_info.value.details.provider_status_reason == (
            "ContainerFailed" if control_status == "container_failed" else None
        )
        assert exc_info.value.details.provider == "opensandbox"
    else:
        with raises_failure(PROVIDER_UNAVAILABLE) as exc_info:
            await provider.exec(handle, "allocate memory", timeout_s=30)
        if control_status == "container_failed":
            assert exc_info.value.details.provider_status_reason == "ContainerFailed"
            assert isinstance(exc_info.value.__cause__.__cause__, Backend410Error)
        else:
            assert isinstance(exc_info.value.__cause__, Backend410Error)

    assert commands.run_calls == 1
    assert commands.status_calls == ["exec-oom", "exec-oom", "exec-oom"]
    assert commands.log_calls == 0
    assert request.await_count == 3
    assert request.await_args.args[1].endswith("/sandboxes/sandbox-oom/diagnostics/inspect")


@pytest.mark.parametrize(
    ("inspect", "oom_confirmed"),
    [
        (
            "Pod Name: sandbox-pod\nContainers:\n  sandbox:\n"
            "    State:          Terminated (exit=137, reason=OOMKilled)\n",
            True,
        ),
        (
            "Pod Name: sandbox-pod\nContainers:\n  sandbox:\n    State:          Running\n",
            False,
        ),
        (
            "Pod Name: sandbox-pod\nContainers:\n  sandbox:\n"
            "    State:          Terminated (exit=137, reason=Error)\n"
            "  egress:\n    State:          Terminated (exit=137, reason=OOMKilled)\n",
            False,
        ),
    ],
)
async def test_exec_background_inspects_current_container_after_persistent_status_timeout(
    monkeypatch: pytest.MonkeyPatch, inspect: str, oom_confirmed: bool
) -> None:
    """A status-only timeout can prove an OOM when get_info is unavailable, but cannot infer one."""
    from nemo_gym.client import http_client

    class FakeCommands:
        def __init__(self) -> None:
            self.run_calls = 0
            self.status_calls: list[str] = []
            self.log_calls = 0

        async def run(self, command: str, *, opts: Any) -> Any:
            self.run_calls += 1
            return SimpleNamespace(id="exec-timeout")

        async def get_command_status(self, execution_id: str) -> Any:
            self.status_calls.append(execution_id)
            raise TimeoutError("command status route timed out")

        async def get_background_command_logs(self, execution_id: str) -> Any:
            self.log_calls += 1
            pytest.fail("logs cannot be fetched without a final status")

    class FakeConnectionConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def get_base_url(self) -> str:
            return "https://sandbox.example/v1"

        def get_api_key(self) -> str:
            return "key"

    class FakeResponse:
        status = 200

        def __init__(self) -> None:
            self.content = SimpleNamespace(read=AsyncMock(return_value=inspect.encode()))

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
            return None

    request = AsyncMock(return_value=FakeResponse())
    monkeypatch.setattr(http_client, "request", request)
    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, FakeConnectionConfig, dict, object, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        diagnostics={"inspect_oom": True},
        connection={"domain": "sandbox.example", "api_key": "key", "request_timeout_s": 5},
        probe={"command": None},
        operations={"background_exec": True, "retries": 0, "background_poll_interval_s": 0.001},
    )
    provider._operations = replace(provider._operations, status_recovery_timeout_s=0.02)
    commands = FakeCommands()
    raw = SimpleNamespace(commands=commands, get_info=AsyncMock(side_effect=RuntimeError("status unavailable")))
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sandbox-timeout", provider_name="opensandbox", raw=raw)

    if oom_confirmed:
        with raises_failure(PROVIDER_OOM_KILLED, match="inspect confirms") as exc_info:
            await provider.exec(handle, "agent command", timeout_s=10_800)
        assert isinstance(exc_info.value.__cause__.__cause__, TimeoutError)
        assert exc_info.value.details.sandbox_id == "sandbox-timeout"
    else:
        with raises_failure(PROVIDER_STATUS_UNAVAILABLE, match="unavailable beyond"):
            await provider.exec(handle, "agent command", timeout_s=10_800)
        assert len(commands.status_calls) >= 2

    assert commands.run_calls == 1
    assert set(commands.status_calls) == {"exec-timeout"}
    assert commands.log_calls == 0
    assert request.await_count >= 1
    assert request.await_args.args[1].endswith("/sandboxes/sandbox-timeout/diagnostics/inspect")


@pytest.mark.parametrize(
    ("inspect", "status_code", "evidence"),
    [
        (
            "Pod Name: sandbox-pod\nContainers:\n  sandbox:\n"
            "    State:          Terminated (exit=137, reason=OOMKilled)\n"
            "  egress:\n    State:          Running (since today)\n",
            200,
            "sandbox State: Terminated (exit=137, reason=OOMKilled)",
        ),
        (
            "Container ID: abc123\nRunning: False\nOOMKilled: True\nExit Code: 137\n",
            200,
            "Docker OOMKilled: True",
        ),
        (
            "Pod Name: sandbox-pod\nContainers:\n  sandbox:\n"
            "    State:          Terminated (exit=137, reason=Error)\n"
            "    Last State:     Terminated (exit=137, reason=OOMKilled)\n",
            200,
            None,
        ),
        (
            "Pod Name: sandbox-pod\nContainers:\n  sandbox:\n"
            "    State:          Terminated (exit=137, reason=Error)\n"
            "  egress:\n    State:          Terminated (exit=137, reason=OOMKilled)\n"
            "Init Containers:\n  setup:\n    State:          Terminated (exit=137, reason=OOMKilled)\n",
            200,
            None,
        ),
        (
            "Pod Name: sandbox-pod\nContainers:\n  sandbox:\n"
            "    State:          Terminated (exit=137, reason=Error)\n"
            "    Message:        not OOMKilled; exit=137\n",
            200,
            None,
        ),
        (
            "Pod Name: sandbox-pod\nContainers:\n  sandbox:\n"
            "    State:          Terminated (exit=137, reason=OOMKilled)\n"
            "    State:          Terminated (exit=137, reason=Error)\n",
            200,
            None,
        ),
        (
            "Pod Name: sandbox-pod\nContainers:\n  sandbox:\n"
            "    State:          Terminated (exit=137, reason=OOMKilled)\n",
            404,
            None,
        ),
    ],
)
@pytest.mark.parametrize("lifecycle_reason", ["ContainerFailed", "OOMKilled"])
@pytest.mark.asyncio
async def test_exec_background_inspect_confirms_only_current_sandbox_oom_after_terminal_502(
    monkeypatch: pytest.MonkeyPatch, inspect: str, status_code: int, evidence: str | None, lifecycle_reason: str
) -> None:
    from nemo_gym.client import http_client

    class Backend502Error(Exception):
        status_code = 502

    class FakeCommands:
        async def run(self, command: str, *, opts: Any) -> Any:
            return SimpleNamespace(id="exec-oom")

        async def get_command_status(self, execution_id: str) -> Any:
            raise Backend502Error("Get command status failed: HTTP 502")

    class FakeConnectionConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def get_base_url(self) -> str:
            return f"{self.kwargs['protocol']}://{self.kwargs['domain']}/v1"

        def get_api_key(self) -> str:
            return self.kwargs["api_key"]

    class FakeResponse:
        def __init__(self) -> None:
            self.status = status_code
            self.content = SimpleNamespace(read=AsyncMock(return_value=inspect.encode()))

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
            return None

    request = AsyncMock(return_value=FakeResponse())
    monkeypatch.setattr(http_client, "request", request)
    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, FakeConnectionConfig, dict, object, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        diagnostics={"inspect_oom": True},
        connection={
            "domain": "https://sandbox.example/prefix/",
            "api_key": "configured-key",
            "tls_verify": False,
            "request_timeout_s": 5,
        },
        probe={"command": None},
        operations={"background_exec": True, "retries": 0},
    )
    provider._operations = replace(provider._operations, status_recovery_timeout_s=0.02)
    monkeypatch.setattr(provider, "_get_transport", lambda: object())
    raw = SimpleNamespace(
        commands=FakeCommands(),
        get_info=AsyncMock(
            return_value=SimpleNamespace(
                status=SimpleNamespace(state="Failed", reason=lifecycle_reason, message="container identity unknown")
            )
        ),
    )
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sandbox-oom", provider_name="opensandbox", raw=raw)

    if evidence is not None:
        with raises_failure(PROVIDER_OOM_KILLED, match="inspect confirms") as exc_info:
            await provider.exec(handle, "allocate memory", timeout_s=30)
        assert isinstance(exc_info.value.__cause__.__cause__, Backend502Error)
        assert evidence in str(exc_info.value)
        assert exc_info.value.details.provider_status_reason == lifecycle_reason
    else:
        with raises_failure(PROVIDER_UNAVAILABLE) as exc_info:
            await provider.exec(handle, "allocate memory", timeout_s=30)
        assert exc_info.value.failure_kind != PROVIDER_OOM_KILLED
        assert exc_info.value.details.provider_status_reason == lifecycle_reason
        assert isinstance(exc_info.value.__cause__.__cause__, Backend502Error)

    assert raw.get_info.await_count >= 1
    assert request.await_count >= 1
    args, kwargs = request.await_args
    assert args == ("GET", "https://sandbox.example/prefix/v1/sandboxes/sandbox-oom/diagnostics/inspect")
    assert kwargs["headers"] == {"OPEN-SANDBOX-API-KEY": "configured-key"}
    assert kwargs["ssl"] is False
    assert 0 < kwargs["timeout"].total <= provider._operations.status_recovery_timeout_s
    assert kwargs["_max_num_tries"] == 1


@pytest.mark.parametrize(
    ("status", "missing"),
    [
        (SimpleNamespace(exit_code=0, error=None), "running"),
        (SimpleNamespace(running=False, error=None), "exit_code"),
    ],
)
@pytest.mark.asyncio
async def test_exec_background_rejects_status_missing_a_field(
    monkeypatch: pytest.MonkeyPatch, status: Any, missing: str
) -> None:
    """An SDK field rename must fail loudly, not score a failed command as success."""

    class FakeCommands:
        async def run(self, command: str, *, opts: Any) -> Any:
            return SimpleNamespace(id="exec-42")

        async def get_command_status(self, execution_id: str) -> Any:
            return status

        async def get_background_command_logs(self, execution_id: str) -> Any:
            return SimpleNamespace(content="combined output", cursor=None)

    monkeypatch.setattr(
        opensandbox_provider, "_require_opensandbox_sdk", lambda: (object, object, dict, object, object)
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
        operations={"background_exec": True, "background_poll_interval_s": 0.01},
    )
    handle = opensandbox_provider.SandboxHandle(
        sandbox_id="sandbox-bg", provider_name="opensandbox", raw=SimpleNamespace(commands=FakeCommands())
    )

    with raises_failure(PROVIDER_COMMAND_OUTCOME_UNKNOWN, match=missing):
        await provider.exec(handle, "make build", timeout_s=30)


@pytest.mark.parametrize(
    ("status", "error_message", "expected_log_reads"),
    [
        (SimpleNamespace(running=None, exit_code=0, error=None), "missing or invalid", 0),
        (SimpleNamespace(running=False, exit_code=None, error=None), "without an exit code", 1),
        (SimpleNamespace(running=False, exit_code=None, error="provider lost status"), "without an exit code", 1),
        (SimpleNamespace(running=False, exit_code=0, error="provider lost status"), "exit code 0 and an error", 1),
    ],
)
@pytest.mark.asyncio
async def test_exec_background_rejects_unknown_outcome_with_readable_logs(
    monkeypatch: pytest.MonkeyPatch, status: Any, error_message: str, expected_log_reads: int
) -> None:
    """Readable output cannot establish success when the SDK omits a terminal outcome."""
    commands = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(id="exec-42")),
        get_command_status=AsyncMock(return_value=status),
        get_background_command_logs=AsyncMock(return_value=SimpleNamespace(content="command output", cursor=None)),
    )
    monkeypatch.setattr(
        opensandbox_provider, "_require_opensandbox_sdk", lambda: (object, object, dict, object, object)
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
        operations={"background_exec": True, "background_poll_interval_s": 0.01},
    )
    handle = opensandbox_provider.SandboxHandle(
        sandbox_id="sandbox-bg", provider_name="opensandbox", raw=SimpleNamespace(commands=commands)
    )

    with raises_failure(PROVIDER_COMMAND_OUTCOME_UNKNOWN, match=error_message):
        await provider.exec(handle, "make build", timeout_s=30)

    commands.run.assert_awaited_once()
    commands.get_command_status.assert_awaited_once_with("exec-42")
    assert commands.get_background_command_logs.await_count == expected_log_reads


async def test_exec_hard_cap_labels_genuinely_wedged_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wedged exec has an unknown outcome, distinct from a command timeout.

    The cap formula floors the real duration at minutes, so the test shrinks it
    by patching asyncio.timeout to ignore the requested duration — the genuine
    cancellation-to-TimeoutError conversion path still runs.
    """

    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeCommands:
        async def run(self, command: str, *, opts: FakeRunCommandOpts, handlers: Any = None) -> Any:
            await asyncio.Event().wait()  # wedged: never returns

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )

    real_timeout = asyncio.timeout
    monkeypatch.setattr(asyncio, "timeout", lambda delay: real_timeout(0.05))

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
        diagnostics={"inspect_oom": False},
    )
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sandbox-wedge", provider_name="opensandbox", raw=FakeRaw())

    with raises_failure(PROVIDER_IO_TIMEOUT, match="hard cap") as caught:
        await provider.exec(handle, "sleep 999", timeout_s=30)
    assert not isinstance(caught.value, TimeoutError)


async def test_inspect_setup_failure_preserves_provider_io_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCommands:
        async def run(self, command: str, *, opts: dict[str, Any]) -> None:
            raise TimeoutError("command submit request expired")

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, dict, object, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    handle = opensandbox_provider.SandboxHandle(
        sandbox_id="unreachable-inspect", provider_name="opensandbox", raw=SimpleNamespace(commands=FakeCommands())
    )

    with raises_failure(PROVIDER_IO_TIMEOUT, match="command run") as caught:
        await provider.exec(handle, "echo unknown", timeout_s=30)

    assert not isinstance(caught.value, TimeoutError)


async def test_foreground_command_request_timeout_is_ungraded_provider_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCommands:
        def __init__(self) -> None:
            self.run_calls = 0

        async def run(self, command: str, *, opts: Any) -> Any:
            self.run_calls += 1
            raise TimeoutError("command run request timed out")

    monkeypatch.setattr(
        opensandbox_provider, "_require_opensandbox_sdk", lambda: (object, object, dict, object, object)
    )
    commands = FakeCommands()
    provider = opensandbox_provider.OpenSandboxProvider(diagnostics={"inspect_oom": False})
    handle = opensandbox_provider.SandboxHandle(
        sandbox_id="foreground-timeout", provider_name="opensandbox", raw=SimpleNamespace(commands=commands)
    )

    with raises_failure(PROVIDER_IO_TIMEOUT, match="command run") as caught:
        await provider.exec(handle, "bash /tests/test.sh", timeout_s=60)

    assert not isinstance(caught.value, TimeoutError)
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert commands.run_calls == 1


@pytest.mark.parametrize("operation", ["upload", "download"])
async def test_file_transfer_request_timeout_is_ungraded_provider_loss(tmp_path: Path, operation: str) -> None:
    files = SimpleNamespace(
        write_file=AsyncMock(side_effect=TimeoutError("write request timed out")),
        read_bytes_stream=AsyncMock(side_effect=TimeoutError("read request timed out")),
    )
    provider = opensandbox_provider.OpenSandboxProvider(operations={"retry_delay_s": 0})
    handle = opensandbox_provider.SandboxHandle(
        sandbox_id="transfer-timeout", provider_name="opensandbox", raw=SimpleNamespace(files=files)
    )
    path = tmp_path / "artifact"
    if operation == "upload":
        path.write_bytes(b"artifact")
        transfer = provider.upload_file(handle, path, "/artifact")
    else:
        transfer = provider.download_file(handle, "/artifact", path)

    with raises_failure(PROVIDER_IO_TIMEOUT) as caught:
        await transfer

    assert not isinstance(caught.value, TimeoutError)
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert files.write_file.await_count + files.read_bytes_stream.await_count == (1 if operation == "upload" else 4)
    if operation == "download":
        assert path.read_bytes() == b""


async def test_exec_hard_cap_does_not_relabel_inner_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A TimeoutError raised inside the dispatch keeps its own message.

    Since Python 3.11 asyncio.TimeoutError IS builtin TimeoutError, so a
    wait_for-based cap caught e.g. an exhausted status-poll budget (a
    minutes-scale failure) and relabeled it as a trip of the hours-scale hard
    cap, corrupting the failure taxonomy. Only the cap's own expiry may carry
    the wedged message.
    """

    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeCommands:
        def __init__(self) -> None:
            self.run_calls = 0
            self.status_calls = 0

        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> Any:
            self.run_calls += 1
            return SimpleNamespace(id="exec-slowpolls")

        async def get_command_status(self, execution_id: str) -> Any:
            self.status_calls += 1
            raise TimeoutError("simulated status poll budget expiry")

        async def get_background_command_logs(self, execution_id: str) -> Any:  # pragma: no cover
            return SimpleNamespace(content="ok", cursor=None)

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 120},
        probe={"command": None},
        operations={
            "background_exec": True,
            "retries": 1,
            "retry_delay_s": 0.001,
            "background_poll_interval_s": 0.01,
        },
    )
    provider._operations = replace(provider._operations, command_timeout_headroom_s=0.0)
    monkeypatch.setattr(provider, "_sandbox_death_evidence", AsyncMock(return_value=None))
    commands = FakeCommands()
    handle = opensandbox_provider.SandboxHandle(
        sandbox_id="sandbox-slowpolls", provider_name="opensandbox", raw=SimpleNamespace(commands=commands)
    )

    with raises_failure(PROVIDER_COMMAND_OUTCOME_UNKNOWN) as exc_info:
        await provider.exec(handle, "echo ok", timeout_s=0.01)

    # The client deadline ends the wait before status recovery could finish;
    # keep the failed read as its cause, without claiming the hard cap tripped.
    assert commands.status_calls >= 1
    assert commands.run_calls == 1
    assert exc_info.value.operation == "command status"
    assert "client wait expired" in exc_info.value.failure_reason
    assert exc_info.value.__cause__.failure_kind == PROVIDER_IO_TIMEOUT
    assert "hard cap" not in str(exc_info.value)


async def test_status_loss_recovery_ends_before_long_agent_command_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCommands:
        def __init__(self) -> None:
            self.run_calls = 0
            self.status_calls: list[str] = []

        async def run(self, command: str, *, opts: Any) -> Any:
            self.run_calls += 1
            return SimpleNamespace(id="one-execution")

        async def get_command_status(self, execution_id: str) -> Any:
            self.status_calls.append(execution_id)
            raise TimeoutError("command status route is unavailable")

        async def get_background_command_logs(self, execution_id: str) -> Any:
            pytest.fail("logs cannot be fetched without a final status")

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, dict, object, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
        operations={"background_exec": True, "retries": 0, "background_poll_interval_s": 0.001},
    )
    provider._operations = replace(provider._operations, status_recovery_timeout_s=0.02)
    inspect = AsyncMock(return_value=None)
    monkeypatch.setattr(provider, "_control_plane_inspect_oom_evidence", inspect)
    commands = FakeCommands()
    handle = opensandbox_provider.SandboxHandle(
        sandbox_id="sandbox-status-loss", provider_name="opensandbox", raw=SimpleNamespace(commands=commands)
    )

    with raises_failure(PROVIDER_STATUS_UNAVAILABLE, match="unavailable beyond"):
        await provider.exec(handle, "long agent command", timeout_s=10_800)

    assert commands.run_calls == 1
    assert len(commands.status_calls) >= 2
    assert set(commands.status_calls) == {"one-execution"}
    assert inspect.await_count == 0  # deprecated route is disabled by default


@pytest.mark.parametrize("evidence_source", ["lifecycle", "inspect", "pending"])
async def test_background_rechecks_late_death_evidence_within_original_recovery_budget(
    monkeypatch: pytest.MonkeyPatch, evidence_source: str
) -> None:
    """Evidence arriving after the first diagnostic slice must still be read without replay."""
    from opensandbox.exceptions import SandboxApiException

    loop = asyncio.get_running_loop()
    now = loop.time()
    started = now
    diagnostic_reads: list[float] = []

    async def advance(delay: float) -> None:
        nonlocal now
        now += delay

    async def get_info() -> Any:
        diagnostic_reads.append(now)
        oom = evidence_source == "lifecycle" and len(diagnostic_reads) >= 2
        return SimpleNamespace(
            status=SimpleNamespace(state="Failed" if oom else "Pending", reason="OOMKilled" if oom else None)
        )

    inspect_reads: list[float] = []

    async def inspect(handle: Any, *, deadline: float) -> str | None:
        inspect_reads.append(now)
        assert now < deadline <= started + 0.08
        return "current sandbox container: OOMKilled" if len(inspect_reads) >= 2 else None

    monkeypatch.setattr(loop, "time", lambda: now)
    monkeypatch.setattr(asyncio, "sleep", advance)
    monkeypatch.setattr(
        opensandbox_provider, "_require_opensandbox_sdk", lambda: (object, object, dict, object, object)
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        diagnostics={"timeout_s": 0.005, "inspect_oom": evidence_source == "inspect"},
        operations={
            "background_exec": True,
            "background_poll_initial_s": 0.02,
            "background_poll_interval_s": 0.02,
            "status_recovery_timeout_s": 0.08,
        },
    )
    monkeypatch.setattr(provider, "_control_plane_inspect_oom_evidence", inspect)
    commands = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(id="one-command")),
        get_command_status=AsyncMock(side_effect=SandboxApiException("backend unreachable", status_code=502)),
        get_background_command_logs=AsyncMock(),
    )
    handle = opensandbox_provider.SandboxHandle(
        "late-evidence", "opensandbox", SimpleNamespace(commands=commands, get_info=AsyncMock(side_effect=get_info))
    )

    expected = {
        "pending": PROVIDER_STATUS_UNAVAILABLE,
        "lifecycle": PROVIDER_UNAVAILABLE,
        "inspect": PROVIDER_OOM_KILLED,
    }[evidence_source]
    with RequestFailureLog.capture() as log, raises_failure(expected) as caught:
        await provider.exec(handle, "agent command", timeout_s=3600)

    commands.run.assert_awaited_once()
    commands.get_background_command_logs.assert_not_awaited()
    assert len(diagnostic_reads) >= 2
    assert diagnostic_reads[1] - diagnostic_reads[0] > provider._diagnostics.timeout_s
    assert now <= started + 0.08
    assert caught.value.__cause__.http_status == 502
    assert isinstance(caught.value.__cause__.__cause__, SandboxApiException)
    assert len(log.report.events) == commands.get_command_status.await_count
    assert all(event.http_status == 502 for event in log.report.events.values())
    if evidence_source == "pending":
        assert now == pytest.approx(started + 0.08)
        assert inspect_reads == []
    else:
        assert commands.get_command_status.await_count == 2
    assert all(call.args == ("one-command",) for call in commands.get_command_status.await_args_list)


async def test_background_diagnostic_retries_share_active_io_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Polling gaps preserve unused I/O time, but repeated diagnostic requests cannot replenish it."""
    loop = asyncio.get_running_loop()
    now = loop.time()
    started = now
    slices: list[float] = []

    async def advance(delay: float) -> None:
        nonlocal now
        now += delay

    async def diagnose(handle: Any, *, deadline: float, poll: bool) -> None:
        nonlocal now
        assert poll is False
        assert deadline <= started + 0.08
        if (remaining := deadline - now) > 0:
            slices.append(remaining)
            now += min(0.002, remaining)

    monkeypatch.setattr(loop, "time", lambda: now)
    monkeypatch.setattr(asyncio, "sleep", advance)
    monkeypatch.setattr(
        opensandbox_provider, "_require_opensandbox_sdk", lambda: (object, object, dict, object, object)
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        diagnostics={"timeout_s": 0.005},
        operations={
            "background_exec": True,
            "background_poll_initial_s": 0.01,
            "background_poll_interval_s": 0.01,
            "status_recovery_timeout_s": 0.08,
        },
    )
    monkeypatch.setattr(provider, "_sandbox_death_evidence", diagnose)
    commands = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(id="one-command")),
        get_command_status=AsyncMock(side_effect=TimeoutError("status unavailable")),
    )
    handle = opensandbox_provider.SandboxHandle("budget", "opensandbox", SimpleNamespace(commands=commands))

    with RequestFailureLog.capture() as log, raises_failure(PROVIDER_STATUS_UNAVAILABLE):
        await provider.exec(handle, "agent command", timeout_s=3600)

    assert slices == pytest.approx([0.005, 0.003, 0.001])
    assert now == pytest.approx(started + 0.08)
    assert commands.get_command_status.await_count > len(slices)
    assert len(log.report.events) == commands.get_command_status.await_count
    commands.run.assert_awaited_once()


@pytest.mark.parametrize(
    "reason,message,expected",
    [
        ("ContainerFailed", "sandbox out of memory", PROVIDER_UNAVAILABLE),
        ("ContainerFailed", "not OOMKilled; exit 137", PROVIDER_UNAVAILABLE),
        ("OOMKilled", "", PROVIDER_UNAVAILABLE),
    ],
)
async def test_exec_and_pty_never_classify_oom_from_lifecycle_reason_alone(reason, message, expected):
    provider = opensandbox_provider.OpenSandboxProvider()
    raw = SimpleNamespace(
        get_info=AsyncMock(
            return_value=SimpleNamespace(status=SimpleNamespace(state="Failed", reason=reason, message=message))
        )
    )
    handle = opensandbox_provider.SandboxHandle("sandbox-1", "opensandbox", raw)
    evidence = await provider._sandbox_death_evidence(handle, deadline=asyncio.get_running_loop().time() + 1)
    assert evidence.failure_kind == expected
    assert evidence.provider_status_reason == reason
    failure = await provider._pty_failure(handle, ConnectionError("socket lost"))
    assert failure.failure_kind == expected
    assert failure.details.provider_status_reason == reason
    assert "lifecycle status" in failure.failure_reason


@pytest.mark.parametrize("request_timeout_s", [120, None])
async def test_exec_background_without_timeout_skips_hard_cap(
    monkeypatch: pytest.MonkeyPatch, request_timeout_s: int | None
) -> None:
    """An uncapped background command stays uncapped; request_timeout_s bounds one poll."""

    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeCommands:
        def __init__(self) -> None:
            self.status_calls: list[str] = []

        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> Any:
            return SimpleNamespace(id="exec-uncapped")

        async def get_command_status(self, execution_id: str) -> Any:
            self.status_calls.append(execution_id)
            running = len(self.status_calls) < 2
            return SimpleNamespace(running=running, exit_code=0, error=None)

        async def get_background_command_logs(self, execution_id: str) -> Any:
            return SimpleNamespace(content="ok", cursor=None)

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    wait_for_timeouts: list[float | None] = []
    real_wait_for = asyncio.wait_for

    async def recording_wait_for(awaitable: Any, timeout: float | None = None) -> Any:
        wait_for_timeouts.append(timeout)
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr(asyncio, "wait_for", recording_wait_for)

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": request_timeout_s},
        probe={"command": None},
        operations={"background_exec": True, "background_poll_interval_s": 0.01},
    )
    handle = opensandbox_provider.SandboxHandle(
        sandbox_id="sandbox-uncapped", provider_name="opensandbox", raw=FakeRaw()
    )

    result = await provider.exec(handle, "echo hi")

    assert result == opensandbox_provider.SandboxExecResult(stdout="ok", stderr=None, return_code=0, error_type=None)
    if request_timeout_s is not None:
        # The per-poll timeout must not become the whole command's hard cap.
        assert 2.0 * request_timeout_s + 30.0 not in wait_for_timeouts


async def test_provider_create_probe_and_close_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"connect_poll_s": 0.01},
        probe={
            "command": "probe",
            "expected_stdout": "ready",
            "timeout_s": 1,
            "deadline_s": 0.01,
        },
    )
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sandbox-1", provider_name="opensandbox", raw=object())

    async def bad_probe(*_args: Any, **_kwargs: Any) -> opensandbox_provider.SandboxExecResult:
        return opensandbox_provider.SandboxExecResult(stdout="not ready", stderr="bad", return_code=1)

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(opensandbox_provider.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(provider, "_exec", bad_probe)
    with pytest.raises(NgFailureEvent) as failure_info:
        await provider._verify_created_handle(handle)
    assert failure_info.value.failure_kind == "provider_not_ready"

    provider = opensandbox_provider.OpenSandboxProvider(
        probe={"command": "probe", "expected_stdout": None, "stable_count": 2, "stable_delay_s": 0.01},
    )
    sleep_calls: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    async def good_probe(*_args: Any, **_kwargs: Any) -> opensandbox_provider.SandboxExecResult:
        return opensandbox_provider.SandboxExecResult(stdout="ready", stderr=None, return_code=0)

    monkeypatch.setattr(opensandbox_provider.asyncio, "sleep", record_sleep)
    monkeypatch.setattr(provider, "_exec", good_probe)
    await provider._verify_created_handle(handle)
    assert sleep_calls == [0.01]

    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": "probe"})

    async def cancelled_probe(*_args: Any, **_kwargs: Any) -> opensandbox_provider.SandboxExecResult:
        raise asyncio.CancelledError()

    monkeypatch.setattr(provider, "_exec", cancelled_probe)
    with pytest.raises(asyncio.CancelledError):
        await provider._verify_created_handle(handle)

    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})

    async def close_raises(_handle: Any) -> None:
        raise RuntimeError("close failed")

    monkeypatch.setattr(provider, "close", close_raises)
    await provider._cleanup_failed_create_handle(handle)
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})

    class StopAlreadyGoneRaw:
        async def kill(self) -> None:
            raise RuntimeError("sandbox sandbox-1 not found")

        async def close(self) -> None:
            return None

    await provider.close(
        opensandbox_provider.SandboxHandle(
            sandbox_id="sandbox-1",
            provider_name="opensandbox",
            raw=StopAlreadyGoneRaw(),
        ),
    )

    class StopAndCloseFailRaw:
        async def kill(self) -> None:
            raise RuntimeError("stop failed")

        async def close(self) -> None:
            raise RuntimeError("close failed")

    with pytest.raises(NgFailureEvent) as observed_failure:
        await provider.close(
            opensandbox_provider.SandboxHandle(
                sandbox_id="sandbox-2",
                provider_name="opensandbox",
                raw=StopAndCloseFailRaw(),
            ),
        )
    assert observed_failure.value.failure_kind == "provider_request_failed"

    class StopFailsCloseSucceedsRaw:
        async def kill(self) -> None:
            raise RuntimeError("stop failed")

        async def close(self) -> None:
            return None

    with pytest.raises(NgFailureEvent) as observed_failure:
        await provider.close(
            opensandbox_provider.SandboxHandle(
                sandbox_id="sandbox-3",
                provider_name="opensandbox",
                raw=StopFailsCloseSucceedsRaw(),
            ),
        )
    assert observed_failure.value.failure_kind == "provider_request_failed"


async def test_close_treats_missing_sandbox_as_terminated_without_retry(caplog: pytest.LogCaptureFixture) -> None:
    from opensandbox.exceptions import SandboxApiException  # noqa: PLC0415

    provider = opensandbox_provider.OpenSandboxProvider(
        probe={"command": None},
        operations={"retries": 2, "retry_delay_s": 0, "retry_max_delay_s": 0},
    )
    not_found = SandboxApiException(
        "Kill sandbox sandbox-1 failed: Sandbox 'sandbox-1' not found | "
        "[KUBERNETES::SANDBOX_NOT_FOUND] Sandbox 'sandbox-1' not found"
    )
    not_found.status_code = 404
    kill_calls = 0

    class AlreadyGoneRaw:
        async def kill(self) -> None:
            nonlocal kill_calls
            kill_calls += 1
            raise not_found

        async def close(self) -> None:
            return None

    with caplog.at_level(logging.DEBUG):
        await provider.close(
            opensandbox_provider.SandboxHandle(
                sandbox_id="sandbox-1",
                provider_name="opensandbox",
                raw=AlreadyGoneRaw(),
            ),
        )

    assert kill_calls == 1
    assert any("already gone; treating terminate as success" in record.message for record in caplog.records)


@pytest.mark.parametrize("already_deleted", [False, True], ids=["deleted", "missing"])
async def test_close_recovers_an_early_lost_delete_reply(already_deleted: bool) -> None:
    from opensandbox.exceptions import SandboxApiException  # noqa: PLC0415

    missing = SandboxApiException("Sandbox sandbox-1 not found")
    missing.status_code = 404
    raw = SimpleNamespace(
        kill=AsyncMock(side_effect=[TimeoutError("delete response lost"), missing if already_deleted else None]),
        close=AsyncMock(),
    )
    handle = opensandbox_provider.SandboxHandle("sandbox-1", "opensandbox", raw)
    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"retries": 1, "retry_delay_s": 0, "retry_max_delay_s": 0},
        probe={"command": None},
    )

    with RequestFailureLog.capture() as log:
        await provider.close(handle)
        report = log.reply_report(None)
    assert [event.failure_kind for event in report.events.values()] == [PROVIDER_IO_TIMEOUT]

    assert raw.kill.await_count == 2
    raw.close.assert_awaited_once_with()
    assert handle.raw is raw


async def test_delete_recovery_respects_configured_fail_action() -> None:
    from opensandbox.exceptions import SandboxApiException

    error = SandboxApiException("temporarily unavailable", status_code=503)
    raw = SimpleNamespace(kill=AsyncMock(side_effect=[error, None]), close=AsyncMock())
    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"retries": 3, "outcome_policy": {"by_status_code": {503: "fail"}}},
        probe={"command": None},
    )
    handle = opensandbox_provider.SandboxHandle("sandbox-1", "opensandbox", raw)

    with raises_failure(PROVIDER_UNAVAILABLE) as caught:
        await provider.close(handle)

    assert caught.value.http_status == 503
    assert caught.value.__cause__ is error
    raw.kill.assert_awaited_once_with()
    raw.close.assert_not_awaited()


@pytest.mark.parametrize("after_timeout", [False, True], ids=["first-attempt", "after-lost-reply"])
@pytest.mark.parametrize("failure", ["permission", "unauthorized", "forbidden", "cancelled"])
async def test_delete_recovery_keeps_permission_and_cancellation_terminal(failure: str, after_timeout: bool) -> None:
    from opensandbox.exceptions import SandboxApiException  # noqa: PLC0415

    if failure == "permission":
        error = PermissionError("Sandbox sandbox-1 not found or permission denied")
    elif failure == "cancelled":
        error = asyncio.CancelledError("connection reset")
    else:
        error = SandboxApiException("Sandbox sandbox-1 not found or permission denied")
        error.status_code = 401 if failure == "unauthorized" else 403
        error.__cause__ = ConnectionError("connection reset")
    errors = ([TimeoutError("delete response lost")] if after_timeout else []) + [error, None]
    raw = SimpleNamespace(kill=AsyncMock(side_effect=errors), close=AsyncMock())
    handle = opensandbox_provider.SandboxHandle("sandbox-1", "opensandbox", raw)
    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"retries": 3, "retry_delay_s": 0, "retry_max_delay_s": 0},
        probe={"command": None},
    )

    with pytest.raises(asyncio.CancelledError if failure == "cancelled" else NgFailureEvent) as caught:
        await provider.close(handle)

    if failure == "cancelled":
        assert caught.value is error
    else:
        assert caught.value.__cause__ is error
        assert caught.value.failure_kind == (
            "provider_credentials_rejected" if failure in ("unauthorized", "forbidden") else "provider_request_failed"
        )
    assert raw.kill.await_count == 1 + after_timeout
    raw.close.assert_not_awaited()
    assert handle.raw is raw


@pytest.mark.parametrize("after_timeout", [False, True], ids=["first-attempt", "after-lost-reply"])
@pytest.mark.parametrize("failure", ["permission", "permission-not-found", "unauthorized", "forbidden", "cancelled"])
async def test_delete_recovery_keeps_wrapped_terminal_errors(failure: str, after_timeout: bool) -> None:
    from opensandbox.adapters.converter.exception_converter import ExceptionConverter  # noqa: PLC0415
    from opensandbox.exceptions import SandboxApiException, SandboxInternalException  # noqa: PLC0415

    if failure.startswith("permission"):
        message = (
            "Sandbox sandbox-1 not found or permission denied"
            if failure.endswith("not-found")
            else "permission denied"
        )
        error = ExceptionConverter.to_sandbox_exception(PermissionError(message))
    else:
        if failure == "cancelled":
            cause = asyncio.CancelledError("connection reset")
        else:
            cause = SandboxApiException("permission denied", status_code=401 if failure == "unauthorized" else 403)
        error = SandboxInternalException("Network connectivity error: Sandbox sandbox-1 not found", cause=cause)
    raw = SimpleNamespace(
        kill=AsyncMock(side_effect=([TimeoutError("delete response lost")] if after_timeout else []) + [error, None]),
        close=AsyncMock(),
    )
    handle = opensandbox_provider.SandboxHandle("sandbox-1", "opensandbox", raw)
    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"retries": 3, "retry_delay_s": 0, "retry_max_delay_s": 0},
        probe={"command": None},
    )

    with pytest.raises(NgFailureEvent) as caught:
        await provider.close(handle)

    assert caught.value.__cause__ is error
    assert caught.value.failure_kind == "provider_internal_error"
    assert raw.kill.await_count == 1 + after_timeout
    raw.close.assert_not_awaited()
    assert handle.raw is raw


@pytest.mark.parametrize("terminal", [False, True])
async def test_delete_terminal_error_check_handles_cyclic_causes(terminal: bool) -> None:
    error = ValueError("invalid delete")
    cause = PermissionError("permission denied") if terminal else ValueError("invalid cause")
    error.__cause__ = cause
    cause.__cause__ = error
    raw = SimpleNamespace(kill=AsyncMock(side_effect=[error, None]), close=AsyncMock())
    handle = opensandbox_provider.SandboxHandle("sandbox-1", "opensandbox", raw)
    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"retries": 3, "retry_delay_s": 0, "retry_max_delay_s": 0},
        probe={"command": None},
    )

    with raises_failure(PROVIDER_REQUEST_FAILED) as caught:
        await provider.close(handle)

    assert caught.value.__cause__ is error
    raw.kill.assert_awaited_once_with()
    raw.close.assert_not_awaited()
    assert handle.raw is raw


async def test_delete_recovery_exhaustion_keeps_the_handle_available_for_owner_retry() -> None:
    error = TimeoutError("delete response lost")
    raw = SimpleNamespace(kill=AsyncMock(side_effect=error), close=AsyncMock())
    handle = opensandbox_provider.SandboxHandle("sandbox-1", "opensandbox", raw)
    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"retries": 2, "retry_delay_s": 0, "retry_max_delay_s": 0},
        probe={"command": None},
    )

    with raises_failure(PROVIDER_IO_TIMEOUT):
        await provider.close(handle)

    assert raw.kill.await_count == 3
    raw.close.assert_not_awaited()
    assert handle.raw is raw
    raw.kill.side_effect = None
    await provider.close(handle)
    assert raw.kill.await_count == 4
    raw.close.assert_awaited_once_with()


@pytest.mark.parametrize("after_timeout", [False, True], ids=["first-attempt", "after-lost-reply"])
async def test_caller_cancellation_during_delete_is_not_retried(after_timeout: bool) -> None:
    entered = asyncio.Event()
    calls = 0

    async def kill() -> None:
        nonlocal calls
        calls += 1
        if after_timeout and calls == 1:
            raise TimeoutError("delete response lost")
        entered.set()
        await asyncio.Event().wait()

    raw = SimpleNamespace(kill=AsyncMock(side_effect=kill), close=AsyncMock())
    handle = opensandbox_provider.SandboxHandle("sandbox-1", "opensandbox", raw)
    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"retries": 3, "retry_delay_s": 0, "retry_max_delay_s": 0},
        probe={"command": None},
    )
    task = asyncio.create_task(provider.close(handle))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel("caller stopped cleanup")
    with pytest.raises(asyncio.CancelledError, match="caller stopped cleanup"):
        await task

    assert raw.kill.await_count == 1 + after_timeout
    raw.close.assert_not_awaited()
    assert handle.raw is raw


async def test_delete_recovery_attempts_share_one_close_deadline() -> None:
    entered, cancelled = asyncio.Event(), asyncio.Event()
    calls = 0

    async def kill() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("delete response lost")
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    raw = SimpleNamespace(kill=AsyncMock(side_effect=kill), close=AsyncMock())
    handle = opensandbox_provider.SandboxHandle("sandbox-1", "opensandbox", raw)
    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"retries": 3, "retry_delay_s": 0, "retry_max_delay_s": 0, "close_timeout_s": 0.1},
        probe={"command": None},
    )

    with raises_failure(PROVIDER_IO_TIMEOUT):
        await asyncio.wait_for(provider.close(handle), 1)

    assert entered.is_set() and cancelled.is_set()
    assert raw.kill.await_count == 2
    raw.close.assert_not_awaited()
    assert handle.raw is raw


@pytest.mark.parametrize("cancel_caller", [False, True], ids=["deadline", "caller-cancellation"])
async def test_delete_recovery_backoff_shares_the_deadline_and_allows_cancellation(
    monkeypatch: pytest.MonkeyPatch, cancel_caller: bool
) -> None:
    import tenacity

    sleeping = asyncio.Event()
    monkeypatch.setattr(tenacity, "wait_random_exponential", lambda **_: tenacity.wait_fixed(30))
    original_log_retry = opensandbox_provider._log_operation_retry

    def log_retry(*args: Any, **kwargs: Any) -> None:
        sleeping.set()
        original_log_retry(*args, **kwargs)

    monkeypatch.setattr(opensandbox_provider, "_log_operation_retry", log_retry)
    raw = SimpleNamespace(kill=AsyncMock(side_effect=TimeoutError("delete response lost")), close=AsyncMock())
    handle = opensandbox_provider.SandboxHandle("sandbox-1", "opensandbox", raw)
    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"retries": 3, "close_timeout_s": 30 if cancel_caller else 0.1},
        probe={"command": None},
    )
    task = asyncio.create_task(provider.close(handle))
    await asyncio.wait_for(sleeping.wait(), 1)
    if cancel_caller:
        task.cancel("caller stopped cleanup")
    else:
        done, _ = await asyncio.wait({task}, timeout=0.5)
        if not done:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert done, "DELETE deadline must interrupt retry backoff before the outer test guard"
    with pytest.raises(asyncio.CancelledError if cancel_caller else NgFailureEvent) as caught:
        await asyncio.wait_for(task, 1)
    if not cancel_caller:
        assert caught.value.failure_kind == PROVIDER_IO_TIMEOUT

    raw.kill.assert_awaited_once_with()
    raw.close.assert_not_awaited()
    assert handle.raw is raw


async def test_delete_recovery_does_not_spend_the_separate_transport_close_budget() -> None:
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def close() -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    raw = SimpleNamespace(
        kill=AsyncMock(side_effect=[TimeoutError("delete response lost"), None]), close=AsyncMock(side_effect=close)
    )
    handle = opensandbox_provider.SandboxHandle("sandbox-1", "opensandbox", raw)
    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"retries": 1, "retry_delay_s": 0, "retry_max_delay_s": 0, "close_timeout_s": 0.1},
        probe={"command": None},
    )

    # Local transport cleanup is best effort after confirmed deletion; its own
    # timeout must not be converted into an unconfirmed DELETE failure.
    await asyncio.wait_for(provider.close(handle), 1)

    assert entered.is_set() and cancelled.is_set()
    assert raw.kill.await_count == 2
    raw.close.assert_awaited_once_with()


async def test_delete_timeout_recovery_does_not_enable_disabled_create_retries() -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"retries": 0}, operations={"retries": 3}, probe={"command": None}
    )
    provider._create_once = AsyncMock(
        side_effect=sandbox_failure(
            operation="create", failure_kind="provider_create_timeout", reason="ambiguous create"
        )
    )

    with raises_failure("provider_create_timeout"):
        await provider.create(SandboxSpec(image="task"))

    provider._create_once.assert_awaited_once()


async def test_non_terminate_operation_still_raises_on_missing_sandbox() -> None:
    from opensandbox.exceptions import SandboxApiException  # noqa: PLC0415

    provider = opensandbox_provider.OpenSandboxProvider(
        probe={"command": None},
        operations={"retries": 2, "retry_delay_s": 0, "retry_max_delay_s": 0},
    )
    not_found = SandboxApiException("Get sandbox sandbox-1 failed: Sandbox 'sandbox-1' not found")
    not_found.status_code = 404
    get_info_calls = 0

    class MissingRaw:
        async def get_info(self) -> Any:
            nonlocal get_info_calls
            get_info_calls += 1
            raise not_found

    with pytest.raises(NgFailureEvent) as observed_failure:
        await provider.status(
            opensandbox_provider.SandboxHandle(
                sandbox_id="sandbox-1",
                provider_name="opensandbox",
                raw=MissingRaw(),
            ),
        )
    assert observed_failure.value.failure_kind == "provider_request_failed"

    assert get_info_calls == 1


async def test_create_once_and_connect_after_create_error_paths(
    fake_opensandbox_sdk: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"timeout_s": 1, "skip_health_check": True},
        probe={"command": None},
    )
    monkeypatch.setattr(opensandbox_provider, "_to_volumes", lambda volumes: volumes)
    spec = SandboxSpec(
        image="image:tag",
        ttl_s=10,
        ready_timeout_s=20,
        resources=SandboxResources(cpu=2, memory_mib=8192, disk_gib=20, gpu=1, gpu_type="H100"),
        entrypoint=["/bin/sh"],
        provider_options={
            "snapshot_id": "snapshot-1",
            "platform": {"os": "linux", "arch": "amd64"},
            "volumes": [{"name": "workspace"}],
            "skip_health_check": False,
        },
    )
    handle = await provider._create_once(spec)
    assert handle.sandbox_id == "sandbox-1"
    assert FakeSandbox.created_kwargs["snapshot_id"] == "snapshot-1"
    assert FakeSandbox.created_kwargs["timeout"] == timedelta(seconds=10)
    assert FakeSandbox.created_kwargs["ready_timeout"] == timedelta(seconds=20)
    assert FakeSandbox.created_kwargs["resource"] == {
        "cpu": "2",
        "memory": "8192Mi",
        "ephemeral-storage": "20Gi",
        "gpu": "1",
        "gpu_type": "H100",
    }
    assert FakeSandbox.created_kwargs["entrypoint"] == ["/bin/sh"]
    assert FakeSandbox.created_kwargs["platform"] == FakePlatformSpec(os="linux", arch="amd64")
    assert FakeSandbox.created_kwargs["volumes"] == [{"name": "workspace"}]
    assert FakeSandbox.created_kwargs["skip_health_check"] is True

    class FailingConnectSandbox(FakeSandbox):
        @classmethod
        async def connect(cls, *args: Any, **kwargs: Any) -> "FakeSandbox":
            del args, kwargs
            raise ConnectionError("pod may still be starting")

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (FailingConnectSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"connect_attempt_timeout_s": 0.01, "connect_poll_s": 0.01},
        probe={"command": None},
    )

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(opensandbox_provider.asyncio, "sleep", no_sleep)
    with pytest.raises(NgFailureEvent) as failure_info:
        await provider._connect_after_create(
            opensandbox_provider.SandboxHandle(sandbox_id="sandbox-1", provider_name="opensandbox", raw=None),
            SandboxSpec(image="image:tag"),
        )
    assert failure_info.value.failure_kind == "provider_create_timeout"

    class CancelledConnectSandbox(FakeSandbox):
        @classmethod
        async def connect(cls, *args: Any, **kwargs: Any) -> "FakeSandbox":
            del args, kwargs
            raise asyncio.CancelledError()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (CancelledConnectSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    with pytest.raises(asyncio.CancelledError):
        await provider._connect_after_create(
            opensandbox_provider.SandboxHandle(sandbox_id="sandbox-1", provider_name="opensandbox", raw=None),
            SandboxSpec(image="image:tag", ready_timeout_s=1),
        )

    class NonRetryableConnectSandbox(FakeSandbox):
        @classmethod
        async def connect(cls, *args: Any, **kwargs: Any) -> "FakeSandbox":
            del args, kwargs
            raise ValueError("bad connection request")

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (NonRetryableConnectSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    with pytest.raises(NgFailureEvent) as observed_failure:
        await provider._connect_after_create(
            opensandbox_provider.SandboxHandle(sandbox_id="sandbox-1", provider_name="opensandbox", raw=None),
            SandboxSpec(image="image:tag", ready_timeout_s=1),
        )
    assert observed_failure.value.failure_kind == "provider_request_failed"

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (FakeSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 3},
        probe={"command": None},
    )
    handle = await provider._create_once(SandboxSpec(image="image:tag", provider_options={"skip_health_check": True}))
    assert handle.sandbox_id == "sandbox-1"
    assert FakeSandbox.created_kwargs["skip_health_check"] is True

    class TimeoutSandbox(FakeSandbox):
        @classmethod
        async def create(cls, **_kwargs: Any) -> "FakeSandbox":
            await asyncio.get_running_loop().create_future()
            return cls()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (TimeoutSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"timeout_s": 0.01},
        probe={"command": None},
    )
    with pytest.raises(NgFailureEvent) as failure_info:
        await provider._create_once(SandboxSpec(image="image:tag"))
    assert failure_info.value.failure_kind == "provider_create_timeout"

    class EmptyCreateSandbox(FakeSandbox):
        @classmethod
        async def create(cls, **_kwargs: Any) -> None:
            return None

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (EmptyCreateSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    with pytest.raises(NgFailureEvent, match="returned no sandbox handle"):
        await provider._create_once(SandboxSpec(image="image:tag"))

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (FakeSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": "probe"})
    cleanup_calls: list[str] = []

    async def fail_verify(_handle: opensandbox_provider.SandboxHandle) -> None:
        raise RuntimeError("probe failed")

    async def cleanup(handle: opensandbox_provider.SandboxHandle) -> None:
        cleanup_calls.append(handle.sandbox_id)

    monkeypatch.setattr(provider, "_verify_created_handle", fail_verify)
    monkeypatch.setattr(provider, "_cleanup_failed_create_handle", cleanup)
    with pytest.raises(RuntimeError, match="probe failed"):
        await provider._create_once(SandboxSpec(image="image:tag"))
    assert cleanup_calls == ["sandbox-1"]


async def test_retry_classification_and_await_sdk_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"retries": 0},
        probe={"command": None},
    )
    assert await provider.aclose() is None
    assert await provider._await_sdk_call(_return_value("ok"), operation="op", sandbox_id="sandbox-1", timeout_s=None)

    async def cancelled() -> None:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await provider._await_sdk_operation(
            cancelled,
            operation="cancelled",
            sandbox_id="sandbox-1",
            timeout_s=None,
        )


async def test_retry_loop_empty_iterator_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyAsyncRetrying:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def __aiter__(self) -> "EmptyAsyncRetrying":
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_tenacity",
        lambda: (EmptyAsyncRetrying, lambda predicate: predicate, lambda attempts: attempts, lambda **kwargs: kwargs),
    )

    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    with pytest.raises(RuntimeError, match="SDK operation retry loop did not run"):
        await provider._await_sdk_operation(
            lambda: _return_value("ok"),
            operation="noop",
            sandbox_id="sandbox-1",
            timeout_s=None,
        )
    with pytest.raises(RuntimeError, match="create retry loop did not run"):
        await provider._create_with_retries(SandboxSpec(image="image:tag"))


async def _return_value(value: Any) -> Any:
    return value


ATTRIBUTION_ENV_VARS = (
    "NEMO_GYM_TEAM",
    "NEMO_GYM_USER",
    "NEMO_GYM_WORKLOAD",
    "NEMO_GYM_RUN_ID",
    "NEMO_GYM_CONFIG_PATH",
    "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_USER",
    "SLURM_JOB_NAME",
)


@pytest.fixture
def clean_attribution_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ATTRIBUTION_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


async def test_create_injects_attribution_metadata(
    fake_opensandbox_sdk: None,
    clean_attribution_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEMO_GYM_TEAM", "gym team")  # sanitized to a valid label value below
    monkeypatch.setenv("NEMO_GYM_USER", "alice")
    monkeypatch.setenv("NEMO_GYM_WORKLOAD", "swe-gym")
    monkeypatch.setenv("NEMO_GYM_RUN_ID", "run-123")
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 10},
        probe={"command": None},
    )

    await provider.create(SandboxSpec(image="image:tag", metadata={"purpose": "test"}))

    assert len(FakeSandbox.created_kwargs["metadata"].pop("nemo-gym.nvidia.com/create-attempt")) == 32
    assert FakeSandbox.created_kwargs["metadata"] == {
        "nemo-gym.nvidia.com/team": "gym_team",
        "nemo-gym.nvidia.com/user": "alice",
        "nemo-gym.nvidia.com/workload": "swe-gym",
        "nemo-gym.nvidia.com/run": "run-123",
        "purpose": "test",
    }


async def test_create_spec_metadata_and_config_win_over_attribution_detection(
    fake_opensandbox_sdk: None,
    clean_attribution_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEMO_GYM_TEAM", "env-team")
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 10},
        probe={"command": None},
        attribution={"team": "cfg-team", "user": "cfg-user", "workload": "cfg-workload", "run": "cfg-run"},
    )

    await provider.create(SandboxSpec(image="image:tag", metadata={"nemo-gym.nvidia.com/team": "explicit-team"}))

    assert len(FakeSandbox.created_kwargs["metadata"].pop("nemo-gym.nvidia.com/create-attempt")) == 32
    assert FakeSandbox.created_kwargs["metadata"] == {
        "nemo-gym.nvidia.com/team": "explicit-team",
        "nemo-gym.nvidia.com/user": "cfg-user",
        "nemo-gym.nvidia.com/workload": "cfg-workload",
        "nemo-gym.nvidia.com/run": "cfg-run",
    }


async def test_create_attribution_disabled(
    fake_opensandbox_sdk: None,
    clean_attribution_env: None,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 10},
        probe={"command": None},
        attribution={"enabled": False, "team": "cfg-team"},
    )

    await provider.create(SandboxSpec(image="image:tag"))

    # Attribution off still leaves the per-attempt key: it is what finds a timed-out create's orphan.
    assert list(FakeSandbox.created_kwargs["metadata"]) == ["nemo-gym.nvidia.com/create-attempt"]


@pytest.mark.parametrize(
    ("key_prefix", "expected_key"),
    [
        ("", "team"),
        ("acme.example.com/", "acme.example.com/team"),
        ("acme.example.com", "acme.example.com/team"),  # trailing slash is normalized in
        ("  ", "team"),
    ],
)
async def test_create_attribution_key_prefix(
    fake_opensandbox_sdk: None,
    clean_attribution_env: None,
    key_prefix: str,
    expected_key: str,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 10},
        probe={"command": None},
        attribution={"team": "cfg-team", "key_prefix": key_prefix},
    )

    await provider.create(SandboxSpec(image="image:tag"))

    metadata = FakeSandbox.created_kwargs["metadata"]
    assert metadata[expected_key] == "cfg-team"


async def test_create_attribution_run_id_generated(
    fake_opensandbox_sdk: None,
    clean_attribution_env: None,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 10},
        probe={"command": None},
    )

    await provider.create(SandboxSpec(image="image:tag"))

    assert FakeSandbox.created_kwargs["metadata"]["nemo-gym.nvidia.com/run"]  # generated per process


@pytest.mark.parametrize("key_prefix", ["Not_A_Valid_Prefix/", "-bad.example.com/", "bad..example.com/"])
def test_attribution_invalid_key_prefix_raises(key_prefix: str) -> None:
    with pytest.raises(ValueError, match="key_prefix"):
        opensandbox_provider.OpenSandboxAttributionConfig(key_prefix=key_prefix)


async def test_connect_health_checks_by_default(fake_opensandbox_sdk: None) -> None:
    """An unchecked handle would defer the exec-daemon startup gap to the first call."""
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})

    handle = await provider.connect({"sandbox_id": "sandbox-9"})

    assert handle.sandbox_id == "sandbox-9"
    assert FakeSandbox.ready_calls == 1


async def test_connect_accepts_a_paused_sandbox_without_health_checking_execd(fake_opensandbox_sdk: None) -> None:
    FakeSandbox.connected_state = "PAUSED"
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})

    handle = await provider.connect({"sandbox_id": "sandbox-9"})

    assert handle.sandbox_id == "sandbox-9"
    assert FakeSandbox.ready_calls == 0


async def test_connect_honours_skip_health_check_opt_out(fake_opensandbox_sdk: None) -> None:
    """Callers that explicitly opt out still get an unchecked handle."""
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"skip_health_check": True},
        probe={"command": None},
    )

    await provider.connect({"sandbox_id": "sandbox-9"})

    assert FakeSandbox.ready_calls == 0


async def test_connect_preserves_failure_when_cleanup_stalls(
    fake_opensandbox_sdk: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(FakeSandbox, "check_ready", AsyncMock(side_effect=ConnectionError("readiness failed")))
    close = AsyncMock(side_effect=asyncio.Event().wait)
    monkeypatch.setattr(FakeSandbox, "close", close)
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"connect_attempt_timeout_s": 0.05},
        operations={"close_timeout_s": 0.05},
        probe={"command": None},
    )

    with pytest.raises(NgFailureEvent) as observed_failure:
        async with asyncio.timeout(0.5):
            await provider.connect({"sandbox_id": "sandbox-9"})
    assert observed_failure.value.failure_kind == "provider_unavailable"
    close.assert_awaited_once_with()


async def test_pause_waits_until_opensandbox_reports_paused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"connect_attempt_timeout_s": 0.000001},
        operations={"pause_resume_timeout_s": 1},
    )
    raw = SimpleNamespace(
        pause=AsyncMock(),
        get_info=AsyncMock(
            side_effect=[
                SimpleNamespace(status=SimpleNamespace(state="PAUSING")),
                SimpleNamespace(status=SimpleNamespace(state="PAUSED")),
            ]
        ),
    )
    handle = opensandbox_provider.SandboxHandle("sandbox-paused", "opensandbox", raw)

    await provider.pause(handle)

    raw.pause.assert_awaited_once_with()
    assert raw.get_info.await_count == 2


async def test_pause_failure_is_not_retried() -> None:
    provider = opensandbox_provider.OpenSandboxProvider(operations={"retries": 5})
    raw = SimpleNamespace(
        pause=AsyncMock(side_effect=ConnectionError("pause request outcome is unknown")),
    )
    handle = opensandbox_provider.SandboxHandle("sandbox-paused", "opensandbox", raw)

    with pytest.raises(NgFailureEvent) as observed_failure:
        await provider.pause(handle)
    assert observed_failure.value.failure_kind == "provider_unavailable"

    raw.pause.assert_awaited_once_with()


async def test_pause_timeout_bounds_the_full_transition() -> None:
    opensandbox_provider._require_tenacity()
    status_started = asyncio.Event()

    async def slow_pause() -> None:
        await asyncio.sleep(0.05)

    async def stalled_status() -> None:
        status_started.set()
        await asyncio.Event().wait()

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 1},
        operations={"pause_resume_timeout_s": 0.1, "retries": 0},
    )
    raw = SimpleNamespace(
        pause=slow_pause,
        get_info=stalled_status,
    )
    handle = opensandbox_provider.SandboxHandle("sandbox-paused", "opensandbox", raw)

    with pytest.raises(NgFailureEvent, match="waiting .* to pause after 0\\.1s") as observed_failure:
        await asyncio.wait_for(provider.pause(handle), timeout=0.14)
    assert observed_failure.value.failure_kind == "provider_io_timeout"
    assert status_started.is_set()


async def test_pause_preserves_a_shorter_request_timeout() -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 0.001},
        operations={"pause_resume_timeout_s": 0.05},
    )
    raw = SimpleNamespace(pause=AsyncMock(side_effect=asyncio.Event().wait))
    handle = opensandbox_provider.SandboxHandle("sandbox-paused", "opensandbox", raw)

    with pytest.raises(NgFailureEvent) as observed_failure:
        await provider.pause(handle)
    assert observed_failure.value.failure_kind == "provider_io_timeout"


async def test_resume_rebuilds_the_sdk_handle_after_readiness(fake_opensandbox_sdk: None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5, "keepalive_expiry_s": None},
        create={
            "connect_poll_s": 0.25,
            "skip_health_check": False,
        },
        operations={"pause_resume_timeout_s": 7},
    )
    raw = SimpleNamespace(close=AsyncMock())
    handle = opensandbox_provider.SandboxHandle("sandbox-paused", "opensandbox", raw)

    await provider.resume(handle)

    assert handle.raw is not raw
    raw.close.assert_awaited_once_with()
    assert FakeSandbox.resumed_args == ("sandbox-paused",)
    assert FakeSandbox.resumed_kwargs["resume_timeout"] == timedelta(seconds=7)
    assert FakeSandbox.resumed_kwargs["health_check_polling_interval"] == timedelta(seconds=0.25)
    assert FakeSandbox.resumed_kwargs["skip_health_check"] is False
    assert FakeSandbox.resumed_kwargs["connection_config"].kwargs["request_timeout"] == timedelta(seconds=5)


async def test_resume_failure_keeps_the_paused_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    resume = AsyncMock(side_effect=ConnectionError("resume request outcome is unknown"))
    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (SimpleNamespace(resume=resume), FakeConnectionConfig, object, FakePlatformSpec, FakeVolume),
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"keepalive_expiry_s": None},
        operations={"retries": 5},
    )
    raw = SimpleNamespace(close=AsyncMock())
    handle = opensandbox_provider.SandboxHandle("sandbox-paused", "opensandbox", raw)

    with pytest.raises(NgFailureEvent) as observed_failure:
        await provider.resume(handle)
    assert observed_failure.value.failure_kind == "provider_unavailable"

    assert resume.await_count == 1
    raw.close.assert_not_awaited()


async def test_resume_timeout_bounds_the_full_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDK applies resume_timeout only to the readiness check; the request and endpoint rebuild must count too."""
    resume_started = asyncio.Event()

    async def stalled_resume(*_args: Any, **_kwargs: Any) -> Any:
        resume_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (SimpleNamespace(resume=stalled_resume), FakeConnectionConfig, object, FakePlatformSpec, FakeVolume),
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"keepalive_expiry_s": None},
        operations={"pause_resume_timeout_s": 0.05},
    )
    raw = SimpleNamespace(close=AsyncMock())
    handle = opensandbox_provider.SandboxHandle("sandbox-paused", "opensandbox", raw)

    with pytest.raises(
        NgFailureEvent, match="to resume after 0\\.05s; reconnect and check status\\(\\)"
    ) as observed_failure:
        await asyncio.wait_for(provider.resume(handle), timeout=1)
    assert observed_failure.value.failure_kind == "provider_io_timeout"

    assert resume_started.is_set()
    assert handle.raw is raw
    raw.close.assert_not_awaited()


async def test_resume_timeout_closes_an_sdk_owned_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """The timeout cancels the SDK call with CancelledError, which the SDK's own cleanup never sees."""
    pytest.importorskip("opensandbox", reason="opensandbox SDK is not installed")
    from opensandbox.config import ConnectionConfig

    transport_close = AsyncMock()

    async def stalled_resume(_sandbox_id: str, *, connection_config: Any, **_kwargs: Any) -> Any:
        assert connection_config.transport is not None and connection_config._owns_transport
        monkeypatch.setattr(connection_config.transport, "aclose", transport_close)
        await asyncio.Event().wait()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (SimpleNamespace(resume=stalled_resume), ConnectionConfig, object, FakePlatformSpec, FakeVolume),
    )
    # The only configuration that leaves the SDK to build its own transport.
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={
            "domain": "sandbox.example",
            "keepalive_expiry_s": None,
            "transport_backend": "httpx",
            "tls_verify": True,
        },
        operations={"pause_resume_timeout_s": 0.05},
    )
    handle = opensandbox_provider.SandboxHandle("sandbox-paused", "opensandbox", SimpleNamespace(close=AsyncMock()))

    with pytest.raises(NgFailureEvent, match="to resume after 0\\.05s") as observed_failure:
        await asyncio.wait_for(provider.resume(handle), timeout=1)
    assert observed_failure.value.failure_kind == "provider_io_timeout"

    transport_close.assert_awaited_once_with()


async def test_resume_installs_the_new_handle_before_closing_the_old_one(fake_opensandbox_sdk: None) -> None:
    """A cancellation during the best-effort close of the old handle must not lose a successful resume."""
    provider = opensandbox_provider.OpenSandboxProvider(connection={"keepalive_expiry_s": None})
    raw = SimpleNamespace(close=AsyncMock(side_effect=asyncio.Event().wait))
    handle = opensandbox_provider.SandboxHandle("sandbox-paused", "opensandbox", raw)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(provider.resume(handle), timeout=0.05)

    assert isinstance(handle.raw, FakeSandbox)
    assert handle.raw.id == "sandbox-paused"
    raw.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_exec_never_replays_an_ambiguous_502_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 502 does not establish non-delivery, regardless of the retry setting."""

    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Backend502Error(Exception):
        status_code = 502

    calls = {"n": 0}

    class FakeCommands:
        async def run(self, command: str, *, opts: FakeRunCommandOpts, handlers: Any = None) -> Any:
            calls["n"] += 1
            if calls["n"] < 3:
                raise Backend502Error("Failed to run command. Status code: 502")
            return SimpleNamespace(logs=SimpleNamespace(stdout=[], stderr=[]), error=None, exit_code=0)

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
        operations={"retries": 3},
    )
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sb-flap", provider_name="opensandbox", raw=FakeRaw())

    with raises_failure(PROVIDER_UNAVAILABLE) as caught:
        await provider.exec(handle, "echo ok", timeout_s=30)
    assert caught.value.http_status == 502
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_exec_persistent_502_raises_typed_backend_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 502 preserves unavailable-backend evidence without replay or inferred death."""

    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Backend502Error(Exception):
        status_code = 502

    calls = {"n": 0}

    class FakeCommands:
        async def run(self, command: str, *, opts: FakeRunCommandOpts, handlers: Any = None) -> Any:
            calls["n"] += 1
            raise Backend502Error("Failed to run command. Status code: 502")

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()

        async def get_info(self) -> Any:
            raise ConnectionError("status API unavailable")

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
        operations={"retries": 2},
    )
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sb-dead", provider_name="opensandbox", raw=FakeRaw())

    with raises_failure(PROVIDER_UNAVAILABLE, match="outcome is unknown"):
        await provider.exec(handle, "echo ok", timeout_s=30)

    assert calls["n"] == 1  # no receipt that replay is safe


@pytest.mark.parametrize(
    ("operations_overrides", "expected_status_timeout_s"),
    [
        ({}, 10.0),  # config default
        ({"status_poll_timeout_s": 5.0}, 5.0),
        ({"status_poll_timeout_s": None}, 120.0),  # falls back to the shared request budget
    ],
)
@pytest.mark.asyncio
async def test_exec_background_status_polls_use_dedicated_timeout(
    monkeypatch: pytest.MonkeyPatch, operations_overrides: dict[str, Any], expected_status_timeout_s: float
) -> None:
    """Status polls get their own short budget; the submit and logs calls keep theirs.

    Status polls are sub-second GETs, so inheriting the shared request budget
    (tuned for long submits) lets each poll against an unreachable sandbox hang
    for minutes. status_poll_timeout_s: None restores the shared budget.
    """

    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeCommands:
        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> Any:
            return SimpleNamespace(id="exec-budget")

        async def get_command_status(self, execution_id: str) -> Any:
            return SimpleNamespace(running=False, exit_code=0, error=None)

        async def get_background_command_logs(self, execution_id: str) -> Any:
            return SimpleNamespace(content="ok", cursor=None)

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    recorded: dict[str, float | None] = {}
    real_await_sdk_call = opensandbox_provider.OpenSandboxProvider._await_sdk_call

    async def recording_await_sdk_call(
        self: Any, awaitable: Any, *, operation: str, sandbox_id: str, timeout_s: float | None
    ) -> Any:
        recorded[operation] = timeout_s
        return await real_await_sdk_call(
            self, awaitable, operation=operation, sandbox_id=sandbox_id, timeout_s=timeout_s
        )

    monkeypatch.setattr(opensandbox_provider.OpenSandboxProvider, "_await_sdk_call", recording_await_sdk_call)

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 120},
        probe={"command": None},
        operations={"background_exec": True, "background_poll_interval_s": 0.01, **operations_overrides},
    )
    handle = opensandbox_provider.SandboxHandle(
        sandbox_id="sandbox-budget", provider_name="opensandbox", raw=SimpleNamespace(commands=FakeCommands())
    )

    result = await provider.exec(handle, "echo ok", timeout_s=30)

    assert result.return_code == 0
    assert 0 < recorded["command status"] <= expected_status_timeout_s
    assert recorded["command status"] <= 90  # whole command budget also bounds this read
    # Everything else keeps its existing budget: submit gets timeout_s + 60
    # headroom, logs the shared request budget (payloads can be large).
    assert recorded["command run (background submit)"] == 90.0
    assert recorded["command logs"] == 120.0


@pytest.mark.asyncio
@pytest.mark.parametrize("read_operation", ["status", "logs"])
@pytest.mark.parametrize("failures", [1, 3])
async def test_exec_background_retries_timed_out_reads(
    monkeypatch: pytest.MonkeyPatch, read_operation: str, failures: int
) -> None:
    """Status read retries preserve the submitted command; log reads use one retry batch."""
    from nemo_gym.client import http_client

    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeConnectionConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def get_base_url(self) -> str:
            return "https://sandbox.example/v1"

        def get_api_key(self) -> None:
            return None

    class FakeResponse:
        status = 404

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
            return None

    class FakeCommands:
        def __init__(self) -> None:
            self.submissions = 0
            self.status_calls: list[str] = []
            self.log_calls: list[str] = []

        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> Any:
            self.submissions += 1
            return SimpleNamespace(id="exec-slowpoll")

        async def get_command_status(self, execution_id: str) -> Any:
            self.status_calls.append(execution_id)
            if read_operation == "status" and len(self.status_calls) <= failures:
                # Surfaces through _await_sdk_call the same way an expired
                # per-call budget does (asyncio.TimeoutError is TimeoutError).
                raise TimeoutError("simulated status poll budget expiry")
            return SimpleNamespace(running=False, exit_code=0, error=None)

        async def get_background_command_logs(self, execution_id: str) -> Any:
            self.log_calls.append(execution_id)
            if read_operation == "logs" and len(self.log_calls) <= failures:
                raise TimeoutError("simulated log fetch budget expiry")
            return SimpleNamespace(content="ok", cursor=None)

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, FakeConnectionConfig, FakeRunCommandOpts, object, object),
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    request = AsyncMock(return_value=FakeResponse())
    monkeypatch.setattr(http_client, "request", request)

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 120},
        probe={"command": None},
        operations={
            "background_exec": True,
            "retries": 2,
            "retry_delay_s": 0.001,
            "background_poll_interval_s": 0.01,
        },
    )
    commands = FakeCommands()
    handle = opensandbox_provider.SandboxHandle(
        sandbox_id="sandbox-slowpoll", provider_name="opensandbox", raw=SimpleNamespace(commands=commands)
    )

    if failures == 3 and read_operation == "logs":
        with raises_failure(PROVIDER_IO_TIMEOUT, match=f"command {read_operation}"):
            await provider.exec(handle, "echo ok", timeout_s=30)
    else:
        result = await provider.exec(handle, "echo ok", timeout_s=30)
        assert result.return_code == 0
        assert result.stdout == "ok"

    assert commands.submissions == 1
    expected_attempts = failures + 1 if read_operation == "status" else min(failures + 1, 3)
    if read_operation == "status":
        assert commands.status_calls == ["exec-slowpoll"] * expected_attempts
        assert commands.log_calls == ["exec-slowpoll"]
        assert request.await_count == 0  # inspect is opt-in
    else:
        assert commands.status_calls == ["exec-slowpoll"]
        assert commands.log_calls == ["exec-slowpoll"] * expected_attempts
        assert request.await_count == 0  # inspect is opt-in


def test_tls_verify_reaches_transports(fake_opensandbox_sdk: None) -> None:
    import ssl

    ConnectionConfig = opensandbox_provider.OpenSandboxConnectionConfig
    assert ConnectionConfig().tls_verify is False
    assert ConnectionConfig(tls_verify=True).tls_verify is True

    default = opensandbox_provider.OpenSandboxProvider(connection={"transport_backend": "httpx"})
    assert default._build_transport()._pool._ssl_context.verify_mode == ssl.CERT_NONE
    verified = opensandbox_provider.OpenSandboxProvider(connection={"transport_backend": "httpx", "tls_verify": True})
    assert verified._build_transport()._pool._ssl_context.verify_mode == ssl.CERT_REQUIRED

    from nemo_gym.sandbox.providers._http_transport import GymAiohttpTransport

    bridge = opensandbox_provider.OpenSandboxProvider(connection={"transport_backend": "aiohttp"})._build_transport()
    assert isinstance(bridge, GymAiohttpTransport)
    assert bridge.verify is False
    bridge_verified = opensandbox_provider.OpenSandboxProvider(
        connection={"transport_backend": "aiohttp", "tls_verify": True}
    )._build_transport()
    assert bridge_verified.verify is True


@pytest.mark.asyncio
async def test_pty_http_client_follows_tls_verify(fake_opensandbox_sdk: None) -> None:
    default = opensandbox_provider.OpenSandboxProvider()._pty_http_client()
    try:
        assert default.connector._ssl is False
    finally:
        await default.close()

    verified = opensandbox_provider.OpenSandboxProvider(connection={"tls_verify": True})._pty_http_client()
    try:
        assert verified.connector._ssl is True
    finally:
        await verified.close()


def test_split_domain_scheme() -> None:
    split = opensandbox_provider.split_domain_scheme
    assert split("sandbox.example") == ("sandbox.example", None)
    assert split(" sandbox.example\n") == ("sandbox.example", None)
    assert split("https://sandbox.example/") == ("sandbox.example", "https")
    assert split("HTTP://sandbox.example:8080/prefix/") == ("sandbox.example:8080/prefix", "http")
    for bad, message in (
        ("ftp://sandbox.example", "unsupported scheme"),
        ("https://", "no host"),
        ("https://sandbox.example/prefix?token=abc", "query string or fragment"),
        ("https://sandbox.example#staging", "query string or fragment"),
    ):
        with pytest.raises(ValueError, match=message):
            split(bad)


def test_connection_config_derives_protocol_from_domain_scheme() -> None:
    ConnectionConfig = opensandbox_provider.OpenSandboxConnectionConfig

    https = ConnectionConfig(domain="https://sandbox.example/")
    assert (https.domain, https.protocol) == ("sandbox.example", "https")

    # The shipped yaml always sets protocol (http), so a scheme in the domain must win over it.
    override = ConnectionConfig(domain="HTTP://sandbox.example:8080/prefix/", protocol="https")
    assert (override.domain, override.protocol) == ("sandbox.example:8080/prefix", "http")

    bare = ConnectionConfig(domain="sandbox.example", protocol="https")
    assert (bare.domain, bare.protocol) == ("sandbox.example", "https")
    assert ConnectionConfig().domain is None

    # Whitespace is normalized for every domain, not only scheme-carrying ones.
    padded = ConnectionConfig(domain=" sandbox.example\n", protocol="http")
    assert (padded.domain, padded.protocol) == ("sandbox.example", "http")

    with pytest.raises(ValueError, match="unsupported scheme"):
        ConnectionConfig(domain="ftp://sandbox.example")
    with pytest.raises(ValueError, match="no host"):
        ConnectionConfig(domain="https://")
    with pytest.raises(ValueError, match="query string or fragment"):
        ConnectionConfig(domain="https://sandbox.example/prefix?token=abc")
    with pytest.raises(ValueError, match="query string or fragment"):
        ConnectionConfig(domain="https://sandbox.example#staging")


def test_scheme_in_domain_reaches_sdk_and_pty_protocol(fake_opensandbox_sdk: None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={
            "domain": "https://sandbox.example/",
            "api_key": "key",  # pragma: allowlist secret
            "protocol": "http",
        }
    )
    kwargs = provider._connection_config().kwargs
    assert kwargs["domain"] == "sandbox.example"
    assert kwargs["protocol"] == "https"
    # The PTY WebSocket URL is built from the provider's own protocol, not the SDK's base URL.
    assert provider._connection.protocol == "https"


@pytest.mark.parametrize("size", ["true", "8Gi", "67108864"])
async def test_shared_memory_metadata_reaches_create_api(fake_opensandbox_sdk, size):
    from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider

    provider = OpenSandboxProvider(attribution={"enabled": False}, probe={"command": None})
    await provider.create(SandboxSpec(image="image:tag", metadata={"nemo.nvidia.com/shm": size}))
    assert FakeSandbox.created_kwargs["metadata"]["nemo.nvidia.com/shm"] == size


def test_images_config_rewrites_prefixes_and_picks_the_longest_auth_prefix() -> None:
    images = opensandbox_provider.OpenSandboxImagesConfig(
        rewrites=[{"from": "harborframework/", "to": "registry.example/mirror/"}],
        auth={
            "registry.example/": {"username": "outer", "password": TEST_REGISTRY_PASSWORD},
            "registry.example/mirror/": {"username": "inner", "password": TEST_REGISTRY_PASSWORD},
        },
    )
    assert images.resolve(None) == (None, None)
    assert images.resolve("python:3.11") == ("python:3.11", None)
    image, auth = images.resolve("harborframework/tb:abc@sha256:0")
    assert image == "registry.example/mirror/tb:abc@sha256:0" and auth["username"] == "inner"
    image, auth = images.resolve("registry.example/other:1")
    assert image == "registry.example/other:1" and auth["username"] == "outer"


async def test_direct_create_applies_the_provider_image_policy(fake_opensandbox_sdk: None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        probe={"command": None},
        images={
            "rewrites": [{"from": "harborframework/", "to": "registry.example/mirror/"}],
            "auth": {"registry.example/": {"username": "mirror", "password": TEST_REGISTRY_PASSWORD}},
        },
    )
    await provider.create(SandboxSpec(image="harborframework/tb:abc"))
    image = FakeSandbox.created_kwargs["image"]
    assert image.image == "registry.example/mirror/tb:abc" and image.auth.username == "mirror"

    # A spec's own credentials win over the provider's, and an unrewritten image with no matching prefix is plain.
    await provider.create(
        SandboxSpec(
            image="harborframework/tb:abc", provider_options={"image_auth": {"username": "spec", "password": "x"}}
        )
    )
    assert FakeSandbox.created_kwargs["image"].auth.username == "spec"
    await provider.create(SandboxSpec(image="python:3.11"))
    assert FakeSandbox.created_kwargs["image"] == "python:3.11"


@pytest.mark.parametrize("return_code,error_type,exit_code", [(1, None, 1), (-1, "timeout", None)])
async def test_direct_create_runs_setup_commands_as_root_and_a_failure_fails_the_create(
    fake_opensandbox_sdk: None,
    monkeypatch: pytest.MonkeyPatch,
    return_code: int,
    error_type: str | None,
    exit_code: int | None,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        probe={"command": None},
        setup={
            "commands": ["apt-get update", "false"],
            "timeout_s": 12,
            "env": {"EXECD_API_GRACE_SHUTDOWN": "50ms", "A": "provider"},
        },
    )
    calls: list[tuple[str, str | None, float | None]] = []
    cleaned: list[str] = []

    async def exec_(handle, command, **kwargs):
        calls.append((command, kwargs.get("user"), kwargs.get("timeout_s")))
        return opensandbox_provider.SandboxExecResult(
            stdout="",
            stderr="boom" if command == "false" else "",
            return_code=return_code if command == "false" else 0,
            error_type=error_type if command == "false" else None,
        )

    async def cleanup(handle):
        cleaned.append(handle.sandbox_id)

    monkeypatch.setattr(provider, "exec", exec_)
    monkeypatch.setattr(provider, "_cleanup_failed_create_handle", cleanup)
    with pytest.raises(NgFailureEvent, match="setup command failed.*false") as failure_info:
        await provider.create(SandboxSpec(image="python:3.11", env={"A": "spec"}))
    assert failure_info.value.failure_kind == "provider_setup_failed"
    assert failure_info.value.details.exit_code is exit_code
    assert failure_info.value.details.provider == "opensandbox"
    assert calls == [("apt-get update", "root", 12), ("false", "root", 12)]
    assert cleaned == ["sandbox-1"]
    # The provider's environment sits under the spec's own variables.
    assert FakeSandbox.created_kwargs["env"] == {"EXECD_API_GRACE_SHUTDOWN": "50ms", "A": "spec"}


async def test_direct_create_does_not_retry_a_failed_setup_command(
    fake_opensandbox_sdk: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        probe={"command": None},
        create={"retries": 3, "retry_delay_s": 0, "retry_max_delay_s": 0},
        setup={"commands": ["false"], "timeout_s": 12},
    )
    calls: list[str] = []
    cleaned: list[str] = []

    async def exec_(handle, command, **kwargs):
        calls.append(command)
        return opensandbox_provider.SandboxExecResult(stdout="", stderr="gateway timeout", return_code=1)

    async def cleanup(handle):
        cleaned.append(handle.sandbox_id)

    monkeypatch.setattr(provider, "exec", exec_)
    monkeypatch.setattr(provider, "_cleanup_failed_create_handle", cleanup)
    # Setup remains a sandbox observation and is not retried by default.
    # Legacy callers must migrate their exception checks (anyswe_agent/app.py,
    # cvdp/testbench_runner.py) classify it as a sandbox failure, while the transient-looking stderr never re-runs
    # the create.
    with pytest.raises(NgFailureEvent) as info:
        await provider.create(SandboxSpec(image="python:3.11"))
    assert info.value.failure_kind == "provider_setup_failed"
    assert calls == ["false"] and cleaned == ["sandbox-1"]
    assert info.value.failure_kind == "provider_setup_failed"
    assert info.value.boundary.value == "sandbox"


# The OpenSandbox runtime's refusal when the exec daemon runs as the image's non-root USER on a cell that drops
# CAP_SETUID (alibaba-ctf softHash, M35 D8: exit 255, the command never ran).
NO_ROOT_REFUSAL = (
    "fork/exec /usr/bin/bash: operation not permitted (switching to uid=0 gid=0 requires CAP_SETUID/CAP_SETGID, "
    "which this sandbox may not have — check the server's docker.drop_capabilities configuration; dropping these "
    "capabilities makes every identity switch fail)"
)


@pytest.mark.parametrize("caller", ["setup", "transfer", "background"])
@pytest.mark.parametrize(
    "refusal",
    [
        NO_ROOT_REFUSAL,
        "fork/exec /usr/bin/bash: operation not permitted",
        "fork/exec /usr/bin/bash: operation not permitted (switching to uid=0 requires CAP_SETUID)",
    ],
)
async def test_root_refusal_chooses_default_user_without_a_failure_event(caller: str, refusal: str) -> None:
    from opensandbox.models.execd import Execution, ExecutionError

    from nemo_gym.sandbox.transfer import exec_as_root

    commands = SimpleNamespace(
        run=AsyncMock(
            side_effect=[
                Execution(error=ExecutionError(name="CommandExecError", value=refusal, timestamp=0)),
                Execution(exit_code=0),
            ]
        )
    )
    raw = SimpleNamespace(commands=commands, get_info=AsyncMock())
    handle = opensandbox_provider.SandboxHandle("root-refusal", "opensandbox", raw)
    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"background_exec": caller == "background"}, probe={"command": None}
    )

    class Sandbox:
        async def exec(self, command, **kwargs):
            return await provider.exec(handle, command, **kwargs)

    with RequestFailureLog.capture() as log:
        if caller == "setup":
            result = await provider._exec_setup_command(handle, "true")
        elif caller == "transfer":
            result = await exec_as_root(Sandbox(), "true", timeout_s=1)
        else:
            result = await provider.exec(handle, "true", user="root", timeout_s=1)
        assert not log.reply_report(None).events

    assert result.error_type == ("user_switch_unavailable" if caller == "background" else None)
    assert result.return_code == (125 if caller == "background" else 0)
    assert [call.kwargs["opts"].uid for call in commands.run.await_args_list] == (
        [0] if caller == "background" else [0, None]
    )
    raw.get_info.assert_not_awaited()


@pytest.mark.parametrize("caller", ["setup", "transfer"])
@pytest.mark.parametrize("refusal_at", ["submit", "status"])
async def test_background_root_fallback_uses_runtime_launch_evidence(caller: str, refusal_at: str) -> None:
    from opensandbox.models.execd import Execution, ExecutionError

    from nemo_gym.sandbox.transfer import exec_as_root

    refused = Execution(id="refused")
    if refusal_at == "submit":
        refused.error = ExecutionError(
            name="RuntimeError",
            value="error running commands failed to start commands: " + NO_ROOT_REFUSAL,
            timestamp=0,
        )

    async def status(execution_id):
        return SimpleNamespace(
            running=False,
            exit_code=255 if execution_id == "refused" else 0,
            error=NO_ROOT_REFUSAL if execution_id == "refused" else None,
        )

    commands = SimpleNamespace(
        run=AsyncMock(side_effect=[refused, Execution(id="ok")]),
        get_command_status=AsyncMock(side_effect=status),
        get_background_command_logs=AsyncMock(return_value=SimpleNamespace(content="ready")),
    )
    handle = opensandbox_provider.SandboxHandle("root-refusal", "opensandbox", SimpleNamespace(commands=commands))
    provider = opensandbox_provider.OpenSandboxProvider(operations={"background_exec": True}, probe={"command": None})

    class Sandbox:
        async def exec(self, command, **kwargs):
            return await provider.exec(handle, command, **kwargs)

    with RequestFailureLog.capture() as log:
        result = (
            await provider._exec_setup_command(handle, "true")
            if caller == "setup"
            else await exec_as_root(Sandbox(), "true", timeout_s=1)
        )
        assert not log.reply_report(None).events
    assert result.return_code == 0 and result.error_type is None
    assert [call.kwargs["opts"].uid for call in commands.run.await_args_list] == [0, None]
    commands.get_background_command_logs.assert_awaited_once_with("ok")


async def test_background_exit_255_with_refusal_text_in_output_does_not_retry() -> None:
    from opensandbox.models.execd import Execution

    from nemo_gym.sandbox.transfer import exec_as_root

    commands = SimpleNamespace(
        run=AsyncMock(return_value=Execution(id="ran")),
        get_command_status=AsyncMock(
            return_value=SimpleNamespace(running=False, exit_code=255, error="exit status 255")
        ),
        get_background_command_logs=AsyncMock(return_value=SimpleNamespace(content=NO_ROOT_REFUSAL)),
    )
    handle = opensandbox_provider.SandboxHandle("already-ran", "opensandbox", SimpleNamespace(commands=commands))
    provider = opensandbox_provider.OpenSandboxProvider(operations={"background_exec": True}, probe={"command": None})

    class Sandbox:
        async def exec(self, command, **kwargs):
            return await provider.exec(handle, command, **kwargs)

    result = await exec_as_root(Sandbox(), "echo side-effect", timeout_s=1)
    assert result.return_code == 255 and result.error_type is None
    commands.run.assert_awaited_once()


@pytest.mark.parametrize(
    "shape", ["stdout", "other-error", "process-exited", "timeout", "exception-text", "other-uid"]
)
async def test_root_fallback_does_not_replay_an_ambiguous_command(shape: str) -> None:
    from opensandbox.models.execd import Execution, ExecutionError, ExecutionLogs, OutputMessage

    from nemo_gym.sandbox.transfer import exec_as_root

    if shape == "stdout":
        execution = Execution(
            exit_code=1, logs=ExecutionLogs(stderr=[OutputMessage(text=NO_ROOT_REFUSAL, timestamp=0)])
        )
    else:
        execution = Execution(
            error=ExecutionError(
                name="OtherError" if shape == "other-error" else "CommandExecError",
                value=NO_ROOT_REFUSAL,
                timestamp=0,
            ),
            exit_code=1 if shape == "process-exited" else None,
        )
    error = TimeoutError(NO_ROOT_REFUSAL) if shape == "timeout" else RuntimeError(NO_ROOT_REFUSAL)
    commands = SimpleNamespace(
        run=AsyncMock(side_effect=error if shape in ("timeout", "exception-text") else None, return_value=execution)
    )
    handle = opensandbox_provider.SandboxHandle("uncertain", "opensandbox", SimpleNamespace(commands=commands))
    provider = opensandbox_provider.OpenSandboxProvider(diagnostics={"timeout_s": 0.001}, probe={"command": None})

    class Sandbox:
        async def exec(self, command, **kwargs):
            return await provider.exec(handle, command, **kwargs)

    if shape in ("stdout", "process-exited"):
        result = await exec_as_root(Sandbox(), "echo side-effect", timeout_s=1)
        assert result.return_code == 1
    else:
        with pytest.raises(NgFailureEvent):
            if shape == "other-uid":
                await provider.exec(handle, "echo side-effect", user=5, timeout_s=1)
            else:
                await exec_as_root(Sandbox(), "echo side-effect", timeout_s=1)
    commands.run.assert_awaited_once()


async def test_setup_commands_run_as_the_images_user_in_a_sandbox_that_cannot_switch_to_root(
    fake_opensandbox_sdk: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A Compose sidecar whose image runs as a non-root ``USER`` (alibaba-ctf's ``softHash`` runtime image, ``User:
    ctf``) cannot switch identity on a cell that drops ``CAP_SETUID``: the runtime refuses the exec before the command
    runs, so no script content can no-op. ``transfer.exec_as_root`` already falls back to the default user on this
    refusal; ``setup.commands`` now degrade the same way, so a command written to no-op where it does not apply gets
    to, and the create succeeds."""
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None}, setup={"commands": ["true"]})
    calls: list[tuple[str, str | None]] = []

    async def exec_(handle, command, **kwargs):
        calls.append((command, kwargs.get("user")))
        if kwargs.get("user") == "root":
            return opensandbox_provider.SandboxExecResult(
                stdout=None, stderr=NO_ROOT_REFUSAL, return_code=125, error_type="user_switch_unavailable"
            )
        return opensandbox_provider.SandboxExecResult(stdout=None, stderr=None, return_code=0)

    monkeypatch.setattr(provider, "exec", exec_)
    with caplog.at_level(logging.DEBUG, logger="nemo_gym.sandbox.providers.opensandbox.provider"):
        handle = await provider.create(SandboxSpec(image="quay.io/alibaba-aaig/alibaba-ctf-bench:runtime.softHash"))
    assert handle.sandbox_id == "sandbox-1"
    assert calls == [("true", "root"), ("true", None)]
    assert any("cannot switch to root" in record.getMessage() for record in caplog.records)


async def test_a_setup_command_that_fails_as_the_images_user_still_fails_the_create(
    fake_opensandbox_sdk: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None}, setup={"commands": ["sed -i x"]})
    cleaned: list[str] = []

    async def exec_(handle, command, **kwargs):
        if kwargs.get("user") == "root":
            return opensandbox_provider.SandboxExecResult(
                stdout=None, stderr=NO_ROOT_REFUSAL, return_code=125, error_type="user_switch_unavailable"
            )
        return opensandbox_provider.SandboxExecResult(stdout=None, stderr="sed: Permission denied", return_code=4)

    async def cleanup(handle):
        cleaned.append(handle.sandbox_id)

    monkeypatch.setattr(provider, "exec", exec_)
    monkeypatch.setattr(provider, "_cleanup_failed_create_handle", cleanup)
    with pytest.raises(NgFailureEvent, match=r"exit 4.*sed -i x\nsed: Permission denied") as failure_info:
        await provider.create(SandboxSpec(image="python:3.11"))
    assert failure_info.value.failure_kind == "provider_setup_failed"
    assert failure_info.value.details.exit_code == 4
    assert cleaned == ["sandbox-1"]


async def test_a_cancelled_create_still_cleans_up_the_new_sandbox(
    fake_opensandbox_sdk: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None}, setup={"commands": ["sleep 600"]})
    cleaned: list[str] = []
    started = asyncio.Event()

    async def exec_(handle, command, **kwargs):
        started.set()
        await asyncio.sleep(3600)

    async def cleanup(handle):
        # The cleanup itself is awaited to completion, not abandoned with the cancelled task.
        await asyncio.sleep(0)
        cleaned.append(handle.sandbox_id)

    monkeypatch.setattr(provider, "exec", exec_)
    monkeypatch.setattr(provider, "_cleanup_failed_create_handle", cleanup)
    task = asyncio.create_task(provider.create(SandboxSpec(image="python:3.11")))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned == ["sandbox-1"]


# -- requests above limits, create budget, orphaned creates ---------------------------------


async def test_direct_create_clamps_requests_above_the_sandbox_limits(
    fake_opensandbox_sdk: None, caplog: pytest.LogCaptureFixture, monkeypatch
) -> None:
    from nemo_gym.telemetry import _fallbacks

    submitted = {}
    monkeypatch.setattr(opensandbox_provider, "is_span_group_enabled", lambda group: True)
    monkeypatch.setattr(_fallbacks, "safe_set_span_attributes", lambda span, values: submitted.update(values))
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    with caplog.at_level(logging.WARNING, logger=opensandbox_provider.LOGGER.name):
        await provider.create(
            SandboxSpec(
                image="image:tag",
                resources={"cpu": 1, "memory_mib": 2048, "disk_gib": 4},
                provider_options={"resource_requests": {"cpu": 0.5, "memory_mib": 512, "disk_gib": 10}},
            ),
        )
    assert FakeSandbox.created_kwargs["resource"] == {"cpu": "1", "memory": "2048Mi", "ephemeral-storage": "4Gi"}
    assert FakeSandbox.created_kwargs["resource_requests"] == {
        "cpu": "0.5",
        "memory": "512Mi",
        "ephemeral-storage": "4Gi",
    }
    assert submitted == {
        "nemo.gym.sandbox.submitted.requests.cpu": 0.5,
        "nemo.gym.sandbox.submitted.requests.memory_mib": 512,
        "nemo.gym.sandbox.submitted.requests.disk_gib": 4,
        "nemo.gym.sandbox.submitted.limits.cpu": 1,
        "nemo.gym.sandbox.submitted.limits.memory_mib": 2048,
        "nemo.gym.sandbox.submitted.limits.disk_gib": 4,
    }
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "disk_gib=10" in warnings[0] and "limit 4" in warnings[0]

    # A request a task declares no limit for is passed through, silently.
    caplog.clear()
    await provider.create(
        SandboxSpec(image="image:tag", resources={"cpu": 1}, provider_options={"resource_requests": {"disk_gib": 10}})
    )
    assert FakeSandbox.created_kwargs["resource_requests"] == {"ephemeral-storage": "10Gi"}
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    # "limits" mirrors the limits and needs no clamp; no requests sends none.
    await provider.create(
        SandboxSpec(image="image:tag", resources={"disk_gib": 4}, provider_options={"resource_requests": "limits"})
    )
    assert FakeSandbox.created_kwargs["resource_requests"] == {"ephemeral-storage": "4Gi"}
    await provider.create(SandboxSpec(image="image:tag", resources={"disk_gib": 4}))
    assert "resource_requests" not in FakeSandbox.created_kwargs
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


async def test_requests_are_clamped_against_the_sdk_default_limits_when_the_spec_declares_none(
    fake_opensandbox_sdk: None, caplog: pytest.LogCaptureFixture
) -> None:
    """The SDK sends cpu 1 / memory 2Gi as limits for a spec without resources (Harbor's default and every
    Compose sidecar), so a deployment's requests are held to those; otherwise the pod never schedules."""
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    with caplog.at_level(logging.WARNING, logger=opensandbox_provider.LOGGER.name):
        await provider.create(
            SandboxSpec(
                image="image:tag", resources={}, provider_options={"resource_requests": {"cpu": 2, "memory_mib": 4096}}
            )
        )
    assert FakeSandbox.created_kwargs["resource"] == {}
    assert FakeSandbox.created_kwargs["resource_requests"] == {"cpu": "1", "memory": "2048Mi"}
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert "cpu=2 exceeds the sandbox's limit 1.0 (the SDK's default" in warnings[0]
    assert "memory_mib=4096 exceeds the sandbox's limit 2048 (the SDK's default" in warnings[1]
    # Requests within the defaults, and keys the defaults do not cover, pass through silently.
    caplog.clear()
    await provider.create(
        SandboxSpec(
            image="image:tag", resources={}, provider_options={"resource_requests": {"cpu": 0.5, "disk_gib": 10}}
        )
    )
    assert FakeSandbox.created_kwargs["resource_requests"] == {"cpu": "0.5", "ephemeral-storage": "10Gi"}
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


async def test_create_attempt_key_is_unique_per_create(fake_opensandbox_sdk: None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    key = f"{provider._attribution.key_prefix}{opensandbox_provider.CREATE_ATTEMPT_KEY}"
    values = []
    for _ in range(2):
        await provider._create_once(SandboxSpec(image="image:tag", metadata={"harbor_task": "t"}))
        values.append(FakeSandbox.created_kwargs["metadata"][key])
    first, second = values
    assert first != second and len(first) == 32
    assert FakeSandbox.created_kwargs["metadata"]["harbor_task"] == "t"
    assert key == "nemo-gym.nvidia.com/create-attempt"


class HangingCreateSandbox(FakeSandbox):
    @classmethod
    async def create(cls, **kwargs: Any) -> "FakeSandbox":
        cls.created_kwargs = kwargs
        await asyncio.get_running_loop().create_future()
        return cls()


@dataclass
class FakeLifecycleService:
    """The SDK's management API without a sandbox handle: list by metadata filter, delete by id."""

    sandboxes: list[SimpleNamespace]
    killed: list[str] = field(default_factory=list)
    filters: list[Any] = field(default_factory=list)
    kill_error: Exception | None = None

    async def list_sandboxes(self, filter: Any) -> SimpleNamespace:
        self.filters.append(filter)
        matches = [
            s
            for s in self.sandboxes
            if all((s.metadata or {}).get(k) == v for k, v in (filter.metadata or {}).items())
        ]
        return SimpleNamespace(sandbox_infos=matches, pagination=SimpleNamespace(has_next_page=False))

    async def kill_sandbox(self, sandbox_id: str) -> None:
        self.killed.append(sandbox_id)
        if self.kill_error is not None:
            raise self.kill_error


def install_lifecycle(monkeypatch: pytest.MonkeyPatch, service: FakeLifecycleService) -> None:
    @dataclass
    class FakeFilter:
        metadata: dict[str, str] | None = None
        page: int | None = None
        page_size: int | None = None

    class FakeFactory:
        def __init__(self, connection_config: Any) -> None:
            self.connection_config = connection_config

        def create_sandbox_service(self) -> FakeLifecycleService:
            return service

    monkeypatch.setattr(opensandbox_provider, "_require_opensandbox_lifecycle_sdk", lambda: (FakeFactory, FakeFilter))
    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (HangingCreateSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )


def orphan(sandbox_id: str, metadata: dict[str, str], state: str = "PENDING", reason: str | None = None):
    return SimpleNamespace(
        id=sandbox_id,
        metadata=metadata,
        status=SimpleNamespace(state=state, reason=reason, message="BatchSandbox is pending allocation"),
    )


async def test_ready_timeout_caps_the_request_but_a_configured_attempt_budget_stands(
    fake_opensandbox_sdk: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """(a) The spec's ready timeout caps a configured SDK request timeout; create.timeout_s is respected as is."""
    install_lifecycle(monkeypatch, FakeLifecycleService(sandboxes=[]))
    monkeypatch.setattr(opensandbox_provider, "_BUDGET_WARNINGS", set())
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"timeout_s": 0.05, "request_timeout_s": 1200}, probe={"command": None}
    )
    started = asyncio.get_running_loop().time()
    with caplog.at_level(logging.WARNING, logger=opensandbox_provider.LOGGER.name):
        with pytest.raises(NgFailureEvent, match="after 0.05s") as info:
            await provider._create_once(SandboxSpec(image="image:tag", ready_timeout_s=0.03))
        assert info.value.failure_kind == "provider_create_timeout"
    assert not info.value.failure_kind == "provider_create_pending"
    assert asyncio.get_running_loop().time() - started < 5
    assert HangingCreateSandbox.created_kwargs["connection_config"].kwargs["request_timeout"] == timedelta(
        seconds=0.03
    )
    assert HangingCreateSandbox.created_kwargs["ready_timeout"] == timedelta(seconds=0.03)
    # The configured budget cannot hold the capped request (0.03) plus the readiness wait (0.03): said once.
    warnings = [r.getMessage() for r in caplog.records if "create.timeout_s=0.05 is below" in r.getMessage()]
    assert len(warnings) == 1 and "0.03s plus the spec's ready_timeout_s of 0.03s" in warnings[0]
    caplog.clear()
    with pytest.raises(NgFailureEvent) as failure_info:
        await provider._create_once(SandboxSpec(image="image:tag", ready_timeout_s=0.03))
    assert failure_info.value.failure_kind == "provider_create_timeout"
    # Nor by the next provider built for the same configuration: servers build one per sandbox.
    with pytest.raises(NgFailureEvent) as failure_info:
        await opensandbox_provider.OpenSandboxProvider(
            create={"timeout_s": 0.05, "request_timeout_s": 1200}, probe={"command": None}
        )._create_once(SandboxSpec(image="image:tag", ready_timeout_s=0.03))
    assert failure_info.value.failure_kind == "provider_create_timeout"
    assert not [r for r in caplog.records if "create.timeout_s=0.05 is below" in r.getMessage()]

    # Without a ready timeout the create budget is the configured one, unchanged.
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"timeout_s": 0.01, "request_timeout_s": 1200}, probe={"command": None}
    )
    with pytest.raises(NgFailureEvent, match="after 0.01s") as failure_info:
        await provider._create_once(SandboxSpec(image="image:tag"))
    assert failure_info.value.failure_kind == "provider_create_timeout"
    assert HangingCreateSandbox.created_kwargs["connection_config"].kwargs["request_timeout"] == timedelta(
        seconds=1200
    )


async def test_create_budget_holds_the_request_plus_the_readiness_wait(fake_opensandbox_sdk: None) -> None:
    def budget(**create: Any) -> Any:
        connection = {"request_timeout_s": create.pop("connection_request_timeout_s", None)}
        return opensandbox_provider.OpenSandboxProvider(create=create, connection=connection, probe={"command": None})

    # Nothing configured: the SDK's default request timeout (30 s) plus the ready timeout plus 30 s slack.
    assert budget()._create_budget(SandboxSpec(image="i", ready_timeout_s=1200)) == (1260.0, None)
    # A configured request timeout is capped at the ready timeout and counted in the default budget.
    assert budget(request_timeout_s=1200)._create_budget(SandboxSpec(image="i", ready_timeout_s=300)) == (630.0, 300)
    assert budget(connection_request_timeout_s=300)._create_budget(SandboxSpec(image="i", ready_timeout_s=1200)) == (
        1530.0,
        300,
    )
    # A configured create.timeout_s stands, above or below the sum (the shipped 1500 over 1200 plus 1200).
    assert budget(timeout_s=3000, request_timeout_s=1200)._create_budget(
        SandboxSpec(image="i", ready_timeout_s=1200)
    ) == (3000, 1200)
    assert budget(timeout_s=1500, request_timeout_s=1200)._create_budget(
        SandboxSpec(image="i", ready_timeout_s=1200)
    ) == (1500, 1200)
    # Without a ready timeout nothing is capped: the configured budget, else the connection's request timeout.
    assert budget(timeout_s=90, request_timeout_s=1200)._create_budget(SandboxSpec(image="i")) == (90, 1200)
    assert budget(connection_request_timeout_s=300)._create_budget(SandboxSpec(image="i")) == (300.0, 300)
    assert budget()._create_budget(SandboxSpec(image="i")) == (None, None)


async def test_unconfigured_request_timeout_is_not_raised_to_the_ready_timeout(fake_opensandbox_sdk: None) -> None:
    """The create's connection config lives on the handle for the sandbox's life: a deployment that configured
    no request timeout keeps the SDK's default (30 s) on status, endpoint and file calls, not a 20-minute one."""
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    await provider.create(SandboxSpec(image="image:tag", ready_timeout_s=1200))
    assert "request_timeout" not in FakeSandbox.created_kwargs["connection_config"].kwargs
    assert FakeSandbox.created_kwargs["ready_timeout"] == timedelta(seconds=1200)
    # The connection's request timeout is capped for the create, as a create-level one is.
    provider = opensandbox_provider.OpenSandboxProvider(connection={"request_timeout_s": 300}, probe={"command": None})
    await provider.create(SandboxSpec(image="image:tag", ready_timeout_s=120))
    assert FakeSandbox.created_kwargs["connection_config"].kwargs["request_timeout"] == timedelta(seconds=120)


@pytest.mark.parametrize("fail_fast", [False, True])
async def test_timed_out_create_deletes_its_orphan_and_a_pending_one_is_retried_unless_fail_fast(
    fake_opensandbox_sdk: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, fail_fast: bool
) -> None:
    """(b) and (c): the orphan is found by run plus attempt key and deleted; still Pending is retried like any
    other timeout (Pending also covers a full cluster and a slow pull) unless the deployment opted out."""
    prefix = opensandbox_provider.DEFAULT_ATTRIBUTION_KEY_PREFIX
    run_key, attempt_key = f"{prefix}run", f"{prefix}{opensandbox_provider.CREATE_ATTEMPT_KEY}"
    service = FakeLifecycleService(sandboxes=[])
    install_lifecycle(monkeypatch, service)
    provider = opensandbox_provider.OpenSandboxProvider(
        create={
            "retries": 2,
            "retry_delay_s": 0,
            "retry_max_delay_s": 0,
            "timeout_s": 0.05,
            "fail_fast_on_pending": fail_fast,
        },
        probe={"command": None},
    )
    attempts = 0
    real_create_once = provider._create_once

    async def create_once(spec: SandboxSpec) -> Any:
        nonlocal attempts
        attempts += 1
        return await real_create_once(spec)

    monkeypatch.setattr(provider, "_create_once", create_once)

    def seed_orphans(metadata: dict[str, str], attempt_key: str) -> Any:
        # What the server holds once the request has gone out: this attempt's sandbox, pending; another
        # attempt's and an unrelated sandbox that the server-side filter must not make us delete.
        service.sandboxes = [
            orphan("this-attempt", dict(metadata), reason="BATCHSANDBOX_PENDING"),
            orphan("other-attempt", {**metadata, attempt_key: "other"}),
            orphan("unrelated", {}),
        ]
        return real_reap(metadata, attempt_key)

    real_reap = provider._reap_orphaned_create
    monkeypatch.setattr(provider, "_reap_orphaned_create", seed_orphans)

    with caplog.at_level(logging.WARNING, logger=opensandbox_provider.LOGGER.name):
        with pytest.raises(NgFailureEvent) as info:
            await provider.create(SandboxSpec(image="image:tag", ready_timeout_s=0.05))
        assert info.value.failure_kind == "provider_create_pending"
    message = str(info.value)
    assert "this-attempt" in message and "PENDING" in message and "BATCHSANDBOX_PENDING" in message
    assert "after 0.05s" in message and "Pending alone does not distinguish" in message
    assert "retry" not in message.lower()
    assert attempts == (1 if fail_fast else 3)  # the default retries a Pending orphan like any other timeout
    assert service.killed == ["this-attempt"] * attempts  # and deletes what each attempt left behind
    assert len(service.filters) == attempts
    assert set(service.filters[0].metadata) == {run_key, attempt_key}
    assert service.filters[0].page == 1 and service.filters[0].page_size == 100
    assert any("Deleted OpenSandbox sandbox this-attempt" in r.getMessage() for r in caplog.records)
    assert opensandbox_provider._is_retryable_create_error(info.value, fail_fast_on_pending=fail_fast) is (
        not fail_fast
    )
    assert opensandbox_provider._is_retryable_create_error(
        sandbox_failure(operation="create", failure_kind="provider_create_timeout", reason="x")
    )


async def test_sdk_request_timeout_reaps_the_orphan_like_the_attempt_budget_does(
    fake_opensandbox_sdk: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SDK's own request timeout (its 30 s default, or a configured one below the attempt budget) leaves
    the same unknown sandbox behind: it is reaped and the failure classified the same way."""
    from opensandbox.exceptions import SandboxApiException, SandboxInternalException, SandboxReadyTimeoutException

    service = FakeLifecycleService(sandboxes=[])
    install_lifecycle(monkeypatch, service)
    failures: list[Exception] = []

    class SdkTimeoutSandbox(FakeSandbox):
        @classmethod
        async def create(cls, **kwargs: Any) -> "FakeSandbox":
            cls.created_kwargs = kwargs
            raise failures.pop(0)

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (SdkTimeoutSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    real_reap = provider._reap_orphaned_create
    state = "PENDING"

    def seed(metadata: dict[str, str], attempt_key: str) -> Any:
        service.sandboxes = [orphan("left-behind", dict(metadata), state=state, reason="CREATING")]
        return real_reap(metadata, attempt_key)

    monkeypatch.setattr(provider, "_reap_orphaned_create", seed)

    # The aiohttp transport's read timeout, as the SDK wraps it (SandboxInternalException with the httpx cause).
    read_timeout = httpx.ReadTimeout("Timeout on reading data from socket")
    failures.append(SandboxInternalException("Network connectivity error: read", cause=read_timeout))
    with pytest.raises(NgFailureEvent) as info:
        await provider._create_once(SandboxSpec(image="image:tag", ready_timeout_s=1200))
    assert info.value.failure_kind == "provider_create_pending"
    assert "left-behind" in str(info.value) and "CREATING" in str(info.value)
    assert "SDK's request timeout of 30s" in str(info.value)  # nothing configured: the SDK default
    assert isinstance(info.value.__cause__, SandboxInternalException) and service.killed == ["left-behind"]
    assert opensandbox_provider._is_retryable_create_error(info.value) is True

    # A sandbox that was merely slow: the plain, retryable timeout; a configured request timeout is named.
    state = "RUNNING"
    provider = opensandbox_provider.OpenSandboxProvider(create={"request_timeout_s": 600}, probe={"command": None})
    monkeypatch.setattr(provider, "_reap_orphaned_create", seed)
    failures.append(SandboxInternalException("Network connectivity error: read", cause=httpx.ReadTimeout("read")))
    with pytest.raises(NgFailureEvent) as info:
        await provider._create_once(SandboxSpec(image="image:tag", ready_timeout_s=1200))
    assert info.value.failure_kind == "provider_create_timeout"
    assert not info.value.failure_kind == "provider_create_pending"
    assert "SDK's request timeout of 600s" in str(info.value) and service.killed == ["left-behind"] * 2

    # Other SDK failures are not request timeouts: no list, no reclassification. A connect timeout sent no
    # request; a readiness timeout is raised with the id known and the sandbox already killed by the SDK.
    listed = len(service.filters)
    for failure in (
        SandboxApiException("boom", status_code=500),
        SandboxInternalException("Network connectivity error", cause=httpx.ConnectTimeout("connect")),
        SandboxReadyTimeoutException("Sandbox health check timed out after 1200s"),
        SandboxInternalException("Unexpected SDK error occurred: x", cause=ValueError("x")),
    ):
        failures.append(failure)
        with pytest.raises(type(failure)) as raised:
            await provider._create_once(SandboxSpec(image="image:tag", ready_timeout_s=1200))
        assert raised.value is failure
    assert len(service.filters) == listed
    assert opensandbox_provider._is_request_timeout(TimeoutError()) is True
    assert opensandbox_provider._is_request_timeout(RuntimeError("not a timeout")) is False


async def test_orphan_reap_failures_never_hide_the_create_timeout(
    fake_opensandbox_sdk: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from opensandbox.exceptions import SandboxApiException

    class BrokenLifecycleService(FakeLifecycleService):
        async def list_sandboxes(self, filter: Any) -> SimpleNamespace:
            raise SandboxApiException("list failed", status_code=500)

    install_lifecycle(monkeypatch, BrokenLifecycleService(sandboxes=[]))
    provider = opensandbox_provider.OpenSandboxProvider(create={"timeout_s": 0.05}, probe={"command": None})
    with caplog.at_level(logging.WARNING, logger=opensandbox_provider.LOGGER.name):
        with pytest.raises(NgFailureEvent) as info:
            await provider._create_once(SandboxSpec(image="image:tag", ready_timeout_s=0.05))
        assert info.value.failure_kind == "provider_create_timeout"
    assert not info.value.failure_kind == "provider_create_pending"
    assert any("Could not reap the sandbox" in r.getMessage() for r in caplog.records)

    # A delete that finds the sandbox already gone is success; any other delete failure is reported, not raised.
    gone = FakeLifecycleService(sandboxes=[], kill_error=SandboxApiException("not found", status_code=404))
    install_lifecycle(monkeypatch, gone)
    real_reap = provider._reap_orphaned_create

    def seed(metadata: dict[str, str], attempt_key: str) -> Any:
        gone.sandboxes = [orphan("gone", dict(metadata))]
        return real_reap(metadata, attempt_key)

    monkeypatch.setattr(provider, "_reap_orphaned_create", seed)
    with pytest.raises(NgFailureEvent) as failure_info:
        await provider._create_once(SandboxSpec(image="image:tag", ready_timeout_s=0.05))
    assert failure_info.value.failure_kind == "provider_create_pending"
    assert gone.killed == ["gone"]


async def test_cancelled_create_request_reaps_its_orphan(
    fake_opensandbox_sdk: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller's deadline (the seed timeout) cancels the open request; the sandbox it started is deleted."""
    service = FakeLifecycleService(sandboxes=[])
    install_lifecycle(monkeypatch, service)
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    real_reap = provider._reap_orphaned_create

    def seed(metadata: dict[str, str], attempt_key: str) -> Any:
        service.sandboxes = [orphan("cancelled", dict(metadata))]
        return real_reap(metadata, attempt_key)

    monkeypatch.setattr(provider, "_reap_orphaned_create", seed)
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await provider._create_once(SandboxSpec(image="image:tag", ready_timeout_s=100))
    assert service.killed == ["cancelled"]


@pytest.mark.parametrize("operation", ["list", "kill"])
@pytest.mark.parametrize("status", [429, 503, None])
async def test_orphan_reap_retries_requests_and_retains_failures(monkeypatch, operation, status):
    from opensandbox.exceptions import SandboxApiException

    attempt_key = "create-attempt"
    scope = {attempt_key: "owned"}
    info = orphan("owned-id", scope)
    page = SimpleNamespace(
        sandbox_infos=[info, orphan("other-attempt", {attempt_key: "other"}), orphan("unrelated", {})],
        pagination=SimpleNamespace(has_next_page=False),
    )
    error = (
        TimeoutError("lost control-plane reply")
        if status is None
        else SandboxApiException("transient control-plane failure", status_code=status)
    )
    service = FakeLifecycleService(sandboxes=[])
    service.list_sandboxes = AsyncMock(side_effect=[error, page] if operation == "list" else [page])
    service.kill_sandbox = AsyncMock(side_effect=[error, None] if operation == "kill" else [None])
    install_lifecycle(monkeypatch, service)
    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"retries": 1, "retry_delay_s": 0, "retry_max_delay_s": 0},
    )
    with RequestFailureLog.capture() as log:
        found = await provider._reap_orphaned_create(scope, attempt_key)
    assert [item["id"] for item in found] == ["owned-id"]
    assert service.list_sandboxes.await_count == (2 if operation == "list" else 1)
    assert [call.args[0] for call in service.kill_sandbox.await_args_list] == ["owned-id"] * (
        2 if operation == "kill" else 1
    )
    event = next(iter(log.report.events.values()))
    assert event.http_status == status and event.boundary.value == "sandbox"
    assert event.details.sandbox_id == (None if operation == "list" else "owned-id")
    assert log.report.failure_id is None


async def test_orphan_reap_retries_share_original_deadline(monkeypatch):
    from functools import partial

    from opensandbox.exceptions import SandboxApiException
    from tenacity import wait_fixed

    loop = asyncio.get_running_loop()
    real_time = loop.time
    elapsed = 0.0
    waits = []
    backoff_cancelled = asyncio.Event()

    async def sleep(delay):
        nonlocal elapsed
        waits.append(float(delay))
        elapsed += float(delay)
        if len(waits) == 2:
            try:
                await asyncio.Event().wait()
            finally:
                backoff_cancelled.set()
        await asyncio.sleep(0)

    retrying, retry_if_exception, stop_after_attempt, _ = opensandbox_provider._require_tenacity()
    # Replace the backoff factory with a fixed wait; keep real retry/cancellation behavior.
    monkeypatch.setattr(
        opensandbox_provider,
        "_require_tenacity",
        lambda: (
            partial(retrying, sleep=sleep),
            retry_if_exception,
            stop_after_attempt,
            lambda **_: wait_fixed(40),
        ),
    )
    service = FakeLifecycleService(sandboxes=[])
    service.list_sandboxes = AsyncMock(side_effect=SandboxApiException("busy", status_code=429))
    install_lifecycle(monkeypatch, service)
    monkeypatch.setattr(opensandbox_provider, "ORPHAN_REAP_TIMEOUT_S", 60)
    provider = opensandbox_provider.OpenSandboxProvider(operations={"retries": 100})
    # Advance through two backoffs: the second crosses the original cleanup
    # deadline. Real asyncio cancellation must interrupt it before a third call.
    with monkeypatch.context() as clock, RequestFailureLog.capture() as log:
        clock.setattr(loop, "time", lambda: real_time() + elapsed)
        assert await asyncio.wait_for(provider._reap_orphaned_create({"attempt": "owned"}, "attempt"), 90) == []
    assert waits == [40, 40]
    assert backoff_cancelled.is_set()
    assert service.list_sandboxes.await_count == 2
    assert any(event.http_status == 429 for event in log.report.events.values())


@pytest.mark.parametrize("status", [401, 403, 404])
async def test_orphan_delete_permission_and_missing_semantics(monkeypatch, status):
    from opensandbox.exceptions import SandboxApiException

    service = FakeLifecycleService(sandboxes=[orphan("owned-id", {"attempt": "owned"})])
    # An auth error whose text says 'not found' must not be swallowed as a missing sandbox.
    service.kill_sandbox = AsyncMock(side_effect=SandboxApiException("sandbox not found", status_code=status))
    install_lifecycle(monkeypatch, service)
    provider = opensandbox_provider.OpenSandboxProvider(operations={"retries": 3, "retry_delay_s": 0})
    with RequestFailureLog.capture() as log:
        result = await provider._reap_orphaned_create({"attempt": "owned"}, "attempt")
    assert [item["id"] for item in result] == (["owned-id"] if status == 404 else [])
    service.kill_sandbox.assert_awaited_once()
    assert any(event.http_status == status for event in log.report.events.values()) is (status != 404)


@pytest.mark.parametrize("read_case", ["running", "recovered", "client-deadline-read"])
async def test_background_client_wait_expiry_preserves_last_running_observation(
    monkeypatch: pytest.MonkeyPatch, read_case: str
) -> None:
    """A successful Running observation is not exhausted status-read recovery."""
    clock = [0.0]
    calls = []
    real_sleep = asyncio.sleep

    class Commands:
        async def run(self, command, *, opts):
            calls.append("submit")
            return SimpleNamespace(id="still-running")

        async def get_command_status(self, execution_id):
            calls.append("status")
            if read_case == "recovered" and calls.count("status") == 1:
                raise TimeoutError("one failed read")
            if read_case == "client-deadline-read" and calls.count("status") == 3:
                # The real request timer clips this slow read at the client
                # deadline, before a status-recovery window could run.
                clock[0] = 0.05
                await real_sleep(0)
                await asyncio.Event().wait()
            return SimpleNamespace(running=True, exit_code=None)

        async def get_background_command_logs(self, execution_id):
            pytest.fail("no logs without a final command status")

    provider, handle = _background_provider(monkeypatch, Commands())
    provider._operations = replace(
        provider._operations,
        command_timeout_headroom_s=0.0,
        status_recovery_timeout_s=0.015 if read_case == "recovered" else 300.0,
    )
    monkeypatch.setattr(provider, "_sandbox_death_evidence", AsyncMock(return_value=None))
    loop = asyncio.get_running_loop()

    async def advance(delay):
        clock[0] += delay

    with monkeypatch.context() as timers:
        timers.setattr(loop, "time", lambda: clock[0])
        timers.setattr(asyncio, "sleep", advance)
        with raises_failure(PROVIDER_COMMAND_OUTCOME_UNKNOWN) as caught:
            await provider.exec(handle, "long command", timeout_s=0.05)

    assert calls.count("submit") == 1
    assert calls.count("status") >= 2
    assert "client wait expired" in caught.value.failure_reason
    assert "last observed running=True" in caught.value.failure_reason
    if read_case == "client-deadline-read":
        assert caught.value.__cause__.failure_kind == PROVIDER_IO_TIMEOUT
        assert isinstance(caught.value.__cause__.__cause__, TimeoutError)
    else:
        assert caught.value.__cause__ is None  # A recovered read is not this expiry's cause.
    assert clock[0] == pytest.approx(0.05)
    assert caught.value.http_status is None
    assert caught.value.details.execution_id == "still-running"


@pytest.mark.parametrize("fail_on_probe_kind,expected_attempts", [(False, 3), (True, 1)])
async def test_create_probe_policy_uses_current_kind_not_http_status_of_lifecycle_cause(
    monkeypatch: pytest.MonkeyPatch, fail_on_probe_kind: bool, expected_attempts: int
) -> None:
    """Probe expiry is a distinct readiness observation above the failed request and lifecycle evidence."""

    class Backend410Error(Exception):
        status_code = 410

    class DeadSandbox(FakeSandbox):
        attempts = 0
        request_errors = []

        def __init__(self):
            super().__init__(f"dead-{type(self).attempts}")
            self.commands = SimpleNamespace(run=self.run)

        @classmethod
        async def create(cls, **kwargs):
            cls.attempts += 1
            return cls()

        async def run(self, command, *, opts):
            error = Backend410Error("sandbox has ended")
            type(self).request_errors.append(error)
            raise error

        async def get_info(self):
            return SimpleNamespace(status=SimpleNamespace(state="Failed", reason="ContainerFailed"))

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (DeadSandbox, FakeConnectionConfig, dict, FakePlatformSpec, object),
    )
    rules = {"by_status_code": {410: "fail"}}
    if fail_on_probe_kind:
        rules["by_failure_kind"] = {"provider_not_ready": "fail"}
    provider = opensandbox_provider.OpenSandboxProvider(
        create={
            "retries": 2,
            "retry_delay_s": 0,
            "retry_max_delay_s": 0,
            "connect_poll_s": 0.02,
            "outcome_policy": rules,
        },
        probe={"command": "readiness", "deadline_s": 0.01},
    )
    cleanup = AsyncMock()
    monkeypatch.setattr(provider, "_cleanup_failed_create_handle", cleanup)

    with raises_failure("provider_not_ready") as caught:
        await provider.create(SandboxSpec(image="image:tag"))

    assert DeadSandbox.attempts == expected_attempts
    assert cleanup.await_count == expected_attempts
    probe = caught.value
    lifecycle = probe.__cause__
    request = lifecycle.__cause__
    assert probe.http_status is None
    assert lifecycle.failure_kind == PROVIDER_UNAVAILABLE
    assert lifecycle.http_status is None
    assert request.http_status == 410
    assert request.__cause__ is DeadSandbox.request_errors[-1]
