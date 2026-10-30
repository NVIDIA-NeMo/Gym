# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline execution lifecycle tests; commands run only in the test provider's temporary directories."""

import asyncio
import importlib
import importlib.util
import shlex
from pathlib import Path

import pytest

from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle, SandboxSpec, SandboxStatus
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    ResourcesSessionConflictError,
    ResourcesSessionLostError,
)


class DirectoryProvider:
    """Exercise the real AsyncSandbox facade without a container service."""

    name = "opensandbox"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.created = []
        self.live = set()
        self.stop_failures = 0
        self.create_failure = False
        self.exec_failure = None
        self.active_commands = 0
        self.peak_commands = 0
        self.closed = False

    async def create(self, spec):
        if self.create_failure:
            raise RuntimeError("synthetic provider create failure")
        assert spec.files == {}, "execution must not upload task data"
        self.created.append(spec)
        name = str(len(self.created))
        (self.root / name).mkdir()
        self.live.add(name)
        return SandboxHandle(name, self.name, None)

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if self.exec_failure is not None:
            return self.exec_failure
        self.active_commands += 1
        self.peak_commands = max(self.peak_commands, self.active_commands)
        try:
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                cwd=self.root / handle.sandbox_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
            return SandboxExecResult(
                stdout.decode(errors="replace"), stderr.decode(errors="replace"), process.returncode
            )
        finally:
            self.active_commands -= 1

    async def close(self, handle):
        if self.stop_failures:
            self.stop_failures -= 1
            raise RuntimeError("synthetic provider stop failure")
        self.live.discard(handle.sandbox_id)

    async def aclose(self):
        self.closed = True

    async def status(self, handle):
        return SandboxStatus.RUNNING if handle.sandbox_id in self.live else SandboxStatus.STOPPED


@pytest.fixture
def api():
    name = "resources_servers.critpt_custom_grader.agentic_tools"
    assert importlib.util.find_spec(name) is not None, "offline execution session helper is not implemented"
    return importlib.import_module(name)


def seed(session="s1", *, task="one", episode="r1", data=None):
    return ResourcesSeedSessionRequest(
        resources_session_id=session,
        episode_id=EpisodeId(rollout_id=episode),
        task_id=TaskId(taskset="synthetic", task_id=task),
        task_data=data or {},
    )


def close(session="s1", *, episode="r1"):
    return ResourcesCloseSessionRequest(resources_session_id=session, episode_id=EpisodeId(rollout_id=episode))


@pytest.fixture
def make_sessions(api, monkeypatch, tmp_path):
    provider = DirectoryProvider(tmp_path)
    monkeypatch.setattr(api, "create_provider", lambda config: provider)

    def make(**kwargs):
        config = api.AgenticExecutionConfig(
            sandbox_provider="execution",
            sandbox_spec={"image": "synthetic:fixture"},
            **kwargs,
        )
        spec = SandboxSpec(
            image="synthetic:fixture",
            workdir="/sandbox",
            ttl_s=21600,
            provider_options={"network_policy": {"defaultAction": "deny", "egress": []}},
        )
        sessions = api.AgenticExecutionSessions(
            config,
            provider_config={"opensandbox": {}},
            sandbox_spec=spec,
            close_retry_window_seconds=60,
        )
        return sessions, provider

    return make


async def test_persistent_files_are_isolated_by_session(make_sessions):
    sessions, provider = make_sessions()
    await sessions.seed(seed(data={"problem": "synthetic task"}))
    await sessions.seed(seed("s2", episode="r2"))
    assert "<exit_code>0</exit_code>" in await sessions.execute("s1", "printf value > saved.txt")
    assert "<stdout>value</stdout>" in await sessions.execute("s1", "cat saved.txt")
    assert "<exit_code>1</exit_code>" in await sessions.execute("s2", "test -f saved.txt")
    await sessions.shutdown()
    assert not provider.live
    assert provider.closed


async def test_seed_is_idempotent_but_conflicting_identity_or_data_is_rejected(make_sessions):
    sessions, provider = make_sessions()
    first = await sessions.seed(seed())
    assert first == await sessions.seed(seed())
    assert len(provider.created) == 1
    for request in (seed(task="two"), seed(episode="r2"), seed(data={"problem": "changed"})):
        with pytest.raises(ValueError, match="bound"):
            await sessions.seed(request)
    await sessions.shutdown()


async def test_commands_and_close_share_the_session_lock(make_sessions):
    sessions, provider = make_sessions()
    await sessions.seed(seed())
    first = asyncio.create_task(sessions.execute("s1", "sleep 0.05; printf first > saved"))
    second = asyncio.create_task(sessions.execute("s1", "cat saved"))
    results = await asyncio.gather(first, second)
    assert "<stdout>first</stdout>" in results[1]
    assert provider.peak_commands == 1
    await sessions.close(close())
    with pytest.raises(ResourcesSessionLostError, match="session"):
        await sessions.execute("s1", "true")
    await sessions.shutdown()


async def test_close_before_seed_and_wrong_episode_are_fenced(make_sessions):
    sessions, _ = make_sessions()
    await sessions.close(close())
    with pytest.raises(ValueError, match="closed"):
        await sessions.seed(seed())
    with pytest.raises(ValueError, match="episode"):
        await sessions.close(close(episode="r2"))
    await sessions.shutdown()


async def test_failed_stop_remains_retryable_but_cannot_execute_or_reseed(make_sessions):
    sessions, provider = make_sessions()
    await sessions.seed(seed())
    provider.stop_failures = 1
    with pytest.raises(RuntimeError, match="cleanup"):
        await sessions.close(close())
    with pytest.raises(ResourcesSessionLostError):
        await sessions.execute("s1", "true")
    with pytest.raises(ResourcesSessionConflictError):
        await sessions.seed(seed())
    assert provider.live
    await sessions.close(close())
    assert not provider.live
    await sessions.shutdown()


async def test_failed_create_cannot_be_resurrected_by_seed_retry(make_sessions):
    sessions, provider = make_sessions()
    provider.create_failure = True
    with pytest.raises(RuntimeError, match="create"):
        await sessions.seed(seed())
    provider.create_failure = False
    with pytest.raises(ResourcesSessionConflictError):
        await sessions.seed(seed())
    assert not provider.created
    await sessions.close(close())
    await sessions.shutdown()


async def test_shutdown_retries_failed_cleanup_and_refuses_new_sessions(make_sessions):
    sessions, provider = make_sessions()
    await sessions.seed(seed())
    provider.stop_failures = 1
    with pytest.raises(RuntimeError, match="cleanup"):
        await sessions.shutdown()
    assert provider.live
    assert not provider.closed
    with pytest.raises(RuntimeError):
        await sessions.seed(seed("s2", episode="r2"))
    await sessions.shutdown()
    assert not provider.live
    assert provider.closed


async def test_command_exit_timeout_and_provider_timeout_are_distinct(api, make_sessions):
    sessions, provider = make_sessions(command_timeout_s=0.1)
    await sessions.seed(seed())
    failed = await sessions.execute("s1", "printf ordinary >&2; exit 7")
    assert "<exit_code>7</exit_code>" in failed
    assert "<stderr>ordinary</stderr>" in failed
    timed_out = await sessions.execute("s1", "sleep 2")
    assert "<error_kind>timeout</error_kind>" in timed_out
    assert "<stdout>alive</stdout>" in await sessions.execute("s1", "printf alive")
    provider.exec_failure = SandboxExecResult("", "synthetic transport timeout", 125, "timeout")
    with pytest.raises(api.AgenticExecutionError, match="provider"):
        await sessions.execute("s1", "true")
    with pytest.raises(ResourcesSessionLostError):
        await sessions.execute("s1", "true")
    await sessions.shutdown()


async def test_shell_quoting_and_upstream_output_truncation(make_sessions):
    sessions, _ = make_sessions()
    await sessions.seed(seed())
    literal = "a'\"$()\\n"
    result = await sessions.execute("s1", "printf %s " + shlex.quote(literal))
    assert f"<stdout>{literal}</stdout>" in result
    result = await sessions.execute(
        "s1", 'python3 -c \'import sys; print("x"*25000); print("y"*25000,file=sys.stderr)\''
    )
    assert result.count("This content has been truncated") == 2
    assert len(result) < 41000
    await sessions.shutdown()


async def test_command_limit_is_repairable_and_capacity_is_bounded(make_sessions):
    sessions, _ = make_sessions(max_command_bytes=16, max_sessions=1)
    await sessions.seed(seed())
    result = await sessions.execute("s1", "x" * 17)
    assert "<error_kind>invalid_argument</error_kind>" in result
    assert "<exit_code>0</exit_code>" in await sessions.execute("s1", "true")
    with pytest.raises(RuntimeError, match="capacity"):
        await sessions.seed(seed("s2", episode="r2"))
    await sessions.close(close())
    await sessions.seed(seed("s2", episode="r2"))
    await sessions.shutdown()


@pytest.mark.parametrize(
    "provider,options",
    [
        ({"local": {}}, {}),
        ({"docker": {"create": {"network": "bridge"}}}, {}),
        ({"docker": {"create": {"network": "none", "extra_run_args": ["--network=host"]}}}, {}),
        ({"opensandbox": {}}, {"network_policy": {"defaultAction": "allow", "egress": []}}),
        (
            {"opensandbox": {}},
            {"network_policy": {"defaultAction": "deny", "egress": [{"action": "allow", "target": "*"}]}},
        ),
        ({"e2b": {"create": {"allow_internet_access": True}}}, {}),
    ],
)
def test_egress_enabled_or_unverified_provider_is_rejected(api, provider, options):
    config = api.AgenticExecutionConfig(sandbox_provider="execution", sandbox_spec={"image": "synthetic:fixture"})
    with pytest.raises(ValueError, match="offline"):
        api.AgenticExecutionSessions(
            config,
            provider_config=provider,
            sandbox_spec=SandboxSpec(image="synthetic:fixture", ttl_s=3600, provider_options=options),
            close_retry_window_seconds=60,
        )


async def test_output_budget_stops_flood_and_session_remains_usable(make_sessions):
    sessions, _ = make_sessions(max_output_bytes=20000)
    await sessions.seed(seed())
    result = await sessions.execute("s1", "python3 -c 'print(\"x\"*2000000)'")
    assert "<error_kind>output_limit</error_kind>" in result
    assert len(result) < 1000
    assert "<stdout>alive</stdout>" in await sessions.execute("s1", "printf alive")
    await sessions.shutdown()


@pytest.mark.parametrize(
    "provider,options",
    [
        ({"docker": {"create": {"network": "none"}}}, {}),
        ({"opensandbox": {}}, {"network_policy": {"defaultAction": "deny", "egress": []}}),
        ({"e2b": {"create": {"allow_internet_access": False}}}, {}),
    ],
)
def test_explicit_offline_providers_are_accepted(api, monkeypatch, tmp_path, provider, options):
    monkeypatch.setattr(api, "create_provider", lambda config: DirectoryProvider(tmp_path))
    config = api.AgenticExecutionConfig(sandbox_provider="execution", sandbox_spec={"image": "synthetic:fixture"})
    api.AgenticExecutionSessions(
        config,
        provider_config=provider,
        sandbox_spec=SandboxSpec(image="synthetic:fixture", ttl_s=3600, provider_options=options),
        close_retry_window_seconds=60,
    )


@pytest.mark.parametrize("overrides", [{"ttl_s": None}, {"ttl_s": 0}, {"files": {"/sandbox/task": "not allowed"}}])
def test_unbounded_or_file_seeded_execution_specs_are_rejected(api, overrides):
    config = api.AgenticExecutionConfig(sandbox_provider="execution", sandbox_spec={"image": "synthetic:fixture"})
    values = {
        "image": "synthetic:fixture",
        "ttl_s": 3600,
        "provider_options": {"network_policy": {"defaultAction": "deny", "egress": []}},
    }
    with pytest.raises(ValueError):
        api.AgenticExecutionSessions(
            config,
            provider_config={"opensandbox": {}},
            sandbox_spec=SandboxSpec(**(values | overrides)),
            close_retry_window_seconds=60,
        )


async def test_expired_session_is_closed_instead_of_reused(api, make_sessions, monkeypatch):
    sessions, provider = make_sessions()
    now = [100.0]
    monkeypatch.setattr(api.time, "monotonic", lambda: now[0])
    await sessions.seed(seed())
    now[0] += 22000
    with pytest.raises(ResourcesSessionLostError, match="expired"):
        await sessions.execute("s1", "true")
    assert not provider.live
    with pytest.raises(ResourcesSessionConflictError):
        await sessions.seed(seed())
    await sessions.shutdown()


@pytest.mark.parametrize("cancel_seed", [False, True], ids=["create-timeout", "caller-cancellation"])
async def test_late_create_is_owned_until_its_handle_can_be_closed(api, make_sessions, monkeypatch, cancel_seed):
    sessions, provider = make_sessions(create_timeout_s=0.02, close_timeout_s=0.02)
    created, release, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_create, original_close = provider.create, provider.close

    async def delayed_create(spec):
        handle = await original_create(spec)
        created.set()
        await release.wait()
        return handle

    async def observed_close(handle):
        await original_close(handle)
        cleaned.set()

    monkeypatch.setattr(provider, "create", delayed_create)
    monkeypatch.setattr(provider, "close", observed_close)
    creating = asyncio.create_task(sessions.seed(seed()))
    await asyncio.wait_for(created.wait(), 1)
    if cancel_seed:
        creating.cancel()
        expected = asyncio.CancelledError
    else:
        expected = api.AgenticExecutionError
    with pytest.raises(expected):
        await asyncio.wait_for(creating, 1)
    assert provider.live
    release.set()
    await asyncio.wait_for(cleaned.wait(), 1)
    assert not provider.live
    with pytest.raises(ResourcesSessionConflictError):
        await sessions.seed(seed())
    await sessions.shutdown()


def test_automatic_renewal_cannot_remove_the_offline_session_ttl(api, monkeypatch):
    monkeypatch.setattr(api, "create_provider", lambda config: object())
    config = api.AgenticExecutionConfig(sandbox_provider="execution", sandbox_spec={"image": "synthetic:fixture"})
    spec = SandboxSpec(
        image="synthetic:fixture",
        ttl_s=3600,
        provider_options={"network_policy": {"defaultAction": "deny", "egress": []}},
    )
    with pytest.raises(ValueError, match="[Oo]ffline"):
        api.AgenticExecutionSessions(
            config,
            provider_config={"opensandbox": {"create": {"renew_interval_s": 30}}},
            sandbox_spec=spec,
            close_retry_window_seconds=60,
        )


async def test_expired_idle_session_releases_capacity_and_preserves_its_fence(api, make_sessions, monkeypatch):
    sessions, provider = make_sessions(max_sessions=1)
    now = [100.0]
    monkeypatch.setattr(api.time, "monotonic", lambda: now[0])
    await sessions.seed(seed())
    now[0] += 22000
    await sessions.seed(seed("s2", episode="r2"))
    assert provider.live == {"2"}
    with pytest.raises(ResourcesSessionConflictError, match="closed"):
        await sessions.seed(seed())
    await sessions.shutdown()


async def test_expired_idle_session_with_failed_cleanup_keeps_its_capacity_slot(api, make_sessions, monkeypatch):
    sessions, provider = make_sessions(max_sessions=1)
    now = [100.0]
    monkeypatch.setattr(api.time, "monotonic", lambda: now[0])
    await sessions.seed(seed())
    now[0] += 22000
    provider.stop_failures = 1
    with pytest.raises(api.AgenticExecutionError, match="capacity"):
        await sessions.seed(seed("s2", episode="r2"))
    assert provider.live == {"1"}
    await sessions.seed(seed("s2", episode="r2"))
    assert provider.live == {"2"}
    await sessions.shutdown()


async def test_delayed_stop_is_owned_and_reused_after_cleanup_timeout(api, make_sessions, monkeypatch):
    sessions, provider = make_sessions(close_timeout_s=0.02)
    await sessions.seed(seed())
    release, cleaned = asyncio.Event(), asyncio.Event()
    original_close = provider.close
    stop_calls = []

    async def delayed_stop(handle):
        stop_calls.append(handle.sandbox_id)
        await release.wait()
        await original_close(handle)
        cleaned.set()

    monkeypatch.setattr(provider, "close", delayed_stop)
    with pytest.raises(api.AgenticExecutionError, match="cleanup"):
        await sessions.close(close())
    assert provider.live
    assert not provider.closed
    release.set()
    await asyncio.wait_for(cleaned.wait(), 1)
    await sessions.close(close())
    assert stop_calls == ["1"]
    assert not provider.live
    await sessions.shutdown()
