# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline execution lifecycle tests; commands run only in the test provider's temporary directories."""

import asyncio
import shlex
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.core import (
    PROVIDER_OOM_KILLED,
    PROVIDER_UNAVAILABLE,
    SESSION_EXPIRED,
    EpisodeId,
    NgFailureEvent,
    SandboxFailureDetails,
    TaskId,
)
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle, SandboxSpec, SandboxStatus
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    ResourcesSessionConflictError,
    ResourcesSessionLostError,
    check_resources_session_contract,
)
from resources_servers.critpt_custom_grader import agentic_tools


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
        self.exec_error = None
        self.active_commands = 0
        self.peak_commands = 0
        self.closed = False
        # Provider records deleted by a successful stop; an ended sandbox's record stays until then.
        self.deleted = []
        # When set, a stop waits for the gate, so a test can act while a delete is in flight.
        self.stop_gate = None
        self.stop_started = asyncio.Event()

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
        if self.exec_error is not None:
            # One command fails; a provider that reports the sandbox ended has also ended it.
            error, self.exec_error = self.exec_error, None
            self.live.discard(handle.sandbox_id)
            raise error
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
        if self.stop_gate is not None:
            self.stop_started.set()
            await self.stop_gate.wait()
        if self.stop_failures:
            self.stop_failures -= 1
            raise RuntimeError("synthetic provider stop failure")
        self.live.discard(handle.sandbox_id)
        self.deleted.append(handle.sandbox_id)

    async def aclose(self):
        self.closed = True

    async def status(self, handle):
        return SandboxStatus.RUNNING if handle.sandbox_id in self.live else SandboxStatus.STOPPED


@pytest.fixture
def api():
    return agentic_tools


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


def test_execution_sessions_follow_the_resources_session_contract(make_sessions):
    from resources_servers.critpt_custom_grader import app as adapter

    sessions, provider = make_sessions()
    config = adapter.CritPtCustomGraderConfig(
        host="127.0.0.1",
        port=8000,
        name="synthetic-grader",
        entrypoint="app.py",
        sandbox_spec={"image": "example/grader:test"},
        startup_probe=False,
        answer_source="finish_tool",
        agentic_execution={"sandbox_provider": "execution", "sandbox_spec": {"image": "synthetic:fixture"}},
    )
    server = adapter.CritPtCustomGraderServer(config=config, server_client=MagicMock(spec=ServerClient))
    # Startup builds the sessions inside the lifespan, which the contract check does not run.
    server._agentic_sessions = sessions
    task = {
        "problem_id": "synthetic-1",
        "reference_source": "def solve(value):\n    return value\n",
        "entrypoint": "solve",
        "test_cases": [{"input": [1], "output": "synthetic"}],
    }
    check_resources_session_contract(
        server.setup_webserver(),
        seed("contract-session", data=task),
        keeps_state=True,
        holds_state=lambda resources_session_id: resources_session_id in sessions._sessions,
    )
    assert len(provider.created) == 2 and not provider.live


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


def sandbox_ended(failure_kind):
    return NgFailureEvent(
        boundary="sandbox",
        failure_kind=failure_kind,
        failure_reason="synthetic sandbox end",
        details=SandboxFailureDetails(provider="opensandbox", sandbox_id="1"),
    )


async def test_command_that_exhausts_memory_is_reported_and_the_episode_continues_in_a_fresh_sandbox(make_sessions):
    sessions, provider = make_sessions()
    await sessions.seed(seed())
    await sessions.execute("s1", "printf value > saved.txt")
    provider.exec_error = sandbox_ended(PROVIDER_OOM_KILLED)
    result = await sessions.execute("s1", "allocate more than the limit")
    assert "<error_kind>out_of_memory</error_kind>" in result
    assert "files and installed packages from earlier commands are gone" in result
    assert len(provider.created) == 2 and provider.live == {"2"}
    assert "<exit_code>1</exit_code>" in await sessions.execute("s1", "test -f saved.txt")
    assert "<stdout>alive</stdout>" in await sessions.execute("s1", "printf alive")
    await sessions.close(close())
    assert not provider.live
    await sessions.shutdown()


async def test_a_failed_delete_of_the_ended_sandbox_is_retried_when_the_session_closes(api, make_sessions):
    sessions, provider = make_sessions()
    await sessions.seed(seed())
    provider.exec_error = sandbox_ended(PROVIDER_OOM_KILLED)
    provider.stop_failures = 1
    assert "out_of_memory" in await sessions.execute("s1", "allocate more than the limit")
    assert "<stdout>alive</stdout>" in await sessions.execute("s1", "printf alive")
    assert provider.deleted == []
    # A stop that fails at close keeps the session; the next close deletes the replacement and the ended record.
    provider.stop_failures = 1
    with pytest.raises(api.AgenticExecutionError, match="cleanup"):
        await sessions.close(close())
    await sessions.close(close())
    assert sorted(provider.deleted) == ["1", "2"]
    await sessions.shutdown()


async def test_a_request_cancelled_during_the_ended_sandbox_delete_leaves_no_usable_session(api, make_sessions):
    sessions, provider = make_sessions(max_sessions=1)
    await sessions.seed(seed())
    provider.exec_error = sandbox_ended(PROVIDER_OOM_KILLED)
    provider.stop_gate = asyncio.Event()
    command = asyncio.create_task(sessions.execute("s1", "allocate more than the limit"))
    await asyncio.wait_for(provider.stop_started.wait(), 10)
    command.cancel()
    with pytest.raises(asyncio.CancelledError):
        await command
    # No replacement was started, and a replayed seed does not revive the dead sandbox.
    assert len(provider.created) == 1
    with pytest.raises(ResourcesSessionConflictError):
        await sessions.seed(seed())
    with pytest.raises(api.AgenticExecutionError, match="capacity"):
        await sessions.seed(seed("s2", episode="r2"))
    # Once the delete finishes, its capacity goes to another episode.
    provider.stop_gate.set()
    await asyncio.sleep(0)
    for _ in range(100):
        if provider.deleted:
            break
        await asyncio.sleep(0.01)
    provider.stop_gate = None
    await sessions.seed(seed("s2", episode="r2"))
    assert provider.deleted[0] == "1"
    await sessions.shutdown()


async def test_memory_message_names_the_configured_limit(api, make_sessions):
    sessions, provider = make_sessions()
    sessions._spec = replace(sessions._spec, resources={"memory_mib": 8192})
    await sessions.seed(seed())
    provider.exec_error = sandbox_ended(PROVIDER_OOM_KILLED)
    assert "memory limit (8192 MiB)" in await sessions.execute("s1", "allocate more than the limit")
    await sessions.shutdown()


async def test_other_sandbox_failures_and_a_failed_replacement_end_the_session(api, make_sessions):
    sessions, provider = make_sessions()
    await sessions.seed(seed())
    provider.exec_error = sandbox_ended(PROVIDER_UNAVAILABLE)
    with pytest.raises(api.AgenticExecutionError, match="provider"):
        await sessions.execute("s1", "true")
    with pytest.raises(ResourcesSessionLostError):
        await sessions.execute("s1", "true")
    await sessions.seed(seed("s2", episode="r2"))
    provider.exec_error = sandbox_ended(PROVIDER_OOM_KILLED)
    provider.create_failure = True
    with pytest.raises(api.AgenticExecutionError, match="create"):
        await sessions.execute("s2", "allocate more than the limit")
    with pytest.raises(ResourcesSessionLostError):
        await sessions.execute("s2", "true")
    assert not provider.live
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


async def test_command_that_outgrows_one_exec_argument_is_repairable(make_sessions):
    sessions, _ = make_sessions()
    await sessions.seed(seed())
    # Within max_command_bytes, but shell-quoting its 60,000 single quotes outgrows one exec argument.
    result = await sessions.execute("s1", "printf %s " + "''" * 30000)
    assert "<error_kind>invalid_argument</error_kind>" in result
    assert "<exit_code>0</exit_code>" in await sessions.execute("s1", "true")
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
        ({"e2b": {"create": {"allow_internet_access": False, "network": {"allow_out": ["192.0.2.10"]}}}}, {}),
        ({"e2b": {"create": {"allow_internet_access": False, "custom_extension_params": {"cache": "all"}}}}, {}),
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
    with pytest.raises(ResourcesSessionLostError, match="expired") as lost:
        await sessions.execute("s1", "true")
    assert lost.value.detail["failure_kind"] == SESSION_EXPIRED
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


async def test_stopped_session_without_a_close_releases_its_capacity_slot(make_sessions):
    sessions, provider = make_sessions(max_sessions=1)
    provider.create_failure = True
    with pytest.raises(RuntimeError, match="create"):
        await sessions.seed(seed())
    provider.create_failure = False
    await sessions.seed(seed("s2", episode="r2"))
    assert provider.live == {"1"}
    with pytest.raises(ResourcesSessionConflictError, match="closed"):
        await sessions.seed(seed())
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
