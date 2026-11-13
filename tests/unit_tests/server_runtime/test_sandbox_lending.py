# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LentSandboxes on the local sandbox provider, in a temporary working directory."""

import asyncio
import logging
import time
from pathlib import Path
from unittest.mock import AsyncMock

import anyio
import pytest
from fastapi import HTTPException

from nemo_gym.core.episode_types import TaskId
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.config import SandboxConfig
from nemo_gym.sandbox.providers.local.provider import LocalProvider
from nemo_gym.server_runtime import sandbox_lending
from nemo_gym.server_runtime.base_resources_server import ResourcesSeedRejectedError
from nemo_gym.server_runtime.sandbox_lending import LentSandboxes


TASK = TaskId(taskset="lending:test", task_id="t" * 80)


@pytest.fixture
def workdir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(sandbox_lending, "get_global_config_dict", lambda: {"sandbox": {"local": {}}})
    # The local provider cannot be reattached from another process, so it has no descriptor of its own.
    monkeypatch.setattr(AsyncSandbox, "serialize", AsyncMock(return_value={"sandbox_id": "local"}))
    return tmp_path / "workspace"


@pytest.fixture
def started(monkeypatch) -> list:
    specs = []
    real_start = AsyncSandbox.start

    async def start(self: AsyncSandbox, spec) -> None:
        specs.append(spec)
        await real_start(self, spec)

    monkeypatch.setattr(AsyncSandbox, "start", start)
    return specs


@pytest.fixture
def stopped(monkeypatch) -> list[AsyncSandbox]:
    stopped: list[AsyncSandbox] = []
    real_stop = AsyncSandbox.stop

    async def stop(self: AsyncSandbox) -> None:
        stopped.append(self)
        await real_stop(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    return stopped


def _lender(workdir: Path) -> LentSandboxes:
    config = SandboxConfig(ttl_s=60, metadata={"team": "gym"})
    return LentSandboxes("sandbox", config, image="unused-by-the-local-provider", workdir=str(workdir))


async def test_lend_starts_one_sandbox_per_session_and_release_stops_it(workdir, started, stopped) -> None:
    lender = _lender(workdir)

    access = await lender.lend("session", TASK, files={str(workdir / "images" / "0.png"): b"png"})
    again = await lender.lend("session", TASK, files={str(workdir / "other.png"): b"ignored"})

    assert access == again
    assert access.workdir == str(workdir)
    assert access.connection.provider_config_ref == "sandbox"
    assert access.connection.descriptor == {"sandbox_id": "local"}
    # The harness gets an operate descriptor: it may use the sandbox, but only this server stops it.
    AsyncSandbox.serialize.assert_awaited_with(scope="operate")
    assert (workdir / "images" / "0.png").read_bytes() == b"png"
    assert not (workdir / "other.png").exists(), "a repeated lend must not start a second sandbox"
    assert len(started) == 1
    spec = started[0]
    assert spec.ttl_s == 60 and spec.workdir == str(workdir)
    assert spec.metadata == {"team": "gym", "task_id": "t" * 63}

    sandbox = lender.get("session")
    await lender.release("session")
    await lender.release("session")
    assert stopped == [sandbox]
    assert lender.get("session") is None


async def test_lend_can_start_an_empty_sandbox(workdir, started, stopped) -> None:
    lender = _lender(workdir)

    await lender.lend("session", TASK)

    assert started[0].files == {}
    await lender.release_all()
    assert len(stopped) == 1


async def test_a_failed_stop_keeps_the_sandbox_so_a_close_can_retry(workdir, started, monkeypatch) -> None:
    lender = _lender(workdir)
    await lender.lend("session", TASK)
    real_stop = AsyncSandbox.stop
    monkeypatch.setattr(AsyncSandbox, "stop", AsyncMock(side_effect=RuntimeError("provider down")))

    with pytest.raises(RuntimeError, match="provider down"):
        await lender.release("session")
    assert lender.get("session") is not None

    monkeypatch.setattr(AsyncSandbox, "stop", real_stop)
    await lender.release("session")
    assert lender.get("session") is None


async def test_release_all_stops_every_sandbox_and_logs_a_failed_stop(workdir, started, monkeypatch, caplog) -> None:
    lender = _lender(workdir)
    await lender.lend("ok", TASK)
    await lender.lend("stuck", TASK)
    stuck = lender.get("stuck")
    real_stop = AsyncSandbox.stop

    async def stop(self: AsyncSandbox) -> None:
        if self is stuck:
            raise RuntimeError("provider down")
        await real_stop(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    with caplog.at_level(logging.ERROR, logger=sandbox_lending.__name__):
        await lender.release_all()

    assert lender.get("ok") is None
    assert lender.get("stuck") is stuck
    assert "resources session stuck" in caplog.text


async def test_a_sandbox_that_cannot_be_described_is_stopped_and_not_kept(
    workdir, started, stopped, monkeypatch
) -> None:
    monkeypatch.setattr(AsyncSandbox, "serialize", AsyncMock(side_effect=RuntimeError("no descriptor")))
    lender = _lender(workdir)

    with pytest.raises(RuntimeError, match="no descriptor"):
        await lender.lend("session", TASK)

    assert len(stopped) == 1
    assert lender.get("session") is None


async def test_release_finishes_the_stop_when_the_close_is_cancelled(workdir, started, monkeypatch) -> None:
    lender = _lender(workdir)
    await lender.lend("session", TASK)
    sandbox = lender.get("session")
    finished: list[AsyncSandbox] = []

    async def stop(self: AsyncSandbox) -> None:
        # A checkpoint where an unshielded stop would be cancelled with its request.
        await anyio.sleep(0)
        finished.append(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    with anyio.CancelScope() as scope:
        scope.cancel()
        await lender.release("session")

    assert finished == [sandbox]
    assert lender.get("session") is None


async def test_a_stop_that_hangs_times_out_and_keeps_the_sandbox(workdir, started, monkeypatch) -> None:
    lender = _lender(workdir)
    await lender.lend("session", TASK)
    monkeypatch.setattr(sandbox_lending, "STOP_TIMEOUT_S", 0.05)

    async def stop(self: AsyncSandbox) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    release = asyncio.create_task(lender.release("session"))
    done, _ = await asyncio.wait({release}, timeout=5)

    assert done == {release}, "release() must give up on a stop that hangs"
    with pytest.raises(TimeoutError):
        release.result()
    assert lender.get("session") is not None


async def test_release_all_stops_the_sandboxes_concurrently(workdir, started, monkeypatch) -> None:
    lender = _lender(workdir)
    for session in ("a", "b", "c"):
        await lender.lend(session, TASK)
    entered: list[AsyncSandbox] = []
    all_entered = asyncio.Event()

    async def stop(self: AsyncSandbox) -> None:
        # Each stop waits until all three have started, so stops run one at a time would never finish.
        entered.append(self)
        if len(entered) == 3:
            all_entered.set()
        await all_entered.wait()

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    await asyncio.wait_for(lender.release_all(), timeout=5)

    assert len(entered) == 3
    assert all(lender.get(session) is None for session in ("a", "b", "c"))


async def test_a_lend_cancelled_while_describing_the_sandbox_still_stops_it(workdir, monkeypatch) -> None:
    lender = _lender(workdir)
    seed = anyio.CancelScope()
    finished: list[AsyncSandbox] = []

    async def serialize(self: AsyncSandbox, *, scope: str | None = None) -> dict[str, str]:
        # The seed request is cancelled while the new sandbox is being described.
        seed.cancel()
        await anyio.sleep(0)
        return {"sandbox_id": "local"}

    async def stop(self: AsyncSandbox) -> None:
        # A checkpoint where an unshielded stop would be cancelled with the seed.
        await anyio.sleep(0)
        finished.append(self)

    monkeypatch.setattr(AsyncSandbox, "serialize", serialize)
    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    with seed:
        await lender.lend("session", TASK)

    assert len(finished) == 1
    assert lender.get("session") is None


async def test_a_lend_cancelled_while_delivering_files_still_stops_the_sandbox(workdir, monkeypatch) -> None:
    lender = _lender(workdir)
    seed = anyio.CancelScope()
    closed: list[object] = []
    real_close = LocalProvider.close

    async def upload_file(self: LocalProvider, handle, source_path, target_path) -> None:
        # The seed request is cancelled while the new sandbox receives its files.
        seed.cancel()
        await anyio.sleep(0)

    async def close(self: LocalProvider, handle) -> None:
        # A checkpoint where a stop that is not shielded is cancelled with the seed, as start()'s own stop is.
        await anyio.sleep(0)
        await real_close(self, handle)
        closed.append(handle)

    monkeypatch.setattr(LocalProvider, "upload_file", upload_file)
    monkeypatch.setattr(LocalProvider, "close", close)
    with seed:
        await lender.lend("session", TASK, files={str(workdir / "0.png"): b"png"})

    assert len(closed) == 1, "the sandbox must be stopped although the seed was cancelled"
    assert lender.get("session") is None
    await lender.release_all()
    assert len(closed) == 1, "nothing is left for the shutdown to stop"


def _stop_failing_first(monkeypatch, failure) -> list[AsyncSandbox]:
    """Make the first stop fail by calling ``failure``, and record each later stop, which really stops."""
    stopped: list[AsyncSandbox] = []
    real_stop = AsyncSandbox.stop
    calls = 0

    async def stop(self: AsyncSandbox) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            await failure()
        await real_stop(self)
        stopped.append(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    return stopped


async def _provider_down() -> None:
    raise RuntimeError("provider down")


async def _hang() -> None:
    await asyncio.Event().wait()


@pytest.mark.parametrize("failure", [_provider_down, _hang], ids=["stop fails", "stop hangs"])
@pytest.mark.parametrize("cleanup", ["release", "release_all"])
async def test_a_failed_seed_whose_stop_fails_is_stopped_later(
    workdir, started, monkeypatch, failure, cleanup
) -> None:
    monkeypatch.setattr(sandbox_lending, "STOP_TIMEOUT_S", 0.05)
    monkeypatch.setattr(AsyncSandbox, "serialize", AsyncMock(side_effect=RuntimeError("no descriptor")))
    stopped = _stop_failing_first(monkeypatch, failure)
    lender = _lender(workdir)

    with pytest.raises(RuntimeError, match="no descriptor"):
        await lender.lend("session", TASK)
    assert lender.get("session") is None, "a failed seed's sandbox is never lent"

    if cleanup == "release":
        await lender.release("session")
        await lender.release("session")
    else:
        await lender.release_all()
        await lender.release_all()
    assert len(stopped) == 1 and len(started) == 1


async def test_a_repeated_seed_stops_the_failed_seed_s_sandbox_before_it_starts_another(
    workdir, started, monkeypatch
) -> None:
    describe = AsyncMock(side_effect=[RuntimeError("no descriptor"), {"sandbox_id": "local"}])
    monkeypatch.setattr(AsyncSandbox, "serialize", describe)
    stopped = _stop_failing_first(monkeypatch, _provider_down)
    lender = _lender(workdir)
    with pytest.raises(RuntimeError, match="no descriptor"):
        await lender.lend("session", TASK)
    assert stopped == [], "the first stop failed"

    # The second seed first stops the sandbox the first seed left, and only then starts its own.
    access = await lender.lend("session", TASK)

    assert len(started) == 2
    assert len(stopped) == 1 and stopped[0] is not lender.get("session")
    assert access.connection.descriptor == {"sandbox_id": "local"}
    await lender.release("session")
    assert len(stopped) == 2


async def test_a_repeated_seed_fails_while_the_failed_seed_s_sandbox_cannot_be_stopped(
    workdir, started, monkeypatch
) -> None:
    monkeypatch.setattr(AsyncSandbox, "serialize", AsyncMock(side_effect=RuntimeError("no descriptor")))
    lender = _lender(workdir)
    real_stop = AsyncSandbox.stop
    monkeypatch.setattr(AsyncSandbox, "stop", AsyncMock(side_effect=RuntimeError("provider down")))
    with pytest.raises(RuntimeError, match="no descriptor"):
        await lender.lend("session", TASK)

    with pytest.raises(RuntimeError, match="provider down"):
        await lender.lend("session", TASK)
    assert len(started) == 1, "no second sandbox starts while the first one may still run"

    monkeypatch.setattr(AsyncSandbox, "stop", real_stop)
    await lender.release("session")
    assert lender.get("session") is None


class _RecordingSandbox:
    """Stands in for AsyncSandbox on a provider this test cannot start: records the provider and the spec."""

    created: list["_RecordingSandbox"] = []

    def __init__(self, provider) -> None:
        self.provider = provider
        self.spec = None
        _RecordingSandbox.created.append(self)

    async def start(self, spec) -> None:
        self.spec = spec

    async def serialize(self, *, scope: str) -> dict:
        return {"sandbox_id": "recorded"}

    async def stop(self) -> None:
        pass


def test_model_only_needs_a_model_server_and_no_explicit_network_policy() -> None:
    with pytest.raises(ValueError, match="needs model_server"):
        LentSandboxes("sandbox", SandboxConfig(), image=None, workdir="/workspace", network_access="model_only")
    with pytest.raises(ValueError, match="not both"):
        LentSandboxes(
            "sandbox",
            SandboxConfig(provider_options={"network_policy": {"defaultAction": "deny"}}),
            image=None,
            workdir="/workspace",
            network_access="model_only",
            model_server="policy_model",
        )


async def test_model_only_lends_a_sandbox_that_reaches_only_the_model_server(monkeypatch) -> None:
    monkeypatch.setattr(sandbox_lending, "get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}})
    asked = []

    def server_url(name: str) -> str:
        asked.append(name)
        return "http://10.1.2.3:8000"

    monkeypatch.setattr(sandbox_lending, "get_server_url", server_url)
    monkeypatch.setattr(_RecordingSandbox, "created", [])
    monkeypatch.setattr(sandbox_lending, "AsyncSandbox", _RecordingSandbox)
    lender = LentSandboxes(
        "sandbox",
        SandboxConfig(provider_options={"keep": True}),
        image="task-image",
        workdir="/workspace",
        network_access="model_only",
        model_server="policy_model",
    )

    await lender.lend("session", TASK)

    (sandbox,) = _RecordingSandbox.created
    assert asked == ["policy_model"]
    assert sandbox.provider == {"opensandbox": {}}
    assert sandbox.spec.provider_options == {
        "keep": True,
        "network_policy": {"defaultAction": "deny", "egress": [{"action": "allow", "target": "10.1.2.3"}]},
    }


def test_model_and_tools_needs_tool_servers_and_only_it_takes_them() -> None:
    with pytest.raises(ValueError, match="needs tool_servers"):
        LentSandboxes(
            "sandbox",
            SandboxConfig(),
            image=None,
            workdir="/workspace",
            network_access="model_and_tools",
            model_server="policy_model",
        )
    with pytest.raises(ValueError, match="needs model_server"):
        LentSandboxes(
            "sandbox",
            SandboxConfig(),
            image=None,
            workdir="/workspace",
            network_access="model_and_tools",
            tool_servers=["agentic_mme"],
        )
    for network_access in ("inherit", "model_only"):
        with pytest.raises(ValueError, match="only to network_access model_and_tools"):
            LentSandboxes(
                "sandbox",
                SandboxConfig(),
                image=None,
                workdir="/workspace",
                network_access=network_access,
                model_server="policy_model",
                tool_servers=["agentic_mme"],
            )


async def test_model_and_tools_lends_a_sandbox_that_reaches_the_model_and_tool_servers(monkeypatch) -> None:
    monkeypatch.setattr(sandbox_lending, "get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}})
    urls = {
        "policy_model": "http://10.1.2.3:8000",
        "agentic_mme": "http://10.4.5.6:9000",
        "search": "http://10.1.2.3:7000",
    }
    asked = []

    def server_url(name: str) -> str:
        asked.append(name)
        return urls[name]

    monkeypatch.setattr(sandbox_lending, "get_server_url", server_url)
    monkeypatch.setattr(_RecordingSandbox, "created", [])
    monkeypatch.setattr(sandbox_lending, "AsyncSandbox", _RecordingSandbox)
    lender = LentSandboxes(
        "sandbox",
        SandboxConfig(),
        image="task-image",
        workdir="/workspace",
        network_access="model_and_tools",
        model_server="policy_model",
        tool_servers=["agentic_mme", "search"],
    )

    await lender.lend("session", TASK)

    (sandbox,) = _RecordingSandbox.created
    assert asked == ["policy_model", "agentic_mme", "search"]
    # One rule per host: the tool server that shares the model server's host adds no rule of its own.
    assert sandbox.spec.provider_options == {
        "network_policy": {
            "defaultAction": "deny",
            "egress": [{"action": "allow", "target": "10.1.2.3"}, {"action": "allow", "target": "10.4.5.6"}],
        },
    }


async def test_model_only_on_a_provider_without_network_policies_is_refused_before_a_sandbox_starts(
    workdir, started, monkeypatch
) -> None:
    monkeypatch.setattr(sandbox_lending, "get_server_url", lambda name: "http://10.1.2.3:8000")
    lender = LentSandboxes(
        "sandbox",
        SandboxConfig(),
        image=None,
        workdir=str(workdir),
        network_access="model_only",
        model_server="policy_model",
    )

    with pytest.raises(ResourcesSeedRejectedError, match="requires the OpenSandbox network-policy provider") as error:
        await lender.lend("session", TASK)

    assert error.value.status_code == 422
    assert started == []
    assert lender.get("session") is None


async def test_inherit_leaves_the_provider_network_alone(workdir, started) -> None:
    lender = _lender(workdir)

    await lender.lend("session", TASK)

    assert "network_policy" not in started[0].provider_options
    await lender.release_all()


async def test_a_lend_gives_its_sandbox_its_own_image_env_and_setup(workdir, started, stopped) -> None:
    lender = _lender(workdir)
    set_up: list[AsyncSandbox] = []

    async def setup(sandbox: AsyncSandbox) -> None:
        # Runs against the started sandbox, its files delivered, before it is lent.
        assert (workdir / "guide.md").read_text() == "read me"
        set_up.append(sandbox)

    await lender.lend(
        "session", TASK, files={str(workdir / "guide.md"): "read me"}, image="row-image", env={"API": "x"}, setup=setup
    )

    assert started[0].image == "row-image" and started[0].env["API"] == "x"
    assert set_up == [lender.get("session")]


async def test_a_setup_that_fails_stops_the_sandbox_and_fails_the_lend(workdir, started, stopped) -> None:
    lender = _lender(workdir)

    async def setup(sandbox: AsyncSandbox) -> None:
        raise RuntimeError("setup failed")

    with pytest.raises(RuntimeError, match="setup failed"):
        await lender.lend("session", TASK, setup=setup)

    # start_with_setup() stops it, and the lend's own stop then finds it stopped.
    assert len({id(sandbox) for sandbox in stopped}) == 1 and lender.get("session") is None


async def test_a_lend_cancelled_while_its_sandbox_is_created_stops_it_once_the_create_returns(
    workdir, monkeypatch, caplog
) -> None:
    # A create cut in flight would leave a sandbox whose id the provider never returned, so the create runs to its end.
    lender = _lender(workdir)
    seed = anyio.CancelScope()
    created: list[object] = []
    closed: list[object] = []
    real_create, real_close = LocalProvider.create, LocalProvider.close

    async def create(self: LocalProvider, spec):
        seed.cancel()
        await anyio.sleep(0)  # where an unshielded create is cancelled with the seed
        handle = await real_create(self, spec)
        created.append(handle)
        return handle

    async def close(self: LocalProvider, handle) -> None:
        await real_close(self, handle)
        closed.append(handle)

    monkeypatch.setattr(LocalProvider, "create", create)
    monkeypatch.setattr(LocalProvider, "close", close)
    with caplog.at_level(logging.INFO, logger=sandbox_lending.__name__), seed:
        await lender.lend("session", TASK)

    assert len(created) == 1 and closed == created
    assert lender.get("session") is None
    assert "resources session session: its seed was cancelled while it started" in caplog.text


async def test_a_start_that_hangs_ends_at_the_start_time_limit_and_lends_nothing(
    workdir, monkeypatch, stopped
) -> None:
    lender = LentSandboxes("sandbox", SandboxConfig(), image=None, workdir=str(workdir), start_timeout_s=0.05)

    async def create(self: LocalProvider, spec):
        await asyncio.Event().wait()  # a provider's create that never returns

    monkeypatch.setattr(LocalProvider, "create", create)
    with pytest.raises(TimeoutError):
        await lender.lend("session", TASK)

    assert len(stopped) == 1 and lender.get("session") is None


async def test_release_all_waits_for_a_lend_still_starting_and_lends_nothing_after(workdir, monkeypatch) -> None:
    lender = _lender(workdir)
    creating, go_on = asyncio.Event(), asyncio.Event()
    closed: list[object] = []
    real_create, real_close = LocalProvider.create, LocalProvider.close

    async def create(self: LocalProvider, spec):
        creating.set()
        await go_on.wait()
        return await real_create(self, spec)

    async def close(self: LocalProvider, handle) -> None:
        await real_close(self, handle)
        closed.append(handle)

    monkeypatch.setattr(LocalProvider, "create", create)
    monkeypatch.setattr(LocalProvider, "close", close)
    seed = asyncio.ensure_future(lender.lend("starting", TASK))
    await creating.wait()
    release = asyncio.ensure_future(lender.release_all(timeout_s=5))
    await asyncio.sleep(0)
    go_on.set()

    assert await release == {"lent": [], "failed_seeds": [], "starting": []}
    # The lend that was starting stopped its own sandbox: release_all would not have found it lent.
    with pytest.raises(HTTPException) as refused:
        await seed
    assert refused.value.status_code == 503
    assert len(closed) == 1 and lender.get("starting") is None
    with pytest.raises(HTTPException):
        await lender.lend("later", TASK)


async def test_release_all_cuts_its_stops_at_its_time_limit_and_reports_what_may_still_run(
    workdir, started, monkeypatch, caplog
) -> None:
    lender = _lender(workdir)
    await lender.lend("hangs", TASK)

    async def stop(self: AsyncSandbox) -> None:
        await asyncio.Event().wait()  # a provider's delete that never returns

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    begun = time.monotonic()
    with caplog.at_level(logging.ERROR, logger=sandbox_lending.__name__):
        left = await lender.release_all(timeout_s=0.05)

    # Cut at its own deadline, not at the stop's (STOP_TIMEOUT_S): a server's exit cannot wait for that.
    assert time.monotonic() - begun < 5
    assert left == {"lent": ["hangs"], "failed_seeds": [], "starting": []}
    assert "a reaper must stop them" in caplog.text and "['hangs']" in caplog.text
