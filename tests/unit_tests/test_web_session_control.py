# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock

import pytest

from nemo_gym.web.api_models import WebSeedSessionRequest, WebSessionIdentity
from nemo_gym.web.session import CapacityUnavailableError, SessionConflictError
from nemo_gym.web.session_control import SessionIdentityError, WebSessionControl
from tests.unit_tests.test_web_session_manager import (
    DelayedBrowserSessionProvider,
    FakeBrowserSessionProvider,
    _manager,
    _task,
)


def _identity(index=0):
    return {"_ng_session_id": f"session-{index:032d}", "_ng_session_close_token": f"capability-{index:032d}"}


def _body(index=0):
    return WebSeedSessionRequest(task=_task(), **_identity(index))


@pytest.mark.asyncio
async def test_lost_seed_response_and_concurrent_close_release_once(tmp_path):
    provider = DelayedBrowserSessionProvider()
    manager, backends = _manager(tmp_path, browser_session_provider=provider)
    control = WebSessionControl(manager, lifetime_seconds=3600)
    waiter = asyncio.create_task(control.seed(_body()))
    await provider.started.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    closers = [asyncio.create_task(control.close(WebSessionIdentity(**_identity()))) for _ in range(10)]
    await asyncio.sleep(0)
    assert not any(task.done() for task in closers)
    provider.finish.set()
    assert all(await asyncio.gather(*closers))
    assert len(provider.acquired) == len(provider.released) == 1
    assert backends[0].close_calls == 1
    assert (await manager.health())["sessions"] == 0
    with pytest.raises(SessionConflictError):
        await control.seed(_body())


@pytest.mark.asyncio
async def test_seed_replay_conflict_capability_and_close_before_seed(tmp_path):
    manager, backends = _manager(tmp_path)
    control = WebSessionControl(manager, lifetime_seconds=3600)
    results = await asyncio.gather(*(control.seed(_body()) for _ in range(10)))
    assert all(result == results[0] for result in results)
    assert len(backends) == 1
    conflict = _body().model_copy(deep=True)
    conflict.task.intent = "different task content"
    with pytest.raises(SessionConflictError, match="different seed payload"):
        await control.seed(conflict)
    bad = {**_identity(), "_ng_session_close_token": "incorrect-capability-000000000000000000"}
    with pytest.raises(SessionIdentityError):
        await control.close(WebSessionIdentity(**bad))
    assert (await manager.health())["sessions"] == 1
    assert await control.close(WebSessionIdentity(**_identity()))
    assert await control.close(WebSessionIdentity(**_identity()))
    assert await control.close(WebSessionIdentity(**_identity(1)))
    with pytest.raises(SessionConflictError):
        await control.seed(_body(1))
    assert len(backends) == 1


@pytest.mark.asyncio
async def test_failed_release_stays_retryable_and_holds_capacity(tmp_path):
    provider = FakeBrowserSessionProvider()
    manager, backends = _manager(tmp_path, browser_session_provider=provider, max_sessions=1)
    control = WebSessionControl(manager, lifetime_seconds=3600)
    await control.seed(_body())
    release = provider.release
    provider.release = AsyncMock(side_effect=RuntimeError("provider unavailable"))
    assert not await control.close(WebSessionIdentity(**_identity()))
    assert (await manager.health())["cleanup_pending"] == 1
    provider.release = release
    assert await control.close(WebSessionIdentity(**_identity()))
    assert (await manager.health())["cleanup_pending"] == 0
    assert len(provider.released) == 1
    assert backends[0].close_calls == 1
    await control.seed(_body(1))
    assert await control.close(WebSessionIdentity(**_identity(1)))


@pytest.mark.asyncio
async def test_64_cancelled_callers_do_not_exhaust_capacity(tmp_path):
    provider = DelayedBrowserSessionProvider()
    manager, _ = _manager(tmp_path, browser_session_provider=provider, max_sessions=64)
    control = WebSessionControl(manager, lifetime_seconds=3600)
    waiters = [asyncio.create_task(control.seed(_body(i))) for i in range(64)]
    for _ in range(100):
        if len(provider.acquired) == 64:
            break
        await asyncio.sleep(0)
    assert len(provider.acquired) == 64
    for waiter in waiters:
        waiter.cancel()
    await asyncio.gather(*waiters, return_exceptions=True)
    provider.finish.set()
    assert all(await asyncio.gather(*(control.close(WebSessionIdentity(**_identity(i))) for i in range(64))))
    health = await manager.health()
    assert health["sessions"] == health["creating"] == health["cleanup_pending"] == 0
    assert health["browser_provider"]["active_leases"] == 0
    assert len(provider.acquired) == len(provider.released) == 64
    await control.seed(_body(64))
    assert await control.close(WebSessionIdentity(**_identity(64)))


def test_capability_is_not_in_repr_or_json():
    body = _body()
    token = _identity()["_ng_session_close_token"]
    assert token not in repr(body)
    assert token not in body.model_dump_json()


@pytest.mark.asyncio
async def test_identity_expiry_does_not_truncate_an_active_rollout(tmp_path):
    manager, _ = _manager(tmp_path)
    control = WebSessionControl(manager, lifetime_seconds=3600)
    await control.seed(_body())
    record = control._records[_identity()["_ng_session_id"]]
    record.created_at -= 7200
    await control._reap_once()
    assert not record.closing
    assert (await manager.health())["sessions"] == 1
    assert await control.close(WebSessionIdentity(**_identity()))


@pytest.mark.asyncio
async def test_late_provider_release_failure_is_not_a_successful_close(tmp_path):
    provider = DelayedBrowserSessionProvider()
    manager, _ = _manager(
        tmp_path, browser_session_provider=provider, browser_acquire_timeout_seconds=0.01, max_sessions=1
    )
    control = WebSessionControl(manager, lifetime_seconds=3600)
    with pytest.raises(CapacityUnavailableError):
        await control.seed(_body())
    identity = WebSessionIdentity(**_identity())
    assert not await control.close(identity)
    with pytest.raises(CapacityUnavailableError):
        await control.seed(_body(1))
    release = provider.release
    provider.release = AsyncMock(side_effect=RuntimeError("release failed"))
    provider.finish.set()
    await asyncio.gather(*tuple(manager._late_browser_cleanup_tasks))
    assert not await control.close(identity)
    assert (await manager.health())["late_release_pending"] == 1
    assert control._records[identity.session_identity].closed_at is None
    provider.release = release
    assert all(await asyncio.gather(*(control.close(identity) for _ in range(10))))
    assert (await manager.health())["browser_provider"]["active_leases"] == 0
    assert len(provider.released) == 1
    await control.seed(_body(1))
    assert await control.close(WebSessionIdentity(**_identity(1)))


@pytest.mark.asyncio
async def test_expired_seed_cannot_hide_failed_release(tmp_path):
    provider = FakeBrowserSessionProvider()
    manager, _ = _manager(tmp_path, browser_session_provider=provider)
    control = WebSessionControl(manager, lifetime_seconds=3600)
    await control.seed(_body())
    release = provider.release
    provider.release = AsyncMock(side_effect=RuntimeError("release failed"))
    assert not await manager.close_session(_identity()["_ng_session_id"])
    from nemo_gym.web.session import SessionNotFoundError

    with pytest.raises(SessionNotFoundError):
        await control.seed(_body())
    assert not await control.close(WebSessionIdentity(**_identity()))
    provider.release = release
    assert await control.close(WebSessionIdentity(**_identity()))


@pytest.mark.asyncio
async def test_close_racing_an_admission_retry_does_not_create_another_browser(tmp_path):
    manager, backends = _manager(tmp_path, max_sessions=1)
    control = WebSessionControl(manager, lifetime_seconds=3600)
    await control.seed(_body())
    with pytest.raises(CapacityUnavailableError):
        await control.seed(_body(1))
    assert await control.close(WebSessionIdentity(**_identity()))
    checking, resume = asyncio.Event(), asyncio.Event()

    async def admission(_session_id):
        checking.set()
        await resume.wait()
        return True

    manager.can_retry_seed = admission
    retry = asyncio.create_task(control.seed(_body(1)))
    await checking.wait()
    assert await control.close(WebSessionIdentity(**_identity(1)))
    resume.set()
    with pytest.raises(SessionConflictError, match="closed while waiting for admission"):
        await retry
    assert len(backends) == 1


@pytest.mark.asyncio
async def test_registry_capacity_prunes_only_expired_tombstones(tmp_path):
    manager, _ = _manager(tmp_path)
    control = WebSessionControl(manager, lifetime_seconds=3600, max_records=1)
    with pytest.raises(SessionIdentityError):
        await control.close(WebSessionIdentity())
    await control.seed(_body())
    with pytest.raises(CapacityUnavailableError, match="identity capacity"):
        await control.seed(_body(1))
    await manager.evaluate(_identity()["_ng_session_id"])
    with pytest.raises(SessionConflictError, match="status='evaluated'"):
        await control.seed(_body())
    assert await control.close(WebSessionIdentity(**_identity()))
    record = control._records[_identity()["_ng_session_id"]]
    assert record.seed is None  # Do not retain screenshots for the whole tombstone TTL.
    record.closed_at -= 7200
    await control.seed(_body(1))
    assert _identity()["_ng_session_id"] not in control._records
    await control.stop(timeout=1)
    assert (await manager.health())["sessions"] == 0


@pytest.mark.asyncio
async def test_expiry_and_bounded_shutdown_keep_ownership_of_an_unfinished_seed(tmp_path, caplog):
    provider = DelayedBrowserSessionProvider()
    manager, _ = _manager(tmp_path, browser_session_provider=provider)
    control = WebSessionControl(manager, lifetime_seconds=0.01)
    waiter = asyncio.create_task(control.seed(_body()))
    await provider.started.wait()
    control.start()
    record = control._records[_identity()["_ng_session_id"]]

    async def wait_for_expiry():
        while not record.closing:
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait_for_expiry(), timeout=1)
    await control.stop(timeout=0.001)
    assert "web_session_shutdown_cleanup_pending" in caplog.text
    assert record.close is not None and not record.close.done()
    provider.finish.set()
    with pytest.raises(SessionConflictError, match="seed was in flight"):
        await waiter
    assert await record.close
    assert len(provider.released) == 1
    assert (await manager.health())["browser_provider"]["active_leases"] == 0


@pytest.mark.asyncio
async def test_expiry_release_retries_are_bounded_but_explicit_close_can_recover(tmp_path):
    provider = FakeBrowserSessionProvider()
    manager, _ = _manager(tmp_path, browser_session_provider=provider)
    control = WebSessionControl(manager, lifetime_seconds=3600)
    await control.seed(_body())
    identity = WebSessionIdentity(**_identity())
    release = provider.release
    provider.release = AsyncMock(side_effect=RuntimeError("release failed"))
    assert not await control.close(identity)
    record = control._records[identity.session_identity]
    record.created_at -= 7200
    for _ in range(3):
        await control._reap_once()
        assert not await record.close
    calls = provider.release.await_count
    await control._reap_once()
    assert record.expiry_close_attempts == 3
    assert provider.release.await_count == calls
    assert (await manager.health())["cleanup_pending"] == 1
    provider.release = release
    assert await control.close(identity)
    assert (await manager.health())["cleanup_pending"] == 0
