# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import AsyncMock

import pytest

from nemo_gym.agent_utils.sandbox_observation import lifecycle_observation
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.trajectory.trajectory_record import SandboxObservation
from tests.unit_tests.test_sandbox import FakeSandboxProvider
from tests.unit_tests.test_sandbox_connect import FakeConnectableProvider


pytestmark = pytest.mark.sandbox


def _observed(sandbox: AsyncSandbox) -> SandboxObservation:
    return SandboxObservation.model_validate(lifecycle_observation(sandbox, role="verifier"))


async def test_the_interval_needs_both_the_provider_create_and_close_returns() -> None:
    sandbox = AsyncSandbox(FakeSandboxProvider())
    assert lifecycle_observation(sandbox, role="verifier") is None
    await sandbox.start(SandboxSpec(image="task"))
    started = _observed(sandbox)
    assert (started.observed_client_interval_s, started.teardown_confirmed, started.outcome) == (None, None, "unknown")
    await sandbox.stop()
    stopped = _observed(sandbox)
    assert stopped.observed_client_interval_s == sandbox.observed_client_interval_s >= 0
    assert (stopped.teardown_confirmed, stopped.cleanup_errors, stopped.outcome) == (True, [], "unknown")
    # A client-side interval is not remote lifetime or usage.
    assert (stopped.wall_time_s, stopped.cpu_time_s, stopped.peak_memory_mib) == (None, None, None)


@pytest.mark.parametrize("error", [RuntimeError("create failed"), asyncio.CancelledError()])
async def test_a_failed_or_cancelled_create_leaves_nothing_to_observe(error) -> None:
    provider = FakeSandboxProvider()
    provider.create = AsyncMock(side_effect=error)
    sandbox = AsyncSandbox(provider)
    with pytest.raises(type(error)):
        await sandbox.start(SandboxSpec(image="task"))
    assert lifecycle_observation(sandbox, role="verifier") is None


async def test_a_connected_handle_has_no_observed_create() -> None:
    owner = AsyncSandbox(FakeConnectableProvider())
    await owner.start(SandboxSpec(image="task"))
    borrowed = await AsyncSandbox.connect(await owner.serialize(), provider=FakeConnectableProvider())
    assert (_observed(borrowed).observed_client_interval_s, _observed(borrowed).teardown_confirmed) == (None, None)
    await borrowed.stop()
    # Teardown is confirmed, but this instance never saw the create, so no interval is reported.
    assert (_observed(borrowed).observed_client_interval_s, _observed(borrowed).teardown_confirmed) == (None, True)


@pytest.mark.parametrize("error", [TimeoutError(), asyncio.CancelledError()])
async def test_a_successful_retry_keeps_the_earlier_close_failure(error) -> None:
    provider = FakeSandboxProvider()
    provider.close = AsyncMock(side_effect=[error, None])
    sandbox = AsyncSandbox(provider)
    await sandbox.start(SandboxSpec(image="task"))
    with pytest.raises(type(error)):
        await sandbox.stop()
    failed = _observed(sandbox)
    name = type(error).__name__
    assert (failed.outcome, failed.teardown_confirmed, failed.cleanup_errors) == (
        "sandbox_error",
        False,
        [f"close:{name}"],
    )
    assert failed.observed_client_interval_s is None
    await sandbox.stop()
    retried = _observed(sandbox)
    assert (retried.teardown_confirmed, retried.cleanup_errors, retried.error_type) == (
        True,
        [f"close:{name}"],
        f"close:{name}",
    )
    assert retried.outcome == "unknown" and provider.close.await_count == 2


async def test_a_client_release_failure_after_confirmed_teardown_is_kept_separately() -> None:
    provider = FakeSandboxProvider()
    provider.aclose = AsyncMock(side_effect=[OSError("client"), None])
    sandbox = AsyncSandbox(provider, owns_provider=True)
    await sandbox.start(SandboxSpec(image="task"))
    with pytest.raises(OSError):
        await sandbox.stop()
    observed = _observed(sandbox)
    assert (observed.teardown_confirmed, observed.cleanup_errors, observed.outcome) == (
        True,
        ["aclose:OSError"],
        "unknown",
    )
    assert observed.observed_client_interval_s is not None
