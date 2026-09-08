# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Modal provider failure and cancellation paths."""

import asyncio

import pytest

from tests.unit_tests.test_modal_provider import FakeProcess, _Aio, _provider, _spec
from tests.unit_tests.test_modal_provider import fake_modal as fake_modal


pytestmark = pytest.mark.sandbox


async def test_timeout_sentinel_does_not_depend_on_client_wall_clock(fake_modal):
    _, created = fake_modal()
    provider = _provider(probe={"command": None})
    handle = await provider.create(_spec())
    created[0].exec_script = {"slow": FakeProcess(returncode=-1)}
    with pytest.raises(TimeoutError):
        await provider.exec(handle, "slow", timeout_s=30)


async def test_cancelled_readiness_terminates_the_allocated_sandbox(fake_modal):
    _, created = fake_modal(exec_script={"printf ok": FakeProcess(wait_delay=30)})
    provider = _provider()
    task = asyncio.create_task(provider.create(_spec()))
    while not created:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert created[0].terminated == 1


async def test_readiness_deadline_bounds_a_stalled_probe(fake_modal):
    _, created = fake_modal(exec_script={"printf ok": FakeProcess(wait_delay=30)})
    provider = _provider(probe={"deadline_s": 0.03})
    from nemo_gym.sandbox.providers.modal import ModalCreateVerificationError

    with pytest.raises(ModalCreateVerificationError):
        async with asyncio.timeout(0.5):
            await provider.create(_spec())
    assert created[0].terminated == 1


async def test_invalid_utf8_is_replaced_in_command_output(fake_modal):
    _, created = fake_modal()
    provider = _provider(probe={"command": None})
    handle = await provider.create(_spec())
    created[0].exec_script = {"binary": FakeProcess(stdout=b"ok\xff", stderr=b"\xfe")}
    result = await provider.exec(handle, "binary")
    assert result.stdout == "ok\ufffd"
    assert result.stderr == "\ufffd"
    assert created[0].exec_calls[-1][1]["text"] is False


async def test_default_tunnel_uses_modal_tls_termination(fake_modal):
    _, created = fake_modal()
    await _provider(probe={"command": None}).create(_spec(ports=[8080]))
    assert created[0].create_kwargs["encrypted_ports"] == [8080]
    assert "unencrypted_ports" not in created[0].create_kwargs


async def test_cancellation_while_allocation_is_inflight_reconciles_the_returned_id(fake_modal):
    modal, created = fake_modal()
    original = modal.Sandbox.create
    submitted, release = asyncio.Event(), asyncio.Event()

    async def delayed_create(*args, **kwargs):
        submitted.set()
        await release.wait()
        return await original.aio(*args, **kwargs)

    modal.Sandbox.create = _Aio(delayed_create)
    task = asyncio.create_task(_provider().create(_spec()))
    await submitted.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(created) == 1
    assert created[0].terminated == 1 and created[0].detached == 1


async def test_cancelled_exec_discards_sandbox_and_cancels_local_readers(fake_modal):
    _, created = fake_modal()
    provider = _provider(probe={"command": None})
    handle = await provider.create(_spec())
    read_started = asyncio.Event()
    cancelled = asyncio.Event()
    process = FakeProcess(wait_delay=30)

    async def read():
        read_started.set()
        try:
            await asyncio.sleep(30)
        finally:
            cancelled.set()

    process.stdout.read = _Aio(read)
    created[0].exec_script = {"slow": process}
    task = asyncio.create_task(provider.exec(handle, "slow"))
    await read_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set() and handle.raw is None
    assert created[0].terminated == 1


async def test_transfer_deadline_discards_the_partial_writer(fake_modal):
    _, created = fake_modal()
    provider = _provider(probe={"command": None}, files={"transfer_timeout_s": 0.02})
    handle = await provider.create(_spec())

    async def hang(*args):
        await asyncio.sleep(30)

    created[0].filesystem.write_bytes = _Aio(hang)
    with pytest.raises(TimeoutError):
        await provider.write_file(handle, "/work/out", b"partial")
    assert created[0].terminated == 1 and handle.raw is None


async def test_close_failure_retains_handle_for_cleanup_retry(fake_modal):
    _, created = fake_modal()
    provider = _provider(probe={"command": None})
    handle = await provider.create(_spec())
    created[0].terminate_exc = ValueError("not a transient error")
    with pytest.raises(ValueError):
        await provider.close(handle)
    assert handle.raw is not None and created[0].detached == 0
    created[0].terminate_exc = None
    await provider.close(handle)
    assert handle.raw is None


async def test_exec_is_submitted_once_on_ambiguous_transport_error(fake_modal):
    modal, created = fake_modal()
    provider = _provider(probe={"command": None}, operations={"retries": 3})
    handle = await provider.create(_spec())
    created[0].exec_script = {"mutate": modal.exception.ServiceError("connection lost")}
    with pytest.raises(modal.exception.ServiceError):
        await provider.exec(handle, "mutate")
    assert len(created[0].exec_calls) == 1


async def test_only_transient_idempotent_operations_are_retried(fake_modal):
    modal, created = fake_modal()
    provider = _provider(probe={"command": None}, operations={"retries": 1})
    handle = await provider.create(_spec())
    calls = 0

    async def flaky_terminate(*, wait):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise modal.exception.ServiceError("temporary failure")
        assert wait

    created[0].terminate = _Aio(flaky_terminate)
    await provider.close(handle)
    assert calls == 2


@pytest.mark.parametrize("user", [0, "-root", ""])
async def test_unsupported_users_fail_before_exec(fake_modal, user):
    _, created = fake_modal()
    provider = _provider(probe={"command": None}, exec={"allow_user_rewrite": True})
    handle = await provider.create(_spec())
    with pytest.raises(ValueError, match="username"):
        await provider.exec(handle, "id", user=user)
    assert not created[0].exec_calls


@pytest.mark.parametrize("ttl", [-1, 0, 86401, float("inf"), float("nan")])
async def test_unsupported_lifetimes_fail_before_allocation(fake_modal, ttl):
    _, created = fake_modal()
    with pytest.raises(RuntimeError, match="ttl_s"):
        await _provider().create(_spec(ttl_s=ttl))
    assert not created


async def test_tag_limit_includes_merged_defaults_and_rejects_before_allocation(fake_modal):
    _, created = fake_modal()
    with pytest.raises(RuntimeError, match="at most 10 tags"):
        await _provider().create(_spec(metadata={f"key-{i}": "value" for i in range(10)}))
    assert not created
    await _provider(probe={"command": None}).create(_spec(metadata={f"key-{i}": "value" for i in range(9)}))
    assert len(created[0].create_kwargs["tags"]) == 10


@pytest.mark.parametrize(
    "config",
    [
        {"connection": {"app_name": ""}},
        {"connection": []},
        {"create": {"keepalive_shell": []}},
        {"create": {"keepalive_cmd": ""}},
        {"create": {"default_tags": []}},
        {"create": {"secrets": [""]}},
        {"create": {"secrets": 42}},
        {"probe": {"command": ""}},
        {"probe": {"stable_count": False}},
        {"exec": {"shell": []}},
        {"files": {"transfer_timeout_s": -1}},
    ],
)
def test_invalid_configuration_fails_before_allocating(config, fake_modal):
    _, created = fake_modal()
    from nemo_gym.sandbox.providers.modal import ModalProvider

    with pytest.raises((ValueError, TypeError)):
        ModalProvider(**config)
    assert not created


@pytest.mark.parametrize(
    "spec_args",
    [
        {"provider_options": []},
        {"provider_options": {"gpu": ""}},
        {"provider_options": {"image_secret": ""}},
        {"provider_options": {"block_network": "false"}},
        {"provider_options": {"volumes": "bad"}},
        {"provider_options": {"tags": "bad"}},
        {"provider_options": {"name": ""}},
        {"ready_timeout_s": 0},
    ],
)
async def test_invalid_spec_options_fail_before_allocating(spec_args, fake_modal):
    _, created = fake_modal()
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        await _provider().create(_spec(**spec_args))
    assert not created


async def test_blocked_network_with_ports_fails_before_allocating(fake_modal):
    _, created = fake_modal()
    with pytest.raises(RuntimeError, match="block_network"):
        await _provider().create(_spec(ports=[8080], provider_options={"block_network": True}))
    assert not created


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("encrypted", "https://tls.example.test"),
        ("h2", "https://tls.example.test"),
        ("unencrypted", "http://tcp.example.test:4567"),
    ],
)
async def test_endpoint_uses_real_sdk_tunnel_semantics(fake_modal, mode, expected):
    from modal import Tunnel

    _, created = fake_modal()
    provider = _provider(probe={"command": None}, create={"port_mode": mode})
    handle = await provider.create(_spec(ports=[8080]))
    created[0].tunnel_map = {
        8080: Tunnel("tls.example.test", 443, "tcp.example.test" if mode == "unencrypted" else "", 4567)
    }
    assert (await provider.endpoint(handle, 8080)).endpoint == expected
    other = _provider(probe={"command": None})
    connected = await other.connect(await provider.serialize_handle(handle))
    assert (await other.endpoint(connected, 8080)).endpoint == expected


async def test_file_bytes_helpers_round_trip(fake_modal):
    _, created = fake_modal()
    provider = _provider(probe={"command": None})
    handle = await provider.create(_spec())
    await provider.write_file(handle, "/work/unicode", "caf\u00e9")
    assert await provider.read_file(handle, "/work/unicode") == "caf\u00e9".encode()


async def test_missing_or_unavailable_tunnel_raises_without_fabricating_a_url(fake_modal):
    modal, created = fake_modal()
    provider = _provider(probe={"command": None})
    handle = await provider.create(_spec(ports=[8080]))
    with pytest.raises(RuntimeError, match="no tunnel"):
        await provider.endpoint(handle, 8080)

    async def failed_tunnels(**kwargs):
        raise modal.exception.AuthError("denied")

    created[0].tunnels = _Aio(failed_tunnels)
    with pytest.raises(RuntimeError, match="denied"):
        await provider.endpoint(handle, 8080)


async def test_exhausted_cleanup_retries_keep_the_original_error_and_handle(fake_modal):
    modal, created = fake_modal()
    provider = _provider(probe={"command": None}, operations={"retries": 1})
    handle = await provider.create(_spec())
    failure = modal.exception.ServiceError("service unavailable")
    created[0].terminate_exc = failure
    with pytest.raises(modal.exception.ServiceError) as caught:
        await provider.close(handle)
    assert caught.value is failure and handle.raw is not None
    assert created[0].terminated == 2


async def test_missing_file_bytes_uses_python_file_not_found(fake_modal):
    modal, created = fake_modal()
    provider = _provider(probe={"command": None})
    handle = await provider.create(_spec())
    created[0].filesystem.error = modal.exception.SandboxFilesystemNotFoundError("missing")
    with pytest.raises(FileNotFoundError):
        await provider.read_file(handle, "/missing")


def test_missing_sdk_has_actionable_install_error(monkeypatch):
    import sys

    from nemo_gym.sandbox.providers.modal._sdk import require_modal_sdk

    monkeypatch.setitem(sys.modules, "modal", None)
    with pytest.raises(ImportError, match=r"modal>=1\.5\.5,<2\.0\.0"):
        require_modal_sdk("test")


def test_broken_transitive_dependency_preserves_its_actual_name(monkeypatch):
    import builtins

    from nemo_gym.sandbox.providers.modal._sdk import require_modal_sdk

    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "modal":
            raise ModuleNotFoundError("broken dependency", name="broken_dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(ModuleNotFoundError) as caught:
        require_modal_sdk("test")
    assert caught.value.name == "broken_dependency"


async def test_closed_handle_status_works_with_real_sdk_without_authentication():
    from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxStatus
    from nemo_gym.sandbox.providers.modal import ModalProvider

    handle = SandboxHandle(sandbox_id="already-closed", provider_name="modal", raw=None)
    assert await ModalProvider().status(handle) is SandboxStatus.STOPPED
