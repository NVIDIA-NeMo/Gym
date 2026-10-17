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
    result = await provider.exec(handle, "slow", timeout_s=30)
    assert result.error_type == "timeout" and result.return_code == 125


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
    result = await provider.exec(handle, "mutate")
    assert result.error_type == "sandbox" and result.return_code == 125
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


@pytest.mark.parametrize("user", [1000, "1000", "-root", "", False])
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
    modal, created = fake_modal()
    with pytest.raises(RuntimeError, match="at most 10 tags"):
        await _provider().create(_spec(metadata={f"key-{i}": "value" for i in range(10)}))
    assert not created and not modal.App.lookups
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
        {"exec": {"shell": "/bin/bash"}},
        {"create": {"keepalive_shell": "/bin/sh"}},
        {"create": {"secrets": ""}},
        {"exec": {"user": "agent"}},
        {"exec": {"user": 1000, "allow_user_rewrite": True}},
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
        {"provider_options": {"volumes": []}},
        {"provider_options": {"volumes": {"relative": "data"}}},
        {"provider_options": {"volumes": {"/data": ""}}},
        {"provider_options": {"idle_timeout_s": -1}},
        {"provider_options": {"tags": "bad"}},
        {"provider_options": {"name": ""}},
        {"ready_timeout_s": 0},
    ],
)
async def test_invalid_spec_options_fail_before_allocating(spec_args, fake_modal):
    modal, created = fake_modal()
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        await _provider().create(_spec(**spec_args))
    assert not created and not modal.App.lookups


async def test_blocked_network_with_ports_fails_before_allocating(fake_modal):
    modal, created = fake_modal()
    with pytest.raises(RuntimeError, match="block_network"):
        await _provider().create(_spec(ports=[8080], provider_options={"block_network": True}))
    assert not created and not modal.App.lookups


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("encrypted", "https://tls.example.test"),
        ("h2", "https://tls.example.test"),
        ("unencrypted", "http://tcp.example.test:4567"),
    ],
)
async def test_endpoint_uses_real_sdk_tunnel_semantics(fake_modal, mode, expected):
    Tunnel = pytest.importorskip("modal").Tunnel

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
    pytest.importorskip("modal")
    from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxStatus
    from nemo_gym.sandbox.providers.modal import ModalProvider

    handle = SandboxHandle(sandbox_id="already-closed", provider_name="modal", raw=None)
    assert await ModalProvider().status(handle) is SandboxStatus.STOPPED


@pytest.mark.parametrize("user", [None, "root", 0])
@pytest.mark.parametrize("allow_rewrite", [False, True])
async def test_root_exec_needs_no_su(fake_modal, user, allow_rewrite):
    _, created = fake_modal()
    provider = _provider(probe={"command": None}, exec={"allow_user_rewrite": allow_rewrite})
    handle = await provider.create(_spec())
    await provider.exec(handle, "id", user=user)
    assert created[0].exec_calls[-1][0] == ("/bin/sh", "-c", "id")


@pytest.mark.parametrize("user", ["root", 0])
async def test_default_root_user_allows_readiness(fake_modal, user):
    _, created = fake_modal()
    await _provider(exec={"user": user}).create(_spec())
    assert created[0].exec_calls[-1][0] == ("/bin/sh", "-c", "printf ok")


async def test_user_rewrite_preserves_login_shell_flags_and_command_quoting(fake_modal):
    import shlex

    _, created = fake_modal()
    provider = _provider(
        probe={"command": None},
        exec={
            "user": "agent",
            "allow_user_rewrite": True,
            "shell": ["/bin/bash", "-lc"],
        },
    )
    handle = await provider.create(_spec())
    command = "printf '%s' \"a b\"; echo '$HOME'"
    await provider.exec(handle, command)
    argv = created[0].exec_calls[-1][0]
    assert argv[:4] == ("su", "-s", "/bin/sh", "-c") and argv[5:] == ("--", "agent")
    assert shlex.split(argv[4]) == ["/bin/bash", "-lc", command]
    await provider.exec(handle, command, user="root")
    assert created[0].exec_calls[-1][0] == ("/bin/bash", "-lc", command)


async def test_string_secret_is_one_secret_name(fake_modal):
    _, created = fake_modal()
    await _provider(create={"secrets": "hf-token"}).create(_spec())
    assert [secret.name for secret in created[0].create_kwargs["secrets"]] == ["hf-token"]


@pytest.mark.parametrize("error", ["AuthError", "NotFoundError", "ServiceError"])
async def test_app_lookup_failure_is_a_create_error(fake_modal, error):
    from nemo_gym.sandbox.providers.modal import ModalCreateError

    modal, created = fake_modal()
    failure = getattr(modal.exception, error)("app unavailable")

    async def fail(*args, **kwargs):
        raise failure

    modal.App.lookup = _Aio(fail)
    with pytest.raises(ModalCreateError, match="app unavailable") as caught:
        await _provider(connection={"create_app_if_missing": False}).create(_spec())
    assert caught.value.__cause__ is failure and not created


@pytest.mark.parametrize("stage", ["lookup", "allocation"])
async def test_readiness_deadline_bounds_lookup_and_allocation(fake_modal, stage, caplog):
    from nemo_gym.sandbox.providers.modal import ModalCreateError

    modal, created = fake_modal()
    stopped = asyncio.Event()

    async def hang(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    if stage == "lookup":
        modal.App.lookup = _Aio(hang)
    else:
        modal.Sandbox.create = _Aio(hang)
    provider = _provider(probe={"command": None}, operations={"close_timeout_s": 0.02})
    async with asyncio.timeout(0.5):
        with pytest.raises(ModalCreateError, match="creation exceeded ready timeout"):
            await provider.create(_spec(ready_timeout_s=0.02))
    assert stopped.is_set() and not created
    if stage == "allocation":
        assert "remote TTL remains the cleanup backstop" in caplog.text


async def test_cancelled_stalled_allocation_is_bounded(fake_modal):
    modal, _ = fake_modal()
    started, stopped = asyncio.Event(), asyncio.Event()

    async def hang(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    modal.Sandbox.create = _Aio(hang)
    task = asyncio.create_task(_provider(operations={"close_timeout_s": 0.02}).create(_spec()))
    await started.wait()
    task.cancel()
    async with asyncio.timeout(0.5):
        with pytest.raises(asyncio.CancelledError):
            await task
    assert stopped.is_set()


async def test_allocation_completing_after_deadline_is_terminated(fake_modal):
    from nemo_gym.sandbox.providers.modal import ModalCreateError

    modal, created = fake_modal()
    original = modal.Sandbox.create

    async def delayed(*args, **kwargs):
        await asyncio.sleep(0.03)
        return await original.aio(*args, **kwargs)

    modal.Sandbox.create = _Aio(delayed)
    async with asyncio.timeout(0.5):
        with pytest.raises(ModalCreateError, match="ready timeout"):
            await _provider(operations={"close_timeout_s": 0.2}).create(_spec(ready_timeout_s=0.01))
    assert created[0].terminated == 1 and created[0].detached == 1


async def test_probe_uses_remaining_creation_budget_and_preserves_diagnostics(fake_modal):
    from nemo_gym.sandbox.providers.modal import ModalCreateVerificationError

    modal, created = fake_modal(exec_script={"printf ok": FakeProcess(stderr="/bin/sh missing", returncode=127)})
    original = modal.Sandbox.create
    allocation_finished = None

    async def delayed(*args, **kwargs):
        nonlocal allocation_finished
        await asyncio.sleep(0.08)
        allocation_finished = asyncio.get_running_loop().time()
        return await original.aio(*args, **kwargs)

    modal.Sandbox.create = _Aio(delayed)
    with pytest.raises(ModalCreateVerificationError, match="return_code=127.*sh missing"):
        await _provider(probe={"stable_delay_s": 1}).create(_spec(ready_timeout_s=0.12))
    assert asyncio.get_running_loop().time() - allocation_finished < 0.10
    assert created[0].terminated == 1


@pytest.mark.parametrize("error", [ValueError, TypeError])
async def test_probe_programming_errors_fail_immediately(fake_modal, error):
    _, created = fake_modal(exec_script={"printf ok": error("bad exec config")})
    with pytest.raises(error, match="bad exec config"):
        async with asyncio.timeout(0.5):
            await _provider().create(_spec())
    assert len(created[0].exec_calls) == 1 and created[0].terminated == 1


@pytest.mark.parametrize("operation", ["exec", "transfer"])
@pytest.mark.parametrize("cleanup_failure", ["error", "timeout"])
async def test_cleanup_failure_preserves_client_timeout(fake_modal, operation, cleanup_failure, caplog):
    _, created = fake_modal()
    provider = _provider(probe={"command": None}, operations={"close_timeout_s": 0.02})
    handle = await provider.create(_spec())

    async def hang(*args, **kwargs):
        await asyncio.Event().wait()

    if cleanup_failure == "error":
        created[0].terminate_exc = ValueError("cleanup failed")
    else:
        created[0].terminate = _Aio(hang)
    created[0].exec_script = {"slow": FakeProcess(wait_delay=30)}
    created[0].filesystem.write_bytes = _Aio(hang)
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.02):
            if operation == "exec":
                await provider.exec(handle, "slow")
            else:
                await provider.write_file(handle, "/work/out", b"partial")
    assert handle.raw is not None and "cleanup after failure raised" in caplog.text


async def test_file_deadline_preserves_original_error_if_cleanup_fails(fake_modal):
    _, created = fake_modal()
    provider = _provider(probe={"command": None}, files={"transfer_timeout_s": 0.01})
    handle = await provider.create(_spec())
    created[0].terminate_exc = ValueError("cleanup failed")

    async def hang(*args):
        await asyncio.Event().wait()

    created[0].filesystem.write_bytes = _Aio(hang)
    with pytest.raises(TimeoutError):
        await provider.write_file(handle, "/out", b"partial")
    assert handle.raw is not None


async def test_exec_stdin_eof_is_drained_before_a_reader_can_finish(fake_modal):
    _, created = fake_modal()
    provider = _provider(probe={"command": None})
    handle = await provider.create(_spec())
    process = FakeProcess()
    drained = asyncio.Event()

    async def drain():
        assert process.stdin.eof
        drained.set()

    async def wait_for_eof():
        await drained.wait()
        return 0

    process.stdin.drain = _Aio(drain)
    process.wait = _Aio(wait_for_eof)
    created[0].exec_script = {"cat": process}
    async with asyncio.timeout(0.5):
        result = await provider.exec(handle, "cat")
    assert result.return_code == 0 and drained.is_set()


@pytest.mark.parametrize("extra", [0, 1])
async def test_exec_argv_limit_includes_shell_arguments(fake_modal, extra):
    _, created = fake_modal()
    provider = _provider(probe={"command": None})
    handle = await provider.create(_spec())
    command = "x" * (65536 - len("/bin/sh") - len("-c") + extra)
    result = await provider.exec(handle, command)
    if extra:
        assert result.return_code == 125 and result.error_type == "sandbox"
        assert "65,536" in result.stderr and "upload a script" in result.stderr
        assert not created[0].exec_calls
    else:
        assert result.return_code == 0 and len(created[0].exec_calls) == 1


@pytest.mark.parametrize("error", ["ExecTimeoutError", "NotFoundError"])
async def test_output_chunks_survive_a_stream_failure(fake_modal, error):
    modal, created = fake_modal()
    provider = _provider(probe={"command": None})
    handle = await provider.create(_spec())

    class PartialStream:
        async def __aiter__(self):
            yield b"caf\xc3"
            yield b"\xa9"
            raise getattr(modal.exception, error)("stream ended")

    process = FakeProcess(wait_delay=30)
    wait_cancelled, reader_cancelled = asyncio.Event(), asyncio.Event()

    async def wait():
        try:
            await asyncio.Event().wait()
        finally:
            wait_cancelled.set()

    async def read():
        try:
            await asyncio.Event().wait()
        finally:
            reader_cancelled.set()

    process.wait = _Aio(wait)
    process.stderr.read = _Aio(read)
    process.stdout = PartialStream()
    created[0].exec_script = {"work": process}
    async with asyncio.timeout(0.5):
        result = await provider.exec(handle, "work")
    assert wait_cancelled.is_set() and reader_cancelled.is_set()
    assert result.stdout == "caf\u00e9" and result.return_code == 125
    assert result.error_type == ("timeout" if error == "ExecTimeoutError" else "sandbox")


@pytest.mark.parametrize("operation", ["exec", "file"])
async def test_borrowed_cancellation_preserves_owner_for_diagnostics(fake_modal, operation):
    _, created = fake_modal()
    provider = _provider(probe={"command": None})
    owner = await provider.create(_spec())
    borrowed = await provider.connect(await provider.serialize_handle(owner))
    created[0].files["/logs"] = b"diagnostics"

    async def hang(*args):
        await asyncio.Event().wait()

    created[0].filesystem.write_bytes = _Aio(hang)
    created[0].exec_script = {"slow": FakeProcess(wait_delay=30)}
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.02):
            if operation == "exec":
                await provider.exec(borrowed, "slow")
            else:
                await provider.write_file(borrowed, "/out", b"partial")
    assert created[0].terminated == 0 and created[0].detached == 0
    assert await provider.read_file(owner, "/logs") == b"diagnostics"
    await provider.close(owner)
    assert created[0].terminated == 1


@pytest.mark.parametrize(
    ("code", "elapsed", "timeout", "expected"),
    [
        (137, 2.1, 2, "timeout"),
        (143, 2, 2, "timeout"),
        (137, 1.9, 2, None),
        (137, 1.5, 1.2, None),
        (137, 2, 1.2, "timeout"),
        (137, 5, 0, None),
        (137, 5, None, None),
        (1, 5, 2, None),
        (0, 5, 2, None),
        (255, 5, 2, None),
    ],
)
async def test_worker_timeout_signal_uses_rounded_deadline(fake_modal, monkeypatch, code, elapsed, timeout, expected):
    from nemo_gym.sandbox.providers.modal import provider as module

    _, created = fake_modal()
    provider = _provider(probe={"command": None}, exec={"default_timeout_s": None})
    handle = await provider.create(_spec())
    created[0].exec_script = {"work": FakeProcess(stdout="partial", returncode=code)}
    clock = iter([10, 10 + elapsed])
    monkeypatch.setattr(module, "monotonic", lambda: next(clock), raising=False)
    result = await provider.exec(handle, "work", timeout_s=timeout)
    assert result.error_type == expected and result.stdout == "partial"
    assert result.return_code == (125 if expected else code)


async def test_slow_output_drain_does_not_turn_early_signal_into_timeout(fake_modal, monkeypatch):
    from nemo_gym.sandbox.providers.modal import provider as module

    _, created = fake_modal()
    provider = _provider(probe={"command": None})
    handle = await provider.create(_spec())
    now = [0]
    exited = asyncio.Event()
    process = FakeProcess(returncode=137)

    async def wait():
        now[0] = 1
        exited.set()
        return 137

    async def read():
        await exited.wait()
        now[0] = 10
        return "partial"

    process.wait = _Aio(wait)
    process.stdout.read = _Aio(read)
    created[0].exec_script = {"work": process}
    monkeypatch.setattr(module, "monotonic", lambda: now[0])
    result = await provider.exec(handle, "work", timeout_s=2)
    assert result.return_code == 137 and result.error_type is None


@pytest.mark.parametrize("error", ["ConflictError", "NotFoundError"])
@pytest.mark.parametrize("state", ["wait_done", "poll_exited", "running"])
async def test_stdin_close_race_respects_process_exit(fake_modal, error, state):
    modal, created = fake_modal()
    provider = _provider(probe={"command": None})
    handle = await provider.create(_spec())
    process = FakeProcess(stdout="ok", returncode=7, wait_delay=0 if state == "wait_done" else 0.01)
    process.poll = _Aio(lambda: 7 if state == "poll_exited" else None)

    async def drain():
        raise getattr(modal.exception, error)("already closed")

    process.stdin.drain = _Aio(drain)
    created[0].exec_script = {"work": process}
    async with asyncio.timeout(0.5):
        result = await provider.exec(handle, "work")
    assert result.stdout == "ok"
    assert result.return_code == (125 if state == "running" else 7)
    assert result.error_type == ("sandbox" if state == "running" else None)


@pytest.mark.parametrize("error", ["ConnectionError", "ServiceError", "InternalError", "builtin"])
async def test_transport_failure_keeps_output_and_never_replays(fake_modal, error):
    modal, created = fake_modal()
    provider = _provider(probe={"command": None}, operations={"retries": 3})
    handle = await provider.create(_spec())
    exc = ConnectionError if error == "builtin" else getattr(modal.exception, error)
    process = FakeProcess(stdout="partial", raise_on_wait=exc("lost connection"), wait_delay=0.001)
    created[0].exec_script = {"mutate": process}
    result = await provider.exec(handle, "mutate")
    assert result.stdout == "partial" and "lost connection" in result.stderr
    assert result.return_code == 125 and result.error_type == "sandbox"
    assert len(created[0].exec_calls) == 1


@pytest.mark.parametrize(
    "section,field",
    [
        ("probe", "timeout_s"),
        ("probe", "stable_delay_s"),
        ("operations", "retry_delay_s"),
        ("operations", "retry_max_delay_s"),
        ("operations", "tunnel_timeout_s"),
        ("operations", "close_timeout_s"),
    ],
)
def test_required_numeric_options_reject_none(section, field):
    with pytest.raises(ValueError, match=field):
        _provider(**{section: {field: None}})


@pytest.mark.parametrize(
    "section,field",
    [
        ("connection", "create_app_if_missing"),
        ("create", "block_network"),
        ("create", "strict_resources"),
        ("exec", "allow_user_rewrite"),
    ],
)
@pytest.mark.parametrize("value", ["false", 1, None])
def test_boolean_options_require_boolean(section, field, value):
    with pytest.raises(ValueError, match=field):
        _provider(**{section: {field: value}})


@pytest.mark.parametrize(
    "descriptor",
    [
        None,
        [],
        {},
        {"sandbox_id": None},
        {"sandbox_id": 1},
        {"sandbox_id": " "},
        {"sandbox_id": "sb-x", "port_mode": "bad"},
        {"sandbox_id": "sb-x", "ports": "80"},
        {"sandbox_id": "sb-x", "ports": [True]},
        {"sandbox_id": "sb-x", "ports": [0]},
        {"sandbox_id": "sb-x", "ports": [65536]},
        {"sandbox_id": "sb-x", "ports": ["80"]},
    ],
)
async def test_bad_connect_descriptor_never_reaches_sdk(fake_modal, descriptor):
    modal, _ = fake_modal()

    async def forbidden(*args):
        pytest.fail("invalid descriptor reached from_id")

    modal.Sandbox.from_id = _Aio(forbidden)
    with pytest.raises((TypeError, ValueError), match="descriptor"):
        await _provider().connect(descriptor)


@pytest.mark.parametrize("entrypoint", [None, ["custom", "arg"]])
async def test_default_keepalive_clears_image_entrypoint_only(fake_modal, entrypoint):
    _, created = fake_modal()
    await _provider(probe={"command": None}).create(_spec(entrypoint=entrypoint))
    assert created[0].create_kwargs["image"].entrypoint_override == ([] if entrypoint is None else None)


async def test_stopped_sandbox_fails_readiness_without_waiting_for_deadline(fake_modal):
    from nemo_gym.sandbox.providers.modal import ModalCreateVerificationError

    modal, created = fake_modal()
    original = modal.Sandbox.create

    async def create(*args, **kwargs):
        sandbox = await original.aio(*args, **kwargs)
        sandbox.poll_result = 1
        sandbox.exec_script = {"printf ok": modal.exception.NotFoundError("gone")}
        return sandbox

    modal.Sandbox.create = _Aio(create)
    with pytest.raises(ModalCreateVerificationError, match="stopped.*gone"):
        async with asyncio.timeout(0.5):
            await _provider(probe={"deadline_s": 30}).create(_spec())
    assert len(created[0].exec_calls) == 1 and created[0].terminated == 1


@pytest.mark.parametrize("scenario", ["transient", "unknown", "consecutive", "no_deadline"])
async def test_readiness_retries_and_consecutive_success_contract(fake_modal, scenario):
    from nemo_gym.sandbox.providers.modal import ModalCreateVerificationError

    modal, created = fake_modal()
    original = modal.Sandbox.create
    outcomes = (
        [modal.exception.ServiceError("transient"), 0]
        if scenario in ("transient", "unknown")
        else [0, 1, 0, 0]
        if scenario == "consecutive"
        else [1]
    )
    calls = []

    async def create(*args, **kwargs):
        sandbox = await original.aio(*args, **kwargs)
        if scenario == "unknown":
            sandbox.poll_exc = modal.exception.ServiceError("status unavailable")

        async def probe(*args, **kwargs):
            calls.append(1)
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return FakeProcess(stdout="ok", returncode=outcome)

        sandbox.exec = _Aio(probe)
        return sandbox

    modal.Sandbox.create = _Aio(create)
    provider = _provider(
        probe={
            "stable_count": 2 if scenario == "consecutive" else 1,
            "deadline_s": None if scenario == "no_deadline" else 2,
        }
    )
    async with asyncio.timeout(0.5):
        if scenario == "no_deadline":
            with pytest.raises(ModalCreateVerificationError, match="return_code=1"):
                await provider.create(_spec())
        else:
            await provider.create(_spec())
    assert len(calls) == {"transient": 2, "unknown": 2, "consecutive": 4, "no_deadline": 1}[scenario]
