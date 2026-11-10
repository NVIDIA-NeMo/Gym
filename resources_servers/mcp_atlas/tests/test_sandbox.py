# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from nemo_gym.sandbox import SandboxExecResult, SandboxSpec
from resources_servers.mcp_atlas import sandbox as sandbox_module
from resources_servers.mcp_atlas.sandbox import SandboxError, SandboxPool, call_sandbox_tool, sandbox_tool_names
from resources_servers.mcp_atlas.tests.fake_agent_environment import FakeAgentEnvironment, free_port


class FakeSandbox:
    """Starts the fake agent-environment when the pool execs the uvicorn command, on the port it names."""

    instances: list["FakeSandbox"] = []

    def __init__(
        self,
        provider,
        spec: SandboxSpec,
        *,
        tools: list[str],
        exit_at_start: bool = False,
        stop_errors: int = 0,
        ready_gate: asyncio.Event | None = None,
        stop_gate: asyncio.Event | None = None,
    ) -> None:
        self.spec = spec
        self.environment = FakeAgentEnvironment(tools, {})
        self.exit_at_start = exit_at_start
        # The first ``stop_errors`` stops fail, as a provider that cannot reach its container does.
        self.stop_errors = stop_errors
        self.stopped = False
        self.ready_gate = ready_gate
        self.stop_gate = stop_gate
        self.exec_started = asyncio.Event()
        self.stop_started = asyncio.Event()
        FakeSandbox.instances.append(self)

    async def start(self):
        return self

    async def exec(self, command: str, *, timeout_s=None):
        if command.startswith("tail "):
            return SandboxExecResult("fake log", "", 0)
        if self.exit_at_start:
            return SandboxExecResult("", "", 1)
        self.exec_started.set()
        if self.ready_gate is not None:
            await self.ready_gate.wait()
        port = int(re.search(r"--port (\d+)", command).group(1))
        await self.environment.start(port)
        await asyncio.Event().wait()

    async def stop(self) -> None:
        self.stop_started.set()
        if self.stop_gate is not None:
            await self.stop_gate.wait()
        if self.stop_errors:
            self.stop_errors -= 1
            raise RuntimeError("provider could not stop the container")
        self.stopped = True
        await self.environment.stop()


def pool(
    tools: list[str],
    *,
    exit_at_start: bool = False,
    ttl_s: float = 3600,
    scratch_root=None,
    stop_errors: int = 0,
    ready_gate: asyncio.Event | None = None,
    stop_gate: asyncio.Event | None = None,
    sandbox_metadata: dict | None = None,
) -> SandboxPool:
    port = free_port()
    return SandboxPool(
        image="/images/mcp-atlas.sif",
        sandbox_provider={"apptainer": {}},
        sandbox_env={"EXA_API_KEY": "secret"},
        sandbox_files={"/root/.npmrc": "before=2026-07-17\n"},
        port_range=(port, port + 20),
        max_concurrency=2,
        startup_timeout_s=10,
        ttl_s=ttl_s,
        scratch_root=scratch_root,
        sandbox_metadata=sandbox_metadata,
        sandbox_factory=lambda provider, spec: FakeSandbox(
            provider,
            spec,
            tools=tools,
            exit_at_start=exit_at_start,
            stop_errors=stop_errors,
            ready_gate=ready_gate,
            stop_gate=stop_gate,
        ),
    )


def test_tool_names_keep_prefixes_and_map_a_single_server() -> None:
    names, missing = sandbox_tool_names(
        ["calculator_calculate", "wikipedia_search"], {"calculator_calculate"}, ["calculator", "wikipedia"]
    )
    assert names == {"calculator_calculate": "calculator_calculate"}
    assert missing == ["wikipedia_search"]
    names, missing = sandbox_tool_names(["calculator_calculate"], {"calculate"}, ["calculator"])
    assert names == {"calculator_calculate": "calculate"}
    assert missing == []


async def test_start_call_and_stop(tmp_path) -> None:
    sandboxes = pool(["calculator_calculate", "whois_whois_domain"], scratch_root=str(tmp_path))
    box = await sandboxes.start("s1", servers=["calculator", "whois"], tools=["calculator_calculate"])
    sandbox = FakeSandbox.instances[-1]
    assert sandbox.spec.env["ENABLED_SERVERS"] == "calculator,whois"
    assert sandbox.spec.env["EXA_API_KEY"] == "secret"
    assert "secret" not in str(sandbox.spec.provider_options)
    assert sandbox.spec.files == {"/root/.npmrc": "before=2026-07-17\n"}
    assert box.missing_tools == []
    assert os.path.isdir(box.scratch_dir)

    status, body = await call_sandbox_tool(box, "calculator_calculate", {"expression": "1+1"}, timeout_s=5)
    assert status == 200 and body == [{"type": "text", "text": "ok"}]
    assert sandbox.environment.calls[-1] == {
        "tool_name": "calculator_calculate",
        "tool_args": {"expression": "1+1"},
        "use_cache": False,
    }

    assert await sandboxes.stop("s1") is True
    assert sandbox.stopped
    assert not os.path.exists(box.scratch_dir)
    assert await sandboxes.stop("s1") is False


async def test_missing_tools_are_reported() -> None:
    sandboxes = pool(["calculator_calculate"])
    box = await sandboxes.start("s1", servers=["calculator", "notion"], tools=["calculator_calculate", "notion_x"])
    assert box.missing_tools == ["notion_x"]
    await sandboxes.close()


async def test_server_exit_at_start_is_a_sandbox_error_and_frees_capacity() -> None:
    sandboxes = pool(["calculator_calculate"], exit_at_start=True)
    for _ in range(3):  # more than max_concurrency: capacity must come back after each failure
        with pytest.raises(SandboxError, match="exited during startup"):
            await sandboxes.start("s1", servers=["calculator"], tools=["calculator_calculate"])
        assert FakeSandbox.instances[-1].stopped


async def test_a_failed_stop_keeps_the_sandbox_and_its_leases(tmp_path) -> None:
    sandboxes = pool(["calculator_calculate"], scratch_root=str(tmp_path), stop_errors=1)
    box = await sandboxes.start("s1", servers=["calculator"], tools=["calculator_calculate"])
    with pytest.raises(SandboxError, match="sandbox stop failed"):
        await sandboxes.stop("s1")
    assert sandboxes.get("s1") is box
    assert box.port in sandboxes._ports._leased
    assert os.path.isdir(box.scratch_dir)
    # The retry stops it and frees everything.
    assert await sandboxes.stop("s1") is True
    assert sandboxes.get("s1") is None
    assert box.port not in sandboxes._ports._leased
    assert not os.path.exists(box.scratch_dir)


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses read-only directory permissions")
async def test_stop_removes_a_readonly_subdirectory_and_releases_its_leases(tmp_path) -> None:
    sandboxes = pool(["calculator_calculate"], scratch_root=str(tmp_path))
    readonly = None
    try:
        box = await sandboxes.start("s1", servers=["calculator"], tools=["calculator_calculate"])
        provider = box.sandbox
        readonly = Path(box.scratch_dir) / "readonly-cache"
        readonly.mkdir()
        cached_file = readonly / "cache.txt"
        cached_file.write_text("cached data")
        readonly.chmod(0o555)
        try:
            cached_file.unlink()
        except PermissionError:
            pass
        else:
            pytest.skip("the filesystem does not enforce read-only directory permissions")

        assert await sandboxes.stop("s1") is True
        assert provider.stopped and box.stopped
        assert sandboxes.get("s1") is None
        assert box.port not in sandboxes._ports._leased
        assert not Path(box.scratch_dir).exists()
        await asyncio.wait_for(sandboxes.start("s2", servers=["calculator"], tools=["calculator_calculate"]), 2)
        await asyncio.wait_for(sandboxes.start("s3", servers=["calculator"], tools=["calculator_calculate"]), 2)
    finally:
        try:
            if readonly is not None and readonly.exists():
                readonly.chmod(0o755)
        finally:
            await sandboxes.close()


async def test_residual_scratch_error_is_logged_without_leaking_or_overreleasing_capacity(
    tmp_path, monkeypatch, caplog
) -> None:
    sandboxes = pool(["calculator_calculate"], scratch_root=str(tmp_path))
    scratch = None
    waiting = None
    try:
        box = await sandboxes.start("s1", servers=["calculator"], tools=["calculator_calculate"])
        provider = box.sandbox
        scratch = Path(box.scratch_dir)
        caplog.set_level(logging.WARNING, logger=sandbox_module.__name__)

        def fail_cleanup(path, **kwargs) -> None:
            assert Path(path) == scratch
            raise OSError("residual disk cleanup failed")

        with monkeypatch.context() as patch:
            patch.setattr(sandbox_module.shutil, "rmtree", fail_cleanup)
            assert await sandboxes.stop("s1") is True

        assert provider.stopped and box.stopped
        assert sandboxes.get("s1") is None
        assert box.port not in sandboxes._ports._leased
        assert scratch.exists()
        assert "could not be removed: residual disk cleanup failed" in caplog.text
        assert str(scratch) in caplog.text
        second = await asyncio.wait_for(
            sandboxes.start("s2", servers=["calculator"], tools=["calculator_calculate"]), 2
        )
        third = await asyncio.wait_for(
            sandboxes.start("s3", servers=["calculator"], tools=["calculator_calculate"]), 2
        )
        assert second.port != third.port
        waiting = asyncio.create_task(sandboxes.start("s4", servers=["calculator"], tools=["calculator_calculate"]))
        await asyncio.sleep(0)
        assert not waiting.done()
        assert sandboxes.get("s4") is None
    finally:
        if waiting is not None:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
        try:
            await sandboxes.close()
        finally:
            if scratch is not None and scratch.exists():
                shutil.rmtree(scratch)


async def test_a_sandbox_that_did_not_stop_after_a_failed_start_is_kept_for_a_retry() -> None:
    sandboxes = pool(["calculator_calculate"], exit_at_start=True, stop_errors=1)
    with pytest.raises(SandboxError, match="exited during startup"):
        await sandboxes.start("s1", servers=["calculator"], tools=["calculator_calculate"])
    assert sandboxes.get("s1") is not None
    assert await sandboxes.stop("s1") is True
    assert FakeSandbox.instances[-1].stopped


async def test_close_reports_sandboxes_that_did_not_stop() -> None:
    sandboxes = pool(["calculator_calculate"], stop_errors=1)
    await sandboxes.start("s1", servers=["calculator"], tools=["calculator_calculate"])
    with pytest.raises(SandboxError, match="1 sandboxes did not stop"):
        await sandboxes.close()
    await sandboxes.close()
    assert sandboxes.get("s1") is None


async def test_idle_sandboxes_are_reaped() -> None:
    sandboxes = pool(["calculator_calculate"], ttl_s=0.01)
    box = await sandboxes.start("s1", servers=["calculator"], tools=["calculator_calculate"])
    await asyncio.sleep(0.05)
    assert await sandboxes.reap_expired() == 1
    assert sandboxes.get("s1") is None
    # Its port is free for the next sandbox, so a late call of the reaped rollout must not be sent.
    assert box.port not in sandboxes._ports._leased
    with pytest.raises(SandboxError, match="stopped"):
        await call_sandbox_tool(box, "calculator_calculate", {"expression": "1"}, timeout_s=5)


async def test_interrupted_failed_start_remains_owned_until_a_later_stop(tmp_path) -> None:
    ready_gate = asyncio.Event()
    stop_gate = asyncio.Event()
    sandboxes = pool(["calculator_calculate"], scratch_root=str(tmp_path), ready_gate=ready_gate, stop_gate=stop_gate)
    task = asyncio.create_task(sandboxes.start("s1", servers=["calculator"], tools=["calculator_calculate"]))
    await asyncio.sleep(0)
    sandbox = FakeSandbox.instances[-1]
    await asyncio.wait_for(sandbox.exec_started.wait(), 2)
    box = sandboxes.get("s1")
    assert box is not None
    task.cancel()
    await asyncio.wait_for(sandbox.stop_started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sandboxes.get("s1") is box
    assert os.path.isdir(box.scratch_dir)
    assert not sandbox.stopped
    stop_gate.set()
    assert await asyncio.wait_for(sandboxes.stop("s1"), 2)
    assert sandbox.stopped
    assert not os.path.exists(box.scratch_dir)
    assert sandboxes.get("s1") is None
    ready_gate.set()
    # Two further starts prove the interrupted attempt released its one slot exactly once.
    second = await asyncio.wait_for(sandboxes.start("s2", servers=["calculator"], tools=["calculator_calculate"]), 2)
    third = await asyncio.wait_for(sandboxes.start("s3", servers=["calculator"], tools=["calculator_calculate"]), 2)
    waiting = asyncio.create_task(sandboxes.start("s4", servers=["calculator"], tools=["calculator_calculate"]))
    await asyncio.sleep(0)
    assert not waiting.done()
    assert sandboxes.get("s4") is None
    assert second.port != third.port
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert sandboxes.get("s4") is None
    await sandboxes.close()


async def test_provider_metadata_reaches_the_sandbox_spec() -> None:
    sandboxes = pool(["calculator_calculate"], sandbox_metadata={"placement_group": "atlas-test"})
    await sandboxes.start("s1", servers=["calculator"], tools=["calculator_calculate"])
    assert FakeSandbox.instances[-1].spec.metadata == {"placement_group": "atlas-test"}
    await sandboxes.close()


class ExecSandbox:
    """The external provider boundary; HTTP bridge responses arrive through exec stdout."""

    def __init__(self, spec, *, startup_result=None, stop_errors=0) -> None:
        self.spec = spec
        self.startup_result = startup_result or SandboxExecResult("", "", 0)
        self.stop_errors = stop_errors
        self.stopped = False
        self.commands = []
        self.requests = []
        self.next_result = None
        self.next_error = None

    async def start(self):
        return self

    async def exec(self, command, *, timeout_s=None, env=None, preserve_background_services=False):
        self.commands.append((command, timeout_s, env, preserve_background_services))
        if preserve_background_services:
            return self.startup_result
        if command.startswith("tail "):
            return SandboxExecResult("service log", "", 0)
        request = json.loads(env[sandbox_module.HTTP_REQUEST_ENV])
        self.requests.append(request)
        if self.next_error is not None:
            error, self.next_error = self.next_error, None
            raise error
        if self.next_result is not None:
            result, self.next_result = self.next_result, None
            return result
        if request["path"] == "/health":
            body = {"status": "health_and_client_connection_ok"}
        elif request["path"] == "/list-tools":
            body = [{"name": "calculate"}]
        else:
            body = {"received": request["json"]}
        return SandboxExecResult(json.dumps({"status": 200, "body": body}), "", 0)

    async def stop(self) -> None:
        if self.stop_errors:
            self.stop_errors -= 1
            raise RuntimeError("provider could not stop the container")
        self.stopped = True


def exec_pool(*, startup_result=None, stop_errors=0, **overrides):
    created = []

    def factory(provider, spec):
        assert next(iter(provider)) == "opensandbox"
        sandbox = ExecSandbox(spec, startup_result=startup_result, stop_errors=stop_errors)
        created.append(sandbox)
        return sandbox

    values = {
        "image": "ghcr.io/scaleapi/mcp-atlas:1.2.7",
        "sandbox_provider": {"opensandbox": {"connection": {"api_key": "provider-only"}}},
        "sandbox_env": {"EXA_API_KEY": "service-only"},
        "sandbox_files": {"/root/.npmrc": "before=2026-07-17\n"},
        "port_range": (27000, 27001),
        "max_concurrency": 2,
        "startup_timeout_s": 10,
        "ttl_s": 3600,
        "sandbox_factory": factory,
        **overrides,
    }
    return SandboxPool(**values), created


async def test_opensandbox_uses_only_exec_and_private_loopback_without_host_leases(monkeypatch) -> None:
    host_http = AsyncMock(side_effect=AssertionError("host HTTP must not be used"))
    monkeypatch.setattr(sandbox_module, "_http_json", host_http)

    def unexpected_host_allocation(*args, **kwargs):
        raise AssertionError("OpenSandbox must not allocate a host port or scratch directory")

    monkeypatch.setattr(sandbox_module.PortAllocator, "acquire", unexpected_host_allocation)
    monkeypatch.setattr(sandbox_module.tempfile, "mkdtemp", unexpected_host_allocation)
    options = {"network_policy": {"defaultAction": "deny", "egress": []}}
    sandboxes, created = exec_pool(sandbox_options=options, sandbox_metadata={"purpose": "atlas"})
    try:
        first = await sandboxes.start("s1", servers=["calculator"], tools=["calculator_calculate"])
        second = await sandboxes.start("s2", servers=["calculator"], tools=["calculator_calculate"])
        for box, provider in zip((first, second), created, strict=True):
            assert box.port is None and box.scratch_dir is None and box.base_url == ""
            assert provider.spec.ports == ()
            assert provider.spec.provider_options == options
            assert provider.spec.metadata == {"purpose": "atlas"}
            assert provider.spec.env == {"EXA_API_KEY": "service-only", "ENABLED_SERVERS": "calculator"}
            assert provider.spec.entrypoint == ["/bin/sh", "-c", "exec sleep infinity"]
            assert sandbox_module.HTTP_HELPER_PATH in provider.spec.files
            command, _, env, preserve = provider.commands[0]
            assert "--host 127.0.0.1 --port 1984" in command
            assert command.endswith(" &") and preserve and env is None
            assert [r["path"] for r in provider.requests] == ["/health", "/list-tools"]
        arguments = {"expression": "'; $(touch /tmp/not-executed) `echo hi`\n$HOME"}
        status, body = await call_sandbox_tool(first, "calculator_calculate", arguments, timeout_s=3)
        assert status == 200
        assert body == {"received": {"tool_name": "calculate", "tool_args": arguments, "use_cache": False}}
        command, timeout_s, env, preserve = created[0].commands[-1]
        assert command == f"/agent-environment/.venv/bin/python {sandbox_module.HTTP_HELPER_PATH}"
        assert arguments["expression"] not in command
        assert set(env) == {sandbox_module.HTTP_REQUEST_ENV}
        assert json.loads(env[sandbox_module.HTTP_REQUEST_ENV]) == created[0].requests[-1]
        assert timeout_s == 8 and preserve is False
        host_http.assert_not_awaited()
    finally:
        await sandboxes.close()
    assert all(provider.stopped for provider in created)


@pytest.mark.parametrize(
    ("result", "exception"),
    [
        (SandboxExecResult("not JSON", "", 0), SandboxError),
        (SandboxExecResult("[]", "", 0), SandboxError),
        (SandboxExecResult('{"status":true,"body":null}', "", 0), SandboxError),
        (SandboxExecResult('{"status":700,"body":null}', "", 0), SandboxError),
        (SandboxExecResult('{"status":200}', "", 0), SandboxError),
        (SandboxExecResult('{"status":200,"body":"ignored"}', "", 1), SandboxError),
        (SandboxExecResult('{"status":200,"body":"ignored"}', "", 0, error_type="provider_failure"), SandboxError),
        (SandboxExecResult(None, None, 124, error_type="timeout"), asyncio.TimeoutError),
        (SandboxExecResult('{"error":"timeout","message":"deadline"}', "", 0), asyncio.TimeoutError),
        (SandboxExecResult('{"error":"transport","message":"RemoteDisconnected"}', "", 0), OSError),
        (SandboxExecResult('{"error":"invalid_request","message":"invalid method"}', "", 0), SandboxError),
    ],
)
async def test_exec_transport_errors_do_not_replay_or_fall_back_to_host_http(monkeypatch, result, exception) -> None:
    host_http = AsyncMock(side_effect=AssertionError("no host HTTP fallback"))
    monkeypatch.setattr(sandbox_module, "_http_json", host_http)
    sandboxes, created = exec_pool()
    try:
        box = await sandboxes.start("s1", servers=["calculator"], tools=["calculator_calculate"])
        provider = created[0]
        before = len(provider.requests)
        provider.next_result = result
        with pytest.raises(exception):
            await call_sandbox_tool(box, "calculator_calculate", {}, timeout_s=1)
        assert len(provider.requests) == before + 1
        host_http.assert_not_awaited()
    finally:
        await sandboxes.close()


async def test_exec_provider_exception_is_an_infrastructure_failure_without_replay() -> None:
    sandboxes, created = exec_pool()
    try:
        box = await sandboxes.start("s1", servers=["calculator"], tools=["calculator_calculate"])
        provider = created[0]
        before = len(provider.requests)
        provider.next_error = RuntimeError("ambiguous provider submission")
        with pytest.raises(SandboxError, match="sandbox HTTP exec failed: RuntimeError"):
            await call_sandbox_tool(box, "calculator_calculate", {}, timeout_s=1)
        assert len(provider.requests) == before + 1
    finally:
        await sandboxes.close()


async def test_exec_transport_preserves_an_http_tool_failure_body() -> None:
    sandboxes, created = exec_pool()
    try:
        box = await sandboxes.start("s1", servers=["calculator"], tools=["calculator_calculate"])
        created[0].next_result = SandboxExecResult('{"status":500,"body":{"detail":"Object 502 not found"}}', "", 0)
        assert await call_sandbox_tool(box, "calculator_calculate", {}, timeout_s=1) == (
            500,
            {"detail": "Object 502 not found"},
        )
    finally:
        await sandboxes.close()


@pytest.mark.parametrize("result", [SandboxExecResult("", "", 1), SandboxExecResult("", "", 0, error_type="runtime")])
async def test_failed_service_start_keeps_retryable_provider_ownership(result) -> None:
    sandboxes, created = exec_pool(startup_result=result, stop_errors=1, max_concurrency=1)
    with pytest.raises(SandboxError, match="sandbox exec failed"):
        await sandboxes.start("s1", servers=["calculator"], tools=["calculator_calculate"])
    assert sandboxes.get("s1") is not None
    assert not created[0].stopped
    assert created[0].requests == []
    assert await sandboxes.stop("s1") is True
    assert created[0].stopped and sandboxes.get("s1") is None
    # A new start acquires the single slot again instead of hanging after failed cleanup.
    with pytest.raises(SandboxError, match="sandbox exec failed"):
        await asyncio.wait_for(sandboxes.start("s2", servers=["calculator"], tools=["calculator_calculate"]), 2)
    await sandboxes.close()


async def test_opensandbox_stop_failure_retains_ownership_then_prevents_late_calls() -> None:
    sandboxes, created = exec_pool(stop_errors=1)
    box = await sandboxes.start("s1", servers=["calculator"], tools=["calculator_calculate"])
    with pytest.raises(SandboxError, match="sandbox stop failed"):
        await sandboxes.stop("s1")
    assert sandboxes.get("s1") is box and not box.stopped
    assert await sandboxes.stop("s1") is True
    assert box.stopped and created[0].stopped
    before = len(created[0].commands)
    with pytest.raises(SandboxError, match="stopped"):
        await call_sandbox_tool(box, "calculator_calculate", {}, timeout_s=1)
    assert len(created[0].commands) == before


@pytest.mark.parametrize(
    "path",
    [
        sandbox_module.HTTP_HELPER_PATH,
        "/tmp/../tmp/nemo-gym-mcp-atlas-http.py",
        "//tmp/nemo-gym-mcp-atlas-http.py",
        "/tmp",
    ],
)
def test_caller_cannot_replace_the_staged_http_helper(path) -> None:
    with pytest.raises(ValueError, match="must not replace the HTTP helper"):
        exec_pool(sandbox_files={path: "caller supplied code"})


def test_caller_cannot_supply_the_reserved_request_environment() -> None:
    with pytest.raises(ValueError, match="reserved MCP_ATLAS_HTTP_REQUEST"):
        exec_pool(sandbox_env={sandbox_module.HTTP_REQUEST_ENV: "caller request"})


@pytest.mark.parametrize("suppresses_cancellation", [False, True])
async def test_bridge_deadline_cancels_and_drains_one_exec_before_rejecting_a_late_response(
    suppresses_cancellation: bool,
) -> None:
    sandboxes, created = exec_pool()
    cancelled = asyncio.Event()
    finished = asyncio.Event()
    response_ready = asyncio.Event()
    cleanup_release = asyncio.Event()
    calls = []
    calling = None

    async def exec(command, **kwargs):
        calls.append((command, kwargs))
        try:
            await response_ready.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await cleanup_release.wait()
            if not suppresses_cancellation:
                raise
        finally:
            finished.set()
        return SandboxExecResult('{"status":200,"body":"late response"}', "", 0)

    try:
        box = await sandboxes.start("s1", servers=["calculator"], tools=["calculator_calculate"])
        created[0].exec = exec
        calling = asyncio.create_task(call_sandbox_tool(box, "calculator_calculate", {}, timeout_s=0.03))
        # The provider's five-second grace cannot supply this cancellation: the host deadline must do it.
        await asyncio.wait_for(cancelled.wait(), 1)
        assert not calling.done(), "the timed-out call must still drain provider cancellation cleanup"
        assert len(calls) == 1
        command, kwargs = calls[0]
        assert kwargs["timeout_s"] == 5.03
        assert command == f"/agent-environment/.venv/bin/python {sandbox_module.HTTP_HELPER_PATH}"
        assert set(kwargs["env"]) == {sandbox_module.HTTP_REQUEST_ENV}
        cleanup_release.set()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(calling, 1)
        assert finished.is_set()
        response_ready.set()
        await asyncio.sleep(0)
        assert len(calls) == 1, "a late response cannot replay a timed-out mutating request"
    finally:
        response_ready.set()
        cleanup_release.set()
        if calling is not None:
            if not calling.done():
                calling.cancel()
            await asyncio.gather(calling, return_exceptions=True)
        await sandboxes.close()


async def test_bridge_deadline_stops_a_real_helper_despite_continuous_socket_data() -> None:
    stop = asyncio.Event()
    chunks = []

    async def drip(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        await response.prepare(request)
        try:
            while not stop.is_set():
                await response.write(b" ")
                chunks.append(1)
                await asyncio.sleep(0.02)
        except ConnectionResetError:
            stop.set()
        return response

    app = web.Application()
    app.router.add_post("/call-tool", drip)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]

    class SubprocessSandbox:
        def __init__(self, process: asyncio.subprocess.Process) -> None:
            self.process = process
            self.cancelled = False
            self.finished = False
            self.calls = 0

        async def exec(self, command, *, timeout_s, env):
            self.calls += 1
            try:
                # The provider boundary hands off the request only after the real helper is loaded.
                # Process startup is outside this streaming-specific regression's measured budget.
                request = env[sandbox_module.HTTP_REQUEST_ENV].encode() + b"\n"
                stdout, stderr = await self.process.communicate(request)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            finally:
                if self.process.returncode is None:
                    self.process.kill()
                await self.process.wait()
                self.finished = True
            return SandboxExecResult(
                stdout.decode(errors="replace"), stderr.decode(errors="replace"), self.process.returncode
            )

    process = None
    calling = None
    try:
        helper = Path(sandbox_module.__file__).with_name("sandbox_http.py")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import os,runpy,sys; helper=runpy.run_path(sys.argv[1]); print('ready',flush=True); "
            "os.environ['MCP_ATLAS_HTTP_REQUEST']=sys.stdin.readline(); helper['main'](port=int(sys.argv[2]))",
            str(helper),
            str(port),
            env={},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert await asyncio.wait_for(process.stdout.readline(), 5) == b"ready\n"
        provider = SubprocessSandbox(process)
        box = sandbox_module.AtlasSandbox(
            base_url="", tool_names={}, missing_tools=[], exec_transport=True, sandbox=provider
        )
        calling = asyncio.create_task(call_sandbox_tool(box, "calculator_calculate", {}, timeout_s=0.5))
        done, _ = await asyncio.wait({calling}, timeout=2)
        assert calling in done, "continuous response bytes must not extend the host's complete-request deadline"
        with pytest.raises(asyncio.TimeoutError):
            await calling
        assert len(chunks) > 2, "the helper received bytes often enough to avoid its per-socket timeout"
        assert provider.cancelled and provider.finished and provider.calls == 1
    finally:
        stop.set()
        if calling is not None:
            if not calling.done():
                calling.cancel()
            await asyncio.gather(calling, return_exceptions=True)
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        await runner.cleanup()
