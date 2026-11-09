# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import re

import pytest

from nemo_gym.sandbox import SandboxSpec
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
            return type("R", (), {"stdout": "fake log", "stderr": ""})()
        if self.exit_at_start:
            return type("R", (), {"stdout": "", "stderr": "", "return_code": 1})()
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
