# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A rollout whose tools died under the policy ends as a sandbox fault, not as a result that gets scored."""

from __future__ import annotations

import asyncio
import inspect
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from responses_api_agents.apex_agent import prebuilt_world_entrypoint, sandbox_entrypoint, stirrup_runtime
from responses_api_agents.apex_agent.stirrup_runtime import (
    ToolPlaneHealth,
    _checkpoint_partial_result,
    mcp_tools_respond,
    require_live_tool_plane,
)


GATEWAY_URL = "http://127.0.0.1:1"


def _healthy(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    timeouts: list[float] = []

    async def wait_for_gateway(_url: str, timeout_seconds: float = 60.0) -> None:
        timeouts.append(timeout_seconds)

    monkeypatch.setattr(stirrup_runtime, "wait_for_gateway", wait_for_gateway)
    return timeouts


def _probe(monkeypatch: pytest.MonkeyPatch, *, answers: bool) -> list[str]:
    """Replace the MCP probe; returns the gateway URLs it was asked about."""
    probed: list[str] = []

    async def mcp_tools_respond(gateway_url: str) -> bool:
        probed.append(gateway_url)
        return answers

    monkeypatch.setattr(stirrup_runtime, "mcp_tools_respond", mcp_tools_respond)
    return probed


def test_unanswered_calls_count_in_total_and_as_the_run_of_them_at_the_end() -> None:
    health = ToolPlaneHealth()
    health.unanswered()
    health.unanswered()
    health.answered()
    assert (health.unanswered_calls, health.trailing_unanswered_calls) == (2, 0)
    health.unanswered()
    assert (health.unanswered_calls, health.trailing_unanswered_calls) == (3, 1)
    assert health.fields() == {"n_mcp_unanswered_calls": 3, "n_mcp_trailing_unanswered_calls": 1}


async def test_a_live_tool_plane_lets_the_result_through_without_probing(monkeypatch: pytest.MonkeyPatch) -> None:
    timeouts = _healthy(monkeypatch)
    probed = _probe(monkeypatch, answers=False)

    await require_live_tool_plane(SimpleNamespace(returncode=None), GATEWAY_URL, {"completed": True})
    await require_live_tool_plane(None, GATEWAY_URL, {"n_mcp_trailing_unanswered_calls": 0})

    assert timeouts == [stirrup_runtime.TOOL_PLANE_HEALTH_TIMEOUT_SECONDS] * 2
    assert probed == [], "a run whose last tool call was answered needs no probe"


async def test_an_exited_gateway_fails_the_rollout_without_waiting_for_its_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeouts = _healthy(monkeypatch)

    with pytest.raises(RuntimeError, match="gateway exited with status 137"):
        await require_live_tool_plane(SimpleNamespace(returncode=137), GATEWAY_URL, {"completed": True})

    assert timeouts == []


async def test_a_gateway_that_stopped_answering_its_health_check_fails_the_rollout() -> None:
    pytest.importorskip("httpx")

    # Nothing listens on port 1, so the real health check runs out its short deadline.
    with pytest.raises(RuntimeError, match="stopped answering its health check"):
        await require_live_tool_plane(
            SimpleNamespace(returncode=None), GATEWAY_URL, {"completed": True}, health_timeout_seconds=0.3
        )


async def test_slow_tool_calls_at_the_end_of_a_run_are_the_policys_when_the_tools_still_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three commands the policy started time out, then it calls finish, which is not an MCP call."""
    _healthy(monkeypatch)
    probed = _probe(monkeypatch, answers=True)
    result = {"completed": True, "n_mcp_unanswered_calls": 3, "n_mcp_trailing_unanswered_calls": 3}

    await require_live_tool_plane(SimpleNamespace(returncode=None), GATEWAY_URL, result)

    assert probed == [GATEWAY_URL]
    assert result["n_mcp_unanswered_calls"] == 3, "the counts stay on the result as diagnostics"


@pytest.mark.parametrize("trailing", [1, 3])
async def test_unanswered_tool_calls_at_the_end_fail_the_rollout_only_when_the_probe_fails_too(
    monkeypatch: pytest.MonkeyPatch, trailing: int
) -> None:
    _healthy(monkeypatch)
    probed = _probe(monkeypatch, answers=False)
    result = {"completed": False, "n_mcp_trailing_unanswered_calls": trailing}

    with pytest.raises(RuntimeError, match="went unanswered and the MCP endpoint no longer lists its tools"):
        await require_live_tool_plane(SimpleNamespace(returncode=None), GATEWAY_URL, result)

    assert probed == [GATEWAY_URL]


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


async def test_the_probe_reports_a_closed_or_silent_mcp_endpoint_as_down() -> None:
    pytest.importorskip("mcp")

    assert await mcp_tools_respond(f"http://127.0.0.1:{_free_port()}", timeout_seconds=5) is False

    async def never_answer(_reader: asyncio.StreamReader, _writer: asyncio.StreamWriter) -> None:
        await asyncio.sleep(3600)

    silent = await asyncio.start_server(never_answer, "127.0.0.1", 0)
    try:
        port = silent.sockets[0].getsockname()[1]
        assert await mcp_tools_respond(f"http://127.0.0.1:{port}", timeout_seconds=1) is False
    finally:
        silent.close()


async def test_the_probe_lists_the_tools_of_a_live_mcp_endpoint() -> None:
    fastmcp = pytest.importorskip("mcp.server.fastmcp")
    uvicorn = pytest.importorskip("uvicorn")
    port = _free_port()
    listed: list[str] = []

    class RecordingServer(fastmcp.FastMCP):
        async def list_tools(self):
            listed.append("tools/list")
            return await super().list_tools()

    server_app = RecordingServer("probe-target", host="127.0.0.1", port=port, streamable_http_path="/mcp/")
    server = uvicorn.Server(
        uvicorn.Config(server_app.streamable_http_app(), host="127.0.0.1", port=port, log_level="critical")
    )
    serving = asyncio.create_task(server.serve())
    try:
        for _ in range(100):
            if server.started:
                break
            await asyncio.sleep(0.05)
        assert await mcp_tools_respond(f"http://127.0.0.1:{port}", timeout_seconds=10) is True
        assert listed == ["tools/list"], "the probe goes past the handshake to a tool listing"
    finally:
        server.should_exit = True
        await serving


async def test_the_partial_result_keeps_the_latest_tool_plane_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A rollout killed by its timeout never reaches the end-of-run check; its partial result still reports them."""
    monkeypatch.setattr(stirrup_runtime, "PARTIAL_RESULT_CHECKPOINT_INTERVAL_SECONDS", 0.01)
    session = SimpleNamespace(_current_run_state=SimpleNamespace(full_msg_history=[], msgs=[]))
    health = ToolPlaneHealth()
    destination = tmp_path / "partial_result.json"

    def fields() -> dict:
        return {"resume_segments": 1} | health.fields()

    async def written(count: int) -> dict:
        for _ in range(500):
            if destination.is_file() and json.loads(destination.read_text())["n_mcp_unanswered_calls"] == count:
                return json.loads(destination.read_text())
            await asyncio.sleep(0.01)
        raise AssertionError(f"the partial result never reported {count} unanswered call(s)")

    task = asyncio.create_task(_checkpoint_partial_result(session, destination, fields))
    try:
        assert (await written(0))["resume_segments"] == 1
        # The same turn is still running when two tool calls time out: the file is rewritten for the counts alone.
        health.unanswered()
        health.unanswered()
        assert (await written(2))["n_mcp_trailing_unanswered_calls"] == 2
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_the_rollout_counts_tool_calls_that_time_out_and_reports_them() -> None:
    source = inspect.getsource(stirrup_runtime.run_stirrup_rollout)

    timeout_branch = source.split("except asyncio.TimeoutError:", 1)[1]
    assert timeout_branch.lstrip().startswith("self.health.unanswered()")
    # A raised transport or protocol error is counted too, and still ends the rollout.
    raised_branch = timeout_branch.split("except Exception:", 1)[1]
    assert [line.strip() for line in raised_branch.splitlines() if line.strip() and "#" not in line][:2] == [
        "self.health.unanswered()",
        "raise",
    ]
    assert "self.health.answered()" in raised_branch
    # The counts go on the final result and, read again at every write, on the partial result.
    assert "**managed_tools.health.fields()," in source
    assert "return segment_fields | managed_tools.health.fields()" in source
    assert "_checkpoint_partial_result(session, checkpoint_path, partial_fields, checkpointer)" in source
    assert "extra=partial_fields()," in source


class _Gateway:
    """The gateway subprocess; ``returncode`` turns non-None when it dies."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    async def wait(self) -> int:
        return self.returncode or 0


async def _run_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    gateway_dies: bool,
    trailing_unanswered: int = 0,
    probe_answers: bool = True,
) -> tuple[_Gateway, Path]:
    """Run the ZIP-world entrypoint with a rollout that returns normally, on stand-ins for the world and gateway."""
    root = tmp_path / "apex-gym"
    output = root / "output"
    root.mkdir()
    (root / "runner_config.json").write_text(json.dumps({"task_id": "task-1", "world_id": "world-1"}))
    gateway = _Gateway()
    snapshots: list[Path] = []

    async def create_subprocess_exec(*_args, **_kwargs) -> _Gateway:
        return gateway

    async def discover(_process, _log_path) -> str:
        return GATEWAY_URL

    async def no_op(*_args, **_kwargs) -> None:
        return None

    def write_snapshot(destination: Path) -> list[str]:
        snapshots.append(destination)
        destination.write_bytes(b"zip")
        return ["filesystem/report.txt"]

    async def run_stirrup_rollout(*_args, **_kwargs) -> dict:
        if gateway_dies:
            gateway.returncode = 1
        return {
            "final_answer": "",
            "completed": False,
            "completion_status": "max_turns",
            "n_mcp_trailing_unanswered_calls": trailing_unanswered,
        }

    monkeypatch.setattr(sandbox_entrypoint, "ROOT", root)
    monkeypatch.setattr(sandbox_entrypoint, "OUTPUT", output)
    monkeypatch.setattr(sandbox_entrypoint, "PARTIAL_RESULT_PATH", tmp_path / "partial_result.json")
    monkeypatch.setattr(sandbox_entrypoint, "_patch_code_mcp_cancellation_race", lambda: None)
    monkeypatch.setattr(sandbox_entrypoint, "populate_world", lambda *_args: None)
    monkeypatch.setattr(sandbox_entrypoint, "gateway_config", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(sandbox_entrypoint, "_discover_gateway_url", discover)
    monkeypatch.setattr(sandbox_entrypoint, "wait_for_gateway", no_op)
    monkeypatch.setattr(sandbox_entrypoint, "configure_gateway", no_op)
    monkeypatch.setattr(sandbox_entrypoint, "write_snapshot", write_snapshot)
    monkeypatch.setattr(sandbox_entrypoint, "run_stirrup_rollout", run_stirrup_rollout)
    monkeypatch.setattr(sandbox_entrypoint.asyncio, "create_subprocess_exec", create_subprocess_exec)
    _healthy(monkeypatch)
    _probe(monkeypatch, answers=probe_answers)

    try:
        await sandbox_entrypoint.main()
    finally:
        assert gateway.terminated or gateway.returncode is not None
    return gateway, output


async def test_the_entrypoint_writes_a_result_when_the_tools_stayed_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Unanswered calls at the end do not cost the policy its result while the tools still answer the probe.
    _gateway, output = await _run_entrypoint(monkeypatch, tmp_path, gateway_dies=False, trailing_unanswered=3)

    result = json.loads((output / "result.json").read_text())
    assert result["completion_status"] == "max_turns"
    assert result["n_mcp_trailing_unanswered_calls"] == 3
    assert (output / "final.zip").is_file()


@pytest.mark.parametrize(
    ("gateway_dies", "trailing_unanswered", "message"),
    [
        (True, 0, "Archipelago gateway exited early"),
        (False, 1, "went unanswered and the MCP endpoint no longer lists its tools"),
    ],
)
async def test_the_entrypoint_fails_a_rollout_that_returned_normally_on_a_dead_tool_plane(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gateway_dies: bool, trailing_unanswered: int, message: str
) -> None:
    """The turn cap on dead tools would otherwise leave a result.json, which the host scores as the policy's zero."""
    with pytest.raises(RuntimeError, match=message):
        await _run_entrypoint(
            monkeypatch,
            tmp_path,
            gateway_dies=gateway_dies,
            trailing_unanswered=trailing_unanswered,
            probe_answers=False,
        )

    output = tmp_path / "apex-gym" / "output"
    # No result file: the process exits non-zero, which the host classes as a retried sandbox_error.
    assert not (output / "result.json").exists()
    assert (output / "final.zip").is_file(), "the final snapshot is still kept for inspection"


def test_the_prebuilt_entrypoint_checks_the_tool_plane_before_it_records_a_result() -> None:
    source = inspect.getsource(prebuilt_world_entrypoint.main)

    check = source.index("await require_live_tool_plane(None, GATEWAY_URL, result)")
    assert source.index("await run_stirrup_rollout(") < check < source.index('"result.json"')
