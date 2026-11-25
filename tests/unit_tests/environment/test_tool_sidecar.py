# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Resources Server's tools served from its tool sidecar: the tool host, and the prepare and state import routes.

The end-to-end tests stand a local directory in for the sidecar: its commands run as local processes there, so the
real tool host starts, is seeded, answers tool calls and exports its state, as it does in a sidecar.
"""

import asyncio
import gzip
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import aiohttp
import pytest
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.client.server_client import ServerClient, rollout_path_prefix
from nemo_gym.config.config_types import BaseServerConfig
from nemo_gym.core.episode_types import EpisodeId, TaskId
from nemo_gym.environment import tool_sidecar, tool_sidecar_host
from nemo_gym.environment.tool_sidecar import ToolSidecarResources, _SidecarShell, secret_paths
from nemo_gym.environment.tool_sidecar_host import build_app
from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle, SandboxStatus
from nemo_gym.server_runtime.attempt_sandbox import ResourcesPrepareSandboxRequest, SandboxGroupSpec
from nemo_gym.server_runtime.base_resources_server import (
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    SimpleResourcesServer,
)
from tests.unit_tests.environment.fixtures.tool_server.app import LedgerConfig, LedgerResourcesServer


SESSION = "resources-session-1"


def _config(port: int = 8000, **overrides: Any) -> LedgerConfig:
    return LedgerConfig.model_validate(
        {
            "name": "ledger",
            "host": "127.0.0.1",
            "port": 9999,
            "entrypoint": "app.py",
            "tool_sidecar": {"image": "python:3.13-slim", "port": port, "ready_timeout_seconds": 60},
        }
        | overrides
    )


def _server(config: LedgerConfig) -> LedgerResourcesServer:
    client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=OmegaConf.create({})
    )
    return LedgerResourcesServer(config=config, server_client=client)


def _seed(opening: Any = "hi") -> dict[str, Any]:
    return ResourcesSeedSessionRequest(
        resources_session_id=SESSION,
        episode_id=EpisodeId(rollout_id="rollout", attempt=1),
        task_id=TaskId(taskset="ledger:test", task_id="t-1"),
        task_data={"opening": opening},
    ).model_dump(mode="json")


def test_a_tool_sidecar_cannot_also_serve_the_tools_over_mcp() -> None:
    with pytest.raises(ValidationError, match="expose_tools_over_mcp"):
        _config(expose_tools_over_mcp=True)


class SecretConfig(LedgerConfig):
    judge_api_key: str | None = None
    refresh_token: bool = False


def test_a_config_that_holds_a_credential_cannot_serve_its_tools_from_a_sidecar() -> None:
    with pytest.raises(ValidationError, match=r"\['judge_api_key'\]"):
        SecretConfig.model_validate(_config().model_dump() | {"judge_api_key": "sk-123"})
    # The same credential is fine while the tools run in process, and a flag named like a credential is no credential.
    assert SecretConfig.model_validate(_config().model_dump() | {"judge_api_key": "sk-123", "tool_sidecar": None})
    assert SecretConfig.model_validate(_config().model_dump() | {"refresh_token": True})
    assert secret_paths(
        {
            "judge": {"API_KEY": "x", "private_key": "k"},
            "tokens": ["a"],
            "passwords": "",
            "aws_secret_access_key": "s",
            "auth": {"user": "u"},
            "base_url": "https://user:pass@example.com/v1",  # pragma: allowlist secret
            "plain_url": "https://example.com/v1",
        }
    ) == ["judge.API_KEY", "judge.private_key", "tokens", "aws_secret_access_key", "auth", "base_url"]


def test_the_tool_host_serves_the_tools_and_their_state_but_not_grading() -> None:
    config = _config()
    spec = _server(config).tool_host_spec()
    # Relative to the directory above the server's own, as resources_servers/<name>/app.py is to the repository.
    assert spec["module"] == "fixtures.tool_server.app"
    assert spec["class"] == "LedgerResourcesServer"
    # The tool host serves the tools itself.
    assert spec["config"]["tool_sidecar"] is None
    # In this process the fixture's module is importable under its test package's name.
    app = build_app(spec | {"module": "tests.unit_tests.environment.fixtures.tool_server.app"})

    # The Resources Server calls from inside the sidecar; the agent container calls from its own address.
    with TestClient(app, client=("127.0.0.1", 40000)) as local, TestClient(app, client=("10.0.0.7", 40000)) as agent:
        assert agent.get("/_tool_sidecar/ready").json() == {"ready": True}
        # Only the Resources Server seeds the tool host.
        assert agent.post("/seed_session", json=_seed()).status_code == 403
        assert local.post("/seed_session", json=_seed()).status_code == 200
        agent.cookies.clear()
        agent.cookies.update(dict(local.cookies))
        assert agent.post("/pay", json={"value": 3}).text == "pay #1"
        # Only the Resources Server reads the state, and the catch-all tool route does not answer for that route. A
        # rollout prefix, which the server strips before routing, does not get around that.
        assert agent.post("/_tool_sidecar/state").status_code == 403
        prefix = rollout_path_prefix("rollout-1")
        assert agent.post(f"{prefix}/_tool_sidecar/state").status_code == 403
        assert agent.post(f"{prefix}/seed_session", json=_seed()).status_code == 403
        assert local.post("/_tool_sidecar/state").json() == {"state": {"ledger": ["opening:hi", "pay:3"]}}
        # The session's lifecycle and grading stay with the Resources Server: these reach the tool route as unknown
        # tools, so nothing is scored or closed.
        for path in ("/verify", "/close_session", "/aggregate_metrics"):
            reply = agent.post(path, json={"resources_session_id": SESSION})
            assert (reply.status_code, "reward" in reply.text) == (200, False), path
        assert local.post("/_tool_sidecar/state").json()["state"]["ledger"][-1] == "aggregate_metrics:None"


class _RecordingSidecar:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    async def exec(self, *argv: str, timeout_s: float) -> SandboxExecResult:
        self.calls.append(argv)
        return SandboxExecResult(stdout="", stderr=None, return_code=0)

    async def upload(self, local_path: Any, remote_path: str) -> None:
        self.calls.append(("upload", remote_path))


async def test_the_runtime_installs_through_a_shell_in_the_sidecar() -> None:
    sidecar = _RecordingSidecar()
    shell = _SidecarShell(sidecar, "opensandbox")  # type: ignore[arg-type]

    await shell.exec("tar -xzf a.tgz", cwd="/tmp/x y", timeout_s=5, env={"HOME": "/tmp/h o"})
    await shell.exec("true", timeout_s=5)
    await shell.upload("/local/a", "/remote/a")

    assert sidecar.calls == [
        ("sh", "-c", "cd '/tmp/x y' && HOME='/tmp/h o' tar -xzf a.tgz"),
        ("sh", "-c", "true"),
        ("upload", "/remote/a"),
    ]
    assert shell.provider_name == "opensandbox"


def test_the_sources_hold_nemo_gym_and_the_servers_package_without_its_tests(tmp_path: Path) -> None:
    server_dir = tmp_path / "resources_servers" / "ledger"
    (server_dir / "tests").mkdir(parents=True)
    (server_dir / "app.py").write_text("SERVED = True\n")
    (server_dir / "tests" / "test_app.py").write_text("raise AssertionError\n")

    archive = tool_sidecar._sources_bytes(((str(server_dir), "resources_servers/ledger"),))

    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(archive))) as tar:
        names = tar.getnames()
    assert "resources_servers/ledger/app.py" in names
    assert "resources_servers/ledger/tests/test_app.py" not in names
    assert "nemo_gym/environment/tool_sidecar_host.py" in names
    # The same sources give the same bytes, so the uploaded archive is cached by its hash.
    tool_sidecar._sources_bytes.cache_clear()
    assert tool_sidecar._sources_bytes(((str(server_dir), "resources_servers/ledger"),)) == archive


class _LocalSidecarProvider:
    """A provider whose one sidecar is a local directory: its commands run as local processes."""

    name = "local_sidecar"

    def __init__(self) -> None:
        self.exec_argvs: list[list[str]] = []

    async def sidecar_exec(
        self, handle: SandboxHandle, sidecar: str, argv: list[str], *, timeout_s: float, on_stdout: Any = None
    ) -> SandboxExecResult:
        self.exec_argvs.append(argv)
        process = await asyncio.create_subprocess_exec(
            *argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
        return SandboxExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=process.returncode,
        )

    async def upload_sidecar_file(
        self, handle: SandboxHandle, sidecar: str, source_path: Path, target_path: str
    ) -> None:
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_path)

    async def download_sidecar_file(
        self, handle: SandboxHandle, sidecar: str, source_path: str, target_path: Path
    ) -> None:
        shutil.copyfile(source_path, target_path)

    async def connect(self, descriptor: dict[str, Any]) -> SandboxHandle:
        return SandboxHandle(sandbox_id=descriptor["sandbox_id"], provider_name=self.name, raw=None)

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id}

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        return SandboxStatus.RUNNING

    async def close(self, handle: SandboxHandle) -> None:
        raise AssertionError("a borrower never stops the group")

    async def aclose(self) -> None:
        return None


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def local_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The sidecar at ``tmp_path/sidecar``, borrowed through the local provider, with this interpreter as its runtime.

    Every tool host started there is killed afterwards.
    """
    root = tmp_path / "sidecar"
    monkeypatch.setattr(tool_sidecar, "_ROOT", str(root))
    monkeypatch.setattr(tool_sidecar, "_HOST", f"{root}/tool_host/nemo_gym/environment/tool_sidecar_host.py")

    class _InstalledRuntime:
        python = sys.executable

        async def is_installed(self, sandbox: Any) -> bool:
            return True

    monkeypatch.setattr(LedgerResourcesServer, "tool_host_runtime", lambda self: _InstalledRuntime())
    provider = _LocalSidecarProvider()

    async def borrow(access: SandboxAccess) -> AsyncSandbox:
        descriptor = dict(access.connection.descriptor)
        return await AsyncSandbox.connect(descriptor, provider=provider, owns_provider=False)

    monkeypatch.setattr(tool_sidecar, "borrow", borrow)
    yield SimpleNamespace(root=root, provider=provider)
    pid_file = root / "host.pid"
    if pid_file.exists():
        try:
            os.killpg(int(pid_file.read_text()), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _access() -> SandboxAccess:
    return SandboxAccess.model_validate(
        {
            "connection": {
                "kind": "direct",
                "provider_config_ref": "attempt_sandboxes",
                "descriptor": {"sandbox_id": "group-1", "sidecars": [{"name": "tools", "image": "python:3.13-slim"}]},
            },
            "workdir": "/workspace",
        }
    )


needs_setsid = pytest.mark.skipif(shutil.which("setsid") is None, reason="the tool host starts detached with setsid")


def _prepare(**task_data: Any) -> ResourcesPrepareSandboxRequest:
    seed = _seed(**task_data)
    return ResourcesPrepareSandboxRequest(
        resources_session_id=SESSION,
        episode_id=seed["episode_id"],
        task_id=seed["task_id"],
        task_data=seed["task_data"],
        sandbox_access=_access(),
    )


def _verify_body() -> dict[str, Any]:
    return {
        "responses_create_params": {"input": "pay twice"},
        "response": {
            "id": "response",
            "created_at": 0,
            "model": "model",
            "object": "response",
            "output": [],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "tools": [],
        },
    }


@needs_setsid
async def test_the_tool_state_the_agents_calls_leave_in_the_sidecar_is_what_the_host_grades(
    local_sidecar: Any,
) -> None:
    port = _free_port()
    server = _server(_config(port))
    with TestClient(server.setup_webserver()) as host:
        # The Resources seed declares the tool sidecar in the attempt's group, and that it prepares it.
        seeded = host.post("/seed_session", json=_seed()).json()
        [sidecar] = seeded["sandbox_group"]["sidecars"]
        assert {k: sidecar[k] for k in ("name", "image", "resources", "egress_hosts", "tools_port")} == {
            "name": "tools",
            "image": "python:3.13-slim",
            "resources": {"cpu": 1, "memory_mib": 2048},
            "egress_hosts": [],
            "tools_port": port,
        }
        assert seeded["sandbox_group"]["prepare"] is True
        cookies = dict(host.cookies)
        # No tool call falls back to the Resources host.
        assert host.post("/pay", json={"value": 1}).status_code == 409

        await server.prepare_sandbox(_prepare())

        # The agent's tool calls go to the tool host with the Resources session's cookies, which it reads as its own.
        async with aiohttp.ClientSession(cookies=cookies) as agent:
            for value in (5, 7):
                async with agent.post(f"http://127.0.0.1:{port}/pay", json={"value": value}) as reply:
                    assert (reply.status, await reply.text()) == (200, f"pay #{1 if value == 5 else 2}")
        assert server._ledgers[SESSION] == ["opening:hi"]
        # Before the copy, the Resources host would grade an untouched ledger.
        assert host.post(
            "/tool_sidecar/import_state", json={"sandbox_access": _access().model_dump(mode="json")}
        ).json() == {"imported": True}
        assert server._ledgers[SESSION] == ["opening:hi", "pay:5", "pay:7"]
        assert host.post("/verify", json=_verify_body()).json()["reward"] == 1.0

    # A second prepare of the same group finds the tool host answering and starts no second one.
    starts = [argv for argv in local_sidecar.provider.exec_argvs if "setsid nohup" in " ".join(argv)]
    await server.prepare_sandbox(_prepare())
    assert [argv for argv in local_sidecar.provider.exec_argvs if "setsid nohup" in " ".join(argv)] == starts
    assert len(starts) == 1


@needs_setsid
async def test_a_tool_host_that_exits_fails_the_prepare_with_its_log(
    local_sidecar: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(_config(_free_port()))
    spec: Callable[[], dict[str, Any]] = server.tool_host_spec
    monkeypatch.setattr(
        LedgerResourcesServer, "tool_host_spec", lambda self: spec() | {"class": "MissingResourcesServer"}
    )
    with pytest.raises(RuntimeError, match=r"(?s)tool host exited.*MissingResourcesServer"):
        await server.prepare_sandbox(_prepare())


@needs_setsid
async def test_a_seed_the_tool_host_refuses_fails_the_prepare(local_sidecar: Any) -> None:
    server = _server(_config(_free_port()))

    # No retry changes a refused seed, so it fails the attempt for good.
    with pytest.raises(ResourcesSeedRejectedError, match="refused the seed with HTTP 422"):
        await server.prepare_sandbox(_prepare(opening=["not", "a", "string"]))


async def test_a_server_whose_tools_run_in_process_prepares_nothing() -> None:
    # Nothing to borrow: a prepare that tried would fail to resolve the access's provider.
    assert await _server(_config().model_copy(update={"tool_sidecar": None})).prepare_sandbox(_prepare()) is None


async def test_a_server_whose_tools_run_in_process_declares_no_group_and_serves_no_sidecar_routes() -> None:
    server = _server(_config().model_copy(update={"tool_sidecar": None}))
    app = server.setup_webserver()
    paths = {getattr(route, "path", None) for route in app.router.routes}

    with TestClient(app) as host:
        assert "sandbox_group" not in host.post("/seed_session", json=_seed()).json()
    assert "/prepare_sandbox" not in paths

    assert "/tool_sidecar/import_state" not in paths
    assert "/tool_sidecar/import_state" in {
        getattr(route, "path", None) for route in _server(_config()).setup_webserver().router.routes
    }


class _ServerThatPrepares(SimpleResourcesServer):
    """A server that declares a file of its own and prepares the group itself."""

    prepared: ClassVar[list[str]] = []

    def sandbox_group(self, body: ResourcesSeedSessionRequest) -> SandboxGroupSpec | None:
        return SandboxGroupSpec(files={"/task/README": "ledger"})

    async def prepare_sandbox(self, body: ResourcesPrepareSandboxRequest) -> None:
        type(self).prepared.append(body.resources_session_id)

    async def verify(self, body: Any) -> Any:
        raise AssertionError("not graded here")


class _ToolsInASidecarToo(ToolSidecarResources, _ServerThatPrepares):
    config: LedgerConfig


def _mixed(config: LedgerConfig) -> _ToolsInASidecarToo:
    return _ToolsInASidecarToo(config=config, server_client=_server(config).server_client)


def test_the_tool_sidecar_joins_the_group_the_server_declares() -> None:
    group = _mixed(_config()).sandbox_declaration(ResourcesSeedSessionRequest.model_validate(_seed()))["sandbox_group"]

    assert group.files == {"/task/README": "ledger"}
    assert [sidecar.name for sidecar in group.sidecars] == ["tools"]
    assert group.prepare is True


async def test_a_server_that_prepares_the_group_itself_still_does_so_without_a_tool_sidecar() -> None:
    _ServerThatPrepares.prepared = []
    server = _mixed(_config().model_copy(update={"tool_sidecar": None}))

    assert server._prepares_sandbox() is True
    await server.prepare_sandbox(_prepare())

    assert _ServerThatPrepares.prepared == [SESSION]


class _OwnGroupLedger(LedgerResourcesServer):
    """A server whose own sandbox_group override does not call super()."""

    def sandbox_group(self, body: ResourcesSeedSessionRequest) -> SandboxGroupSpec | None:
        return SandboxGroupSpec(files={"/task/README": "own"})


def test_a_servers_own_group_override_keeps_the_tool_sidecar() -> None:
    server = _OwnGroupLedger(config=_config(), server_client=_server(_config()).server_client)

    group = server.sandbox_declaration(ResourcesSeedSessionRequest.model_validate(_seed()))["sandbox_group"]

    assert (group.files, [sidecar.name for sidecar in group.sidecars], group.prepare) == (
        {"/task/README": "own"},
        ["tools"],
        True,
    )


def test_the_tool_host_runs_from_its_spec_and_answers_the_calls_the_resources_server_sends_from_the_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    port = _free_port()
    spec = _server(_config(port)).tool_host_spec() | {
        "module": "tests.unit_tests.environment.fixtures.tool_server.app"
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    started: list[uvicorn.Server] = []

    def run(app: Any, **options: Any) -> None:
        # uvicorn.run with a handle, so the test can stop the tool host it serves.
        server = uvicorn.Server(uvicorn.Config(app, **options))
        started.append(server)
        server.run()

    monkeypatch.setattr(uvicorn, "run", run)
    # Nothing answers on the port yet.
    assert tool_sidecar_host.main(["tool_sidecar_host.py", "--ready", str(port)]) == 1

    thread = threading.Thread(target=tool_sidecar_host.main, args=(["tool_sidecar_host.py", str(spec_path)],))
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while tool_sidecar_host.main(["tool_sidecar_host.py", "--ready", str(port)]) != 0:
            assert thread.is_alive() and time.monotonic() < deadline, "the tool host did not start"
            time.sleep(0.05)
        # The agent container reaches the tool host by the sidecar's address.
        assert (started[0].config.host, started[0].config.port) == ("0.0.0.0", port)

        def call(path: str, body: Any, cookies: dict[str, str]) -> dict[str, Any]:
            request, reply = tmp_path / "request.json", tmp_path / "reply.json"
            request.write_text(json.dumps({"method": "POST", "path": path, "body": body, "cookies": cookies}))
            argv = ["tool_sidecar_host.py", "--call", str(port), str(request), str(reply)]
            assert tool_sidecar_host.main(argv) == 0
            return json.loads(reply.read_text())

        seeded = call("/seed_session", _seed(), {})
        assert seeded["status"] == 200
        # The seed's session cookie names the session in later calls, as the Resources session's cookie does.
        assert seeded["cookies"]
        assert call("/pay", {"value": 4}, seeded["cookies"])["body"] == "pay #1"
        assert json.loads(call("/_tool_sidecar/state", {}, seeded["cookies"])["body"]) == {
            "state": {"ledger": ["opening:hi", "pay:4"]}
        }
        # Without the cookie the state route names no session, and the reply still reaches the caller.
        assert call("/_tool_sidecar/state", {}, {})["status"] == 500
        # The reply is written whole: no partial file is left beside it.
        assert sorted(path.name for path in tmp_path.iterdir()) == ["reply.json", "request.json", "spec.json"]
    finally:
        if started:
            started[0].should_exit = True
        thread.join(30)
    assert not thread.is_alive()


class _ScriptedSidecar:
    """A sidecar whose commands answer from ``answers``: the first pattern found in the command picks the result."""

    def __init__(self, answers: list[tuple[str, SandboxExecResult]]) -> None:
        self.answers = answers
        self.commands: list[str] = []
        self.uploads: list[str] = []

    async def exec(self, *argv: str, timeout_s: float) -> SandboxExecResult:
        command = " ".join(argv)
        self.commands.append(command)
        for pattern, result in self.answers:
            if pattern in command:
                return result
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def upload(self, local_path: Any, remote_path: str) -> None:
        self.uploads.append(remote_path)

    async def download(self, remote_path: str, local_path: Path) -> None:
        raise AssertionError("no reply to download")


class _FakeRuntime:
    python = "/opt/tool-host/bin/python"

    def __init__(self) -> None:
        self.installs = 0

    async def is_installed(self, sandbox: Any) -> bool:
        return False

    async def install(self, sandbox: Any, *, timeout_s: float) -> None:
        self.installs += 1


_NOT_READY = ("--ready", SandboxExecResult(stdout="", stderr="", return_code=1))


async def test_a_sidecar_without_the_runtime_gets_it_and_a_failed_step_names_its_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr(LedgerResourcesServer, "tool_host_runtime", lambda self: runtime)
    config = _config()
    sidecar = _ScriptedSidecar(
        [_NOT_READY, ("mkdir -p", SandboxExecResult(stdout="", stderr="read-only", return_code=1))]
    )

    with pytest.raises(RuntimeError, match="Could not prepare its directories in the tool sidecar: read-only"):
        await _server(config)._start_tool_host(sidecar, "opensandbox", config.tool_sidecar)  # type: ignore[arg-type]

    assert runtime.installs == 1
    # Nothing was uploaded or started after the failed step.
    assert sidecar.uploads == []
    assert not any("setsid" in command for command in sidecar.commands)


async def test_a_tool_host_whose_process_cannot_be_checked_fails_the_prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LedgerResourcesServer, "tool_host_runtime", lambda self: _FakeRuntime())
    config = _config()
    sidecar = _ScriptedSidecar(
        [_NOT_READY, ("kill -0", SandboxExecResult(stdout=None, stderr=None, return_code=None, error_type="timeout"))]
    )

    with pytest.raises(RuntimeError, match="tool host's process could not be checked: timeout"):
        await _server(config)._start_tool_host(sidecar, "opensandbox", config.tool_sidecar)  # type: ignore[arg-type]

    # The sources and the spec went up before the tool host was started.
    assert sidecar.uploads == [
        "/tmp/nemo-gym-tool-sidecar/sources.tar.gz",
        "/tmp/nemo-gym-tool-sidecar/spec.json",
    ]
    assert any("setsid nohup" in command for command in sidecar.commands)


class _BorrowedGroup:
    def __init__(self) -> None:
        self.disconnected = 0
        self.sidecars: list[str] = []

    def sidecar(self, name: str) -> str:
        self.sidecars.append(name)
        return name

    @property
    def provider_name(self) -> str:
        return "opensandbox"

    async def disconnect(self) -> None:
        self.disconnected += 1


@pytest.fixture
def scripted_tool_host(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Borrow a group whose tool host answers each call with the next of ``replies``."""
    state = SimpleNamespace(group=_BorrowedGroup(), replies=[], calls=[])

    async def borrow(access: SandboxAccess) -> _BorrowedGroup:
        return state.group

    async def call(sidecar: str, python: str, port: int, method: str, path: str, body: Any, *, cookies: Any) -> Any:
        state.calls.append((sidecar, path, dict(cookies)))
        return state.replies.pop(0)

    async def started(self: Any, sidecar: Any, provider_name: str, config: Any) -> None:
        return None

    monkeypatch.setattr(tool_sidecar, "borrow", borrow)
    monkeypatch.setattr(tool_sidecar, "_call", call)
    monkeypatch.setattr(LedgerResourcesServer, "_start_tool_host", started)
    return state


async def test_a_seed_the_tool_host_fails_is_an_infrastructure_failure(scripted_tool_host: SimpleNamespace) -> None:
    scripted_tool_host.replies = [{"status": 503, "body": "starting", "cookies": {}}]

    # A RuntimeError, unlike a refused seed, is retried on a fresh group.
    with pytest.raises(RuntimeError, match="failed the seed with HTTP 503: starting"):
        await _server(_config()).prepare_sandbox(_prepare())

    assert scripted_tool_host.group.disconnected == 1


async def test_tool_state_that_the_tool_host_does_not_export_is_not_imported(
    scripted_tool_host: SimpleNamespace,
) -> None:
    server = _server(_config())
    access = {"sandbox_access": _access().model_dump(mode="json")}
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as host:
        host.post("/seed_session", json=_seed())
        scripted_tool_host.replies = [{"status": 500, "body": "boom", "cookies": {}}]
        assert host.post("/tool_sidecar/import_state", json=access).status_code == 500
        # The state route answers None for tools that keep no state, and the session here is left as it was.
        scripted_tool_host.replies = [{"status": 200, "body": '{"state": null}', "cookies": {}}]
        assert host.post("/tool_sidecar/import_state", json=access).json() == {"imported": False}
    assert server._ledgers[SESSION] == ["opening:hi"]
    # Each call read the session the Resources session's cookie names, and every borrow was let go.
    assert [(sidecar, path) for sidecar, path, _ in scripted_tool_host.calls] == [
        ("tools", "/_tool_sidecar/state")
    ] * 2
    assert all(cookies for _, _, cookies in scripted_tool_host.calls)
    assert scripted_tool_host.group.disconnected == 2


def test_the_tool_host_installs_the_servers_own_requirements_after_gyms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LedgerResourcesServer, "tool_sidecar_requirements", ("ledger-backend==1.2",))

    requirements = _server(_config()).tool_host_runtime().requirements

    assert requirements[: len(tool_sidecar.TOOL_HOST_REQUIREMENTS)] == tool_sidecar.TOOL_HOST_REQUIREMENTS
    assert requirements[-1] == "ledger-backend==1.2"


def test_a_server_that_signs_sessions_with_a_secret_cannot_serve_its_tools_from_a_sidecar() -> None:
    signed = type("SignedLedger", (LedgerResourcesServer,), {"signs_sessions_with_run_secret": True})

    with pytest.raises(ValueError, match="signs sessions with a secret the tool host does not share"):
        signed(config=_config(), server_client=_server(_config()).server_client)
    # With the tools in process, the server may sign with a secret.
    signed(config=_config().model_copy(update={"tool_sidecar": None}), server_client=_server(_config()).server_client)


def test_a_server_whose_tools_run_in_process_has_no_tool_host() -> None:
    with pytest.raises(HTTPException) as refused:
        _server(_config().model_copy(update={"tool_sidecar": None})).tool_host_spec()
    assert refused.value.status_code == 404


def test_a_credential_is_found_under_a_header_name_a_key_name_or_a_position_in_a_list() -> None:
    assert secret_paths(
        {
            "headers": {"X-Api-Key": "k", "Accept": "application/json"},
            "secret_key": "s",
            "exa_key": "e",
            "sort_key": 3,
            "session_cookie": "c",
            "mirrors": ["https://example.com", "https://user:pass@example.com"],  # pragma: allowlist secret
        }
    ) == ["headers.X-Api-Key", "secret_key", "exa_key", "session_cookie", "mirrors[1]"]


async def test_verify_grades_only_a_session_whose_tool_state_was_read_back_from_the_sidecar(
    scripted_tool_host: SimpleNamespace,
) -> None:
    access = {"sandbox_access": _access().model_dump(mode="json")}
    with TestClient(_server(_config()).setup_webserver()) as host:
        host.post("/seed_session", json=_seed())
        # An Environment Server that never read the state back would grade a ledger no tool call touched.
        refused = host.post("/verify", json=_verify_body())
        assert refused.status_code == 409
        assert "never read back" in refused.json()["detail"]

        scripted_tool_host.replies = [
            {"status": 200, "body": '{"state": {"ledger": ["opening:hi", "pay:2"]}}', "cookies": {}}
        ]
        assert host.post("/tool_sidecar/import_state", json=access).json() == {"imported": True}
        assert host.post("/verify", json=_verify_body()).json()["reward"] == 1.0
        # This server keeps no grade to replay, so another /verify of the session needs another read.
        assert host.post("/verify", json=_verify_body()).status_code == 409


async def test_a_failed_call_still_removes_the_seed_it_left_in_the_sidecar() -> None:
    sidecar = _ScriptedSidecar(
        [
            ("--call", SandboxExecResult(stdout="", stderr="tool host died", return_code=1)),
            ("rm -f", SandboxExecResult(stdout="", stderr="busy", return_code=1)),
        ]
    )

    # The call's own failure is raised, not the failed removal after it.
    with pytest.raises(
        RuntimeError, match="Could not call /seed_session on the tool host in the tool sidecar: tool host"
    ):
        await tool_sidecar._call(sidecar, "/py", 8000, "POST", "/seed_session", {"task_data": {}}, cookies={})  # type: ignore[arg-type]

    [uploaded] = sidecar.uploads
    assert any(command.startswith("rm -f") and uploaded in command for command in sidecar.commands)


class _LedgerWithALookup(LedgerResourcesServer):
    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.get("/lookup")(self.lookup)
        return app

    async def lookup(self) -> dict[str, str]:
        return {"found": "in process"}


def test_no_tool_route_answers_on_the_resources_host_whatever_its_method() -> None:
    server = _LedgerWithALookup(config=_config(), server_client=_server(_config()).server_client)
    with TestClient(server.setup_webserver()) as host:
        assert host.get("/lookup").status_code == 409
        # The base's own routes still answer.
        assert host.get("/reverify_mode").status_code == 200


class _StatefulWithoutStateHooks(ToolSidecarResources, SimpleResourcesServer):
    config: LedgerConfig

    def has_session_state(self, session_id: str) -> bool:
        return False

    async def initialize_session_state(self, session_id: str, body: Any) -> None:
        return None

    async def release_session_state(self, session_id: str) -> None:
        return None

    async def verify(self, body: Any) -> Any:
        raise AssertionError("not graded here")


def test_a_server_that_keeps_session_state_must_bring_its_tool_state_to_verify() -> None:
    client = _server(_config()).server_client
    with pytest.raises(ValueError, match=r"implement \['export_tool_state', 'import_tool_state'\]"):
        _StatefulWithoutStateHooks(config=_config(), server_client=client)
    # With the tools in process, there is no state to bring back.
    _StatefulWithoutStateHooks(config=_config().model_copy(update={"tool_sidecar": None}), server_client=client)


class _Lookup(ToolSidecarResources, SimpleResourcesServer):
    """Tools that keep no state, such as a lookup: the tool host has no state to hand back."""

    config: LedgerConfig

    async def verify(self, body: Any) -> Any:
        raise AssertionError("not graded here")


def test_a_tool_host_whose_tools_keep_no_state_exports_none() -> None:
    lookup = _Lookup(config=_config(), server_client=_server(_config()).server_client)
    spec = lookup.tool_host_spec() | {"module": __name__, "class": "_Lookup"}

    with TestClient(build_app(spec), client=("127.0.0.1", 40000)) as local:
        local.post("/seed_session", json=_seed())
        assert local.post("/_tool_sidecar/state").json() == {"state": None}
