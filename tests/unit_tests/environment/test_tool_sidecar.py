# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Resources Server's tools served from its tool sidecar: the tool host, and the prepare and state import routes.

The end-to-end tests stand a local directory in for the sidecar: its commands run as local processes there, so the
real tool host starts, is seeded, answers tool calls and exports its state, as it does in a sidecar.
"""

import asyncio
import gzip
import io
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import aiohttp
import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.client.server_client import ServerClient, rollout_path_prefix
from nemo_gym.config.config_types import BaseServerConfig
from nemo_gym.core.episode_types import EpisodeId, TaskId
from nemo_gym.environment import tool_sidecar
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
