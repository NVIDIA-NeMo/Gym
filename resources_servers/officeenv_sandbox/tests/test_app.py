# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sandboxed OfficeEnv server: each Environment Server episode borrows a sandbox, reaches OfficeEnv's tools over MCP and
moves file content over the episode's file API, and the server refuses callers that are not Gym's.
Sandboxes come from a fake provider, so these tests start no container. Environment Server episodes
without a sandbox and the MCP exposure itself are tested in resources_servers/officeenv/tests/test_app.py. What each app does with file
content in workspace mode is OfficeEnv's, and its tests' (tests/test_workspace.py); these test the
server's side of the mode, with Google Drive as the example app and one Gmail attachment flow: the
routes, the reads and writes of the sandbox's files with their checks, the refusals and the metrics."""

import asyncio
import base64
import hashlib
import io
import ipaddress
import json
import logging
import os
import re
import shlex
import shutil
import socket
import stat
import statistics
import subprocess
import threading
import time
import typing
import zipfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anyio
import httpx
import pytest
import yaml
from fastapi import HTTPException
from fastapi.testclient import TestClient


# OfficeEnv installs apart from the server's locked dependencies until it is published: without it these tests skip.
pytest.importorskip("officeenv", reason="OfficeEnv is not installed (README.md, 'The OfficeEnv version')")

from officeenv import __version__ as officeenv_version
from officeenv.files import FileApp, FileCallError, FileContent, ReadPlan
from officeenv.service import Reply

from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import (
    NEMO_GYM_MCP_SESSION_TOKEN_HEADER,
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    check_resources_session_contract,
    maybe_auto_expose,
    sandbox_lending,
)
from nemo_gym.server_runtime.base_resources_server import mcp_session_token_serializer
from resources_servers.officeenv.app import (
    OfficeEnvResourcesServer,
    OfficeEnvResourcesServerConfig,
    tools_exposure_catalog,
)
from resources_servers.officeenv_sandbox import app as sandbox_app
from resources_servers.officeenv_sandbox.app import (
    APP_COMMAND_PATH,
    APP_COMMAND_SOURCE,
    CLI_USER_AGENT,
    CREDENTIALS_PATH,
    DOCS_TOOLS,
    RELAY_PATH,
    RELAY_SOURCE,
    SANDBOX_EXPOSURES,
    SEARCH_TOOLS,
    STAT_SCRIPT,
    TOOL_SETS,
    UNLIMITED_INLINE_FILE_BYTES,
    UPLOAD_HINT,
    OfficeEnvSandboxConfig,
    OfficeEnvSandboxResourcesServer,
    OfficeEnvSandboxResourcesServerConfig,
    ToolSet,
    api_credentials,
    egress_urls,
    is_trusted_caller,
    relay_ports,
    with_http_calls,
    workspace_guide,
)


pytest.importorskip("mcp")

SANDBOX_ROWS = Path(__file__).parents[1] / "data" / "sandbox_pilot.jsonl"
SANDBOX_CONFIG = Path(__file__).parents[1] / "configs" / "officeenv_sandbox.yaml"
ME = "alex.morgan@example.com"
RPC_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
LEDGER = "invoice_id,vendor,amount\nINV-1,Acme,10.50\nINV-2,Globex,4.25\n"
TOTALS = "vendor,total\nAcme,10.50\nGlobex,4.25\n"
EXPECTED = [{"vendor": "Acme", "total": 10.5}, {"vendor": "Globex", "total": 4.25}]
SPEC = {
    "task_id": "sandbox-files",
    "actor": {"email": ME},
    "apps": {
        "gdrive": {
            "data": {
                "users": [{"emailAddress": ME, "displayName": "Alex Morgan"}],
                "files": [
                    {"id": "F_FIN", "name": "Finance", "type": "folder", "owner": ME},
                    {"id": "F_LEDGER", "name": "ledger.csv", "parents": ["F_FIN"], "owner": ME, "content": LEDGER},
                ],
            }
        },
        "user": {},
    },
    "exposure": {"mode": "tools"},
    "checks": [
        {
            "check": "world.records_match",
            "args": {
                "app": "gdrive",
                "collection": "files",
                "where": {"name": "totals.csv"},
                "new_only": True,
                "content": {"file_field": "content", "key": "vendor", "mode": "exact", "expected": EXPECTED},
            },
        }
    ],
}
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
XLSX_CHECK = {
    "check": "world.records_match",
    "args": {
        "app": "gdrive",
        "collection": "files",
        "where": {"name": "totals.xlsx"},
        "new_only": True,
        "content": {
            "file_field": "content",
            "file_format": "xlsx",
            "sheet": "Totals",
            "key": "vendor",
            "mode": "exact",
            "expected": EXPECTED,
        },
    },
}
RESPONSE = {
    "id": "resp_1",
    "created_at": 0.0,
    "model": "x",
    "object": "response",
    "output": [],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}
VERIFY_BODY = {"responses_create_params": {"input": []}, "response": RESPONSE}
# Gym's run-wide shutdown grace, as a config sets it, beside the provider config the fixture serves: the server
# keeps 2 s of it for the exit
GRACE_OF_3_S = {"sandbox": {"docker": {}}, "server_shutdown_timeout_seconds": 3.0}
IMAGE = {"image": "officeenv-sandbox:pilot"}
LOCAL = ("127.0.0.1", 50000)
SANDBOX = ("203.0.113.47", 50000)  # a public address (TEST-NET-3), as a cluster node sees a cloud sandbox's calls


def workbook(rows: list[list]) -> bytes:
    """A minimal XLSX workbook holding ``rows`` on one sheet named Totals (inline strings)."""
    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

    def cell(value) -> str:
        if isinstance(value, str):
            return f'<c t="inlineStr"><is><t>{value}</t></is></c>'
        return f"<c><v>{value}</v></c>"

    sheet = "".join(f"<row>{''.join(cell(v) for v in row)}</row>" for row in rows)
    parts = {
        "xl/workbook.xml": f'<workbook xmlns="{main}" xmlns:r="{rel}"><sheets>'
        '<sheet name="Totals" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
        f'relationships"><Relationship Id="rId1" Type="{rel}/worksheet" Target="worksheets/sheet1.xml"/>'
        "</Relationships>",
        "xl/worksheets/sheet1.xml": f'<worksheet xmlns="{main}"><sheetData>{sheet}</sheetData></worksheet>',
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in parts.items():
            archive.writestr(name, text)
    return buffer.getvalue()


BOOK = workbook([["vendor", "total"], ["Acme", 10.5], ["Globex", 4.25]])


class FakeSandbox:
    """A sandbox provider stand-in: records what each sandbox was started with, ran, was given and when it stopped."""

    started: list["FakeSandbox"] = []
    stopped: list["FakeSandbox"] = []
    fail_on: str | None = None  # "start", "serialize", "upload" or "stop": that step raises
    refusing: tuple[str, ...] = ()  # commands that start with one of these exit 1
    on_stop: Callable[[], None] | None = None  # runs as a sandbox stops
    on_download: Callable[[], None] | None = None  # runs as a file is downloaded from a sandbox
    links: dict[str, str] = {}  # sandbox paths that hold a link, and the path it points to
    folders: set[str] = set()  # sandbox paths that hold a folder
    kinds: dict[str, str] = {}  # what stat says of a path, for one the fake holds no bytes for (a device, a huge file)
    partial: dict[str, bytes] = {}  # paths whose copy is cut short: these bytes arrive, then the download raises

    def __init__(self, provider_config: dict) -> None:
        self.provider_config = provider_config
        self.commands: list[str] = []
        self.files: dict[str, bytes] = {}  # what the server uploaded into it, by path
        self.modes: dict[str, int] = {}  # the mode of each upload's local file
        self.downloaded: list[str] = []  # the paths copied out of it, in order
        self.sandbox_id: str | None = None  # the provider's id, once started
        self.scopes: list[str | None] = []  # the scope of each descriptor asked of it

    async def start(self, spec) -> None:
        self.spec = spec
        if FakeSandbox.fail_on == "start":
            raise RuntimeError("the provider could not create the sandbox")
        FakeSandbox.started.append(self)
        self.sandbox_id = f"sbx-{len(FakeSandbox.started)}"

    async def start_with_setup(self, spec, setup) -> None:
        await self.start(spec)
        await setup(self)

    async def exec(self, command: str, cwd: str | None = None, timeout_s: float | None = None) -> SimpleNamespace:
        self.commands.append(command)
        argv = shlex.split(command) if command.startswith("sh -c ") else []
        if argv[:3] == ["sh", "-c", STAT_SCRIPT]:  # the attachments' check: each path's type and size, one per line
            return SimpleNamespace(
                error_type=None, return_code=0, stdout="".join(f"{self.stat(p)}\n" for p in argv[4:])
            )
        refused = command.startswith(FakeSandbox.refusing)
        return SimpleNamespace(error_type=None, return_code=int(refused), stdout="", stderr="refused" * refused)

    def stat(self, path: str) -> str:
        """What STAT_SCRIPT prints of ``path``, a final link checked as a link and not followed."""
        if path in FakeSandbox.kinds:
            return FakeSandbox.kinds[path]
        if path in FakeSandbox.links:
            return "link|0"
        if path in FakeSandbox.folders:
            return "other|0"
        data = self.files.get(path, FakeSandbox.partial.get(path))
        return "missing|0" if data is None else f"regular file|{len(data)}"

    async def upload(self, local_path: str, remote_path: str) -> None:
        if FakeSandbox.fail_on == "upload":
            raise RuntimeError("the sandbox is gone")
        self.files[remote_path] = Path(local_path).read_bytes()
        self.modes[remote_path] = stat.S_IMODE(os.stat(local_path).st_mode)

    async def download(self, remote_path: str, local_path) -> None:
        """A file the server wrote, or the test as the model; a link or a folder is copied as it is, as ``docker cp``
        copies them."""
        if FakeSandbox.on_download is not None:
            FakeSandbox.on_download()
        self.downloaded.append(remote_path)
        if remote_path in FakeSandbox.partial:  # the copy is cut short: part of the file arrives, then an error
            Path(local_path).write_bytes(FakeSandbox.partial[remote_path])
            raise RuntimeError("the copy was cut short")
        if remote_path in FakeSandbox.links:
            os.symlink(FakeSandbox.links[remote_path], local_path)
        elif remote_path in FakeSandbox.folders:
            os.mkdir(local_path)
        elif remote_path in self.files:
            Path(local_path).write_bytes(self.files[remote_path])
        else:
            raise FileNotFoundError(remote_path)

    async def serialize(self, *, scope: str | None = None) -> dict:
        self.scopes.append(scope)
        if FakeSandbox.fail_on == "serialize":
            raise RuntimeError("the provider cannot describe the sandbox")
        return {"container": "c-1"}

    async def stop(self) -> None:
        if FakeSandbox.fail_on == "stop":
            raise RuntimeError("the provider could not stop the sandbox")
        if FakeSandbox.on_stop is not None:
            FakeSandbox.on_stop()
        FakeSandbox.stopped.append(self)


@pytest.fixture
def sandboxes(monkeypatch) -> type[FakeSandbox]:
    FakeSandbox.started, FakeSandbox.stopped, FakeSandbox.fail_on, FakeSandbox.on_stop = [], [], None, None
    FakeSandbox.refusing, FakeSandbox.on_download, FakeSandbox.links, FakeSandbox.folders = (), None, {}, set()
    FakeSandbox.kinds, FakeSandbox.partial = {}, {}
    monkeypatch.setattr(sandbox_app, "AsyncSandbox", FakeSandbox)
    monkeypatch.setattr(sandbox_lending, "AsyncSandbox", FakeSandbox)  # Gym's LentSandboxes starts the sandboxes
    monkeypatch.setattr(sandbox_app, "get_global_config_dict", lambda: {"sandbox": {"docker": {}}})
    # LentSandboxes reads the config this server does, as a test sets it
    monkeypatch.setattr(sandbox_lending, "get_global_config_dict", lambda: sandbox_app.get_global_config_dict())
    return FakeSandbox


SETTINGS = {"host": "10.0.0.1", "port": 8080, "entrypoint": "", "name": "oe", "expose_tools_over_mcp": True}


def make_server(**config: object) -> OfficeEnvSandboxResourcesServer:
    """The sandboxed server, on the fixture's provider with the pilot image unless ``config`` says otherwise."""
    settings = {**SETTINGS, "sandbox_provider": "sandbox", "sandbox_config": IMAGE, **config}
    return OfficeEnvSandboxResourcesServer(
        config=OfficeEnvSandboxResourcesServerConfig(**settings), server_client=MagicMock(spec=ServerClient)
    )


def make_base_server(**config: object) -> OfficeEnvResourcesServer:
    """The OfficeEnv server the sandboxed one subclasses, which lends no sandbox."""
    return OfficeEnvResourcesServer(
        config=OfficeEnvResourcesServerConfig(**SETTINGS, **config), server_client=MagicMock(spec=ServerClient)
    )


class MCPClient(TestClient):
    """A test client whose lifespan starts the MCP session manager, then the MCP handshake."""

    def __enter__(self) -> "MCPClient":
        super().__enter__()
        params = {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}
        initialize = {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": params}
        self.post("/mcp", headers=RPC_HEADERS, json=initialize)
        self.post("/mcp", headers=RPC_HEADERS, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
        return self


def mcp_client(server: OfficeEnvResourcesServer, client: tuple[str, int] = LOCAL) -> MCPClient:
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    return MCPClient(app, base_url="http://127.0.0.1:8000", client=client)


def rpc(client: TestClient, method: str, params: dict, token: str) -> dict:
    headers = {**RPC_HEADERS, NEMO_GYM_MCP_SESSION_TOKEN_HEADER: token}
    return client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})


def seed_request(
    spec: dict = SPEC, *, session_id: str = "rs-1", rollout_id: str = "r1"
) -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id=rollout_id),
        task_id=TaskId(taskset="oe", task_id=spec["task_id"]),
        task_data={"officeenv": spec, "task_id": spec["task_id"]},
    )


def seed_body(spec: dict = SPEC, *, session_id: str = "rs-1", rollout_id: str = "r1") -> dict:
    return seed_request(spec, session_id=session_id, rollout_id=rollout_id).model_dump(mode="json")


def close_body(session_id: str = "rs-1", rollout_id: str = "r1") -> dict:
    return {"resources_session_id": session_id, "episode_id": EpisodeId(rollout_id=rollout_id).model_dump(mode="json")}


def file_auth(server: OfficeEnvResourcesServer, session_id: str = "rs-1") -> dict:
    return {"Authorization": f"Bearer {server._episodes[session_id].token}"}


def configured_mcp_settings() -> dict:
    """The MCP settings the sandbox configs give the server (configs/officeenv_sandbox_episodes.yaml)."""
    episodes = SANDBOX_CONFIG.with_name("officeenv_sandbox_episodes.yaml")
    settings = yaml.safe_load(episodes.read_text())["oe"]["resources_servers"]["officeenv_sandbox"]
    return {key: settings[key] for key in ("mcp_server_name", "mcp_bearer_token")}


def test_tool_names_fit_pis_limit_with_the_configured_server_name() -> None:
    names = set(tools_exposure_catalog())
    assert {"gdrive_create_file", "gdrive_download_file", "user_complete_task"} <= names
    prefix = configured_mcp_settings()["mcp_server_name"] + "_"  # Pi names a tool <server>_<tool>, in 64 characters
    assert all(len(prefix + name) <= 64 for name in names)


# Lending a sandbox per episode.


def test_a_seed_lends_a_sandbox_with_the_episode_file_api_and_close_stops_it(sandboxes) -> None:
    server = make_server()
    with mcp_client(server) as client:
        seeded = client.post("/seed_session", json=seed_body())
        assert seeded.status_code == 200, seeded.text
        access = seeded.json()["sandbox_access"]
        assert access["workdir"] == "/workspace" and access["connection"]["provider_config_ref"] == "sandbox"
        assert access["connection"]["descriptor"] == {"container": "c-1"}
        (sandbox,) = sandboxes.started
        assert sandbox.provider_config == {"docker": {}}
        assert sandbox.spec.image == "officeenv-sandbox:pilot" and sandbox.spec.workdir == "/workspace"
        assert sandbox.spec.metadata["task_id"] == "sandbox-files"
        assert sandbox.spec.env["COMPANY_API_URL"] == "http://10.0.0.1:8080"  # this server's host and port
        assert sandbox.spec.env["COMPANY_API_TOKEN"] == server._episodes["rs-1"].token
        (drive,) = server.service.file_apps("rs-1")  # its file apps, with OfficeEnv's note on each
        assert drive.app == "gdrive" and drive.folder == "drive"
        assert sandbox.spec.files == {
            "/workspace/README.md": workspace_guide(8192, ("gdrive", "user"), file_apps=[drive]),  # its apps
            # the URL and token again, for a shell started without the sandbox's environment
            CREDENTIALS_PATH: api_credentials("http://10.0.0.1:8080", server._episodes["rs-1"].token),
        }

        token = seeded.json()["resources_tools"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]
        listed = {tool["name"] for tool in rpc(client, "tools/list", {}, token).json()["result"]["tools"]}
        assert "gdrive_create_file" in listed and not any(name.startswith("slack_") for name in listed)

        assert client.post("/close_session", json=close_body(), cookies=seeded.cookies).status_code == 200
        assert sandboxes.stopped == [sandbox]
        assert server.service.store.get("rs-1") is None and not server._episodes and not server._tokens


def test_files_api_url_overrides_the_address_a_sandbox_calls(sandboxes) -> None:
    with mcp_client(make_server(files_api_url="http://node-7:8080/")) as client:
        assert client.post("/seed_session", json=seed_body()).status_code == 200
        assert sandboxes.started[0].spec.env["COMPANY_API_URL"] == "http://node-7:8080"


def test_a_repeated_seed_lends_the_same_sandbox(sandboxes) -> None:
    with mcp_client(make_server()) as client:
        first = client.post("/seed_session", json=seed_body())
        again = client.post("/seed_session", json=seed_body())
        assert again.status_code == 200 and again.json()["sandbox_access"] == first.json()["sandbox_access"]
        assert client.post("/seed_session", json=seed_body(rollout_id="r2")).status_code == 409
        assert len(sandboxes.started) == 1


def test_a_refused_seed_starts_no_sandbox(sandboxes) -> None:
    server = make_server()
    with mcp_client(server) as client:
        unknown_app = client.post("/seed_session", json=seed_body({**SPEC, "apps": {"nope": {}}}))
        assert unknown_app.status_code == 400
        # an exposure a sandbox cannot be given, and tools the MCP endpoint does not serve under their names
        for exposure in ({"mode": "mcp"}, {"mode": "tools", "options": {"separator": "."}}):
            not_served = client.post("/seed_session", json=seed_body({**SPEC, "exposure": exposure}))
            assert not_served.status_code == 400 and "unsupported_exposure" in not_served.text, not_served.text
        assert sandboxes.started == [] and not server._episodes and len(server.service.store) == 0


def test_an_api_search_episode_whose_documentation_tool_mcp_does_not_serve_is_refused(sandboxes, monkeypatch) -> None:
    """An OfficeEnv naming the documentation tool otherwise than MCP serves it (DOCS_TOOLS read from an OfficeEnv
    that lists no names) would leave the episode no documentation: its seed is refused before a sandbox starts."""
    monkeypatch.setattr(sandbox_app, "SEARCH_TOOLS", sandbox_app.SEARCH_TOOLS[:1])  # api_search alone
    server = make_server()
    with mcp_client(server) as client:
        refused = client.post("/seed_session", json=seed_body(in_exposure(SPEC, "api_search")))
        assert refused.status_code == 400 and "unsupported_exposure" in refused.text, refused.text
        assert "not ['api_fetch']" in refused.text
        assert sandboxes.started == [] and not server._episodes and len(server.service.store) == 0


@pytest.mark.parametrize("step", ["start", "serialize"])
def test_a_sandbox_that_fails_to_start_leaves_no_world_and_no_sandbox(sandboxes, step: str) -> None:
    sandboxes.fail_on = step
    server = make_server()
    with mcp_client(server) as client:
        with pytest.raises(RuntimeError, match="the provider"):
            client.post("/seed_session", json=seed_body())
        assert len(server.service.store) == 0 and not server._episodes and not server._tokens
        assert len(sandboxes.stopped) == 1  # stopped, or its provider closed when it never started
        sandboxes.fail_on = None
        assert client.post("/seed_session", json=seed_body()).status_code == 200  # the seed can be retried


def test_a_failed_seed_whose_stop_fails_is_stopped_by_its_sessions_next_seed(sandboxes, monkeypatch, caplog) -> None:
    """The provider cannot describe the started sandbox, and its stop fails too: the seed's own error answers, no
    world or token is left, and the retried seed stops the sandbox the failed one left before it lends a new one
    (Gym's LentSandboxes), which shutdown then stops."""
    sandboxes.fail_on = "serialize"
    stop = FakeSandbox.stop
    refused: list[FakeSandbox] = []

    async def stop_refusing_once(sandbox: FakeSandbox) -> None:
        if not refused:
            refused.append(sandbox)
            raise RuntimeError("the provider could not stop the sandbox")
        await stop(sandbox)

    monkeypatch.setattr(FakeSandbox, "stop", stop_refusing_once)
    server = make_server()
    with caplog.at_level("ERROR"):
        with mcp_client(server) as client:
            with pytest.raises(RuntimeError, match="the provider cannot describe the sandbox"):
                client.post("/seed_session", json=seed_body())
            assert len(server.service.store) == 0 and not server._episodes and not server._tokens
            assert sandboxes.stopped == [] and "Failed to stop the sandbox of a failed seed" in caplog.text
            sandboxes.fail_on = None
            assert client.post("/seed_session", json=seed_body()).status_code == 200  # a new sandbox is lent
            assert sandboxes.stopped == [sandboxes.started[0]] and len(sandboxes.started) == 2
        # leaving the client shuts the server down
    assert refused == [sandboxes.started[0]] and set(sandboxes.stopped) == set(sandboxes.started)
    assert "a reaper must stop them" not in caplog.text


def test_sandbox_episodes_keep_the_resources_session_contract(sandboxes, monkeypatch) -> None:
    # The contract's client calls from "testclient", which is no address; the caller check is tested below.
    monkeypatch.setattr(sandbox_app, "is_trusted_caller", lambda host, **_: True)
    server = make_server()

    def holds_state(session_id: str) -> bool:  # the episode, which holds its sandbox, or its world
        return session_id in server._episodes or session_id in server.service.store.sessions

    check_resources_session_contract(
        server.setup_webserver(), seed_request(), keeps_state=True, holds_state=holds_state, verify_body=VERIFY_BODY
    )
    assert sandboxes.started and sandboxes.stopped == sandboxes.started  # every sandbox lent was stopped


def test_a_failed_stop_keeps_the_episode_for_a_retried_close(sandboxes) -> None:
    server = make_server()
    with mcp_client(server) as client:
        assert client.post("/seed_session", json=seed_body()).status_code == 200
        sandboxes.fail_on = "stop"
        with pytest.raises(RuntimeError, match="could not stop"):
            client.post("/close_session", json=close_body())
        assert "rs-1" in server._episodes and server.service.store.get("rs-1") is not None
        sandboxes.fail_on = None
        assert client.post("/close_session", json=close_body()).status_code == 200
        assert len(sandboxes.stopped) == 1 and not server._episodes


def test_retiring_a_session_stops_its_sandbox(sandboxes) -> None:
    server = make_server()
    with mcp_client(server) as client:
        assert client.post("/seed_session", json=seed_body()).status_code == 200
        auth = file_auth(server)
        asyncio.run(server.retire_session_state("rs-1"))
        assert sandboxes.stopped == sandboxes.started and len(server.service.store) == 0
        assert client.get("/v1/files/gdrive/F_LEDGER", headers=auth).status_code == 401


def test_verify_stops_the_sandbox_and_closes_the_tools_before_grading(sandboxes) -> None:
    server = make_server()
    when_stopped = []  # (the world still there, the episode's tools closed)
    sandboxes.on_stop = lambda: when_stopped.append(
        (server.service.store.get("rs-1") is not None, server._episodes["rs-1"].grading)
    )
    with mcp_client(server) as client:
        seeded = client.post("/seed_session", json=seed_body())
        auth = file_auth(server)
        grant = seeded.json()["resources_tools"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]
        verified = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies)
        assert verified.status_code == 200 and verified.json()["reward"] == 0.0, verified.text
        assert when_stopped == [(True, True)]  # the tools closed and the sandbox stopped before grading
        assert sandboxes.stopped == sandboxes.started
        # a process left running in the sandbox reaches neither the file API nor the episode's tools any more
        assert client.get("/v1/files/gdrive/F_LEDGER", headers=auth).status_code == 401
        called = rpc(client, "tools/call", {"name": "gdrive_list_files", "arguments": {}}, grant).json()["result"]
        assert called["isError"] is True
        assert client.post("/close_session", json=close_body(), cookies=seeded.cookies).status_code == 200
        assert len(sandboxes.stopped) == 1  # close stops nothing twice


def test_while_verification_stops_the_sandbox_the_episodes_tools_refuse(sandboxes) -> None:
    server = make_server()
    with mcp_client(server) as client:
        seeded = client.post("/seed_session", json=seed_body())
        grant = seeded.json()["resources_tools"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]
        server._episodes["rs-1"].grading = True  # as /verify sets it before it awaits the sandbox's stop
        called = rpc(client, "tools/call", {"name": "gdrive_list_files", "arguments": {}}, grant).json()["result"]
        assert called["isError"] is True and "The task has ended: its tools are closed" in json.dumps(called)
        refused = client.post("/gdrive_list_files", json={}, cookies=seeded.cookies)
        assert refused.status_code == 409 and server.service.store.get("rs-1") is not None


def test_an_upload_streaming_in_when_verification_begins_writes_nothing(sandboxes, monkeypatch) -> None:
    """Verification revokes the file API token, but an upload the token admitted earlier may still be streaming in:
    once its body has arrived it is refused, so it cannot change what is graded."""
    server = make_server()
    with mcp_client(server) as client:
        seeded = client.post("/seed_session", json=seed_body())
        auth = file_auth(server)
        read_body = OfficeEnvSandboxResourcesServer._content_args

        async def body_then_verification(request):
            args = await read_body(request)
            server._episodes["rs-1"].grading = True  # as /verify sets it, while this body streamed in
            return args

        monkeypatch.setattr(OfficeEnvSandboxResourcesServer, "_content_args", staticmethod(body_then_verification))
        replaced = client.put("/v1/files/gdrive/F_LEDGER", content=b"vendor,total\n", headers=auth)
        server._episodes["rs-1"].grading = False
        created = client.post(
            "/v1/files", params={"app": "gdrive", "name": "late.csv"}, content=b"a,b\n", headers=auth
        )
        assert replaced.status_code == 409 and created.status_code == 409, (replaced.text, created.text)
        server._episodes["rs-1"].grading = False
        assert client.get("/v1/files/gdrive/F_LEDGER", headers=auth).text == LEDGER  # unchanged
        listed = client.post("/gdrive_list_files", json={}, cookies=seeded.cookies)
        assert listed.status_code == 200 and "late.csv" not in listed.text


@pytest.mark.parametrize("close_tools_on_task_end", [True, False])
def test_with_tools_closed_at_task_end_the_file_api_closes_too(sandboxes, close_tools_on_task_end: bool) -> None:
    """Once a call has ended the task, close_tools_on_task_end refuses the file API as it refuses the tools, so a
    file uploaded after complete_task cannot make the task pass. Without the setting the file API stays open."""
    server = make_server(close_tools_on_task_end=close_tools_on_task_end)
    with mcp_client(server) as client:
        seeded = client.post("/seed_session", json=seed_body())
        auth = file_auth(server)
        ended = client.post("/user_complete_task", json={"answer": "done"}, cookies=seeded.cookies)
        assert ended.status_code == 200, ended.text
        created = client.post(
            "/v1/files", params={"app": "gdrive", "name": "totals.csv"}, content=TOTALS.encode(), headers=auth
        )
        replaced = client.put("/v1/files/gdrive/F_LEDGER", content=b"vendor,total\n", headers=auth)
        downloaded = client.get("/v1/files/gdrive/F_LEDGER", headers=auth)
        answers = [created.status_code, replaced.status_code, downloaded.status_code]
        verified = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies)
    if close_tools_on_task_end:
        assert answers == [409, 409, 409] and "its files are closed" in created.text, created.text
        assert verified.json()["reward"] == 0.0, verified.text  # the late upload made nothing pass
    else:
        assert answers == [200, 200, 200], (created.text, replaced.text)
        assert verified.json()["reward"] == 1.0, verified.text


FILE_CALLS = {  # each file API call, as a sandbox makes it
    "download": lambda client, auth: client.get("/v1/files/gdrive/F_LEDGER", headers=auth),
    "create": lambda client, auth: client.post(
        "/v1/files", params={"app": "gdrive", "name": "notes.csv"}, content=b"a,b\n", headers=auth
    ),
    "replace": lambda client, auth: client.put("/v1/files/gdrive/F_LEDGER", content=b"a,b\n", headers=auth),
}


@pytest.mark.parametrize("fails", [False, True], ids=["answered", "failed"])
@pytest.mark.parametrize("verb", list(FILE_CALLS))
def test_a_file_api_call_that_ends_the_task_publishes_it(sandboxes, monkeypatch, verb: str, fails: bool) -> None:
    """A file API call runs the app's op through the same pipeline as a tool call (its injections among it), so it may
    end the task too: the task state it leaves is published once it is over, whatever it answers, as a tool call's
    is. /task_state reads the end, and with close_tools_on_task_end the next call is refused, so a file uploaded after
    it cannot make the task pass."""
    server = make_server(close_tools_on_task_end=True)
    with mcp_client(server) as client:
        seeded = client.post("/seed_session", json=seed_body())
        auth = file_auth(server)
        file_call = getattr(server.service, f"file_{verb}")

        def ending_the_task(session_id: str, *args, **kwargs):
            # as an op the call runs might: the user's task leaves pending
            server.service.call(session_id, "user_complete_task", json.dumps({"answer": "done"}))
            if fails:
                raise FileCallError(503, {"error": {"code": 503, "message": "The service is unavailable"}})
            return file_call(session_id, *args, **kwargs)

        monkeypatch.setattr(server.service, f"file_{verb}", ending_the_task)
        answered = FILE_CALLS[verb](client, auth)
        assert answered.status_code == (503 if fails else 200), answered.text
        assert client.get("/task_state", cookies=seeded.cookies).json() == {"task_ended": True}
        late = client.post(
            "/v1/files", params={"app": "gdrive", "name": "totals.csv"}, content=TOTALS.encode(), headers=auth
        )
        assert late.status_code == 409 and "its files are closed" in late.text, late.text
        verified = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies)
        assert verified.json()["reward"] == 0.0, verified.text  # the late upload made nothing pass


def test_a_failed_stop_before_grading_still_grades_and_close_stops_it(sandboxes) -> None:
    server = make_server()
    with mcp_client(server) as client:
        seeded = client.post("/seed_session", json=seed_body())
        auth = file_auth(server)
        sandboxes.fail_on = "stop"
        verified = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies)
        assert verified.status_code == 200 and verified.json()["reward"] == 0.0, verified.text
        assert sandboxes.stopped == [] and server._episodes["rs-1"].sandbox is not None
        assert client.get("/v1/files/gdrive/F_LEDGER", headers=auth).status_code == 401
        sandboxes.fail_on = None
        assert client.post("/close_session", json=close_body(), cookies=seeded.cookies).status_code == 200
        assert sandboxes.stopped == sandboxes.started and not server._episodes


def test_shutdown_stops_the_sandboxes_of_open_episodes(sandboxes) -> None:
    with mcp_client(make_server()) as client:
        assert client.post("/seed_session", json=seed_body()).status_code == 200
        assert sandboxes.stopped == []
    assert sandboxes.stopped == sandboxes.started and len(sandboxes.started) == 1


@pytest.mark.parametrize("stop", ["shutdown", "retire"])
def test_shutdown_and_retiring_wait_for_a_seed_still_starting_its_sandbox(sandboxes, monkeypatch, stop: str) -> None:
    """A seed puts its episode in before the sandbox has started. Retiring the session waits for the seed's lock and
    stops that sandbox. Shutdown waits for the start, which then lends nothing (503) and stops its sandbox, since
    shutdown would not find it lent. A seed that comes after shutdown starts none."""
    server = make_server()
    start = FakeSandbox.start

    async def scenario() -> None:
        starting, go = asyncio.Event(), asyncio.Event()

        async def slow_start(sandbox: FakeSandbox, spec) -> None:
            starting.set()
            await go.wait()
            await start(sandbox, spec)

        monkeypatch.setattr(FakeSandbox, "start", slow_start)
        seed = asyncio.create_task(server._seed_sandboxed_episode(SimpleNamespace(session={}), seed_request()))
        await starting.wait()
        stopping = asyncio.create_task(
            server.shutdown() if stop == "shutdown" else server.retire_session_state("rs-1")
        )
        await asyncio.sleep(0)
        go.set()
        seeded, _ = await asyncio.gather(seed, stopping, return_exceptions=True)
        if stop == "shutdown":
            assert isinstance(seeded, HTTPException) and seeded.status_code == 503, seeded
            with pytest.raises(HTTPException) as refused:
                await server._seed_sandboxed_episode(SimpleNamespace(session={}), seed_request(session_id="rs-2"))
            assert refused.value.status_code == 503
        else:
            assert isinstance(seeded, ResourcesSeedSessionResponse), seeded

    asyncio.run(scenario())
    assert len(sandboxes.started) == 1 and sandboxes.stopped == sandboxes.started


def test_shutdown_stops_the_sandbox_a_seed_failing_meanwhile_could_not_stop(sandboxes, monkeypatch, caplog) -> None:
    """A seed still starting its sandbox when shutdown begins fails, and so does its own stop of that sandbox:
    shutdown, which waited for the start, stops it on a second try and leaves nothing for a reaper."""
    server = make_server()
    start, stop = FakeSandbox.start, FakeSandbox.stop
    refused: list[FakeSandbox] = []

    async def stop_refusing_once(sandbox: FakeSandbox) -> None:
        if not refused:
            refused.append(sandbox)
            raise RuntimeError("the provider could not stop the sandbox")
        await stop(sandbox)

    async def scenario() -> None:
        starting, go = asyncio.Event(), asyncio.Event()

        async def slow_failing_start(sandbox: FakeSandbox, spec) -> None:
            starting.set()
            await go.wait()
            await start(sandbox, spec)
            raise RuntimeError("the provider lost the sandbox")

        monkeypatch.setattr(FakeSandbox, "start", slow_failing_start)
        monkeypatch.setattr(FakeSandbox, "stop", stop_refusing_once)
        seed = asyncio.create_task(server._seed_sandboxed_episode(SimpleNamespace(session={}), seed_request()))
        await starting.wait()
        stopping = asyncio.create_task(server.shutdown())
        await asyncio.sleep(0)  # shutdown has begun, and waits for the seed's lock
        go.set()
        failed, stopped = await asyncio.gather(seed, stopping, return_exceptions=True)
        assert isinstance(failed, RuntimeError) and stopped is None, (failed, stopped)

    with caplog.at_level("ERROR"):
        asyncio.run(scenario())
    assert refused == sandboxes.started and sandboxes.stopped == sandboxes.started  # the second stop was shutdown's
    assert "a reaper must stop them" not in caplog.text


def test_shutdown_stops_every_lent_sandbox_at_once_within_its_grace(sandboxes, monkeypatch) -> None:
    """A job's end gives the servers server_shutdown_timeout_seconds before they are killed; stopped one after another,
    a collection's worth of sandboxes (64 in a training smoke) would outlast it and stay on the cell until their TTL."""
    monkeypatch.setattr(sandbox_app, "get_global_config_dict", lambda: GRACE_OF_3_S)  # a stop budget of 1 s
    server = make_server()
    stop = FakeSandbox.stop

    async def slow_stop(sandbox: FakeSandbox) -> None:
        await asyncio.sleep(0.25)
        await stop(sandbox)

    async def scenario() -> float:
        for n in range(20):
            await server._seed_sandboxed_episode(
                SimpleNamespace(session={}), seed_request(session_id=f"rs-{n}", rollout_id=f"r{n}")
            )
        monkeypatch.setattr(FakeSandbox, "stop", slow_stop)
        started = time.monotonic()
        await server.shutdown()
        return time.monotonic() - started

    elapsed = asyncio.run(scenario())
    assert len(sandboxes.started) == 20 and set(sandboxes.stopped) == set(sandboxes.started)
    assert elapsed < 1.0  # one after another: 5 s


def test_shutdown_ends_at_its_deadline_and_logs_the_sandboxes_left(sandboxes, monkeypatch, caplog) -> None:
    monkeypatch.setattr(sandbox_app, "get_global_config_dict", lambda: GRACE_OF_3_S)  # a stop budget of 1 s
    server = make_server()
    stop = FakeSandbox.stop

    async def stop_or_hang(sandbox: FakeSandbox) -> None:
        if sandbox is FakeSandbox.started[0]:
            await asyncio.Event().wait()  # the provider never answers
        await stop(sandbox)

    async def scenario() -> None:
        for n in range(3):
            await server._seed_sandboxed_episode(
                SimpleNamespace(session={}), seed_request(session_id=f"rs-{n}", rollout_id=f"r{n}")
            )
        monkeypatch.setattr(FakeSandbox, "stop", stop_or_hang)
        await asyncio.wait_for(server.shutdown(), timeout=3.0)  # unbounded, it would wait for the hung stop

    with caplog.at_level("ERROR"):
        asyncio.run(scenario())
    assert set(sandboxes.stopped) == set(sandboxes.started[1:])
    assert "1 lent sandboxes may still run" in caplog.text and "a lent sandbox: ['rs-0']" in caplog.text


def test_shutdown_names_the_creates_still_in_flight_at_its_deadline_apart(sandboxes, monkeypatch, caplog) -> None:
    """A sandbox whose create is still in flight at the deadline is counted and named apart from the lent ones, so a
    reaper's count can be checked against the line."""
    monkeypatch.setattr(sandbox_app, "get_global_config_dict", lambda: GRACE_OF_3_S)  # a stop budget of 1 s
    server = make_server()
    start, stop = FakeSandbox.start, FakeSandbox.stop

    async def scenario() -> None:
        in_flight = asyncio.Event()

        async def start_or_hang(sandbox: FakeSandbox, spec) -> None:
            if FakeSandbox.started:  # the second create never answers
                in_flight.set()
                await asyncio.Event().wait()
            await start(sandbox, spec)

        async def stop_or_hang(sandbox: FakeSandbox) -> None:
            if FakeSandbox.started and sandbox is FakeSandbox.started[0]:
                await asyncio.Event().wait()  # the lent sandbox's stop never answers either
            await stop(sandbox)

        monkeypatch.setattr(FakeSandbox, "start", start_or_hang)
        monkeypatch.setattr(FakeSandbox, "stop", stop_or_hang)
        await server._seed_sandboxed_episode(
            SimpleNamespace(session={}), seed_request(session_id="rs-0", rollout_id="r0")
        )
        starting = asyncio.create_task(
            server._seed_sandboxed_episode(
                SimpleNamespace(session={}), seed_request(session_id="rs-1", rollout_id="r1")
            )
        )
        await in_flight.wait()
        await asyncio.wait_for(server.shutdown(), timeout=3.0)
        starting.cancel()
        await asyncio.gather(starting, return_exceptions=True)

    with caplog.at_level("ERROR"):
        asyncio.run(scenario())
    assert "2 lent sandboxes may still run" in caplog.text
    assert "a lent sandbox: ['rs-0']. Failed seeds: []. Starting: ['rs-1']" in caplog.text


def as_a_request(handler: Callable[[], Awaitable[object]]) -> asyncio.Task:
    """``handler`` run as Gym runs a route, in an anyio task group (ClientDisconnectCancellationMiddleware), in a task
    of its own, as uvicorn runs a request. Cancelling the task, as uvicorn cancels the requests in flight 0.5 s into a
    shutdown and a client's disconnect cancels its own, cancels every later await in the handler that is not
    shielded."""

    async def request() -> None:
        async with anyio.create_task_group() as group:
            group.start_soon(handler)

    return asyncio.ensure_future(request())


@pytest.mark.parametrize("shutting_down", [False, True], ids=["client_gave_up", "shutdown"])
def test_a_seed_cancelled_as_its_sandbox_starts_stops_that_sandbox(
    sandboxes, monkeypatch, caplog, shutting_down: bool
) -> None:
    """A seed whose request is cancelled while its sandbox starts still finishes the start, then stops the sandbox by
    its id, logs it and lends nothing; a shutdown that began meanwhile waits for it. Before, the cancellation reached
    the provider's create and the cleanup's unshielded stop alike, and neither the seed nor shutdown knew the sandbox
    any more: a two-step training smoke ended with 8 sandboxes still running, as many as the requests uvicorn
    cancelled, and no line naming them."""
    monkeypatch.setattr(sandbox_app, "get_global_config_dict", lambda: GRACE_OF_3_S)  # a shutdown budget of 1 s
    server = make_server()
    start, stop = FakeSandbox.start, FakeSandbox.stop

    async def scenario() -> None:
        creating, created = asyncio.Event(), asyncio.Event()

        async def slow_start(sandbox: FakeSandbox, spec) -> None:
            creating.set()
            await created.wait()  # the provider's create, in flight
            await start(sandbox, spec)

        async def provider_stop(sandbox: FakeSandbox) -> None:
            await asyncio.sleep(0)  # a request to the provider, which a cancelled scope cuts
            await stop(sandbox)

        monkeypatch.setattr(FakeSandbox, "start", slow_start)
        monkeypatch.setattr(FakeSandbox, "stop", provider_stop)
        seed = as_a_request(lambda: server._seed_sandboxed_episode(SimpleNamespace(session={}), seed_request()))
        await creating.wait()
        seed.cancel()
        await asyncio.sleep(0.05)  # the cancellation has reached the seed, whose start goes on
        assert not seed.done() and not server._episodes["rs-1"].access  # it lent nothing yet
        shutdown = asyncio.ensure_future(server.shutdown()) if shutting_down else None
        await asyncio.sleep(0.05)
        assert shutdown is None or not shutdown.done()  # it waits for the seed still starting
        created.set()
        with pytest.raises(asyncio.CancelledError):
            await seed
        if shutdown is not None:
            await shutdown

    with caplog.at_level(logging.INFO):
        asyncio.run(scenario())
    (sandbox,) = sandboxes.started
    assert sandboxes.stopped == [sandbox]
    assert not server._episodes and not server._tokens
    stopped = (
        f"Stopped sandbox {sandbox.sandbox_id} of resources session rs-1: its seed was cancelled while it started"
    )
    assert stopped in caplog.text and "a reaper must stop them" not in caplog.text, caplog.text


def test_a_close_cancelled_as_it_stops_the_sandbox_still_stops_it(sandboxes, monkeypatch) -> None:
    """A close whose request is cancelled while it stops the episode's sandbox (a verification's stop, alike) still
    stops it. Before, the stop was cancelled with the request, and the sandbox ran on until shutdown."""
    server = make_server()

    async def scenario() -> None:
        await server._seed_sandboxed_episode(SimpleNamespace(session={}), seed_request())
        stopping, stopped = asyncio.Event(), asyncio.Event()
        stop = FakeSandbox.stop

        async def slow_stop(sandbox: FakeSandbox) -> None:
            stopping.set()
            await stopped.wait()  # the provider's stop, in flight
            await stop(sandbox)

        monkeypatch.setattr(FakeSandbox, "stop", slow_stop)
        caller = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
        body = ResourcesCloseSessionRequest(**close_body())
        close = as_a_request(lambda: server.close_resources_session(caller, body))
        await stopping.wait()
        close.cancel()
        await asyncio.sleep(0.05)
        assert not close.done()  # the stop goes on
        stopped.set()
        await asyncio.gather(close, return_exceptions=True)

    asyncio.run(scenario())
    (sandbox,) = sandboxes.started
    assert sandboxes.stopped == [sandbox]
    assert all(episode.sandbox is None for episode in server._episodes.values())


# Seeds take turns.

QUEUED_SEEDS = 20
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
}


def slow_seeds(server: OfficeEnvResourcesServer, monkeypatch, built: list[str]) -> None:
    """Each world build holds the event loop for a moment, as a large world's does, and is recorded."""
    seed = server.service.seed

    def slow_seed(session_id: str, row: dict) -> object:
        time.sleep(0.01)
        built.append(session_id)
        return seed(session_id, row)

    monkeypatch.setattr(server.service, "seed", slow_seed)


def seed_concurrently(server: OfficeEnvResourcesServer, count: int) -> list[asyncio.Task]:
    return [
        asyncio.create_task(
            server._seed_sandboxed_episode(
                SimpleNamespace(session={}), seed_request(session_id=f"rs-{n}", rollout_id=f"r{n}")
            )
        )
        for n in range(count)
    ]


def test_a_harness_is_answered_while_seeds_queue(sandboxes, monkeypatch) -> None:
    """A seed builds its world synchronously, and one iteration of the event loop runs every task that is ready: a
    wave of seeds ran back to back while a harness's MCP initialize waited behind all of them (minutes, at a training
    cap; Pi gives up after 30 s). Seeds take turns, and the loop runs what is ready before each, so the initialize is
    answered before more than a seed or two of 20 queued ones are built. With only a bare yield before each seed it
    advanced a step per seed: of 40 queued seeds, it was answered after 13."""
    server = make_server()
    built: list[str] = []
    slow_seeds(server, monkeypatch, built)
    app = server.setup_webserver()
    maybe_auto_expose(server, app)

    async def scenario() -> tuple[int, int]:
        transport = httpx.ASGITransport(app=app, client=SANDBOX)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as harness:
            seeds = seed_concurrently(server, QUEUED_SEEDS)
            answer = await asyncio.create_task(harness.post("/mcp", headers=RPC_HEADERS, json=INITIALIZE))
            answered_after = len(built)
            await asyncio.gather(*seeds)
        return answer.status_code, answered_after

    with MCPClient(app, base_url="http://127.0.0.1:8000", client=LOCAL) as client:  # its lifespan runs MCP
        status, answered_after = client.portal.call(scenario)
    assert status == 200
    assert answered_after <= 2, f"answered after {answered_after} of {QUEUED_SEEDS} seeds"
    assert len(sandboxes.started) == QUEUED_SEEDS  # and each seed still had its turn


def test_a_burst_of_harness_setups_is_answered_between_world_builds(sandboxes, monkeypatch) -> None:
    """At a training cap a wave of seeds and a wave of harness setups arrive together: a cap-256 smoke queued up to 166
    seeds while its harnesses set up, and 3.5% of Pi's setups outwaited its 30 s. A setup is a few MCP requests of a few
    dozen steps of the loop each, so the gap between two builds must hold a burst of them: with a 10 ms gap the median
    request waited a build out. With SEED_TURN_YIELD_S the median request is answered in the gap it arrives in, and none
    waits more than a few builds out, however long the queue (back to back, it waited for every seed ahead of it)."""
    server = make_server()
    built: list[str] = []
    slow_seeds(server, monkeypatch, built)
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    seeds_queued, setups = 30, 20

    async def scenario(token: str) -> list[int]:
        headers = {**RPC_HEADERS, NEMO_GYM_MCP_SESSION_TOKEN_HEADER: token}
        transport = httpx.ASGITransport(app=app, client=SANDBOX)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as harness:
            seeds = seed_concurrently(server, seeds_queued)
            waited: list[int] = []  # the builds each request waited out

            async def request(body: dict) -> None:
                before = len(built)
                answer = await harness.post("/mcp", headers=headers, json=body)
                assert answer.status_code < 300, answer.text
                waited.append(len(built) - before)

            async def setup() -> None:  # Pi's tool discovery
                await request(INITIALIZE)
                await request({"jsonrpc": "2.0", "method": "notifications/initialized"})
                await request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})

            await asyncio.gather(*(setup() for _ in range(setups)))
            await asyncio.gather(*seeds)
        return waited

    with MCPClient(app, base_url="http://127.0.0.1:8000", client=LOCAL) as client:
        seeded = client.post("/seed_session", json=seed_body(session_id="rs-harness", rollout_id="r-harness"))
        token = seeded.json()["resources_tools"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]
        waited = client.portal.call(scenario, token)
    assert len(waited) == 3 * setups and len(sandboxes.started) == seeds_queued + 1
    assert statistics.median(waited) == 0, sorted(waited)
    assert max(waited) <= 3, sorted(waited)


def test_a_seed_is_admitted_again_in_its_turn(sandboxes, monkeypatch) -> None:
    """While a seed waits for its turn, shutdown may begin: in its turn it is admitted again, and refused (503) before
    its world is built. A seed whose world was being built as shutdown began is refused before it lends a sandbox,
    which shutdown would not stop, and its world goes."""
    server = make_server()
    built: list[str] = []
    slow_seeds(server, monkeypatch, built)
    seed = server.service.seed

    def seed_as_shutdown_begins(session_id: str, row: dict) -> object:
        server._shutting_down = True
        return seed(session_id, row)

    monkeypatch.setattr(server.service, "seed", seed_as_shutdown_begins)

    async def scenario() -> list:
        return await asyncio.gather(*seed_concurrently(server, 2), return_exceptions=True)

    first, second = asyncio.run(scenario())
    for refused in (first, second):
        assert isinstance(refused, HTTPException) and refused.status_code == 503, refused
    assert built == ["rs-0"] and not server._episodes and len(server.service.store) == 0
    assert sandboxes.started == []


def test_each_seed_logs_its_place_its_wait_its_build_time_and_its_sandbox(sandboxes, monkeypatch, caplog) -> None:
    """One line per seed, for a CPU sample of the server to be read against: the seeds ahead of it, its wait for its
    turn and its world's build. A second names the sandbox lent to the session, and its start time: the harness's and
    the provider's logs name the sandbox, this server's the session. Gym's servers configure no logging, so the lines
    have a handler of their own."""
    server = make_server()
    slow_seeds(server, monkeypatch, [])

    async def scenario() -> None:
        await asyncio.gather(*seed_concurrently(server, 3))

    with caplog.at_level(logging.INFO, logger=sandbox_app.SEED_LOGGER.name):
        asyncio.run(scenario())
    seeds = re.compile(r"Seed of OfficeEnv session (rs-\d): (\d+) ahead, waited \d+\.\d\d s, built in \d+\.\d\d s")
    lent = re.compile(r"Sandbox (sbx-\d) of OfficeEnv session (rs-\d) \(task sandbox-files\) started in \d+\.\d\d s")
    lines = [record.getMessage() for record in caplog.records if record.name == sandbox_app.SEED_LOGGER.name]
    seed_lines = [line for line in lines if line.startswith("Seed ")]
    assert [seeds.fullmatch(line).groups() for line in seed_lines] == [("rs-0", "0"), ("rs-1", "1"), ("rs-2", "2")]
    lent_lines = [lent.fullmatch(line) for line in lines if line not in seed_lines]
    sandbox_ids = {match.group(2): match.group(1) for match in lent_lines}
    assert sandbox_ids == {session: episode.sandbox.sandbox_id for session, episode in server._episodes.items()}
    assert len(lines) == 6 and set(sandbox_ids.values()) == {"sbx-1", "sbx-2", "sbx-3"}
    assert sandbox_app.SEED_LOGGER.handlers and sandbox_app.SEED_LOGGER.level == logging.INFO


def test_an_episodes_lock_is_kept_only_while_a_request_holds_it(sandboxes) -> None:
    server = make_server()
    with mcp_client(server) as client:
        assert client.post("/seed_session", json=seed_body()).status_code == 200
        assert client.post("/close_session", json=close_body()).status_code == 200
        assert client.post("/close_session", json=close_body(session_id="rs-2")).status_code == 200  # never seeded
        assert len(server._locks) == 0


def test_relay_ports_are_this_servers_and_the_model_servers() -> None:
    named = {
        "policy_model": {"responses_api_models": {"inference_provider": {"port": 18173}}},
        "judge": {"responses_api_models": {"vllm_model": {"port": 18200}}},
        "pi_agent": {"responses_api_agents": {"pi_agent": {"port": 18300}}},
        "sandbox": {"docker": {}},
    }
    assert relay_ports(named, own_port=15824) == [15824, 18173, 18200]


def test_a_host_relay_starts_in_the_sandbox_before_it_is_lent(sandboxes, monkeypatch) -> None:
    named = {
        "sandbox": {"docker": {}},
        "policy_model": {"responses_api_models": {"inference_provider": {"port": 18173}}},
    }
    monkeypatch.setattr(sandbox_app, "get_global_config_dict", lambda: named)
    server = make_server(host="127.0.0.1", sandbox_config={**IMAGE, "host_relay": "host.lima.internal"})
    with mcp_client(server) as client:
        assert client.post("/seed_session", json=seed_body()).status_code == 200
        (sandbox,) = sandboxes.started
        assert sandbox.spec.files[RELAY_PATH] == RELAY_SOURCE
        assert sandbox.spec.env["COMPANY_API_URL"] == "http://127.0.0.1:8080"
        assert "relay.py host.lima.internal 8080 18173" in sandbox.commands[0]
        assert "relay ready" in sandbox.commands[1]


# Sandboxes without the internet.

OPENSANDBOX_NAMED = {
    "sandbox": {"opensandbox": {}},
    "policy_model": {"responses_api_models": {"vllm_model": {"host": "10.0.0.2", "port": 18173}}},
    "judge_model": {"responses_api_models": {"vllm_model": {"host": "127.0.0.1", "port": 18200}}},
    "pi_agent": {"responses_api_agents": {"pi_agent": {"host": "10.0.0.9", "port": 18300}}},
}


def test_a_sandbox_without_the_internet_reaches_only_this_server_and_the_model_servers(sandboxes, monkeypatch) -> None:
    monkeypatch.setattr(sandbox_app, "get_global_config_dict", lambda: OPENSANDBOX_NAMED)
    with mcp_client(make_server(sandbox_config={**IMAGE, "block_internet": True})) as client:
        assert client.post("/seed_session", json=seed_body()).status_code == 200
    guide = sandboxes.started[0].spec.files["/workspace/README.md"]
    assert guide.endswith("The sandbox has no internet access, so nothing else can be installed.\n")
    policy = sandboxes.started[0].spec.provider_options["network_policy"]
    # this server (MCP and the file API) and the policy model; the judge on loopback is no sandbox's to reach
    assert policy == {
        "defaultAction": "deny",
        "egress": [{"action": "allow", "target": "10.0.0.1"}, {"action": "allow", "target": "10.0.0.2"}],
    }
    assert egress_urls(["http://10.0.0.1:8080"], OPENSANDBOX_NAMED) == [
        "http://10.0.0.1:8080",
        "http://10.0.0.2:18173",
    ]
    # Gym's OpenSandbox provider takes the policy as it is.
    pytest.importorskip("opensandbox")
    from opensandbox.models.sandboxes import NetworkPolicy

    from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProviderOptions

    assert OpenSandboxProviderOptions.from_mapping({"network_policy": policy}).network_policy == policy
    assert NetworkPolicy.model_validate(policy).default_action == "deny"


def test_blocking_the_internet_needs_opensandbox_a_reachable_server_and_one_policy(sandboxes, monkeypatch) -> None:
    blocked = {**IMAGE, "block_internet": True}
    with mcp_client(make_server(sandbox_config=blocked)) as client:  # the fixture's provider is Docker
        with pytest.raises(ValueError, match="OpenSandbox"):
            client.post("/seed_session", json=seed_body())
    monkeypatch.setattr(sandbox_app, "get_global_config_dict", lambda: OPENSANDBOX_NAMED)
    with mcp_client(make_server(host="127.0.0.1", sandbox_config=blocked)) as client:
        with pytest.raises(ValueError, match="sandbox-reachable"):
            client.post("/seed_session", json=seed_body())
    assert not sandboxes.started
    with pytest.raises(ValueError, match="one or the other"):
        OfficeEnvSandboxConfig(block_internet=True, provider_options={"network_policy": {"defaultAction": "deny"}})
    with pytest.raises(ValueError, match="host_relay"):
        OfficeEnvSandboxConfig(block_internet=True, host_relay="host.lima.internal")


# Images and what a row needs.

DOCUMENTS_IMAGE = {"image": "officeenv-sandbox:documents", "provides": ["documents", "office"]}


def needing(needs: object, *, session_id: str = "rs-1", rollout_id: str = "r1") -> dict:
    body = seed_body(session_id=session_id, rollout_id=rollout_id)
    body["task_data"]["sandbox_needs"] = needs
    return body


def test_a_row_gets_the_first_image_with_what_it_needs_and_the_guide_names_its_tools(sandboxes) -> None:
    server = make_server(sandbox_config={**IMAGE, "images": [DOCUMENTS_IMAGE]})
    with mcp_client(server) as client:
        for body in (
            seed_body(),
            needing(["documents"], session_id="rs-2", rollout_id="r2"),
            needing([], session_id="rs-3", rollout_id="r3"),
        ):
            assert client.post("/seed_session", json=body).status_code == 200
    plain, documents, nothing_more = sandboxes.started
    assert [sandbox.spec.image for sandbox in sandboxes.started] == [
        "officeenv-sandbox:pilot",
        "officeenv-sandbox:documents",
        "officeenv-sandbox:pilot",
    ]
    assert "Installed here" not in plain.spec.files["/workspace/README.md"]
    guide = documents.spec.files["/workspace/README.md"]
    assert (
        "\nInstalled here:\n\n- bash, curl and Python 3.12 with pandas and openpyxl\n- python-docx, python-pptx, pypdf, "
        "pdfplumber and reportlab for Python, and Poppler's pdftotext, pdfinfo and pdftoppm\n- LibreOffice without a "
        "display, which converts documents (soffice --headless --convert-to pdf <file>, or docx, xlsx, pptx)"
    ) in guide
    assert guide.endswith("formulas as it converts it\n") and "internet" not in guide
    assert not ENVIRONMENT_NAMES.search(guide)
    assert set(typing.get_args(ToolSet)) == set(TOOL_SETS)


def test_the_configured_image_serves_the_rows_that_need_what_it_provides(sandboxes) -> None:
    with mcp_client(make_server(sandbox_config=DOCUMENTS_IMAGE)) as client:
        assert client.post("/seed_session", json=needing(["office", "documents"])).status_code == 200
    assert sandboxes.started[0].spec.image == "officeenv-sandbox:documents"


def test_a_row_no_image_serves_is_refused_before_its_world_is_built(sandboxes) -> None:
    server = make_server()
    with mcp_client(server) as client:
        for needs, message in [
            (["office"], "No configured sandbox image provides office (sandbox_config.provides and"),
            (["documents", "scanner"], "sandbox_needs names scanner; a sandbox can provide documents, office"),
            ("documents", "sandbox_needs lists what the task needs in its sandbox: some of documents, office"),
        ]:
            refused = client.post("/seed_session", json=needing(needs))
            assert refused.status_code == 400, refused.text
            error = refused.json()["detail"]["error"]
            assert error["code"] == "unsupported_needs" and error["message"].startswith(message), error
    assert sandboxes.started == [] and not server._episodes and len(server.service.store) == 0
    with mcp_client(make_server(sandbox_config={})) as client:
        with pytest.raises(ValueError, match="Set sandbox_config.image"):
            client.post("/seed_session", json=seed_body())


def test_the_guide_says_a_sandbox_without_the_internet_installs_nothing_else() -> None:
    guide = workspace_guide(8192, ("gdrive",), offline=True)
    assert guide.endswith(
        "\nInstalled here:\n\n- bash, curl and Python 3.12 with pandas and openpyxl\n\n"
        "The sandbox has no internet access, so nothing else can be installed.\n"
    )
    with_tools = workspace_guide(8192, ("gdrive",), tools=("office",), offline=True)
    assert "- LibreOffice without a display" in with_tools and with_tools.endswith("nothing else can be installed.\n")


# The file API.


def test_the_file_api_moves_text_files_with_the_episode_token(sandboxes) -> None:
    server = make_server()
    with mcp_client(server) as client:
        seeded = client.post("/seed_session", json=seed_body())
        auth = file_auth(server)

        downloaded = client.get("/v1/files/gdrive/F_LEDGER", headers=auth)
        assert downloaded.status_code == 200 and downloaded.text == LEDGER

        uploaded = client.post(
            "/v1/files",
            params={"app": "gdrive", "name": "totals.csv", "parent": "F_FIN", "mime_type": "text/csv"},
            content=TOTALS.encode(),
            headers=auth,
        )
        assert uploaded.status_code == 200, uploaded.text
        file_id = uploaded.json()["id"]
        assert client.get(f"/v1/files/gdrive/{file_id}", headers=auth).text == TOTALS

        replaced = client.put(f"/v1/files/gdrive/{file_id}", content=b"vendor,total\n", headers=auth)
        assert replaced.status_code == 200
        assert client.get(f"/v1/files/gdrive/{file_id}", headers=auth).text == "vendor,total\n"
        client.put(f"/v1/files/gdrive/{file_id}", content=TOTALS.encode(), headers=auth)

        verified = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies)
        assert verified.json()["reward"] == 1.0, verified.text


def test_the_file_api_moves_binary_files_byte_for_byte(sandboxes) -> None:
    server = make_server()
    with mcp_client(server) as client:
        seeded = client.post("/seed_session", json=seed_body({**SPEC, "checks": [XLSX_CHECK]}))
        auth = file_auth(server)

        uploaded = client.post(
            "/v1/files",
            params={"app": "gdrive", "name": "totals.xlsx", "parent": "F_FIN"},
            content=BOOK,
            headers=auth,
        )
        assert uploaded.status_code == 200, uploaded.text
        created = uploaded.json()
        assert created["mimeType"] == XLSX and created["size"] == str(len(BOOK))
        downloaded = client.get(f"/v1/files/gdrive/{created['id']}", headers=auth)
        assert downloaded.status_code == 200 and downloaded.content == BOOK
        assert downloaded.headers["content-type"] == XLSX

        noise = bytes(range(256)) * 4  # every byte value, no UTF-8 text
        replaced = client.put(f"/v1/files/gdrive/{created['id']}", content=noise, headers=auth)
        assert replaced.status_code == 200 and replaced.json()["size"] == str(len(noise))
        assert client.get(f"/v1/files/gdrive/{created['id']}", headers=auth).content == noise
        blob = client.post("/v1/files", params={"app": "gdrive", "name": "blob"}, content=noise, headers=auth)
        got = client.get(f"/v1/files/gdrive/{blob.json()['id']}", headers=auth)
        assert got.content == noise and got.headers["content-type"] == "application/octet-stream"

        client.put(f"/v1/files/gdrive/{created['id']}", content=BOOK, headers=auth)
        verified = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies)
        assert verified.json()["reward"] == 1.0, verified.text


def test_the_file_api_refuses_other_tokens_apps_and_oversized_uploads(sandboxes, monkeypatch) -> None:
    server = make_server()
    with mcp_client(server) as client:
        seeded = client.post("/seed_session", json=seed_body())
        auth = file_auth(server)
        assert client.get("/v1/files/gdrive/F_LEDGER").status_code == 401
        assert client.get("/v1/files/gdrive/F_LEDGER", headers={"Authorization": "Bearer x"}).status_code == 401
        assert client.get("/v1/files/box/F_LEDGER", headers=auth).status_code == 404
        missing = client.get("/v1/files/gdrive/F_MISSING", headers=auth)
        assert missing.status_code == 404 and "error" in missing.text
        monkeypatch.setattr(sandbox_app, "MAX_UPLOAD_BYTES", 8)
        too_big = client.post(
            "/v1/files", params={"app": "gdrive", "name": "x.bin"}, content=b"\xff" * 9, headers=auth
        )
        assert too_big.status_code == 413

        assert client.post("/close_session", json=close_body(), cookies=seeded.cookies).status_code == 200
        assert client.get("/v1/files/gdrive/F_LEDGER", headers=auth).status_code == 401  # revoked at close


def test_a_download_without_a_media_type_is_text_or_bytes_by_its_content(sandboxes, monkeypatch) -> None:
    """A file its app gives no media type (Box keeps none) comes as text/plain when its bytes are UTF-8 text, and as
    application/octet-stream otherwise."""
    server = make_server()
    with mcp_client(server) as client:
        client.post("/seed_session", json=seed_body())
        auth = file_auth(server)
        for data, media_type in ((LEDGER.encode(), "text/plain; charset=utf-8"), (BOOK, "application/octet-stream")):
            monkeypatch.setattr(server.service, "file_download", lambda *_, data=data: FileContent(data, "f", None))
            got = client.get("/v1/files/box/B_FILE", headers=auth)
            assert got.status_code == 200 and got.content == data and got.headers["content-type"] == media_type


# Drive file content stays out of a sandboxed episode's conversation (OfficeEnv's workspace mode).

LONG = "invoice_id,vendor,amount\n" + "".join(f"INV-{n},Vendor {n % 7},{n}.25\n" for n in range(600))
SHORT = "vendor,total\nAcme,10.50\n"
BOOK_B64 = base64.b64encode(BOOK).decode()
POLICY = "Book economy.\n"


def content_spec() -> dict:
    """SPEC with a workbook, a long CSV (its name holding a '/') and a Google Doc beside the short ledger."""
    spec = json.loads(json.dumps(SPEC))
    spec["apps"]["gdrive"]["data"]["files"] += [
        {"id": "F_BOOK", "name": "ledger.xlsx", "parents": ["F_FIN"], "owner": ME, "contentBase64": BOOK_B64},
        {"id": "F_LONG", "name": "Q3/Q4 ledger.csv", "parents": ["F_FIN"], "owner": ME, "content": LONG},
        {
            "id": "F_DOC",
            "name": "Travel policy",
            "mimeType": "application/vnd.google-apps.document",
            "parents": ["F_FIN"],
            "owner": ME,
            "content": POLICY,
        },
    ]
    return spec


def seed_with_content(client: TestClient) -> tuple:
    """Seed content_spec() and return the seed's answer, a function that calls a tool over MCP and reads its JSON
    output, and the tools' descriptions as the episode lists them."""
    seeded = client.post("/seed_session", json=seed_body(content_spec()))
    assert seeded.status_code == 200, seeded.text
    token = seeded.json()["resources_tools"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]

    def call(tool: str, /, **arguments: object) -> dict:
        result = rpc(client, "tools/call", {"name": tool, "arguments": arguments}, token).json()["result"]
        return result.get("structuredContent") or json.loads(result["content"][0]["text"])

    listed = rpc(client, "tools/list", {}, token).json()["result"]["tools"]
    return seeded, call, {tool["name"]: tool["description"] for tool in listed}


def test_a_download_saves_the_file_in_the_workspace_and_answers_with_its_path(sandboxes) -> None:
    """The server writes each file a call saved (its reply's workspace_writes) into the sandbox before it answers:
    byte for byte, its folder made first, mode 644. /verify counts the writes that landed, and passes on the episode's
    workspace trace, which holds the mode's options and every call."""
    with mcp_client(make_server()) as client:
        seeded, call, _ = seed_with_content(client)
        (sandbox,) = sandboxes.started

        book = call("gdrive_download_file", file_id="F_BOOK")
        assert "contentBase64" not in book and "content" not in book, book
        assert book["savedTo"] == "/workspace/drive/F_BOOK/ledger.xlsx" and book["savedBytes"] == len(BOOK)
        assert sandbox.files["/workspace/drive/F_BOOK/ledger.xlsx"] == BOOK  # byte for byte
        assert "mkdir -p /workspace/drive/F_BOOK" in sandbox.commands
        # readable whoever the harness runs as, executable by no one, whatever mode the provider writes
        assert sandbox.modes[book["savedTo"]] == 0o644
        assert "chmod 644 /workspace/drive/F_BOOK/ledger.xlsx" in sandbox.commands

        short = call("gdrive_download_file", file_id="F_LEDGER")  # short text: inline as well
        assert short["content"] == LEDGER and short["savedTo"] == "/workspace/drive/F_LEDGER/ledger.csv"
        assert sandbox.files[short["savedTo"]] == LEDGER.encode()
        long = call("gdrive_download_file", file_id="F_LONG")
        assert "content" not in long and sandbox.files[long["savedTo"]] == LONG.encode()

        result = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert result["sandbox_files_saved"] == 3.0 and result["sandbox_uploads_refused"] == 0.0
        assert result["sandbox_bytes_saved"] == float(len(BOOK) + len(LEDGER) + len(LONG))
        trace = result["officeenv_workspace"]
        assert trace["workspace"] == {
            "folder": "/workspace",
            "inline_bytes": 8192,
            "upload_hint": {"replace": UPLOAD_HINT.replace, "create": UPLOAD_HINT.create},
            # this server's caps on the files one call reads
            "max_files": 20,
            "max_total_bytes": 25_000_000,
            "max_file_bytes": 20_000_000,
        }
        assert [entry["tool"] for entry in trace["calls"]] == ["gdrive_download_file"] * 3


def test_an_upload_that_pastes_its_content_is_refused_with_the_file_api_command(sandboxes) -> None:
    server = make_server()
    with mcp_client(server) as client:
        seeded, call, _ = seed_with_content(client)

        pasted = call("gdrive_create_file", name="Q3 totals.xlsx", parents=["F_FIN"], content_base64=BOOK_B64)
        # Drive's own error envelope, OfficeEnv's words: the path form first, then the file API's command
        assert "File content is not pasted into this call (binary content)" in pasted["error"]["message"], pasted
        assert '{"content_path": "/workspace/Q3 totals.xlsx"}' in pasted["error"]["message"]
        assert (
            'curl -sf -X POST -H "Authorization: Bearer $COMPANY_API_TOKEN" --data-binary @<local path> '
            '"$COMPANY_API_URL/v1/files?app=gdrive&name=Q3%20totals.xlsx&parent=F_FIN"'
        ) in pasted["error"]["message"]
        too_long = call("gdrive_create_file", name="big.csv", content=LONG)
        assert "text over 8192 bytes" in too_long["error"]["message"]
        replaced = call("gdrive_update_file", file_id="F_LEDGER", content_base64=BOOK_B64)
        assert "-X PUT" in replaced["error"]["message"]
        assert '"$COMPANY_API_URL/v1/files/gdrive/F_LEDGER"' in replaced["error"]["message"]
        # short text still goes in the call, and the file API takes what the tools refused
        assert call("gdrive_create_file", name="totals.csv", parents=["F_FIN"], content=TOTALS)["name"] == "totals.csv"
        auth = file_auth(server)
        sent = client.post("/v1/files", params={"app": "gdrive", "name": "big.xlsx"}, content=BOOK, headers=auth)
        assert sent.status_code == 200, sent.text
        assert client.get(f"/v1/files/gdrive/{sent.json()['id']}", headers=auth).content == BOOK
        assert sandboxes.started[0].files == {}  # the file API writes nothing into the sandbox

        result = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert result["reward"] == 1.0 and result["sandbox_uploads_refused"] == 3.0


def test_tools_are_served_as_workspace_mode_describes_them_and_the_guide_says_where_files_go(sandboxes) -> None:
    """MCP serves a sandboxed episode's tools in workspace mode's words, OfficeEnv's, as the episode itself lists
    them (the seed checks they agree); an op that moves no content keeps its own words. The guide lists the file API's
    apps with OfficeEnv's notes and the folders their downloads land in."""
    server = make_server()
    in_mode = tools_exposure_catalog(server._workspace())
    without = tools_exposure_catalog()
    with mcp_client(server) as client:
        _, _, described = seed_with_content(client)
        assert described["gdrive_download_file"] == in_mode["gdrive_download_file"]["description"]
        assert described["gdrive_download_file"] != without["gdrive_download_file"]["description"]
        assert "savedTo" in described["gdrive_download_file"] and "content_path" in described["gdrive_create_file"]
        assert described["gdrive_list_files"] == without["gdrive_list_files"]["description"]
        session = server.service.store.get("rs-1")
        assert {tool["name"]: tool for tool in session.tools()}["gdrive_create_file"] == in_mode["gdrive_create_file"]
        guide = sandboxes.started[0].spec.files["/workspace/README.md"]
        file_apps = server.service.file_apps("rs-1")
        assert guide == workspace_guide(8192, ("gdrive", "user"), file_apps=file_apps)
        assert "gdrive (Google Drive)" in guide and f"Google Drive: {file_apps[0].note}." in guide
        assert "drive/<file id>/<name>" in guide and "at most 8192" in guide and "content_path" in guide
    assert "drive/<file id>" not in workspace_guide(8192, ("slack",))  # no file API app, no file API


def test_an_upload_takes_a_workspace_file_by_its_path(sandboxes) -> None:
    """An upload that names a workspace file (content_path) has the server read it from the sandbox before the call
    runs, as it reads a message's attachments: the file goes in the call byte for byte, and /verify counts it."""
    server = make_server()
    with mcp_client(server) as client:
        seeded, call, _ = seed_with_content(client)
        (sandbox,) = sandboxes.started
        sandbox.files["/workspace/out/totals.xlsx"] = BOOK  # as the model wrote it
        created = call("gdrive_create_file", name="totals.xlsx", parents=["F_FIN"], content_path="out/totals.xlsx")
        assert created["name"] == "totals.xlsx", created
        assert client.get(f"/v1/files/gdrive/{created['id']}", headers=file_auth(server)).content == BOOK
        missing = call("gdrive_create_file", name="x.csv", content_path="/workspace/none.csv")
        assert "No file at /workspace/none.csv" in refused_message(missing), missing
        assert sandbox.downloaded == ["/workspace/out/totals.xlsx"]  # the missing file was checked, never copied
        result = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert result["sandbox_files_uploaded"] == 1.0 and result["sandbox_bytes_uploaded"] == float(len(BOOK))
        assert "sandbox_trace_failures" not in result  # no trace store, nothing to count


@pytest.mark.parametrize("store_fails", [False, True], ids=["kept", "unwritable"])
def test_the_trace_store_keeps_what_an_episode_is_handed_and_its_failures_fail_no_call(
    sandboxes, tmp_path, caplog, store_fails: bool
) -> None:
    """With workspace_trace_store set, the files a call reads and the content a file API call sends are kept by
    SHA-256, for the episode's trace to replay without a sandbox. The store is a record beside the episode: a write
    it fails (here a store whose folder cannot be made; a full disk fails alike) is logged and counted in
    sandbox_trace_failures, and the call goes on, so the episode is graded on what the policy did, unmasked. Before,
    the upload failed (an internal_error tool output, a 500 from the file API) and the reward counted it."""
    if store_fails:
        (tmp_path / "blocked").write_text("a file where the store's parent folder goes")
    store = tmp_path / ("blocked" if store_fails else "ok") / "trace"
    server = make_server(workspace_trace_store=str(store))
    with mcp_client(server) as client:
        seeded, call, _ = seed_with_content(client)
        (sandbox,) = sandboxes.started
        sandbox.files["/workspace/out/totals.xlsx"] = BOOK  # as the model wrote it
        with caplog.at_level(logging.ERROR):
            created = call("gdrive_create_file", name="totals.xlsx", parents=["F_FIN"], content_path="out/totals.xlsx")
            sent = client.post(
                "/v1/files",
                params={"app": "gdrive", "name": "totals.csv"},
                content=TOTALS.encode(),
                headers=file_auth(server),
            )
        assert created.get("name") == "totals.xlsx", created
        assert sent.status_code == 200, sent.text
        result = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
    assert result["reward"] == 1.0 and not result.get("mask_sample"), result
    if store_fails:
        assert result["sandbox_trace_failures"] == 2.0
        assert caplog.text.count("Could not keep") == 2 and "NotADirectoryError" in caplog.text, caplog.text
    else:
        assert result["sandbox_trace_failures"] == 0.0
        handed = (BOOK, TOTALS.encode())  # the file the upload read, and the file API's content
        assert {path.name for path in store.iterdir()} == {hashlib.sha256(data).hexdigest() for data in handed}


def test_without_an_inline_limit_text_comes_inline_and_other_files_through_the_workspace(sandboxes) -> None:
    """inline_file_bytes None keeps workspace mode, which the episode's trace needs, with as much text inline as the
    file API takes: a long text file comes inline (and is saved), long text is taken pasted, and a binary file still
    moves through the workspace."""
    with mcp_client(make_server(sandbox_config={**IMAGE, "inline_file_bytes": None})) as client:
        seeded, call, _ = seed_with_content(client)
        (sandbox,) = sandboxes.started
        long = call("gdrive_download_file", file_id="F_LONG")
        assert long["content"] == LONG and sandbox.files[long["savedTo"]] == LONG.encode()
        book = call("gdrive_download_file", file_id="F_BOOK")
        assert "contentBase64" not in book and sandbox.files[book["savedTo"]] == BOOK
        assert call("gdrive_create_file", name="long.csv", parents=["F_FIN"], content=LONG)["name"] == "long.csv"
        assert f"at most {UNLIMITED_INLINE_FILE_BYTES} bytes" in sandbox.spec.files["/workspace/README.md"]
        verified = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert verified["officeenv_workspace"]["workspace"]["inline_bytes"] == UNLIMITED_INLINE_FILE_BYTES
        assert verified["sandbox_files_saved"] == 2.0 and verified["sandbox_uploads_refused"] == 0.0


def test_a_fault_the_task_injects_fails_a_file_api_call_with_the_apps_own_error(sandboxes) -> None:
    """A file API call is the agent's own action: an outage the task injects in Drive fails it as it fails Drive's
    tools, and the route answers with the app's status and body."""
    spec = content_spec()
    spec["faults"] = [{"id": "drive-down", "kind": "unavailable", "match": {"app": "gdrive"}}]
    server = make_server()
    with mcp_client(server) as client:
        seeded = client.post("/seed_session", json=seed_body(spec))
        assert seeded.status_code == 200, seeded.text
        auth = file_auth(server)
        down = client.get("/v1/files/gdrive/F_LEDGER", headers=auth)
        assert down.status_code == 503 and down.json()["detail"]["error"]["code"] == 503, down.text  # Drive's body
        sent = client.post("/v1/files", params={"app": "gdrive", "name": "t.csv"}, content=b"a,b\n", headers=auth)
        assert sent.status_code == 503, sent.text
        listed = client.post("/gdrive_list_files", json={}, cookies=seeded.cookies)
        assert listed.status_code == 200 and listed.json()["error"]["code"] == 503  # the tools meet it too


def test_a_download_the_sandbox_cannot_take_fails_the_call(sandboxes) -> None:
    with mcp_client(make_server()) as client:
        seeded, _, _ = seed_with_content(client)
        sandboxes.fail_on = "upload"
        got = client.post("/gdrive_download_file", json={"file_id": "F_BOOK"}, cookies=seeded.cookies)
        assert got.status_code == 502 and "/workspace/drive/F_BOOK/ledger.xlsx" in got.text
        assert "the sandbox is gone" not in got.text and "Error" not in got.text  # the provider's words stay logged
        sandboxes.fail_on = None  # the next call writes only its own file
        again = client.post("/gdrive_download_file", json={"file_id": "F_LEDGER"}, cookies=seeded.cookies)
        assert again.status_code == 200
        assert set(sandboxes.started[0].files) == {"/workspace/drive/F_LEDGER/ledger.csv"}
        # the provider failed, not the policy: the episode's reward is masked
        verified = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert verified["mask_sample"] is True and verified["failure_kind"] == "provider_unavailable"
        assert "the sandbox is gone" in verified["failure_reason"]


def test_a_save_the_workspace_is_in_the_way_of_fails_the_call_and_is_graded(sandboxes) -> None:
    """A file where the download's folder goes, or a folder at the file's path: what the workspace holds is the
    model's doing, so the call fails (502) and the episode is graded as usual."""
    with mcp_client(make_server()) as client:
        seeded, _, _ = seed_with_content(client)
        sandboxes.refusing = ("mkdir",)
        got = client.post("/gdrive_download_file", json={"file_id": "F_BOOK"}, cookies=seeded.cookies)
        assert got.status_code == 502 and "ledger.xlsx in the workspace: its folder could not be made" in got.text
        sandboxes.refusing = ("test ! -d",)
        got = client.post("/gdrive_download_file", json={"file_id": "F_BOOK"}, cookies=seeded.cookies)
        assert got.status_code == 502 and "ledger.xlsx in the workspace: a folder is there" in got.text
        verified = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert not verified["mask_sample"] and verified["failure_kind"] is None, verified


def test_a_file_put_where_a_download_folder_was_fails_the_next_save_and_is_graded(sandboxes) -> None:
    """A folder a download made is made again for the next save into it: the model may have removed it (it comes
    back), or put a file in its place. Then the call fails (502) as the workspace's doing, and the episode is graded,
    not masked as if the provider had failed."""
    with mcp_client(make_server()) as client:
        seeded, _, _ = seed_with_content(client)
        (sandbox,) = sandboxes.started
        for _ in range(2):
            got = client.post("/gdrive_download_file", json={"file_id": "F_BOOK"}, cookies=seeded.cookies)
            assert got.status_code == 200, got.text
        assert sandbox.commands.count("mkdir -p /workspace/drive/F_BOOK") == 2
        # A file where the folder was: mkdir refuses, and the provider's own copy would fail.
        sandboxes.refusing, sandboxes.fail_on = ("mkdir",), "upload"
        got = client.post("/gdrive_download_file", json={"file_id": "F_BOOK"}, cookies=seeded.cookies)
        assert got.status_code == 502 and "ledger.xlsx in the workspace: its folder could not be made" in got.text
        sandboxes.refusing, sandboxes.fail_on = (), None
        verified = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert not verified["mask_sample"] and verified["failure_kind"] is None, verified


def test_verification_waits_for_a_call_saving_its_download(sandboxes, monkeypatch) -> None:
    """/verify stops the sandbox under the episode's lock, which a call holds until its downloads are in the
    sandbox: a call under way when verification begins saves its file, and the sandbox stops after."""
    server = make_server()
    run = FakeSandbox.exec
    stopping: list[asyncio.Task] = []

    async def exec_as_verification_begins(sandbox: FakeSandbox, command: str, **options: object) -> SimpleNamespace:
        if command.startswith("mkdir") and not stopping:  # /verify comes in as the call saves its download
            stopping.append(asyncio.create_task(server._stop_before_grading("rs-1")))
            for _ in range(5):
                await asyncio.sleep(0)
        return await run(sandbox, command, **options)

    with mcp_client(server) as client:
        seeded, _, _ = seed_with_content(client)
        sandbox = sandboxes.started[0]
        held = []  # what the sandbox held as it stopped
        sandboxes.on_stop = lambda: held.append(set(sandbox.files))
        monkeypatch.setattr(FakeSandbox, "exec", exec_as_verification_begins)
        got = client.post("/gdrive_download_file", json={"file_id": "F_BOOK"}, cookies=seeded.cookies)
        assert got.status_code == 200, got.text
        verified = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies)  # after the stop under way
        assert verified.status_code == 200 and not verified.json()["mask_sample"], verified.text
        assert held == [{"/workspace/drive/F_BOOK/ledger.xlsx"}]  # the file saved, then the sandbox stopped, once


def test_a_download_whose_path_holds_a_folder_fails_the_call(sandboxes) -> None:
    """A folder at the file's path would take the copy inside it, and the answer would name the folder."""
    with mcp_client(make_server()) as client:
        seeded, _, _ = seed_with_content(client)
        sandboxes.refusing = ("test ! -d",)  # the model made a folder where the file goes
        got = client.post("/gdrive_download_file", json={"file_id": "F_BOOK"}, cookies=seeded.cookies)
        assert (
            got.status_code == 502
            and "/workspace/drive/F_BOOK/ledger.xlsx in the workspace: a folder is there" in got.text
        )
        assert sandboxes.started[0].files == {}


def test_a_saved_download_whose_mode_cannot_be_set_is_answered(sandboxes) -> None:
    with mcp_client(make_server()) as client:
        seeded, _, _ = seed_with_content(client)
        sandboxes.refusing = ("chmod",)
        got = client.post("/gdrive_download_file", json={"file_id": "F_BOOK"}, cookies=seeded.cookies)
        assert got.status_code == 200 and json.loads(got.text)["savedTo"] == "/workspace/drive/F_BOOK/ledger.xlsx"
        assert sandboxes.started[0].files["/workspace/drive/F_BOOK/ledger.xlsx"] == BOOK


def test_without_a_sandbox_file_content_moves_as_before(sandboxes) -> None:
    """An episode of the OfficeEnv server, and a plain row's seed on the sandboxed one, answer and take file content
    in full, with OfficeEnv's descriptions."""
    catalog = tools_exposure_catalog()
    with mcp_client(make_base_server()) as client:
        seeded, call, described = seed_with_content(client)
        book = call("gdrive_download_file", file_id="F_BOOK")
        assert book["contentBase64"] == BOOK_B64 and "savedTo" not in book
        assert call("gdrive_download_file", file_id="F_LONG")["content"] == LONG
        assert call("gdrive_create_file", name="t.xlsx", content_base64=BOOK_B64)["name"] == "t.xlsx"
        assert described["gdrive_download_file"] == catalog["gdrive_download_file"]["description"]
        assert "sandbox_files_saved" not in client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
    with mcp_client(make_server()) as client:
        row = {"responses_create_params": {"input": []}, "officeenv": content_spec(), "task_id": "sandbox-files"}
        seeded = client.post("/seed_session", json=row)
        assert seeded.status_code == 200, seeded.text
        got = client.post("/gdrive_download_file", json={"file_id": "F_BOOK"}, cookies=seeded.cookies).json()
        assert got["contentBase64"] == BOOK_B64 and "savedTo" not in got and not sandboxes.started


def refused_message(answer: dict) -> str:
    """An app's error message, in Drive's, Box's or Graph's envelope, or a failed command's error text."""
    error = answer.get("error")
    if isinstance(error, str):
        return error
    return str((error if isinstance(error, dict) else answer).get("message") or "")


# Gmail's attachments move through the workspace too.

MAIL_FILES = {"/workspace/out/totals.xlsx": BOOK, "/workspace/notes.csv": LONG.encode()}  # as the model writes them
MESSAGE = {"to": ["sam@example.com"], "subject": "Totals", "body": "Attached."}


def with_mail(spec: dict) -> dict:
    """``spec`` with Gmail connected: Alex's mailbox holds a message (M_LEDGER) carrying the workbook, the long CSV
    and a short one."""
    spec["apps"]["gmail"] = {
        "data": {
            "mailboxes": [{"email": ME, "name": "Alex Morgan"}],
            "messages": [
                {
                    "id": "M_LEDGER",
                    "owner": ME,
                    "sender": "sam@example.com",
                    "to": [ME],
                    "subject": "Ledgers",
                    "body": "All three attached.",
                    "date": "2026-09-20T08:00:00Z",
                    "labelIds": ["INBOX"],
                    "attachments": [
                        {"filename": "ledger.xlsx", "content_base64": BOOK_B64},
                        {"filename": "ledger long.csv", "content": LONG},
                        {"filename": "totals.csv", "content": SHORT},
                    ],
                }
            ],
        }
    }
    return spec


def seed_mail(client: TestClient, sandboxes: type[FakeSandbox], files: dict | None = None) -> tuple:
    """Seed with_mail(content_spec()), with ``files`` (MAIL_FILES) in the workspace as if the model had written them:
    the seed's answer, a function that calls a tool over MCP and reads its JSON output, and the tools as listed."""
    seeded = client.post("/seed_session", json=seed_body(with_mail(content_spec())))
    assert seeded.status_code == 200, seeded.text
    token = seeded.json()["resources_tools"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]
    sandboxes.started[-1].files.update(MAIL_FILES if files is None else files)

    def call(tool: str, /, **arguments: object) -> dict:
        result = rpc(client, "tools/call", {"name": tool, "arguments": arguments}, token).json()["result"]
        return result.get("structuredContent") or json.loads(result["content"][0]["text"])

    listed = rpc(client, "tools/list", {}, token).json()["result"]["tools"]
    return seeded, call, {tool["name"]: tool for tool in listed}


def test_a_message_attaches_a_workspace_file_by_its_path(sandboxes) -> None:
    """The server reads the files a message names from the sandbox before the call runs, and the message carries them
    byte for byte; getting one back saves it in the workspace. A relative path is in the workspace."""
    with mcp_client(make_server()) as client:
        seeded, call, _ = seed_mail(client, sandboxes)
        sandbox = sandboxes.started[0]
        attachments = [{"path": "/workspace/out/totals.xlsx"}, {"path": "notes.csv", "filename": "Q3 notes.csv"}]
        sent = call("gmail_send_message", **MESSAGE, attachments=attachments)
        assert "error" not in sent, sent
        assert sandbox.downloaded == ["/workspace/out/totals.xlsx", "/workspace/notes.csv"]
        listed = call("gmail_get_message", id=sent["id"])["attachments"]
        assert [(a["filename"], a["size"]) for a in listed] == [
            ("totals.xlsx", len(BOOK)),
            ("Q3 notes.csv", len(LONG)),
        ]
        back = call("gmail_get_attachment", message_id=sent["id"], attachment_id="totals.xlsx")
        assert sandbox.files[back["savedTo"]] == BOOK  # through the message and back, byte for byte
        result = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert result["sandbox_files_attached"] == 2.0 and result["sandbox_uploads_refused"] == 0.0
        assert result["sandbox_bytes_attached"] == float(len(BOOK) + len(LONG))
        assert result["sandbox_files_saved"] == 1.0 and result["sandbox_bytes_saved"] == float(len(BOOK))


def test_an_attachment_is_read_only_from_a_regular_file_of_the_sandbox(sandboxes, tmp_path, monkeypatch) -> None:
    """``docker cp`` copies a link as a link, which the server would read on its own host: a link, like a folder, is
    no file to attach, and neither is one over the upload limit. Each is refused before anything is copied out."""
    secret = tmp_path / "host.txt"
    secret.write_text("a file of the host\n")
    sandboxes.links, sandboxes.folders = {"/workspace/link.txt": str(secret)}, {"/workspace/out"}
    monkeypatch.setattr(sandbox_app, "MAX_UPLOAD_BYTES", 8)  # the episode's Workspace takes it at its seed
    server = make_server()
    with mcp_client(server) as client:
        _, call, _ = seed_mail(client, sandboxes)
        for path in ("/workspace/link.txt", "/workspace/out"):
            refused = call("gmail_send_message", **MESSAGE, attachments=[{"path": path}])
            assert f"No file at {path}" in refused_message(refused), refused
        too_big = call("gmail_send_message", **MESSAGE, attachments=[{"path": "notes.csv"}])
        assert "/workspace/notes.csv is over 8 bytes" in refused_message(too_big), too_big
        assert sandboxes.started[-1].downloaded == []  # the check in the sandbox refused each of them
        messages = server.service.store.get("rs-1").world.collection("gmail", "messages").rows.values()
        assert [row["subject"] for row in messages] == ["Ledgers"]  # nothing was sent


def test_attachments_are_checked_in_the_sandbox_before_anything_is_copied(sandboxes, monkeypatch) -> None:
    """A device, even through a linked folder, a file over the upload limit, more files than a message attaches, and
    files together over its limit are refused without a download."""
    sandboxes.kinds = {
        "/workspace/d/zero": "character special file|0",
        "/workspace/huge.bin": "regular file|1073741824",
    }
    # Gmail's attachments have no caps of their own, so these are this server's, which the episode's Workspace takes
    # at its seed and OfficeEnv's refusals name
    monkeypatch.setattr(sandbox_app, "MAX_ATTACHMENT_BYTES", len(BOOK) + len(LONG) - 1)
    server = make_server()
    with mcp_client(server) as client:
        _, call, _ = seed_mail(client, sandboxes)
        sandbox = sandboxes.started[-1]

        def refusal(*attachments: object) -> str:
            return refused_message(call("gmail_send_message", **MESSAGE, attachments=list(attachments)))

        assert "No file at /workspace/d/zero" in refusal({"path": "d/zero"})  # /dev/zero through a link
        assert "/workspace/huge.bin is over 20 MB" in refusal({"path": "huge.bin"})
        many = [{"path": f"f{i}.csv"} for i in range(sandbox_app.MAX_ATTACHMENTS + 1)]
        assert "one call sends at most 20 files from the workspace" in refusal(*many)
        both = refusal({"path": "out/totals.xlsx"}, {"path": "notes.csv"})
        assert f"together are over {len(BOOK) + len(LONG) - 1} bytes" in both, both
        assert sandbox.downloaded == []  # none of them was copied out
        messages = server.service.store.get("rs-1").world.collection("gmail", "messages").rows.values()
        assert [row["subject"] for row in messages] == ["Ledgers"]  # nothing was sent


def test_an_attachment_the_provider_fails_to_read_fails_the_call_and_masks_the_episode(sandboxes, monkeypatch) -> None:
    """A copy cut short attaches nothing, and the file is still there: the provider failed, not the model. The call
    fails without the provider's words, nothing is sent, and the episode's verification is masked; so it is when the
    check cannot run."""
    sandboxes.partial = {"/workspace/cut.csv": b"vendor,to"}
    server = make_server()
    with mcp_client(server) as client:
        seeded, _, _ = seed_mail(client, sandboxes)
        cut = client.post(
            "/gmail_send_message", json={**MESSAGE, "attachments": [{"path": "cut.csv"}]}, cookies=seeded.cookies
        )
        assert cut.status_code == 502 and cut.json()["detail"] == "Could not read /workspace/cut.csv in the workspace"
        run = FakeSandbox.exec

        async def check_fails(sandbox: FakeSandbox, command: str, **options: object) -> SimpleNamespace:
            if shlex.split(command)[:3] == ["sh", "-c", STAT_SCRIPT]:
                return SimpleNamespace(error_type="timeout", return_code=None, stdout="", stderr="")
            return await run(sandbox, command, **options)

        monkeypatch.setattr(FakeSandbox, "exec", check_fails)
        unchecked = client.post(
            "/gmail_send_message", json={**MESSAGE, "attachments": [{"path": "notes.csv"}]}, cookies=seeded.cookies
        )
        assert unchecked.status_code == 502
        assert unchecked.json()["detail"] == "Could not check the files in the workspace"
        messages = server.service.store.get("rs-1").world.collection("gmail", "messages").rows.values()
        assert [row["subject"] for row in messages] == ["Ledgers"]  # nothing was sent
        verified = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert verified["mask_sample"] is True and verified["failure_kind"] == "provider_unavailable"
        assert "the copy was cut short" in verified["failure_reason"]


def test_an_attachment_gone_since_the_check_is_missing(sandboxes) -> None:
    """A file removed between the check and the copy is missing, as the model left it: the call is refused with its
    path, and the episode is graded as usual."""
    with mcp_client(make_server()) as client:
        seeded, call, _ = seed_mail(client, sandboxes)
        sandbox = sandboxes.started[-1]
        sandboxes.on_download = lambda: sandbox.files.pop("/workspace/notes.csv")
        gone = call("gmail_send_message", **MESSAGE, attachments=[{"path": "notes.csv"}])
        assert "No file at /workspace/notes.csv" in refused_message(gone), gone
        verified = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert not verified["mask_sample"] and verified["failure_kind"] is None, verified


def test_a_call_whose_verification_begins_while_its_attachments_are_read_is_refused(sandboxes) -> None:
    server = make_server()
    with mcp_client(server) as client:
        seeded, _, _ = seed_mail(client, sandboxes)
        sandboxes.on_download = lambda: setattr(server._episodes["rs-1"], "grading", True)
        body = {**MESSAGE, "attachments": [{"path": "notes.csv"}]}
        sent = client.post("/gmail_send_message", json=body, cookies=seeded.cookies)
        assert sent.status_code == 409 and "its tools are closed" in sent.text
        messages = server.service.store.get("rs-1").world.collection("gmail", "messages").rows.values()
        assert [row["subject"] for row in messages] == ["Ledgers"]  # nothing was sent


# The cli and api_search exposures: the apps as commands, or as endpoints found over MCP and called with curl.


def _names_the_docs_tool() -> bool:
    from officeenv.exposure import Catalog, build_exposure

    tools = build_exposure("api_search", Catalog({}), {"docs_tool": "api_docs"}).tools()
    return "api_docs" in {tool["name"] for tool in tools}


# api_search's documentation tool named api_docs (exposure option docs_tool) is OfficeEnv 1.11.0's: an older one
# ignores the option. Run these against a newer OfficeEnv: install it, or put its src on PYTHONPATH.
NEEDS_DOCS_TOOL = pytest.mark.skipif(not _names_the_docs_tool(), reason="needs OfficeEnv 1.11.0's option docs_tool")


def in_exposure(spec: dict, exposure: str, **options: object) -> dict:
    """A copy of ``spec`` with its task in exposure ``exposure``, with the exposure ``options`` given."""
    spec = json.loads(json.dumps(spec))
    spec["exposure"] = {"mode": exposure, **({"options": options} if options else {})}
    return spec


def nul_args(*argv: str) -> bytes:
    """Arguments as an app's command sends them: each one followed by a NUL."""
    return b"".join(arg.encode() + b"\0" for arg in argv)


def command_flags(arguments: dict) -> list[str]:
    """An op's arguments as its command takes them: --kebab-case flags, JSON for any value that is not text."""
    return [
        f"--{key.replace('_', '-')}={value if isinstance(value, str) else json.dumps(value)}"
        for key, value in arguments.items()
    ]


def seed_in(
    client: TestClient,
    sandboxes: type[FakeSandbox],
    spec: dict,
    exposure: str,
    files: dict | None = None,
    options: dict | None = None,
):
    """Seed ``spec`` in ``exposure`` (with the exposure ``options``), with ``files`` in the workspace as if the model
    had written them: the seed's answer, the bearer header of the episode's API, an MCP caller
    (``tool(name, **arguments)``), and ``call(app, op, **arguments)``, which calls an op the exposure's way (an MCP
    tool, the app's command, an endpoint) and reads its JSON answer, or a failed command's error text as
    ``{"error": <stderr>}``."""
    seeded = client.post("/seed_session", json=seed_body(in_exposure(spec, exposure, **(options or {}))))
    assert seeded.status_code == 200, seeded.text
    sandbox = sandboxes.started[-1]
    sandbox.files.update(files or {})
    auth = {"Authorization": f"Bearer {sandbox.spec.env['COMPANY_API_TOKEN']}"}
    grant = seeded.json()["resources_tools"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]

    def tool(name: str, /, **arguments: object) -> dict:
        result = rpc(client, "tools/call", {"name": name, "arguments": arguments}, grant).json()["result"]
        return result.get("structuredContent") or json.loads(result["content"][0]["text"])

    def call(app: str, op: str, /, **arguments: object) -> dict:
        if exposure == "tools":
            return tool(f"{app}_{op}", **arguments)
        if exposure == "api_search":
            return client.post(f"/v1/{app}/{op}", headers=auth, json=arguments).json()
        command = nul_args(app, op.replace("_", "-"), *command_flags(arguments))
        ran = client.post("/v1/cli", headers={**auth, "User-Agent": CLI_USER_AGENT.format(app=app)}, content=command)
        return json.loads(ran.text) if ran.headers["x-exit-code"] == "0" else {"error": ran.text}

    return seeded, auth, tool, call


def test_a_cli_episode_has_its_apps_as_commands_and_no_mcp_tools(sandboxes) -> None:
    server = make_server()
    with mcp_client(server) as local, mcp_client(server, client=SANDBOX) as remote:
        seeded = local.post("/seed_session", json=seed_body(in_exposure(SPEC, "cli")))
        assert seeded.status_code == 200, seeded.text
        grant = seeded.json()["resources_tools"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]
        assert rpc(local, "tools/list", {}, grant).json()["result"]["tools"] == []
        (sandbox,) = sandboxes.started
        assert sandbox.spec.files[APP_COMMAND_PATH] == APP_COMMAND_SOURCE
        guide = sandbox.spec.files["/workspace/README.md"]
        file_apps = server.service.file_apps("rs-1")
        assert guide == workspace_guide(8192, ("gdrive", "user"), exposure="cli", file_apps=file_apps)
        assert "commands here, one per connected app: gdrive, user." in guide
        setup = sandbox.commands[0]  # before the sandbox is lent
        assert f"chmod 755 {APP_COMMAND_PATH}" in setup and f"ln -s {APP_COMMAND_PATH} /usr/local/bin/gdrive" in setup
        assert f"ln -s {APP_COMMAND_PATH} /usr/local/bin/user" in setup

        # the commands reach the episode from the sandbox, with its token and as the app's command
        token = {"Authorization": f"Bearer {sandbox.spec.env['COMPANY_API_TOKEN']}"}
        auth = {**token, "User-Agent": CLI_USER_AGENT.format(app="gdrive")}
        listed = remote.post(
            "/v1/cli", headers=auth, content=nul_args("gdrive", "list-files", "--q=name contains 'led'")
        )
        assert listed.status_code == 200 and listed.headers["x-exit-code"] == "0", listed.text
        assert "ledger.csv" in listed.text and listed.text.endswith("\n")
        missing = remote.post("/v1/cli", headers=auth, content=nul_args("gdrive", "get-file", "--file-id=F_NONE"))
        assert missing.headers["x-exit-code"] == "1" and missing.text.startswith("error:")
        helped = remote.post("/v1/cli", headers=auth, content=nul_args("gdrive", "--help"))
        assert helped.headers["x-exit-code"] == "0" and "list-files" in helped.text
        assert remote.post("/v1/cli", content=nul_args("gdrive", "--help")).status_code == 401
        # a call made by hand rather than by the command is refused, pointed at the command, and counted
        by_hand = remote.post("/v1/cli", headers=token, content=nul_args("gdrive", "list-files"))
        assert by_hand.status_code == 403 and "gdrive <command>" in by_hand.text
        assert "x-exit-code" not in by_hand.headers
        # the apps are commands here, not endpoints
        assert remote.post("/v1/gdrive/list_files", headers=token, json={}).status_code == 404

        created = remote.post(
            "/v1/cli",
            headers=auth,
            content=nul_args("gdrive", "create-file", "--name=totals.csv", "--parents=F_FIN", f"--content={TOTALS}"),
        )
        assert created.headers["x-exit-code"] == "0", created.text
        verified = local.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert verified["reward"] == 1.0, verified
        assert verified["sandbox_commands"] == 4.0 and verified["sandbox_commands_refused"] == 1.0
        assert verified["sandbox_endpoint_calls"] == 0.0


@pytest.mark.parametrize("docs_tool", [None, "api_fetch", pytest.param("api_docs", marks=NEEDS_DOCS_TOOL)])
def test_an_api_search_episode_finds_endpoints_over_mcp_and_calls_them_over_http(sandboxes, docs_tool) -> None:
    """The finding tools over MCP, the documentation tool by the name the row's exposure gives it (docs_tool, by
    default api_fetch), with each endpoint's curl command, and the endpoints over HTTP."""
    docs = docs_tool or "api_fetch"
    other = "api_docs" if docs == "api_fetch" else "api_fetch"
    server = make_server()
    with mcp_client(server) as local, mcp_client(server, client=SANDBOX) as remote:
        options = {"docs_tool": docs_tool} if docs_tool else None
        seeded, auth, tool, _ = seed_in(local, sandboxes, SPEC, "api_search", options=options)
        grant = seeded.json()["resources_tools"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]
        tools = rpc(local, "tools/list", {}, grant).json()["result"]["tools"]
        assert {listed["name"] for listed in tools} == {"api_search", docs}
        guide = sandboxes.started[0].spec.files["/workspace/README.md"]
        file_apps = server.service.file_apps("rs-1")
        assert guide == workspace_guide(
            8192, ("gdrive", "user"), exposure="api_search", docs_tool=docs_tool, file_apps=file_apps
        )
        assert f"read its parameters with your {docs} tool" in guide and other not in guide
        found = tool("api_search", query="list drive files")
        hit = next(result for result in found["results"] if result["endpoint"] == "gdrive.list_files")
        assert hit["path"] == "/v1/gdrive/list_files"
        fetched = tool(docs, endpoint="gdrive.list_files")
        assert fetched["http"]["path"] == "/v1/gdrive/list_files"
        assert '"$COMPANY_API_URL/v1/gdrive/list_files"' in fetched["http"]["curl"]
        # the endpoints that move content are documented as workspace mode runs them, whatever the tool's name
        assert "savedTo" in tool(docs, endpoint="gdrive.download_file")["description"]
        refused = rpc(local, "tools/call", {"name": other, "arguments": {"endpoint": "gdrive.list_files"}}, grant)
        assert refused.json()["result"]["isError"] is True  # the other name is no tool of this episode's

        listed = remote.post("/v1/gdrive/list_files", headers=auth, json={"q": "name contains 'ledger'"})
        assert listed.status_code == 200 and "ledger.csv" in listed.text
        missing = remote.post("/v1/gdrive/get_file", headers=auth, json={"file_id": "F_NONE"})
        assert missing.status_code == 404 and "error" in missing.json()  # the app's error, under its status
        assert remote.post("/v1/gdrive/list_files", headers=auth, content=b"[1]").status_code == 400
        assert remote.post("/v1/gdrive/list_files", json={}).status_code == 401
        user_agent = {**auth, "User-Agent": CLI_USER_AGENT.format(app="gdrive")}
        assert remote.post("/v1/cli", headers=user_agent, content=nul_args("gdrive")).status_code == 404

        created = remote.post(
            "/v1/gdrive/create_file",
            headers=auth,
            json={"name": "totals.csv", "parents": ["F_FIN"], "content": TOTALS},
        )
        assert created.status_code == 200, created.text
        verified = local.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert verified["reward"] == 1.0, verified
        assert verified["sandbox_endpoint_calls"] == 3.0  # the body that was no JSON object never ran


def test_the_sandbox_gets_the_command_without_its_comments() -> None:
    lines = APP_COMMAND_SOURCE.splitlines()
    assert lines[0] == "#!/bin/sh" and not any(line.lstrip().startswith("#") for line in lines[1:])
    assert '-A "${0##*/}-cli/1.0"' in APP_COMMAND_SOURCE and CLI_USER_AGENT == "{app}-cli/1.0"
    assert f". {CREDENTIALS_PATH}" in APP_COMMAND_SOURCE  # for a shell started without the API's variables
    assert not ENVIRONMENT_NAMES.search(APP_COMMAND_SOURCE)


def test_the_finding_tools_take_the_api_search_exposures_arguments() -> None:
    """MCP serves api_search's search and its documentation tool under each name the installed OfficeEnv's exposure
    may give it (DOCS_TOOLS), each taking the arguments the exposure's own tool takes."""
    from officeenv.session import Session
    from officeenv.task import TaskSpec

    own = {}
    for docs in DOCS_TOOLS:
        session = Session(TaskSpec.model_validate(in_exposure(SPEC, "api_search", docs_tool=docs)))
        own |= {tool["name"]: tool["parameters"] for tool in session.tools()}
    assert [name for name, _, _ in SEARCH_TOOLS] == ["api_search", *DOCS_TOOLS]
    assert DOCS_TOOLS[0] == "api_fetch"  # the exposure's default name
    for name, schema, _ in SEARCH_TOOLS:
        assert (
            set(schema["properties"]) == set(own[name]["properties"]) and schema["required"] == own[name]["required"]
        )


@NEEDS_DOCS_TOOL
def test_the_documentation_tools_names_are_the_installed_officeenvs() -> None:
    """An OfficeEnv that names the documentation tool api_docs lists its names, and MCP serves the tool under each:
    the names are OfficeEnv's, not the fallback for the versions before 1.11.0, which name it api_fetch alone."""
    assert "api_docs" in DOCS_TOOLS


def test_the_documentation_tool_is_the_one_the_episode_names() -> None:
    """with_http_calls gives the curl command in the documentation of the tool the episode's exposure names
    (docs_tool), and leaves any other tool's answer as it is."""
    doc = json.dumps({"endpoint": "gdrive.list_files", "examples": [{"q": "trashed = false"}]})
    assert "curl" in json.loads(with_http_calls("api_docs", doc, docs_tool="api_docs"))["http"]
    assert "curl" in json.loads(with_http_calls("api_fetch", doc))["http"]  # the default name
    assert with_http_calls("api_docs", doc) == doc
    assert with_http_calls("api_fetch", doc, docs_tool="api_docs") == doc


def test_api_fetch_shows_a_curl_command_that_keeps_its_example_intact() -> None:
    doc = {"endpoint": "gdrive.list_files", "examples": [{"q": "name contains 'budget' and trashed = false"}]}
    curl = shlex.split(json.loads(with_http_calls("api_fetch", json.dumps(doc)))["http"]["curl"])
    assert curl[curl.index("-d") + 1] == json.dumps(doc["examples"][0])
    error = '{"error":{"code":"unknown_endpoint"}}'
    assert with_http_calls("api_fetch", error) == error
    assert with_http_calls("gdrive_list_files", "{}") == "{}"


@pytest.mark.skipif(not (shutil.which("sh") and shutil.which("curl")), reason="needs sh and curl")
def test_an_apps_command_runs_through_the_server(sandboxes, tmp_path) -> None:
    """The command as a sandbox has it (the script it is given, linked under the app's name) run by sh against a
    server on a socket: its answers and exit codes, and the file API with the guide's curl commands beside it."""
    import httpx
    import uvicorn

    server = make_server()
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    web = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=web.run, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if web.started:
                break
            time.sleep(0.05)
        with httpx.Client(base_url=f"http://127.0.0.1:{port}") as http:
            assert http.post("/seed_session", json=seed_body(in_exposure(content_spec(), "cli"))).status_code == 200
        script = tmp_path / "app"
        script.write_text(APP_COMMAND_SOURCE)
        command = tmp_path / "gdrive"
        command.symlink_to(script)
        env = {
            **os.environ,
            "COMPANY_API_URL": f"http://127.0.0.1:{port}",
            "COMPANY_API_TOKEN": sandboxes.started[0].spec.env["COMPANY_API_TOKEN"],
        }

        def run(*argv: str, **overrides: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["sh", str(command), *argv], env={**env, **overrides}, capture_output=True, text=True, timeout=60
            )

        listed = run("list-files", "--q=name contains 'ledger'")
        assert listed.returncode == 0 and "ledger.csv" in listed.stdout and listed.stderr == "", listed
        missing = run("get-file", "--file-id=F_NONE")
        assert missing.returncode == 1 and missing.stdout == "" and missing.stderr.startswith("error:"), missing
        refused = run("--help", COMPANY_API_TOKEN="x")
        assert refused.returncode == 1 and "invalid bearer token" in refused.stderr, refused
        saved = json.loads(run("download-file", "--file-id=F_BOOK").stdout)
        assert saved["savedTo"] == "/workspace/drive/F_BOOK/ledger.xlsx" and "contentBase64" not in saved

        # a shell started without the API's variables: the command reads them from the file the seed wrote (here at a
        # path of the test's own, since /etc is the host's), and so does a shell that sources it, as the guide says
        credentials = tmp_path / "api.env"
        written = sandboxes.started[0].spec.files[CREDENTIALS_PATH]
        credentials.write_text(written.replace("http://10.0.0.1:8080", f"http://127.0.0.1:{port}"))
        script.write_text(APP_COMMAND_SOURCE.replace(CREDENTIALS_PATH, str(credentials)))
        bare = {key: value for key, value in env.items() if not key.startswith("COMPANY_API_")}
        listed = subprocess.run(
            ["sh", str(command), "list-files"], env=bare, capture_output=True, text=True, timeout=60
        )
        assert listed.returncode == 0 and "ledger.csv" in listed.stdout, listed
        sourced = subprocess.run(
            ["sh", "-c", f". {shlex.quote(str(credentials))} && sh -c 'echo \"$COMPANY_API_TOKEN\"'"],
            env=bare,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert sourced.stdout.strip() == env["COMPANY_API_TOKEN"]  # set and exported

        # the file API as the guide gives it (curl -sf) moves files on a cli episode too, and an app's error is a
        # failure curl reports
        book = tmp_path / "book.xlsx"
        book.write_bytes(BOOK)

        def curl(*argv: str) -> subprocess.CompletedProcess:
            bearer = f"Authorization: Bearer {env['COMPANY_API_TOKEN']}"
            return subprocess.run(["curl", "-sf", "-H", bearer, *argv], capture_output=True, timeout=60)

        files = f"{env['COMPANY_API_URL']}/v1/files"
        sent = curl("-X", "POST", "--data-binary", f"@{book}", f"{files}?app=gdrive&name=totals.xlsx")
        assert sent.returncode == 0, sent
        back = tmp_path / "back.xlsx"
        assert curl("-o", str(back), f"{files}/gdrive/{json.loads(sent.stdout)['id']}").returncode == 0
        assert back.read_bytes() == BOOK
        assert curl("-X", "PUT", "--data-binary", f"@{book}", f"{files}/gdrive/F_NONE").returncode == 22  # HTTP 404
    finally:
        web.should_exit = True
        thread.join(timeout=10)


@pytest.mark.parametrize("exposure", SANDBOX_EXPOSURES)
def test_the_file_api_moves_files_whatever_the_exposure(sandboxes, exposure: str) -> None:
    """A cli episode has no gdrive_create_file tool, and an api_search episode neither: the file API's calls are the
    agent's own, whatever its exposure, which the episode's trace records, and an app's error comes back under its
    HTTP status."""
    server = make_server()
    with mcp_client(server) as client:
        seeded, auth, _, _ = seed_in(client, sandboxes, content_spec(), exposure)
        params = {"app": "gdrive", "name": "totals.xlsx", "parent": "F_FIN"}
        uploaded = client.post("/v1/files", params=params, content=BOOK, headers=auth)
        assert uploaded.status_code == 200, uploaded.text
        created = uploaded.json()
        assert created["name"] == "totals.xlsx" and created["size"] == str(len(BOOK)), created
        assert client.get(f"/v1/files/gdrive/{created['id']}", headers=auth).content == BOOK
        noise = bytes(range(256)) * 4  # every byte value, no UTF-8 text
        assert client.put(f"/v1/files/gdrive/{created['id']}", content=noise, headers=auth).status_code == 200
        assert client.get(f"/v1/files/gdrive/{created['id']}", headers=auth).content == noise
        assert client.get("/v1/files/gdrive/F_LONG", headers=auth).content == LONG.encode()  # over the inline limit
        missing = [
            client.get("/v1/files/gdrive/F_NONE", headers=auth),
            client.put("/v1/files/gdrive/F_NONE", content=b"a,b\n", headers=auth),
            client.post("/v1/files", params={**params, "parent": "F_NONE"}, content=b"a,b\n", headers=auth),
        ]
        assert [answer.status_code for answer in missing] == [404, 404, 404], [answer.text for answer in missing]
        assert sandboxes.started[-1].files == {}  # the file API writes nothing into the sandbox
        verified = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert verified["sandbox_uploads_refused"] == 0.0  # the file API pastes nothing
        moved = {entry["file"] for entry in verified["officeenv_workspace"]["calls"] if "file" in entry}
        assert moved == {"create", "download", "replace"}


@pytest.mark.parametrize("exposure", ["cli", "api_search"])
def test_commands_and_endpoints_keep_file_content_out_of_the_conversation(sandboxes, exposure: str) -> None:
    """A command and an endpoint save a download in the workspace as a tool does, the endpoint's files written by
    this server too, and refuse pasted content with the file API's command."""
    with mcp_client(make_server()) as client:
        seeded, _, _, call = seed_in(client, sandboxes, content_spec(), exposure)
        sandbox = sandboxes.started[-1]
        book = call("gdrive", "download_file", file_id="F_BOOK")
        assert not {"content", "contentBase64", "content_base64", "preview"} & set(book), book
        assert book["savedTo"] == "/workspace/drive/F_BOOK/ledger.xlsx" and sandbox.files[book["savedTo"]] == BOOK
        where = {"name": "Q3 totals.csv", "parents": ["F_FIN"]}
        refused = call("gdrive", "create_file", **where, content_base64=BOOK_B64)
        assert "binary content" in refused_message(refused) and "curl -sf" in refused_message(refused), refused
        assert "error" not in call("gdrive", "create_file", **where, content=SHORT)  # short text still goes in a call
        result = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert result["sandbox_files_saved"] == 1.0 and result["sandbox_uploads_refused"] == 1.0


@pytest.mark.parametrize("exposure", ["cli", "api_search"])
def test_commands_and_endpoints_move_mail_attachments_through_the_workspace(sandboxes, exposure: str) -> None:
    with mcp_client(make_server()) as client:
        seeded, _, _, call = seed_in(client, sandboxes, with_mail(content_spec()), exposure, files=MAIL_FILES)
        sandbox = sandboxes.started[-1]
        book = call("gmail", "get_attachment", message_id="M_LEDGER", attachment_id="ledger.xlsx")
        assert book["savedTo"].startswith("/workspace/gmail/M_LEDGER/") and sandbox.files[book["savedTo"]] == BOOK
        attachments = [{"path": "/workspace/out/totals.xlsx"}, {"path": "notes.csv", "filename": "Q3 notes.csv"}]
        sent = call("gmail", "send_message", **MESSAGE, attachments=attachments)
        assert "error" not in sent, sent
        listed = call("gmail", "get_message", id=sent["id"])["attachments"]
        assert [(a["filename"], a["size"]) for a in listed] == [
            ("totals.xlsx", len(BOOK)),
            ("Q3 notes.csv", len(LONG)),
        ]
        pasted = call(
            "gmail", "send_message", **MESSAGE, attachments=[{"filename": "t.xlsx", "content_base64": BOOK_B64}]
        )
        assert '{"path": "/workspace/t.xlsx"}' in refused_message(pasted), pasted
        result = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert result["sandbox_files_attached"] == 2.0 and result["sandbox_uploads_refused"] == 1.0


@pytest.mark.parametrize("exposure", ["cli", "api_search"])
def test_commands_and_endpoints_refuse_once_verification_begins(sandboxes, exposure: str) -> None:
    server = make_server()
    with mcp_client(server) as client:
        _, auth, _, _ = seed_in(client, sandboxes, content_spec(), exposure)
        (episode,) = server._episodes.values()
        episode.grading = True  # as /verify sets it before it stops the sandbox
        if exposure == "cli":
            as_drive = {**auth, "User-Agent": CLI_USER_AGENT.format(app="gdrive")}
            got = client.post("/v1/cli", headers=as_drive, content=nul_args("gdrive", "list-files"))
        else:
            got = client.post("/v1/gdrive/list_files", headers=auth, json={})
        assert got.status_code == 409 and "The task has ended" in got.text, got.text


@pytest.mark.parametrize("exposure", ["cli", "api_search"])
def test_a_save_the_provider_fails_fails_a_command_or_an_endpoint_as_a_tool(sandboxes, exposure: str) -> None:
    """Commands and endpoints run under the episode's lock as the tools do: a save the provider fails answers 502
    without the provider's words, and the episode's verification is masked."""
    with mcp_client(make_server()) as client:
        seeded, auth, _, _ = seed_in(client, sandboxes, content_spec(), exposure)
        sandboxes.fail_on = "upload"
        if exposure == "cli":
            as_drive = {**auth, "User-Agent": CLI_USER_AGENT.format(app="gdrive")}
            command = nul_args("gdrive", "download-file", "--file-id=F_BOOK")
            got = client.post("/v1/cli", headers=as_drive, content=command)
        else:
            got = client.post("/v1/gdrive/download_file", headers=auth, json={"file_id": "F_BOOK"})
        assert got.status_code == 502, got.text
        assert got.json()["detail"] == "Could not save /workspace/drive/F_BOOK/ledger.xlsx in the workspace"
        verified = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert verified["mask_sample"] is True and verified["failure_kind"] == "provider_unavailable"


@pytest.mark.parametrize("exposure", ["cli", "api_search"])
def test_verification_waits_for_a_command_or_an_endpoint_saving_its_download(
    sandboxes, monkeypatch, exposure: str
) -> None:
    server = make_server()
    run = FakeSandbox.exec
    stopping: list[asyncio.Task] = []

    async def exec_as_verification_begins(sandbox: FakeSandbox, command: str, **options: object) -> SimpleNamespace:
        if command.startswith("mkdir") and not stopping:  # /verify comes in as the call saves its download
            stopping.append(asyncio.create_task(server._stop_before_grading("rs-1")))
            for _ in range(5):
                await asyncio.sleep(0)
        return await run(sandbox, command, **options)

    with mcp_client(server) as client:
        seeded, _, _, call = seed_in(client, sandboxes, content_spec(), exposure)
        sandbox = sandboxes.started[-1]
        held = []  # what the sandbox held as it stopped
        sandboxes.on_stop = lambda: held.append(set(sandbox.files))
        monkeypatch.setattr(FakeSandbox, "exec", exec_as_verification_begins)
        book = call("gdrive", "download_file", file_id="F_BOOK")
        assert book["savedTo"] == "/workspace/drive/F_BOOK/ledger.xlsx", book
        verified = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies)  # after the stop under way
        assert verified.status_code == 200 and not verified.json()["mask_sample"], verified.text
        assert held == [{"/workspace/drive/F_BOOK/ledger.xlsx"}]  # the file saved, then the sandbox stopped, once


def test_an_endpoint_that_fails_answers_without_its_exception(sandboxes, monkeypatch) -> None:
    server = make_server()
    with mcp_client(server) as client:
        _, auth, _, _ = seed_in(client, sandboxes, content_spec(), "api_search")

        def broken(*args: object, **kwargs: object) -> None:
            raise RuntimeError("the store broke at /srv/internal")

        monkeypatch.setattr(server.service.store, "call", broken)
        got = client.post("/v1/gdrive/list_files", headers=auth, json={})
        assert got.status_code == 500 and "broke" not in got.text and "RuntimeError" not in got.text, got.text


@pytest.mark.parametrize("status", [200, 301, 404])
def test_an_endpoint_answers_an_apps_error_under_the_status_the_app_gives_it(sandboxes, monkeypatch, status) -> None:
    """An app may answer an error with a status below 400, as its vendor's API does (Snipe-IT's 200 with
    {"status": "error"}, Gorgias's 301): the endpoint answers the call's own status, never one of this server's."""
    server = make_server()
    with mcp_client(server) as client:
        _, auth, _, _ = seed_in(client, sandboxes, content_spec(), "api_search")
        answer = Reply(status, '{"status":"error","messages":"Asset not found"}', is_error=True)
        monkeypatch.setattr(server.service, "endpoint_call", lambda *args, **kwargs: answer)
        got = client.post("/v1/gdrive/list_files", headers=auth, json={}, follow_redirects=False)
        assert got.status_code == status and got.json() == {"status": "error", "messages": "Asset not found"}


def test_a_cli_episode_whose_commands_cannot_be_linked_leaves_no_world_and_no_sandbox(sandboxes) -> None:
    """The app commands are linked as the sandbox starts (ln refuses a name the image already has there): a failure
    fails the seed as a failed start does, and the seed can be retried."""
    sandboxes.refusing = (f"chmod 755 {APP_COMMAND_PATH}",)
    server = make_server()
    row = seed_body(in_exposure(content_spec(), "cli"))
    with mcp_client(server) as client:
        with pytest.raises(RuntimeError, match="Sandbox setup failed"):
            client.post("/seed_session", json=row)
        assert len(server.service.store) == 0 and not server._episodes and not server._tokens
        assert len(sandboxes.stopped) == 1
        sandboxes.refusing = ()
        assert client.post("/seed_session", json=row).status_code == 200


def test_the_guide_says_how_each_exposure_reaches_the_apps() -> None:
    apps = ("gdrive", "gmail", "user")
    drive = FileApp("gdrive", "Google Drive", "drive", "parent optional")  # as EpisodeService.file_apps lists it
    tools, cli, endpoints = (
        workspace_guide(8192, apps, exposure=exposure, prefix="company_", file_apps=[drive])
        for exposure in SANDBOX_EXPOSURES
    )
    assert "are your tools, one per operation of an app\n(company_<app>_<operation>)" in tools
    assert "Downloading a file with the apps' tools saves it" in tools
    assert "commands here, one per connected app: gdrive, gmail, user." in cli
    assert "Downloading a file with the apps' commands saves it" in cli
    assert 'curl -sS -X POST "$COMPANY_API_URL/v1/<app>/<operation>"' in endpoints
    assert "your company_api_search\ntool" in endpoints and "with the apps' endpoints" in endpoints
    for guide in (tools, cli, endpoints):
        assert guide.startswith("# Company apps\n\n") and "$COMPANY_API_URL/v1/files" in guide
        assert f"a shell\nstarted without them gets them with `. {CREDENTIALS_PATH}`" in guide
        assert "<app> is gdrive (Google Drive)." in guide and "Google Drive: parent optional.\n" in guide
        assert "    drive/<file id>/<name>   Google Drive" in guide


# The caps on the files a call reads: the stricter of the call's own (its ReadPlan) and the server's.


def test_the_host_reads_a_calls_files_within_the_stricter_of_its_caps_and_the_calls(sandboxes, monkeypatch) -> None:
    """A call whose own caps are below the host's has its files refused by them before anything is copied out: more
    files than it takes, a file over its cap, or more bytes together."""
    server = make_server()
    with mcp_client(server) as client:
        _, call, _ = seed_mail(client, sandboxes)
        planned = server.service.workspace_reads

        def capped(**caps: int) -> Callable[..., ReadPlan]:
            return lambda *args: ReadPlan(planned(*args).paths, **caps)

        attachments = [{"path": "out/totals.xlsx"}, {"path": "notes.csv"}]
        for caps in ({"max_files": 1}, {"max_file_bytes": len(BOOK) - 1}, {"max_total_bytes": len(BOOK)}):
            monkeypatch.setattr(server.service, "workspace_reads", capped(**caps))
            assert "error" in call("gmail_send_message", **MESSAGE, attachments=attachments), caps
        assert sandboxes.started[-1].downloaded == []  # none of them was copied out
        messages = server.service.store.get("rs-1").world.collection("gmail", "messages").rows.values()
        assert [row["subject"] for row in messages] == ["Ledgers"]  # nothing was sent


# Guarding a server that sandboxes can reach.


def test_in_sandbox_mode_callers_from_another_host_reach_only_mcp_and_the_file_api(sandboxes) -> None:
    server = make_server()
    with mcp_client(server) as local, mcp_client(server, client=SANDBOX) as remote:
        seeded = local.post("/seed_session", json=seed_body())
        assert seeded.status_code == 200, seeded.text
        for path, body in [
            ("/seed_session", seed_body(session_id="rs-2", rollout_id="r2")),
            ("/close_session", close_body()),
            ("/verify", VERIFY_BODY),
            ("/gdrive_list_files", {}),
        ]:
            refused = remote.post(path, json=body, cookies=seeded.cookies)
            assert refused.status_code == 404 and refused.json() == {"detail": "Not Found"}, (path, refused.text)
        assert remote.get("/officeenv/apps").status_code == 404
        for docs in ("/openapi.json", "/docs", "/redoc"):  # they would show a sandbox every route
            for answer in (local.get(docs), remote.get(docs)):  # 405: the tool route takes POST /<name>
                assert answer.status_code in (404, 405) and "paths" not in answer.text, (docs, answer.text)
        assert remote.get("/v1/files/gdrive/F_LEDGER", headers=file_auth(server)).status_code == 200
        grant = seeded.json()["resources_tools"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]
        called = rpc(remote, "tools/call", {"name": "gdrive_list_files", "arguments": {}}, grant).json()["result"]
        assert called.get("isError") is not True, called


def test_in_sandbox_mode_a_token_signed_with_the_servers_name_is_refused(sandboxes) -> None:
    server = make_server()
    with mcp_client(server) as local, mcp_client(server, client=SANDBOX) as remote:
        seeded = local.post("/seed_session", json=seed_body())
        grant = seeded.json()["resources_tools"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]
        claims = mcp_session_token_serializer(server.get_session_secret()).loads(grant)
        # What code in a sandbox could sign: the episode's session with every tool.
        forged = mcp_session_token_serializer(server.get_session_middleware_key()).dumps({**claims, "tools": None})
        call = {"name": "user_complete_task", "arguments": {"answer": "done"}}
        refused = rpc(remote, "tools/call", call, forged).json()["result"]
        assert refused["isError"] is True and "Invalid Gym MCP session token" in json.dumps(refused)


def test_sandbox_mode_signs_with_a_per_process_secret_and_names_its_cookie_neutrally(sandboxes) -> None:
    server = make_server()
    named_key = f"{type(server).__name__}___oe"
    assert server.get_session_middleware_key() == named_key  # appears in logs and error text
    assert server.get_session_secret() not in (named_key, make_server().get_session_secret())
    # a sandbox's calls get the cookie too, so its name says nothing of the environment, and stays unique
    cookie = server.get_session_cookie_name()
    assert cookie.startswith("sid_") and cookie != make_server(name="oe2").get_session_cookie_name()
    with mcp_client(server) as client:
        assert list(client.post("/seed_session", json=seed_body()).cookies) == [cookie]
    base = make_base_server()  # Gym's own name for the cookie, on the server the sandboxed one subclasses
    assert base.get_session_cookie_name() == base.get_session_middleware_key()


def test_sandbox_mode_needs_one_worker() -> None:
    with pytest.raises(ValueError, match="num_workers=1"):
        make_server(num_workers=2)


def test_trusted_callers_are_this_node_and_the_trusted_networks(monkeypatch) -> None:
    local = frozenset({"127.0.0.1", "::1", "10.0.0.1"})
    networks = (ipaddress.ip_network("10.109.0.0/16"),)
    for host in ("127.0.0.1", "::1", "::ffff:127.0.0.1", "10.0.0.1", "::ffff:10.0.0.1", "10.109.30.14"):
        assert is_trusted_caller(host, local=local, networks=networks), host
    for host in ("203.0.113.47", "10.0.0.2", "172.17.0.2", "testclient", "", None):
        assert not is_trusted_caller(host, local=local, networks=networks), host
    # The node's own addresses beyond those, when the host can tell them
    monkeypatch.setattr(sandbox_app, "is_own_address", lambda address: address == "10.0.0.2")
    assert is_trusted_caller("::ffff:10.0.0.2", local=local, networks=networks, own_addresses=True)
    assert not is_trusted_caller("10.0.0.2", local=local, networks=networks, own_addresses=False)
    assert not is_trusted_caller("172.17.0.2", local=local, networks=networks, own_addresses=True)


# The OfficeEnv server the sandboxed one subclasses.


def test_the_officeenv_server_it_subclasses_changes_nothing_for_any_caller(monkeypatch) -> None:
    def no_lookup() -> frozenset[str]:
        raise AssertionError("only the sandboxed server looks up this node's addresses")

    monkeypatch.setattr(sandbox_app, "node_addresses", no_lookup)
    server = make_base_server()
    assert server.get_session_secret() == server.get_session_middleware_key()  # Gym's default signing
    with mcp_client(server, client=SANDBOX) as remote:
        seeded = remote.post("/seed_session", json=seed_body())
        assert seeded.status_code == 200 and seeded.json()["sandbox_access"] is None, seeded.text
        assert remote.get("/officeenv/apps").status_code == 200 and remote.get("/openapi.json").status_code == 200
        assert remote.get("/v1/files/gdrive/F_LEDGER").status_code == 404  # no file API
        assert remote.post("/gdrive_list_files", json={}, cookies=seeded.cookies).status_code == 200
        assert remote.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).status_code == 200
        assert remote.post("/close_session", json=close_body(), cookies=seeded.cookies).status_code == 200


# Nothing a sandbox reads names the environment.

# What names the environment behind the company's API, or Gym: a sandboxed model reads none of it.
ENVIRONMENT_NAMES = re.compile(
    r"officeenv|office_env|\bgym\b|nemo|rollout|episode|simulat|seed|verif|surface|exposure|harness|dummy|token.capture|ng-trace",
    re.I,
)


# Gmail's own vocabulary, which holds one of those words without naming the environment: verifying a send-as address
# (its sendAs.verify operation, as a tool, a command and an endpoint, and the send-as addresses Gmail has verified)
VENDOR_WORDS = re.compile(
    r"verify[_-]send[_-]as|sendAs\.verify|send-as addresses that is verified|is verified at once"
    r"|whether each is verified"
)


def environment_names(text: str) -> set[str]:
    return {found.group(0).lower() for found in ENVIRONMENT_NAMES.finditer(VENDOR_WORDS.sub("", text))}


# The calls that move content, as sandbox_view makes them on each exposure: Drive's download saved, its pasted uploads
# refused (binary content, long text, a replacement) and an upload of a workspace file that is not there, and Gmail's
# attachments: one saved, a workspace file attached, and attachments refused (pasted, a filename alone, no such file);
# then a call to an app the episode does not connect.
CONTENT_CALLS = (
    ("gdrive", "download_file", {"file_id": "F_BOOK"}),
    ("gdrive", "create_file", {"name": "x.bin", "content_base64": "AAEC"}),
    ("gdrive", "create_file", {"name": "x.csv", "parents": ["F_FIN"], "content": LONG}),
    ("gdrive", "update_file", {"file_id": "F_LONG", "content": LONG}),
    ("gdrive", "create_file", {"name": "x.csv", "content_path": "/workspace/missing.csv"}),
    ("gmail", "get_attachment", {"message_id": "M_LEDGER", "attachment_id": "ledger.xlsx"}),
    ("gmail", "send_message", {**MESSAGE, "attachments": [{"path": "/workspace/out/totals.xlsx"}]}),
    ("gmail", "send_message", {**MESSAGE, "attachments": [{"filename": "x.xlsx", "content_base64": "AAEC"}]}),
    ("gmail", "send_message", {**MESSAGE, "attachments": ["x.xlsx"]}),
    ("gmail", "send_message", {**MESSAGE, "attachments": [{"path": "/workspace/missing.xlsx"}]}),
    ("slack", "post_message", {"channel": "C1", "text": "hi"}),
)


def harness_model_url(*, token_capture: bool) -> str:
    """The policy's URL as Gym's agents hand it to a sandboxed harness for one rollout (Pi writes it into the
    sandbox, in its models.json); traces off."""
    from nemo_gym.client import apply_rollout_prefix

    return (
        apply_rollout_prefix("http://10.0.0.1:9000", "17-0", token_capture=token_capture, trace_context=False) + "/v1"
    )


def sandbox_view(server: OfficeEnvResourcesServer, sandboxes: type[FakeSandbox], exposure: str = "tools") -> str:
    """Everything a sandboxed model can read of ``server`` in a ``exposure`` episode: its variables, the files and
    commands its sandbox gets, the MCP endpoint's name, header and tools, and the answers of every route a sandbox
    reaches, right and wrong (saved downloads, refused uploads, a call outside the episode's apps, a command made by
    hand, the routes of other exposures and those it may not use, the docs), with the cookies they set, for Drive and
    Gmail's attachments. The finding tools are described by this server, and Drive's and Gmail's tools by OfficeEnv in
    workspace mode, with this server's file API commands (in MCP, in a command's --help, in api_fetch's
    documentation); OfficeEnv's other descriptions are OfficeEnv's to check. Tokens are masked."""
    seen: list[str] = []
    with mcp_client(server) as local, mcp_client(server, client=SANDBOX) as remote:
        seeded = local.post("/seed_session", json=seed_body(in_exposure(with_mail(content_spec()), exposure)))
        assert seeded.status_code == 200, seeded.text
        (sandbox,) = sandboxes.started
        sandbox.files.update(MAIL_FILES)
        token = sandbox.spec.env["COMPANY_API_TOKEN"]
        mcp = seeded.json()["resources_tools"]
        header, grant = next(iter(mcp["headers"].items()))
        seen += [f"{name}={value}" for name, value in sandbox.spec.env.items()]
        seen += [f"{path}\n{text}" for path, text in sandbox.spec.files.items()]
        seen += [mcp["server_name"], mcp["url_path"], header]
        granted = {**RPC_HEADERS, header: grant}

        def read(answer) -> str:
            seen.extend(cookie.split("=", 1)[0] for cookie in answer.headers.get_list("set-cookie"))
            return answer.text

        def mcp_call(method: str, params: dict, headers: dict) -> str:
            body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
            return read(remote.post("/mcp", headers=headers, json=body))

        tools = json.loads(mcp_call("tools/list", {}, granted))["result"]["tools"]
        mail = {"gmail_get_attachment", "gmail_send_message", "gmail_create_draft", "gmail_update_draft"}
        described = {tool["name"] for tool in tools if tool["name"].startswith(("gdrive_", "api_"))} | mail
        seen += [json.dumps(tool) if tool["name"] in described else tool["name"] for tool in tools]
        auth = {"Authorization": f"Bearer {token}"}
        for app, op, arguments in CONTENT_CALLS:  # the exposure's way
            if exposure == "tools":
                seen.append(mcp_call("tools/call", {"name": f"{app}_{op}", "arguments": arguments}, granted))
            elif exposure == "cli":
                as_app = {**auth, "User-Agent": CLI_USER_AGENT.format(app=app)}
                for argv in (command_flags(arguments), ["--help"]):
                    command = nul_args(app, op.replace("_", "-"), *argv)
                    seen.append(read(remote.post("/v1/cli", headers=as_app, content=command)))
            else:
                seen.append(read(remote.post(f"/v1/{app}/{op}", headers=auth, json=arguments)))
                fetch = {"name": "api_fetch", "arguments": {"endpoint": f"{app}.{op}"}}
                seen.append(mcp_call("tools/call", fetch, granted))
        if exposure == "cli":  # each app's list of its commands
            for app in sorted(with_mail(content_spec())["apps"]):
                as_app = {**auth, "User-Agent": CLI_USER_AGENT.format(app=app)}
                seen.append(read(remote.post("/v1/cli", headers=as_app, content=nul_args(app, "--help"))))
        if exposure == "api_search":
            for name, arguments in [
                ("api_search", {"query": "list files"}),
                ("api_search", {"query": "download a file's content"}),
                ("api_search", {"query": "send mail from another address", "app": "gmail"}),
                ("api_fetch", {"endpoint": "gdrive.list_files"}),
            ]:
                seen.append(mcp_call("tools/call", {"name": name, "arguments": arguments}, granted))
        seen.append(mcp_call("tools/call", {"name": "slack_post_message", "arguments": {}}, granted))  # no grant
        seen.append(mcp_call("tools/list", {}, RPC_HEADERS))
        forged = {**RPC_HEADERS, "Authorization": "Bearer forged", NEMO_GYM_MCP_SESSION_TOKEN_HEADER: "forged"}
        seen.append(mcp_call("tools/call", {"name": "gdrive_list_files", "arguments": {}}, forged))
        for method, path, kwargs in [
            ("post", "/v1/cli", {"headers": auth, "content": nul_args("gdrive", "--help")}),  # by hand
            ("post", "/v1/cli", {"content": nul_args("gdrive", "--help")}),  # no token
            ("post", "/v1/gdrive/list_files", {"headers": auth, "json": {}}),
            ("post", "/v1/gdrive/list_files", {"headers": auth, "content": b"[1]"}),
            ("post", "/v1/gdrive/list_files", {"json": {}}),
            ("get", "/openapi.json", {}),
            ("get", "/docs", {}),
            ("get", "/redoc", {}),
            ("post", "/seed_session", {"json": seed_body(session_id="rs-2", rollout_id="r2")}),
            ("post", "/verify", {"json": VERIFY_BODY}),
            ("post", "/close_session", {"json": close_body()}),
            ("post", "/gdrive_list_files", {"json": {}}),
            ("get", "/officeenv/apps", {}),
            ("get", "/v1/files/gdrive/F_LEDGER", {"headers": auth}),
            ("get", "/v1/files/gdrive/F_LEDGER", {}),
            ("get", "/v1/files/gdrive/F_LEDGER", {"headers": {"Authorization": "Bearer forged"}}),
            ("get", "/v1/files/gdrive/F_NONE", {"headers": auth}),
            ("get", "/v1/files/slack/F1", {"headers": auth}),
            ("post", "/v1/files", {"headers": auth, "params": {"app": "gdrive", "name": "t.csv"}, "content": b"a,b"}),
            ("put", "/v1/files/gdrive/F_NONE", {"headers": auth, "content": b"a,b"}),
            ("get", "/v1/files/box/B_SHORT", {"headers": auth}),  # a file app the episode does not connect
        ]:
            answer = getattr(remote, method)(path, **kwargs)
            assert answer.status_code != 500, (path, answer.text)
            seen.append(read(answer))
        seen += sandbox.commands  # the relay, and the saved download's folder and mode
        seen += list(sandbox.files)
    return "\n".join(seen).replace(token, "<token>").replace(grant.removeprefix("Bearer "), "<token>")


@pytest.mark.parametrize("exposure", SANDBOX_EXPOSURES)
def test_a_sandbox_reads_environment_names_only_in_the_model_url(sandboxes, exposure: str) -> None:
    server = make_server(sandbox_config={**IMAGE, "host_relay": "host.lima.internal"}, **configured_mcp_settings())
    text = sandbox_view(server, sandboxes, exposure)
    assert "COMPANY_API_URL=http://10.0.0.1:8080" in text and "/workspace/README.md" in text
    assert "savedTo" in text and "$COMPANY_API_URL/v1/files" in text  # a saved download, a refused pasted upload
    # Drive's: a download saved in its folder, and its refused uploads' file API commands
    assert "/workspace/drive/F_BOOK/" in text and "files?app=gdrive&" in text and "/v1/files/gdrive/F_LONG" in text
    assert "No file at /workspace/missing.csv" in text  # an upload's workspace file that is not there
    # Gmail's attachments: one saved, and the refusals that name the workspace file to attach
    assert "/workspace/gmail/M_LEDGER/" in text and "No file at /workspace/missing.xlsx" in text
    assert "carries no content" in text and "attach it by its path" in text
    assert RELAY_PATH in text and server.get_session_cookie_name() in text  # the relay, a cookie an answer sets
    if exposure == "cli":  # the apps' command and the links that make each app one
        assert APP_COMMAND_PATH in text and "/usr/local/bin/gmail" in text
    if exposure == "api_search":  # an endpoint of Gmail's own send-as vocabulary, found and let pass by name
        assert "/v1/gmail/verify_send_as" in text
    assert environment_names(text) == set()  # nothing of this server's
    # What is left, from Gym's agents: the policy's URL carries Gym's own path names, the rollout prefix from rollout
    # correlation alone, the token-capture segment with token capture (and /ng-trace/<traceparent> while traces
    # export). Making them configurable is a follow-up.
    assert harness_model_url(token_capture=False) == "http://10.0.0.1:9000/ng-rollout/17-0/v1"
    assert environment_names(harness_model_url(token_capture=False)) == {"rollout"}
    assert environment_names(harness_model_url(token_capture=True)) == {"rollout", "token-capture"}


def test_without_the_company_mcp_settings_the_sandbox_reads_gym(sandboxes) -> None:
    """The check above can fail: Gym's MCP header is what the config's settings replace."""
    assert {"nemo", "gym"} <= environment_names(sandbox_view(make_server(), sandboxes))


def test_the_opensandbox_probe_prints_a_plain_word() -> None:
    """OpenSandbox's exec daemon keeps each command's output in the sandbox, and the provider's default post-create
    probe prints a marker that names Gym: the OpenSandbox config's probe, merged into the provider's settings, prints
    a plain word."""
    from omegaconf import OmegaConf

    from nemo_gym.sandbox import OpenSandboxProvider
    from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle
    from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProbeConfig

    merged = OmegaConf.merge(
        OmegaConf.load(Path(__file__).parents[3] / "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml"),
        OmegaConf.load(SANDBOX_CONFIG.with_name("officeenv_opensandbox.yaml")),
    )
    probe = OmegaConf.to_container(merged["sandbox"]["opensandbox"]["probe"])
    assert probe["stable_count"] == 2  # the provider's own probe settings stay
    assert ENVIRONMENT_NAMES.search(OpenSandboxProbeConfig().command)  # the provider's default marker
    provider = OpenSandboxProvider(probe=probe)
    execute = AsyncMock(return_value=SandboxExecResult("ready", "", 0))
    provider._exec = execute
    asyncio.run(provider._verify_created_handle(SandboxHandle("box", "opensandbox", None)))
    assert {call.args[1] for call in execute.call_args_list} == {"printf ready"}


def test_the_rows_name_the_company_api_as_the_sandbox_does() -> None:
    for line in SANDBOX_ROWS.read_text().splitlines():
        row = json.loads(line)
        (system,) = [item["content"] for item in row["responses_create_params"]["input"] if item["role"] == "system"]
        assert "/workspace/README.md" in system and "COMPANY_API_URL" in system and "COMPANY_API_TOKEN" in system
        prefix = configured_mcp_settings()["mcp_server_name"]
        assert f"{prefix}_user_complete_task" in system  # the tool as Pi lists it
        assert not ENVIRONMENT_NAMES.search(system), system
        # every file app's content goes through the file API, and mail attaches workspace files by their path
        assert "between the workspace and the apps with the file API" in system and "Google Drive with" not in system
        assert "attach workspace files to email by their path" in system


def load_prepare_rows():
    """sandbox/prepare_rows.py as a module."""
    import importlib.util

    path = Path(sandbox_app.__file__).parent / "sandbox" / "prepare_rows.py"
    spec = importlib.util.spec_from_file_location("prepare_rows", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_row_prompts_name_the_company_api_and_how_each_exposure_reaches_the_apps() -> None:
    prepare_rows = load_prepare_rows()
    assert (
        prepare_rows.system_prompt()
        == prepare_rows.SYSTEM_PROMPT
        == json.loads(SANDBOX_ROWS.read_text().splitlines()[0])["responses_create_params"]["input"][0]["content"]
    )  # the committed rows' prompt
    for exposure in SANDBOX_EXPOSURES:
        prompt = prepare_rows.system_prompt(exposure)
        assert "/workspace/README.md" in prompt and "COMPANY_API_URL" in prompt and "COMPANY_API_TOKEN" in prompt
        assert "in a shell without them, run `. /etc/company-api.env` first" in prompt  # Codex's shell, for one
        assert "attach workspace files to email by their path" in prompt
        assert not ENVIRONMENT_NAMES.search(prompt), prompt
    assert "run `user complete-task --answer=" in prepare_rows.system_prompt("cli")
    assert "your mcp__company__api_search tool" in prepare_rows.system_prompt("api_search", "mcp__company__")
    assert "your company_api_fetch tool" in prepare_rows.system_prompt("api_search")  # the tool's default name


@NEEDS_DOCS_TOOL
def test_an_api_search_rows_prompt_names_the_documentation_tool_as_its_task_does(tmp_path) -> None:
    """A task whose exposure names its documentation tool api_docs (docs_tool) gets a prompt that names it so."""
    prepare_rows = load_prepare_rows()
    spec = prepare_rows.rows("api_search")[0]["officeenv"]
    spec["exposure"]["options"] = {"docs_tool": "api_docs"}
    (tmp_path / "api_docs.json").write_text(json.dumps({"prompt": "Do the task.", "spec": spec}), encoding="utf-8")
    (row,) = prepare_rows.rows("api_search", folder=tmp_path)
    system_message = row["responses_create_params"]["input"][0]["content"]
    assert "read its parameters with your company_api_docs tool" in system_message
    assert "api_fetch" not in system_message and "api_call" not in system_message
    assert row["officeenv"]["exposure"]["options"] == {"docs_tool": "api_docs"}


@pytest.mark.parametrize("exposure", ["cli", "api_search"])
def test_prepared_rows_seed_as_sandboxed_episodes_on_their_exposure(sandboxes, exposure: str) -> None:
    """prepare_rows.py --exposure: every sandbox task, its prompt saying how its apps are reached, seeds as an
    episode of that exposure on the pinned OfficeEnv, with the grant and the guide of that exposure."""
    rows = load_prepare_rows().rows(exposure)
    assert {row["officeenv"]["exposure"]["mode"] for row in rows} == {exposure}
    server = make_server()
    with mcp_client(server) as client:
        for index, row in enumerate(rows):
            seed = ResourcesSeedSessionRequest(
                resources_session_id=f"rs-{index}",
                episode_id=EpisodeId(rollout_id=f"r{index}"),
                task_id=TaskId(taskset="oe", task_id=row["task_id"]),
                task_data={key: value for key, value in row.items() if key != "responses_create_params"},
            )
            seeded = client.post("/seed_session", json=seed.model_dump(mode="json"))
            assert seeded.status_code == 200, (row["task_id"], seeded.text)
            grant = seeded.json()["resources_tools"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]
            tools = {tool["name"] for tool in rpc(client, "tools/list", {}, grant).json()["result"]["tools"]}
            assert tools == ({"api_search", "api_fetch"} if exposure == "api_search" else set()), row["task_id"]
            episode = server._episodes[f"rs-{index}"]
            guide = sandboxes.started[-1].spec.files["/workspace/README.md"]
            file_apps = server.service.file_apps(f"rs-{index}") or server.service.file_apps()
            assert guide == workspace_guide(8192, episode.apps, exposure=exposure, file_apps=file_apps)


def test_a_task_that_needs_document_tools_keeps_that_in_its_row(sandboxes, tmp_path) -> None:
    """prepare_rows.py copies an example's sandbox_needs to its row, whose episode gets an image that has them."""
    prepare_rows = load_prepare_rows()
    (row,) = prepare_rows.rows()[:1]
    task = {"prompt": "Write the memo.", "spec": row["officeenv"], "sandbox_needs": ["documents"]}
    (tmp_path / "memo.json").write_text(json.dumps(task), encoding="utf-8")
    (needing_documents,) = prepare_rows.rows(folder=tmp_path)
    assert needing_documents["sandbox_needs"] == ["documents"]
    assert not any("sandbox_needs" in each for each in prepare_rows.rows())  # OfficeEnv's examples need no more
    seed = ResourcesSeedSessionRequest(
        resources_session_id="rs-1",
        episode_id=EpisodeId(rollout_id="r1"),
        task_id=TaskId(taskset="oe", task_id=row["task_id"]),
        task_data={key: value for key, value in needing_documents.items() if key != "responses_create_params"},
    ).model_dump(mode="json")
    with mcp_client(make_server()) as client:
        refused = client.post("/seed_session", json=seed)
        assert refused.status_code == 400 and refused.json()["detail"]["error"]["code"] == "unsupported_needs"
    with mcp_client(make_server(sandbox_config={**IMAGE, "images": [DOCUMENTS_IMAGE]})) as client:
        assert client.post("/seed_session", json=seed).status_code == 200
    assert [sandbox.spec.image for sandbox in sandboxes.started] == ["officeenv-sandbox:documents"]


def test_each_harness_config_binds_its_harness_to_the_shared_episodes() -> None:
    """Pi on MCP tools, Codex, mini-SWE, OpenCode, OpenClaw and Terminus 2 on commands: each config composes the same
    episodes (configs/officeenv_sandbox_episodes.yaml) with its harness, and no other harness."""
    configs = SANDBOX_CONFIG.parent
    for name, agent, transports in [
        ("officeenv_sandbox.yaml", "pi_agent", ["mcp"]),
        ("officeenv_sandbox_codex.yaml", "codex_agent", []),
        ("officeenv_sandbox_miniswe.yaml", "miniswe_sandboxed_agent", []),
        ("officeenv_sandbox_opencode.yaml", "opencode_agent", []),
        ("officeenv_sandbox_openclaw.yaml", "openclaw_agent", []),
        ("officeenv_sandbox_terminus_2.yaml", "terminus_2_sandboxed_agent", []),
    ]:
        config = yaml.safe_load((configs / name).read_text())
        assert config["config_paths"] == [
            "resources_servers/officeenv_sandbox/configs/officeenv_sandbox_episodes.yaml",
            f"responses_api_agents/{agent}/configs/{agent}.yaml",
            "environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml",
        ], name
        binding = config["single_agent_turn_legacy"]["environment_servers"]["single_agent_turn_legacy"]
        assert binding["agent_server"]["name"] == agent and binding["resources_server"]["name"] == "oe", name
        assert binding["resources_tool_transports"] == transports, name
        assert set(config) - {"config_paths", "oe", "single_agent_turn_legacy"} == {agent}, name


# The committed rows.


def test_the_sandbox_rows_seed_on_the_pinned_officeenv() -> None:
    """The rows sandbox/prepare_rows.py wrote seed as an Environment Server sends them, each granting a tool that
    reads a stored file: Drive's, Box's, SharePoint's, OneDrive's or a Gmail attachment's. The workbook task seeds a
    binary Drive file, which OfficeEnv releases up to 1.7.0 refuse."""
    rows = [json.loads(line) for line in SANDBOX_ROWS.read_text().splitlines()]
    downloads = {
        "gdrive_download_file",
        "box_get_file_content",
        "sharepoint_get_item_content",
        "onedrive_get_content",
        "gmail_get_attachment",
    }
    assert {row["task_id"] for row in rows} >= {"sandbox_vendor_spend", "sandbox_vendor_spend_xlsx"}
    with mcp_client(make_base_server()) as client:
        for index, row in enumerate(rows):
            assert row["officeenv_version"] == officeenv_version, "run sandbox/prepare_rows.py again"
            assert "tools" not in row["responses_create_params"]  # the harness takes its tools from the grant
            seed = ResourcesSeedSessionRequest(
                resources_session_id=f"rs-{index}",
                episode_id=EpisodeId(rollout_id=f"r{index}"),
                task_id=TaskId(taskset="oe", task_id=row["task_id"]),
                task_data={key: value for key, value in row.items() if key != "responses_create_params"},
            )
            seeded = client.post("/seed_session", json=seed.model_dump(mode="json"))
            assert seeded.status_code == 200, (row["task_id"], seeded.text)
            grant = seeded.json()["resources_tools"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]
            tools = {tool["name"] for tool in rpc(client, "tools/list", {}, grant).json()["result"]["tools"]}
            assert "user_complete_task" in tools and tools & downloads, row["task_id"]


# Lending through Gym's LentSandboxes, and the guards, as the review asked.


def gym_call() -> SimpleNamespace:
    """A request from Gym's servers on this node, for calling a route's handler directly."""
    return SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"), scope={}, session={})


def test_the_harness_gets_a_descriptor_that_lets_it_operate_the_sandbox_but_not_destroy_it(sandboxes) -> None:
    with mcp_client(make_server()) as client:
        assert client.post("/seed_session", json=seed_body()).status_code == 200
    assert sandboxes.started[0].scopes == ["operate"]


def test_a_close_waits_for_a_seed_still_starting_its_sandbox_and_stops_it(sandboxes, monkeypatch) -> None:
    """The environment server's cleanup close can come while a cancelled seed's start goes on: the close waits for
    the seed's lock, so the sandbox the seed then lends is stopped, not left with an episode nothing tracks."""
    server = make_server()
    start = FakeSandbox.start

    async def scenario() -> None:
        starting, go = asyncio.Event(), asyncio.Event()

        async def slow_start(sandbox: FakeSandbox, spec) -> None:
            starting.set()
            await go.wait()
            await start(sandbox, spec)

        monkeypatch.setattr(FakeSandbox, "start", slow_start)
        seed = asyncio.create_task(server._seed_sandboxed_episode(SimpleNamespace(session={}), seed_request()))
        await starting.wait()
        close = asyncio.create_task(
            server.close_resources_session(gym_call(), ResourcesCloseSessionRequest.model_validate(close_body()))
        )
        await asyncio.sleep(0)
        go.set()
        await asyncio.gather(seed, close)

    asyncio.run(scenario())
    assert len(sandboxes.started) == 1 and sandboxes.stopped == sandboxes.started
    assert not server._episodes and not server._tokens


def test_shutdown_during_a_seeds_turn_starts_no_sandbox(sandboxes, monkeypatch) -> None:
    """A seed that has not put its episode in yet, as while it waits SEED_TURN_YIELD_S in its turn, finds shutdown begun
    when its turn comes, and starts no sandbox (503): shutdown could not have stopped one it started."""
    monkeypatch.setattr(sandbox_app, "SEED_TURN_YIELD_S", 0.2)
    server = make_server()

    async def scenario() -> None:
        seed = asyncio.create_task(
            server._seed_sandboxed_episode(SimpleNamespace(session={}), seed_request(session_id="rs-2"))
        )
        await asyncio.sleep(0.05)  # the seed waits in its turn
        assert not seed.done() and "rs-2" not in server._episodes
        await server.shutdown()
        with pytest.raises(HTTPException) as refused:
            await seed
        assert refused.value.status_code == 503

    asyncio.run(scenario())
    assert sandboxes.started == [] or set(sandboxes.stopped) == set(sandboxes.started)
    assert "rs-2" not in server._episodes


def test_a_retired_attempts_replacement_seeds_the_same_session_id(sandboxes) -> None:
    """Gym never closes a retired session: retiring stops its sandbox and forgets its episode, so the replacement
    attempt seeds the same ID with its own episode and gets a sandbox of its own."""
    server = make_server()
    with mcp_client(server) as client:
        assert client.post("/seed_session", json=seed_body()).status_code == 200
        asyncio.run(server.retire_session_state("rs-1"))
        assert "rs-1" not in server._episodes and sandboxes.stopped == sandboxes.started
        replacement = seed_request().model_copy(update={"episode_id": EpisodeId(rollout_id="r1", attempt=1)})
        reseeded = client.post("/seed_session", json=replacement.model_dump(mode="json"))
        assert reseeded.status_code == 200, reseeded.text
        assert len(sandboxes.started) == 2


def test_a_failed_seeds_sandbox_whose_stop_failed_is_stopped_at_its_sessions_close(sandboxes, monkeypatch) -> None:
    """A sandbox may run on, billed, after its seed failed and its stop failed too: the session's close tries the stop
    again (Gym's LentSandboxes keeps it for that), rather than shutdown, which may come days later."""
    sandboxes.fail_on = "serialize"
    stop = FakeSandbox.stop
    refused: list[FakeSandbox] = []

    async def stop_refusing_once(sandbox: FakeSandbox) -> None:
        if not refused:
            refused.append(sandbox)
            raise RuntimeError("the provider could not stop the sandbox")
        await stop(sandbox)

    monkeypatch.setattr(FakeSandbox, "stop", stop_refusing_once)
    server = make_server()
    with mcp_client(server) as client:
        with pytest.raises(RuntimeError, match="the provider cannot describe the sandbox"):
            client.post("/seed_session", json=seed_body())
        assert sandboxes.stopped == []
        assert client.post("/close_session", json=close_body()).status_code == 200
        assert sandboxes.stopped == sandboxes.started == refused


def test_gyms_servers_on_this_node_are_trusted_whatever_the_hostname_resolves_to(sandboxes, monkeypatch) -> None:
    """On a node whose hostname resolves to 127.0.1.1 (Debian, Ubuntu), Gym's servers bound to another of its addresses
    (+default_host, a Docker bridge's say) reach this one from the node's primary address: any address of the node's
    own is this node. A sandbox's is not."""
    monkeypatch.setattr(sandbox_app, "node_addresses", lambda: frozenset({"127.0.0.1", "::1", "127.0.1.1"}))
    monkeypatch.setattr(sandbox_app, "is_own_address", lambda address: address == "10.136.223.148")
    server = make_server()
    app = server.setup_webserver()
    gym = TestClient(app, base_url="http://172.17.0.1:8080", client=("10.136.223.148", 50000))
    sandbox = TestClient(app, base_url="http://172.17.0.1:8080", client=("172.17.0.2", 50000))
    assert gym.get("/officeenv/apps").status_code == 200
    assert sandbox.get("/officeenv/apps").status_code == 404


def test_an_own_address_is_one_a_socket_here_binds() -> None:
    assert sandbox_app.binds_only_own_addresses()
    assert sandbox_app.is_own_address("127.0.0.1")
    for other in ("192.0.2.1", "203.0.113.47", "testclient", ""):  # TEST-NET addresses, and no address
        assert not sandbox_app.is_own_address(other), other


def test_every_route_of_gyms_answers_a_sandbox_as_missing_before_reading_its_body(sandboxes) -> None:
    """Metrics and the reverify mode are Gym's routes too, and a body the guarded routes would refuse (422) names
    Gym's request schema: a sandbox gets 404 before the body is read."""
    with mcp_client(make_server(), client=SANDBOX) as remote:
        for path in ("/seed_session", "/close_session", "/verify", "/aggregate_metrics"):
            refused = remote.post(path, json={"not": "a request"})
            assert refused.status_code == 404 and refused.json() == {"detail": "Not Found"}, (path, refused.text)
        assert remote.get("/reverify_mode").status_code == 404
    with mcp_client(make_server()) as local:
        assert local.get("/reverify_mode").status_code == 200
        assert local.post("/aggregate_metrics", json={"verify_responses": []}).status_code == 200
        assert local.post("/seed_session", json={"not": "a request"}).status_code == 422


def test_an_api_search_row_that_lists_no_documentation_tool_is_refused(sandboxes, monkeypatch) -> None:
    """OfficeEnv's api_search exposure lists its documentation tool beside api_search; a row that did not would leave
    the episode no way to read an endpoint's parameters, and is refused before a sandbox starts."""
    monkeypatch.setattr(sandbox_app, "finding_tools", lambda tools: [name for name in tools if name == "api_search"])
    server = make_server()
    with mcp_client(server) as client:
        refused = client.post("/seed_session", json=seed_body(in_exposure(SPEC, "api_search")))
        assert refused.status_code == 400 and "a documentation tool" in refused.text, refused.text
        assert sandboxes.started == [] and not server._episodes


def test_the_attachment_check_reads_each_paths_kind_and_size_with_any_posix_shell(tmp_path: Path) -> None:
    """STAT_SCRIPT runs in the sandbox's shell, which may be GNU's, BusyBox's, or macOS's on the local provider."""
    (tmp_path / "file").write_bytes(b"12345")
    (tmp_path / "empty").write_bytes(b"")
    (tmp_path / "link").symlink_to(tmp_path / "file")
    (tmp_path / "folder").mkdir()
    paths = [str(tmp_path / name) for name in ("file", "empty", "link", "folder", "missing")]

    done = subprocess.run(["sh", "-c", STAT_SCRIPT, "sh", *paths], capture_output=True, text=True, check=True)

    assert done.stdout.splitlines() == ["regular file|5", "regular file|0", "link|0", "other|0", "missing|0"]
