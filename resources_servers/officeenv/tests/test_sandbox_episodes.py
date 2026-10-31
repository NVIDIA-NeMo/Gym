# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox mode: each Environment Server episode borrows a sandbox, reaches OfficeEnv's tools over MCP and
moves file content over the episode's file API, and the server refuses callers that are not Gym's.
Sandboxes come from a fake provider, so these tests start no container. Environment Server episodes
without a sandbox and the MCP exposure itself are tested in test_app.py."""

import asyncio
import base64
import io
import ipaddress
import json
import os
import re
import shlex
import shutil
import socket
import stat
import subprocess
import threading
import time
import typing
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from fastapi import HTTPException
from fastapi.testclient import TestClient
from officeenv import __version__ as officeenv_version

from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import NEMO_GYM_MCP_SESSION_TOKEN_HEADER, ResourcesSeedSessionRequest
from nemo_gym.server_runtime.base_resources_server import mcp_session_token_serializer
from nemo_gym.server_runtime.mcp_auto_exposure import maybe_auto_expose
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.officeenv import app as officeenv_app
from resources_servers.officeenv.app import (
    APP_COMMAND_PATH,
    APP_COMMAND_SOURCE,
    ATTACHMENT_DOWNLOADS,
    ATTACHMENT_UPLOADS,
    CLI_USER_AGENT,
    CONTENT_DOWNLOADS,
    CONTENT_UPLOADS,
    CREDENTIALS_PATH,
    FILE_APPS,
    RELAY_PATH,
    RELAY_SOURCE,
    SANDBOX_SURFACES,
    SEARCH_TOOLS,
    STAT_SCRIPT,
    TOOL_SETS,
    OfficeEnvResourcesServer,
    OfficeEnvResourcesServerConfig,
    OfficeEnvSandboxConfig,
    ToolSet,
    _path_part,
    api_credentials,
    content_description,
    content_schema,
    egress_urls,
    example_attachments,
    is_trusted_caller,
    relay_ports,
    saved_download,
    tools_exposure_catalog,
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

    async def start(self, spec) -> None:
        self.spec = spec
        if FakeSandbox.fail_on == "start":
            raise RuntimeError("the provider could not create the sandbox")
        FakeSandbox.started.append(self)

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
        """What ``stat -c "%F|%s"`` prints of ``path``, without following a final link."""
        if path in FakeSandbox.kinds:
            return FakeSandbox.kinds[path]
        if path in FakeSandbox.links:
            return "symbolic link|1"
        if path in FakeSandbox.folders:
            return "directory|4096"
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

    async def serialize(self) -> dict:
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
    monkeypatch.setattr(officeenv_app, "AsyncSandbox", FakeSandbox)
    monkeypatch.setattr(officeenv_app, "get_global_config_dict", lambda: {"sandbox": {"docker": {}}})
    return FakeSandbox


def make_server(**config: object) -> OfficeEnvResourcesServer:
    """A server in sandbox mode unless ``sandbox_provider=None``."""
    settings = {
        "host": "10.0.0.1",
        "port": 8080,
        "entrypoint": "",
        "name": "oe",
        "expose_tools_over_mcp": True,
        "sandbox_provider": "sandbox",
        "sandbox_config": IMAGE,
        **config,
    }
    return OfficeEnvResourcesServer(
        config=OfficeEnvResourcesServerConfig(**settings), server_client=MagicMock(spec=ServerClient)
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
    settings = yaml.safe_load(episodes.read_text())["oe"]["resources_servers"]["officeenv"]
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
        assert sandbox.spec.metadata["officeenv-task"] == "sandbox-files"
        assert sandbox.spec.env["COMPANY_API_URL"] == "http://10.0.0.1:8080"  # this server's host and port
        assert sandbox.spec.env["COMPANY_API_TOKEN"] == server._episodes["rs-1"].token
        assert sandbox.spec.files == {
            "/workspace/README.md": workspace_guide(8192, ("gdrive", "user")),  # its apps
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
        # a surface a sandbox cannot be given, and tools the MCP endpoint does not serve under their names
        for exposure in ({"mode": "mcp"}, {"mode": "tools", "options": {"separator": "."}}):
            not_served = client.post("/seed_session", json=seed_body({**SPEC, "exposure": exposure}))
            assert not_served.status_code == 400 and "unsupported_exposure" in not_served.text, not_served.text
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


def test_sandbox_episodes_keep_the_resources_session_contract(sandboxes, monkeypatch) -> None:
    # The contract's client calls from "testclient", which is no address; the caller check is tested below.
    monkeypatch.setattr(officeenv_app, "is_trusted_caller", lambda host, **_: True)
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
        read_body = OfficeEnvResourcesServer._content_args

        async def body_then_verification(request):
            args = await read_body(request)
            server._episodes["rs-1"].grading = True  # as /verify sets it, while this body streamed in
            return args

        monkeypatch.setattr(OfficeEnvResourcesServer, "_content_args", staticmethod(body_then_verification))
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
    """A seed puts its episode in before the sandbox has started: shutdown, and retiring the session, wait for the
    seed's lock and stop that sandbox too. A seed that comes after shutdown starts none."""
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
        await asyncio.gather(seed, stopping)
        if stop == "shutdown":
            with pytest.raises(HTTPException) as refused:
                await server._seed_sandboxed_episode(SimpleNamespace(session={}), seed_request(session_id="rs-2"))
            assert refused.value.status_code == 503

    asyncio.run(scenario())
    assert len(sandboxes.started) == 1 and sandboxes.stopped == sandboxes.started


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
    monkeypatch.setattr(officeenv_app, "get_global_config_dict", lambda: named)
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
    monkeypatch.setattr(officeenv_app, "get_global_config_dict", lambda: OPENSANDBOX_NAMED)
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
    monkeypatch.setattr(officeenv_app, "get_global_config_dict", lambda: OPENSANDBOX_NAMED)
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
        monkeypatch.setattr(officeenv_app, "MAX_UPLOAD_BYTES", 8)
        too_big = client.post(
            "/v1/files", params={"app": "gdrive", "name": "x.bin"}, content=b"\xff" * 9, headers=auth
        )
        assert too_big.status_code == 413

        assert client.post("/close_session", json=close_body(), cookies=seeded.cookies).status_code == 200
        assert client.get("/v1/files/gdrive/F_LEDGER", headers=auth).status_code == 401  # revoked at close


# Drive file content stays out of a sandboxed episode's conversation (sandbox_config.inline_file_bytes).

LONG = "invoice_id,vendor,amount\n" + "".join(f"INV-{n},Vendor {n % 7},{n}.25\n" for n in range(600))
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
    with mcp_client(make_server()) as client:
        seeded, call, _ = seed_with_content(client)
        (sandbox,) = sandboxes.started

        book = call("gdrive_download_file", file_id="F_BOOK")
        assert "contentBase64" not in book and "content" not in book, book
        assert book["savedTo"] == "/workspace/drive/F_BOOK/ledger.xlsx" and book["savedBytes"] == len(BOOK)
        assert "binary content" in book["note"] and "preview" not in book
        assert sandbox.files["/workspace/drive/F_BOOK/ledger.xlsx"] == BOOK  # byte for byte
        assert "mkdir -p /workspace/drive/F_BOOK" in sandbox.commands
        # readable whoever the harness runs as, executable by no one, whatever mode the provider writes
        assert sandbox.modes[book["savedTo"]] == 0o644
        assert "chmod 644 /workspace/drive/F_BOOK/ledger.xlsx" in sandbox.commands

        short = call("gdrive_download_file", file_id="F_LEDGER")  # short text: inline as well
        assert short["content"] == LEDGER and short["savedTo"] == "/workspace/drive/F_LEDGER/ledger.csv"
        assert sandbox.files[short["savedTo"]] == LEDGER.encode()

        long = call("gdrive_download_file", file_id="F_LONG")  # a Drive name is no path
        assert "content" not in long and long["savedTo"] == "/workspace/drive/F_LONG/Q3_Q4 ledger.csv"
        assert sandbox.files[long["savedTo"]] == LONG.encode()
        assert LONG.startswith(long["preview"]) and long["preview"].endswith("\n") and len(long["preview"]) <= 2000

        revision = call("gdrive_get_revision", file_id="F_BOOK", revision_id="1")
        assert revision["savedTo"] == "/workspace/drive/F_BOOK/revisions/1/ledger.xlsx"
        assert sandbox.files[revision["savedTo"]] == BOOK
        exported = call("gdrive_export_file", file_id="F_DOC", mime_type="text/plain")
        assert exported["content"] == POLICY and exported["savedTo"] == "/workspace/drive/F_DOC/Travel policy.txt"
        assert sandbox.commands.count("mkdir -p /workspace/drive/F_BOOK") == 1  # a file's folder is made once

        result = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert result["sandbox_files_saved"] == 5.0 and result["sandbox_uploads_refused"] == 0.0
        assert result["sandbox_bytes_saved"] == float(2 * len(BOOK) + len(LEDGER) + len(LONG) + len(POLICY))


def test_an_upload_that_pastes_its_content_is_refused_with_the_file_api_command(sandboxes) -> None:
    server = make_server()
    with mcp_client(server) as client:
        seeded, call, _ = seed_with_content(client)

        pasted = call("gdrive_create_file", name="Q3 totals.xlsx", parents=["F_FIN"], content_base64=BOOK_B64)
        assert pasted["error"]["code"] == "upload_from_workspace", pasted
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


def test_tool_descriptions_and_the_guide_say_where_file_content_goes(sandboxes) -> None:
    catalog = tools_exposure_catalog()
    with mcp_client(make_server()) as client:
        _, _, described = seed_with_content(client)
        download = described["gdrive_download_file"]
        assert download.startswith("[Google Drive] Download a file's content (alt=media) into the workspace")
        assert "at most 8192 bytes" in download and "contentBase64" not in download
        create = described["gdrive_create_file"]
        assert "file API (/workspace/README.md)" in create and "content_base64" not in create
        assert described["gdrive_list_files"] == catalog["gdrive_list_files"]["description"]  # OfficeEnv's own
        guide = sandboxes.started[0].spec.files["/workspace/README.md"]
        assert (
            guide == workspace_guide(8192, ("gdrive", "user"))
            and "drive/<file id>/<name>" in guide
            and "at most 8192" in guide
        )
    assert "drive/<file id>" not in workspace_guide(None)


def test_the_uploads_content_parameters_send_a_binary_file_by_the_file_api(sandboxes) -> None:
    catalog = tools_exposure_catalog()
    for app, op in sorted(CONTENT_UPLOADS):
        schema = catalog[f"{app}_{op}"]["parameters"]
        properties = content_schema(app, op, schema, 8192)["properties"]
        assert properties["content_base64"]["description"] == (
            "Not taken here: text goes in content (at most 8192 bytes), and a binary file (an XLSX workbook, a PDF, an "
            "image) goes from the workspace with the file API (/workspace/README.md), never pasted here"
        ), (app, op)
        assert properties["content"]["description"].startswith("Text of at most 8192 bytes; a longer or binary file")
        assert content_schema(app, op, schema, None) == schema  # content through the conversation: OfficeEnv's
    with mcp_client(make_server()) as client:
        seeded = client.post("/seed_session", json=seed_body(content_spec()))
        token = seeded.json()["resources_tools"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]
        listed = {tool["name"]: tool for tool in rpc(client, "tools/list", {}, token).json()["result"]["tools"]}
    created = listed["gdrive_create_file"]["inputSchema"]["properties"]["content_base64"]["description"]
    assert "never pasted here" in created and "base64-encoded" not in created
    # as described: a call that pastes base64, even of short text, is refused with the file API command
    with mcp_client(make_server()) as client:
        seeded = client.post("/seed_session", json=seed_body(content_spec()))
        pasted = client.post(
            "/gdrive_create_file", json={"name": "a.txt", "content_base64": "aGVsbG8="}, cookies=seeded.cookies
        )
        assert "upload_from_workspace" in pasted.text and "curl -sf" in pasted.text, pasted.text


def test_with_no_limit_file_content_moves_through_the_conversation(sandboxes) -> None:
    catalog = tools_exposure_catalog()
    with mcp_client(make_server(sandbox_config={**IMAGE, "inline_file_bytes": None})) as client:
        seeded, call, described = seed_with_content(client)
        assert call("gdrive_download_file", file_id="F_BOOK")["contentBase64"] == BOOK_B64
        created = call("gdrive_create_file", name="totals.xlsx", parents=["F_FIN"], content_base64=BOOK_B64)
        assert created["name"] == "totals.xlsx" and sandboxes.started[0].files == {}
        assert "drive/<file id>" not in sandboxes.started[0].spec.files["/workspace/README.md"]
        assert described["gdrive_download_file"] == catalog["gdrive_download_file"]["description"]
        verified = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert verified["sandbox_files_saved"] == 0.0 and verified["sandbox_uploads_refused"] == 0.0


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
    """An episode without a sandbox, and a plain row's seed even in sandbox mode, answer and take file content in
    full, with OfficeEnv's descriptions."""
    catalog = tools_exposure_catalog()
    with mcp_client(make_server(sandbox_provider=None)) as client:
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


def test_workspace_paths_take_one_segment_per_name() -> None:
    assert _path_part("Q3/Q4 notes.txt", "F1") == "Q3_Q4 notes.txt"
    assert _path_part("..", "F1") == "F1" and _path_part(None, "F1") == "F1" and _path_part(" ", "F1") == "F1"
    long = "é" * 300 + ".xlsx"
    assert _path_part(long, "F1").endswith(".xlsx") and len(_path_part(long, "F1").encode()) <= 200
    answer = {"files": []}
    assert saved_download("gdrive", "list_files", {}, answer, folder="/w", limit=10, file_id="F1", name=None) is None


def test_an_export_to_an_office_format_is_saved_under_its_extension() -> None:
    office = "application/vnd.openxmlformats-officedocument"
    for mime_type, extension in [
        (f"{office}.wordprocessingml.document", ".docx"),
        (f"{office}.spreadsheetml.sheet", ".xlsx"),
        (f"{office}.presentationml.presentation", ".pptx"),
    ]:
        answer = {"mimeType": mime_type, "contentBase64": base64.b64encode(b"PK\x03\x04").decode()}
        params = {"file_id": "F_DOC", "mime_type": mime_type}
        saved = saved_download(
            "gdrive", "export_file", params, answer, folder="/w/drive", limit=8192, file_id="F_DOC", name="Policy"
        )
        assert saved is not None
        out, target, data = saved
        assert target == f"/w/drive/F_DOC/Policy{extension}" and data == b"PK\x03\x04" and "content" not in out


# Box, SharePoint and OneDrive: their file content stays out of the conversation as Drive's does.

SP_DOCS, OD_ME = "b!opsdocs", "b!alexdrive"
SHORT = "vendor,total\nAcme,10.50\n"


def file_apps_spec() -> dict:
    """content_spec() with Box, SharePoint and OneDrive connected: in a Finance folder of each, a long CSV
    (``<X>_LONG``), a short one (``<X>_SHORT``) and the workbook (``<X>_BOOK``); X is B, S or O."""
    spec = content_spec()

    def files(prefix: str, *, binary_key: str, **place: object) -> list[dict]:
        return [
            {"id": f"{prefix}_LONG", "name": "ledger long.csv", **place, "content": LONG},
            {"id": f"{prefix}_SHORT", "name": "totals.csv", **place, "content": SHORT},
            {"id": f"{prefix}_BOOK", "name": "ledger.xlsx", **place, binary_key: BOOK_B64},
        ]

    graph_user = {"id": "u-alex", "displayName": "Alex Morgan", "mail": ME}
    spec["apps"]["box"] = {
        "data": {
            "users": [{"id": "11000000001", "name": "Alex Morgan", "login": ME, "role": "user"}],
            "folders": [{"id": "B_FIN", "name": "Finance", "owned_by": ME}],
            "files": files("B", binary_key="content_base64", parent="B_FIN"),
        }
    }
    spec["apps"]["sharepoint"] = {
        "data": {
            "users": [graph_user],
            "sites": [{"id": "S_OPS", "name": "ops", "displayName": "Ops", "owners": [ME]}],
            "drives": [{"id": SP_DOCS, "siteId": "S_OPS", "name": "Documents", "default": True}],
            "driveItems": [
                {"id": "S_FIN", "driveId": SP_DOCS, "name": "Finance", "folder": {}},
                *files("S", binary_key="contentBase64", driveId=SP_DOCS, parentId="S_FIN"),
            ],
        }
    }
    spec["apps"]["onedrive"] = {
        "data": {
            "users": [graph_user],
            "drives": [{"id": OD_ME, "owner": ME}],
            "driveItems": [
                {"id": "O_FIN", "driveId": OD_ME, "name": "Finance", "folder": {}},
                *files("O", binary_key="contentBase64", driveId=OD_ME, parentId="O_FIN"),
            ],
        }
    }
    return spec


# Per app: how its tools download a file by id, and where the download is saved.
DOWNLOADS = {
    "box": ("box_get_file_content", lambda file_id: {"file_id": file_id}),
    "sharepoint": ("sharepoint_get_item_content", lambda file_id: {"drive_id": SP_DOCS, "item_id": file_id}),
    "onedrive": ("onedrive_get_content", lambda file_id: {"item_id": file_id}),
}
FILE_APPS_UNDER_TEST = tuple(DOWNLOADS)


def seed_file_apps(client: TestClient) -> tuple:
    seeded = client.post("/seed_session", json=seed_body(file_apps_spec()))
    assert seeded.status_code == 200, seeded.text
    token = seeded.json()["resources_tools"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]

    def call(tool: str, /, **arguments: object) -> dict:
        result = rpc(client, "tools/call", {"name": tool, "arguments": arguments}, token).json()["result"]
        return result.get("structuredContent") or json.loads(result["content"][0]["text"])

    listed = rpc(client, "tools/list", {}, token).json()["result"]["tools"]
    return seeded, call, {tool["name"]: tool["description"] for tool in listed}


@pytest.mark.parametrize("app", FILE_APPS_UNDER_TEST)
def test_a_file_apps_download_is_saved_in_the_workspace(sandboxes, app: str) -> None:
    prefix = app[0].upper()
    tool, args = DOWNLOADS[app]
    with mcp_client(make_server()) as client:
        seeded, call, _ = seed_file_apps(client)
        (sandbox,) = sandboxes.started
        long = call(tool, **args(f"{prefix}_LONG"))
        assert "content" not in long and long["savedTo"] == f"/workspace/{app}/{prefix}_LONG/ledger long.csv", long
        assert long["savedBytes"] == len(LONG) and LONG.startswith(long["preview"]) and len(long["preview"]) <= 2000
        assert sandbox.files[long["savedTo"]] == LONG.encode() and sandbox.modes[long["savedTo"]] == 0o644
        assert f"mkdir -p /workspace/{app}/{prefix}_LONG" in sandbox.commands
        assert any(command.startswith(f"chmod 644 '/workspace/{app}/{prefix}_LONG/") for command in sandbox.commands)
        short = call(tool, **args(f"{prefix}_SHORT"))  # short text: inline as well
        assert short["content"] == SHORT and sandbox.files[short["savedTo"]] == SHORT.encode()
        result = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert result["sandbox_files_saved"] == 2.0 and result["sandbox_bytes_saved"] == float(len(LONG) + len(SHORT))


@pytest.mark.parametrize("app", FILE_APPS_UNDER_TEST)
def test_a_file_apps_workbook_is_saved_byte_for_byte(sandboxes, app: str) -> None:
    prefix = app[0].upper()
    tool, args = DOWNLOADS[app]
    with mcp_client(make_server()) as client:
        _, call, _ = seed_file_apps(client)
        book = call(tool, **args(f"{prefix}_BOOK"))
        assert not {"content", "contentBase64", "content_base64"} & set(book), book
        assert book["savedTo"] == f"/workspace/{app}/{prefix}_BOOK/ledger.xlsx" and book["savedBytes"] == len(BOOK)
        assert "binary content" in book["note"] and "preview" not in book
        assert sandboxes.started[0].files[book["savedTo"]] == BOOK


def test_versions_are_saved_in_their_own_folder(sandboxes) -> None:
    server = make_server()
    with mcp_client(server) as client:
        _, call, _ = seed_file_apps(client)
        auth = file_auth(server)
        for app, file_id in (("box", "B_LONG"), ("onedrive", "O_LONG")):  # a second version, through the file API
            assert client.put(f"/v1/files/{app}/{file_id}", content=SHORT.encode(), headers=auth).status_code == 200
        files = sandboxes.started[0].files
        first = call("box_get_file_content", file_id="B_LONG", version_id="1")
        assert first["savedTo"] == "/workspace/box/B_LONG/versions/1/ledger long.csv"
        assert files[first["savedTo"]] == LONG.encode()
        assert call("box_get_file_content", file_id="B_LONG")["content"] == SHORT  # the current one, inline
        old = call("onedrive_get_version_content", item_id="O_LONG", version_id="1.0")
        assert old["savedTo"] == "/workspace/onedrive/O_LONG/versions/1.0/ledger long.csv"
        assert files[old["savedTo"]] == LONG.encode()
        by_path = call("onedrive_get_version_content", path="/Finance/ledger long.csv", version_id="1.0")
        assert by_path["savedTo"] == old["savedTo"]  # the item a path names


@pytest.mark.parametrize(
    ("tool", "arguments", "command"),
    [
        (
            "box_upload_file",
            {"name": "Q3 totals.csv", "parent_id": "B_FIN"},
            '-X POST {send} "$COMPANY_API_URL/v1/files?app=box&name=Q3%20totals.csv&parent=B_FIN"',
        ),
        ("box_upload_file_version", {"file_id": "B_LONG"}, '-X PUT {send} "$COMPANY_API_URL/v1/files/box/B_LONG"'),
        (
            "sharepoint_upload_content",
            {"drive_id": SP_DOCS, "parent_id": "S_FIN", "name": "Q3 totals.csv"},
            '-X POST {send} "$COMPANY_API_URL/v1/files?app=sharepoint&name=Q3%20totals.csv&parent=S_FIN"',
        ),
        (
            "sharepoint_upload_content",  # by path: the folder the path names
            {"drive_id": SP_DOCS, "path": "/Finance/Q3 totals.csv"},
            '-X POST {send} "$COMPANY_API_URL/v1/files?app=sharepoint&name=Q3%20totals.csv&parent=S_FIN"',
        ),
        (
            "sharepoint_upload_content",  # the library by its name within the site, as SharePoint takes it
            {"drive_id": "Documents", "site": "ops", "path": "/Finance/Q3 totals.csv"},
            '-X POST {send} "$COMPANY_API_URL/v1/files?app=sharepoint&name=Q3%20totals.csv&parent=S_FIN"',
        ),
        (
            "onedrive_upload_file",  # a folder that is not there: the command names the parameter to fill in
            {"path": "/Nowhere/Q3 totals.csv"},
            '-X POST {send} "$COMPANY_API_URL/v1/files?app=onedrive&name=Q3%20totals.csv&parent=<folder_id>"',
        ),
        (
            "onedrive_upload_file",
            {"path": "/Finance/Q3 totals.csv"},
            '-X POST {send} "$COMPANY_API_URL/v1/files?app=onedrive&name=Q3%20totals.csv&parent=O_FIN"',
        ),
        ("onedrive_upload_file", {"item_id": "O_LONG"}, '-X PUT {send} "$COMPANY_API_URL/v1/files/onedrive/O_LONG"'),
    ],
)
def test_a_file_apps_pasted_upload_is_refused_with_the_file_api_command(
    sandboxes, tool: str, arguments: dict, command: str
) -> None:
    send = '-H "Authorization: Bearer $COMPANY_API_TOKEN" --data-binary @<local path>'
    with mcp_client(make_server()) as client:
        seeded, call, _ = seed_file_apps(client)
        refused = call(tool, **arguments, content=LONG)
        message = json.dumps(refused)
        assert "upload_from_workspace" in message and "text over 8192 bytes" in message, refused
        assert "curl -sf " + command.format(send=send) in refused_message(refused)
        accepted = call(tool, **arguments, content=SHORT)  # short text still goes in the call
        assert "upload_from_workspace" not in json.dumps(accepted), accepted
        pasted = call(tool, **arguments, content_base64=BOOK_B64)
        assert "binary content" in refused_message(pasted)
        result = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert result["sandbox_uploads_refused"] == 2.0


def refused_message(answer: dict) -> str:
    """An app's error message, in Drive's, Box's or Graph's envelope, or a failed command's error text."""
    error = answer.get("error")
    if isinstance(error, str):
        return error
    return str((error if isinstance(error, dict) else answer).get("message") or "")


@pytest.mark.parametrize(
    ("app", "parent"),
    [("box", "B_FIN"), ("box", None), ("sharepoint", "S_FIN"), ("onedrive", "O_FIN"), ("onedrive", None)],
)
def test_the_file_api_moves_a_file_apps_files(sandboxes, app: str, parent: str | None) -> None:
    server = make_server()
    with mcp_client(server) as client:
        seed_file_apps(client)
        auth = file_auth(server)
        where = {"app": app, "name": "totals new.csv", **({"parent": parent} if parent else {})}
        sent = client.post("/v1/files", params=where, content=LONG.encode(), headers=auth)
        assert sent.status_code == 200, sent.text
        created = sent.json()
        file_id = (created.get("entries") or [created])[0]["id"]  # Box answers {total_count, entries}
        assert client.get(f"/v1/files/{app}/{file_id}", headers=auth).content == LONG.encode()
        replaced = client.put(f"/v1/files/{app}/{file_id}", content=SHORT.encode(), headers=auth)
        assert replaced.status_code == 200, replaced.text
        assert client.get(f"/v1/files/{app}/{file_id}", headers=auth).content == SHORT.encode()
        book = client.put(f"/v1/files/{app}/{file_id}", content=BOOK, headers=auth)
        assert book.status_code == 200, book.text
        assert client.get(f"/v1/files/{app}/{file_id}", headers=auth).content == BOOK  # byte for byte
        assert client.get(f"/v1/files/{app}/{app[0].upper()}_BOOK", headers=auth).content == BOOK
        assert sandboxes.started[0].files == {}  # the file API writes nothing into the sandbox
        (episode,) = server._episodes.values()
        assert episode.direct is False and episode.uploads_refused == 0


def test_a_file_api_call_a_file_app_refuses_answers_with_its_http_status(sandboxes) -> None:
    server = make_server()
    with mcp_client(server) as client:
        seed_file_apps(client)
        auth = file_auth(server)
        for app in FILE_APPS_UNDER_TEST:
            missing = client.get(f"/v1/files/{app}/X_NONE", headers=auth)
            assert missing.status_code == 404, (app, missing.text)
            assert client.put(f"/v1/files/{app}/X_NONE", content=b"a,b", headers=auth).status_code == 404, app
            into_nothing = client.post(
                "/v1/files", params={"app": app, "name": "a.csv", "parent": "X_NONE"}, content=b"a,b", headers=auth
            )
            assert into_nothing.status_code == 404, (app, into_nothing.text)
        taken = client.post(
            "/v1/files", params={"app": "box", "name": "totals.csv", "parent": "B_FIN"}, content=b"a,b", headers=auth
        )
        assert taken.status_code == 409 and "item_name_in_use" in taken.text  # Box's own status
        no_folder = client.post(
            "/v1/files", params={"app": "sharepoint", "name": "a.csv"}, content=b"a,b", headers=auth
        )
        assert no_folder.status_code == 400 and "parent" in no_folder.text  # SharePoint needs a folder


def test_file_apps_tools_and_the_guide_say_where_content_goes(sandboxes) -> None:
    catalog = tools_exposure_catalog()
    with mcp_client(make_server()) as client:
        _, _, described = seed_file_apps(client)
        assert described["box_get_file_content"].startswith(
            "[Box] Download a file's content (GET /files/{id}/content), optionally a given version, into the workspace"
        )
        assert described["sharepoint_get_item_content"].startswith("[SharePoint] A file's content (GET /drives/")
        assert "saved in the workspace" in described["onedrive_get_version_content"]
        for tool in (
            "box_upload_file",
            "box_upload_file_version",
            "sharepoint_upload_content",
            "onedrive_upload_file",
        ):
            assert "file API (/workspace/README.md)" in described[tool] and "content_base64" not in described[tool]
        for tool in ("box_get_file_content", "sharepoint_get_item_content", "onedrive_get_content"):
            assert "base64" not in described[tool].lower() and "at most 8192 bytes" in described[tool]
        assert described["onedrive_list_versions"] == catalog["onedrive_list_versions"]["description"]  # OfficeEnv's
        guide = sandboxes.started[0].spec.files["/workspace/README.md"]
        assert guide == workspace_guide(8192, ("box", "gdrive", "onedrive", "sharepoint", "user"))
        for folder in ("drive", "box", "sharepoint", "onedrive"):
            assert f"{folder}/<file id>/<name>" in guide
        assert "SharePoint: parent required" in guide and "$COMPANY_API_URL/v1/files/<app>/<file id>" in guide


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


def test_a_mail_attachment_is_saved_in_the_workspace(sandboxes) -> None:
    with mcp_client(make_server()) as client:
        seeded, call, _ = seed_mail(client, sandboxes, files={})
        (sandbox,) = sandboxes.started
        book = call("gmail_get_attachment", message_id="M_LEDGER", attachment_id="ledger.xlsx")
        folder = f"/workspace/gmail/M_LEDGER/{book['attachmentId']}"  # a message may hold two of one name
        assert book["savedTo"] == f"{folder}/ledger.xlsx" and book["savedBytes"] == len(BOOK), book
        assert not {"content", "contentBase64", "preview"} & set(book) and "binary content" in book["note"]
        assert sandbox.files[book["savedTo"]] == BOOK and sandbox.modes[book["savedTo"]] == 0o644
        long = call("gmail_get_attachment", message_id="M_LEDGER", attachment_id="ledger long.csv")
        assert "content" not in long and LONG.startswith(long["preview"])
        assert sandbox.files[long["savedTo"]] == LONG.encode()
        short = call("gmail_get_attachment", message_id="M_LEDGER", attachment_id="totals.csv")
        assert short["content"] == SHORT and sandbox.files[short["savedTo"]] == SHORT.encode()  # inline as well
        result = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert result["sandbox_files_saved"] == 3.0
        assert result["sandbox_bytes_saved"] == float(len(BOOK) + len(LONG) + len(SHORT))


def test_a_message_attaches_a_workspace_file_by_its_path(sandboxes) -> None:
    """The server reads the file from the sandbox before the call runs, and the message carries it byte for byte. A
    relative path is in the workspace, and the attachment is named after the file unless the call names it."""
    with mcp_client(make_server()) as client:
        seeded, call, _ = seed_mail(client, sandboxes)
        attachments = [{"path": "/workspace/out/totals.xlsx"}, {"path": "notes.csv", "filename": "Q3 notes.csv"}]
        sent = call("gmail_send_message", **MESSAGE, attachments=attachments)
        assert "error" not in sent, sent
        listed = call("gmail_get_message", id=sent["id"])["attachments"]
        assert [(a["filename"], a["size"]) for a in listed] == [
            ("totals.xlsx", len(BOOK)),
            ("Q3 notes.csv", len(LONG)),
        ]
        back = call("gmail_get_attachment", message_id=sent["id"], attachment_id="totals.xlsx")
        assert sandboxes.started[0].files[back["savedTo"]] == BOOK  # through the message and back, byte for byte
        # a draft takes them too, and update_draft replaces them
        draft = call("gmail_create_draft", **MESSAGE, attachments=[{"path": "notes.csv"}])
        assert "error" not in draft, draft
        assert "error" not in call("gmail_update_draft", id=draft["id"], attachments=[{"path": "out/totals.xlsx"}])
        final = call("gmail_send_draft", id=draft["id"])
        assert [a["filename"] for a in call("gmail_get_message", id=final["id"])["attachments"]] == ["totals.xlsx"]
        result = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert result["sandbox_files_attached"] == 4.0 and result["sandbox_uploads_refused"] == 0.0
        assert result["sandbox_bytes_attached"] == float(2 * len(BOOK) + 2 * len(LONG))


def test_pasted_or_unnamed_mail_attachments_are_refused_with_the_path_to_use(sandboxes) -> None:
    with mcp_client(make_server()) as client:
        seeded, call, _ = seed_mail(client, sandboxes)

        def send(*attachments: object) -> dict:
            return call("gmail_send_message", **MESSAGE, attachments=list(attachments))

        pasted = send({"filename": "totals.xlsx", "content_base64": BOOK_B64})
        assert "upload_from_workspace" in json.dumps(pasted) and "binary content" in refused_message(pasted), pasted
        assert '{"path": "/workspace/totals.xlsx"}' in refused_message(pasted)
        assert "text over 8192 bytes" in refused_message(send({"filename": "notes.csv", "content": LONG}))
        unnamed = send("totals.xlsx")  # a filename alone
        assert "attach_from_workspace" in json.dumps(unnamed), unnamed
        assert '{"path": "/workspace/totals.xlsx"}' in refused_message(unnamed)
        for path in ("/workspace/missing.xlsx", "/dev/zero", "../../proc/self/environ"):
            assert "there is no file at" in refused_message(send({"path": path})), path
        sandbox = sandboxes.started[-1]  # nothing under /dev or /proc is checked or copied, not merely found missing
        assert not any("/dev/zero" in command or "/proc/self" in command for command in sandbox.commands)
        assert not {"/dev/zero", "/proc/self/environ"} & set(sandbox.downloaded)
        assert "path or its content, not both" in refused_message(send({"path": "notes.csv", "content": "a,b"}))
        assert "error" not in send({"filename": "short.csv", "content": SHORT})  # short text still goes in the call
        result = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert result["sandbox_uploads_refused"] == 2.0 and result["sandbox_files_attached"] == 0.0


def test_a_bare_string_naming_a_workspace_file_attaches_it(sandboxes) -> None:
    """A model may write a file's path as a bare string (``--attachments=/workspace/out/totals.xlsx``). A string that
    writes a path, or names a file the workspace holds, attaches that file."""
    with mcp_client(make_server()) as client:
        seeded, call, _ = seed_mail(client, sandboxes)
        sent = call("gmail_send_message", **MESSAGE, attachments="/workspace/out/totals.xlsx")
        assert "error" not in sent, sent
        assert [(a["filename"], a["size"]) for a in call("gmail_get_message", id=sent["id"])["attachments"]] == [
            ("totals.xlsx", len(BOOK))
        ]
        listed = call("gmail_send_message", **MESSAGE, attachments=["out/totals.xlsx", "notes.csv"])
        assert "error" not in listed, listed
        assert [a["filename"] for a in call("gmail_get_message", id=listed["id"])["attachments"]] == [
            "totals.xlsx",
            "notes.csv",
        ]
        result = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert result["sandbox_files_attached"] == 3.0 and result["sandbox_uploads_refused"] == 0.0
        assert result["sandbox_bytes_attached"] == float(2 * len(BOOK) + len(LONG))


def test_a_refusal_names_the_workspace_path_once(sandboxes) -> None:
    """A name already under the workspace is suggested as itself, never as ``/workspace//workspace/...``."""
    with mcp_client(make_server()) as client:
        seeded, call, _ = seed_mail(client, sandboxes)

        def send(attachments: object) -> dict:
            return call("gmail_send_message", **MESSAGE, attachments=attachments)

        missing = send("/workspace/aging_2026-10.xlsx")  # a path the workspace does not hold
        assert "there is no file at /workspace/aging_2026-10.xlsx" in refused_message(missing), missing
        for named in ({"filename": "/workspace/aging_2026-10.xlsx"}, {"name": "/workspace/aging_2026-10.xlsx"}):
            unnamed = send([named])
            assert "attach_from_workspace" in json.dumps(unnamed), unnamed
            assert '{"path": "/workspace/aging_2026-10.xlsx"}' in refused_message(unnamed)
            assert "//workspace" not in json.dumps(unnamed)
        pasted = send([{"filename": "/workspace/out/totals.xlsx", "content_base64": BOOK_B64}])
        assert '{"path": "/workspace/out/totals.xlsx"}' in refused_message(pasted)
        assert "//workspace" not in json.dumps(pasted)
        assert '{"path": "/workspace/totals.xlsx"}' in refused_message(send("totals.xlsx"))  # a name it does not hold
        result = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert result["sandbox_files_attached"] == 0.0


def test_an_attachment_is_read_only_from_a_regular_file_of_the_sandbox(sandboxes, tmp_path, monkeypatch) -> None:
    """``docker cp`` copies a link as a link, which the server would read on its own host: a link, like a folder, is
    no file to attach, and neither is one over the upload limit. Each is refused before anything is copied out."""
    secret = tmp_path / "host.txt"
    secret.write_text("a file of the host\n")
    sandboxes.links, sandboxes.folders = {"/workspace/link.txt": str(secret)}, {"/workspace/out"}
    server = make_server()
    with mcp_client(server) as client:
        _, call, _ = seed_mail(client, sandboxes)
        for path in ("/workspace/link.txt", "/workspace/out"):
            refused = call("gmail_send_message", **MESSAGE, attachments=[{"path": path}])
            assert f"there is no file at {path}" in refused_message(refused), refused
        monkeypatch.setattr(officeenv_app, "MAX_UPLOAD_BYTES", 8)
        too_big = call("gmail_send_message", **MESSAGE, attachments=[{"path": "notes.csv"}])
        assert "/workspace/notes.csv is over" in refused_message(too_big), too_big
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
    server = make_server()
    with mcp_client(server) as client:
        _, call, _ = seed_mail(client, sandboxes)
        sandbox = sandboxes.started[-1]

        def refusal(*attachments: object) -> str:
            return refused_message(call("gmail_send_message", **MESSAGE, attachments=list(attachments)))

        assert "there is no file at /workspace/d/zero" in refusal({"path": "d/zero"})  # /dev/zero through a link
        assert "/workspace/huge.bin is over 20 MB" in refusal({"path": "huge.bin"})
        many = [{"path": f"f{i}.csv"} for i in range(officeenv_app.MAX_ATTACHMENTS + 1)]
        assert f"at most {officeenv_app.MAX_ATTACHMENTS} workspace files" in refusal(*many)
        monkeypatch.setattr(officeenv_app, "MAX_ATTACHMENT_BYTES", len(BOOK) + len(LONG) - 1)
        assert "together are over" in refusal({"path": "out/totals.xlsx"}, {"path": "notes.csv"})
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
        assert unchecked.json()["detail"] == "Could not check the attachments in the workspace"
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
        assert "there is no file at /workspace/notes.csv" in refused_message(gone), gone
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


def test_mail_attachments_are_described_as_they_move_here(sandboxes) -> None:
    catalog = tools_exposure_catalog()
    with mcp_client(make_server()) as client:
        _, _, listed = seed_mail(client, sandboxes)
        for op in ("send_message", "create_draft", "update_draft"):
            text = listed[f"gmail_{op}"]["inputSchema"]["properties"]["attachments"]["description"]
            assert "{path, filename?, mimeType?} attaches a file from the workspace" in text, text
            assert "content_base64" not in text and "at most 8192 bytes" in text and "/workspace/totals.xlsx" in text
        got = listed["gmail_get_attachment"]["description"]
        assert "saved in the workspace" in got and "contentBase64" not in got and "at most 8192 bytes" in got
        assert listed["gmail_list_messages"]["description"] == catalog["gmail_list_messages"]["description"]
        guide = sandboxes.started[0].spec.files["/workspace/README.md"]
        assert guide == workspace_guide(8192, tuple(sorted(with_mail(content_spec())["apps"])))
        assert "gmail/<message id>/<attachment id>/<name>" in guide and "Getting one (gmail_get_attachment)" in guide
        assert '{"path": "/workspace/totals.xlsx"}' in guide
    assert "Mail attachments" not in workspace_guide(8192, ("gdrive",))  # an episode without Gmail
    assert "Mail attachments" not in workspace_guide(None, ("gmail",))  # content through the conversation


def test_without_the_content_limit_mail_attachments_move_as_before(sandboxes) -> None:
    with mcp_client(make_server(sandbox_config={**IMAGE, "inline_file_bytes": None})) as client:
        _, call, listed = seed_mail(client, sandboxes)
        got = call("gmail_get_attachment", message_id="M_LEDGER", attachment_id="ledger.xlsx")
        assert got["contentBase64"] == BOOK_B64 and "savedTo" not in got
        pasted = [{"filename": "t.xlsx", "content_base64": BOOK_B64}]
        assert "error" not in call("gmail_send_message", **MESSAGE, attachments=pasted)
        by_path = call("gmail_send_message", **MESSAGE, attachments=[{"path": "/workspace/out/totals.xlsx"}])
        assert "unknown fields" in refused_message(by_path)  # Gmail's own answer: nothing reads the workspace here
        schema = listed["gmail_send_message"]["inputSchema"]
        assert "content_base64" in schema["properties"]["attachments"]["description"]  # OfficeEnv's


# The cli and api_search surfaces: the apps as commands, or as endpoints found over MCP and called with curl.

GUIDE = "/workspace/README.md"
# Per file app: an upload into its Finance folder (file_apps_spec), as its op and arguments, content aside
UPLOADS = {
    "gdrive": ("create_file", {"name": "Q3 totals.csv", "parents": ["F_FIN"]}),
    "box": ("upload_file", {"name": "Q3 totals.csv", "parent_id": "B_FIN"}),
    "sharepoint": ("upload_content", {"drive_id": SP_DOCS, "parent_id": "S_FIN", "name": "Q3 totals.csv"}),
    "onedrive": ("upload_file", {"path": "/Finance/Q3 totals.csv"}),
}


def in_surface(spec: dict, surface: str) -> dict:
    """A copy of ``spec`` with its task in exposure ``surface``."""
    spec = json.loads(json.dumps(spec))
    spec["exposure"] = {"mode": surface}
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


def seed_in(client: TestClient, sandboxes: type[FakeSandbox], spec: dict, surface: str, files: dict | None = None):
    """Seed ``spec`` in ``surface``, with ``files`` in the workspace as if the model had written them: the seed's
    answer, the bearer header of the episode's API, an MCP caller (``tool(name, **arguments)``), and
    ``call(app, op, **arguments)``, which calls an op the surface's way (an MCP tool, the app's command, an endpoint)
    and reads its JSON answer, or a failed command's error text as ``{"error": <stderr>}``."""
    seeded = client.post("/seed_session", json=seed_body(in_surface(spec, surface)))
    assert seeded.status_code == 200, seeded.text
    sandbox = sandboxes.started[-1]
    sandbox.files.update(files or {})
    auth = {"Authorization": f"Bearer {sandbox.spec.env['COMPANY_API_TOKEN']}"}
    grant = seeded.json()["resources_tools"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]

    def tool(name: str, /, **arguments: object) -> dict:
        result = rpc(client, "tools/call", {"name": name, "arguments": arguments}, grant).json()["result"]
        return result.get("structuredContent") or json.loads(result["content"][0]["text"])

    def call(app: str, op: str, /, **arguments: object) -> dict:
        if surface == "tools":
            return tool(f"{app}_{op}", **arguments)
        if surface == "api_search":
            return client.post(f"/v1/{app}/{op}", headers=auth, json=arguments).json()
        command = nul_args(app, op.replace("_", "-"), *command_flags(arguments))
        ran = client.post("/v1/cli", headers={**auth, "User-Agent": CLI_USER_AGENT.format(app=app)}, content=command)
        return json.loads(ran.text) if ran.headers["x-exit-code"] == "0" else {"error": ran.text}

    return seeded, auth, tool, call


def test_a_cli_episode_has_its_apps_as_commands_and_no_mcp_tools(sandboxes) -> None:
    server = make_server()
    with mcp_client(server) as local, mcp_client(server, client=SANDBOX) as remote:
        seeded = local.post("/seed_session", json=seed_body(in_surface(SPEC, "cli")))
        assert seeded.status_code == 200, seeded.text
        grant = seeded.json()["resources_tools"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]
        assert rpc(local, "tools/list", {}, grant).json()["result"]["tools"] == []
        (sandbox,) = sandboxes.started
        assert sandbox.spec.files[APP_COMMAND_PATH] == APP_COMMAND_SOURCE
        guide = sandbox.spec.files["/workspace/README.md"]
        assert guide == workspace_guide(8192, ("gdrive", "user"), surface="cli")
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


def test_an_api_search_episode_finds_endpoints_over_mcp_and_calls_them_over_http(sandboxes) -> None:
    server = make_server()
    with mcp_client(server) as local, mcp_client(server, client=SANDBOX) as remote:
        seeded, auth, tool, _ = seed_in(local, sandboxes, SPEC, "api_search")
        grant = seeded.json()["resources_tools"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]
        tools = rpc(local, "tools/list", {}, grant).json()["result"]["tools"]
        assert {listed["name"] for listed in tools} == {"api_search", "api_fetch"}
        assert sandboxes.started[0].spec.files["/workspace/README.md"] == workspace_guide(
            8192, ("gdrive", "user"), surface="api_search"
        )
        found = tool("api_search", query="list drive files")
        hit = next(result for result in found["results"] if result["endpoint"] == "gdrive.list_files")
        assert hit["path"] == "/v1/gdrive/list_files"
        fetched = tool("api_fetch", endpoint="gdrive.list_files")
        assert fetched["http"]["path"] == "/v1/gdrive/list_files"
        assert '"$COMPANY_API_URL/v1/gdrive/list_files"' in fetched["http"]["curl"]

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
    from officeenv.session import Session
    from officeenv.task import TaskSpec

    own = {
        tool["name"]: tool["parameters"]
        for tool in Session(TaskSpec.model_validate(in_surface(SPEC, "api_search"))).tools()
    }
    for name, schema, _ in SEARCH_TOOLS:
        assert (
            set(schema["properties"]) == set(own[name]["properties"]) and schema["required"] == own[name]["required"]
        )


def test_api_fetch_shows_a_curl_command_that_keeps_its_example_intact() -> None:
    doc = {"endpoint": "gdrive.list_files", "examples": [{"q": "name contains 'budget' and trashed = false"}]}
    curl = shlex.split(json.loads(with_http_calls("api_fetch", json.dumps(doc), guide=GUIDE))["http"]["curl"])
    assert curl[curl.index("-d") + 1] == json.dumps(doc["examples"][0])
    error = '{"error":{"code":"unknown_endpoint"}}'
    assert with_http_calls("api_fetch", error, guide=GUIDE) == error
    assert with_http_calls("gdrive_list_files", "{}", guide=GUIDE) == "{}"


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
            assert http.post("/seed_session", json=seed_body(in_surface(content_spec(), "cli"))).status_code == 200
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


@pytest.mark.parametrize("surface", SANDBOX_SURFACES)
def test_the_file_api_moves_files_whatever_the_surface(sandboxes, surface: str) -> None:
    """A cli episode has no gdrive_create_file tool, and an api_search episode neither: the file API's calls are
    answered as the tools exposure answers them, and an app's error comes back under its HTTP status."""
    server = make_server()
    with mcp_client(server) as client:
        _, auth, _, _ = seed_in(client, sandboxes, content_spec(), surface)
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
        (episode,) = server._episodes.values()
        assert episode.direct is False and episode.uploads_refused == 0  # its own calls are its surface's again


@pytest.mark.parametrize("surface", ["cli", "api_search"])
@pytest.mark.parametrize("app", ["gdrive", *FILE_APPS_UNDER_TEST])
def test_commands_and_endpoints_keep_file_content_out_of_the_conversation(sandboxes, app: str, surface: str) -> None:
    prefix = {"gdrive": "F"}.get(app, app[0].upper())
    op, args = DOWNLOADS.get(app, ("gdrive_download_file", lambda file_id: {"file_id": file_id}))
    op = op.removeprefix(f"{app}_")
    with mcp_client(make_server()) as client:
        seeded, _, _, call = seed_in(client, sandboxes, file_apps_spec(), surface)
        sandbox = sandboxes.started[-1]
        book = call(app, op, **args(f"{prefix}_BOOK"))
        assert not {"content", "contentBase64", "content_base64", "preview"} & set(book), book
        folder = FILE_APPS[app].folder
        assert book["savedTo"] == f"/workspace/{folder}/{prefix}_BOOK/ledger.xlsx" and "binary content" in book["note"]
        assert sandbox.files[book["savedTo"]] == BOOK
        upload, where = UPLOADS[app]
        refused = call(app, upload, **where, content_base64=BOOK_B64)
        assert "binary content" in refused_message(refused) and "curl -sf" in refused_message(refused), refused
        assert "error" not in call(app, upload, **where, content=SHORT)  # short text still goes in the call
        result = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert result["sandbox_files_saved"] == 1.0 and result["sandbox_uploads_refused"] == 1.0


def test_commands_and_endpoints_describe_content_as_it_moves_here(sandboxes) -> None:
    """A command's --help (the app's list and the command's own) and api_fetch's and api_search's documentation say
    what the ops moving content do in a sandbox, in place of OfficeEnv's words."""
    moving = sorted(CONTENT_DOWNLOADS | CONTENT_UPLOADS | ATTACHMENT_DOWNLOADS)
    spec = with_mail(file_apps_spec())
    with mcp_client(make_server()) as client:
        _, auth, _, _ = seed_in(client, sandboxes, spec, "cli")
        for app, op in moving:
            as_app = {**auth, "User-Agent": CLI_USER_AGENT.format(app=app)}
            command = op.replace("_", "-")
            listed = client.post("/v1/cli", headers=as_app, content=nul_args(app, "--help")).text
            helped = client.post("/v1/cli", headers=as_app, content=nul_args(app, command, "--help")).text
            text = content_description(app, op, "", 8192, guide=GUIDE)
            assert text in listed and helped.startswith(f"{app} {command}: {text}"), (app, op, helped[:300])
            assert "content_base64" not in listed and "contentBase64" not in listed
    with mcp_client(make_server()) as client:
        _, _, tool, _ = seed_in(client, sandboxes, spec, "api_search")
        for app, op in moving:
            fetched = tool("api_fetch", endpoint=f"{app}.{op}")
            assert fetched["description"] == content_description(app, op, "", 8192, guide=GUIDE), (app, op)
            hits = tool("api_search", query=fetched["description"], app=app)["results"]
            assert (
                next(hit for hit in hits if hit["endpoint"] == f"{app}.{op}")["description"] == fetched["description"]
            )


@pytest.mark.parametrize("surface", ["cli", "api_search"])
def test_commands_and_endpoints_move_mail_attachments_through_the_workspace(sandboxes, surface: str) -> None:
    with mcp_client(make_server()) as client:
        seeded, _, _, call = seed_in(client, sandboxes, with_mail(content_spec()), surface, files=MAIL_FILES)
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


@pytest.mark.parametrize("surface", ["cli", "api_search"])
def test_commands_and_endpoints_attach_a_workspace_file_named_by_a_bare_string(sandboxes, surface: str) -> None:
    """A command line writes an attachment's path as ``--attachments=/workspace/out/totals.xlsx``. A bare string that
    writes a path, or names a file the workspace holds, attaches that file, and a refusal names a path once."""
    with mcp_client(make_server()) as client:
        seeded, _, _, call = seed_in(client, sandboxes, with_mail(content_spec()), surface, files=MAIL_FILES)
        sent = call("gmail", "send_message", **MESSAGE, attachments="/workspace/out/totals.xlsx")
        assert "error" not in sent, sent
        listed = call("gmail", "get_message", id=sent["id"])["attachments"]
        assert [(a["filename"], a["size"]) for a in listed] == [("totals.xlsx", len(BOOK))]
        both = call("gmail", "send_message", **MESSAGE, attachments=["out/totals.xlsx", "notes.csv"])
        assert "error" not in both, both
        assert [a["filename"] for a in call("gmail", "get_message", id=both["id"])["attachments"]] == [
            "totals.xlsx",
            "notes.csv",
        ]
        unnamed = call("gmail", "send_message", **MESSAGE, attachments=[{"filename": "/workspace/aging_2026-10.xlsx"}])
        assert '{"path": "/workspace/aging_2026-10.xlsx"}' in refused_message(unnamed), unnamed
        assert "//workspace" not in json.dumps(unnamed)
        result = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert result["sandbox_files_attached"] == 3.0 and result["sandbox_uploads_refused"] == 0.0


def test_example_attachments_name_the_workspace_path_once() -> None:
    assert example_attachments(["totals.xlsx", "/workspace/out/totals.xlsx", {"filename": "/workspace/a.csv"}]) == [
        {"path": "/workspace/totals.xlsx"},
        {"path": "/workspace/out/totals.xlsx"},
        {"path": "/workspace/a.csv"},
    ]
    assert example_attachments([{"path": "x"}, {"filename": "y", "content": "z"}]) == [
        {"path": "x"},
        {"filename": "y", "content": "z"},
    ]


def test_only_help_is_rewritten_whole_lines_at_a_time(sandboxes) -> None:
    """--help says what the ops that move content do here, whole descriptions replaced line by line (no OfficeEnv
    tail survives, whatever the hash seed); an op's own answer is the app's data and keeps OfficeEnv's words."""
    with mcp_client(make_server()) as client:
        _, auth, _, call = seed_in(client, sandboxes, content_spec(), "cli")
        as_drive = {**auth, "User-Agent": CLI_USER_AGENT.format(app="gdrive")}
        helped = client.post("/v1/cli", headers=as_drive, content=nul_args("gdrive", "create-file", "--help")).text
        options = {line.split("=", 1)[0].strip(): line for line in helped.splitlines() if line.startswith("  --")}
        assert options["--content"].endswith("with the file API (/workspace/README.md)"), options["--content"]
        assert options["--content-base64"].endswith("never pasted here"), options["--content-base64"]
        assert "XLSX date cells" not in helped and "base64-encoded" not in helped
        listed = client.post("/v1/cli", headers=as_drive, content=nul_args("gdrive", "--help")).text
        assert "base64-encoded" not in listed and "contentBase64" not in listed
        # the app's data, never rewritten: a file whose text is one of OfficeEnv's descriptions comes back as it is
        official = "Text content to upload"
        created = call("gdrive", "create_file", name="note.txt", content=official)
        assert call("gdrive", "download_file", file_id=created["id"])["content"] == official


@pytest.mark.parametrize("surface", ["cli", "api_search"])
def test_commands_and_endpoints_refuse_once_verification_begins(sandboxes, surface: str) -> None:
    server = make_server()
    with mcp_client(server) as client:
        _, auth, _, _ = seed_in(client, sandboxes, content_spec(), surface)
        (episode,) = server._episodes.values()
        episode.grading = True  # as /verify sets it before it stops the sandbox
        if surface == "cli":
            as_drive = {**auth, "User-Agent": CLI_USER_AGENT.format(app="gdrive")}
            got = client.post("/v1/cli", headers=as_drive, content=nul_args("gdrive", "list-files"))
        else:
            got = client.post("/v1/gdrive/list_files", headers=auth, json={})
        assert got.status_code == 409 and "The task has ended" in got.text, got.text


@pytest.mark.parametrize("surface", ["cli", "api_search"])
def test_a_save_the_provider_fails_fails_a_command_or_an_endpoint_as_a_tool(sandboxes, surface: str) -> None:
    """Commands and endpoints run under the episode's lock as the tools do: a save the provider fails answers 502
    without the provider's words, and the episode's verification is masked."""
    with mcp_client(make_server()) as client:
        seeded, auth, _, _ = seed_in(client, sandboxes, content_spec(), surface)
        sandboxes.fail_on = "upload"
        if surface == "cli":
            as_drive = {**auth, "User-Agent": CLI_USER_AGENT.format(app="gdrive")}
            command = nul_args("gdrive", "download-file", "--file-id=F_BOOK")
            got = client.post("/v1/cli", headers=as_drive, content=command)
        else:
            got = client.post("/v1/gdrive/download_file", headers=auth, json={"file_id": "F_BOOK"})
        assert got.status_code == 502, got.text
        assert got.json()["detail"] == "Could not save /workspace/drive/F_BOOK/ledger.xlsx in the workspace"
        verified = client.post("/verify", json=VERIFY_BODY, cookies=seeded.cookies).json()
        assert verified["mask_sample"] is True and verified["failure_kind"] == "provider_unavailable"


@pytest.mark.parametrize("surface", ["cli", "api_search"])
def test_verification_waits_for_a_command_or_an_endpoint_saving_its_download(
    sandboxes, monkeypatch, surface: str
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
        seeded, _, _, call = seed_in(client, sandboxes, content_spec(), surface)
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


def test_a_cli_episode_whose_commands_cannot_be_linked_leaves_no_world_and_no_sandbox(sandboxes) -> None:
    """The app commands are linked as the sandbox starts (ln refuses a name the image already has there): a failure
    fails the seed as a failed start does, and the seed can be retried."""
    sandboxes.refusing = (f"chmod 755 {APP_COMMAND_PATH}",)
    server = make_server()
    row = seed_body(in_surface(content_spec(), "cli"))
    with mcp_client(server) as client:
        with pytest.raises(RuntimeError, match="Sandbox setup failed"):
            client.post("/seed_session", json=row)
        assert len(server.service.store) == 0 and not server._episodes and not server._tokens
        assert len(sandboxes.stopped) == 1
        sandboxes.refusing = ()
        assert client.post("/seed_session", json=row).status_code == 200


def test_commands_and_endpoints_describe_mail_attachments_as_they_move_here(sandboxes) -> None:
    spec = with_mail(content_spec())
    with mcp_client(make_server()) as client:
        _, auth, _, _ = seed_in(client, sandboxes, spec, "cli")
        as_gmail = {**auth, "User-Agent": CLI_USER_AGENT.format(app="gmail")}
        helped = client.post("/v1/cli", headers=as_gmail, content=nul_args("gmail", "send-message", "--help")).text
        assert "attaches a file from the workspace" in helped and "content_base64" not in helped
        assert """'--attachments=[{"path":"/workspace/forecast.xlsx"}]'""" in helped  # OfficeEnv's example, here
        assert "Getting one (gmail get-attachment)" in sandboxes.started[-1].spec.files["/workspace/README.md"]
    with mcp_client(make_server()) as client:
        _, _, tool, _ = seed_in(client, sandboxes, spec, "api_search")
        fetched = tool("api_fetch", endpoint="gmail.send_message")
        assert (
            "attaches a file from the workspace" in fetched["parameters"]["properties"]["attachments"]["description"]
        )
        assert fetched["examples"][0]["attachments"] == [{"path": "/workspace/forecast.xlsx"}]
        assert '"attachments": [{"path": "/workspace/forecast.xlsx"}]' in fetched["http"]["curl"]
        assert (
            "Getting one (the gmail.get_attachment endpoint)"
            in sandboxes.started[-1].spec.files["/workspace/README.md"]
        )


def test_the_guide_says_how_each_surface_reaches_the_apps() -> None:
    apps = ("gdrive", "gmail", "user")
    tools, cli, endpoints = (
        workspace_guide(8192, apps, surface=surface, prefix="company_") for surface in SANDBOX_SURFACES
    )
    assert "are your tools, one per operation of an app\n(company_<app>_<operation>)" in tools
    assert "Getting one (company_gmail_get_attachment)" in tools
    assert "commands here, one per connected app: gdrive, gmail, user." in cli
    assert "Downloading a file with the apps' commands saves it" in cli and "Getting one (gmail get-attachment)" in cli
    assert 'curl -sS -X POST "$COMPANY_API_URL/v1/<app>/<operation>"' in endpoints
    assert "your company_api_search\ntool" in endpoints and "with the apps' endpoints" in endpoints
    for guide in (tools, cli, endpoints):
        assert guide.startswith("# Company apps\n\n") and "$COMPANY_API_URL/v1/files" in guide
        assert f"a shell\nstarted without them gets them with `. {CREDENTIALS_PATH}`" in guide


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
    assert make_server(sandbox_provider=None).get_session_cookie_name() == named_key


def test_sandbox_mode_needs_one_worker() -> None:
    with pytest.raises(ValueError, match="num_workers=1"):
        make_server(num_workers=2)


def test_trusted_callers_are_this_node_and_the_trusted_networks() -> None:
    local = frozenset({"127.0.0.1", "::1", "10.0.0.1"})
    networks = (ipaddress.ip_network("10.109.0.0/16"),)
    for host in ("127.0.0.1", "::1", "::ffff:127.0.0.1", "10.0.0.1", "::ffff:10.0.0.1", "10.109.30.14"):
        assert is_trusted_caller(host, local=local, networks=networks), host
    for host in ("203.0.113.47", "10.0.0.2", "172.17.0.2", "testclient", "", None):
        assert not is_trusted_caller(host, local=local, networks=networks), host


# Without a sandbox provider.


def test_without_a_sandbox_provider_nothing_changes_for_any_caller(monkeypatch) -> None:
    def no_lookup() -> frozenset[str]:
        raise AssertionError("only sandbox mode looks up this node's addresses")

    monkeypatch.setattr(officeenv_app, "node_addresses", no_lookup)
    server = make_server(sandbox_provider=None)
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
    r"officeenv|office_env|\bgym\b|nemo|rollout|episode|simulat|seed|verif|surface|harness|dummy|token.capture|ng-trace",
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


# The calls that move content, as sandbox_view makes them on each surface: for each file app a download saved and a
# pasted upload refused, and Gmail's attachments: one saved, a workspace file attached, and attachments refused
# (pasted, a filename alone, no such file); then a call to an app the episode does not connect.
CONTENT_CALLS = (
    ("gdrive", "download_file", {"file_id": "F_BOOK"}),
    ("gdrive", "create_file", {"name": "x.bin", "content_base64": "AAEC"}),
    ("box", "get_file_content", {"file_id": "B_LONG"}),
    ("box", "upload_file", {"name": "x.csv", "parent_id": "B_FIN", "content": LONG}),
    ("sharepoint", "get_item_content", {"drive_id": SP_DOCS, "item_id": "S_LONG"}),
    ("sharepoint", "upload_content", {"drive_id": SP_DOCS, "path": "/Finance/x.csv", "content": LONG}),
    ("onedrive", "get_content", {"item_id": "O_LONG"}),
    ("onedrive", "upload_file", {"item_id": "O_LONG", "content": LONG}),
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


def sandbox_view(server: OfficeEnvResourcesServer, sandboxes: type[FakeSandbox], surface: str = "tools") -> str:
    """Everything a sandboxed model can read of ``server`` in a ``surface`` episode: its variables, the files and
    commands its sandbox gets, the MCP endpoint's name, header and tools, and the answers of every route a sandbox
    reaches, right and wrong (saved downloads, refused uploads, a call outside the episode's apps, a command made by
    hand, the routes of other surfaces and those it may not use, the docs), with the cookies they set, for each file
    app and Gmail's attachments. The finding tools and the ops moving content are described by this server (in MCP,
    in a command's --help, in api_fetch's documentation); OfficeEnv's other descriptions are OfficeEnv's to check.
    Tokens are masked."""
    seen: list[str] = []
    moving = CONTENT_DOWNLOADS | CONTENT_UPLOADS | ATTACHMENT_DOWNLOADS | ATTACHMENT_UPLOADS
    content_tools = {f"{app}_{op}" for app, op in moving}
    with mcp_client(server) as local, mcp_client(server, client=SANDBOX) as remote:
        seeded = local.post("/seed_session", json=seed_body(in_surface(with_mail(file_apps_spec()), surface)))
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
        described = {tool["name"] for tool in tools if tool["name"].startswith(("gdrive_", "api_"))} | content_tools
        seen += [json.dumps(tool) if tool["name"] in described else tool["name"] for tool in tools]
        auth = {"Authorization": f"Bearer {token}"}
        for app, op, arguments in CONTENT_CALLS:  # the surface's way
            if surface == "tools":
                seen.append(mcp_call("tools/call", {"name": f"{app}_{op}", "arguments": arguments}, granted))
            elif surface == "cli":
                as_app = {**auth, "User-Agent": CLI_USER_AGENT.format(app=app)}
                for argv in (command_flags(arguments), ["--help"]):
                    command = nul_args(app, op.replace("_", "-"), *argv)
                    seen.append(read(remote.post("/v1/cli", headers=as_app, content=command)))
            else:
                seen.append(read(remote.post(f"/v1/{app}/{op}", headers=auth, json=arguments)))
                fetch = {"name": "api_fetch", "arguments": {"endpoint": f"{app}.{op}"}}
                seen.append(mcp_call("tools/call", fetch, granted))
        if surface == "cli":  # each app's list of its commands
            for app in sorted(with_mail(file_apps_spec())["apps"]):
                as_app = {**auth, "User-Agent": CLI_USER_AGENT.format(app=app)}
                seen.append(read(remote.post("/v1/cli", headers=as_app, content=nul_args(app, "--help"))))
        if surface == "api_search":
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
            ("get", "/v1/files/box/B_SHORT", {"headers": auth}),
            ("get", "/v1/files/box/X_NONE", {"headers": auth}),
            (
                "post",
                "/v1/files",
                {
                    "headers": auth,
                    "params": {"app": "box", "name": "totals.csv", "parent": "B_FIN"},
                    "content": b"a,b",
                },
            ),
            ("get", "/v1/files/sharepoint/X_NONE", {"headers": auth}),
            (
                "post",
                "/v1/files",
                {"headers": auth, "params": {"app": "sharepoint", "name": "a.csv"}, "content": b"a,b"},
            ),
            ("put", "/v1/files/onedrive/X_NONE", {"headers": auth, "content": b"a,b"}),
        ]:
            answer = getattr(remote, method)(path, **kwargs)
            assert answer.status_code != 500, (path, answer.text)
            seen.append(read(answer))
        seen += sandbox.commands  # the relay, and the saved download's folder and mode
        seen += list(sandbox.files)
    return "\n".join(seen).replace(token, "<token>").replace(grant.removeprefix("Bearer "), "<token>")


@pytest.mark.parametrize("surface", SANDBOX_SURFACES)
def test_a_sandbox_reads_environment_names_only_in_the_model_url(sandboxes, surface: str) -> None:
    server = make_server(sandbox_config={**IMAGE, "host_relay": "host.lima.internal"}, **configured_mcp_settings())
    text = sandbox_view(server, sandboxes, surface)
    assert "COMPANY_API_URL=http://10.0.0.1:8080" in text and "/workspace/README.md" in text
    assert "savedTo" in text and "$COMPANY_API_URL/v1/files" in text  # a saved download, a refused pasted upload
    # every file app's: a download saved in its folder, and its refused upload's file API command
    assert all(f"/workspace/{app}/" in text for app in ("drive", "box", "sharepoint", "onedrive"))
    assert "files?app=box&" in text and "files?app=sharepoint&" in text and "/v1/files/onedrive/O_LONG" in text
    # Gmail's attachments: one saved, and the refusals that name the workspace file to attach
    assert "/workspace/gmail/M_LEDGER/" in text and "there is no file at /workspace/missing.xlsx" in text
    assert "carries no content" in text and "attach it by its path" in text
    assert RELAY_PATH in text and server.get_session_cookie_name() in text  # the relay, a cookie an answer sets
    if surface == "cli":  # the apps' command and the links that make each app one
        assert APP_COMMAND_PATH in text and "/usr/local/bin/gmail" in text
    if surface == "api_search":  # an endpoint of Gmail's own send-as vocabulary, found and let pass by name
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

    from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle
    from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProbeConfig, OpenSandboxProvider

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

    path = Path(officeenv_app.__file__).parent / "sandbox" / "prepare_rows.py"
    spec = importlib.util.spec_from_file_location("prepare_rows", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_row_prompts_name_the_company_api_and_how_each_surface_reaches_the_apps() -> None:
    prepare_rows = load_prepare_rows()
    assert (
        prepare_rows.system_prompt()
        == prepare_rows.SYSTEM_PROMPT
        == json.loads(SANDBOX_ROWS.read_text().splitlines()[0])["responses_create_params"]["input"][0]["content"]
    )  # the committed rows' prompt
    for surface in SANDBOX_SURFACES:
        prompt = prepare_rows.system_prompt(surface)
        assert "/workspace/README.md" in prompt and "COMPANY_API_URL" in prompt and "COMPANY_API_TOKEN" in prompt
        assert "in a shell without them, run `. /etc/company-api.env` first" in prompt  # Codex's shell, for one
        assert "attach workspace files to email by their path" in prompt
        assert not ENVIRONMENT_NAMES.search(prompt), prompt
    assert "run `user complete-task --answer=" in prepare_rows.system_prompt("cli")
    assert "your mcp__company__api_search tool" in prepare_rows.system_prompt("api_search", "mcp__company__")


@pytest.mark.parametrize("surface", ["cli", "api_search"])
def test_prepared_rows_seed_as_sandboxed_episodes_on_their_surface(sandboxes, surface: str) -> None:
    """prepare_rows.py --exposure: every sandbox task, its prompt saying how its apps are reached, seeds as an
    episode of that surface on the pinned OfficeEnv, with the grant and the guide of that surface."""
    rows = load_prepare_rows().rows(surface)
    assert {row["officeenv"]["exposure"]["mode"] for row in rows} == {surface}
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
            assert tools == ({"api_search", "api_fetch"} if surface == "api_search" else set()), row["task_id"]
            episode = server._episodes[f"rs-{index}"]
            guide = sandboxes.started[-1].spec.files["/workspace/README.md"]
            assert guide == workspace_guide(8192, episode.apps, surface=surface)


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
            "resources_servers/officeenv/configs/officeenv_sandbox_episodes.yaml",
            f"responses_api_agents/{agent}/configs/{agent}.yaml",
            "environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml",
        ], name
        binding = config["single_agent_turn_legacy"]["environment_servers"]["single_agent_turn_legacy"]
        assert binding["agent_server"]["name"] == agent and binding["resources_server"]["name"] == "oe", name
        assert binding["resources_tool_transports"] == transports, name
        assert set(config) - {"config_paths", "oe", "single_agent_turn_legacy"} == {agent}, name


AGENTS = Path(__file__).parents[3] / "responses_api_agents"
HARNESS_RUNTIME = Path(__file__).parents[1] / "sandbox" / "harness_runtime.sh"


@pytest.mark.parametrize(
    ("harness", "agent", "variable", "version_of"),
    [
        ("pi", "pi_agent", "PI_VERSION", "pi_version"),
        ("codex", "codex_agent", "CODEX_VERSION", "codex_version"),
        ("openclaw", "openclaw_agent", "OPENCLAW_VERSION", "openclaw_version"),
        ("opencode", "opencode_agent", "OPENCODE_VERSION", "opencode_version"),
    ],
)
def test_a_harness_image_holds_the_runtime_its_adapter_looks_for(
    harness: str, agent: str, variable: str, version_of: str
) -> None:
    """sandbox/harness_runtime.sh puts a harness's runtime in its image at the adapter's configured version and in the
    adapter's runtime folder: an adapter that changes either installs again at session start, which a sandbox
    without the internet cannot."""
    block = HARNESS_RUNTIME.read_text().split(f"\n  {harness})\n", 1)[1].split(";;", 1)[0]
    version = re.search(rf"version=\$\{{{variable}:-([^}}]+)\}}", block).group(1)
    folder = re.search(r"runtime=(\S+)", block).group(1).replace("$version", f"{{self.config.{version_of}}}")
    assert f'runtime = f"{folder}"' in (AGENTS / agent / "app.py").read_text()
    if harness == "opencode":
        runtime_module = (AGENTS / agent / "runtime.py").read_text()
        configured = re.search(r'^OPENCODE_VERSION = "([^"]+)"', runtime_module, re.MULTILINE).group(1)
    else:
        settings = yaml.safe_load((AGENTS / agent / "configs" / f"{agent}.yaml").read_text())
        configured = settings[agent]["responses_api_agents"][agent][version_of]
    assert version == str(configured)


@pytest.mark.parametrize("harness", ["pi", "codex", "openclaw", "opencode"])
def test_a_harness_image_keeps_no_npm_cache_its_installer_left(harness: str) -> None:
    """npm's cache keeps debug logs that name the runtime folder. An installer that moves HOME puts it in the runtime
    (``npm_config_cache``, as Codex's and OpenClaw's do), any other in ~/.npm: sandbox/harness_runtime.sh removes it
    wherever the adapter's installer puts it, once the harness has run, which leaves Node's compile cache too."""
    script = HARNESS_RUNTIME.read_text()
    tidy = script.split("tidy() {", 1)[1].split("\n}", 1)[0]
    block = script.split(f"\n  {harness})\n", 1)[1].split(";;", 1)[0]
    installer = (AGENTS / re.search(r'bash "\$agents/([^"]+)"', block).group(1)).read_text()
    cache = re.search(r'npm_config_cache="\$runtime/([^"]+)"', installer)
    assert (f'"$1/{cache.group(1)}"' if cache else '"${HOME:-/root}/.npm"') in tidy
    assert "tidy" not in block and script.rindex('tidy "$runtime"') > script.rindex("esac")


# The committed rows.


def test_the_sandbox_rows_seed_on_the_pinned_officeenv() -> None:
    """The rows sandbox/prepare_rows.py wrote seed as an Environment Server sends them. The workbook task
    seeds a binary Drive file, which OfficeEnv releases up to 1.7.0 refuse."""
    rows = [json.loads(line) for line in SANDBOX_ROWS.read_text().splitlines()]
    assert {row["task_id"] for row in rows} >= {"sandbox_vendor_spend", "sandbox_vendor_spend_xlsx"}
    with mcp_client(make_server(sandbox_provider=None)) as client:
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
            assert {"gdrive_download_file", "user_complete_task"} <= tools, row["task_id"]
