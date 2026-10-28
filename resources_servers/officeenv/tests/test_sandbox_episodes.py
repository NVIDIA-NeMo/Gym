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
import stat
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
    RELAY_PATH,
    RELAY_SOURCE,
    WORKSPACE_GUIDE,
    OfficeEnvResourcesServer,
    OfficeEnvResourcesServerConfig,
    OfficeEnvSandboxConfig,
    _path_part,
    egress_urls,
    is_trusted_caller,
    relay_ports,
    saved_download,
    tools_exposure_catalog,
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

    def __init__(self, provider_config: dict) -> None:
        self.provider_config = provider_config
        self.commands: list[str] = []
        self.files: dict[str, bytes] = {}  # what the server uploaded into it, by path
        self.modes: dict[str, int] = {}  # the mode of each upload's local file

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
        refused = command.startswith(FakeSandbox.refusing)
        return SimpleNamespace(error_type=None, return_code=int(refused), stdout="", stderr="refused" * refused)

    async def upload(self, local_path: str, remote_path: str) -> None:
        if FakeSandbox.fail_on == "upload":
            raise RuntimeError("the sandbox is gone")
        self.files[remote_path] = Path(local_path).read_bytes()
        self.modes[remote_path] = stat.S_IMODE(os.stat(local_path).st_mode)

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
    FakeSandbox.refusing = ()
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
    """The MCP settings configs/officeenv_sandbox.yaml gives the server."""
    settings = yaml.safe_load(SANDBOX_CONFIG.read_text())["oe"]["resources_servers"]["officeenv"]
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
        assert sandbox.spec.files == {"/workspace/README.md": WORKSPACE_GUIDE}

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
        not_served = client.post("/seed_session", json=seed_body({**SPEC, "exposure": {"mode": "api_search"}}))
        assert not_served.status_code == 400
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
        assert guide == WORKSPACE_GUIDE and "drive/<file id>/<name>" in guide and "at most 8192" in guide
    assert "drive/<file id>" not in workspace_guide(None)


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
        assert got.status_code == 502 and "/workspace/drive/F_BOOK/ledger.xlsx in the workspace: a folder is there" in got.text
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
    assert saved_download("list_files", {}, {"files": []}, folder="/w", limit=10, name_of=lambda _: None) is None


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


def environment_names(text: str) -> set[str]:
    return {found.group(0).lower() for found in ENVIRONMENT_NAMES.finditer(text)}


def harness_model_url(*, token_capture: bool) -> str:
    """The policy's URL as Gym's agents hand it to a sandboxed harness for one rollout (Pi writes it into the
    sandbox, in its models.json); traces off."""
    from nemo_gym.client import apply_rollout_prefix

    return (
        apply_rollout_prefix("http://10.0.0.1:9000", "17-0", token_capture=token_capture, trace_context=False) + "/v1"
    )


def sandbox_view(server: OfficeEnvResourcesServer, sandboxes: type[FakeSandbox]) -> str:
    """Everything a sandboxed model can read of ``server``: its variables, the files and commands its sandbox gets,
    the MCP endpoint's name, header and tools, and the answers of every route a sandbox reaches, right and wrong
    (a saved download, a refused upload, a tool outside the grant, the routes it may not use, the docs), with the
    cookies they set. Drive's tools are described by this server; OfficeEnv's other descriptions are OfficeEnv's to
    check. Tokens are masked."""
    seen: list[str] = []
    with mcp_client(server) as local, mcp_client(server, client=SANDBOX) as remote:
        seeded = local.post("/seed_session", json=seed_body(content_spec()))
        assert seeded.status_code == 200, seeded.text
        (sandbox,) = sandboxes.started
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
        seen += [json.dumps(tool) if tool["name"].startswith("gdrive_") else tool["name"] for tool in tools]
        for name, arguments in [
            ("gdrive_download_file", {"file_id": "F_BOOK"}),  # saved in the workspace
            ("gdrive_create_file", {"name": "x.bin", "content_base64": "AAEC"}),  # refused: pasted content
            ("slack_post_message", {"channel": "C1", "text": "hi"}),  # outside the grant
        ]:
            seen.append(mcp_call("tools/call", {"name": name, "arguments": arguments}, granted))
        seen.append(mcp_call("tools/list", {}, RPC_HEADERS))
        forged = {**RPC_HEADERS, "Authorization": "Bearer forged", NEMO_GYM_MCP_SESSION_TOKEN_HEADER: "forged"}
        seen.append(mcp_call("tools/call", {"name": "gdrive_list_files", "arguments": {}}, forged))
        auth = {"Authorization": f"Bearer {token}"}
        for method, path, kwargs in [
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
        ]:
            answer = getattr(remote, method)(path, **kwargs)
            assert answer.status_code != 500, (path, answer.text)
            seen.append(read(answer))
        seen += sandbox.commands  # the relay, and the saved download's folder and mode
        seen += list(sandbox.files)
    return "\n".join(seen).replace(token, "<token>").replace(grant.removeprefix("Bearer "), "<token>")


def test_a_sandbox_reads_environment_names_only_in_the_model_url(sandboxes) -> None:
    server = make_server(sandbox_config={**IMAGE, "host_relay": "host.lima.internal"}, **configured_mcp_settings())
    text = sandbox_view(server, sandboxes)
    assert "COMPANY_API_URL=http://10.0.0.1:8080" in text and "/workspace/README.md" in text
    assert "savedTo" in text and "$COMPANY_API_URL/v1/files" in text  # a saved download, a refused pasted upload
    assert RELAY_PATH in text and server.get_session_cookie_name() in text  # the relay, a cookie an answer sets
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
