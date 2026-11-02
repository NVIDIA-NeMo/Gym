# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The mock-tool server in sandboxed mode: each row's own backend runs in a fresh sandbox per rollout.

The sandbox provider here is a stand-in that runs the real runner as a local process over the files the server
uploads, so the server, the runner and the bundle meet as they do in a real sandbox.
"""

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import aiohttp
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym import PARENT_DIR
from nemo_gym.client import ServerClient
from nemo_gym.sandbox import AsyncSandbox, SandboxEndpoint, SandboxExecResult, SandboxHandle, SandboxStatus
from nemo_gym.server_runtime import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.server_runtime.mcp_auto_exposure import harvest_tools
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.partner_if import mock_tools_app as api
from resources_servers.partner_if import tool_sandbox
from resources_servers.partner_if.task_data import TaskData, ToolBackendBundle
from resources_servers.partner_if.tests.test_configs import server_config
from resources_servers.partner_if.tests.test_mock_tools_app import TOOLS, answer, call, episode, output, row, seed_for
from resources_servers.partner_if.tests.test_tool_runner import NOTES, PYDANTIC_PIN, free_port
from resources_servers.partner_if.tool_sandbox import ROOT, RUNNER_PORT, ToolSandboxConfig, network_options


# Records the rows it is given, so a test can see what reached the sandbox; `boom` fails like a backend bug.
PROBE = """
import json


class Probe:
    requires_final_answer = False

    def __init__(self, fixture_path):
        self.rows = []

    def tools(self, row):
        self.rows.append(row)
        return {name: {"description": name} for name in ("lookup", "append", "boom")}

    def new_state(self, row):
        self.rows.append(row)
        return []

    def call(self, state, name, arguments):
        if name == "boom":
            raise RuntimeError("secret backend detail")
        state.append(name)
        return {"error": "a backend's own error output"} if name == "append" else {"note": "HELLO"}

    def artifacts(self, state):
        return {"log": "HELLO", "rows": json.dumps(self.rows)}

    def receipts(self, state):
        return {"calls": state}
"""
PROBE_BUNDLE = {
    "entrypoint": "probe:Probe",
    "files": {"probe.py": PROBE, "fixture.json": "{}"},
    "fixture": "fixture.json",
    "requirements": [],
}


class LocalRunnerProvider:
    """Stands in for OpenSandbox: uploads land under a directory per sandbox, the runner starts as a local process
    on a free port, and an install is recorded but skipped, since this venv already has the bundles' pins."""

    name = "opensandbox"

    def __init__(self, root: Path):
        self.root = root
        self.specs = []
        self.uploads = {}
        self.commands = []
        self.processes = {}
        self.ports = {}
        self.stopped = []
        self.failing_closes = 0
        self.create_gate: asyncio.Event | None = None
        self.runner_gate: asyncio.Event | None = None

    def path(self, handle: SandboxHandle, path: str) -> Path:
        return self.root / handle.sandbox_id / path.lstrip("/")

    def files(self, index: int) -> dict[str, str]:
        """What the server uploaded into a sandbox, by path."""
        return self.uploads.get(f"sandbox-{index}", {})

    async def create(self, spec):
        self.specs.append(spec)
        if self.create_gate is not None:
            await self.create_gate.wait()
        return SandboxHandle(sandbox_id=f"sandbox-{len(self.specs) - 1}", provider_name=self.name, raw=None)

    async def upload_file(self, handle, source_path, target_path):
        target = self.path(handle, target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
        self.uploads.setdefault(handle.sandbox_id, {})[target_path] = source_path.read_text()

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.commands.append(command)
        root = self.path(handle, ROOT)
        if "pip install" in command:
            return SandboxExecResult(stdout="", stderr="", return_code=0)
        if "tool_runner.py" in command:
            if self.runner_gate is not None:
                await self.runner_gate.wait()
            port = free_port()
            config = json.loads((root / "runner.json").read_text())
            (root / "runner.json").write_text(json.dumps(config | {"host": "127.0.0.1", "port": port}))
            with open(root / "runner.log", "wb") as log:
                self.processes[handle.sandbox_id] = subprocess.Popen(
                    [sys.executable, "-I", str(root / "tool_runner.py"), str(root / "runner.json")],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            self.ports[handle.sandbox_id] = port
            return SandboxExecResult(stdout="", stderr="", return_code=0)
        if command.startswith("tail "):
            return SandboxExecResult(stdout=(root / "runner.log").read_text(), stderr="", return_code=0)
        raise AssertionError(f"Unexpected sandbox command: {command}")

    async def endpoint(self, handle, port):
        assert port == RUNNER_PORT
        return SandboxEndpoint(endpoint=f"http://127.0.0.1:{self.ports[handle.sandbox_id]}")

    async def close(self, handle):
        if self.failing_closes:
            self.failing_closes -= 1
            raise RuntimeError("Provider unavailable")
        if process := self.processes.pop(handle.sandbox_id, None):
            process.terminate()
            process.wait()
        self.stopped.append(handle.sandbox_id)

    async def download_file(self, handle, source_path, target_path):
        raise NotImplementedError

    async def status(self, handle):
        return SandboxStatus.STOPPED if handle.sandbox_id in self.stopped else SandboxStatus.RUNNING

    async def aclose(self):
        pass


@pytest.fixture
def provider(tmp_path, monkeypatch):
    provider = LocalRunnerProvider(tmp_path / "sandboxes")
    monkeypatch.setattr(tool_sandbox, "get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}})
    monkeypatch.setattr(tool_sandbox, "AsyncSandbox", lambda config: AsyncSandbox(provider, owns_provider=False))
    yield provider
    for process in provider.processes.values():
        process.terminate()
        process.wait()


@pytest.fixture(autouse=True)
def requests_sent(monkeypatch):
    """Each request on its own aiohttp session: Gym's shared client stays on one event loop, and TestClient runs
    each request on its own. Records the requests' retry options."""
    sent = []

    async def send(method, url, **options):
        recorded = {key: value for key, value in options.items() if key.startswith("_") or key == "ssl"}
        sent.append((url.rsplit("/", 1)[-1], recorded))
        async with aiohttp.ClientSession() as session:
            response = await session.request(method, url, **{k: v for k, v in options.items() if k[0] != "_"})
            await response.read()
            return response

    monkeypatch.setattr(tool_sandbox, "request", send)
    return sent


def sandboxed_row(bundle=None):
    return row() | {"tool_backend": NOTES | (bundle or {})}


def make_server(tmp_path, tasks=None, **config):
    tasks = tasks or [sandboxed_row()]
    (tmp_path / "tasks.jsonl").write_text("".join(json.dumps(task) + "\n" for task in tasks))
    return api.MockToolsResourcesServer(
        config=api.MockToolsServerConfig(
            host="127.0.0.1",
            port=8001,
            name="tools",
            entrypoint="mock_tools_app.py",
            sandbox_provider="sandbox",
            tasks_path=tmp_path / "tasks.jsonl",
            judge_model_server={"type": "responses_api_models", "name": "judge"},
            **config,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def test_a_rollout_runs_its_rows_own_backend_in_a_fresh_sandbox_and_is_graded(tmp_path, provider, requests_sent):
    task = sandboxed_row()
    resource = make_server(tmp_path, [task])
    client = TestClient(resource.setup_webserver())
    seed = seed_for(task)
    assert client.post("/seed_session", json=seed).status_code == 200
    assert client.post("/seed_session", json=seed).status_code == 200  # A repeated seed keeps its sandbox.
    [spec] = provider.specs
    assert spec.image == "python:3.13-slim" and spec.ports == (RUNNER_PORT,)
    assert spec.provider_options["network_policy"] == {
        "defaultAction": "deny",
        "egress": [{"action": "allow", "target": "files.pythonhosted.org"}, {"action": "allow", "target": "pypi.org"}],
    }
    assert spec.metadata == {"task": "smoke", "nemo_gym_agent": "tools"}
    install = next(command for command in provider.commands if "pip install" in command)
    assert "--only-binary=:all:" in install and "--index-url https://pypi.org/simple" in install
    assert provider.files(0)[f"{ROOT}/requirements.txt"] == f"{PYDANTIC_PIN}\n"
    # Only the runner goes to the background, with none of the command's streams, or the command never ends.
    start = next(command for command in provider.commands if "tool_runner.py" in command)
    assert start.endswith("< /dev/null &") and "&&" not in start

    found = client.post("/lookup", json={"key": "greeting"}).json()
    missing = client.post("/lookup", json={"key": "missing"})
    appended = client.post("/append", json={"text": "HELLO"})
    assert found == {"note": "HELLO"} and appended.text == "Appended"
    assert missing.status_code == 200 and missing.json() == {"error": "No note has the key 'missing'"}
    # A call that may have run in the sandbox is never sent twice.
    assert {options["_resend_after_disconnect"] for route, options in requests_sent if route == "call"} == {False}
    assert {options["ssl"] for _, options in requests_sent} == {True}

    events = [
        call(),
        output(found),
        call(args={"key": "missing"}, identifier="call2"),
        output(missing.json(), "call2"),
        call("append", {"text": "HELLO"}, "call3"),
        output(appended.text, "call3"),
        answer(),
    ]
    body = client.post("/verify", json=task | {"response": episode(events).model_dump(mode="json")}).json()
    assert body["reward"] == 1 and not body["mask_sample"]
    assert body["environment_receipts"] == {"appended": ["HELLO"]}
    assert [recorded["output"] for recorded in body["tool_calls"]] == [found, missing.json(), "Appended"]
    assert "tool_backend" not in body  # The bundle never travels with a result.
    assert body["tool_sandbox_id"] == "sandbox-0"

    close = {key: seed[key] for key in ("resources_session_id", "episode_id")}
    assert client.post("/close_session", json=close).status_code == 200
    assert provider.stopped == ["sandbox-0"] and not provider.processes


def test_the_sandbox_holds_only_the_backend_and_the_backend_sees_only_what_the_actor_sees(tmp_path, provider):
    task = sandboxed_row(PROBE_BUNDLE)
    resource = make_server(tmp_path, [task])
    client = TestClient(resource.setup_webserver())
    client.post("/seed_session", json=seed_for(task))
    files = provider.files(0)
    assert set(files) == {
        f"{ROOT}/tool_runner.py",
        f"{ROOT}/lib/resources_servers/partner_if/tool_backend.py",
        f"{ROOT}/runner.json",
        f"{ROOT}/bundle/probe.py",
        f"{ROOT}/bundle/fixture.json",
    }
    contract = task["verifier_metadata"]["partner_if"]
    grading_text = [
        contract["source"],
        contract["rubric_version"],
        *(item["constraint"] for item in contract["rubrics"]),
    ]
    assert not any(text in content for text in grading_text for content in files.values())
    assert "pip install" not in "".join(provider.commands)  # Nothing to install.

    rows = json.loads(asyncio.run(resource._episodes["session-one"].backend.artifacts())["rows"])
    assert rows == [{"responses_create_params": task["responses_create_params"]}] * 2  # From tools and new_state.

    client.post("/lookup", json={"key": "greeting"})
    raw = episode([call(), output({"note": "HELLO"})]).model_dump(mode="json")
    body = client.post("/verify", json=task | {"response": raw}).json()
    # The task's backend needs no final answer, so an episode ending on a tool result is graded.
    assert body["failure_kind"] is None and body["environment_receipts"] == {"calls": ["lookup"]}
    assert {item["id"]: item["passed"] for item in body["grading_result"]["items"]} == {
        "logged": True,
        "answered": False,
    }


def test_the_backend_that_runs_is_the_configured_snapshots(tmp_path, provider):
    task = sandboxed_row()
    client = TestClient(make_server(tmp_path, [task]).setup_webserver())
    seed = seed_for(task)
    other = seed | {"task_data": seed["task_data"] | {"tool_backend": NOTES | {"fixture": "example_tools.py"}}}
    assert client.post("/seed_session", json=other).status_code == 409
    assert not provider.specs
    same = seed | {"task_data": seed["task_data"] | {"tool_backend": task["tool_backend"]}}
    assert client.post("/seed_session", json=same).status_code == 200


def test_each_rollout_gets_its_own_sandbox_and_state(tmp_path, provider):
    task = sandboxed_row()
    resource = make_server(tmp_path, [task])
    first, second = TestClient(resource.setup_webserver()), TestClient(resource.setup_webserver())
    first.post("/seed_session", json=seed_for(task))
    second.post("/seed_session", json=seed_for(task, session="session-two"))
    first.post("/append", json={"text": "HELLO"})
    assert len(provider.specs) == 2
    assert asyncio.run(resource._episodes["session-two"].backend.receipts()) == {"appended": []}
    seed = seed_for(task)
    first.post("/close_session", json={key: seed[key] for key in ("resources_session_id", "episode_id")})
    assert provider.stopped == ["sandbox-0"]


def test_a_backend_failure_is_a_masked_infrastructure_error_whose_details_stay_in_the_log(tmp_path, provider, caplog):
    task = sandboxed_row(PROBE_BUNDLE)
    task["responses_create_params"]["tools"] = [*TOOLS, TOOLS[0] | {"name": "boom"}]
    client = TestClient(make_server(tmp_path, [task]).setup_webserver())
    client.post("/seed_session", json=seed_for(task))
    failed = client.post("/boom", json={})
    # The model sees the response body: no traceback, no exception and no sandbox detail reach it.
    assert failed.status_code == 500 and failed.text == "Internal Server Error"
    assert "probe:Probe failed in boom" in caplog.text and "secret backend detail" in caplog.text
    # A backend's own error output is an ordinary output, recorded like any other.
    errored = client.post("/append", json={"text": "HELLO"})
    assert errored.json() == {"error": "a backend's own error output"}
    events = [
        call("boom", {}),
        output(failed.text),
        call("append", {"text": "HELLO"}, "c2"),
        output(errored.json(), "c2"),
    ]
    body = client.post(
        "/verify", json=task | {"response": episode([*events, answer()]).model_dump(mode="json")}
    ).json()
    # The environment failed, not the model: the rollout is masked and names the failure.
    assert body["mask_sample"] and body["reward"] == 0 and body["failure_kind"] == "partner_if:tool_backend_failed"
    assert body["failure_reason"] == "The tool backend failed in boom" and body["tool_sandbox_id"] == "sandbox-0"
    assert [recorded["name"] for recorded in body["tool_calls"]] == ["append"]


def test_a_runner_lost_before_verify_masks_the_rollout(tmp_path, provider, caplog):
    task = sandboxed_row()
    client = TestClient(make_server(tmp_path, [task]).setup_webserver())
    client.post("/seed_session", json=seed_for(task))
    appended = client.post("/append", json={"text": "HELLO"}).text
    # The sandbox dies after the last call, such as a killed container.
    runner = provider.processes["sandbox-0"]
    runner.terminate()
    runner.wait()
    events = [call("append", {"text": "HELLO"}), output(appended), answer()]
    checked = client.post("/verify", json=task | {"response": episode(events).model_dump(mode="json")})
    assert checked.status_code == 200, checked.text
    body = checked.json()
    assert body["mask_sample"] and body["reward"] == 0 and body["failure_kind"] == "partner_if:tool_backend_failed"
    assert body["failure_reason"] == "The tool backend failed before verify" and body["environment_receipts"] == {}
    assert "failed before verify" in caplog.text


def test_a_task_whose_backend_does_not_serve_its_tools_fails_its_seed(tmp_path, provider):
    task = sandboxed_row(PROBE_BUNDLE)
    task["responses_create_params"]["tools"] = [*TOOLS, TOOLS[0] | {"name": "erase"}]
    seeded = TestClient(make_server(tmp_path, [task]).setup_webserver()).post("/seed_session", json=seed_for(task))
    assert seeded.status_code == 422 and "does not serve: erase" in seeded.json()["detail"]
    assert provider.stopped == ["sandbox-0"]


def test_a_backend_that_cannot_load_fails_its_seed_and_its_sandbox_stops(tmp_path, provider):
    task = sandboxed_row({"requirements": ["pydantic==0.0.1"]})
    resource = make_server(tmp_path, [task])
    client = TestClient(resource.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/seed_session", json=seed_for(task)).status_code == 500
    assert provider.stopped == ["sandbox-0"] and not resource._episodes


def test_an_install_the_provider_reports_as_failed_fails_the_seed_and_its_sandbox_stops(
    tmp_path, provider, monkeypatch
):
    original = provider.exec

    async def exec_(handle, command, **options):
        if "pip install" in command:
            # A provider can report a failure without a process exit code.
            return SandboxExecResult(stdout="", stderr="", return_code=0, error_type="timeout")
        return await original(handle, command, **options)

    monkeypatch.setattr(provider, "exec", exec_)
    task = sandboxed_row()
    resource = make_server(tmp_path, [task])
    client = TestClient(resource.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/seed_session", json=seed_for(task)).status_code == 500
    assert provider.stopped == ["sandbox-0"] and not resource._episodes


def test_tool_sandboxes_always_have_a_ttl_backstop():
    assert ToolSandboxConfig().ttl_s == 3600
    with pytest.raises(ValidationError):
        ToolSandboxConfig(ttl_s=None)


def test_sandboxed_sessions_keep_the_resources_session_contract(tmp_path, provider):
    task = sandboxed_row()
    resource = make_server(tmp_path, [task])
    check_resources_session_contract(
        resource.setup_webserver(), ResourcesSeedSessionRequest.model_validate(seed_for(task)), keeps_state=True
    )
    # The contract check also seeds a session whose reply is lost, then closes it without a cookie.
    assert provider.stopped == ["sandbox-0", "sandbox-1"]
    assert not resource._episodes


def test_a_sandbox_that_fails_to_stop_keeps_its_episode_for_a_retried_close(tmp_path, provider):
    task = sandboxed_row()
    resource = make_server(tmp_path, [task])
    client = TestClient(resource.setup_webserver(), raise_server_exceptions=False)
    seed = seed_for(task)
    client.post("/seed_session", json=seed)
    provider.failing_closes = 1
    close = {key: seed[key] for key in ("resources_session_id", "episode_id")}
    assert client.post("/close_session", json=close).status_code == 500
    assert "session-one" in resource._episodes and "session-one" not in resource._closed
    assert client.post("/close_session", json=close).status_code == 200
    assert provider.stopped == ["sandbox-0"] and not resource._episodes


def test_retire_and_shutdown_stop_sandboxes(tmp_path, provider):
    task = sandboxed_row()
    resource = make_server(tmp_path, [task])
    with TestClient(resource.setup_webserver()) as client:
        client.post("/seed_session", json=seed_for(task))
        client.post("/seed_session", json=seed_for(task, session="session-two"))
        client.portal.call(resource.retire_session_state, "session-one")
        assert provider.stopped == ["sandbox-0"]
        assert client.post("/seed_session", json=seed_for(task)).status_code == 409
    assert provider.stopped == ["sandbox-0", "sandbox-1"] and not resource._episodes


def seed_request(task, session="session-one"):
    return ResourcesSeedSessionRequest.model_validate(seed_for(task, session))


async def test_concurrent_seeds_of_one_session_start_one_sandbox(tmp_path, provider):
    task = sandboxed_row()
    resource = make_server(tmp_path, [task])
    provider.create_gate = asyncio.Event()
    first, second = SimpleNamespace(session={}), SimpleNamespace(session={})
    seeds = [asyncio.create_task(resource.seed_session(request, seed_request(task))) for request in (first, second)]
    await asyncio.sleep(0.1)
    provider.create_gate.set()
    await asyncio.gather(*seeds)
    assert len(provider.specs) == 1 and first.session == second.session
    await resource._stop_all()


async def test_a_close_racing_a_seed_waits_for_the_sandbox_and_stops_it(tmp_path, provider):
    task = sandboxed_row()
    resource = make_server(tmp_path, [task])
    provider.create_gate = asyncio.Event()
    seeding = asyncio.create_task(resource.seed_session(SimpleNamespace(session={}), seed_request(task)))
    await asyncio.sleep(0.1)
    close = ResourcesCloseSessionRequest(resources_session_id="session-one", episode_id=seed_for(task)["episode_id"])
    closing = asyncio.create_task(resource.close_resources_session(close))
    await asyncio.sleep(0.1)
    assert not closing.done()  # It holds off until the seed records its sandbox, so the sandbox cannot leak.
    provider.create_gate.set()
    await seeding
    await closing
    assert provider.stopped == ["sandbox-0"] and not resource._episodes and "session-one" in resource._closed
    with pytest.raises(api.HTTPException) as refused:
        await resource.seed_session(SimpleNamespace(session={}), seed_request(task))
    assert refused.value.status_code == 409 and len(provider.specs) == 1


async def test_a_seed_that_fails_after_its_sandbox_started_stops_it(tmp_path, provider):
    task = sandboxed_row()
    resource = make_server(tmp_path, [task])
    provider.runner_gate = asyncio.Event()
    seeding = asyncio.create_task(resource.seed_session(SimpleNamespace(session={}), seed_request(task)))
    while not any("tool_runner.py" in command for command in provider.commands):
        await asyncio.sleep(0.05)
    seeding.cancel()  # Gym cancels a request whose client disconnects, here while the sandbox sets up.
    with pytest.raises(asyncio.CancelledError):
        await seeding
    assert provider.stopped == ["sandbox-0"] and not resource._episodes
    provider.runner_gate = None
    await resource.seed_session(SimpleNamespace(session={}), seed_request(task))  # A retry starts afresh.
    assert len(provider.specs) == 2 and "session-one" in resource._episodes
    await resource._stop_all()


def test_mcp_exposes_each_rows_advertised_tools(tmp_path, provider):
    task = sandboxed_row()
    task["responses_create_params"]["tools"] = TOOLS[:1]
    resource = make_server(tmp_path, [task])
    assert set(harvest_tools(resource.setup_webserver(), resource)) == {"lookup"}
    assert resource.mcp_allowed_tools_for_session(seed_for(task)["task_data"]) == ["lookup"]
    other = sandboxed_row(PROBE_BUNDLE)
    other["verifier_metadata"]["partner_if"]["task_id"] = "other"
    other["responses_create_params"]["tools"] = [TOOLS[0] | {"description": "Another lookup"}]
    conflicting = make_server(tmp_path, [task, other])
    with pytest.raises(ValueError, match="gives tool lookup another schema"):
        harvest_tools(conflicting.setup_webserver(), conflicting)


def test_a_sandboxed_server_needs_a_bundle_on_every_row_and_one_backend_mode(tmp_path, provider):
    plain = row()
    with pytest.raises(ValueError, match="Task smoke has no tool_backend"):
        make_server(tmp_path, [plain])
    with pytest.raises(ValidationError, match="Set either backend"):
        make_server(tmp_path, backend="synthetic_tools:NotesBackend")
    with pytest.raises(ValidationError, match="fixture_path goes with an in-process backend"):
        make_server(tmp_path, fixture_path=tmp_path / "tasks.jsonl")


@pytest.mark.parametrize(
    "change,error",
    [
        ({"files": {"../escape.py": "", "notes.json": ""}, "entrypoint": "escape:B"}, "normalized relative paths"),
        ({"files": {"/abs.py": "", "notes.json": ""}, "entrypoint": "abs:B"}, "normalized relative paths"),
        ({"files": {"a//b.py": "", "notes.json": ""}, "entrypoint": "a.b:B"}, "normalized relative paths"),
        ({"fixture": "missing.json"}, "fixture 'missing.json' is not one of"),
        ({"entrypoint": "elsewhere:NotesBackend"}, "module 'elsewhere' is not one of"),
        ({"requirements": ["pydantic>=2"]}, "requirements.0"),
        ({"entrypoint": "example_tools"}, "entrypoint"),
    ],
)
def test_a_bundle_must_be_self_contained_and_pinned(change, error):
    with pytest.raises(ValidationError, match=error):
        ToolBackendBundle.model_validate(NOTES | change)
    package = NOTES | {"files": {"pkg/__init__.py": "", "notes.json": ""}, "entrypoint": "pkg:Backend"}
    assert ToolBackendBundle.model_validate(package).entrypoint == "pkg:Backend"


def test_the_network_is_restricted_where_the_provider_enforces_it_and_otherwise_refused():
    package_index, deny_all = ToolSandboxConfig(), ToolSandboxConfig(network="none")
    assert network_options(package_index, "opensandbox")["network_policy"]["defaultAction"] == "deny"
    assert network_options(deny_all, "opensandbox") == {"network_policy": {"defaultAction": "deny", "egress": []}}
    for config in (package_index, deny_all):
        with pytest.raises(ValueError, match="OpenSandbox"):
            network_options(config, "docker")
    assert network_options(ToolSandboxConfig(network="provider_default"), "docker") == {}
    with pytest.raises(ValueError, match="sandbox_config.network"):
        network_options(ToolSandboxConfig(provider_options={"network_policy": {}}), "opensandbox")


def test_with_no_egress_nothing_is_installed_and_the_image_must_have_the_pins(tmp_path, provider):
    task = sandboxed_row()
    client = TestClient(make_server(tmp_path, [task], sandbox_config={"network": "none"}).setup_webserver())
    assert client.post("/seed_session", json=seed_for(task)).status_code == 200  # This venv has the pin.
    assert "pip install" not in "".join(provider.commands)
    assert f"{ROOT}/requirements.txt" not in provider.files(0)
    assert provider.specs[0].provider_options["network_policy"] == {"defaultAction": "deny", "egress": []}


EXAMPLE = PARENT_DIR / "resources_servers/partner_if/data/mock_tools_sandboxed_example.jsonl"


def test_the_sandboxed_example_runs_the_example_backend_on_the_example_tasks():
    root = PARENT_DIR / "resources_servers/partner_if"
    rows = [json.loads(line) for line in EXAMPLE.read_text().splitlines()]
    in_process = [json.loads(line) for line in (root / "data/mock_tools_example.jsonl").read_text().splitlines()]
    assert [{key: value for key, value in task.items() if key != "tool_backend"} for task in rows] == in_process
    for task in rows:
        bundle = ToolBackendBundle.model_validate(task["tool_backend"])
        assert bundle.files == {
            "example_tools.py": (root / "example_tools.py").read_text(),
            "notes.json": (root / "data/mock_tools_notes.json").read_text(),
        }
        assert bundle.requirements == ["pydantic==2.13.4"]  # Installed in the sandbox, whatever this venv has.


def test_the_sandboxed_example_config_starts_and_gym_validates_its_rows(monkeypatch):
    from nemo_gym.task_data import TaskDataValidator, load_task_data_schema

    monkeypatch.setattr(tool_sandbox, "get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}})
    config = api.MockToolsServerConfig.model_validate(
        server_config("mock_tools_sandboxed.yaml", "partner_if_mock_tools_sandboxed")
    )
    assert config.sandbox_config.network == "package_index" and config.backend is None
    resource = api.MockToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    assert resource.mcp_allowed_tools_for_session(
        seed_for(json.loads(EXAMPLE.read_text().splitlines()[0]))["task_data"]
    ) == ["append", "lookup"]
    validator = TaskDataValidator("partner_if", load_task_data_schema(EXAMPLE.parent.parent), EXAMPLE.name)
    for i, line in enumerate(EXAMPLE.read_text().splitlines()):
        validator.validate_row(i, json.loads(line))
    assert validator.report.clean, validator.report.summary()


def test_undeclared_fields_are_rejected_rather_than_dropped():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ToolBackendBundle.model_validate(NOTES | {"timeout_s": 5})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ToolSandboxConfig.model_validate({"netwrok": "none"})
    contract = sandboxed_row()["verifier_metadata"]["partner_if"]
    assert TaskData.model_validate({"partner_if": contract, "tool_backend": NOTES}).tool_backend.entrypoint
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TaskData.model_validate({"partner_if": contract, "tool_backends": NOTES})


@pytest.mark.parametrize(
    "connection,verify", [({}, True), ({"tls_verify": True}, True), ({"tls_verify": False}, False)]
)
def test_the_runner_is_reached_with_the_providers_own_tls_setting(monkeypatch, connection, verify):
    # The endpoint is the provider's server, which may use a self-signed certificate its connection accepts.
    config = {"sandbox": {"opensandbox": {"connection": connection}}}
    monkeypatch.setattr(tool_sandbox, "get_global_config_dict", lambda: config)
    assert tool_sandbox.ToolSandboxes("sandbox", ToolSandboxConfig(), server_name="tools")._verify_tls is verify
