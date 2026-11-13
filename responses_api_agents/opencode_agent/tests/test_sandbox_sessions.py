# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import hashlib
import json
import shlex
import sqlite3
import subprocess
import threading
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AgentCloseSessionRequest,
    AgentSeedSessionRequest,
    is_harness_termination_failure,
    parse_cleanup_receipt,
    raises_agent_failure,
)
from nemo_gym.client import ServerClient
from nemo_gym.core import (
    PROVIDER_OOM_KILLED,
    EpisodeId,
    MediaFile,
    NeMoGymResponseCreateParamsNonStreaming,
    NgFailureEvent,
    SandboxFailureDetails,
    TaskId,
)
from nemo_gym.data.tasks.harbor import load_task, materialize_task
from nemo_gym.sandbox import SandboxHandle
from nemo_gym.trajectory import NG_AGENT_SANDBOX_OBSERVATION_KEY, SandboxObservation
from responses_api_agents.opencode_agent.app import OpenCodeAgent, OpenCodeAgentConfig, _sandbox_prepare_command
from responses_api_agents.opencode_agent.runtime import UNRESTRICTED_IMAGE_ATTACHMENT


def sandbox_oom(reason: str, *, sandbox_id: str = "task-5") -> NgFailureEvent:
    return NgFailureEvent(
        boundary="sandbox",
        failure_kind=PROVIDER_OOM_KILLED,
        failure_reason=reason,
        details=SandboxFailureDetails(provider="opensandbox", sandbox_id=sandbox_id),
    )


def seed() -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id=f"opencode-test-{uuid4().hex}",
        episode_id=EpisodeId(rollout_id="opencode-smoke", attempt=2),
        task_id=TaskId(taskset="swe-pro", task_id="task"),
        sandbox_access={
            "connection": {
                "kind": "direct",
                "provider_config_ref": "sandbox",
                "descriptor": {"sandbox_id": "resources-owned"},
            },
            "workdir": "/app",
        },
    )


def events() -> str:
    return json.dumps(
        {
            "messages": [
                {"info": {"role": "user"}, "parts": [{"type": "text", "text": "task"}]},
                {
                    "info": {
                        "role": "assistant",
                        "time": {"completed": 100},
                        "finish": "stop",
                        "tokens": {"input": 10, "output": 2, "reasoning": 1, "cache": {"read": 3}, "total": 12},
                    },
                    "parts": [
                        {"type": "reasoning", "text": "Inspect"},
                        {
                            "type": "tool",
                            "callID": "tool-1",
                            "tool": "bash",
                            "state": {"input": {"command": "pwd"}, "output": "/app"},
                        },
                        {"type": "text", "text": "Fixed"},
                    ],
                },
            ]
        }
    )


def rejected_events(*, after_work: bool = False) -> str:
    """Persist a terminal request rejection, optionally after a completed model turn."""
    export = json.loads(events())
    rejected = {
        "info": {
            "role": "assistant",
            "time": {"created": 99, "completed": 100},
            "tokens": {"input": 0, "output": 0, "reasoning": 0, "cache": {"read": 0, "write": 0}},
            "error": {
                "name": "APIError",
                "data": {"message": "model rejected request", "statusCode": 400, "isRetryable": False},
            },
        },
        "parts": [],
    }
    if after_work:
        export["messages"].append(rejected)
    else:
        export["messages"][-1] = rejected
    return json.dumps(export)


def read_params(text: str) -> dict:
    """Source runner PARAMS with bash, as the sandbox runner does."""
    script = 'source /dev/stdin; printf "%s\\0" "$cwd" "$stdin" "$setup" "${#command[@]}" "${command[@]}" "${env[@]}"'
    fields = subprocess.run(["bash", "-c", script], input=text, capture_output=True, text=True, check=True).stdout
    cwd, stdin, setup, count, *rest = fields.split("\0")[:-1]
    count = int(count)
    env = dict(item.split("=", 1) for item in rest[count:])
    return {"cwd": cwd, "stdin": stdin, "setup": setup, "command": rest[:count], "env": env}


class Sandbox:
    def __init__(self):
        self.sandbox_id = "provider-sandbox-id"
        self.provider_name = "opensandbox"
        self.files = {}
        self.snapshots = []
        self.result = {
            "return_code": 0,
            "timed_out": False,
            "cleanup_confirmed": True,
            "error": None,
        }
        self.runtime_info = {"hostname": "task-container", "pid": 123}
        self.task_environment_receipt = "ready\n"
        self.events = events()
        self.blocked = False
        self.started = asyncio.Event()
        self.exited = asyncio.Event()
        self.exec = AsyncMock(
            side_effect=self.run_exec,
            return_value=SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None),
        )
        self.stop = AsyncMock()
        self.disconnect = AsyncMock()
        self.launch = AsyncMock(side_effect=self.create)
        self.request_stop = AsyncMock(side_effect=self.signal)

    def _require_handle(self):
        return SandboxHandle(sandbox_id=self.sandbox_id, provider_name=self.provider_name, raw=None)

    @property
    def pty(self):
        raise AssertionError("This provider supports exec only; no PTY API")

    async def run_exec(self, command, **kwargs):
        if "--receipt" in command and "process_supervisor.sh" in command:
            return await self.launch(command=command, **kwargs)
        if "opencode.db-wal" in command:
            return SimpleNamespace(return_code=0, stdout="opencode.db\n", stderr="", error_type=None)
        if "stop.request" in command:
            await self.request_stop()
        if command.startswith("{ [ ! -d /tmp/nemo-gym-media/"):
            # stage_media renames its partial uploads onto their content-named paths in one command.
            words = shlex.split(command)
            for index, word in enumerate(words):
                if words[index : index + 3] == ["mv", "-f", "--"]:
                    source, destination = words[index + 3 : index + 5]
                    self.files[destination] = self.files.pop(source)
        return self.exec.return_value

    async def upload(self, source, destination):
        data = Path(source).read_bytes()
        try:
            self.files[destination] = data.decode()
        except UnicodeDecodeError:
            self.files[destination] = data

    async def download(self, source, destination):
        if source.endswith("/opencode.db") and source not in self.files:
            Path(destination).write_bytes(b"")
            return
        Path(destination).write_text(self.files[source])

    def params(self, directory=None):
        """The PARAMS and prompt staged for the runner in ``directory`` (default: the last launch)."""
        params = read_params(self.files[f"{directory or self.directory}/params.sh"])
        params["prompt"] = self.files[params["stdin"]]
        return params

    def snapshot(self, database, directory, *, harness_session_id, previous_message_ids):
        """Stand in for the SQLite export (see test_snapshot.py): the staged export, the copied database."""
        self.snapshots.append({"harness_session_id": harness_session_id, "previous_message_ids": previous_message_ids})
        (directory / "export.json").write_text(self.files[f"{self.directory}/export.json"])
        if database.stat().st_size:  # an empty database stands for a failed observation copy
            (directory / "observations.db").write_bytes(database.read_bytes())

    async def create(self, **kwargs):
        runner_params = next(arg for arg in shlex.split(kwargs["command"]) if arg.endswith("/params.sh"))
        self.directory = runner_params.removesuffix("/params.sh")
        assert self.params()["cwd"] == getattr(self, "expected_workdir", "/app")
        assert kwargs["cwd"] == getattr(self, "expected_workdir", "/app")
        assert "/harness_runner.sh " in kwargs["command"]
        self.started.set()
        if not self.blocked:
            self.exited.set()
        await self.wait_exit()
        return SimpleNamespace(error_type=None, return_code=0, stdout="", stderr="")

    async def signal(self):
        if hasattr(self, "directory"):
            self.result["timed_out"] = True
            self.exited.set()
            await asyncio.sleep(0)

    async def wait_exit(self):
        await self.exited.wait()
        self.files[f"{self.directory}/cleanup.json"] = json.dumps(self.result)
        self.files[f"{self.directory}/runtime.json"] = json.dumps(self.runtime_info)
        if self.task_environment_receipt is not None:
            self.files[f"{self.directory}/task-environment.ready"] = self.task_environment_receipt
        self.files[f"{self.directory}/export.json"] = self.events
        return 0


@pytest.fixture
def setup():
    sandbox = Sandbox()
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = OmegaConf.create(
        {"policy": {"responses_api_models": {"openai_model": {"host": "model.example", "port": 9000}}}}
    )
    client._build_server_base_url.return_value = "http://model.example:9000"
    config = OpenCodeAgentConfig(
        name="opencode",
        host="localhost",
        port=8001,
        entrypoint="app.py",
        num_workers=1,
        model_server={"type": "responses_api_models", "name": "policy"},
        opencode_version="1.17.11",
        resources_server={"type": "resources_servers", "name": "resources"},
        sandbox_provider="unused",
        sandbox_config={},
        timeout=30,
        context_window=32000,
        max_output_tokens=4096,
        # Bounds every close-phase step, including the artifact capture each HTTP-flow test runs; 1 s flaked on loaded
        # CI runners. A test of the close deadline itself sets agent.config.session_close_timeout_seconds locally.
        session_close_timeout_seconds=30,
    )
    module = "responses_api_agents.opencode_agent.app"
    with (
        patch(
            f"{module}.ensure_opencode", side_effect=AssertionError("Sandbox sessions must not install on the host")
        ),
        patch(f"{module}.resolve_provider_config"),
        patch(f"{module}.get_global_config_dict", return_value={}),
        patch(f"{module}.create_provider") as provider_factory,
        patch(f"{module}.AsyncSandbox.connect", AsyncMock(return_value=sandbox)),
        patch("responses_api_agents.opencode_agent.sandbox.snapshot", sandbox.snapshot),
    ):
        provider_factory.return_value.name = "opensandbox"
        agent = OpenCodeAgent(config=config, server_client=client)
        yield agent, sandbox


def active_sessions(agent):
    return {key: record.state for key, record in agent._session_records.items() if record.state is not None}


@pytest.mark.parametrize("stage", ["prepare", "install"])
@pytest.mark.parametrize("error_type", ["timeout", "sandbox"])
@pytest.mark.parametrize("owned", [False, True])
async def test_provider_error_type_blocks_setup_even_with_zero_exit(setup, stage, error_type, owned):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}})
    body = seed()
    if owned:
        agent.config.sandbox_provider = "sandbox"
        agent.config.sandbox_config = {"image": "test-image", "workdir": "/app"}
        body = body.model_copy(update={"sandbox_access": None})
        sandbox.start = AsyncMock()
    ok = SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")
    error = SimpleNamespace(return_code=0, error_type=error_type, stdout="bootstrap output", stderr="provider failed")
    sandbox.exec.side_effect = ([ok] if owned else []) + ([error, ok] if stage == "prepare" else [ok, error, ok])
    with patch("responses_api_agents.opencode_agent.app.AsyncSandbox", return_value=sandbox) as sandbox_class:
        sandbox_class.connect = AsyncMock(return_value=sandbox)
        with pytest.raises(NgFailureEvent, match=f"error_type={error_type}") as failed:
            await agent.seed_agent_session(request, body)
    assert failed.value.boundary == "harness"
    assert failed.value.failure_kind == "agent_route_failed"
    assert failed.value.operation == "opencode.setup"
    assert "bootstrap output" in str(failed.value)
    assert "provider failed" in str(failed.value)
    assert "return_code=0" in failed.value.failure_reason
    assert not request.session
    sandbox.launch.assert_not_awaited()
    if owned:
        sandbox.start.assert_awaited_once()
        sandbox.stop.assert_awaited_once()
    else:
        sandbox.disconnect.assert_awaited_once()
        sandbox.stop.assert_not_awaited()
    assert not any(record.state is not None for record in agent._session_records.values())


def test_setup_failure_retains_status_and_bounded_output_tails():
    result = SimpleNamespace(
        return_code=17,
        error_type=None,
        stdout="discard stdout prefix" + "x" * 4000 + "stdout end",
        stderr="discard stderr prefix" + "y" * 4000 + "stderr end",
    )
    with pytest.raises(NgFailureEvent) as failed:
        OpenCodeAgent._check_sandbox_setup("install runtime", result)
    assert failed.value.stage == "seed"
    reason = failed.value.failure_reason
    assert "return_code=17; error_type=None" in reason
    assert "stdout end" in reason and "stderr end" in reason
    assert "discard stdout prefix" not in reason and "discard stderr prefix" not in reason
    assert len(reason) < 2000


def close_body(session_id):
    return {"agent_session_id": session_id, "episode_id": seed().episode_id.model_dump()}


def capture_observations(sandbox, tmp_path):
    """Persist the same assistant turns used by the fake runner as real SQLite artifacts."""
    database = tmp_path / "observations.db"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            create table session(id text, parent_id text, time_created integer);
            create table message(id text, session_id text, data text, time_created integer);
            create table part(id text, message_id text, session_id text, data text, time_created integer);
            insert into session values('root', null, 0);
        """)
        for index, message in enumerate(json.loads(sandbox.events)["messages"]):
            message_id = f"message-{index}"
            connection.execute(
                "insert into message values (?, ?, ?, ?)",
                (message_id, "root", json.dumps(message["info"]), index),
            )
            for part_index, part in enumerate(message["parts"]):
                connection.execute(
                    "insert into part values (?, ?, ?, ?, ?)",
                    (f"part-{index}-{part_index}", message_id, "root", json.dumps(part), part_index),
                )
    original = sandbox.download

    async def download(source, destination):
        if source.endswith("/opencode.db"):
            Path(destination).write_bytes(database.read_bytes())
        else:
            await original(source, destination)

    sandbox.download = download
    return database


PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
PNG_URI = "data:image/png;base64," + base64.b64encode(PNG).decode()
JPEG = b"\xff\xd8\xff\xe0\x00\xfe"
JPEG_URI = "data:image/jpeg;base64," + base64.b64encode(JPEG).decode()


def count_media_decodes(monkeypatch):
    """Count base64 data-URI decodes, which should happen once per image in each request."""
    decodes = []
    decode = MediaFile.from_data_uri.__func__

    def counting(cls, uri):
        decodes.append(uri)
        return decode(cls, uri)

    monkeypatch.setattr(MediaFile, "from_data_uri", classmethod(counting))
    return decodes


def test_session_attaches_prompt_images_and_declares_image_input(setup, monkeypatch):
    agent, sandbox = setup
    decodes = count_media_decodes(monkeypatch)
    prompt = [
        {"type": "input_text", "text": "What is shown?"},
        {"type": "input_image", "image_url": PNG_URI, "detail": "auto"},
        {"type": "input_image", "image_url": JPEG_URI, "detail": "auto"},
    ]
    body = {"input": [{"role": "user", "content": prompt}]}
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json=body)
        assert result.status_code == 200, result.text
        # A retry replays the activation without decoding its images again.
        retried = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json=body)
        assert retried.status_code == 200, retried.text
        assert decodes == [PNG_URI, JPEG_URI]
        payload = sandbox.params()
        assert payload["prompt"] == "What is shown?"
        command = payload["command"]
        paths = [command[index + 1] for index, arg in enumerate(command) if arg == "--file"]
        # The path reaches the model's prompt, so it depends only on the image.
        assert paths == [
            f"/tmp/nemo-gym-media/{hashlib.sha256(PNG).hexdigest()}.png",
            f"/tmp/nemo-gym-media/{hashlib.sha256(JPEG).hexdigest()}.jpg",
        ]
        assert [sandbox.files[path] for path in paths] == [PNG, JPEG]
        assert not [path for path in sandbox.files if path.endswith(".partial")]
        config = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
        model = config["provider"]["nemo_gym"]["models"]["dummy_model"]
        assert model["modalities"] == {"input": ["text", "image"], "output": ["text"]}
        assert config["attachment"] == UNRESTRICTED_IMAGE_ATTACHMENT
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text


@pytest.mark.parametrize("image_input", [False, True])
def test_text_prompt_declares_image_input_only_when_configured(setup, image_input):
    agent, sandbox = setup
    agent.config.image_input = image_input
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == 200, result.text
        config = json.loads(sandbox.params()["env"]["OPENCODE_CONFIG_CONTENT"])
        # Without the declaration OpenCode replaces tool images with an error text, which a text-only model accepts.
        assert ("modalities" in config["provider"]["nemo_gym"]["models"]["dummy_model"]) is image_input
        # Image limits are lifted only with the declaration, so the model gets each image's own bytes.
        assert config.get("attachment") == (UNRESTRICTED_IMAGE_ATTACHMENT if image_input else None)
        client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))


def test_session_attachment_config_restores_image_limits(setup):
    agent, sandbox = setup
    agent.config.image_input = True
    agent.config.opencode_config = {"attachment": {"image": {"max_width": 4096}}}
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == 200, result.text
        config = json.loads(sandbox.params()["env"]["OPENCODE_CONFIG_CONTENT"])
        # The session's setting replaces the lifted limits; OpenCode applies its defaults to the rest.
        assert config["attachment"] == {"image": {"max_width": 4096}}
        client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))


def test_session_close_stops_media_staging_before_releasing_the_sandbox(setup):
    agent, sandbox = setup
    events = []
    started = threading.Event()
    upload = sandbox.upload

    async def stall_media(source, destination):
        if "/nemo-gym-media/" not in destination:
            return await upload(source, destination)
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            events.append("upload cancelled")
            raise

    async def released(*args, **kwargs):
        events.append("sandbox released")

    sandbox.upload = stall_media
    sandbox.stop.side_effect = released
    sandbox.disconnect.side_effect = released
    prompt = [
        {"type": "input_text", "text": "What is shown?"},
        {"type": "input_image", "image_url": PNG_URI, "detail": "auto"},
    ]
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        activation = threading.Thread(
            target=client.post,
            args=("/ng-rollout/opencode-smoke-a2/v1/responses",),
            kwargs={"json": {"input": [{"role": "user", "content": prompt}]}},
        )
        activation.start()
        assert started.wait(10)
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text
        activation.join(10)
        assert not activation.is_alive()
    # Staging belongs to the activation, so close stops it before giving the sandbox back.
    assert events[:2] == ["upload cancelled", "sandbox released"], events
    assert not sandbox.launch.called


def test_first_activation_budget_bounds_media_staging(setup):
    agent, sandbox = setup
    upload = sandbox.upload
    stalled = SimpleNamespace(cancelled=False)

    async def stall_media(source, destination):
        if "/nemo-gym-media/" not in destination:
            return await upload(source, destination)
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            stalled.cancelled = True
            raise

    sandbox.upload = stall_media
    prompt = [
        {"type": "input_text", "text": "What is shown?"},
        {"type": "input_image", "image_url": PNG_URI, "detail": "auto"},
    ]
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        started = monotonic()
        result = client.post(
            "/ng-rollout/opencode-smoke-a2/v1/responses",
            json={"input": [{"role": "user", "content": prompt}], "metadata": {"agent_timeout_sec": "0.5"}},
        )
        # The budget, not the stalled upload, ends the activation, and its slot is free for the next one.
        assert monotonic() - started < 10
        assert result.status_code == 500, result.text
        assert stalled.cancelled
        assert agent.sem._value == agent.config.concurrency
        client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
    assert not sandbox.launch.called


@pytest.mark.parametrize(
    "items",
    [
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is shown?"},
                    {"type": "input_image", "image_url": "https://example.com/a.png", "detail": "auto"},
                ],
            }
        ],
        [
            {"role": "system", "content": [{"type": "input_image", "image_url": PNG_URI, "detail": "auto"}]},
            {"role": "user", "content": "task"},
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": " "},
                    {"type": "input_image", "image_url": PNG_URI, "detail": "auto"},
                ],
            }
        ],
    ],
    ids=["url", "system-media", "image-without-text"],
)
def test_session_rejects_images_it_cannot_attach(setup, items):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": items})
        assert result.status_code == 422, result.text
        # A plain string: request validation would return a list of errors before the agent runs.
        assert result.json()["detail"].startswith("OpenCode sandbox execution"), result.text
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text
    assert not sandbox.launch.called


@pytest.mark.parametrize("output_budget", [4096, 32000])
def test_http_session_flow_runs_opencode_in_borrowed_sandbox(setup, output_budget):
    agent, sandbox = setup
    agent.config.max_output_tokens = output_budget
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        session_id = created.json()["agent_session_id"]
        assert not sandbox.launch.called
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "Fix the code"})
        assert result.status_code == 200, result.text
        body = result.json()
        assert body["status"] == "completed"
        assert [item["type"] for item in body["output"]] == [
            "reasoning",
            "function_call",
            "function_call_output",
            "message",
        ]
        assert body["usage"]["total_tokens"] == 16
        payload = sandbox.params()
        assert payload["prompt"] == "Fix the code"
        assert payload["command"][0].endswith("nemo-gym-opencode-runtime-1.17.11/opencode")
        assert payload["env"]["HOME"] == f"{sandbox.directory}/home"
        config = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
        assert config["provider"]["nemo_gym"]["models"]["dummy_model"]["limit"] == {
            "context": 32000,
            "input": 32000,
            "output": output_budget,
        }
        assert payload["env"]["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] == str(output_budget)
        assert (
            config["provider"]["nemo_gym"]["options"]["baseURL"]
            == "http://model.example:9000/ng-rollout/opencode-smoke-a2/v1"
        )
        # As on opencode_sandboxed_agent with opencode_model_call_timeout null, only the run's own deadline bounds a
        # model call. Here the provider timeout is off and there is no chunkTimeout; the relay's chunkTimeout is its
        # run budget.
        options = config["provider"]["nemo_gym"]["options"]
        assert options["timeout"] is False
        assert "chunkTimeout" not in options
        assert client.post("/v1/responses", json={"input": "task"}).status_code == 409
        assert client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"}).status_code == 409
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text
    assert not active_sessions(agent)
    assert not any(path.startswith("/app/") for path in sandbox.files)
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()
    agent.server_client.post.assert_not_called()


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("origin_stage", [None, "agent", "seed", "verification"])
def test_confirmed_agent_execution_oom_retains_sandbox_id_and_closes_dead_session(setup, owned, origin_stage):
    agent, sandbox = setup
    agent.config.report_agent_sandbox_oom = True
    body = seed()
    if owned:
        agent.config.sandbox_provider = "agent-provider"
        agent.config.sandbox_config = {"image": "test-image", "workdir": "/app"}
        body.sandbox_access = None
        sandbox.sandbox_id = "agent-owned"
        sandbox.start = AsyncMock()
    error = sandbox_oom(
        f"Sandbox was OOM-killed. OpenSandbox status: reason='OOMKilled'; sandbox_id='{sandbox.sandbox_id}'",
        sandbox_id=sandbox.sandbox_id,
    )
    error.stage = origin_stage
    sandbox.launch.side_effect = error

    webapp = agent.setup_webserver()
    agent.setup_exception_middleware(webapp)
    module = "responses_api_agents.opencode_agent.app"
    with (
        patch(f"{module}.AsyncSandbox", return_value=sandbox) as factory,
        TestClient(webapp) as client,
    ):
        factory.connect = AsyncMock(return_value=sandbox)
        created = client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
        assert created.status_code == 200, created.text
        response = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "Fix the code"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "failed"
        assert body["output"] == []
        from nemo_gym.agents import AgentOutcome, AgentOutcomePolicyConfig, OutcomeAction, agent_outcome_policy
        from nemo_gym.core import NeMoGymResponse

        outcome = AgentOutcome.model_validate_json(body["metadata"][AGENT_OUTCOME_METADATA_KEY])
        assert outcome.failure_kind == PROVIDER_OOM_KILLED
        assert outcome.status_code is None
        typed_response = NeMoGymResponse.model_validate(body)
        assert agent_outcome_policy(typed_response) == OutcomeAction.VERIFY
        for action in (OutcomeAction.FAIL, OutcomeAction.RETRY):
            assert (
                agent_outcome_policy(
                    typed_response, config=AgentOutcomePolicyConfig(by_failure_kind={PROVIDER_OOM_KILLED: action})
                )
                == action
            )
        observation = SandboxObservation.model_validate_json(body["metadata"][NG_AGENT_SANDBOX_OBSERVATION_KEY])
        assert observation.role == "agent"
        assert observation.outcome == "oom"
        assert observation.stage == (origin_stage or "agent")
        assert "phase" not in json.loads(body["metadata"][NG_AGENT_SANDBOX_OBSERVATION_KEY])
        assert observation.provider == "opensandbox"
        assert observation.sandbox_id == sandbox.sandbox_id
        assert observation.error_type is None
        from nemo_gym.core import FailureHeader

        report = FailureHeader.decode(response.headers)
        assert report.failure_id is None
        assert report.events[observation.failure_id].failure_kind == PROVIDER_OOM_KILLED
        assert observation.failure_id == error.event_id
        assert report.events[observation.failure_id].stage == origin_stage
        assert observation.error_message is None

        retried = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "Fix the code"})
        assert retried.status_code == 200
        retry_report = FailureHeader.decode(retried.headers)
        assert set(retry_report.events) == set(report.events)
        assert retry_report.failure_id is None
        assert retried.json() == body
        exec_count_at_failure = sandbox.exec.await_count
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text
        assert observation.model_dump() in closed.json()["agent_observations"]["records"]
        assert any(
            record["kind"] == "agent_invocation" and record["invocation_id"] == "opencode-smoke-a2"
            for record in closed.json()["agent_observations"]["records"]
        )
        assert any(gap["code"] == "observation_capture_failed" for gap in closed.json()["agent_observations"]["gaps"])

    assert sandbox.launch.await_count == 1
    assert sandbox.exec.await_count == exec_count_at_failure
    assert not any("rm -rf" in call.args[0] for call in sandbox.exec.await_args_list)
    if owned:
        sandbox.stop.assert_awaited_once()
        sandbox.disconnect.assert_not_awaited()
    else:
        sandbox.disconnect.assert_awaited_once()
        sandbox.stop.assert_not_awaited()


async def test_agent_execution_oom_reporting_is_opt_in(setup):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}, "path_params": {"rollout_id": "opencode-smoke-a2"}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    error = sandbox_oom("OpenSandbox reason='OOMKilled'; sandbox_id='resources-owned'")
    sandbox.launch.side_effect = error

    with pytest.raises(NgFailureEvent) as caught:
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))

    assert caught.value is error
    closed = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert not any(isinstance(record, SandboxObservation) for record in closed.agent_observations.records)
    sandbox.disconnect.assert_awaited_once()


def test_http_splits_the_reasoning_block_a_gym_reply_starts_with(setup, tmp_path):
    # The Gym Model Server returns a reply's reasoning ahead of its text as <think>...</think>, which OpenCode stores
    # as a text part; graded output and observations carry it as reasoning, not as the answer.
    agent, sandbox = setup
    export = json.loads(sandbox.events)
    assistant = export["messages"][1]
    tool = assistant["parts"][1]
    assistant["parts"] = [
        {"type": "text", "text": "<think>run pwd</think>"},
        tool,
        {"type": "text", "text": "<think>it printed /app</think>\n\nAnswer: /app"},
    ]
    sandbox.events = json.dumps(export)
    capture_observations(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == 200, result.text
        output = result.json()["output"]
        assert [item["type"] for item in output] == [
            "reasoning",
            "function_call",
            "function_call_output",
            "reasoning",
            "message",
        ]
        assert output[0]["summary"][0]["text"] == "run pwd"
        assert output[3]["summary"][0]["text"] == "it printed /app"
        assert output[4]["content"][0]["text"] == "Answer: /app"
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text
    invocation = next(
        record for record in closed.json()["agent_observations"]["records"] if record["kind"] == "agent_invocation"
    )
    persisted = [item for item in invocation["conversation"] if item.get("role") != "user"]
    assert [item["type"] for item in persisted] == [item["type"] for item in output]
    assert persisted[4]["content"][0]["text"] == "Answer: /app"


async def test_http_keeps_an_empty_answer_when_the_final_gym_reply_is_only_reasoning(setup, tmp_path):
    # Without the empty final message, the judge's backward search would grade the earlier answer "4".
    from responses_api_agents.opencode_agent.tests.test_app import judge_final_answer

    agent, sandbox = setup
    export = json.loads(sandbox.events)
    assistant = export["messages"][1]
    tool = assistant["parts"][1]
    assistant["info"]["finish"] = "tool-calls"
    assistant["parts"] = [{"type": "text", "text": "4"}, {"type": "text", "text": "<think>run pwd</think>"}, tool]
    final = {"info": {**assistant["info"], "finish": "stop"}, "parts": [{"type": "text", "text": "<think>r</think>"}]}
    export["messages"].append(final)
    sandbox.events = json.dumps(export)
    capture_observations(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == 200, result.text
        output = result.json()["output"]
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text
    assert [item["type"] for item in output] == [
        "message",
        "reasoning",
        "function_call",
        "function_call_output",
        "reasoning",
        "message",
    ]
    assert output[0]["content"][0]["text"] == "4"
    assert output[-1]["content"][0]["text"] == ""
    assert await judge_final_answer(output) == (0.0, "empty_final_answer", False)
    invocation = next(
        record for record in closed.json()["agent_observations"]["records"] if record["kind"] == "agent_invocation"
    )
    # The observed conversation keeps the same empty answer for the final reply.
    observed = [item for item in invocation["conversation"] if item.get("id") == "part-2-0"]
    assert [(item["type"], item["content"][0]["text"]) for item in observed] == [("message", "")]


@pytest.mark.parametrize("part_type", ["text", "reasoning"])
@pytest.mark.parametrize("text", ["literal <think> opening", "literal </think> closing", " <think>x</think> \n"])
def test_http_preserves_typed_literal_think_tags_and_unique_ids(setup, tmp_path, part_type, text):
    agent, sandbox = setup
    export = json.loads(sandbox.events)
    assistant = export["messages"][1]
    tool = assistant["parts"][1]
    assistant["parts"] = [{"type": part_type, "text": text}, tool, {"type": part_type, "text": text}]
    sandbox.events = json.dumps(export)
    capture_observations(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == 200, result.text
        body = result.json()
        assert body["status"] == "completed"
        output = body["output"]
        expected_type = "message" if part_type == "text" else "reasoning"
        text_key = "content" if part_type == "text" else "summary"
        assert [item["type"] for item in output] == [
            expected_type,
            "function_call",
            "function_call_output",
            expected_type,
        ]
        assert output[0][text_key][0]["text"] == output[3][text_key][0]["text"] == text
        assert output[0]["id"] != output[3]["id"]
        assert output[1]["call_id"] == output[2]["call_id"] == "tool-1"
        assert output[1]["name"] == "bash"
        assert json.loads(output[1]["arguments"]) == {"command": "pwd"}
        assert output[2]["output"] == "/app"
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text
    invocation = next(
        record for record in closed.json()["agent_observations"]["records"] if record["kind"] == "agent_invocation"
    )
    persisted_parts = [item for item in invocation["conversation"] if item.get("role") != "user"]
    assert [item["type"] for item in persisted_parts] == [item["type"] for item in output]
    assert persisted_parts[0][text_key][0]["text"] == persisted_parts[3][text_key][0]["text"] == text
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize(
    "override",
    [
        {"max_output_tokens": 123},
        {"top_logprobs": 2},
        {"reasoning": {"effort": "high"}},
        {"tools": [{"type": "function", "name": "foo"}]},
        {"input": [{"role": "assistant", "content": "old turn"}]},
    ],
)
def test_unsupported_request_is_not_silently_ignored(setup, override):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task", **override})
        assert result.status_code == 422, result.text
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200
    sandbox.launch.assert_not_awaited()


@pytest.mark.parametrize("limit", [4096, 16384])
def test_a_nemo_rl_row_output_limit_passes_when_opencode_stays_within_it(setup, limit):
    """NeMo RL sets max_output_tokens on every row. OpenCode's own limit for each model call (4096 here) obeys it."""
    agent, sandbox = setup
    row = {"input": "task", "temperature": 1.0, "top_p": 1.0, "max_output_tokens": limit}
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json=row)
        assert result.status_code == 200, result.text
    assert sandbox.params()["env"]["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] == "4096"


def test_rejected_request_does_not_consume_activation(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        path = "/ng-rollout/opencode-smoke-a2/v1/responses"
        assert client.post(path, json={"input": "task", "max_output_tokens": 5}).status_code == 422
        sandbox.launch.assert_not_awaited()
        accepted = client.post(path, json={"input": "task"})
        assert accepted.status_code == 200, accepted.text
        assert client.post(path, json={"input": "task"}).json() == accepted.json()
        assert client.post(path, json={"input": "changed"}).status_code == 409
    sandbox.launch.assert_awaited_once()


async def activate(agent, sandbox):
    request = Request(
        {"type": "http", "headers": [], "session": {}, "path_params": {"rollout_id": "opencode-smoke-a2"}}
    )
    seeded = await agent.seed_agent_session(request, seed())
    task = asyncio.create_task(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task")))
    await asyncio.wait_for(sandbox.started.wait(), 2)
    return request, seeded.agent_session_id, task


async def test_close_cancels_active_opencode_before_detaching(setup, tmp_path):
    agent, sandbox = setup
    sandbox.blocked = True
    capture_observations(sandbox, tmp_path)
    request, session_id, task = await activate(agent, sandbox)
    closed = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    with pytest.raises(asyncio.CancelledError):
        await task
    sandbox.request_stop.assert_awaited_once()
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()
    invocation = next(record for record in closed.agent_observations.records if record.kind == "agent_invocation")
    assert invocation.status == "incomplete"
    assert invocation.error_type == "cancelled"
    assert invocation.conversation[-1].content[0].text == "Fixed"


async def test_close_reports_the_cancel_on_the_sandbox_and_running_tools(setup, tmp_path):
    agent, sandbox = setup
    sandbox.blocked = True
    export = json.loads(sandbox.events)
    export["messages"][-1]["parts"].append(
        {
            "type": "tool",
            "callID": "running-shell",
            "tool": "bash",
            "state": {"status": "running", "input": {"command": "sleep 60"}, "metadata": {"output": "started"}},
        }
    )
    sandbox.events = json.dumps(export)
    capture_observations(sandbox, tmp_path)

    async def stop_requested():
        # The process supervisor reports a requested stop as a forced exit, not as a timeout.
        sandbox.result["return_code"] = 241
        sandbox.exited.set()
        await asyncio.sleep(0)

    sandbox.request_stop.side_effect = stop_requested
    request, session_id, task = await activate(agent, sandbox)
    closed = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    with pytest.raises(asyncio.CancelledError):
        await task
    records = closed.agent_observations.records
    agent_sandbox = next(record for record in records if record.kind == "sandbox")
    assert agent_sandbox.stage == "agent"
    assert (agent_sandbox.outcome, agent_sandbox.error_type, agent_sandbox.exit_code) == (
        "cancelled",
        "cancelled",
        241,
    )
    tools = {
        record.native_tool_call_id or record.tool_call_id: record for record in records if record.kind == "tool_call"
    }
    assert (tools["running-shell"].status, tools["running-shell"].error_type) == ("incomplete", None)
    assert tools["tool-1"].status != "cancelled"
    invocation = next(record for record in records if record.kind == "agent_invocation")
    assert {
        "type": "function_call_output",
        "call_id": tools["running-shell"].tool_call_id,
        "output": "started",
    }.items() <= next(
        item.model_dump()
        for item in invocation.conversation
        if getattr(item, "call_id", None) == tools["running-shell"].tool_call_id
        and item.type == "function_call_output"
    ).items()


async def test_terminal_termination_failure_releases_borrowed_session(setup):
    agent, sandbox = setup
    sandbox.result["cleanup_confirmed"] = False
    request, session_id, task = await activate(agent, sandbox)
    with pytest.raises(NgFailureEvent, match="harness termination was not confirmed") as failure:
        await task
    assert is_harness_termination_failure(failure.value)
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert session_id not in active_sessions(agent)
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()
    assert any("rm -rf" in call.args[0] for call in sandbox.exec.await_args_list)


async def test_disconnect_failure_retains_session_for_retry(setup):
    agent, sandbox = setup
    request, session_id, task = await activate(agent, sandbox)
    await task
    sandbox.disconnect.side_effect = [RuntimeError("provider unavailable"), None]
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert session_id in active_sessions(agent)
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert session_id not in active_sessions(agent)
    sandbox.launch.assert_awaited_once()


def test_cleanup_receipt_is_required():
    with pytest.raises(ValueError):
        parse_cleanup_receipt({"return_code": 0, "error": None})


@pytest.mark.parametrize("mode", ["append", "replace"])
def test_instructions_and_text_parts_reach_opencode(setup, mode):
    agent, sandbox = setup
    agent.config.system_prompt_mode = mode
    agent.config.opencode_config = {"tools": {"*": False}, "permission": {"bash": "deny"}}
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        result = client.post(
            "/ng-rollout/opencode-smoke-a2/v1/responses",
            json={
                "instructions": "outer",
                "input": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": [{"type": "input_text", "text": "task"}]},
                ],
            },
        )
        assert result.status_code == 200, result.text
        payload = sandbox.params()
        config = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
        assert config["tools"] == {"*": False}
        assert config["permission"] == {"bash": "deny"}
        if mode == "replace":
            assert payload["command"][payload["command"].index("--agent") + 1] == "build"
            assert config["agent"]["build"]["prompt"] == "outer\n\nsystem"
            assert "instructions" not in config
            assert f"{sandbox.directory}/instructions.md" not in sandbox.files
        else:
            assert "--agent" not in payload["command"]
            assert sandbox.files[config["instructions"][0]] == "outer\n\nsystem"
        assert payload["prompt"] == "task"
        assert config["enabled_providers"] == ["nemo_gym"]


@pytest.mark.parametrize("mode", ["append", "replace"])
def test_compaction_and_build_steps_reach_opencode(setup, mode):
    agent, sandbox = setup
    agent.config.system_prompt_mode = mode
    agent.config.system_prompt = "Answer briefly"
    agent.config.opencode_config = {"compaction": {"auto": False}, "agent": {"build": {"steps": 8}}}
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == 200, result.text
        config = json.loads(sandbox.params()["env"]["OPENCODE_CONFIG_CONTENT"])
    assert config["compaction"] == {"auto": False}
    if mode == "replace":
        # The replacement prompt joins the configured step limit instead of replacing the agent entry.
        assert config["agent"] == {"build": {"steps": 8, "prompt": "Answer briefly"}}
    else:
        assert config["agent"] == {"build": {"steps": 8}}
    # The session's prompt stays out of the configuration that later sessions copy.
    assert agent.config.opencode_config == {"compaction": {"auto": False}, "agent": {"build": {"steps": 8}}}


def test_compaction_reserved_reaches_opencode(setup):
    """OpenCode then compacts after a call of 32000 - 8000 tokens, instead of 32000 - min(20000, 4096)."""
    agent, sandbox = setup
    agent.config.opencode_config = {"compaction": {"auto": True, "reserved": 8000}}
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == 200, result.text
        config = json.loads(sandbox.params()["env"]["OPENCODE_CONFIG_CONTENT"])
    assert config["compaction"] == {"auto": True, "reserved": 8000}
    assert config["provider"]["nemo_gym"]["models"]["dummy_model"]["limit"]["input"] == 32000


def test_request_sampling_joins_the_configured_build_steps(setup):
    agent, sandbox = setup
    agent.config.opencode_config = {"agent": {"build": {"steps": 8}}}
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task", "temperature": 0.6})
        assert result.status_code == 200, result.text
        config = json.loads(sandbox.params()["env"]["OPENCODE_CONFIG_CONTENT"])
    assert config["agent"]["build"] == {"steps": 8, "temperature": 0.6}
    assert config["agent"]["plan"] == {"temperature": 0.6}
    # The request's sampling stays out of the configuration that later sessions copy.
    assert agent.config.opencode_config == {"agent": {"build": {"steps": 8}}}


def step_events(model_calls: int) -> str:
    """A prompt that made ``model_calls`` model calls: tool calls, then a final answer."""
    export = json.loads(events())
    answer = export["messages"][-1]
    tool_call = {**answer, "info": {**answer["info"], "finish": "tool-calls"}, "parts": answer["parts"][:2]}
    export["messages"] = [export["messages"][0], *[tool_call] * (model_calls - 1), answer]
    return json.dumps(export)


@pytest.mark.parametrize(
    ("steps", "model_calls", "marked"),
    [(3, 3, True), (3, 4, True), (3, 9, True), (4, 3, False), (None, 3, False)],
)
def test_step_limit_marks_the_response(setup, steps, model_calls, marked):
    """A prompt that made at least agent.build.steps model calls reached OpenCode's step limit; the response says so."""
    agent, sandbox = setup
    agent.config.opencode_config = {"agent": {"build": {"steps": steps}}} if steps is not None else {}
    sandbox.events = step_events(model_calls)
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["status"] == "completed"
    assert body["metadata"].get("harness_step_limit_reached") == ("true" if marked else None)


def test_step_limit_counts_only_the_last_prompt():
    from responses_api_agents.opencode_agent.app import reached_step_limit

    user, assistant = {"info": {"role": "user"}}, {"info": {"role": "assistant"}}
    export = {"messages": [user, assistant, assistant, assistant, user, assistant]}
    assert not reached_step_limit(export, 2)
    assert reached_step_limit(export, 1)
    assert not reached_step_limit({"messages": []}, 1)
    assert not reached_step_limit(export, None)


# Message shapes OpenCode 1.17.11 saves around an automatic compaction (packages/opencode/src/session/compaction.ts).
_PROMPT = {"info": {"role": "user"}, "parts": [{"type": "text", "text": "task"}]}
_BUILD = {"info": {"role": "assistant", "finish": "tool-calls"}}
_ANSWER = {"info": {"role": "assistant", "finish": "stop"}}
_SUMMARY = {"info": {"role": "assistant", "summary": True, "mode": "compaction"}}
_CONTINUE = {
    "info": {"role": "user"},
    "parts": [{"type": "text", "synthetic": True, "metadata": {"compaction_continue": True}, "text": "Continue"}],
}


def _compaction(**part: object) -> dict:
    return {"info": {"role": "user"}, "parts": [{"type": "compaction", "auto": True, **part}]}


@pytest.mark.parametrize(
    ("messages", "passes"),
    [
        # The loop starts a compaction in a pass of its own when the last finished call overflowed: no overflow field.
        ([_PROMPT, _BUILD, _BUILD, _compaction(), _SUMMARY, _CONTINUE, _ANSWER], 5),
        # A build call that overflows starts the compaction itself (overflow set), and replays the prompt afterwards.
        ([_PROMPT, _BUILD, _BUILD, _compaction(overflow=True), _SUMMARY, _PROMPT, _ANSWER], 4),
        ([_PROMPT, _BUILD, _BUILD, _compaction(overflow=False), _SUMMARY, _CONTINUE, _ANSWER], 4),
        # Without a continuing user message, as in an older OpenCode.
        ([_PROMPT, _BUILD, _compaction(), _SUMMARY, _ANSWER], 4),
        # A new prompt after a compacted one starts the count again.
        ([_PROMPT, _BUILD, _compaction(), _SUMMARY, _CONTINUE, _ANSWER, _PROMPT, _ANSWER], 1),
    ],
)
def test_step_limit_counts_compaction_passes_and_keeps_the_prompt(messages, passes):
    """OpenCode's step counter counts compaction passes and does not restart at the user messages compaction adds."""
    from responses_api_agents.opencode_agent.app import reached_step_limit

    assert reached_step_limit({"messages": messages}, passes)
    assert not reached_step_limit({"messages": messages}, passes + 1)


def compacted_events() -> str:
    """events() with an automatic compaction before the final answer, in the message shapes OpenCode 1.17.11 saves.
    Every message's info is also in usage_messages, as in the export opencode_agent writes, which usage counts from."""
    prompt, answer = json.loads(events())["messages"]
    summary = {
        "info": {
            "role": "assistant",
            "summary": True,
            "mode": "compaction",
            "agent": "compaction",
            "finish": "stop",
            "tokens": {"input": 100, "output": 20, "reasoning": 5, "cache": {"read": 7, "write": 0}},
        },
        "parts": [{"type": "text", "text": "Summary of the work so far"}],
    }
    messages = [prompt, _compaction(), summary, _CONTINUE, answer]
    return json.dumps({"messages": messages, "usage_messages": [message["info"] for message in messages]})


@pytest.mark.parametrize("exclude", [False, True])
def test_exclude_compaction_summaries_leaves_them_out_of_the_output(setup, exclude):
    agent, sandbox = setup
    agent.config.exclude_compaction_summaries = exclude
    sandbox.events = compacted_events()
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
    assert result.status_code == 200, result.text
    output = [
        (item["type"], item["content"][0]["text"] if item["type"] == "message" else None)
        for item in result.json()["output"]
    ]
    answer = [("reasoning", None), ("function_call", None), ("function_call_output", None), ("message", "Fixed")]
    assert output == (answer if exclude else [("message", "Summary of the work so far"), *answer])
    # Either way, usage counts the summary call (107 input, 25 output tokens) and the answer's (13, 3).
    usage = result.json()["usage"]
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (120, 28, 148)
    assert usage["input_tokens_details"]["cached_tokens"] == 7 + 3
    assert usage["output_tokens_details"]["reasoning_tokens"] == 5 + 1


@pytest.mark.parametrize("marker", [{"summary": True}, {"agent": "compaction"}])
def test_either_marker_identifies_a_compaction_summary(setup, marker):
    """OpenCode 1.17.11 sets both markers on a summary; either one alone still identifies it."""
    from nemo_gym.trajectory import AgentObservationBundle

    agent, _ = setup
    agent.config.exclude_compaction_summaries = True
    export = {
        "messages": [
            {"info": {"role": "assistant", **marker}, "parts": [{"type": "text", "text": "Summary"}]},
            {"info": {"role": "assistant", "agent": "build"}, "parts": [{"type": "text", "text": "Answer"}]},
        ]
    }
    output = agent._session_output(export, AgentObservationBundle(source="opencode"))
    assert [item.content[0].text for item in output] == ["Answer"]


@pytest.mark.parametrize(
    ("opencode_config", "detail"),
    [
        ({"provider": {"other": {}}}, "unsupported: ['provider']"),
        ({"compaction": {"auto": False, "prune": True}}, "compaction supports only auto and reserved"),
        ({"compaction": {"reserved": 8000, "prune": False}}, "compaction supports only auto and reserved"),
        ({"compaction": {"auto": "false"}}, "compaction.auto must be true or false"),
        ({"compaction": None}, "compaction supports only auto and reserved"),
        ({"compaction": {"reserved": 0}}, "compaction.reserved must be a positive integer"),
        ({"compaction": {"reserved": -8000}}, "compaction.reserved must be a positive integer"),
        ({"compaction": {"reserved": 8000.5}}, "compaction.reserved must be a positive integer"),
        ({"compaction": {"reserved": "8000"}}, "compaction.reserved must be a positive integer"),
        ({"compaction": {"reserved": True}}, "compaction.reserved must be a positive integer"),
        # The whole context window (32000 here) or more: OpenCode would compact after every call.
        ({"compaction": {"reserved": 32000}}, "compaction.reserved must be below context_window"),
        ({"compaction": {"reserved": 40000}}, "compaction.reserved must be below context_window"),
        ({"agent": {"plan": {"steps": 4}}}, "agent supports only build"),
        ({"agent": {"build": {"prompt": "other"}}}, "agent.build supports only steps"),
        ({"agent": {"build": {"steps": 0}}}, "agent.build.steps must be a positive integer"),
        ({"agent": {"build": {"steps": True}}}, "agent.build.steps must be a positive integer"),
    ],
)
def test_seed_rejects_opencode_config_outside_the_session_allowlist(setup, opencode_config, detail):
    agent, sandbox = setup
    agent.config.opencode_config = opencode_config
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
    assert response.status_code == 422
    assert detail in response.json()["detail"]
    sandbox.exec.assert_not_awaited()


def test_token_capture_routes_every_opencode_model_call_to_the_rollout(setup):
    agent, sandbox = setup
    agent.server_client.global_config_dict = OmegaConf.create(
        {
            "policy": {"responses_api_models": {"openai_model": {"host": "model.example", "port": 9000}}},
            "token_id_capture": {"enabled": True, "all_agents": True},
        }
    )
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        # An Environment Server activates a captured rollout on the tagged path; without the tag it would not capture.
        result = client.post(
            "/ng-rollout/opencode-smoke-a2/training-token-capture/v1/responses", json={"input": "task"}
        )
        assert result.status_code == 200, result.text
        config = json.loads(sandbox.params()["env"]["OPENCODE_CONFIG_CONTENT"])
    # OpenCode can reach only the Gym provider, for its main and small model alike.
    assert config["enabled_providers"] == ["nemo_gym"]
    assert config["model"].startswith("nemo_gym/") and config["small_model"].startswith("nemo_gym/")
    assert list(config["provider"]) == ["nemo_gym"]
    assert (
        config["provider"]["nemo_gym"]["options"]["baseURL"]
        == "http://model.example:9000/ng-rollout/opencode-smoke-a2/training-token-capture/v1"
    )


def test_with_token_capture_a_seed_accepts_build_steps(setup):
    agent, sandbox = setup
    # Capture recognizes OpenCode's step-limit note (harness_notes), so the step limit and capture go together.
    agent.server_client.global_config_dict = OmegaConf.create(
        {
            "policy": {"responses_api_models": {"openai_model": {"host": "model.example", "port": 9000}}},
            "token_id_capture": {"enabled": True, "all_agents": True},
        }
    )
    agent.config.opencode_config = {"compaction": {"auto": False}, "agent": {"build": {"steps": 8}}}
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()


def test_http_close_retry_survives_other_session_closes(setup, monkeypatch):
    agent, sandbox = setup
    monkeypatch.setattr("nemo_gym.agents.base_responses_api_agent.monotonic", lambda: 100.0)
    with TestClient(agent.setup_webserver()) as client:

        def seed_and_close(index):
            client.cookies.clear()
            body = seed().model_dump(mode="json")
            body["episode_id"] = {"rollout_id": f"episode-{index}"}
            created = client.post("/v1/agent_sessions", json=body)
            assert created.status_code == 200
            cookies = dict(client.cookies)
            close = {"agent_session_id": created.json()["agent_session_id"], "episode_id": body["episode_id"]}
            result = client.post("/v1/agent_sessions/close", json=close)
            assert result.status_code == 200
            return cookies, close, result.json()

        cookies, close, first = seed_and_close(0)
        for index in range(1, 66):
            seed_and_close(index)
        client.cookies.clear()
        client.cookies.update(cookies)
        retry = client.post("/v1/agent_sessions/close", json=close)
        assert retry.status_code == 200
        assert retry.json() == first
    assert sandbox.disconnect.await_count == 66


async def test_fence_expires_without_extending_on_retry(setup, monkeypatch):
    agent, sandbox = setup
    clock = [100.0]
    monkeypatch.setattr("nemo_gym.agents.base_responses_api_agent.monotonic", lambda: clock[0])
    agent.config.session_close_retry_window_seconds = 10
    request = Request({"type": "http", "session": {}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    close = AgentCloseSessionRequest(**close_body(session_id))
    first = await agent.close_agent_session(request, close)
    clock[0] = 109.0
    assert await agent.close_agent_session(request, close) == first
    clock[0] = 110.0
    with pytest.raises(HTTPException) as error:
        await agent.close_agent_session(request, close)
    assert error.value.status_code == 409
    assert not agent._fenced_session_records
    with pytest.raises(HTTPException) as error:
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert error.value.status_code == 409
    sandbox.disconnect.assert_awaited_once()


async def test_close_retry_window_starts_after_cleanup(setup, monkeypatch):
    agent, sandbox = setup
    clock = [100.0]
    monkeypatch.setattr("nemo_gym.agents.base_responses_api_agent.monotonic", lambda: clock[0])
    agent.config.session_close_retry_window_seconds = 10
    request = Request({"type": "http", "session": {}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id

    async def disconnect():
        clock[0] = 200.0

    sandbox.disconnect.side_effect = disconnect
    close = AgentCloseSessionRequest(**close_body(session_id))
    first = await agent.close_agent_session(request, close)
    clock[0] = 209.0
    assert await agent.close_agent_session(request, close) == first
    sandbox.disconnect.assert_awaited_once()


async def test_concurrent_closes_share_the_saved_reply(setup):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    entered, release = asyncio.Event(), asyncio.Event()

    async def disconnect():
        entered.set()
        await release.wait()

    sandbox.disconnect.side_effect = disconnect
    close = AgentCloseSessionRequest(**close_body(session_id))
    first = asyncio.create_task(agent.close_agent_session(request, close))
    await asyncio.wait_for(entered.wait(), 2)
    second = asyncio.create_task(agent.close_agent_session(request, close))
    await asyncio.sleep(0)
    release.set()
    first_result, second_result = await asyncio.wait_for(asyncio.gather(first, second), 2)
    assert first_result == second_result
    assert first_result is not second_result
    sandbox.disconnect.assert_awaited_once()


async def test_seed_prunes_expired_fences(setup, monkeypatch):
    agent, _ = setup
    clock = [100.0]
    monkeypatch.setattr("nemo_gym.agents.base_responses_api_agent.monotonic", lambda: clock[0])
    request = Request({"type": "http", "session": {}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    clock[0] += agent.config.session_close_retry_window_seconds
    await agent.seed_agent_session(Request({"type": "http", "session": {}}), seed())
    assert not agent._fenced_session_records


@pytest.mark.parametrize("window", [0, -1, float("inf")])
def test_close_retry_window_must_be_positive_and_finite(setup, window):
    agent, _ = setup
    with pytest.raises(ValidationError):
        OpenCodeAgentConfig(**(agent.config.model_dump() | {"session_close_retry_window_seconds": window}))


async def test_unknown_launch_outcome_fails_closed(setup):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}, "path_params": {"rollout_id": "opencode-smoke-a2"}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    sandbox.launch.side_effect = TimeoutError("lost launch response")
    with pytest.raises(TimeoutError, match="lost launch response"):
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    with pytest.raises(RuntimeError, match="launch outcome is unknown"):
        await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert session_id in active_sessions(agent)
    sandbox.disconnect.assert_not_awaited()
    sandbox.stop.assert_not_awaited()


async def test_install_failure_disconnects_without_stopping_owner(setup):
    agent, sandbox = setup
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=0, error_type=None),
        SimpleNamespace(return_code=1, stderr="curl failed", stdout="", error_type=None),
        SimpleNamespace(return_code=0, error_type=None),
    ]
    request = Request({"type": "http", "session": {}})
    with pytest.raises(NgFailureEvent, match="curl failed"):
        await agent.seed_agent_session(request, seed())
    assert not active_sessions(agent)
    assert not request.session
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


async def test_cancelled_install_never_publishes_session_or_launches_opencode(setup):
    agent, sandbox = setup
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=0, error_type=None),
        asyncio.CancelledError(),
        SimpleNamespace(return_code=0, error_type=None),
    ]
    request = Request({"type": "http", "session": {}})
    with pytest.raises(asyncio.CancelledError):
        await agent.seed_agent_session(request, seed())
    assert not active_sessions(agent)
    assert not request.session
    sandbox.launch.assert_not_awaited()
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


def test_reported_usage_restores_cached_and_reasoning_tokens_across_subagents():
    export = {
        "usage_messages": [
            {
                "role": "assistant",
                "tokens": {"input": 10, "output": 3, "reasoning": 2, "cache": {"read": 4, "write": 1}},
            },
            {"role": "assistant", "tokens": {"input": 7, "output": 5, "reasoning": 1, "cache": {"read": 2}}},
            {"role": "user"},
        ]
    }
    usage = OpenCodeAgent._session_usage(export)
    assert usage.input_tokens == 24
    assert usage.output_tokens == 11
    assert usage.total_tokens == 35
    assert usage.input_tokens_details.cached_tokens == 6
    assert usage.output_tokens_details.reasoning_tokens == 3
    assert OpenCodeAgent._session_usage({"messages": []}) is None


@pytest.mark.parametrize("field", ["cache", "reasoning"])
@pytest.mark.parametrize("value", [None, 0, -1, True, 1.5, "3", "bad"])
def test_reported_usage_does_not_treat_defaulted_or_invalid_details_as_measurements(field, value):
    tokens = {"input": 10, "output": 3, "reasoning": 2, "cache": {"read": 4, "write": 1}}
    if field == "cache":
        tokens["cache"]["read"] = value
    else:
        tokens["reasoning"] = value
    usage = OpenCodeAgent._session_usage({"usage_messages": [{"role": "assistant", "tokens": tokens}]})
    # Invalid optional fields neither erase measured base counts nor get coerced into totals.
    assert usage.input_tokens == (11 if field == "cache" else 15)
    assert usage.output_tokens == (3 if field == "reasoning" else 5)
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens
    assert usage.input_tokens_details.cached_tokens == (None if field == "cache" else 4)
    assert usage.output_tokens_details.reasoning_tokens == (None if field == "reasoning" else 2)


@pytest.mark.parametrize("unknown_turn", [0, 1])
@pytest.mark.parametrize("unknown_kind", ["zero", "absent", "missing_usage"])
def test_optional_reported_usage_stays_unknown_across_root_and_subagent_calls(unknown_turn, unknown_kind):
    infos = [
        {"role": "assistant", "tokens": {"input": 10, "output": 3, "reasoning": 2, "cache": {"read": 4, "write": 1}}},
        {"role": "assistant", "tokens": {"input": 7, "output": 5, "reasoning": 1, "cache": {"read": 2}}},
    ]
    if unknown_kind == "missing_usage":
        infos[unknown_turn].pop("tokens")
    elif unknown_kind == "absent":
        infos[unknown_turn]["tokens"].pop("reasoning")
        infos[unknown_turn]["tokens"]["cache"].pop("read")
    else:
        infos[unknown_turn]["tokens"]["reasoning"] = 0
        infos[unknown_turn]["tokens"]["cache"]["read"] = 0
    usage = OpenCodeAgent._session_usage({"usage_messages": infos})
    assert usage.input_tokens_details.cached_tokens is None
    assert usage.output_tokens_details.reasoning_tokens is None
    if unknown_kind == "missing_usage":
        assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (
            (9, 6, 15) if unknown_turn == 0 else (15, 5, 20)
        )
    else:
        assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (
            (20, 9, 29) if unknown_turn == 0 else (22, 10, 32)
        )


@pytest.mark.parametrize("cache,reasoning", [(0, 0), (4, 0), (0, 2), (4, 2)])
def test_session_http_usage_gaps_follow_persisted_optional_counter_availability(setup, tmp_path, cache, reasoning):
    agent, sandbox = setup
    export = json.loads(sandbox.events)
    tokens = export["messages"][1]["info"]["tokens"]
    tokens["cache"] = {"read": cache, "write": 1}
    tokens["reasoning"] = reasoning
    sandbox.events = json.dumps(export)
    capture_observations(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        created.raise_for_status()
        activated = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        activated.raise_for_status()
        response = activated.json()
        assert response["status"] == "completed"
        assert response["usage"] == {
            "input_tokens": 11 + cache,
            "output_tokens": 2 + reasoning,
            "total_tokens": 13 + cache + reasoning,
            "input_tokens_details": {"cached_tokens": cache or None, "cache_write_tokens": None},
            "output_tokens_details": {"reasoning_tokens": reasoning or None},
        }
        assert [item["type"] for item in response["output"]] == [
            "reasoning",
            "function_call",
            "function_call_output",
            "message",
        ]
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        closed.raise_for_status()
    gaps = {gap["code"] for gap in closed.json()["agent_observations"]["gaps"]}
    assert ("token_usage_detail_unavailable" in gaps) == (cache == 0 or reasoning == 0)


@pytest.mark.parametrize("option", ["missing-sandbox", "required-tool", "workdir", "unpinned", "provider"])
def test_invalid_seed_never_connects(setup, option):
    agent, sandbox = setup
    body = seed().model_dump(mode="json")
    if option == "missing-sandbox":
        body["sandbox_access"] = None
        agent.config.sandbox_provider = ""
    elif option == "required-tool":
        body["tool_accesses"] = [
            {"kind": "direct_http", "name": "tools", "base_url": "http://resources", "required": True}
        ]
    elif option == "workdir":
        body["sandbox_access"]["workdir"] = "/"
    elif option == "unpinned":
        agent.config.opencode_version = "latest"
    else:
        agent.config.opencode_config = {"provider": {"other": {"apiKey": "must-not-copy"}}}
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=body).status_code == 422
    sandbox.exec.assert_not_awaited()


def test_several_workers_seed_sessions(setup):
    """Session routing sends a session's later requests to the worker that seeded it."""
    agent, sandbox = setup
    agent.config.num_workers = 2
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        assert created.json()["agent_session_id"] in agent._session_records


@pytest.mark.parametrize("output_budget", [0, -1, 32001])
def test_invalid_output_budget_rejected_before_connecting(setup, output_budget):
    agent, sandbox = setup
    agent.config.max_output_tokens = output_budget
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
    assert response.status_code == 422
    assert "max_output_tokens" in response.json()["detail"]
    sandbox.exec.assert_not_awaited()
    sandbox.launch.assert_not_awaited()


@pytest.mark.parametrize("project_config", [False, True])
def test_task_repository_project_config_follows_the_session_setting(setup, project_config):
    agent, sandbox = setup
    agent.config.session_project_config = project_config
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == 200, result.text
        payload = sandbox.params()
        client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
    # OpenCode reads the repository's AGENTS.md/CLAUDE.md and opencode.json only without this flag.
    assert payload["env"].get("OPENCODE_DISABLE_PROJECT_CONFIG") == (None if project_config else "true")
    # The rest of the isolation stays, and Gym's inline config is still passed.
    assert payload["env"]["OPENCODE_DISABLE_AUTOUPDATE"] == "true"
    assert "OPENCODE_CONFIG_CONTENT" in payload["env"]


@pytest.mark.parametrize("kind", ["error", "timeout", "length"])
def test_partial_output_survives_model_failure_and_timeout(setup, kind, tmp_path):
    agent, sandbox = setup
    export = json.loads(sandbox.events)
    if kind == "error":
        export["messages"][-1]["info"]["error"] = {"message": "model rejected request"}
    elif kind == "timeout":
        sandbox.result["timed_out"] = True
        sandbox.result["return_code"] = -9
    else:
        export["messages"][-1]["info"]["finish"] = "length"
    sandbox.events = json.dumps(export)
    capture_observations(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        session_id = created.json()["agent_session_id"]
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        if kind == "error":
            assert result.status_code == 502
            assert "model rejected request" in result.json()["detail"]
        else:
            assert result.json()["status"] == "incomplete"
            assert result.json()["output"][-1]["content"][0]["text"] == "Fixed"
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200
        invocation = next(
            record for record in closed.json()["agent_observations"]["records"] if record["kind"] == "agent_invocation"
        )
        assert invocation["status"] == ("failed" if kind == "error" else "incomplete")
        assert invocation["conversation"][-1]["content"][0]["text"] == "Fixed"
        retry = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert retry.json() == closed.json()
        assert client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"}).status_code == 409
    sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("case", ["initial", "later", "partial", "auxiliary", "usage", "unknown-usage"])
def test_terminal_model_error_reports_partial_output_and_usage(setup, tmp_path, case):
    agent, sandbox = setup
    export = json.loads(rejected_events(after_work=case == "later"))
    sandbox.result["return_code"] = 1
    if case == "partial":
        export["messages"][-1]["parts"] = [{"type": "text", "text": "partial output"}]
    elif case == "auxiliary":
        export["usage_messages"] = [
            export["messages"][-1]["info"],
            json.loads(events())["messages"][-1]["info"],
        ]
    elif case == "usage":
        export["messages"][-1]["info"]["tokens"]["output"] = 1
    elif case == "unknown-usage":
        del export["messages"][-1]["info"]["tokens"]
    sandbox.events = json.dumps(export)
    capture_observations(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        response = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "failed"
        if case == "later":
            assert [part["type"] for part in response.json()["output"]] == [
                "reasoning",
                "function_call",
                "function_call_output",
                "message",
            ]
            assert response.json()["output"][-1]["content"][0]["text"] == "Fixed"
        elif case == "partial":
            assert response.json()["output"][-1]["content"][0]["text"] == "partial output"
        else:
            assert response.json()["output"] == []
        if case == "unknown-usage":
            assert response.json()["usage"] is None
        else:
            assert response.json()["usage"]["total_tokens"] == (
                16 if case in {"later", "auxiliary"} else 1 if case == "usage" else 0
            )
        assert response.json()["error"]["message"] == "HTTP 400: model rejected request"
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text
        invocation = next(
            record for record in closed.json()["agent_observations"]["records"] if record["kind"] == "agent_invocation"
        )
        assert invocation["status"] == "failed"
        observation = next(
            record for record in closed.json()["agent_observations"]["records"] if record["kind"] == "sandbox"
        )
        assert (observation["outcome"], observation["exit_code"], observation["error_type"]) == (
            "failed",
            1,
            "process_exit",
        )
        if case not in {"later", "partial"}:
            assert [item["role"] for item in invocation["conversation"]] == ["user"]


@pytest.mark.parametrize(("status_code", "retryable"), [(401, False), (403, False), (429, True), (500, True)])
def test_terminal_provider_errors_report_failure_to_environment(setup, tmp_path, status_code, retryable):
    agent, sandbox = setup
    export = json.loads(rejected_events())
    export["messages"][-1]["info"]["error"]["data"].update(statusCode=status_code, isRetryable=retryable)
    sandbox.result["return_code"] = 1
    sandbox.events = json.dumps(export)
    capture_observations(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        response = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        response.raise_for_status()
        assert response.json()["status"] == "failed"
        assert response.json()["error"]["message"] == f"HTTP {status_code}: model rejected request"
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        invocation = next(
            record for record in closed.json()["agent_observations"]["records"] if record["kind"] == "agent_invocation"
        )
        assert invocation["status"] == "failed"
        assert invocation["error_type"] == "provider_error"


@pytest.mark.parametrize(
    "case",
    [
        "unterminated",
        "not-last",
        "cleanup-error",
        "timeout",
        "zero-exit",
        "other-exit",
    ],
)
def test_model_error_does_not_hide_independent_execution_failures(setup, tmp_path, case):
    agent, sandbox = setup
    export = json.loads(rejected_events(after_work=True))
    info = export["messages"][-1]["info"]
    sandbox.result["return_code"] = 1
    if case == "unterminated":
        del info["time"]["completed"]
    elif case == "not-last":
        export["messages"].append(json.loads(events())["messages"][-1])
    elif case == "cleanup-error":
        sandbox.result["error"] = "worker failed"
    elif case == "timeout":
        sandbox.result["timed_out"] = True
    elif case in {"zero-exit", "other-exit"}:
        sandbox.result["return_code"] = 0 if case == "zero-exit" else 2
    sandbox.events = json.dumps(export)
    capture_observations(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        response = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert response.status_code == 502, response.text
        if case == "cleanup-error":
            assert "worker failed" in response.json()["detail"]
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        closed.raise_for_status()
        invocation = next(
            record for record in closed.json()["agent_observations"]["records"] if record["kind"] == "agent_invocation"
        )
        assert invocation["status"] == "failed"
        assert invocation["conversation"][-1]["content"][0]["text"] == "Fixed"


@pytest.mark.parametrize(
    ("finish", "expected"),
    [
        ("stop", "completed"),
        ("length", "incomplete"),
        ("content-filter", "incomplete"),
        ("tool-calls", "incomplete"),
        ("error", "failed"),
        ("unknown", "failed"),
        (None, "failed"),
    ],
)
def test_completed_model_turn_is_not_always_terminal(setup, tmp_path, finish, expected):
    agent, sandbox = setup
    export = json.loads(sandbox.events)
    export["messages"][-1]["info"]["finish"] = finish
    sandbox.events = json.dumps(export)
    capture_observations(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        if expected == "failed":
            assert result.status_code == 502
            assert "terminal assistant result" in result.json()["detail"]
        else:
            assert result.status_code == 200
            assert result.json()["status"] == expected
            assert result.json()["output"][-1]["content"][0]["text"] == "Fixed"
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200
        invocation = next(
            record for record in closed.json()["agent_observations"]["records"] if record["kind"] == "agent_invocation"
        )
        assert invocation["status"] == expected
        assert invocation["conversation"][-1]["content"][0]["text"] == "Fixed"


@pytest.mark.parametrize(("finish", "expected"), [("stop", "completed"), ("tool-calls", "incomplete")])
def test_timeout_preserves_completed_children_and_marks_unfinished_children(setup, tmp_path, finish, expected):
    agent, sandbox = setup
    sandbox.result["timed_out"] = True
    sandbox.result["return_code"] = -9
    database = capture_observations(sandbox, tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("insert into session values ('child', 'root', 1)")
        for index, reason in enumerate(("tool-calls", finish)):
            connection.execute(
                "insert into message values (?, ?, ?, ?)",
                (
                    f"child-message-{index}",
                    "child",
                    json.dumps({"role": "assistant", "finish": reason, "time": {"completed": 100 + index}}),
                    index,
                ),
            )
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert result.json()["status"] == "incomplete"
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        invocations = {
            record["invocation_id"]: record["status"]
            for record in closed.json()["agent_observations"]["records"]
            if record["kind"] == "agent_invocation"
        }
        assert invocations == {"root": "incomplete", "child": expected}


async def test_file_cleanup_failure_keeps_connection_and_can_retry(setup):
    agent, sandbox = setup
    request, session_id, task = await activate(agent, sandbox)
    await task
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=1, error_type=None, stderr="file cleanup failed"),
        SimpleNamespace(return_code=0, error_type=None, stderr=""),
    ]
    close = AgentCloseSessionRequest(**close_body(session_id))
    with pytest.raises(RuntimeError, match="session files"):
        await agent.close_agent_session(request, close)
    sandbox.disconnect.assert_not_awaited()
    assert agent._session_records[session_id].closing
    await agent.close_agent_session(request, close)
    sandbox.disconnect.assert_awaited_once()


def test_malformed_later_artifact_keeps_partial_output(setup):
    from nemo_gym.trajectory import AgentObservationBundle

    agent, _ = setup
    export = json.loads(events())
    export["messages"][1]["parts"].extend(
        [
            {"type": "future-event"},
            {
                "type": "tool",
                "callID": "failed-tool",
                "tool": "bash",
                "state": {"input": {}, "status": "error", "error": "failed tool"},
            },
        ]
    )
    observations = AgentObservationBundle(source="opencode")
    output = agent._session_output(export, observations)
    assert output[3].content[0].text == "Fixed"
    assert output[-1].output == "failed tool"
    assert output[-1].status == "incomplete"
    assert observations.gaps[0].code == "agent_artifact_record_unparseable"


@pytest.mark.parametrize(
    "workdir",
    [
        "/",
        "/tmp",
        "/tmp/",
        "/tmp/../tmp/task",
        "/tmp/nemo-gym-opencode-sessions",
        "/tmp/nemo-gym-opencode-runtime-1.17.11/repo",
    ],
)
async def test_adapter_workdir_overlap_is_rejected_before_connection(setup, workdir):
    agent, sandbox = setup
    body = seed()
    body.sandbox_access.workdir = workdir
    with pytest.raises(HTTPException) as error:
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    assert error.value.status_code == 422
    sandbox.exec.assert_not_awaited()


@pytest.mark.parametrize(
    ("workdir", "runtime", "overlaps"),
    [
        ("task", "runtime", True),  # a symlink to the session root
        ("sessions/nested", "runtime", True),
        ("repo", "link/runtime", True),  # a missing root resolved through a symlinked parent
        ("repo", "runtime", False),
        ("repo/../repo", "runtime", False),
    ],
)
def test_resolved_workdir_check_runs_before_session_files_are_created(tmp_path, workdir, runtime, overlaps):
    sessions = tmp_path / "sessions"
    (sessions / "nested").mkdir(parents=True)
    (tmp_path / "task").symlink_to(sessions, target_is_directory=True)
    (tmp_path / "repo").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "repo", target_is_directory=True)
    session = sessions / "session"
    command = _sandbox_prepare_command(str(tmp_path / workdir), str(session), str(tmp_path / runtime))
    result = subprocess.run(["/bin/sh", "-c", command], capture_output=True, text=True)
    if overlaps:
        assert result.returncode != 0
        assert "overlaps the task workdir" in result.stderr
        assert not session.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert session.is_dir()


def test_missing_workdir_is_rejected(tmp_path):
    command = _sandbox_prepare_command(str(tmp_path / "missing"), str(tmp_path / "s/session"), str(tmp_path / "r"))
    result = subprocess.run(["/bin/sh", "-c", command], capture_output=True, text=True)
    assert result.returncode != 0
    assert "OpenCode task workdir is missing" in result.stderr
    assert not (tmp_path / "s").exists()


def test_close_response_cookie_blocks_legacy_run(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"])).raise_for_status()
        response = client.post("/run", json={"responses_create_params": {"input": "task"}})
        assert response.status_code == 409
    agent.server_client.post.assert_not_called()


async def test_caller_assigned_seed_is_serialized_and_binds_all_inputs(setup):
    agent, sandbox = setup
    body = seed()
    first_request = Request({"type": "http", "session": {}})
    retry_request = Request({"type": "http", "session": {}})
    first, retry = await asyncio.gather(
        agent.seed_agent_session(first_request, body), agent.seed_agent_session(retry_request, body)
    )
    assert first.agent_session_id == retry.agent_session_id == body.agent_session_id
    assert sandbox.exec.await_count == 2
    changed = body.model_copy(deep=True)
    changed.sandbox_access.workdir = "/different"
    with pytest.raises(HTTPException, match="another seed request"):
        await agent.seed_agent_session(retry_request, changed)
    await agent.close_agent_session(
        Request({"type": "http", "session": {}}), AgentCloseSessionRequest(**close_body(body.agent_session_id))
    )
    sandbox.disconnect.assert_awaited_once()


async def test_close_without_seed_cookie_blocks_delayed_seed(setup):
    agent, sandbox = setup
    body = seed()
    request = Request({"type": "http", "session": {}})
    close = AgentCloseSessionRequest(**close_body(body.agent_session_id))
    assert (await agent.close_agent_session(request, close)).agent_session_id == body.agent_session_id
    with pytest.raises(HTTPException, match="already closed"):
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    sandbox.exec.assert_not_awaited()
    sandbox.disconnect.assert_not_awaited()


async def test_close_waits_for_inflight_seed_even_without_cookie(setup):
    agent, sandbox = setup
    body = seed()
    entered, release = asyncio.Event(), asyncio.Event()
    initialize = agent._seed_agent_session_state

    async def delayed_initialize(*args):
        entered.set()
        await release.wait()
        return await initialize(*args)

    with patch.object(agent, "_seed_agent_session_state", delayed_initialize):
        seeded = asyncio.create_task(agent.seed_agent_session(Request({"type": "http", "session": {}}), body))
        await entered.wait()
        closed = asyncio.create_task(
            agent.close_agent_session(
                Request({"type": "http", "session": {}}), AgentCloseSessionRequest(**close_body(body.agent_session_id))
            )
        )
        await asyncio.sleep(0)
        assert not closed.done()
        release.set()
        await seeded
        await closed
    assert not active_sessions(agent)
    sandbox.disconnect.assert_awaited_once()


async def test_active_session_has_no_adapter_expiry_timer(setup):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}})
    body = seed()
    await agent.seed_agent_session(request, body)
    await asyncio.sleep(0)
    assert "session_lifetime_seconds" not in type(agent.config).model_fields
    assert body.agent_session_id in active_sessions(agent)
    sandbox.disconnect.assert_not_awaited()
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(body.agent_session_id)))


async def test_caller_session_id_is_never_used_as_a_filesystem_path(setup):
    agent, sandbox = setup
    body = seed().model_copy(update={"agent_session_id": "../../outside; echo untrusted"})
    request = Request({"type": "http", "session": {}})
    await agent.seed_agent_session(request, body)
    state = active_sessions(agent)[body.agent_session_id]
    assert Path(state.session.session_dir).parent == Path("/tmp/nemo-gym-opencode-sessions")
    assert body.agent_session_id not in state.session.session_dir
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(body.agent_session_id)))


async def test_cookie_of_an_expired_fence_cannot_seed_or_activate(setup, monkeypatch):
    agent, sandbox = setup
    clock = [100.0]
    monkeypatch.setattr("nemo_gym.agents.base_responses_api_agent.monotonic", lambda: clock[0])
    agent.config.session_close_retry_window_seconds = 10
    body = seed()
    request = Request({"type": "http", "session": {}})
    await agent.seed_agent_session(request, body)
    close = AgentCloseSessionRequest(**close_body(body.agent_session_id))
    await agent.close_agent_session(request, close)
    clock[0] = 111.0
    with pytest.raises(HTTPException, match="no longer available"):
        await agent.close_agent_session(request, close)
    with pytest.raises(HTTPException, match="no longer available"):
        await agent.seed_agent_session(request, body)
    with pytest.raises(HTTPException):
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("failure", ["remove", "disconnect"])
async def test_failed_setup_cleanup_retains_state_for_cookieless_close(setup, failure):
    agent, sandbox = setup
    body = seed()
    success = SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None)
    failed = SimpleNamespace(return_code=1, stdout="", stderr="installer failed", error_type=None)
    sandbox.exec.side_effect = [success, failed, failed if failure == "remove" else success]
    if failure == "disconnect":
        sandbox.disconnect.side_effect = RuntimeError("disconnect failed")
    with pytest.raises(NgFailureEvent, match="installer failed"):
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    assert agent._session_records[body.agent_session_id].closing
    with pytest.raises(HTTPException, match="closing"):
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    sandbox.exec.side_effect = None
    sandbox.exec.return_value = success
    sandbox.disconnect.side_effect = None
    closed = await agent.close_agent_session(
        Request({"type": "http", "session": {}}), AgentCloseSessionRequest(**close_body(body.agent_session_id))
    )
    assert closed.agent_observations is not None
    assert body.agent_session_id not in active_sessions(agent)
    sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize("owned", [False, True])
def test_sandbox_source_controls_ownership_and_session_routing(setup, owned):
    agent, sandbox = setup
    agent.config.sandbox_provider = "agent-provider"
    agent.config.sandbox_config = {"image": "test-image", "workdir": "/agent-workspace"}
    sandbox.expected_workdir = "/agent-workspace" if owned else "/app"
    body = seed()
    if owned:
        body.sandbox_access = None
    sandbox.start = AsyncMock()
    module = "responses_api_agents.opencode_agent.app"
    with (
        patch(f"{module}.AsyncSandbox", return_value=sandbox) as factory,
        patch(f"{module}.resolve_provider_config") as resolve,
        TestClient(agent.setup_webserver()) as client,
    ):
        factory.connect = AsyncMock(return_value=sandbox)
        created = client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
        assert created.status_code == 200, created.text
        state = active_sessions(agent)[body.agent_session_id]
        assert state.session.owns_sandbox is owned
        assert state.sandbox_id == sandbox.sandbox_id
        assert state.session.workdir == sandbox.expected_workdir
        resolve.assert_called_once_with("agent-provider" if owned else "sandbox", {})
        workspace_calls = [call for call in sandbox.exec.await_args_list if call.args[0].startswith("mkdir -p --")]
        assert len(workspace_calls) == int(owned)
        if owned:
            assert workspace_calls[0].args[0] == "mkdir -p -- /agent-workspace"
            assert workspace_calls[0].kwargs["cwd"] == "/"
        if owned:
            factory.connect.assert_not_awaited()
            spec = sandbox.start.await_args.args[0]
            assert spec.image == "test-image"
            assert spec.workdir == "/agent-workspace"
        else:
            factory.assert_not_called()
            factory.connect.assert_awaited_once()
        response = client.post(
            f"/ng-rollout/{body.episode_id.capture_key}/v1/responses", json={"input": "Fix the code"}
        )
        assert response.status_code == 200, response.text
        close_request = {"agent_session_id": body.agent_session_id, "episode_id": body.episode_id.model_dump()}
        closed = client.post("/v1/agent_sessions/close", json=close_request)
        assert closed.status_code == 200, closed.text
        observation = next(
            record for record in closed.json()["agent_observations"]["records"] if record["kind"] == "sandbox"
        )
        assert observation["sandbox_id"] == sandbox.sandbox_id
        assert observation["provider"] == ("agent-provider" if owned else "sandbox")
        assert client.post("/v1/agent_sessions/close", json=close_request).json() == closed.json()
        if owned:
            sandbox.stop.assert_awaited_once()
            sandbox.disconnect.assert_not_awaited()
        else:
            sandbox.stop.assert_not_awaited()
            sandbox.disconnect.assert_awaited_once()
        assert (
            client.post(f"/ng-rollout/{body.episode_id.capture_key}/v1/responses", json={"input": "task"}).status_code
            == 409
        )


def owned_seed_with_search_grant() -> dict:
    body = seed().model_dump(mode="json")
    body["sandbox_access"] = None
    body["tool_accesses"] = [
        {
            "kind": "mcp",
            "name": "search",
            "required": False,
            "connection": {"url": "http://resources.example:8000/search/mcp", "headers": {"x-session": "token"}},
        }
    ]
    return body


def owned_seed_with_several_grants(*, required: bool = True) -> dict:
    """A required and an optional MCP grant on different hosts, plus an optional direct HTTP grant OpenCode skips."""
    body = owned_seed_with_search_grant()
    body["tool_accesses"] = [
        {
            "kind": "mcp",
            "name": "search",
            "required": required,
            "connection": {"url": "http://resources.example:8000/search/mcp", "headers": {"x-session": "token"}},
        },
        {
            "kind": "mcp",
            "name": "docs",
            "required": False,
            "connection": {"url": "http://docs.example:9000/mcp", "headers": {"x-session": "token"}},
        },
        {"kind": "direct_http", "name": "direct", "base_url": "http://direct.example", "required": False},
    ]
    return body


@pytest.mark.parametrize(
    ("network_access", "required", "allowed"),
    [
        ("inherit", True, None),
        # A required grant under model_only is refused (see the next test), so this case grants only optional ones.
        ("model_only", False, ["model.example"]),
        ("model_and_tools", True, ["docs.example", "model.example", "resources.example"]),
    ],
)
def test_owned_sandbox_egress_allows_only_model_and_granted_tool_hosts(setup, network_access, required, allowed):
    agent, sandbox = setup
    agent.config.sandbox_provider = "agent-provider"
    agent.config.sandbox_config = {"image": "test-image", "workdir": "/workspace"}
    agent.config.network_access = network_access
    sandbox.expected_workdir = "/workspace"
    sandbox.start = AsyncMock()
    module = "responses_api_agents.opencode_agent.app"
    with (
        patch(f"{module}.AsyncSandbox", return_value=sandbox),
        patch(f"{module}.create_provider", return_value=SimpleNamespace(name="opensandbox", aclose=AsyncMock())),
        TestClient(agent.setup_webserver()) as client,
    ):
        created = client.post("/v1/agent_sessions", json=owned_seed_with_several_grants(required=required))
        assert created.status_code == 200, created.text
    # Every MCP grant's host, required or optional, and never the direct HTTP grant OpenCode does not use.
    policy = sandbox.start.await_args.args[0].provider_options.get("network_policy")
    if allowed is None:
        assert policy is None
    else:
        assert policy == {"defaultAction": "deny", "egress": [{"action": "allow", "target": host} for host in allowed]}


@pytest.mark.parametrize("network_access", ["model_only", "model_and_tools"])
def test_network_access_with_borrowed_sandbox_is_rejected_before_connecting(setup, network_access):
    agent, sandbox = setup
    agent.config.network_access = network_access
    module = "responses_api_agents.opencode_agent.app"
    with (
        patch(f"{module}.AsyncSandbox") as factory,
        TestClient(agent.setup_webserver()) as client,
    ):
        factory.connect = AsyncMock(return_value=sandbox)
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 422
        assert "borrowed" in created.text
        factory.connect.assert_not_awaited()


def test_network_access_and_explicit_policy_conflict_is_a_terminal_seed_refusal(setup):
    agent, sandbox = setup
    agent.config.sandbox_provider = "agent-provider"
    agent.config.sandbox_config = {
        "image": "test-image",
        "workdir": "/workspace",
        "provider_options": {"network_policy": {}},
    }
    agent.config.network_access = "model_only"
    provider = SimpleNamespace(name="opensandbox", aclose=AsyncMock())
    module = "responses_api_agents.opencode_agent.app"
    with (
        patch(f"{module}.AsyncSandbox") as factory,
        patch(f"{module}.create_provider", return_value=provider),
        TestClient(agent.setup_webserver()) as client,
    ):
        refused = client.post("/v1/agent_sessions", json=owned_seed_with_search_grant())
    # A configuration another attempt cannot fix: HTTP 422 fails the episode once instead of a retried 500.
    assert refused.status_code == 422, refused.text
    assert "not both" in refused.text
    factory.assert_not_called()
    provider.aclose.assert_awaited_once()


def test_required_tool_grant_under_model_only_is_a_terminal_seed_refusal(setup):
    agent, sandbox = setup
    agent.config.sandbox_provider = "agent-provider"
    agent.config.sandbox_config = {"image": "test-image", "workdir": "/workspace"}
    agent.config.network_access = "model_only"
    provider = SimpleNamespace(name="opensandbox", aclose=AsyncMock())
    body = owned_seed_with_several_grants(required=True)
    module = "responses_api_agents.opencode_agent.app"
    with (
        patch(f"{module}.AsyncSandbox") as factory,
        patch(f"{module}.create_provider", return_value=provider),
        TestClient(agent.setup_webserver()) as client,
    ):
        refused = client.post("/v1/agent_sessions", json=body)
    # model_only blocks the grant's host, so the session would fail only after the sandbox and runtime exist.
    assert refused.status_code == 422, refused.text
    assert "model_and_tools" in refused.json()["detail"]
    factory.assert_not_called()
    provider.aclose.assert_awaited_once()
    assert body["agent_session_id"] not in agent._session_records


def test_owned_stop_failure_blocks_close_until_retry(setup):
    agent, sandbox = setup
    agent.config.sandbox_provider = "agent-provider"
    agent.config.sandbox_config = {"image": "test-image", "workdir": "/workspace"}
    sandbox.start = AsyncMock()
    body = seed()
    body.sandbox_access = None
    with (
        patch("responses_api_agents.opencode_agent.app.AsyncSandbox", return_value=sandbox),
        TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client,
    ):
        created = client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
        assert created.status_code == 200, created.text
        state = active_sessions(agent)[body.agent_session_id]
        assert state.session.workdir == "/workspace"
        # An owned sandbox can be destroyed even when no runner receipt was returned.
        state.session.launch_started = True
        sandbox.stop.side_effect = [RuntimeError("provider stop failed"), None]
        close_request = {"agent_session_id": body.agent_session_id, "episode_id": body.episode_id.model_dump()}
        assert client.post("/v1/agent_sessions/close", json=close_request).status_code == 500
        assert not state.session.closed
        assert (
            client.post(f"/ng-rollout/{body.episode_id.capture_key}/v1/responses", json={"input": "task"}).status_code
            == 409
        )
        assert client.post("/v1/agent_sessions/close", json=close_request).status_code == 200
        assert sandbox.stop.await_count == 2
        sandbox.disconnect.assert_not_awaited()


@pytest.mark.parametrize("stage", ["start", "workdir", "install"])
def test_owned_setup_failure_preserves_error_and_retryable_cleanup(setup, stage):
    agent, sandbox = setup
    agent.config.sandbox_provider = "agent-provider"
    agent.config.sandbox_config = {"image": "test-image", "workdir": "/app"}
    sandbox.start = AsyncMock()
    body = seed()
    body.sandbox_access = None
    original = RuntimeError(f"{stage} failed")
    if stage == "start":
        sandbox.start.side_effect = original
    else:
        execute = sandbox.exec.side_effect

        async def fail_at_stage(command, **kwargs):
            if (stage == "workdir" and command.startswith("mkdir -p --")) or (
                stage == "install" and command.startswith("bash ") and "install_opencode_runtime.sh" in command
            ):
                raise original
            return await execute(command, **kwargs)

        sandbox.exec.side_effect = fail_at_stage
    sandbox.stop.side_effect = [RuntimeError("stop failed"), None]
    with (
        patch("responses_api_agents.opencode_agent.app.AsyncSandbox", return_value=sandbox),
        TestClient(agent.setup_webserver()) as client,
    ):
        with raises_agent_failure(RuntimeError) as error:
            client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
        assert error.value is original
        assert client.post("/v1/agent_sessions", json=body.model_dump(mode="json")).status_code == 409
        state = active_sessions(agent)[body.agent_session_id]
        assert state.session.closing and not state.session.closed
        response = client.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": body.agent_session_id, "episode_id": body.episode_id.model_dump()},
        )
        assert response.status_code == 200, response.text
        assert sandbox.stop.await_count == 2
        sandbox.disconnect.assert_not_awaited()


def test_failed_connection_closes_provider_without_publishing_session(setup):
    agent, sandbox = setup
    agent.config.sandbox_provider = "fallback-must-not-be-used"
    module = "responses_api_agents.opencode_agent.app"
    provider = SimpleNamespace(aclose=AsyncMock())
    with (
        patch(f"{module}.AsyncSandbox") as factory,
        patch(f"{module}.create_provider", return_value=provider),
        TestClient(agent.setup_webserver()) as client,
    ):
        factory.connect = AsyncMock(side_effect=RuntimeError("borrow failed"))
        with raises_agent_failure(RuntimeError, match="borrow failed"):
            client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        factory.assert_not_called()
        sandbox.stop.assert_not_awaited()
        provider.aclose.assert_awaited_once()
        assert not active_sessions(agent)


@pytest.mark.parametrize("workdir", [None, "relative"])
def test_owned_workdir_is_validated_before_creation(setup, workdir):
    agent, sandbox = setup
    agent.config.sandbox_provider = "agent-provider"
    agent.config.sandbox_config = {"workdir": workdir}
    body = seed()
    body.sandbox_access = None
    with (
        patch("responses_api_agents.opencode_agent.app.AsyncSandbox") as factory,
        TestClient(agent.setup_webserver()) as client,
    ):
        assert client.post("/v1/agent_sessions", json=body.model_dump(mode="json")).status_code == 422
        factory.assert_not_called()
        sandbox.exec.assert_not_awaited()


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("invalid_runtime", [False, True])
async def test_close_keeps_captured_events_after_sandbox_release(setup, tmp_path, owned, invalid_runtime):
    agent, sandbox = setup
    body = seed()
    if owned:
        body.sandbox_access = None
        agent.config.sandbox_provider = "agent-provider"
        agent.config.sandbox_config = {"image": "test-image"}
    if invalid_runtime:
        sandbox.runtime_info = {"hostname": "worker", "pid": "not-an-integer"}
        sandbox.result["return_code"] = "not-an-integer"
    sandbox.start = AsyncMock()
    capture_observations(sandbox, tmp_path)
    request = Request({"type": "http", "session": {}, "path_params": {"rollout_id": "opencode-smoke-a2"}})
    with patch("responses_api_agents.opencode_agent.app.AsyncSandbox", return_value=sandbox) as factory:
        factory.connect = AsyncMock(return_value=sandbox)
        await agent.seed_agent_session(request, body)
        state = agent._session_records[body.agent_session_id].state
        directory = state.session.session_dir
        assert f"{directory}/process_supervisor.sh" not in sandbox.files
        sandbox.blocked = True
        task = asyncio.create_task(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task")))
        await asyncio.wait_for(sandbox.started.wait(), 2)
        assert f"{directory}/process_supervisor.sh" in sandbox.files

        async def release():
            assert state.session.cleanup["cleanup_confirmed"] is True
            assert state.session.artifacts is not None
            sandbox.files.clear()
            sandbox.download = AsyncMock(side_effect=AssertionError("output read after release"))

        sandbox.stop.side_effect = release
        sandbox.disconnect.side_effect = release
        close = AgentCloseSessionRequest(**close_body(body.agent_session_id))
        response = await agent.close_agent_session(request, close)
        with pytest.raises(asyncio.CancelledError):
            await task
        assert response.agent_observations is not None
        invocations = [record for record in response.agent_observations.records if record.kind == "agent_invocation"]
        assert any(record.conversation for record in invocations)
        assert all(record.agent_instance == state.agent_instance for record in invocations)
        tools = [record for record in response.agent_observations.records if record.kind == "tool_call"]
        assert tools and all(record.agent_instance == state.agent_instance for record in tools)
        assert state.session.closed
        assert await agent.close_agent_session(request, close) == response
        (sandbox.stop if owned else sandbox.disconnect).assert_awaited_once()
        (sandbox.disconnect if owned else sandbox.stop).assert_not_awaited()


@pytest.mark.parametrize("diagnostics", [{"return_code": "0"}, {"extra": 1}])
def test_optional_diagnostics_preserve_valid_terminal_result(setup, diagnostics):
    agent, sandbox = setup
    sandbox.result.update(diagnostics)
    sandbox.runtime_info = {"pid": "invalid"}
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        created.raise_for_status()
        response = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        response.raise_for_status()
        assert response.json()["status"] == "completed"
        assert response.json()["output"][-1]["content"][0]["text"] == "Fixed"
        assert "harness_hostname" not in response.json()["metadata"]
        assert "harness_pid" not in response.json()["metadata"]
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        closed.raise_for_status()
        gaps = {gap["code"] for gap in closed.json()["agent_observations"]["gaps"]}
        assert "runtime_info_unavailable" in gaps
        assert ("worker_exit_code_unavailable" in gaps) == ("return_code" in diagnostics)


async def test_disconnected_waiter_and_identical_retry_share_one_activation(setup):
    agent, sandbox = setup
    sandbox.blocked = True
    request, session_id, waiter = await activate(agent, sandbox)
    retry = asyncio.create_task(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task")))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    state = agent._session_records[session_id].state
    assert not state.task.done()
    sandbox.request_stop.assert_not_awaited()
    with pytest.raises(HTTPException, match="retry the same request"):
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="different task"))
    sandbox.exited.set()
    response = await asyncio.wait_for(retry, 2)
    original = response.model_copy(deep=True)
    response.output.clear()
    assert await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task")) == original
    sandbox.launch.assert_awaited_once()
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))


@pytest.mark.parametrize("receipt", [None, "", "partial", "ready"])
async def test_missing_task_environment_hook_rejects_otherwise_valid_result(setup, receipt):
    agent, sandbox = setup
    sandbox.task_environment_receipt = receipt
    request, session_id, task = await activate(agent, sandbox)
    state = active_sessions(agent)[session_id]
    with pytest.raises(RuntimeError, match="task environment initialization was not confirmed"):
        await task
    assert state.session.cleanup["cleanup_confirmed"] is True
    assert state.session.artifacts is None
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize(
    "budget,expected", [(None, 1800), ("900.0", 900), ("12.5", 12.5), ("3600", 1800), ("nan", 1800)]
)
def test_task_timeout_metadata_reaches_sandbox_supervisor(
    setup: tuple[OpenCodeAgent, Sandbox], tmp_path: Path, budget: str | None, expected: float
) -> None:
    agent, sandbox = setup
    agent.config.timeout = 1800
    (tmp_path / "task.toml").write_text("[agent]\ntimeout_sec = 900.0\n")
    (tmp_path / "instruction.md").write_text("Fix the code")
    (tmp_path / "tests").mkdir()
    body = materialize_task(load_task(tmp_path), "timeout-test").task_input.responses_create_params
    if budget != "900.0":
        body.metadata = None if budget is None else {"agent_timeout_sec": budget}
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        session_id = created.json()["agent_session_id"]
        response = client.post(
            "/ng-rollout/opencode-smoke-a2/v1/responses", json=body.model_dump(mode="json", exclude_unset=True)
        )
        assert response.status_code == 200, response.text
        command = shlex.split(sandbox.launch.await_args.kwargs["command"])
        # Staging spends the budget, and OpenCode runs for what remains.
        assert expected - 1 < float(command[command.index("--timeout") + 1]) <= expected
        assert agent.config.timeout == 1800
        assert response.json()["metadata"]["agent_timeout_sec"] == str(float(expected))
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text


def test_task_timeout_does_not_allow_other_metadata_or_consume_activation(
    setup: tuple[OpenCodeAgent, Sandbox],
) -> None:
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        session_id = created.json()["agent_session_id"]
        path = "/ng-rollout/opencode-smoke-a2/v1/responses"
        rejected = client.post(
            path, json={"input": "Fix the code", "metadata": {"agent_timeout_sec": "900.0", "ignored": "value"}}
        )
        assert rejected.status_code == 422
        assert "metadata" in rejected.json()["detail"]
        sandbox.launch.assert_not_awaited()
        accepted = client.post(path, json={"input": "Fix the code", "metadata": {"agent_timeout_sec": "900.0"}})
        assert accepted.status_code == 200, accepted.text
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text


@pytest.mark.parametrize("status", [400, 401, 413, 429, 503, None])
async def test_native_model_error_is_reported_with_classification(setup, status):
    agent, sandbox = setup
    export = json.loads(sandbox.events)
    assistant = export["messages"][-1]["info"]
    assistant.pop("finish")
    assistant["error"] = {
        "name": "APIError",
        "data": {
            "statusCode": status,
            "message": "provider error",
            "responseHeaders": {"x-provider-detail": "private-header"},
            "responseBody": "private-body",
        },
    }
    sandbox.events = json.dumps(export)
    sandbox.result["return_code"] = 1
    request, session_id, task = await activate(agent, sandbox)
    response = await task
    assert response.status == "failed"
    assert response.error.message == (f"HTTP {status}: provider error" if status is not None else "provider error")
    expected = (
        {"failure_kind": "unknown"}
        if status is None
        else {"failure_kind": "model_request_error", "status_code": status}
    )
    assert json.loads(response.metadata["ng_agent_outcome"]) == expected
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))


@pytest.mark.parametrize("message", [None, "", {"responseBody": "private-body"}])
async def test_native_model_error_without_message_uses_safe_fallback(setup, message):
    agent, sandbox = setup
    export = json.loads(sandbox.events)
    assistant = export["messages"][-1]["info"]
    assistant.pop("finish")
    assistant["error"] = {
        "name": "APIError",
        "data": {"statusCode": 503, "message": message, "responseBody": "private-body"},
    }
    sandbox.events = json.dumps(export)
    sandbox.result["return_code"] = 1
    request, session_id, task = await activate(agent, sandbox)
    response = await task
    assert response.error.message == "HTTP 503: OpenCode provider request failed"
    assert json.loads(response.metadata["ng_agent_outcome"]) == {
        "failure_kind": "model_request_error",
        "status_code": 503,
    }
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))


@pytest.mark.parametrize(
    "change", ["wrong_error", "finished", "not_completed", "worker_exit", "cleanup_error", "timeout"]
)
def test_provider_report_does_not_hide_other_execution_outcomes(change):
    native = {"error": {"name": "APIError"}, "time": {"completed": 1}}
    cleanup = {"return_code": 1, "timed_out": False, "error": None}
    if change == "wrong_error":
        native["error"]["name"] = "UnknownError"
    elif change == "finished":
        native["finish"] = "stop"
    elif change == "not_completed":
        native["time"] = {}
    elif change == "worker_exit":
        cleanup["return_code"] = 7
    elif change == "cleanup_error":
        cleanup["error"] = "cleanup failed"
    else:
        cleanup["timed_out"] = True
    assert not OpenCodeAgent._reported_model_error(native, cleanup)


@pytest.mark.parametrize(
    "message, code",
    [
        ("Connection reset by server", "ECONNRESET"),
        ("Response decompression failed", "ZlibError"),
        ("Timed out waiting for response headers", "HeaderTimeoutError"),
        ("Response stream interrupted", "ResponseStreamError"),
    ],
)
async def test_statusless_transient_provider_failure_stays_retryable(setup, message, code):
    from nemo_gym.agents import OutcomeAction, agent_outcome_policy

    agent, sandbox = setup
    export = json.loads(sandbox.events)
    assistant = export["messages"][-1]["info"]
    assistant.pop("finish")
    assistant["error"] = {
        "name": "APIError",
        "data": {"message": message, "isRetryable": True, "metadata": {"code": code}},
    }
    sandbox.events = json.dumps(export)
    sandbox.result["return_code"] = 1
    request, session_id, task = await activate(agent, sandbox)
    response = await task
    assert response.status == "failed"
    assert message in response.error.message
    assert json.loads(response.metadata["ng_agent_outcome"]) == {"failure_kind": "model_transport_error"}
    policy = agent_outcome_policy(response)
    assert policy == OutcomeAction.RETRY
    closed = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert closed.agent_observations is not None


@pytest.mark.parametrize("retryable", [None, False, "true", 1])
def test_statusless_error_without_native_retry_flag_remains_unknown(setup, retryable):
    agent, _ = setup
    data = {} if retryable is None else {"isRetryable": retryable}
    outcome = agent.classify_outcome({"error": {"name": "APIError", "data": data}})
    assert outcome.failure_kind == "unknown"


@pytest.mark.parametrize("cleanup_error", [None, "cleanup failed"])
async def test_context_limit_preserves_partial_work_without_hiding_cleanup_failure(setup, tmp_path, cleanup_error):
    from nemo_gym.agents import OutcomeAction, agent_outcome_policy

    agent, sandbox = setup
    export = json.loads(sandbox.events)
    assistant = export["messages"][-1]["info"]
    assistant["finish"] = "error"
    assistant["error"] = {
        "name": "ContextOverflowError",
        "data": {"message": "context is too large", "responseBody": "provider detail"},
    }
    sandbox.events = json.dumps(export)
    sandbox.result.update(return_code=1, error=cleanup_error)
    capture_observations(sandbox, tmp_path)
    request, session_id, task = await activate(agent, sandbox)
    if cleanup_error:
        with pytest.raises(HTTPException):
            await task
    else:
        response = await task
        assert response.status == "incomplete"
        assert response.error is None
        assert response.output
        assert response.usage.total_tokens > 0
        assert agent.classify_outcome(assistant).failure_kind == "model_context_limit"
        assert agent_outcome_policy(response) == OutcomeAction.VERIFY
    closed = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    root = next(r for r in closed.agent_observations.records if r.kind == "agent_invocation")
    assert root.status == ("failed" if cleanup_error else "incomplete")
    if not cleanup_error:
        assert root.error_type == "context_limit"


async def test_reported_native_error_is_bounded_without_losing_classification(setup):
    agent, sandbox = setup
    export = json.loads(sandbox.events)
    assistant = export["messages"][-1]["info"]
    assistant.pop("finish")
    assistant["error"] = {"name": "APIError", "data": {"statusCode": 429, "message": "start" + "x" * 20000 + "end"}}
    sandbox.events = json.dumps(export)
    sandbox.result["return_code"] = 1
    request, session_id, task = await activate(agent, sandbox)
    response = await task
    assert len(response.error.message) <= 16000
    assert "...[truncated]..." in response.error.message
    assert "start" in response.error.message and "end" in response.error.message
    assert json.loads(response.metadata["ng_agent_outcome"]) == {
        "failure_kind": "model_request_error",
        "status_code": 429,
    }
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))


def test_explicit_native_model_and_runtime_paths_are_passed_to_the_harness(setup):
    agent, sandbox = setup
    agent.config.sandbox_model_id = "gpt-5.5"
    agent.config.sandbox_path_prefix = ["/runtime/bin"]
    agent.config.sandbox_executable_path = "/runtime/bin/opencode"
    agent.config.sandbox_executable_sha256 = "1" * 64
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).status_code == 200
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "Fix code"})
        assert result.status_code == 200, result.text
        params = sandbox.params()
        config = json.loads(params["env"]["OPENCODE_CONFIG_CONTENT"])
        assert config["model"] == config["small_model"] == "nemo_gym/gpt-5.5"
        assert set(config["provider"]["nemo_gym"]["models"]) == {"gpt-5.5"}
        assert config["provider"]["nemo_gym"]["models"]["gpt-5.5"]["reasoning"] is True
        # The runner puts the preinstalled runtime first and keeps the image PATH after it.
        assert params["setup"].endswith("PATH=/runtime/bin:$PATH\n")
