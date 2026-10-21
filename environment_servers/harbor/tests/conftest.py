# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fakes and helpers of the harbor environment server tests.

The task fixtures, the FakeSandbox family and the Compose fakes, the fake agent client, the
server factory and the episode runner that test_app.py, test_overlay.py, test_entrypoint.py,
test_review_regressions.py and test_verifier_filesystem.py all use. None of these are pytest
fixtures, so the test modules import them by name from this file.
"""

import asyncio
import inspect
import io
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import orjson
from omegaconf import OmegaConf
from pydantic import ConfigDict
from pytest import MonkeyPatch

from environment_servers.harbor.app import (
    HarborEnvironmentServer,
    HarborEnvironmentServerConfig,
    HarborEpisodeResponse,
)
from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from nemo_gym.single_agent_turn_types import SingleAgentTurnRequest
from nemo_gym.tasks.harbor import DIGEST_KEY, load_task


TASK_TOML = """
schema_version = "1.4"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 300.0

[environment]
cpus = 1
memory_mb = 2048
storage_mb = 10240
"""


def write_task(root: Path, *, toml: str = TASK_TOML, dockerfile: str = "FROM ubuntu:24.04\nWORKDIR /app") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "task.toml").write_text(toml)
    (root / "instruction.md").write_text("Create hello.txt\n")
    (root / "environment").mkdir(exist_ok=True)
    (root / "environment" / "Dockerfile").write_text(dockerfile)
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    return root


def archive_of(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name=f"./{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@dataclass
class FakeSandbox:
    """Records sandbox calls; ``verifier_files`` is what ``/logs/verifier`` holds after test.sh."""

    verifier_files: dict[str, str] = field(default_factory=lambda: {"reward.txt": "1\n"})
    test_result: SandboxExecResult = SandboxExecResult(stdout="", stderr="", return_code=0)
    execs: list[dict] = field(default_factory=list)
    uploads: list[tuple[Path, str]] = field(default_factory=list)
    uploaded_archives: list[bytes] = field(default_factory=list)
    stopped: bool = False

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.execs.append({"command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s, "user": user})
        if "test.sh" in command:
            return self.test_result
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def upload(self, local_path, remote_path):
        self.uploads.append((Path(local_path), remote_path))
        self.uploaded_archives.append(Path(local_path).read_bytes())

    async def download(self, remote_path, local_path):
        Path(local_path).write_bytes(archive_of(self.verifier_files))

    async def serialize(self, *, scope=None):
        return {"sandbox_id": "sb-1", "workdir": "/app"}

    async def stop(self):
        self.stopped = True


class _Cookie:
    value = "cookie-value"


class _Response:
    ok = True
    cookies = {"session": _Cookie()}

    def __init__(self, body: dict) -> None:
        self.body = orjson.dumps(body)

    async def read(self) -> bytes:
        return self.body


def agent_response() -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "id": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": "echo hello > hello.txt", "annotations": []}],
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )


class FakeAgentClient(ServerClient):
    """Stands in for the agent server: answers the session seed, the turn and the close.

    ``fail_path`` makes the next call to that path time out (a lost reply). ``slow_path`` delays
    that path so an episode deadline can expire during it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    calls: list[tuple[str, str, dict]]
    fail_path: str | None = None
    # Raise this exception (instead of losing the reply) the next time this path is called.
    fail_with: dict = {}
    slow_path: str | None = None
    observations: dict | None = None
    turn_response: dict | None = None
    # Called when the harness takes its turn, between the seed and the verification of an episode.
    turn_hooks: list = []

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Response:
        body = kwargs.get("json")
        self.calls.append((server_name, url_path, kwargs))
        if url_path.endswith("/v1/responses"):
            for hook in self.turn_hooks:
                outcome = hook()
                if inspect.isawaitable(outcome):
                    await outcome
        if url_path == self.slow_path:
            await asyncio.sleep(0.5)
        if url_path == self.fail_path:
            self.fail_path = None
            raise TimeoutError(f"response lost for {url_path}")
        if url_path in self.fail_with:
            raise self.fail_with.pop(url_path)
        if url_path == "/v1/agent_sessions":
            AgentSeedSessionRequest.model_validate(body)
            return _Response({"agent_session_id": body["agent_session_id"]})
        if url_path == "/v1/agent_sessions/close":
            return _Response({"agent_session_id": body["agent_session_id"], "agent_observations": self.observations})
        if url_path.endswith("/v1/responses"):
            return _Response(self.turn_response or agent_response().model_dump(mode="json"))
        raise AssertionError(f"unexpected agent call {url_path}")

    def seed_bodies(self) -> list[dict]:
        return [call[2]["json"] for call in self.calls if call[1] == "/v1/agent_sessions"]

    def paths(self) -> list[str]:
        return [call[1] for call in self.calls]


def make_client() -> FakeAgentClient:
    global_config = OmegaConf.create(
        {"agent": {"responses_api_agents": {"test": {"host": "agent", "port": 8001, "entrypoint": "app.py"}}}}
    )
    return FakeAgentClient(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=global_config,
        calls=[],
        turn_hooks=[],
        fail_with={},
    )


def make_server(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    sandbox: FakeSandbox | None = None,
    *,
    toml: str = TASK_TOML,
    dockerfile: str = "FROM ubuntu:24.04\nWORKDIR /app",
    **config_overrides,
):
    folder = tmp_path / "datasets" / "ds"
    task = load_task(write_task(folder / "hello", toml=toml, dockerfile=dockerfile))
    config = HarborEnvironmentServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="harbor_environment_server",
        agent_server={"type": "responses_api_agents", "name": "agent"},
        tasksets={"ds": {"folder": str(folder), "tasks": {"hello": task.digest}}},
        artifacts_dir=tmp_path / "artifacts",
        cleanup_timeout_seconds=10,
        **config_overrides,
    )
    server = HarborEnvironmentServer(config=config, server_client=make_client())
    sandbox = sandbox or FakeSandbox()
    created: list[tuple] = []

    async def create(task, workdir, *, episode_key=None):
        created.append((task.task_id, workdir))
        return sandbox

    monkeypatch.setattr(server, "_create_sandbox", create)
    return server, task, sandbox, created


def episode(task, *, rollout="r1", attempt=0, digest=None, task_id="hello", taskset="ds") -> SingleAgentTurnRequest:
    return SingleAgentTurnRequest.model_validate(
        {
            "episode_id": {"rollout_id": rollout, "attempt": attempt},
            "task": {
                "task_id": {"taskset": taskset, "task_id": task_id},
                "task_input": {
                    "responses_create_params": {"input": [{"role": "user", "content": "Create hello.txt"}]},
                    "task_data": {DIGEST_KEY: digest or task.digest},
                },
            },
        }
    )


async def run_episode(server: HarborEnvironmentServer, task, **kwargs) -> HarborEpisodeResponse:
    return await server.run_request(episode(task, **kwargs))
