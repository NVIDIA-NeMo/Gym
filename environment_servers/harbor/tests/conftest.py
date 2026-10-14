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
from environment_servers.harbor.verifier import EXIT_RECORD_NAME
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
    """Records sandbox calls; ``verifier_files`` is what ``/logs/verifier`` holds after test.sh.

    The verifier wrapper's exit record is added when the wrapper ran to its end: ``test_elapsed`` is how
    long the run took inside the container. A provider timeout, or ``wrapper_ran=False`` (no ``timeout``
    or ``bash`` in the image), leaves no record.
    """

    verifier_files: dict[str, str] = field(default_factory=lambda: {"reward.txt": "1\n"})
    test_result: SandboxExecResult = SandboxExecResult(stdout="", stderr="", return_code=0)
    test_elapsed: int = 1
    wrapper_ran: bool = True
    execs: list[dict] = field(default_factory=list)
    uploads: list[tuple[Path, str]] = field(default_factory=list)
    uploaded_archives: list[bytes] = field(default_factory=list)
    stopped: bool = False

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.execs.append({"command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s, "user": user})
        if "test.sh" in command:
            if self.wrapper_ran and self.test_result.error_type != "timeout":
                start = 1_700_000_000
                self.verifier_files.setdefault(
                    EXIT_RECORD_NAME, f"{self.test_result.return_code} {start} {start + self.test_elapsed}\n"
                )
            return self.test_result
        if command == "pwd":
            # The image's own WORKDIR, asked for when a task declares none.
            return SandboxExecResult(stdout="/app\n", stderr="", return_code=0)
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
    trajectory: dict | None = None
    # What the seed response declares: a harness that reads the task's MCP servers and skills directory.
    supports_task_tool_context: bool = True
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
            return _Response(
                {
                    "agent_session_id": body["agent_session_id"],
                    "supports_task_tool_context": self.supports_task_tool_context,
                }
            )
        if url_path == "/v1/agent_sessions/close":
            return _Response(
                {
                    "agent_session_id": body["agent_session_id"],
                    "agent_observations": self.observations,
                    "trajectory": self.trajectory,
                }
            )
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


SEPARATE_TOML = """
schema_version = "1.4"

artifacts = ["/app/output/report.json", { source = "/var/log/api", service = "api" }]

[verifier]
timeout_sec = 300.0
user = "root"
environment_mode = "separate"

[verifier.env]
CHECK = "strict"

[verifier.environment]
docker_image = "org/verifier:1"
cpus = 2
memory_mb = 4096

[[verifier.collect]]
command = "cp /app/state.db /logs/artifacts/state.db"
timeout_sec = 10.0

[[verifier.collect]]
command = "kafka-dump"
service = "kafka"

[agent]
timeout_sec = 120.0

[environment]
docker_image = "org/agent:1"
cpus = 1
"""


@dataclass
class AgentSandbox(FakeSandbox):
    """The agent's sandbox in separate mode: holds /logs/artifacts and one report file.

    ``archives`` maps a folder the server archives (``tar ... -C <folder>`` or ``cd <folder>``) to the
    archive bytes the fake hands back for it; ``probe_result`` replaces every probe answer when set.
    """

    present: dict[str, str] = field(
        default_factory=lambda: {"/logs/artifacts": "dir", "/app/output/report.json": "file"}
    )
    archives: dict[str, bytes] = field(
        default_factory=lambda: {
            "/logs/artifacts": archive_of({"state.db": "db"}),
            "/app/output": archive_of({"report.json": '{"ok": true}'}),
        }
    )
    probe_result: SandboxExecResult | None = None
    archived: list[str] = field(default_factory=list)

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.execs.append({"command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s, "user": user})
        if command.startswith("if [ -d "):
            if self.probe_result is not None:
                return self.probe_result
            path = command.split("if [ -d ")[1].split(" ]")[0].strip("'")
            return SandboxExecResult(stdout=self.present.get(path, "none") + "\n", stderr="", return_code=0)
        if "tar -czf" in command:
            marker = " -C " if " -C " in command else "cd "
            self.archived.append(command.split(marker)[1].split(" ")[0].strip("'"))
        self.execs.pop()  # recorded again by the base fake
        return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

    async def download(self, remote_path, local_path):
        Path(local_path).write_bytes(self.archives[self.archived[-1]])

    async def test_a_probe_the_sandbox_cannot_answer_masks_but_absence_does_not(self, tmp_path, monkeypatch):
        server, task, agent, verifier, created = self.prepared(tmp_path, monkeypatch)
        agent.probe_result = SandboxExecResult(
            stdout="", stderr="container gone", return_code=125, error_type="sandbox"
        )
        result = (await run_episode(server, task)).result
        # The outage is Gym's, not an empty artifact: masked, and no verifier was started for it.
        assert result.mask_sample is True and result.failure_kind == "provider_unavailable"
        assert "SandboxUnavailableError" in result.failure_reason and "container gone" in result.failure_reason
        assert created == [] and verifier.execs == [] and agent.stopped

        server, task, agent, verifier, _ = self.prepared(tmp_path, monkeypatch)
        agent.probe_result = SandboxExecResult(stdout="none\n", stderr="", return_code=0)
        result = (await run_episode(server, task)).result
        # A plain "none" is still absence: nothing restored, the verifier runs and scores.
        assert result.reward == 1.0 and not result.mask_sample
        assert not any("tar -xzf" in c["command"] and "/logs/artifacts" in c["command"] for c in verifier.execs)

    async def test_artifact_exclusions_reach_the_archive_command(self, tmp_path, monkeypatch):
        toml = SEPARATE_TOML.replace(
            'artifacts = ["/app/output/report.json", { source = "/var/log/api", service = "api" }]',
            'artifacts = [{ source = "/app/build", exclude = ["node_modules", "./cache", "*.log"] }]',
        )
        agent = AgentSandbox(
            present={"/logs/artifacts": "dir", "/app/build": "dir"},
            archives={"/logs/artifacts": archive_of({}), "/app/build": archive_of({"a.out": "elf"})},
        )
        server, task, _, _, _ = self.prepared(tmp_path, monkeypatch, toml=toml, agent=agent)
        result = (await run_episode(server, task)).result
        assert result.reward == 1.0, result.failure_reason
        build = next(c["command"] for c in agent.execs if "tar -czf" in c["command"] and "/app/build" in c["command"])
        # Verbatim, unanchored, as Harbor passes them: `node_modules` at any depth, `./cache` at the top only.
        assert " --exclude=node_modules --exclude=./cache --exclude='*.log' -C /app/build ." in build
        artifacts_archive = next(
            c["command"] for c in agent.execs if "tar -czf" in c["command"] and "/logs/artifacts" in c["command"]
        )
        assert "--exclude" not in artifacts_archive

    async def test_each_artifact_stages_under_its_own_root(self, tmp_path, monkeypatch):
        """A link kept from one artifact is never a path component of another artifact's destination."""
        victim = tmp_path / "victim"
        victim.mkdir()
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            link = tarfile.TarInfo(name="./output")
            link.type, link.linkname = tarfile.SYMTYPE, str(victim)
            tar.addfile(link)
        toml = SEPARATE_TOML.replace(
            'artifacts = ["/app/output/report.json", { source = "/var/log/api", service = "api" }]',
            'artifacts = ["/app", "/app/output/report.json"]',
        )
        agent = AgentSandbox(
            present={"/logs/artifacts": "none", "/app": "dir", "/app/output/report.json": "file"},
            archives={"/app": buffer.getvalue(), "/app/output": archive_of({"report.json": '{"ok": true}'})},
        )
        server, task, _, _, _ = self.prepared(tmp_path, monkeypatch, toml=toml, agent=agent)
        result = (await run_episode(server, task)).result
        assert result.reward == 1.0, result.failure_reason
        artifacts = Path(result.verifier_logs_dir) / "artifacts"
        # The link travelled as a link under the first artifact's root; the report landed under the second.
        assert (artifacts / "01" / "app" / "output").is_symlink()
        assert (artifacts / "02" / "app" / "output" / "report.json").read_bytes() == b'{"ok": true}'
        assert list(victim.iterdir()) == []
