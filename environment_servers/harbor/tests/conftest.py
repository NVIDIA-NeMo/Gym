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
import json
import posixpath
import re
import shlex
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import orjson
from aiohttp import ClientResponseError
from omegaconf import OmegaConf
from pydantic import ConfigDict
from pytest import MonkeyPatch

from environment_servers.harbor.app import (
    HarborEnvironmentServer,
    HarborEnvironmentServerConfig,
    HarborEpisodeRequest,
    HarborEpisodeResponse,
)
from environment_servers.harbor.verifier import EXIT_RECORD_NAME
from nemo_gym.agents import AGENT_ACTIVATION_HEADER, AgentSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.config import BaseServerConfig
from nemo_gym.core import NeMoGymResponse
from nemo_gym.data import BUILT_IMAGES_FILE, DIGEST_KEY, RECORD_VERSION, load_task
from nemo_gym.data.tasks.harbor.digest import context_digest
from nemo_gym.data.tasks.harbor.records import IMAGE_CONFIGS_FILE
from nemo_gym.sandbox import SandboxExecResult, SandboxSpec


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


# The PATH the test base images record, as Docker's default for an image whose Dockerfile set none.
IMAGE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


SOLVE_SH = "#!/bin/bash\necho hi > hello.txt\n"


def write_image_config(parent: Path, image: str, **config) -> None:
    """Record ``image``'s OCI configuration next to the task folders, as preparing the dataset does."""
    path = parent / IMAGE_CONFIGS_FILE
    recorded = json.loads(path.read_text()) if path.is_file() else {}
    recorded[image] = {
        "os": "linux",
        "architecture": "amd64",
        "image": image,
        "config": config,
        "record_version": RECORD_VERSION,
    }
    path.write_text(json.dumps(recorded))


def write_task(
    root: Path,
    *,
    toml: str = TASK_TOML,
    dockerfile: str = "FROM ubuntu:24.04\nWORKDIR /app",
    image_config: dict | None = None,
    solution: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "task.toml").write_text(toml)
    (root / "instruction.md").write_text("Create hello.txt\n")
    (root / "environment").mkdir(exist_ok=True)
    (root / "environment" / "Dockerfile").write_text(dockerfile)
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    if solution:
        # A reference solution with a subfolder, an author's private mode on the script, and an editor
        # leftover the task digest skips; the digest covers solution/, so it is written before the load.
        (root / "solution").mkdir(exist_ok=True)
        (root / "solution" / "solve.sh").write_text(SOLVE_SH)
        (root / "solution" / "solve.sh").chmod(0o700)
        (root / "solution" / "data").mkdir(exist_ok=True)
        (root / "solution" / "data" / "answer.txt").write_text("42\n")
        (root / "solution" / "__pycache__").mkdir(exist_ok=True)
        (root / "solution" / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    # The loader resolves the Dockerfile against its base image's configuration, recorded at prepare.
    write_image_config(root.parent, dockerfile.split()[1], **(image_config or {"Env": [f"PATH={IMAGE_PATH}"]}))
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


OK = SandboxExecResult(stdout="", stderr="", return_code=0)


def _fails(message: str) -> SandboxExecResult:
    return SandboxExecResult(stdout="", stderr=message, return_code=1)


@dataclass
class FakeSandbox:
    """Records sandbox calls and behaves like the container they would reach.

    ``verifier_files`` is what ``/logs/verifier`` holds after test.sh. The verifier wrapper's exit
    record is added when the wrapper ran to its end: ``test_elapsed`` is how long the run took inside
    the container. A provider timeout, or ``wrapper_ran=False`` (no ``timeout`` or ``bash`` in the
    image), leaves no record. The container side: a command runs as the exec's ``user``, else as the
    image's default user; ``spec_env`` is the environment the sandbox was created with, and the
    ``unset``/``export``/``cd`` prefix the server puts before a Dockerfile ``RUN`` line is applied to
    it, so each exec's record carries under ``seen`` the command, environment, user and directory the
    container would run. Directories exist only once something made them (``dirs`` maps each to its
    owner; a non-root user can only create under its own directories, and ``cd`` into a missing one
    fails), and ``passwd`` is the image's uid-to-name table.
    """

    verifier_files: dict[str, str] = field(default_factory=lambda: {"reward.txt": "1\n"})
    test_result: SandboxExecResult = OK
    test_elapsed: int = 1
    wrapper_ran: bool = True
    execs: list[dict] = field(default_factory=list)
    uploads: list[tuple[Path, str]] = field(default_factory=list)
    uploaded_archives: list[bytes] = field(default_factory=list)
    stopped: bool = False
    spec_env: dict[str, str] = field(default_factory=dict)
    image_user: str = "root"
    dirs: dict[str, str] = field(default_factory=lambda: {"/": "root", "/tmp": "root", "/home": "root"})
    passwd: dict[str, str] = field(default_factory=lambda: {"0": "root", "1000": "agent"})

    default_user: str | int | None = None

    def set_default_user(self, user):
        self.default_user = user

    def run(self, fragment: str) -> dict:
        """The record of the one exec whose command, prefix aside, holds ``fragment``."""
        (record,) = [c for c in self.execs if c.get("seen") and fragment in c["seen"]["command"]]
        return record

    def created_in(self, workdir: str | None) -> None:
        """What a provider does with the spec's workdir: the directory exists, root-owned, before anything runs."""
        if workdir is not None:
            self._mkdir(workdir, "root")

    def _mkdir(self, path: str, user: str) -> SandboxExecResult:
        if path in self.dirs:
            return OK
        parent = posixpath.dirname(path)
        while parent not in self.dirs:
            parent = posixpath.dirname(parent)
        if user.split(":")[0] not in ("root", "0") and self.dirs[parent] != user:
            return _fails(f"mkdir: cannot create directory '{path}': Permission denied")
        while path not in self.dirs:
            self.dirs[path] = user
            path = posixpath.dirname(path)
        return OK

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        # The facade's rule (`AsyncSandbox.exec`): an exec that names no user runs as the sandbox's default, and
        # a sandbox with no default runs as the image's own user. `user` is the raw argument, `ran_as` the user
        # the container ran the command as.
        effective = user if user is not None else self.default_user
        who = str(effective) if effective is not None else self.image_user
        record = {
            "command": command,
            "cwd": cwd,
            "env": env,
            "timeout_s": timeout_s,
            "user": user,
            "ran_as": who,
            "seen": None,
        }
        self.execs.append(record)
        if "test.sh" in command:
            if "timeout" not in command:
                # Harbor's `chmod +x` on test.sh before the run, or a separate verifier's probe for the file.
                return SandboxExecResult(stdout="", stderr="", return_code=0)
            if self.wrapper_ran and self.test_result.error_type != "timeout":
                start = 1_700_000_000
                self.verifier_files.setdefault(
                    EXIT_RECORD_NAME, f"{self.test_result.return_code} {start} {start + self.test_elapsed}\n"
                )
            return self.test_result
        if command == "pwd":
            # The image's own WORKDIR, asked for when a task declares none.
            return SandboxExecResult(stdout="/app\n", stderr="", return_code=0)
        if match := re.fullmatch(r"getent passwd (\S+) \| cut -d: -f1", command):
            name = self.passwd.get(shlex.split(match.group(1))[0], "")
            return SandboxExecResult(stdout=f"{name}\n" if name else "", stderr="", return_code=0)
        if match := re.fullmatch(r"stat -c %u:%g (\S+)", command):
            # The owner's uid and gid (one group per user here); a name the passwd table lacks is its own uid.
            owner = self.dirs.get(shlex.split(match.group(1))[0], "root")
            uid = next((uid for uid, name in self.passwd.items() if name == owner), owner)
            return SandboxExecResult(stdout=f"{uid}:{uid}\n", stderr="", return_code=0)
        if match := re.fullmatch(r"chown (\S+) (\S+)", command):
            owner, path = shlex.split(match.group(1))[0], shlex.split(match.group(2))[0]
            if who.split(":")[0] not in ("root", "0"):
                return _fails(f"chown: changing ownership of '{path}': Operation not permitted")
            uid = owner.split(":")[0]
            self.dirs[path] = self.passwd.get(uid, uid)
            return OK
        if match := re.fullmatch(r"test -d (\S+) \|\| \(mkdir -p \1 && chown (\S+) \1\)", command):
            path, owner = shlex.split(match.group(1))[0], shlex.split(match.group(2))[0]
            if path in self.dirs:
                return OK
            result = self._mkdir(path, who)
            if result.return_code == 0:
                self.dirs[path] = owner
            return result
        if match := re.fullmatch(r"mkdir -p (\S+)(?: && chown (\S+) \1)?", command):
            path = shlex.split(match.group(1))[0]
            result = self._mkdir(path, who)
            if result.return_code == 0 and match.group(2):
                self.dirs[path] = shlex.split(match.group(2))[0]
            return result
        # Anything else runs as a shell line: the server's `unset`/`export`/`cd` prefix shapes what the
        # command itself sees.
        statements = command.split("; ")
        seen_env, seen_cwd = dict(self.spec_env), cwd
        while statements and statements[0].startswith(("unset ", "export ")):
            words = shlex.split(statements.pop(0))
            if words[0] == "unset":
                seen_env.pop(words[1], None)
            else:
                key, _, value = words[1].partition("=")
                seen_env[key] = value
        if statements and statements[0].startswith("cd ") and statements[0].endswith(" || exit 1"):
            seen_cwd = shlex.split(statements.pop(0))[1]
            if seen_cwd not in self.dirs:
                return _fails(f"cd: {seen_cwd}: No such file or directory")
        rest = "; ".join(statements)
        if rest.startswith("exec /bin/sh -c "):
            # A default-shell step names its shell explicitly; what the step sees is the quoted line.
            rest = shlex.split(rest)[3]
        record["seen"] = {"command": rest, "env": seen_env, "user": who, "cwd": seen_cwd}
        # Who owned the directory the command ran in at that moment (a build step fails there otherwise).
        record["cwd_owner"] = self.dirs.get(seen_cwd)
        return OK

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


def _http_error(status: int, message: str) -> ClientResponseError:
    """What ``raise_for_status`` raises for a non-2xx agent reply."""
    error = ClientResponseError(SimpleNamespace(real_url=f"http://agent/{status}"), (), status=status, message=message)
    error.response_content = message.encode()
    return error


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
    that path by ``slow_seconds`` so an episode deadline can expire during it, or so a turn costs real time
    within a budget a test sizes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    calls: list[tuple[str, str, dict]]
    fail_path: str | None = None
    # Raise this exception (instead of losing the reply) the next time this path is called.
    fail_with: dict = {}
    slow_path: str | None = None
    slow_seconds: float = 0.5
    observations: dict | None = None
    trajectory: dict | None = None
    # What the seed response declares: a harness that reads the task's MCP servers and skills directory.
    supports_task_tool_context: bool = True
    # What GET /v1/agent_sessions/capabilities and a continuation seed declare: None is a harness without
    # multi-turn continuation (OpenCode declares one). ``capabilities_status`` 404 stands for an agent server
    # built before the route existed.
    continuation_capabilities: dict | None = None
    capabilities_status: int = 200
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
            await asyncio.sleep(self.slow_seconds)
        if url_path == self.fail_path:
            self.fail_path = None
            raise TimeoutError(f"response lost for {url_path}")
        if url_path in self.fail_with:
            raise self.fail_with.pop(url_path)
        if url_path == "/v1/agent_sessions":
            AgentSeedSessionRequest.model_validate(body)
            if body.get("continuation") is not None and self.continuation_capabilities is None:
                raise _http_error(422, "Agent does not support the required multi-turn continuation mode")
            return _Response(
                {
                    "agent_session_id": body["agent_session_id"],
                    "supports_task_tool_context": self.supports_task_tool_context,
                    **(
                        {"capabilities": self.continuation_capabilities}
                        if body.get("continuation") is not None
                        else {}
                    ),
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

    async def get(self, server_name: str, url_path: str, **kwargs) -> _Response:
        self.calls.append((server_name, url_path, kwargs))
        if url_path == "/v1/agent_sessions/capabilities":
            if self.capabilities_status != 200:
                raise _http_error(self.capabilities_status, "Not Found")
            return _Response(
                {"supports_task_tool_context": self.supports_task_tool_context}
                | ({"continuation": self.continuation_capabilities} if self.continuation_capabilities else {})
            )
        raise AssertionError(f"unexpected agent GET {url_path}")

    def seed_bodies(self) -> list[dict]:
        return [call[2]["json"] for call in self.calls if call[1] == "/v1/agent_sessions"]

    def activation_headers(self) -> list[str | None]:
        """The activation header of each turn call, in order (None when the turn named none)."""
        return [
            (call[2].get("headers") or {}).get(AGENT_ACTIVATION_HEADER)
            for call in self.calls
            if call[1].endswith("/v1/responses")
        ]

    def paths(self) -> list[str]:
        return [call[1] for call in self.calls]


def make_client(implementation: str = "test") -> FakeAgentClient:
    """``implementation`` is the key under the agent instance's ``responses_api_agents`` block in the run config:
    the server reads it to tell the oracle (``harbor_oracle_agent``) from every other harness."""
    global_config = OmegaConf.create(
        {"agent": {"responses_api_agents": {implementation: {"host": "agent", "port": 8001, "entrypoint": "app.py"}}}}
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
    image_config: dict | None = None,
    agent_implementation: str = "test",
    solution: bool = False,
    **config_overrides,
):
    folder = tmp_path / "datasets" / "ds"
    task = load_task(
        write_task(folder / "hello", toml=toml, dockerfile=dockerfile, image_config=image_config, solution=solution)
    )
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
    server = HarborEnvironmentServer(config=config, server_client=make_client(agent_implementation))
    sandbox = sandbox or FakeSandbox()
    created: list[tuple] = []

    async def create(task, workdir, *, episode_key=None):
        created.append((task.task_id, workdir))
        sandbox.created_in(workdir)
        return sandbox

    monkeypatch.setattr(server, "_create_sandbox", create)
    return server, task, sandbox, created


def episode(task, *, rollout="r1", attempt=0, digest=None, task_id="hello", taskset="ds") -> HarborEpisodeRequest:
    return HarborEpisodeRequest.model_validate(
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


OVERLAY_DOCKERFILE = """\
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y curl
WORKDIR /app
ENV PATH="/opt/tools/bin:$PATH" MSG="hi there"
RUN mkdir -p /opt/tools/bin && echo ok > ready.txt
USER agent
RUN echo "as agent" > /tmp/who
ENTRYPOINT []
"""

HEALTHCHECK_TOML = (
    TASK_TOML
    + """
[environment.healthcheck]
command = "test -f /app/ready.txt"
interval_sec = 0.01
start_interval_sec = 0.01
timeout_sec = 5.0
retries = 3
"""
)


@dataclass
class OverlaySandbox(FakeSandbox):
    """Fails the RUN line whose command holds ``failing`` with the given result."""

    failing: str | None = None
    failure: SandboxExecResult = SandboxExecResult(
        stdout="line 1\nline 2\n", stderr="E: no package\n", return_code=100
    )
    spec: SandboxSpec | None = None

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        result = await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)
        seen = self.execs[-1]["seen"]
        if self.failing and seen and self.failing in seen["command"]:
            return self.failure
        return result


SEPARATE_IN_TASK_IMAGE_TOML = TASK_TOML.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n')

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


COMPOSE_YAML = """
services:
  main:
    image: org/agent:1
  api:
    image: org/api:1
    expose: ["3080"]
  kafka:
    image: org/kafka:1
"""

IMAGE_CONFIGS = {
    "org/agent:1": {
        "os": "linux",
        "architecture": "amd64",
        "image": "org/agent@sha256:" + "a" * 64,
        "config": {},
        "record_version": RECORD_VERSION,
    },
    "org/api:1": {
        "os": "linux",
        "architecture": "amd64",
        "image": "org/api@sha256:" + "b" * 64,
        "config": {"Cmd": ["serve"]},
        "record_version": RECORD_VERSION,
    },
    "org/kafka:1": {
        "os": "linux",
        "architecture": "amd64",
        "image": "org/kafka@sha256:" + "c" * 64,
        "config": {"Cmd": ["kafka"], "User": "appuser"},
        "record_version": RECORD_VERSION,
    },
}


class FakeCompose:
    def __init__(self, services):
        self.services = services
        self.stopped = False

    async def stop(self):
        self.stopped = True


VERIFIER_TESTS_DOCKERFILE = "FROM python:3.13-slim-bookworm\nRUN pip install pytest==8.4.1\nCOPY . /tests/\n"


def record_verifier_build(task_folder: Path, image: str, **config) -> str:
    """Record ``image`` as the verifier image prepare built from ``tests/Dockerfile``: under Harbor's hash of ``tests/``
    as service ``verifier`` in ``built-images.json``, its configuration in ``compose-images.json``."""
    parent = task_folder.parent
    digest = context_digest(task_folder / "tests")
    path = parent / BUILT_IMAGES_FILE
    built = json.loads(path.read_text()) if path.is_file() else {}
    built[digest] = {
        "tasks": [task_folder.name],
        "tool": "buildah",
        "pushed": True,
        "services": {"verifier": {"image": image, "tool": "buildah", "pushed": True}},
    }
    path.write_text(json.dumps(built))
    write_image_config(parent, image, **(config or {"Env": [f"PATH={IMAGE_PATH}"], "WorkingDir": "/tests"}))
    return digest
