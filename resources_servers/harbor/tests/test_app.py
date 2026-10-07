# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxSpec
from nemo_gym.server_utils import ServerClient
from nemo_gym.tasks.harbor import DIGEST_KEY, load_task
from nemo_gym.tasks.harbor.dockerfile import OverlayRun
from nemo_gym.tasks.harbor.models import HarborEnvironment
from nemo_gym.tasks.harbor.task import IMAGE_CONFIGS_FILE
from resources_servers.harbor.app import (
    OVERLAY_FAILED_KIND,
    HarborResourcesServer,
    HarborResourcesServerConfig,
    _verifier_image,
    overlay_shell_command,
    overlay_workdir_command,
    parse_reward_file,
    select_reward,
)


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


def write_image_config(parent: Path, image: str, **config) -> None:
    """Record ``image``'s OCI configuration next to the task folders, as preparing the dataset does."""
    path = parent / IMAGE_CONFIGS_FILE
    recorded = json.loads(path.read_text()) if path.is_file() else {}
    recorded[image] = {"os": "linux", "architecture": "amd64", "image": image, "config": config}
    path.write_text(json.dumps(recorded))


def write_task(
    root: Path, *, dockerfile: str = "FROM ubuntu:24.04\nWORKDIR /app", image_config: dict | None = None
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "task.toml").write_text(TASK_TOML)
    (root / "instruction.md").write_text("Create hello.txt\n")
    (root / "environment").mkdir(exist_ok=True)
    (root / "environment" / "Dockerfile").write_text(dockerfile)
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
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

    ``verifier_files`` is what ``/logs/verifier`` holds after test.sh. The container side: a
    command runs as the exec's ``user``, else as the image's default user; ``spec_env`` is the
    environment the sandbox was created with, and the ``unset``/``export``/``cd`` prefix the server
    puts before a Dockerfile ``RUN`` line is applied to it, so each exec's record carries under
    ``seen`` the command, environment, user and directory the container would run. Directories
    exist only once something made them (``dirs`` maps each to its owner; a non-root user can
    only create under its own directories, and ``cd`` into a missing one fails), and ``passwd``
    is the image's uid-to-name table.
    """

    verifier_files: dict[str, str] = field(default_factory=lambda: {"reward.txt": "1\n"})
    test_result: SandboxExecResult = OK
    execs: list[dict] = field(default_factory=list)
    uploads: list[tuple[Path, str]] = field(default_factory=list)
    stopped: bool = False
    spec_env: dict[str, str] = field(default_factory=dict)
    image_user: str = "root"
    dirs: dict[str, str] = field(default_factory=lambda: {"/": "root", "/tmp": "root", "/home": "root"})
    passwd: dict[str, str] = field(default_factory=lambda: {"0": "root", "1000": "agent"})

    def run(self, fragment: str) -> dict:
        """The record of the one exec whose command, prefix aside, holds ``fragment``."""
        (record,) = [c for c in self.execs if c.get("seen") and fragment in c["seen"]["command"]]
        return record

    def _mkdir(self, path: str, user: str) -> SandboxExecResult:
        if path in self.dirs:
            return OK
        parent = posixpath.dirname(path)
        while parent not in self.dirs:
            parent = posixpath.dirname(parent)
        if user != "root" and self.dirs[parent] != user:
            return _fails(f"mkdir: cannot create directory '{path}': Permission denied")
        while path not in self.dirs:
            self.dirs[path] = user
            path = posixpath.dirname(path)
        return OK

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        record = {"command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s, "user": user, "seen": None}
        self.execs.append(record)
        if "test.sh" in command:
            return self.test_result
        who = str(user) if user is not None else self.image_user
        if match := re.fullmatch(r"getent passwd (\S+) \| cut -d: -f1", command):
            name = self.passwd.get(shlex.split(match.group(1))[0], "")
            return SandboxExecResult(stdout=f"{name}\n" if name else "", stderr="", return_code=0)
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
        record["seen"] = {"command": "; ".join(statements), "env": seen_env, "user": who, "cwd": seen_cwd}
        return OK

    async def upload(self, local_path, remote_path):
        self.uploads.append((Path(local_path), remote_path))

    async def download(self, remote_path, local_path):
        Path(local_path).write_bytes(archive_of(self.verifier_files))

    async def serialize(self, *, scope=None):
        return {"sandbox_id": "sb-1", "workdir": "/app"}

    async def stop(self):
        self.stopped = True


def make_server(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    sandbox: FakeSandbox | None = None,
    *,
    dockerfile: str = "FROM ubuntu:24.04\nWORKDIR /app",
    image_config: dict | None = None,
):
    folder = tmp_path / "datasets" / "ds"
    task = load_task(write_task(folder / "hello", dockerfile=dockerfile, image_config=image_config))
    config = HarborResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="harbor_resources_server",
        tasksets={"ds": {"folder": str(folder), "tasks": {"hello": task.digest}}},
        artifacts_dir=tmp_path / "artifacts",
    )
    server = HarborResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    sandbox = sandbox or FakeSandbox()
    created: list[tuple] = []

    async def create(task, workdir):
        created.append((task.task_id, workdir))
        return sandbox

    monkeypatch.setattr(server, "_create_sandbox", create)
    return server, task, sandbox, created


def seed_body(task, *, session="rs-1", digest=None, task_id="hello", taskset="ds"):
    return {
        "resources_session_id": session,
        "episode_id": {"rollout_id": "r1", "attempt": 0},
        "task_id": {"taskset": taskset, "task_id": task_id},
        "task_data": {DIGEST_KEY: digest or task.digest},
    }


def verify_body(*, digest=None):
    """The flat verify body the environment server posts: task_data keys beside the params and response."""
    body = {
        "responses_create_params": {"input": [{"role": "user", "content": "Create hello.txt"}]},
        "response": {
            "output": [],
            "id": "resp",
            "created_at": 0,
            "model": "m",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        },
    }
    if digest is not None:
        body[DIGEST_KEY] = digest
    return body


class TestRewardFile:
    def test_text_and_json(self, tmp_path):
        (tmp_path / "reward.txt").write_text("0.5\n")
        assert parse_reward_file(tmp_path) == ({"reward": 0.5}, None)
        (tmp_path / "reward.json").write_text(json.dumps({"accuracy": 1, "speed": 0.25}))
        rewards, problem = parse_reward_file(tmp_path)
        assert problem is None and rewards == {"accuracy": 1.0, "speed": 0.25}
        assert select_reward(rewards) is None
        assert select_reward({"accuracy": 0.25}) == 0.25
        assert select_reward({"reward": 1.0, "other": 0.0}) == 1.0

    @pytest.mark.parametrize(
        ("name", "text", "fragment"),
        [
            ("reward.txt", "", "empty"),
            ("reward.txt", "yes", "not valid"),
            ("reward.json", "[1]", "non-empty JSON object"),
            ("reward.json", '{"reward": "1"}', "not a finite number"),
            ("reward.json", '{"reward": true}', "not a finite number"),
        ],
    )
    def test_problems(self, tmp_path, name, text, fragment):
        (tmp_path / name).write_text(text)
        rewards, problem = parse_reward_file(tmp_path)
        assert rewards is None and fragment in problem

    def test_missing(self, tmp_path):
        assert parse_reward_file(tmp_path)[1] == "no reward.json or reward.txt was written"

    def test_invalid_json_falls_back_to_text(self, tmp_path):
        (tmp_path / "reward.json").write_text("{not json")
        (tmp_path / "reward.txt").write_text("0.5\n")
        assert parse_reward_file(tmp_path) == ({"reward": 0.5}, None)

        # A well-formed but unusable reward.json also falls back.
        (tmp_path / "reward.json").write_text("[]")
        assert parse_reward_file(tmp_path) == ({"reward": 0.5}, None)

        # When both are unusable, the problem names both files.
        (tmp_path / "reward.txt").write_text("maybe")
        rewards, problem = parse_reward_file(tmp_path)
        assert rewards is None
        assert "reward.json must hold a non-empty JSON object" in problem
        assert "reward.txt is not valid" in problem

    def test_non_utf8_bytes_do_not_raise(self, tmp_path):
        (tmp_path / "reward.txt").write_bytes(b"\xff\xfe1\n")
        rewards, problem = parse_reward_file(tmp_path)
        assert rewards is None and "reward.txt is not valid" in problem

        (tmp_path / "reward.json").write_bytes(b'{"reward": 1}\xff')
        (tmp_path / "reward.txt").write_text("0.25\n")
        assert parse_reward_file(tmp_path) == ({"reward": 0.25}, None)


class TestSeed:
    def test_seed_starts_sandbox_and_returns_access(self, tmp_path, monkeypatch):
        server, task, sandbox, created = make_server(tmp_path, monkeypatch)
        client = TestClient(server.setup_webserver())

        response = client.post("/seed_session", json=seed_body(task))

        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["resources_session_id"] == "rs-1"
        assert payload["sandbox_access"] == {
            "connection": {
                "kind": "direct",
                "provider_config_ref": "sandbox",
                "descriptor": {"sandbox_id": "sb-1", "workdir": "/app"},
            },
            "workdir": "/app",
        }
        assert created == [("hello", "/app")]
        assert sandbox.execs[0]["command"] == "mkdir -p /app"

        # Re-seeding the same session is idempotent.
        again = client.post("/seed_session", json=seed_body(task))
        assert again.status_code == 200 and created == [("hello", "/app")]

        # Another episode cannot reuse the session id.
        other = seed_body(task)
        other["episode_id"]["rollout_id"] = "r2"
        assert client.post("/seed_session", json=other).status_code == 409

    def test_seed_parses_and_hashes_the_task_once(self, tmp_path, monkeypatch):
        server, task, _, created = make_server(tmp_path, monkeypatch)
        client = TestClient(server.setup_webserver())
        loads: list[Path] = []
        real_load_task = load_task

        def counting_load_task(folder):
            loads.append(Path(folder))
            return real_load_task(folder)

        monkeypatch.setattr("resources_servers.harbor.app.load_task", counting_load_task)

        assert client.post("/seed_session", json=seed_body(task)).status_code == 200
        second = seed_body(task, session="rs-2")
        second["episode_id"]["rollout_id"] = "r2"
        assert client.post("/seed_session", json=second).status_code == 200

        assert created == [("hello", "/app"), ("hello", "/app")]
        assert loads == [task.path]

    def test_seed_rejects_bad_identity(self, tmp_path, monkeypatch):
        server, task, _, created = make_server(tmp_path, monkeypatch)
        client = TestClient(server.setup_webserver())

        assert client.post("/seed_session", json=seed_body(task, taskset="nope")).status_code == 404
        assert client.post("/seed_session", json=seed_body(task, task_id="missing")).status_code == 404
        assert client.post("/seed_session", json=seed_body(task, digest="0" * 64)).status_code == 422
        assert created == []

    def test_seed_rejects_changed_folder(self, tmp_path, monkeypatch):
        server, task, _, created = make_server(tmp_path, monkeypatch)
        (task.path / "tests" / "test.sh").write_text("#!/bin/bash\necho 0 > /logs/verifier/reward.txt\n")

        response = TestClient(server.setup_webserver()).post("/seed_session", json=seed_body(task))

        assert response.status_code == 409
        assert "changed since materialization" in response.json()["detail"]
        assert created == []

    def test_seed_reports_sandbox_failure_as_retryable(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)

        async def boom(task, workdir):
            raise RuntimeError("pull failed")

        monkeypatch.setattr(server, "_create_sandbox", boom)
        response = TestClient(server.setup_webserver()).post("/seed_session", json=seed_body(task))
        assert response.status_code == 503
        assert "pull failed" in response.json()["detail"]

    def test_seed_after_close_is_rejected(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        client = TestClient(server.setup_webserver())
        assert client.post("/seed_session", json=seed_body(task)).status_code == 200

        close = client.post(
            "/close_session",
            json={"resources_session_id": "rs-1", "episode_id": {"rollout_id": "r1", "attempt": 0}},
        )
        assert close.status_code == 200 and sandbox.stopped

        assert client.post("/seed_session", json=seed_body(task)).status_code == 409

    def test_sandbox_spec_from_task(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "resources_servers.harbor.app.get_global_config_dict",
            lambda: {"sandbox": {"opensandbox": {}, "default_metadata": {"sandbox-api": "osb"}}},
        )

        spec = server._sandbox_spec(task, "/app")

        assert spec.image == "ubuntu:24.04"
        assert spec.workdir == "/app"
        assert spec.ttl_s == 300 + 120 + server.config.sandbox_ttl_slack_s
        assert (spec.resources.cpu, spec.resources.memory_mib, spec.resources.disk_gib) == (1.0, 2048, 10)
        assert spec.metadata["sandbox-api"] == "osb"
        assert spec.metadata["harbor_task"] == "hello"


class TestSeedWorkdirAndResources:
    def test_seed_creates_the_logs_folders_and_per_task_env(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        # Per-task environment comes from the dataset's own file, applied by the loader; the server sees plain tasks.
        (task.path.parent / "dataset.toml").write_text(
            '[gym.tasks."hello".environment]\nenv = { CIRCLE_NODE_TOTAL = "3" }\n[gym.tasks."other".environment]\nenv = { X = "1" }\n'
        )
        task = load_task(task.path)
        monkeypatch.setattr(
            "resources_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        assert TestClient(server.setup_webserver()).post("/seed_session", json=seed_body(task)).status_code == 200
        commands = [c["command"] for c in sandbox.execs]
        assert commands[0] == "mkdir -p /app"
        logs = next(c for c in sandbox.execs if "/logs/artifacts" in c["command"])
        assert "chmod 777 /logs/agent /logs/verifier /logs/artifacts" in logs["command"] and logs["user"] == "root"
        spec = server._sandbox_spec(task, "/app")
        assert spec.env["CIRCLE_NODE_TOTAL"] == "3" and "X" not in spec.env

    def test_image_workdir_used_when_task_sets_none(self, tmp_path, monkeypatch):
        server, task, sandbox, created = make_server(tmp_path, monkeypatch)
        # A task with a real Dockerfile and a prebuilt image declares no workdir; the image's WORKDIR is used.
        (task.path / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nRUN true\n")
        (task.path / "task.toml").write_text(TASK_TOML.replace("cpus = 1", 'docker_image = "org/task:1"\ncpus = 1'))
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        assert task.workdir is None

        class PwdSandbox(FakeSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if command == "pwd":
                    return SandboxExecResult(stdout="/work\n", stderr="", return_code=0)
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

        pwd_sandbox = PwdSandbox()

        async def create(task, workdir):
            created.append((task.task_id, workdir))
            return pwd_sandbox

        monkeypatch.setattr(server, "_create_sandbox", create)
        response = TestClient(server.setup_webserver()).post("/seed_session", json=seed_body(task))

        assert response.status_code == 200, response.text
        assert created == [("hello", None)]
        assert response.json()["sandbox_access"]["workdir"] == "/work"

    def test_resources_override_and_no_injected_env(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "resources_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        spec = server._sandbox_spec(task, "/app")
        # Like Docker and Harbor, the sandbox gets exactly the task's [environment].env and nothing derived.
        assert spec.resources.cpu == 1.0 and spec.env == {}

        server.config.sandbox_resources_override = {"cpu": 4, "memory_mib": 16384, "disk_gib": 30}
        spec = server._sandbox_spec(task, "/app")
        assert (spec.resources.cpu, spec.resources.memory_mib, spec.resources.disk_gib) == (4.0, 16384, 30)
        assert spec.env == {}

        # The override merges over the task's resources, so a GPU request survives a CPU/memory override.
        task.config.environment.gpus = 1
        task.config.environment.gpu_types = ["H100"]
        spec = server._sandbox_spec(task, "/app")
        assert spec.resources.gpu == 1 and spec.resources.cpu == 4.0
        task.config.environment.gpus = None
        task.config.environment.gpu_types = None


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


class TestOverlay:
    def overlay_server(self, tmp_path, monkeypatch, *, dockerfile=OVERLAY_DOCKERFILE, toml=TASK_TOML, sandbox=None):
        sandbox = sandbox or OverlaySandbox()
        server, task, _, created = make_server(tmp_path, monkeypatch, sandbox, dockerfile=dockerfile)
        (task.path / "task.toml").write_text(toml)
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setattr(
            "resources_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )

        async def create(task, workdir):
            # The sandbox starts with the spec's environment, as the provider's container would.
            sandbox.spec = server._sandbox_spec(task, workdir)
            sandbox.spec_env = dict(sandbox.spec.env)
            created.append((task.task_id, workdir))
            return sandbox

        monkeypatch.setattr(server, "_create_sandbox", create)
        return server, task, sandbox, created

    def test_shell_command_exports_literal_values_and_only_enters_the_workdir(self):
        step = OverlayRun(
            command="make all", workdir="/app src", env={"PATH": "/opt/bin:/usr/bin", "Q": 'say "hi" `x` $NOT'}
        )
        # Values are already resolved: they are quoted so nothing expands again, `$NOT` included.
        assert overlay_shell_command(step) == (
            "export PATH=/opt/bin:/usr/bin; export Q='say \"hi\" `x` $NOT'; cd '/app src' || exit 1; make all"
        )
        # Keys the sandbox was created with that this step does not have yet are unset first.
        assert overlay_shell_command(step, {"PATH": "/x", "LATE": "1", "ZED": "2"}) == (
            "unset LATE; unset ZED; export PATH=/opt/bin:/usr/bin; export Q='say \"hi\" `x` $NOT'; "
            "cd '/app src' || exit 1; make all"
        )
        assert overlay_shell_command(OverlayRun(command="true")) == "cd / || exit 1; true"

    def test_workdir_command_creates_as_root_and_hands_the_directory_to_the_user(self):
        assert overlay_workdir_command("/app", None) == "mkdir -p /app"
        assert overlay_workdir_command("/app", "root") == overlay_workdir_command("/app", "0:0") == "mkdir -p /app"
        assert overlay_workdir_command("/home/agent/w", "agent") == (
            "test -d /home/agent/w || (mkdir -p /home/agent/w && chown agent /home/agent/w)"
        )
        assert overlay_workdir_command("/a b", "agent:staff") == (
            "test -d '/a b' || (mkdir -p '/a b' && chown agent:staff '/a b')"
        )

    def test_run_lines_see_the_environment_directory_and_user_of_their_point_in_the_dockerfile(
        self, tmp_path, monkeypatch
    ):
        server, task, sandbox, created = self.overlay_server(tmp_path, monkeypatch, toml=HEALTHCHECK_TOML)
        assert len(task.overlay) == 3 and task.user == "agent"

        response = TestClient(server.setup_webserver()).post("/seed_session", json=seed_body(task))

        assert response.status_code == 200, response.text
        assert created == [("hello", "/app")]
        # The sandbox is created with the Dockerfile's final environment, resolved: no live `$` reaches the provider.
        assert sandbox.spec.env == {"PATH": f"/opt/tools/bin:{IMAGE_PATH}", "MSG": "hi there"}
        assert not any("$" in value for value in sandbox.spec.env.values())
        first = sandbox.run("apt-get update")
        second = sandbox.run("echo ok > ready.txt")
        third = sandbox.run("as agent")
        # The line before `ENV PATH=...` sees the image's PATH and no MSG; the lines after see the extended PATH.
        extended = {"PATH": f"/opt/tools/bin:{IMAGE_PATH}", "MSG": "hi there"}
        assert first["seen"] == {
            "command": "apt-get update && apt-get install -y curl",
            "env": {"PATH": IMAGE_PATH},
            "user": "root",
            "cwd": "/",
        }
        assert second["seen"] == {
            "command": "mkdir -p /opt/tools/bin && echo ok > ready.txt",
            "env": extended,
            "user": "root",
            "cwd": "/app",
        }
        assert third["seen"] == {
            "command": 'echo "as agent" > /tmp/who',
            "env": extended,
            "user": "agent",
            "cwd": "/app",
        }
        # Lines before `USER agent` run as root (explicitly, since the task's user is non-root); after it, as agent.
        assert [c["user"] for c in (first, second, third)] == ["root", "root", "agent"]
        assert all(c["cwd"] == "/" and c["env"] is None for c in (first, second, third))
        commands = [c["command"] for c in sandbox.execs]
        # /app is made as root when the first RUN after `WORKDIR /app` comes up, once, then only entered.
        assert commands[:4] == [first["command"], "mkdir -p /app", second["command"], third["command"]]
        assert first["command"] == f"unset MSG; export PATH={IMAGE_PATH}; cd / || exit 1; {first['seen']['command']}"
        assert sandbox.execs[1]["user"] == "root" and commands.count("mkdir -p /app") == 1
        # The overlay shares the task's build budget: the first line sees all of it, later ones what is left.
        assert first["timeout_s"] == pytest.approx(task.config.environment.build_timeout_sec, abs=1)
        assert second["timeout_s"] <= first["timeout_s"] and third["timeout_s"] <= second["timeout_s"]
        # Only then does Gym prepare the agent's workdir, poll the healthcheck and hand the sandbox over.
        assert commands[4] == "mkdir -p /app && chown agent /app" and sandbox.dirs["/app"] == "agent"
        assert commands.index("test -f /app/ready.txt") > commands.index(third["command"])

    def test_root_image_without_user_keeps_the_plain_exec_path(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile="FROM img\nWORKDIR /app\nRUN true\nRUN false\n"
        )
        assert TestClient(server.setup_webserver()).post("/seed_session", json=seed_body(task)).status_code == 200
        prepare, *runs = sandbox.execs[:3]
        assert prepare["command"] == "mkdir -p /app"
        # The image's own PATH is exported again, harmlessly: a step exports its whole resolved environment.
        assert [c["command"] for c in runs] == [
            f"export PATH={IMAGE_PATH}; cd /app || exit 1; true",
            f"export PATH={IMAGE_PATH}; cd /app || exit 1; false",
        ]
        assert [c["seen"]["cwd"] for c in runs] == ["/app", "/app"]
        # No USER anywhere: root is the image's default user, so no override is needed (every provider supports this).
        assert [c["user"] for c in (prepare, *runs)] == [None, None, None]

    def test_workdir_after_user_is_made_as_root_and_owned_by_the_user(self, tmp_path, monkeypatch):
        # The fake enforces what a container would: a non-root user cannot create under a root-owned directory.
        assert OverlaySandbox()._mkdir("/home/agent/work", "agent").return_code == 1
        server, task, sandbox, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile="FROM img\nUSER agent\nWORKDIR /home/agent/work\nRUN touch made\n"
        )
        assert task.user == "agent" and task.workdir == "/home/agent/work"

        response = TestClient(server.setup_webserver()).post("/seed_session", json=seed_body(task))

        assert response.status_code == 200, response.text
        prepare, run = sandbox.execs[:2]
        assert prepare["command"] == (
            "test -d /home/agent/work || (mkdir -p /home/agent/work && chown agent /home/agent/work)"
        )
        assert prepare["user"] == "root"
        assert sandbox.dirs["/home/agent/work"] == "agent" and sandbox.dirs["/home/agent"] == "root"
        assert run["seen"] == {
            "command": "touch made",
            "env": {"PATH": IMAGE_PATH},
            "user": "agent",
            "cwd": "/home/agent/work",
        }

    def test_numeric_user_is_resolved_to_its_name_in_the_image(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile="FROM img\nUSER 1000\nWORKDIR /home/agent/work\nRUN id\nRUN id -g\n"
        )
        assert task.user == "1000"

        response = TestClient(server.setup_webserver()).post("/seed_session", json=seed_body(task))

        assert response.status_code == 200, response.text
        lookup, prepare, first, second = sandbox.execs[:4]
        assert lookup["command"] == "getent passwd 1000 | cut -d: -f1" and lookup["user"] == "root"
        # The name serves the directory's owner and both lines; the lookup happens once.
        assert prepare["command"].endswith("chown agent /home/agent/work)") and prepare["user"] == "root"
        assert [c["seen"]["user"] for c in (first, second)] == ["agent", "agent"]
        assert sum("getent" in c["command"] for c in sandbox.execs) == 1

    def test_unknown_numeric_user_is_a_clear_task_error(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile="FROM img\nUSER 4242\nRUN id\n"
        )
        response = TestClient(server.setup_webserver()).post("/seed_session", json=seed_body(task))
        assert response.status_code == 422, response.text
        assert "RUN line 1/1 'id': USER 4242 names no user in img" in response.json()["detail"]
        assert [c["command"] for c in sandbox.execs] == ["getent passwd 4242 | cut -d: -f1"] and sandbox.stopped

    def test_failing_run_is_a_terminal_task_error_with_the_end_of_its_output(self, tmp_path, monkeypatch):
        stdout = "\n".join(f"line {number}" for number in range(1, 401)) + "\n"
        sandbox = OverlaySandbox(
            failing="apt-get install",
            failure=SandboxExecResult(stdout=stdout, stderr="E: no package\n", return_code=100),
        )
        server, task, sandbox, _ = self.overlay_server(tmp_path, monkeypatch, sandbox=sandbox)
        client = TestClient(server.setup_webserver())

        response = client.post("/seed_session", json=seed_body(task))

        assert response.status_code == 422, response.text
        detail = response.json()["detail"]
        assert "Dockerfile overlay failed for 'hello'" in detail
        assert "RUN line 1/3 'apt-get update && apt-get install -y curl' exited 100" in detail
        # The tail keeps the end of a long log, where the error is, and drops the start.
        assert "line 400\nE: no package" in detail and "line 1\nline 2\n" not in detail
        # Nothing after the failing line ran, the sandbox is gone and no session was kept.
        assert len(sandbox.execs) == 1 and sandbox.stopped
        assert client.post("/verify", json=verify_body()).status_code == 404

    def test_provider_failure_during_a_run_is_retriable(self, tmp_path, monkeypatch):
        sandbox = OverlaySandbox(
            failing="apt-get install",
            failure=SandboxExecResult(stdout=None, stderr="connection reset", return_code=-1, error_type="sandbox"),
        )
        server, task, sandbox, _ = self.overlay_server(tmp_path, monkeypatch, sandbox=sandbox)
        client = TestClient(server.setup_webserver())

        response = client.post("/seed_session", json=seed_body(task))

        assert response.status_code == 503, response.text
        detail = response.json()["detail"]
        assert "Could not set up sandbox for 'hello'" in detail
        assert (
            "RUN line 1/3 'apt-get update && apt-get install -y curl' could not run in the sandbox: sandbox" in detail
        )
        assert len(sandbox.execs) == 1 and sandbox.stopped
        assert client.post("/verify", json=verify_body()).status_code == 404

    def test_timed_out_run_names_the_budget(self, tmp_path, monkeypatch):
        sandbox = OverlaySandbox(
            failing="echo ok", failure=SandboxExecResult(stdout="", stderr="", return_code=-1, error_type="timeout")
        )
        server, task, sandbox, _ = self.overlay_server(tmp_path, monkeypatch, sandbox=sandbox)
        response = TestClient(server.setup_webserver()).post("/seed_session", json=seed_body(task))
        assert response.status_code == 422
        assert "RUN line 2/3" in response.json()["detail"]
        assert "exceeded [environment].build_timeout_sec=600" in response.json()["detail"]
        assert sandbox.stopped

    def test_exhausted_budget_skips_the_remaining_lines(self, tmp_path, monkeypatch):
        class SlowSandbox(OverlaySandbox):
            async def exec(self, command, **kwargs):
                await asyncio.sleep(0.05)  # longer than the whole build budget below
                return await super().exec(command, **kwargs)

        toml = TASK_TOML.replace("cpus = 1", "build_timeout_sec = 0.01\ncpus = 1")
        server, task, sandbox, _ = self.overlay_server(tmp_path, monkeypatch, toml=toml, sandbox=SlowSandbox())
        response = TestClient(server.setup_webserver()).post("/seed_session", json=seed_body(task))
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "RUN line 2/3" in detail and "build_timeout_sec=0.01 is used up" in detail
        assert len(sandbox.execs) == 1 and sandbox.stopped

    def test_sandbox_ttls_include_the_build_budget_when_there_is_an_overlay(self, tmp_path, monkeypatch):
        server, task, _, _ = self.overlay_server(tmp_path, monkeypatch, toml=SEPARATE_IN_TASK_IMAGE_TOML)
        config, slack = task.config, server.config.sandbox_ttl_slack_s
        build = config.environment.build_timeout_sec
        assert build == 600 and task.overlay
        agent_spec = server._sandbox_spec(task, "/app")
        assert agent_spec.ttl_s == config.agent.timeout_sec + config.verifier.timeout_sec + slack + build

        specs = []

        class RecordingSandbox:
            def __init__(self, provider_config):
                pass

            async def start(self, spec):
                specs.append(spec)

        monkeypatch.setattr("resources_servers.harbor.app.AsyncSandbox", RecordingSandbox)
        asyncio.run(server._create_verifier_sandbox(task))
        (verifier_spec,) = specs
        assert verifier_spec.ttl_s == config.verifier.timeout_sec + slack + build
        # The verifier runs in the task's own image: it gets the same resolved Dockerfile environment as the agent.
        assert verifier_spec.image == task.image == "debian:bookworm-slim"
        assert verifier_spec.env == agent_spec.env == {"PATH": f"/opt/tools/bin:{IMAGE_PATH}", "MSG": "hi there"}

        # A pull-mode task has no overlay, so no build budget is added anywhere.
        server, task, _, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile="FROM img\nWORKDIR /app\n", toml=SEPARATE_IN_TASK_IMAGE_TOML
        )
        assert not task.overlay
        assert (
            server._sandbox_spec(task, "/app").ttl_s == config.agent.timeout_sec + config.verifier.timeout_sec + slack
        )
        specs.clear()
        asyncio.run(server._create_verifier_sandbox(task))
        assert specs[0].ttl_s == config.verifier.timeout_sec + slack

    def separate_verifier(self, tmp_path, monkeypatch, *, toml=SEPARATE_IN_TASK_IMAGE_TOML, verifier=None):
        server, task, agent, _ = self.overlay_server(tmp_path, monkeypatch, toml=toml)
        verifier = verifier or OverlaySandbox()

        async def create_verifier(task):
            verifier.spec_env = dict(
                server._sandbox_spec(task, None, environment=task.config.verifier.environment).env
            )
            return verifier

        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        client = TestClient(server.setup_webserver())
        assert client.post("/seed_session", json=seed_body(task)).status_code == 200
        return server, task, agent, verifier, client

    def test_separate_verifier_in_the_task_image_gets_the_overlay(self, tmp_path, monkeypatch):
        server, task, agent, verifier, client = self.separate_verifier(tmp_path, monkeypatch)
        assert not task.config.is_shared_verifier and _verifier_image(task) == "debian:bookworm-slim"

        payload = client.post("/verify", json=verify_body()).json()

        assert payload["reward"] == 1.0 and payload["verifier_mode"] == "separate", payload
        assert payload["mask_sample"] is False and payload["failure_kind"] is None
        verifier_commands = [c["command"] for c in verifier.execs]
        # The same three lines, in the same state, before anything else touches the verifier's sandbox.
        assert verifier.execs[0]["seen"]["command"] == "apt-get update && apt-get install -y curl"
        assert verifier.run("apt-get update")["seen"]["env"] == {"PATH": IMAGE_PATH}
        assert verifier.run("as agent")["seen"] == {
            "command": 'echo "as agent" > /tmp/who',
            "env": {"PATH": f"/opt/tools/bin:{IMAGE_PATH}", "MSG": "hi there"},
            "user": "agent",
            "cwd": "/app",
        }
        assert verifier_commands.index(verifier.run("as agent")["command"]) < next(
            i for i, c in enumerate(verifier_commands) if "test.sh" in c
        )
        assert agent.stopped and verifier.stopped

    def test_separate_verifier_overlay_failure_masks_the_sample(self, tmp_path, monkeypatch):
        verifier = OverlaySandbox(failing="apt-get install")
        server, task, agent, verifier, client = self.separate_verifier(tmp_path, monkeypatch, verifier=verifier)

        response = client.post("/verify", json=verify_body())

        # The rollout is complete; a verifier sandbox that cannot take the overlay must not discard it as a 422.
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["reward"] == 0.0 and payload["mask_sample"] is True
        assert payload["failure_kind"] == OVERLAY_FAILED_KIND == "harbor:overlay_failed"
        assert "Dockerfile overlay failed in the verifier sandbox" in payload["failure_reason"]
        assert "RUN line 1/3 'apt-get update && apt-get install -y curl' exited 100" in payload["failure_reason"]
        assert "E: no package" in payload["failure_reason"]
        assert not any("test.sh" in c["command"] for c in verifier.execs)
        assert agent.stopped and verifier.stopped

    def test_separate_verifier_provider_failure_during_the_overlay_masks_the_sample(self, tmp_path, monkeypatch):
        verifier = OverlaySandbox(
            failing="apt-get install",
            failure=SandboxExecResult(stdout=None, stderr="gone", return_code=-1, error_type="sandbox"),
        )
        _, _, _, verifier, client = self.separate_verifier(tmp_path, monkeypatch, verifier=verifier)
        payload = client.post("/verify", json=verify_body()).json()
        assert payload["mask_sample"] is True and payload["failure_kind"] == "provider_unavailable"
        assert "could not run in the sandbox: sandbox" in payload["failure_reason"] and verifier.stopped

    def test_prebuilt_verifier_image_gets_no_overlay(self, tmp_path, monkeypatch):
        toml = TASK_TOML.replace(
            "[verifier]\n",
            '[verifier]\nenvironment_mode = "separate"\n\n[verifier.environment]\ndocker_image = "org/verifier:1"\n',
        )
        _, task, _, verifier, client = self.separate_verifier(tmp_path, monkeypatch, toml=toml)
        assert _verifier_image(task) == "org/verifier:1"
        assert client.post("/verify", json=verify_body()).json()["reward"] == 1.0
        assert not any("apt-get" in c["command"] for c in verifier.execs)
        # Nor the agent image's Dockerfile environment.
        assert verifier.spec_env == {}


class TestVerify:
    def seeded_client(self, tmp_path, monkeypatch, sandbox=None):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, sandbox)
        client = TestClient(server.setup_webserver())
        assert client.post("/seed_session", json=seed_body(task)).status_code == 200
        return server, task, sandbox, client

    def test_verify_runs_test_sh_and_reads_reward(self, tmp_path, monkeypatch):
        server, task, sandbox, client = self.seeded_client(tmp_path, monkeypatch)

        response = client.post("/verify", json=verify_body())

        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["reward"] == 1.0
        assert payload["mask_sample"] is False
        assert payload["failure_kind"] is None
        assert payload["verifier_rewards"] == {"reward": 1.0}
        assert payload["verifier_return_code"] == 0
        assert payload["verifier_seconds"] >= 0
        assert payload["responses_create_params"]["input"][0]["content"] == "Create hello.txt"

        run = next(call for call in sandbox.execs if "test.sh" in call["command"])
        assert run["command"] == "timeout --signal=KILL 120 bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1"
        assert run["cwd"] == "/app"
        # The exec itself is bounded by the budget plus the grace period; the provider keeps it alive that long.
        assert run["timeout_s"] == 120 + server.config.verifier_grace_s
        # A root image needs no user override for the prepare step.
        prepare = next(
            call for call in sandbox.execs if "chmod 777" in call["command"] and "/tests" in call["command"]
        )
        assert prepare["user"] is None and prepare["cwd"] == "/"
        # tests/ was uploaded as an archive and unpacked into /tests.
        assert any(remote.endswith(".tar.gz") for _, remote in sandbox.uploads)
        assert any("tar -xzf" in call["command"] and "/tests" in call["command"] for call in sandbox.execs)
        assert Path(payload["verifier_logs_dir"]).is_absolute()
        assert (Path(payload["verifier_logs_dir"]) / "reward.txt").read_text() == "1\n"

    def test_prepare_runs_as_root_on_a_non_root_image(self, tmp_path, monkeypatch):
        class NonRootSandbox(FakeSandbox):
            """The image's default user may not touch root-owned /logs/verifier."""

            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                result = await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)
                if ("chmod" in command or "chown" in command) and user != "root":
                    return SandboxExecResult(stdout="", stderr="chmod: Permission denied", return_code=1)
                return result

        server, task, sandbox, _ = make_server(
            tmp_path, monkeypatch, NonRootSandbox(), dockerfile="FROM ubuntu:24.04\nWORKDIR /app\nUSER app"
        )
        assert task.user == "app"
        client = TestClient(server.setup_webserver())
        assert client.post("/seed_session", json=seed_body(task)).status_code == 200

        payload = client.post("/verify", json=verify_body()).json()

        assert payload["reward"] == 1.0
        assert payload["mask_sample"] is False
        assert payload["failure_kind"] is None
        prepare = next(
            call for call in sandbox.execs if "chmod 777" in call["command"] and "/tests" in call["command"]
        )
        assert prepare["user"] == "root"
        # test.sh itself still runs as the verifier's user, not root.
        run = next(call for call in sandbox.execs if "test.sh" in call["command"])
        assert run["user"] == task.config.verifier.user

    def test_verify_is_idempotent_per_session(self, tmp_path, monkeypatch):
        _, _, sandbox, client = self.seeded_client(tmp_path, monkeypatch)

        first = client.post("/verify", json=verify_body())
        second = client.post("/verify", json=verify_body())

        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
        assert first.json()["reward"] == 1.0
        assert sum("test.sh" in call["command"] for call in sandbox.execs) == 1
        # The retry did not re-run the prepare step that wipes /tests either.
        assert sum("chmod 777" in call["command"] and "/tests" in call["command"] for call in sandbox.execs) == 1

    def test_json_reward_with_components(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(verifier_files={"reward.json": json.dumps({"reward": 0.5, "tests_passed": 3})})
        _, _, _, client = self.seeded_client(tmp_path, monkeypatch, sandbox)
        payload = client.post("/verify", json=verify_body()).json()
        assert payload["reward"] == 0.5
        assert payload["verifier_rewards"] == {"reward": 0.5, "tests_passed": 3.0}

    def test_ambiguous_reward_is_an_authoring_error(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(verifier_files={"reward.json": json.dumps({"a": 1, "b": 0})})
        _, _, _, client = self.seeded_client(tmp_path, monkeypatch, sandbox)
        response = client.post("/verify", json=verify_body())
        assert response.status_code == 422
        assert "several keys" in response.json()["detail"]

    def test_missing_reward_scores_zero_and_is_measured(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(
            verifier_files={"test-stdout.txt": "pytest exploded"},
            test_result=SandboxExecResult(stdout="", stderr="", return_code=1),
        )
        _, _, _, client = self.seeded_client(tmp_path, monkeypatch, sandbox)
        payload = client.post("/verify", json=verify_body()).json()
        assert payload["reward"] == 0.0
        assert payload["mask_sample"] is False
        assert payload["failure_kind"] == "harbor:missing_reward"
        assert "pytest exploded" in payload["failure_reason"]
        assert payload["verifier_return_code"] == 1

    def test_invalid_reward_scores_zero(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(verifier_files={"reward.txt": "maybe"})
        _, _, _, client = self.seeded_client(tmp_path, monkeypatch, sandbox)
        payload = client.post("/verify", json=verify_body()).json()
        assert payload["reward"] == 0.0 and payload["failure_kind"] == "harbor:invalid_reward"

    def test_verifier_timeout_when_exit_code_never_appears(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(
            test_result=SandboxExecResult(stdout=None, stderr=None, return_code=125, error_type="timeout")
        )
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, sandbox)
        server.config.verifier_grace_s = 0
        (task.path / "task.toml").write_text(
            TASK_TOML.replace("timeout_sec = 120.0\n\n[agent]", "timeout_sec = 0.01\n\n[agent]")
        )
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        client = TestClient(server.setup_webserver())
        assert client.post("/seed_session", json=seed_body(task)).status_code == 200
        payload = client.post("/verify", json=verify_body()).json()
        assert payload["reward"] == 0.0 and payload["failure_kind"] == "harbor:verifier_timeout"

    def test_sandbox_runtime_failure_masks(self, tmp_path, monkeypatch):
        class BrokenLaunch(FakeSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if "test.sh" in command:
                    return SandboxExecResult(stdout=None, stderr="gone", return_code=125, error_type="sandbox")
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

        sandbox = BrokenLaunch()
        _, _, _, client = self.seeded_client(tmp_path, monkeypatch, sandbox)
        payload = client.post("/verify", json=verify_body()).json()
        assert payload["reward"] == 0.0
        assert payload["mask_sample"] is True
        assert payload["failure_kind"] == "verifier_error"

    def test_transfer_exception_masks(self, tmp_path, monkeypatch):
        class BrokenUpload(FakeSandbox):
            async def upload(self, local_path, remote_path):
                raise ConnectionError("lost")

        _, _, _, client = self.seeded_client(tmp_path, monkeypatch, BrokenUpload())
        payload = client.post("/verify", json=verify_body()).json()
        assert payload["mask_sample"] is True
        assert payload["failure_kind"] == "provider_unavailable"
        assert "lost" in payload["failure_reason"]

    def test_verify_needs_a_seeded_session(self, tmp_path, monkeypatch):
        server, _, _, _ = make_server(tmp_path, monkeypatch)
        response = TestClient(server.setup_webserver()).post("/verify", json=verify_body())
        assert response.status_code == 404

    def test_verify_identity_must_match(self, tmp_path, monkeypatch):
        _, _, _, client = self.seeded_client(tmp_path, monkeypatch)
        assert client.post("/verify", json=verify_body(digest="other")).status_code == 409


class TestClose:
    def test_close_unknown_session_is_idempotent(self, tmp_path, monkeypatch):
        server, _, _, _ = make_server(tmp_path, monkeypatch)
        response = TestClient(server.setup_webserver()).post(
            "/close_session",
            json={"resources_session_id": "never", "episode_id": {"rollout_id": "r1", "attempt": 0}},
        )
        assert response.status_code == 200

    def test_close_checks_episode(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        client = TestClient(server.setup_webserver())
        assert client.post("/seed_session", json=seed_body(task)).status_code == 200
        response = client.post(
            "/close_session",
            json={"resources_session_id": "rs-1", "episode_id": {"rollout_id": "other", "attempt": 0}},
        )
        assert response.status_code == 409 and not sandbox.stopped

    @pytest.mark.asyncio
    async def test_shutdown_stops_sandboxes(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        client = TestClient(server.setup_webserver())
        assert client.post("/seed_session", json=seed_body(task)).status_code == 200
        await server.shutdown()
        assert sandbox.stopped


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
    """The agent's sandbox in separate mode: holds /logs/artifacts and one report file."""

    present: dict[str, str] = field(
        default_factory=lambda: {"/logs/artifacts": "dir", "/app/output/report.json": "file"}
    )

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.execs.append({"command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s, "user": user})
        if command.startswith("if [ -d "):
            path = command.split("if [ -d ")[1].split(" ]")[0].strip("'")
            return SandboxExecResult(stdout=self.present.get(path, "none") + "\n", stderr="", return_code=0)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def download(self, remote_path, local_path):
        if remote_path.startswith("/tmp/.nemo-gym-download-"):
            Path(local_path).write_bytes(archive_of({"state.db": "db"}))
        else:
            Path(local_path).write_bytes(b'{"ok": true}')


class TestSeparateVerification:
    def seeded(self, tmp_path, monkeypatch, *, toml=SEPARATE_TOML, verifier=None):
        agent = AgentSandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent)
        (task.path / "task.toml").write_text(toml)
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        verifier = verifier or FakeSandbox()
        created = []

        async def create_verifier(task):
            created.append(task.task_id)
            return verifier

        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        client = TestClient(server.setup_webserver())
        assert client.post("/seed_session", json=seed_body(task)).status_code == 200, "seed"
        return server, task, agent, verifier, created, client

    def test_collects_artifacts_then_verifies_in_a_fresh_sandbox(self, tmp_path, monkeypatch):
        server, task, agent, verifier, created, client = self.seeded(tmp_path, monkeypatch)

        payload = client.post("/verify", json=verify_body()).json()

        assert payload["reward"] == 1.0 and payload["verifier_mode"] == "separate", payload
        assert created == ["hello"]
        # The main-container collect hook ran in the agent sandbox; the sidecar hook was skipped.
        hooks = [c for c in agent.execs if "state.db" in c["command"] or "kafka-dump" in c["command"]]
        assert [h["command"] for h in hooks] == ["cp /app/state.db /logs/artifacts/state.db"]
        # /logs/artifacts (a directory) and the report (a file) were pulled from the agent sandbox...
        artifacts = server.config.artifacts_dir / "rs-1" / "artifacts"
        assert (artifacts / "logs" / "artifacts" / "state.db").read_text() == "db"
        assert (artifacts / "app" / "output" / "report.json").read_bytes() == b'{"ok": true}'
        # ...the agent sandbox was stopped before test.sh ran, and the verifier got tests plus artifacts back.
        assert agent.stopped
        uploads = [remote for _, remote in verifier.uploads]
        assert any(remote.endswith(".tar.gz") for remote in uploads)  # tests/ and /logs/artifacts archives
        assert "/app/output/report.json" in uploads
        # Restored artifact directories are world-writable, as Harbor leaves them, so a verifier that
        # drops privileges can still write scratch files next to the agent's output.
        assert any(c["command"] == "mkdir -p /app/output && chmod 777 /app/output" for c in verifier.execs)
        run = next(c for c in verifier.execs if "test.sh" in c["command"] and "timeout" in c["command"])
        assert "timeout --signal=KILL 300 bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1" in run["command"]
        assert (run["env"], run["user"], run["cwd"]) == ({"CHECK": "strict"}, "root", None)
        assert run["timeout_s"] == 300 + server.config.verifier_grace_s
        assert verifier.stopped
        # Closing the session does not stop the agent sandbox a second time.
        agent.stopped = False
        close = client.post(
            "/close_session", json={"resources_session_id": "rs-1", "episode_id": {"rollout_id": "r1", "attempt": 0}}
        )
        assert close.status_code == 200 and not agent.stopped

    def test_sandbox_spec_applies_dockerfile_env(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(
            tmp_path, monkeypatch, dockerfile="FROM ubuntu:24.04\nWORKDIR /app\nENV FOO=bar\nENV MODE=image\n"
        )
        (task.path / "task.toml").write_text(TASK_TOML + '\n[environment.env]\nMODE = "toml"\n')
        task = load_task(task.path)
        monkeypatch.setattr(
            "resources_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )

        spec = server._sandbox_spec(task, "/app")

        # Dockerfile ENV reaches the agent's sandbox, and task.toml [environment.env] wins over it.
        assert spec.env["FOO"] == "bar" and spec.env["MODE"] == "toml"

        verifier_environment = HarborEnvironment(docker_image="org/verifier:1", env={"ONLY": "verifier"})
        verifier_spec = server._sandbox_spec(task, None, environment=verifier_environment, role="verifier")
        assert "FOO" not in verifier_spec.env and verifier_spec.env["ONLY"] == "verifier"

    def test_verifier_sandbox_spec_uses_the_verifier_environment(self, tmp_path, monkeypatch):
        server, task, _, _, _, _ = self.seeded(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "resources_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        spec = server._sandbox_spec(
            task,
            None,
            environment=task.config.verifier.environment,
            image=_verifier_image(task),
            role="verifier",
            ttl=42,
        )
        assert spec.image == "org/verifier:1"
        assert (spec.resources.cpu, spec.resources.memory_mib) == (2, 4096)
        assert spec.ttl_s == 42 and spec.metadata["harbor_role"] == "verifier"

    def test_separate_mode_without_an_image_falls_back_to_the_task_image(self, tmp_path, monkeypatch):
        toml = (
            SEPARATE_TOML.split("[verifier.environment]")[0]
            + '[agent]\ntimeout_sec = 120.0\n\n[environment]\ndocker_image = "org/agent:1"\n'
        )
        server, task, _, _, _, _ = self.seeded(tmp_path, monkeypatch, toml=toml)
        assert _verifier_image(task) == "org/agent:1"

    def test_seed_rejects_a_verifier_that_must_be_built(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        toml = SEPARATE_TOML.replace('docker_image = "org/verifier:1"\n', "")
        (task.path / "task.toml").write_text(toml)
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        response = TestClient(server.setup_webserver()).post("/seed_session", json=seed_body(task))
        assert response.status_code == 422 and "tests/Dockerfile" in response.json()["detail"]

    def test_verifier_failure_masks_and_stops_both_sandboxes(self, tmp_path, monkeypatch):
        class Broken(FakeSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                raise RuntimeError("verifier sandbox lost")

        broken = Broken()
        _, _, agent, verifier, _, client = self.seeded(tmp_path, monkeypatch, verifier=broken)
        payload = client.post("/verify", json=verify_body()).json()
        assert payload["mask_sample"] is True and payload["failure_kind"] == "provider_unavailable"
        assert agent.stopped and verifier.stopped


class TestTransfersWithoutRoot:
    def test_falls_back_to_the_default_user_when_root_is_refused(self, tmp_path):
        import asyncio

        from resources_servers.harbor.sandbox_io import upload_dir

        class NoRoot(FakeSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                self.execs.append({"command": command, "user": user})
                if user == "root":
                    return SandboxExecResult(
                        stdout="",
                        stderr="fork/exec /usr/bin/bash: operation not permitted (switching to uid=0 requires CAP_SETUID)",
                        return_code=1,
                    )
                return SandboxExecResult(stdout="", stderr="", return_code=0)

        sandbox = NoRoot()
        source = tmp_path / "src"
        source.mkdir()
        (source / "a.txt").write_text("a")
        asyncio.run(upload_dir(sandbox, source, "/tests"))
        users = [call["user"] for call in sandbox.execs if "tar -xzf" in call["command"]]
        assert users == ["root", None]

    def test_other_root_failures_propagate(self, tmp_path):
        import asyncio

        from resources_servers.harbor.sandbox_io import SandboxTransferError, upload_dir

        class Broken(FakeSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                return SandboxExecResult(stdout="", stderr="tar: corrupt archive", return_code=2)

        source = tmp_path / "src"
        source.mkdir()
        (source / "a.txt").write_text("a")
        with pytest.raises(SandboxTransferError, match="corrupt archive"):
            asyncio.run(upload_dir(Broken(), source, "/tests"))


COMPOSE_TOML = """
schema_version = "1.4"

artifacts = [{ source = "/var/log/api", service = "api" }]

[verifier]
timeout_sec = 300.0
environment_mode = "separate"

[verifier.environment]
docker_image = "org/verifier:1"

[[verifier.collect]]
command = "dump-topics > /logs/artifacts/topics.txt"
service = "kafka"

[agent]
timeout_sec = 120.0

[environment]
docker_image = "org/agent:1"
cpus = 1
skills_dir = "/app/.skills"

[[environment.mcp_servers]]
name = "playwright"
transport = "sse"
url = "http://api:3080/sse"

[environment.healthcheck]
command = "test -f /tmp/ready"
interval_sec = 0.01
start_interval_sec = 0.01
timeout_sec = 5.0
retries = 3
"""

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
    "org/agent:1": {"os": "linux", "architecture": "amd64", "image": "org/agent@sha256:" + "a" * 64, "config": {}},
    "org/api:1": {
        "os": "linux",
        "architecture": "amd64",
        "image": "org/api@sha256:" + "b" * 64,
        "config": {"Cmd": ["serve"]},
    },
    "org/kafka:1": {
        "os": "linux",
        "architecture": "amd64",
        "image": "org/kafka@sha256:" + "c" * 64,
        "config": {"Cmd": ["kafka"], "User": "appuser"},
    },
}


@dataclass
class HealthySandbox(AgentSandbox):
    """Main service whose healthcheck passes on the third probe."""

    probes: int = 0

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if command == "test -f /tmp/ready":
            self.probes += 1
            self.execs.append({"command": command, "user": user})
            return SandboxExecResult(stdout="", stderr="", return_code=0 if self.probes >= 3 else 1)
        return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)


class FakeCompose:
    def __init__(self, services):
        self.services = services
        self.stopped = False

    async def stop(self):
        self.stopped = True


class TestComposeAndRouting:
    def compose_server(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "task.toml").write_text(COMPOSE_TOML)
        (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (task.path.parent / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setattr(
            "resources_servers.harbor.app.get_global_config_dict",
            lambda: {"sandbox": {"opensandbox": {}}, "sandbox_gpu": {"opensandbox": {}}},
        )
        main, api, kafka = HealthySandbox(), FakeSandbox(), FakeSandbox()
        compose = FakeCompose({"main": main, "api": api, "kafka": kafka})
        documents = []

        async def start_compose(task, session_id):
            documents.append(server._compose_document(task))
            return compose

        monkeypatch.setattr(server, "_create_compose", start_compose)
        return server, task, compose, documents

    def test_compose_task_seeds_a_group_and_leaves_task_context(self, tmp_path, monkeypatch):
        server, task, compose, documents = self.compose_server(tmp_path, monkeypatch)
        client = TestClient(server.setup_webserver())

        response = client.post("/seed_session", json=seed_body(task))

        assert response.status_code == 200, response.text
        document = documents[0]
        assert set(document["services"]) == {"main", "api", "kafka"}
        # The non-root sidecar keeps its image user and gets no host injection.
        assert "user" not in document["services"]["kafka"] and document["services"]["kafka"]["x-sandbox"] == {
            "hosts": []
        }
        main = compose.services["main"]
        # The healthcheck was polled until it passed, then the task context was written for the harness.
        assert main.probes == 3
        context_write = next(c for c in main.execs if "/tmp/.nemo-gym/task.json" in c["command"])
        assert context_write["user"] == "root"
        assert "playwright" in context_write["command"] and "/app/.skills" in context_write["command"]
        payload = response.json()
        assert payload["sandbox_access"]["connection"]["provider_config_ref"] == "sandbox"

    def test_separate_verifier_collects_from_sidecars_and_stops_the_group(self, tmp_path, monkeypatch):
        server, task, compose, _ = self.compose_server(tmp_path, monkeypatch)
        kafka, api = compose.services["kafka"], compose.services["api"]
        api.__class__ = AgentSandbox  # gives it `present`
        api.present = {"/var/log/api": "dir"}
        verifier = FakeSandbox()

        async def create_verifier(task):
            return verifier

        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        client = TestClient(server.setup_webserver())
        assert client.post("/seed_session", json=seed_body(task)).status_code == 200

        payload = client.post("/verify", json=verify_body()).json()

        assert payload["reward"] == 1.0, payload
        hook = next(c for c in kafka.execs if "dump-topics" in c["command"])
        assert hook["command"].startswith("sh -c ")
        assert any("/var/log/api" in c["command"] for c in api.execs)
        assert compose.stopped and verifier.stopped
        assert (
            client.post(
                "/close_session",
                json={"resources_session_id": "rs-1", "episode_id": {"rollout_id": "r1", "attempt": 0}},
            ).status_code
            == 200
        )

    def test_gpu_tasks_route_to_the_gpu_provider(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "task.toml").write_text(TASK_TOML.replace("cpus = 1", 'cpus = 1\ngpus = 1\ngpu_types = ["H100"]'))
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        server.config.gpu_sandbox_provider = "sandbox_gpu"
        monkeypatch.setattr(
            "resources_servers.harbor.app.get_global_config_dict",
            lambda: {"sandbox": {"opensandbox": {}}, "sandbox_gpu": {"opensandbox": {}}},
        )
        assert server._provider_ref(task) == "sandbox_gpu"
        spec = server._sandbox_spec(task, "/app")
        assert spec.resources.gpu == 1 and spec.resources.gpu_type is None
        server.config.request_gpu_type = True
        assert server._sandbox_spec(task, "/app").resources.gpu_type == "H100"
        client = TestClient(server.setup_webserver())
        response = client.post("/seed_session", json=seed_body(task))
        assert response.status_code == 200, response.text
        assert response.json()["sandbox_access"]["connection"]["provider_config_ref"] == "sandbox_gpu"

    def test_failed_healthcheck_is_a_retryable_seed_failure(self, tmp_path, monkeypatch):
        server, task, compose, _ = self.compose_server(tmp_path, monkeypatch)
        compose.services["main"].probes = -100  # never reaches 3 within 3 retries
        response = TestClient(server.setup_webserver()).post("/seed_session", json=seed_body(task))
        assert response.status_code == 503 and "Healthcheck failed" in response.json()["detail"]
        assert compose.stopped


def test_server_config_holds_only_deployment_settings():
    """A setting belongs here only if every dataset on the same deployment wants the same value.

    Task facts stay in task.toml, dataset settings in dataset.toml's [gym] table, image mirrors and
    setup commands in the sandbox provider. Adding a field means changing this set in the same PR.
    """
    own = set(HarborResourcesServerConfig.model_fields) - set(BaseResourcesServerConfig.model_fields)
    assert own == {
        "tasksets",
        "sandbox_provider",
        "sandbox_ready_timeout_s",
        "sandbox_ttl_slack_s",
        "sandbox_provider_options",
        "sandbox_metadata",
        "sandbox_resources_override",
        "gpu_sandbox_provider",
        "request_gpu_type",
        "verifier_grace_s",
        "artifacts_dir",
    }
    source = inspect.getsource(HarborResourcesServerConfig).lower()
    assert not any(word in source for word in ("terminal", "tb4", "tb2", "swe", "nextjs", "hello-world"))
