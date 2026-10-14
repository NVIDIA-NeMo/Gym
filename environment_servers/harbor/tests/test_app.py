# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import errno
import inspect
import io
import json
import posixpath
import re
import shlex
import shutil
import subprocess
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import orjson
import pytest
import yaml
from aiohttp import ClientResponseError, RequestInfo
from fastapi.testclient import TestClient
from multidict import CIMultiDict, CIMultiDictProxy
from omegaconf import OmegaConf
from pydantic import ConfigDict
from pytest import MonkeyPatch
from yarl import URL

import nemo_gym.sandbox.providers.base as sandbox_base
from environment_servers.harbor.app import (
    EXIT_RECORD_NAME,
    OVERLAY_FAILED_KIND,
    SEED_CREATE_GRACE_S,
    TIMEOUT_WRAPPER_GUARD,
    HarborEnvironmentServer,
    HarborEnvironmentServerConfig,
    HarborEpisodeResponse,
    _exec_as_root_user,
    _verifier_image,
    overlay_shell_command,
    overlay_workdir_command,
    parse_exit_record,
    parse_reward_file,
    select_reward,
    verifier_command,
)
from nemo_gym.base_environment_server import BaseEnvironmentServerConfig
from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxSpec
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from nemo_gym.single_agent_turn_types import SingleAgentTurnRequest
from nemo_gym.tasks.harbor import DIGEST_KEY, load_task
from nemo_gym.tasks.harbor.dockerfile import RECORD_VERSION, OverlayRun
from nemo_gym.tasks.harbor.models import HarborEnvironment
from nemo_gym.tasks.harbor.task import IMAGE_CONFIGS_FILE


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
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "task.toml").write_text(toml)
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
        record = {"command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s, "user": user, "seen": None}
        self.execs.append(record)
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
    image_config: dict | None = None,
    **config_overrides,
):
    folder = tmp_path / "datasets" / "ds"
    task = load_task(write_task(folder / "hello", toml=toml, dockerfile=dockerfile, image_config=image_config))
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
        sandbox.created_in(workdir)
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
        ],
    )
    def test_problems(self, tmp_path, name, text, fragment):
        (tmp_path / name).write_text(text)
        rewards, problem = parse_reward_file(tmp_path)
        assert rewards is None and fragment in problem

    def test_missing(self, tmp_path):
        assert parse_reward_file(tmp_path)[1] == "no reward.json or reward.txt was written"

    @pytest.mark.parametrize("contents", ["{not json", "[]", "", '{"reward": "1"}'])
    def test_invalid_json_does_not_fall_back_to_text(self, tmp_path, contents):
        (tmp_path / "reward.json").write_text(contents)
        (tmp_path / "reward.txt").write_text("0.5\n")
        rewards, problem = parse_reward_file(tmp_path)
        assert rewards is None
        assert "reward.json" in problem

    def test_boolean_rewards_match_harbor(self, tmp_path):
        (tmp_path / "reward.json").write_text('{"passed": true, "failed": false}')
        assert parse_reward_file(tmp_path) == ({"passed": 1.0, "failed": 0.0}, None)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("137 1700000000 1700000120\n", (137, 120)),
            ("0 1700000000 1700000001", (0, 1)),
            # An image without `date` leaves empty timestamps: the exit code is still known.
            ("137  \n", (137, None)),
            ("137 x y", (137, None)),
            ("", None),
            ("nope", None),
        ],
    )
    def test_exit_record(self, tmp_path, text, expected):
        (tmp_path / EXIT_RECORD_NAME).write_text(text)
        assert parse_exit_record(tmp_path) == expected

    def test_exit_record_missing(self, tmp_path):
        assert parse_exit_record(tmp_path) is None

    def test_non_utf8_bytes_do_not_raise(self, tmp_path):
        (tmp_path / "reward.txt").write_bytes(b"\xff\xfe1\n")
        rewards, problem = parse_reward_file(tmp_path)
        assert rewards is None and "reward.txt is not valid" in problem

        (tmp_path / "reward.json").write_bytes(b'{"reward": 1}\xff')
        (tmp_path / "reward.txt").write_text("0.25\n")
        assert parse_reward_file(tmp_path)[0] is None
        assert "reward.json is not valid" in parse_reward_file(tmp_path)[1]


class TestEpisode:
    async def test_episode_seeds_runs_the_agent_verifies_and_stops_the_sandbox(self, tmp_path, monkeypatch):
        server, task, sandbox, created = make_server(tmp_path, monkeypatch)
        client = server.server_client
        closes_before_verify: list[int] = []
        real_exec = sandbox.exec

        async def exec_recording_closes(command, **kwargs):
            if "test.sh" in command:
                closes_before_verify.append(client.paths().count("/v1/agent_sessions/close"))
            return await real_exec(command, **kwargs)

        monkeypatch.setattr(sandbox, "exec", exec_recording_closes)

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        result = response.result
        assert result.reward == 1.0 and result.mask_sample is False and result.failure_kind is None
        assert result.verifier_rewards == {"reward": 1.0} and result.verifier_return_code == 0
        assert result.response.id == "response"
        assert result.responses_create_params.input[0].content == "Create hello.txt"
        # The sandbox was created for the task, prepared, and handed to the harness through its session seed.
        assert created == [("hello", "/app")]
        assert sandbox.execs[0]["command"] == "mkdir -p /app"
        assert client.paths() == ["/v1/agent_sessions", "/ng-rollout/r1/v1/responses", "/v1/agent_sessions/close"]
        assert client.seed_bodies()[0]["sandbox_access"] == {
            "connection": {
                "kind": "direct",
                "provider_config_ref": "sandbox",
                "descriptor": {"sandbox_id": "sb-1", "workdir": "/app"},
            },
            "workdir": "/app",
        }
        assert client.seed_bodies()[0]["task_id"] == {"taskset": "ds", "task_id": "hello"}
        # The turn's cookie jar, not the seed's, travels to the close.
        assert client.calls[2][2]["cookies"] == {"session": "cookie-value"}
        # Verification ran only after the harness let go of the sandbox, then the sandbox was stopped.
        assert closes_before_verify == [1]
        assert sandbox.stopped and not server._sandboxes and not server._sessions
        assert (Path(result.verifier_logs_dir) / "reward.txt").read_text() == "1\n"

    def test_run_endpoint_returns_the_episode_without_a_sandbox_handle(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        with TestClient(server.setup_webserver()) as client:
            response = client.post("/run", json=episode(task).model_dump(mode="json"))
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["result"]["reward"] == 1.0 and payload["failure"] is None
        assert payload["episode_id"] == {"rollout_id": "r1", "attempt": 0}
        assert "sb-1" not in response.text and "sandbox" not in payload["result"]
        assert sandbox.stopped

    async def test_episodes_parse_and_hash_the_task_once(self, tmp_path, monkeypatch):
        server, task, _, created = make_server(tmp_path, monkeypatch)
        loads: list[Path] = []
        real_load_task = load_task

        def counting_load_task(folder):
            loads.append(Path(folder))
            return real_load_task(folder)

        monkeypatch.setattr("environment_servers.harbor.app.load_task", counting_load_task)

        assert (await run_episode(server, task)).result.reward == 1.0
        assert (await run_episode(server, task, rollout="r2")).result.reward == 1.0

        assert created == [("hello", "/app"), ("hello", "/app")]
        assert len(loads) == 1
        assert loads[0] != task.path
        assert load_task(loads[0]).digest == task.digest

    @pytest.mark.parametrize(
        ("selector", "fragment"),
        [
            ({"taskset": "nope"}, "not served"),
            ({"task_id": "missing"}, "not in taskset"),
            ({"digest": "0" * 64}, "does not match the taskset mapping"),
        ],
    )
    async def test_bad_identity_is_a_terminal_seed_failure(self, tmp_path, monkeypatch, selector, fragment):
        server, task, _, created = make_server(tmp_path, monkeypatch)
        response = await run_episode(server, task, **selector)
        failure = response.failure
        assert response.result is None and failure.stage == "seed" and failure.terminal
        assert fragment in failure.failure_reason
        assert created == [] and server.server_client.calls == []

    async def test_dataset_toml_edit_after_materialization_is_refused(self, tmp_path, monkeypatch):
        server, task, _, created = make_server(tmp_path, monkeypatch)
        dataset_file = task.path.parent / "dataset.toml"
        dataset_file.write_text("[gym.defaults]\ntimeout_multiplier = 2.0\n")
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        # The rows were materialized with the multiplier at 2; the file is edited before the server runs.
        dataset_file.write_text("[gym.defaults]\ntimeout_multiplier = 4.0\n")
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "changed since materialization" in response.failure.failure_reason and created == []
        # Restored, the same mapping serves the task again.
        dataset_file.write_text("[gym.defaults]\ntimeout_multiplier = 2.0\n")
        assert (await run_episode(server, task, rollout="r2")).failure is None

    async def test_changed_folder_is_a_terminal_seed_failure(self, tmp_path, monkeypatch):
        server, task, _, created = make_server(tmp_path, monkeypatch)
        (task.path / "tests" / "test.sh").write_text("#!/bin/bash\necho 0 > /logs/verifier/reward.txt\n")

        response = await run_episode(server, task)

        assert response.failure.stage == "seed" and response.failure.terminal
        assert "changed since materialization" in response.failure.failure_reason
        assert created == []

    async def test_host_copy_failure_is_a_retryable_seed_failure(self, tmp_path, monkeypatch):
        """A failed snapshot copy (disk full, I/O error) is not a task defect: the collector may retry it."""
        server, task, _, created = make_server(tmp_path, monkeypatch)

        def no_space(*args, **kwargs):
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(shutil, "copytree", no_space)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "No space left on device" in response.failure.failure_reason
        assert created == [] and server.server_client.calls == []

    async def test_malformed_task_folder_is_a_terminal_seed_failure(self, tmp_path, monkeypatch):
        server, task, _, created = make_server(tmp_path, monkeypatch)
        (task.path / "task.toml").write_text("[verifier\n")
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "Could not snapshot task" in response.failure.failure_reason
        assert created == []

    async def test_sandbox_failure_is_a_retryable_seed_failure(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)

        async def boom(task, workdir, *, episode_key):
            raise RuntimeError("pull failed")

        monkeypatch.setattr(server, "_create_sandbox", boom)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "pull failed" in response.failure.failure_reason
        assert server.server_client.calls == []

    async def test_shutdown_refuses_new_episodes(self, tmp_path, monkeypatch):
        server, task, _, created = make_server(tmp_path, monkeypatch)
        await server.shutdown()
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "shutting down" in response.failure.failure_reason and created == []

    def test_sandbox_spec_from_task(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict",
            lambda: {"sandbox": {"opensandbox": {}, "default_metadata": {"sandbox-api": "osb"}}},
        )

        spec = server._sandbox_spec(task, "/app")

        assert spec.image == "ubuntu:24.04"
        assert spec.workdir == "/app"
        assert spec.ttl_s == 21600 + 10 + server.config.sandbox_ttl_slack_s
        assert (spec.resources.cpu, spec.resources.memory_mib, spec.resources.disk_gib) == (1.0, 2048, 10)
        assert spec.metadata["sandbox-api"] == "osb"
        assert spec.metadata["harbor_task"] == "hello"
        assert spec.metadata["nemo_gym_environment_server"] == "harbor_environment_server"


class TestSeedWorkdirAndResources:
    async def test_seed_creates_the_logs_folders_and_per_task_env(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        # Per-task environment comes from the dataset's own file, applied by the loader; the server sees plain tasks.
        (task.path.parent / "dataset.toml").write_text(
            '[gym.tasks."hello".environment]\nenv = { CIRCLE_NODE_TOTAL = "3" }\n[gym.tasks."other".environment]\nenv = { X = "1" }\n'
        )
        task = load_task(task.path)
        # The applied dataset slice is part of the digest the rows and the mapping carry.
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        seeded: list = []
        server.server_client.turn_hooks.append(lambda: seeded.append(server._sessions["r1"].task))
        assert (await run_episode(server, task)).failure is None
        commands = [c["command"] for c in sandbox.execs]
        assert commands[0] == "mkdir -p /app"
        logs = next(c for c in sandbox.execs if "/logs/artifacts" in c["command"])
        assert "chmod 777 /logs/agent /logs/verifier /logs/artifacts" in logs["command"] and logs["user"] == "root"
        # The server resolves the task from its own snapshot; the dataset's settings must reach that copy.
        assert seeded[0].path != task.path and seeded[0].env["CIRCLE_NODE_TOTAL"] == "3" and "X" not in seeded[0].env
        spec = server._sandbox_spec(seeded[0], "/app")
        assert spec.env["CIRCLE_NODE_TOTAL"] == "3" and "X" not in spec.env

    async def test_image_workdir_used_when_task_sets_none(self, tmp_path, monkeypatch):
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

        async def create(task, workdir, *, episode_key=None):
            created.append((task.task_id, workdir))
            return pwd_sandbox

        monkeypatch.setattr(server, "_create_sandbox", create)
        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert created == [("hello", None)]
        assert server.server_client.seed_bodies()[0]["sandbox_access"]["workdir"] == "/work"

    async def test_root_workdir_is_neither_created_nor_handed_over(self, tmp_path, monkeypatch):
        """An image whose WORKDIR is `/` with a user override: `/` keeps root's ownership."""
        server, task, _, _ = make_server(tmp_path, monkeypatch, dockerfile="FROM ubuntu:24.04\nUSER runner\n")
        assert task.user == "runner" and task.workdir in (None, "/")

        class RootPwdSandbox(FakeSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if command == "pwd":
                    return SandboxExecResult(stdout="/\n", stderr="", return_code=0)
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

        def workdir_commands(sandbox) -> list[str]:
            # The /logs folders are prepared by the same helper; only the working directory commands matter here.
            return [call["command"] for call in sandbox.execs if "/logs" not in call["command"]]

        sandbox = RootPwdSandbox()
        assert await server._prepare_workdir(sandbox, task, None) == "/"
        # Only the `pwd` probe ran (answered above the recording fake): no `mkdir -p /`, no `chown ... /`.
        assert workdir_commands(sandbox) == []

        # A named directory below the root is still created and handed over.
        named = RootPwdSandbox()
        assert await server._prepare_workdir(named, task, "/work") == "/work"
        assert workdir_commands(named) == ["mkdir -p /work && chown runner /work"]

    def test_resources_override_and_no_injected_env(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
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


class TestAgentStage:
    async def test_lost_agent_seed_is_a_retryable_agent_failure_and_stops_the_sandbox(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        server.server_client.fail_path = "/v1/agent_sessions"

        response = await run_episode(server, task)

        assert response.failure.stage == "agent" and not response.failure.terminal
        assert "response lost" in response.failure.failure_reason
        # The harness never worked, so nothing was verified; the sandbox and the agent session are closed anyway.
        assert not any("test.sh" in call["command"] for call in sandbox.execs)
        assert server.server_client.paths() == ["/v1/agent_sessions", "/v1/agent_sessions/close"]
        assert sandbox.stopped and not server._sandboxes

    @pytest.mark.parametrize(("status", "terminal"), [(422, True), (503, False), (429, False)])
    async def test_agent_http_errors_are_terminal_only_for_client_faults(
        self, tmp_path, monkeypatch, status, terminal
    ):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        request_info = RequestInfo(URL("http://agent/v1/responses"), "POST", CIMultiDictProxy(CIMultiDict()))
        server.server_client.fail_with["/ng-rollout/r1/v1/responses"] = ClientResponseError(
            request_info, (), status=status, message="agent said no"
        )

        response = await run_episode(server, task)

        assert response.failure.stage == "agent" and response.failure.terminal is terminal
        assert "agent said no" in response.failure.failure_reason
        assert sandbox.stopped and not server._sandboxes

    async def test_failed_turn_closes_the_session_and_keeps_the_agents_evidence(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        server.server_client.fail_path = "/ng-rollout/r1/v1/responses"
        server.server_client.observations = {"source": "harness", "records": [], "gaps": []}

        response = await run_episode(server, task)

        assert response.failure.stage == "agent" and not response.failure.terminal
        assert response.failure.agent_observations is not None
        assert server.server_client.paths()[-1] == "/v1/agent_sessions/close"
        assert not any("test.sh" in call["command"] for call in sandbox.execs)
        assert sandbox.stopped

    async def test_failed_close_is_a_cleanup_failure_with_the_turn_attached(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        server.server_client.fail_path = "/v1/agent_sessions/close"

        response = await run_episode(server, task)

        assert response.failure.stage == "cleanup" and not response.failure.terminal
        assert response.failure.partial_response.id == "response"
        # Verification never ran: the harness may still have been working in the sandbox.
        assert not any("test.sh" in call["command"] for call in sandbox.execs)
        # The final unwind retried the close, then stopped the sandbox.
        assert server.server_client.paths().count("/v1/agent_sessions/close") == 2
        assert sandbox.stopped

    async def test_observations_travel_with_the_result(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        server.server_client.observations = {"source": "harness", "records": [], "gaps": []}
        response = await run_episode(server, task)
        assert response.result.ng_agent_observations is not None
        assert response.result.model_dump()["ng_agent_observations"]["source"] == "harness"

    async def test_episode_deadline_stops_the_sandbox(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, default_episode_timeout_seconds=0.1)
        server.server_client.slow_path = "/ng-rollout/r1/v1/responses"

        response = await run_episode(server, task)

        assert response.failure is not None and "timed out" in response.failure.failure_reason
        assert not response.failure.terminal
        assert server.server_client.paths()[-1] == "/v1/agent_sessions/close"
        assert sandbox.stopped and not server._sandboxes


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
    def overlay_server(
        self, tmp_path, monkeypatch, *, dockerfile=OVERLAY_DOCKERFILE, toml=TASK_TOML, sandbox=None, image_config=None
    ):
        sandbox = sandbox or OverlaySandbox()
        server, task, _, created = make_server(
            tmp_path, monkeypatch, sandbox, dockerfile=dockerfile, image_config=image_config
        )
        (task.path / "task.toml").write_text(toml)
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )

        async def create(task, workdir, *, episode_key=None):
            # The sandbox starts with the spec's environment, as the provider's container would.
            sandbox.spec = server._sandbox_spec(task, workdir)
            sandbox.spec_env = dict(sandbox.spec.env)
            sandbox.created_in(workdir)
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
        assert overlay_workdir_command("/app", "root") == overlay_workdir_command("/app", "0") == "mkdir -p /app"
        # root with a group of its own is not plain root: the directory gets that group, as in a build.
        assert overlay_workdir_command("/app", "0:0") == "test -d /app || (mkdir -p /app && chown 0:0 /app)"
        assert overlay_workdir_command("/home/agent/w", "agent") == (
            "test -d /home/agent/w || (mkdir -p /home/agent/w && chown agent /home/agent/w)"
        )
        assert overlay_workdir_command("/a b", "agent:staff") == (
            "test -d '/a b' || (mkdir -p '/a b' && chown agent:staff '/a b')"
        )

    async def test_run_lines_see_the_environment_directory_and_user_of_their_point_in_the_dockerfile(
        self, tmp_path, monkeypatch
    ):
        server, task, sandbox, created = self.overlay_server(tmp_path, monkeypatch, toml=HEALTHCHECK_TOML)
        assert len(task.overlay) == 3 and task.user == "agent"

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        # An overlay task's sandbox starts without the task workdir: the overlay's WORKDIR step makes it.
        assert created == [("hello", None)]
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
        # Lines before `USER agent` run as root, which is the image's own default user, so without an override
        # (every provider supports the plain exec path); after it, as agent.
        assert task.sandbox_user is None
        assert [c["user"] for c in (first, second, third)] == [None, None, "agent"]
        assert all(c["cwd"] == "/" and c["env"] is None for c in (first, second, third))
        commands = [c["command"] for c in sandbox.execs]
        # /app is made as root when the first RUN after `WORKDIR /app` comes up, once, then only entered.
        assert commands[:4] == [first["command"], "mkdir -p /app", second["command"], third["command"]]
        assert first["command"] == f"unset MSG; export PATH={IMAGE_PATH}; cd / || exit 1; {first['seen']['command']}"
        assert sandbox.execs[1]["user"] is None and commands.count("mkdir -p /app") == 1
        # The overlay shares the task's build budget: the first line sees all of it, later ones what is left.
        assert first["timeout_s"] == pytest.approx(task.config.environment.build_timeout_sec, abs=1)
        assert second["timeout_s"] <= first["timeout_s"] and third["timeout_s"] <= second["timeout_s"]
        # Only then does Gym prepare the agent's workdir, poll the healthcheck and hand the sandbox over.
        assert commands[4] == "mkdir -p /app && chown agent /app" and sandbox.dirs["/app"] == "agent"
        assert commands.index("test -f /app/ready.txt") > commands.index(third["command"])
        # The harness got its session only after all of it.
        assert server.server_client.seed_bodies()

    async def test_root_image_without_user_keeps_the_plain_exec_path(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile="FROM img\nWORKDIR /app\nRUN true\nRUN false\n"
        )
        assert (await run_episode(server, task)).failure is None
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

    async def test_workdir_after_user_is_made_as_root_and_owned_by_the_user(self, tmp_path, monkeypatch):
        # The fake enforces what a container would: a non-root user cannot create under a root-owned directory.
        assert OverlaySandbox()._mkdir("/home/agent/work", "agent").return_code == 1
        server, task, sandbox, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile="FROM img\nUSER agent\nWORKDIR /home/agent/work\nRUN touch made\n"
        )
        assert task.user == "agent" and task.workdir == "/home/agent/work"

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        prepare, run = sandbox.execs[:2]
        assert prepare["command"] == (
            "test -d /home/agent/work || (mkdir -p /home/agent/work && chown agent /home/agent/work)"
        )
        assert prepare["user"] is None  # the image's default user is root already
        assert sandbox.dirs["/home/agent/work"] == "agent" and sandbox.dirs["/home/agent"] == "root"
        assert run["seen"] == {
            "command": "touch made",
            "env": {"PATH": IMAGE_PATH},
            "user": "agent",
            "cwd": "/home/agent/work",
        }

    async def test_numeric_user_is_resolved_to_its_name_in_the_image(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile="FROM img\nUSER 1000\nWORKDIR /home/agent/work\nRUN id\nRUN id -g\n"
        )
        assert task.user == "1000"

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        lookup, prepare, first, second = sandbox.execs[:4]
        assert lookup["command"] == "getent passwd 1000 | cut -d: -f1" and lookup["user"] is None
        # The name serves the directory's owner and both lines; the lookup happens once.
        assert prepare["command"].endswith("chown agent /home/agent/work)") and prepare["user"] is None
        assert [c["seen"]["user"] for c in (first, second)] == ["agent", "agent"]
        assert sum("getent" in c["command"] for c in sandbox.execs) == 1

    async def test_root_steps_on_a_non_root_image_override_the_image_user(self, tmp_path, monkeypatch):
        # jupyter/base-notebook style: the image runs as jovyan, the Dockerfile switches to root and stays there.
        # The sandbox's default exec user is the image's, so every root step needs the override, including the
        # WORKDIR directory and the agent's workdir, both made by root.
        sandbox = OverlaySandbox(image_user="jovyan")
        server, task, sandbox, _ = self.overlay_server(
            tmp_path,
            monkeypatch,
            dockerfile="FROM img\nUSER root\nWORKDIR /opt/tools\nRUN apt-get install -y gcc\n",
            sandbox=sandbox,
            image_config={"Env": [f"PATH={IMAGE_PATH}"], "User": "jovyan"},
        )
        assert task.user == "root" and task.sandbox_user == "jovyan"

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        # The overlay's WORKDIR step and the agent workdir preparation both create it, as root.
        assert [c["user"] for c in sandbox.execs if "mkdir -p /opt/tools" in c["command"]] == ["root", "root"]
        run = sandbox.run("apt-get install")
        assert run["user"] == "root" and run["seen"]["user"] == "root" and run["seen"]["cwd"] == "/opt/tools"
        assert sandbox.dirs["/opt/tools"] == "root"

    async def test_steps_on_a_non_root_image_without_a_user_line_run_as_the_image_user(self, tmp_path, monkeypatch):
        sandbox = OverlaySandbox(image_user="jovyan")
        server, task, sandbox, _ = self.overlay_server(
            tmp_path,
            monkeypatch,
            dockerfile="FROM img\nWORKDIR /home/jovyan/work\nRUN touch made\n",
            sandbox=sandbox,
            image_config={"Env": [f"PATH={IMAGE_PATH}"], "User": "jovyan"},
        )
        assert task.user == "jovyan" and task.sandbox_user == "jovyan"
        sandbox.dirs["/home/jovyan"] = "jovyan"

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        prepare, run = sandbox.execs[:2]
        # The directory is still made as root (with the override) and handed to the image user; the line runs
        # as that user with no override, as it would in a build.
        assert prepare["user"] == "root" and prepare["command"].endswith("chown jovyan /home/jovyan/work)")
        assert run["user"] == "jovyan" and run["seen"]["user"] == "jovyan"
        assert sandbox.dirs["/home/jovyan/work"] == "jovyan"

    async def test_a_numeric_user_without_a_passwd_entry_is_passed_through_as_a_uid(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile="FROM img\nUSER 4242\nRUN id\n"
        )
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        lookup, run = sandbox.execs[:2]
        # The uid goes to the exec as a number: Docker's --user and OpenSandbox's uid take it, where the string
        # "4242" would be looked up as an account name by the su-backed providers and fail.
        assert lookup["command"] == "getent passwd 4242 | cut -d: -f1"
        assert run["user"] == 4242 and isinstance(run["user"], int) and run["seen"]["user"] == "4242"
        # A uid:gid pair stays as written: only Docker's --user takes the pair.
        server, task, sandbox, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile="FROM img\nUSER 4242:4242\nRUN id\n"
        )
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert sandbox.execs[1]["user"] == "4242:4242"
        # A provider that refuses the uid fails the line itself, with its message: the Dockerfile does not apply.
        sandbox = OverlaySandbox(
            failing="id", failure=SandboxExecResult(stdout="", stderr="unable to find user 4242\n", return_code=125)
        )
        server, task, sandbox, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile="FROM img\nUSER 4242\nRUN id\n", sandbox=sandbox
        )
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "RUN line 1/1 'id' exited 125" in response.failure.failure_reason
        assert "unable to find user 4242" in response.failure.failure_reason and sandbox.stopped

    @pytest.mark.parametrize("user", ["root:1234", "0:0"])
    async def test_root_with_a_group_keeps_the_group_on_its_workdir_and_runs_the_line_as_root(
        self, tmp_path, monkeypatch, user
    ):
        server, task, sandbox, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile=f"FROM img\nUSER {user}\nWORKDIR /srv/data\nRUN id -g\n"
        )
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        prepare, run = sandbox.execs[:2]
        # The directory gets the group a build would give it; the line itself runs as plain root, since the
        # su-backed providers (OpenSandbox, enroot, apptainer) reject "root:1234" or "0:0" as an account name.
        assert prepare["command"] == f"test -d /srv/data || (mkdir -p /srv/data && chown {user} /srv/data)"
        assert prepare["user"] is None and sandbox.dirs["/srv/data"] == user
        assert run["user"] is None and run["seen"]["user"] == "root"
        # On a non-root image the same line takes the root override, as every other root step does.
        server, task, sandbox, _ = self.overlay_server(
            tmp_path,
            monkeypatch,
            dockerfile=f"FROM img\nUSER {user}\nRUN id -g\n",
            sandbox=OverlaySandbox(image_user="jovyan"),
            image_config={"Env": [f"PATH={IMAGE_PATH}"], "User": "jovyan"},
        )
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert sandbox.execs[0]["user"] == "root" and sandbox.execs[0]["seen"]["user"] == "root"

    async def test_the_final_workdir_belongs_to_the_user_of_its_instruction(self, tmp_path, monkeypatch):
        # node:20 style: USER node, then WORKDIR, then a RUN that writes there as node. A sandbox created with the
        # task workdir already has it, root-owned, and the RUN line fails; the overlay must make it instead.
        sandbox = OverlaySandbox(passwd={"0": "root", "1000": "node"})
        sandbox.dirs["/home/node"] = "node"
        server, task, sandbox, created = self.overlay_server(
            tmp_path,
            monkeypatch,
            dockerfile="FROM node:20\nUSER node\nWORKDIR /home/node/app\nRUN npm init -y\n",
            sandbox=sandbox,
        )
        assert task.user == "node" and task.workdir == "/home/node/app"
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert created == [("hello", None)]
        run = sandbox.run("npm init")
        assert run["seen"]["user"] == "node" and run["seen"]["cwd"] == "/home/node/app"
        assert run["cwd_owner"] == "node", "the directory was root-owned when the RUN line ran"
        prepare = next(c for c in sandbox.execs if c["command"].endswith("chown node /home/node/app)"))
        assert sandbox.execs.index(prepare) < sandbox.execs.index(run)
        assert server.server_client.seed_bodies()[0]["sandbox_access"]["workdir"] == "/home/node/app"

    async def test_the_compose_main_service_of_an_overlay_task_starts_without_the_task_workdir_too(
        self, tmp_path, monkeypatch
    ):
        # The same node:20 shape as a Compose task: the main service's spec must not name the task workdir
        # either, or the provider creates it root-owned before the overlay's WORKDIR step can.
        sandbox = OverlaySandbox(passwd={"0": "root", "1000": "node"})
        sandbox.dirs["/home/node"] = "node"
        server, task, _, _ = make_server(
            tmp_path,
            monkeypatch,
            sandbox,
            dockerfile="FROM node:20\nUSER node\nWORKDIR /home/node/app\nRUN npm init -y\n",
        )
        (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (task.path.parent / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        write_image_config(task.path.parent, "node:20", Env=[f"PATH={IMAGE_PATH}"])
        task = load_task(task.path)
        assert task.needs_compose and task.overlay and task.workdir == "/home/node/app" and task.user == "node"
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        specs: dict[str, SandboxSpec] = {}

        class CapturingCompose(FakeCompose):
            def __init__(self, provider_config, path, *, service_specs, timeout_s):
                super().__init__({"main": sandbox, "api": FakeSandbox(), "kafka": FakeSandbox()})
                specs.update(service_specs)
                # What a provider does with the spec's workdir: it exists, root-owned, before anything runs.
                sandbox.created_in(service_specs["main"].workdir)

            async def start(self):
                pass

        monkeypatch.setattr("environment_servers.harbor.app.AsyncSandboxCompose", CapturingCompose)

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        # Neither main nor the sidecars derived from it start in the task workdir.
        assert specs["main"].workdir is None and specs["api"].workdir is None
        run = sandbox.run("npm init")
        assert run["seen"]["user"] == "node" and run["seen"]["cwd"] == "/home/node/app"
        assert run["cwd_owner"] == "node", "the directory was root-owned when the RUN line ran"
        assert server.server_client.seed_bodies()[0]["sandbox_access"]["workdir"] == "/home/node/app"

    async def test_every_workdir_instruction_is_created_in_order_even_without_a_run_after_it(
        self, tmp_path, monkeypatch
    ):
        server, task, sandbox, _ = self.overlay_server(
            tmp_path,
            monkeypatch,
            dockerfile="FROM img\nWORKDIR /tmp/a\nWORKDIR /\nRUN test -d /tmp/a\nUSER agent\nWORKDIR /home/agent/last\n",
        )
        assert [(w.path, w.user, w.before) for w in task.overlay_workdirs] == [
            ("/tmp/a", None, 0),
            ("/", None, 0),
            ("/home/agent/last", "agent", 1),
        ]
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        commands = [c["command"] for c in sandbox.execs]
        run = sandbox.run("test -d /tmp/a")
        # /tmp/a was made (as root, the user of its instruction) before the RUN line that checks it...
        assert commands.index("mkdir -p /tmp/a") < sandbox.execs.index(run) and sandbox.dirs["/tmp/a"] == "root"
        # ...and the WORKDIR after the last RUN line exists too, owned by agent, before the workdir preparation.
        trailing = next(c for c in sandbox.execs if c["command"].endswith("chown agent /home/agent/last)"))
        assert sandbox.execs.index(run) < sandbox.execs.index(trailing) and sandbox.dirs["/home/agent/last"] == "agent"
        assert commands.index("mkdir -p /home/agent/last && chown agent /home/agent/last") > sandbox.execs.index(
            trailing
        )

    @pytest.mark.skipif(shutil.which("bash") is None or shutil.which("sh") is None, reason="sh and bash are required")
    def test_an_inherited_shell_runs_the_line_and_exec_form_bypasses_it(self):
        pipefail = OverlayRun(command="false | true", shell=("/bin/bash", "-o", "pipefail", "-c"))
        line = overlay_shell_command(pipefail)
        assert line == "cd / || exit 1; exec /bin/bash -o pipefail -c 'false | true'"
        # The build failed this line under bash -o pipefail; the provider's sh would have passed it.
        assert subprocess.run(["sh", "-c", line]).returncode != 0
        assert subprocess.run(["sh", "-c", overlay_shell_command(OverlayRun(command="false | true"))]).returncode == 0
        exec_form = OverlayRun(command=shlex.join(["bash", "-lc", "echo hi"]), shell=None)
        assert overlay_shell_command(exec_form) == "cd / || exit 1; exec bash -lc 'echo hi'"
        assert subprocess.run(["sh", "-c", overlay_shell_command(exec_form)], capture_output=True).stdout == b"hi\n"

    async def test_a_declared_image_built_from_a_user_line_gets_the_root_override(self, tmp_path, monkeypatch):
        toml = TASK_TOML.replace("[environment]\n", '[environment]\ndocker_image = "org/app:1"\n')
        server, task, sandbox, _ = make_server(
            tmp_path, monkeypatch, FakeSandbox(image_user="app"), toml=toml, dockerfile="FROM python:3.12\nUSER app\n"
        )
        # The declared image runs as the Dockerfile's final USER, so root work needs the override.
        assert task.image == "org/app:1" and task.sandbox_user == "app"
        await _exec_as_root_user(sandbox, "true", configured_user=task.sandbox_user)
        assert sandbox.execs[-1]["user"] == "root"

    async def test_failing_run_is_a_terminal_task_error_with_the_end_of_its_output(self, tmp_path, monkeypatch):
        stdout = "\n".join(f"line {number}" for number in range(1, 401)) + "\n"
        sandbox = OverlaySandbox(
            failing="apt-get install",
            failure=SandboxExecResult(stdout=stdout, stderr="E: no package\n", return_code=100),
        )
        server, task, sandbox, _ = self.overlay_server(tmp_path, monkeypatch, sandbox=sandbox)

        response = await run_episode(server, task)

        failure = response.failure
        assert failure.stage == "seed" and failure.terminal
        assert "Dockerfile overlay failed for 'hello'" in failure.failure_reason
        assert "RUN line 1/3 'apt-get update && apt-get install -y curl' exited 100" in failure.failure_reason
        # The tail keeps the end of a long log, where the error is, and drops the start.
        assert "line 400\nE: no package" in failure.failure_reason and "line 1\nline 2\n" not in failure.failure_reason
        # Nothing after the failing line ran, the sandbox is gone and the harness was never engaged.
        assert len(sandbox.execs) == 1 and sandbox.stopped and not server._sandboxes
        assert server.server_client.calls == []

    async def test_provider_failure_during_a_run_is_retriable(self, tmp_path, monkeypatch):
        sandbox = OverlaySandbox(
            failing="apt-get install",
            failure=SandboxExecResult(stdout=None, stderr="connection reset", return_code=-1, error_type="sandbox"),
        )
        server, task, sandbox, _ = self.overlay_server(tmp_path, monkeypatch, sandbox=sandbox)

        response = await run_episode(server, task)

        failure = response.failure
        assert failure.stage == "seed" and not failure.terminal
        assert "Could not set up sandbox for 'hello'" in failure.failure_reason
        assert (
            "RUN line 1/3 'apt-get update && apt-get install -y curl' could not run in the sandbox: sandbox"
            in failure.failure_reason
        )
        assert len(sandbox.execs) == 1 and sandbox.stopped
        assert server.server_client.calls == []

    async def test_timed_out_run_names_the_budget(self, tmp_path, monkeypatch):
        sandbox = OverlaySandbox(
            failing="echo ok", failure=SandboxExecResult(stdout="", stderr="", return_code=-1, error_type="timeout")
        )
        server, task, sandbox, _ = self.overlay_server(tmp_path, monkeypatch, sandbox=sandbox)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "RUN line 2/3" in response.failure.failure_reason
        assert "exceeded [environment].build_timeout_sec=600" in response.failure.failure_reason
        assert sandbox.stopped

    async def test_exhausted_budget_skips_the_remaining_lines(self, tmp_path, monkeypatch):
        class SlowSandbox(OverlaySandbox):
            async def exec(self, command, **kwargs):
                await asyncio.sleep(0.05)  # longer than the whole build budget below
                return await super().exec(command, **kwargs)

        toml = TASK_TOML.replace("cpus = 1", "build_timeout_sec = 0.01\ncpus = 1")
        server, task, sandbox, _ = self.overlay_server(tmp_path, monkeypatch, toml=toml, sandbox=SlowSandbox())
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "RUN line 2/3" in response.failure.failure_reason
        assert "build_timeout_sec=0.01 is used up" in response.failure.failure_reason
        assert len(sandbox.execs) == 1 and sandbox.stopped

    def test_sandbox_ttls_stay_episode_sized_when_there_is_an_overlay(self, tmp_path, monkeypatch):
        server, task, _, _ = self.overlay_server(tmp_path, monkeypatch, toml=SEPARATE_IN_TASK_IMAGE_TOML)
        config, slack = task.config, server.config.sandbox_ttl_slack_s
        assert config.environment.build_timeout_sec == 600 and task.overlay
        agent_spec = server._sandbox_spec(task, "/app")
        episode = server.config.default_episode_timeout_seconds + server.config.cleanup_timeout_seconds
        # The agent sandbox lives for the environment's whole episode, which already covers the overlay.
        assert agent_spec.ttl_s == episode + slack

        specs = []

        class RecordingSandbox:
            def __init__(self, provider_config):
                pass

            async def start(self, spec):
                specs.append(spec)

        monkeypatch.setattr("environment_servers.harbor.app.AsyncSandbox", RecordingSandbox)
        asyncio.run(server._create_verifier_sandbox(task))
        (verifier_spec,) = specs
        # The verifier gets the same episode-sized lifetime, overlay included: `timeout_sec + slack (+ build)`
        # would not clear a provider's renew interval (TB4's 1800 s), which masks every separate-mode episode.
        assert verifier_spec.ttl_s == agent_spec.ttl_s == episode + slack >= 1800 + config.verifier.timeout_sec
        # The verifier runs in the task's own image: it gets the same resolved Dockerfile environment as the agent.
        assert verifier_spec.image == task.image == "debian:bookworm-slim"
        assert verifier_spec.env == agent_spec.env == {"PATH": f"/opt/tools/bin:{IMAGE_PATH}", "MSG": "hi there"}

        # A pull-mode task has no overlay and the same lifetimes.
        server, task, _, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile="FROM img\nWORKDIR /app\n", toml=SEPARATE_IN_TASK_IMAGE_TOML
        )
        assert not task.overlay
        assert server._sandbox_spec(task, "/app").ttl_s == episode + slack
        specs.clear()
        asyncio.run(server._create_verifier_sandbox(task))
        assert specs[0].ttl_s == episode + slack

    def separate_verifier(self, tmp_path, monkeypatch, *, toml=SEPARATE_IN_TASK_IMAGE_TOML, verifier=None):
        server, task, agent, _ = self.overlay_server(tmp_path, monkeypatch, toml=toml)
        verifier = verifier or OverlaySandbox()

        async def create_verifier(task, *, episode_key=None):
            verifier.spec_env = dict(
                server._sandbox_spec(task, None, environment=task.config.verifier.environment).env
            )
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        return server, task, agent, verifier

    async def test_separate_verifier_in_the_task_image_gets_the_overlay(self, tmp_path, monkeypatch):
        server, task, agent, verifier = self.separate_verifier(tmp_path, monkeypatch)
        assert not task.config.is_shared_verifier and _verifier_image(task) == "debian:bookworm-slim"

        result = (await run_episode(server, task)).result

        assert result.reward == 1.0 and result.verifier_mode == "separate", result
        assert result.mask_sample is False and result.failure_kind is None
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

    async def test_separate_verifier_overlay_failure_masks_the_sample(self, tmp_path, monkeypatch):
        verifier = OverlaySandbox(failing="apt-get install")
        server, task, agent, verifier = self.separate_verifier(tmp_path, monkeypatch, verifier=verifier)

        response = await run_episode(server, task)

        # The rollout is complete; a verifier sandbox that cannot take the overlay must not discard it.
        assert response.failure is None, response.failure
        result = response.result
        assert result.reward == 0.0 and result.mask_sample is True
        assert result.failure_kind == OVERLAY_FAILED_KIND == "harbor:overlay_failed"
        assert "Dockerfile overlay failed in the verifier sandbox" in result.failure_reason
        assert "RUN line 1/3 'apt-get update && apt-get install -y curl' exited 100" in result.failure_reason
        assert "E: no package" in result.failure_reason
        assert not any("test.sh" in c["command"] for c in verifier.execs)
        assert agent.stopped and verifier.stopped

    async def test_separate_verifier_provider_failure_during_the_overlay_masks_the_sample(self, tmp_path, monkeypatch):
        verifier = OverlaySandbox(
            failing="apt-get install",
            failure=SandboxExecResult(stdout=None, stderr="gone", return_code=-1, error_type="sandbox"),
        )
        server, task, _, verifier = self.separate_verifier(tmp_path, monkeypatch, verifier=verifier)
        result = (await run_episode(server, task)).result
        assert result.mask_sample is True and result.failure_kind == "provider_unavailable"
        assert "could not run in the sandbox: sandbox" in result.failure_reason and verifier.stopped

    async def test_prebuilt_verifier_image_gets_no_overlay(self, tmp_path, monkeypatch):
        toml = TASK_TOML.replace(
            "[verifier]\n",
            '[verifier]\nenvironment_mode = "separate"\n\n[verifier.environment]\ndocker_image = "org/verifier:1"\n',
        )
        server, task, _, verifier = self.separate_verifier(tmp_path, monkeypatch, toml=toml)
        assert _verifier_image(task) == "org/verifier:1"
        assert (await run_episode(server, task)).result.reward == 1.0
        assert not any("apt-get" in c["command"] for c in verifier.execs)
        # Nor the agent image's Dockerfile environment.
        assert verifier.spec_env == {}


class TestVerify:
    async def verified(self, tmp_path, monkeypatch, sandbox=None, **kwargs):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, sandbox, **kwargs)
        response = await run_episode(server, task)
        return server, task, sandbox, response

    async def test_verify_runs_test_sh_and_reads_reward(self, tmp_path, monkeypatch):
        server, task, sandbox, response = await self.verified(tmp_path, monkeypatch)

        result = response.result
        assert result.reward == 1.0 and result.mask_sample is False and result.failure_kind is None
        assert result.verifier_rewards == {"reward": 1.0} and result.verifier_return_code == 0

        run = next(call for call in sandbox.execs if "test.sh" in call["command"])
        assert run["command"] == verifier_command(120)
        # The portable short form (`--signal=` fails on BusyBox), guarded so a missing wrapper exits 127
        # before anything is redirected, and followed by the exit record the server reads the deadline from.
        assert run["command"].startswith(f"{TIMEOUT_WRAPPER_GUARD}; start=$(date +%s); ")
        assert (
            "timeout -s KILL 120 bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1; rc=$?; " in run["command"]
        )
        assert run["command"].endswith('echo "$rc $start $(date +%s)" > /logs/verifier/.ng-exit; exit $rc')
        assert "--signal" not in run["command"]
        assert run["cwd"] == "/app"
        # The exec itself is bounded by the budget plus the grace period; the provider keeps it alive that long.
        assert run["timeout_s"] == 120 + server.config.verifier_grace_s
        commands = [call["command"] for call in sandbox.execs]
        wipe = next(i for i, command in enumerate(commands) if "-mindepth 1 -delete" in command)
        upload = next(i for i, command in enumerate(commands) if "tar -xzf" in command and "/tests" in command)
        assert wipe < upload < commands.index(run["command"])
        # An image may have a non-root default even when the task has no user override.
        prepare = next(
            call for call in sandbox.execs if "chmod 777" in call["command"] and "/tests" in call["command"]
        )
        assert prepare["user"] == "root" and prepare["cwd"] == "/"
        # tests/ was uploaded as an archive and unpacked into /tests.
        assert any(remote.endswith(".tar.gz") for _, remote in sandbox.uploads)
        assert Path(result.verifier_logs_dir).is_absolute()
        assert (Path(result.verifier_logs_dir) / "reward.txt").read_text() == "1\n"

    async def test_prepare_runs_as_root_on_a_non_root_image(self, tmp_path, monkeypatch):
        class NonRootSandbox(FakeSandbox):
            """The image's default user may not touch root-owned /logs/verifier."""

            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                result = await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)
                if ("chmod" in command or "chown" in command) and user != "root":
                    return SandboxExecResult(stdout="", stderr="chmod: Permission denied", return_code=1)
                return result

        # The recorded base image runs as `app`, as the NonRootSandbox models.
        server, task, sandbox, response = await self.verified(
            tmp_path,
            monkeypatch,
            NonRootSandbox(),
            dockerfile="FROM ubuntu:24.04\nWORKDIR /app\nUSER app",
            image_config={"Env": [f"PATH={IMAGE_PATH}"], "User": "app"},
        )
        assert task.user == "app" and task.sandbox_user == "app"
        assert response.result.reward == 1.0 and response.result.failure_kind is None
        prepare = next(
            call for call in sandbox.execs if "chmod 777" in call["command"] and "/tests" in call["command"]
        )
        assert prepare["user"] == "root"
        # test.sh itself still runs as the verifier's user, not root.
        run = next(call for call in sandbox.execs if "test.sh" in call["command"])
        assert run["user"] == task.config.verifier.user

    async def test_json_reward_with_components(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(verifier_files={"reward.json": json.dumps({"reward": 0.5, "tests_passed": 3})})
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        assert response.result.reward == 0.5
        assert response.result.verifier_rewards == {"reward": 0.5, "tests_passed": 3.0}

    async def test_ambiguous_reward_is_a_terminal_verification_failure(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(verifier_files={"reward.json": json.dumps({"a": 1, "b": 0})})
        _, _, sandbox, response = await self.verified(tmp_path, monkeypatch, sandbox)
        failure = response.failure
        assert failure.stage == "verification" and failure.terminal
        assert "several keys" in failure.failure_reason
        assert failure.partial_response.id == "response"
        assert sandbox.stopped

    async def test_missing_reward_scores_zero_and_is_measured(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(
            verifier_files={"test-stdout.txt": "pytest exploded"},
            test_result=SandboxExecResult(stdout="", stderr="", return_code=1),
        )
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        result = response.result
        assert result.reward == 0.0 and result.mask_sample is False
        assert result.failure_kind == "harbor:missing_reward"
        assert "pytest exploded" in result.failure_reason
        assert result.verifier_return_code == 1

    async def test_invalid_reward_scores_zero(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(verifier_files={"reward.txt": "maybe"})
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        assert response.result.reward == 0.0 and response.result.failure_kind == "harbor:invalid_reward"

    async def test_verifier_timeout_when_exit_code_never_appears(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(
            verifier_files={},
            test_result=SandboxExecResult(stdout=None, stderr=None, return_code=125, error_type="timeout"),
        )
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, sandbox)
        server.config.verifier_grace_s = 0
        (task.path / "task.toml").write_text(
            TASK_TOML.replace("timeout_sec = 120.0\n\n[agent]", "timeout_sec = 0.01\n\n[agent]")
        )
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        result = (await run_episode(server, task)).result
        assert result.reward == 0.0 and result.failure_kind == "harbor:verifier_timeout"
        assert result.mask_sample is False
        # A sub-second budget is rounded up to one whole second: `timeout 0` would mean no limit at all.
        run = next(call for call in sandbox.execs if "test.sh" in call["command"])
        assert "timeout -s KILL 1 bash /tests/test.sh" in run["command"]

    async def test_exit_137_before_the_deadline_keeps_the_reward(self, tmp_path, monkeypatch):
        """A grader that writes its reward and then ends 137 (a killed helper under `wait`) did not time out."""
        sandbox = FakeSandbox(
            verifier_files={"reward.txt": "1\n"},
            test_result=SandboxExecResult(stdout="", stderr="", return_code=137),
            test_elapsed=5,
        )
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        result = response.result
        assert result.reward == 1.0 and result.mask_sample is False and result.failure_kind is None
        assert result.verifier_return_code == 137

    async def test_exit_137_at_the_deadline_without_a_reward_is_a_timeout(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(
            verifier_files={"test-stdout.txt": "still running"},
            test_result=SandboxExecResult(stdout="", stderr="", return_code=137),
            test_elapsed=120,
        )
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        result = response.result
        assert result.reward == 0.0 and result.mask_sample is False
        assert result.failure_kind == "harbor:verifier_timeout"
        assert "still running" in result.failure_reason

    async def test_reward_written_before_the_kill_still_counts(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(
            verifier_files={"reward.txt": "0.5\n"},
            test_result=SandboxExecResult(stdout="", stderr="", return_code=137),
            test_elapsed=120,
        )
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        assert response.result.reward == 0.5 and response.result.failure_kind is None

    async def test_exit_137_without_timestamps_falls_back_to_the_exit_code(self, tmp_path, monkeypatch):
        """An image without `date` leaves no elapsed time: 137 then means the timer, as before."""
        sandbox = FakeSandbox(
            verifier_files={EXIT_RECORD_NAME: "137  \n"},
            test_result=SandboxExecResult(stdout="", stderr="", return_code=137),
        )
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        assert response.result.failure_kind == "harbor:verifier_timeout"

    async def test_missing_timeout_wrapper_masks_the_sample(self, tmp_path, monkeypatch):
        """BusyBox without `timeout`, or an image without `bash`: the guard exits 127 before test.sh starts."""
        sandbox = FakeSandbox(
            verifier_files={},
            test_result=SandboxExecResult(stdout="", stderr="", return_code=127),
            wrapper_ran=False,
        )
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        result = response.result
        assert result.reward == 0.0 and result.mask_sample is True
        assert result.failure_kind == "verifier_error"
        assert "could not be started" in result.failure_reason and "exit 127" in result.failure_reason

    async def test_test_sh_exiting_127_itself_is_measured(self, tmp_path, monkeypatch):
        """The wrapper ran: a script that ends 127 (a missing command of its own) is an ordinary measured zero."""
        sandbox = FakeSandbox(verifier_files={}, test_result=SandboxExecResult(stdout="", stderr="", return_code=127))
        _, _, _, response = await self.verified(tmp_path, monkeypatch, sandbox)
        result = response.result
        assert result.mask_sample is False and result.failure_kind == "harbor:missing_reward"
        assert result.verifier_return_code == 127

    async def test_sandbox_runtime_failure_masks(self, tmp_path, monkeypatch):
        class BrokenLaunch(FakeSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if "test.sh" in command:
                    return SandboxExecResult(stdout=None, stderr="gone", return_code=125, error_type="sandbox")
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

        _, _, _, response = await self.verified(tmp_path, monkeypatch, BrokenLaunch())
        assert response.result.reward == 0.0 and response.result.mask_sample is True
        assert response.result.failure_kind == "verifier_error"

    async def test_transfer_exception_masks(self, tmp_path, monkeypatch):
        class BrokenUpload(FakeSandbox):
            async def upload(self, local_path, remote_path):
                raise ConnectionError("lost")

        _, _, _, response = await self.verified(tmp_path, monkeypatch, BrokenUpload())
        assert response.result.mask_sample is True
        assert response.result.failure_kind == "provider_unavailable"
        assert "lost" in response.result.failure_reason

    async def test_aggregate_metrics_scores_the_episode_results(self, tmp_path, monkeypatch):
        server, task, _, response = await self.verified(tmp_path, monkeypatch)
        row = response.result.model_dump(mode="json")
        rows = [
            row | {"_ng_task_index": 0, "_ng_rollout_index": 0},
            row | {"_ng_task_index": 0, "_ng_rollout_index": 1, "reward": 0.0},
        ]
        metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
        assert metrics.agent_metrics["mean/reward"] == 0.5


class TestOwnership:
    async def test_shutdown_stops_sandboxes_of_episodes_in_flight(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        entered, release = asyncio.Event(), asyncio.Event()

        async def hold():
            entered.set()
            await release.wait()

        server.server_client.turn_hooks.append(hold)
        running = asyncio.create_task(run_episode(server, task))
        await asyncio.wait_for(entered.wait(), 5)
        assert server._sandboxes and not sandbox.stopped
        await server.shutdown()
        assert sandbox.stopped and not server._sandboxes
        release.set()
        response = await running
        # The verifier still ran its commands against the stopped sandbox object here; a real provider would
        # fail them and the sample would be masked. The episode's own cleanup finds nothing left to stop.
        assert response.failure is not None or response.result is not None
        assert not server._sandboxes

    async def test_a_collect_hook_the_sandbox_cannot_run_masks(self, tmp_path, monkeypatch):
        class HookLost(AgentSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if command.startswith("cp /app/state.db"):
                    return SandboxExecResult(stdout="", stderr="", return_code=None, error_type="timeout")
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

        agent = HookLost()
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent, toml=SEPARATE_TOML)
        result = (await run_episode(server, task)).result
        assert result.mask_sample is True and result.failure_kind == "provider_unavailable"
        assert "collect hook could not run: timeout" in result.failure_reason and agent.stopped

    async def test_shared_mode_runs_collect_hooks_before_preparing_tests(self, tmp_path, monkeypatch):
        toml = (
            TASK_TOML
            + '\n[[verifier.collect]]\ncommand = "cp /app/state.db /logs/artifacts/state.db"\nuser = "root"\n'
        )
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=toml)
        result = (await run_episode(server, task)).result
        assert result.reward == 1.0 and result.verifier_mode == "shared"
        commands = [c["command"] for c in sandbox.execs]
        hook = commands.index("cp /app/state.db /logs/artifacts/state.db")
        prepare = next(i for i, command in enumerate(commands) if "-mindepth 1 -delete" in command)
        assert hook < prepare
        assert sandbox.execs[hook]["user"] == "root" and sandbox.execs[hook]["cwd"] == "/app"
        assert sandbox.execs[hook]["timeout_s"] == 60 + 30

    async def test_verifier_sandbox_lifetime_covers_the_whole_episode(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch, toml=SEPARATE_TOML)
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        specs = []

        class Capturing:
            def __init__(self, provider_config):
                pass

            async def start(self, spec):
                specs.append(spec)

        monkeypatch.setattr("environment_servers.harbor.app.AsyncSandbox", Capturing)
        await server._create_verifier_sandbox(task)
        agent_spec = server._sandbox_spec(task, "/app")
        (verifier_spec,) = specs
        # Readiness, transfers, test.sh with its grace and reward collection all fit, and so does the TB4
        # provider's 1800 s renew interval, which the old `timeout_sec + slack` (300 + 900) did not clear.
        assert verifier_spec.ttl_s == agent_spec.ttl_s >= 1800 + task.config.verifier.timeout_sec
        assert verifier_spec.image == "org/verifier:1" and verifier_spec.metadata["harbor_role"] == "verifier"


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
        assert " --anchored --exclude=./node_modules --exclude=./cache --exclude='./*.log' -C /app/build ." in build
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


class TestSeparateVerification:
    def prepared(self, tmp_path, monkeypatch, *, toml=SEPARATE_TOML, verifier=None, agent=None):
        agent = agent or AgentSandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent)
        (task.path / "task.toml").write_text(toml)
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        verifier = verifier or FakeSandbox()
        created = []

        async def create_verifier(task, *, episode_key=None):
            assert agent.stopped, "the agent sandbox must be stopped before the verifier is created"
            created.append(task.task_id)
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        return server, task, agent, verifier, created

    async def test_collects_artifacts_then_verifies_in_a_fresh_sandbox(self, tmp_path, monkeypatch):
        server, task, agent, verifier, created = self.prepared(tmp_path, monkeypatch)

        response = await run_episode(server, task)

        result = response.result
        assert result is not None, response.failure
        assert result.reward == 1.0 and result.verifier_mode == "separate"
        assert created == ["hello"]
        # The main-container collect hook ran in the agent sandbox; the sidecar hook was skipped.
        hooks = [c for c in agent.execs if "state.db" in c["command"] or "kafka-dump" in c["command"]]
        assert [h["command"] for h in hooks] == ["cp /app/state.db /logs/artifacts/state.db"]
        # /logs/artifacts (a directory) and the report (a file) were pulled from the agent sandbox, each
        # under its own staging root. Verifier logs live in a fresh per-verification folder; the row names it.
        artifacts = Path(result.verifier_logs_dir) / "artifacts"
        assert (artifacts / "00" / "logs" / "artifacts" / "state.db").read_text() == "db"
        assert (artifacts / "01" / "app" / "output" / "report.json").read_bytes() == b'{"ok": true}'
        # ...the agent sandbox was stopped before test.sh ran, and the verifier got tests plus artifacts back.
        assert agent.stopped
        # tests/, /logs/artifacts and the report all travel as archives, so the report keeps its mode bits.
        assert len(verifier.uploads) == 3 and all(remote.endswith(".tar.gz") for _, remote in verifier.uploads)
        unpacked = [c["command"] for c in verifier.execs if "tar -xzf" in c["command"]]
        assert any("-C /logs/artifacts;" in c for c in unpacked) and any("-C /app/output;" in c for c in unpacked)
        # A restored directory replaces what the image ships there and is world-writable, as Harbor leaves
        # it, so a verifier that drops privileges can still write scratch files next to the agent's output.
        assert any(
            c["command"]
            == "mkdir -p /logs/artifacts && find /logs/artifacts -mindepth 1 -delete && chmod 777 /logs/artifacts"
            for c in verifier.execs
        )
        # A file's parent is created world-writable when missing and widened when it exists; when the chmod is
        # refused (an image that cannot switch to root) the parent only has to be writable.
        prepare = (
            "if [ -d /app/output ]; then chmod a+rwx /app/output 2>/dev/null || test -w /app/output; "
            "else mkdir -p /app/output && chmod 777 /app/output; fi"
        )
        assert any(c["command"] == prepare for c in verifier.execs)
        run = next(c for c in verifier.execs if "test.sh" in c["command"] and "timeout" in c["command"])
        assert run["command"] == f"chmod +x /tests/test.sh; {verifier_command(300)}"
        assert (run["env"], run["user"], run["cwd"]) == ({"CHECK": "strict"}, "root", None)
        assert run["timeout_s"] == 300 + server.config.verifier_grace_s
        assert verifier.stopped
        # The episode's final cleanup found nothing left to stop: the agent sandbox is no longer owned.
        assert not server._sandboxes and not server._sessions

    def test_sandbox_spec_applies_dockerfile_env(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(
            tmp_path, monkeypatch, dockerfile="FROM ubuntu:24.04\nWORKDIR /app\nENV FOO=bar\nENV MODE=image\n"
        )
        (task.path / "task.toml").write_text(TASK_TOML + '\n[environment.env]\nMODE = "toml"\n')
        task = load_task(task.path)
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )

        spec = server._sandbox_spec(task, "/app")

        # Dockerfile ENV reaches the agent's sandbox, and task.toml [environment.env] wins over it.
        assert spec.env["FOO"] == "bar" and spec.env["MODE"] == "toml"

        verifier_environment = HarborEnvironment(docker_image="org/verifier:1", env={"ONLY": "verifier"})
        verifier_spec = server._sandbox_spec(task, None, environment=verifier_environment, role="verifier")
        assert "FOO" not in verifier_spec.env and verifier_spec.env["ONLY"] == "verifier"

    def test_verifier_sandbox_spec_uses_the_verifier_environment(self, tmp_path, monkeypatch):
        server, task, _, _, _ = self.prepared(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
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
        server, task, _, _, _ = self.prepared(tmp_path, monkeypatch, toml=toml)
        assert _verifier_image(task) == "org/agent:1"

    async def test_seed_rejects_a_verifier_that_must_be_built(self, tmp_path, monkeypatch):
        server, task, _, created = make_server(tmp_path, monkeypatch)
        toml = SEPARATE_TOML.replace('docker_image = "org/verifier:1"\n', "")
        (task.path / "task.toml").write_text(toml)
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "tests/Dockerfile" in response.failure.failure_reason and created == []

    async def test_verifier_failure_masks_and_stops_both_sandboxes(self, tmp_path, monkeypatch):
        class Broken(FakeSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                raise RuntimeError("verifier sandbox lost")

        broken = Broken()
        server, task, agent, verifier, _ = self.prepared(tmp_path, monkeypatch, verifier=broken)
        result = (await run_episode(server, task)).result
        assert result.mask_sample is True and result.failure_kind == "provider_unavailable"
        assert agent.stopped and verifier.stopped


class TestTransfersWithoutRoot:
    def test_falls_back_to_the_default_user_when_root_is_refused(self, tmp_path):
        import asyncio

        from environment_servers.harbor.sandbox_io import upload_dir

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

        from environment_servers.harbor.sandbox_io import SandboxTransferError, upload_dir

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


@dataclass
class MainWatchingSandbox(AgentSandbox):
    """A sidecar that notes, at each collect hook and artifact lookup, whether the main container was stopped."""

    main: FakeSandbox | None = None
    main_stopped_when_seen: list[bool] = field(default_factory=list)

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if "dump-topics" in command or command.startswith("if [ -d "):
            self.main_stopped_when_seen.append(self.main.stopped)
        return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)


def _record_stop(stops: list[str], name: str, sandbox: FakeSandbox):
    async def stop():
        stops.append(name)
        sandbox.stopped = True

    return stop


class TestComposeAndRouting:
    def compose_server(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "task.toml").write_text(COMPOSE_TOML)
        (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (task.path.parent / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict",
            lambda: {"sandbox": {"opensandbox": {}}, "sandbox_gpu": {"opensandbox": {}}},
        )
        main, api, kafka = HealthySandbox(), FakeSandbox(), FakeSandbox()
        compose = FakeCompose({"main": main, "api": api, "kafka": kafka})
        documents = []
        verifier = FakeSandbox()

        async def start_compose(task, episode_key):
            documents.append(server._compose_document(task))
            return compose

        async def create_verifier(task, *, episode_key=None):
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr(server, "_create_compose", start_compose)
        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        return server, task, compose, documents, verifier

    async def test_compose_document_resolves_variables_and_env_file_from_the_task_then_the_process_environment(
        self, tmp_path, monkeypatch
    ):
        # M19: the server reads the Compose file through the same resolver prepare used, so ${MAIN_IMAGE_NAME},
        # ${CONTEXT_DIR} and an env_file resolve from the task. G3: a variable the task does not set is read from
        # the server's process environment, as Harbor's docker compose reads it, so prepare and the server must
        # run with the same variables.
        server, task, compose, documents, _ = self.compose_server(tmp_path, monkeypatch)
        monkeypatch.setenv("KAFKA_TOPIC", "from-the-shell")
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n"
            "  main:\n    image: ${MAIN_IMAGE_NAME}\n    volumes: ['${CONTEXT_DIR}/seed.json:/app/seed.json:ro']\n"
            "  api:\n    image: org/api:1\n    expose: ['3080']\n    env_file: api.env\n"
            "  kafka:\n    image: org/kafka:1\n    environment: ['KAFKA_TOPIC=${KAFKA_TOPIC:-events}']\n"
        )
        (task.path / "environment" / "api.env").write_text("API_MODE=replay\n")
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        services = documents[0]["services"]
        assert services["main"]["image"] == IMAGE_CONFIGS["org/agent:1"]["image"]
        # The server resolves the task from its snapshot, so the bind mount points into that copy.
        (volume,) = services["main"]["volumes"]
        assert volume.endswith("/hello/environment/seed.json:/app/seed.json:ro")
        assert volume.startswith(server._snapshots.name) and not volume.startswith(str(task.path))
        assert services["api"]["environment"] == {"API_MODE": "replay"} and "env_file" not in services["api"]
        assert services["kafka"]["environment"] == {"KAFKA_TOPIC": "from-the-shell"}

    async def test_compose_task_seeds_a_group_and_leaves_task_context(self, tmp_path, monkeypatch):
        server, task, compose, documents, _ = self.compose_server(tmp_path, monkeypatch)

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
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
        # Everything above happened before the harness got its session.
        seed = server.server_client.seed_bodies()[0]
        assert seed["sandbox_access"]["connection"]["provider_config_ref"] == "sandbox"

    async def test_separate_verifier_collects_from_sidecars_and_stops_the_group(self, tmp_path, monkeypatch):
        server, task, compose, _, verifier = self.compose_server(tmp_path, monkeypatch)
        main = compose.services["main"]
        api = compose.services["api"] = MainWatchingSandbox(
            main=main, present={"/var/log/api": "dir"}, archives={"/var/log/api": archive_of({"api.log": "ok"})}
        )
        kafka = compose.services["kafka"] = MainWatchingSandbox(main=main)

        result = (await run_episode(server, task)).result

        assert result.reward == 1.0, result
        hook = next(c for c in kafka.execs if "dump-topics" in c["command"])
        assert hook["command"].startswith("sh -c ")
        # Harbor's phased collection: the agent's container was already stopped when the sidecar hook ran
        # and when the sidecar's artifact was looked up, while the sidecars themselves were still up.
        assert kafka.main_stopped_when_seen == [True] and api.main_stopped_when_seen == [True]
        assert any("/var/log/api" in c["command"] for c in api.execs)
        # The main container's own /logs/artifacts was looked up (before it was stopped: it answered).
        assert any(c["command"].startswith("if [ -d /logs/artifacts") for c in main.execs)
        assert compose.stopped and verifier.stopped
        # The group left the ownership table when it was stopped; the final cleanup had nothing to do.
        assert not server._sandboxes and not server._sessions

    async def test_a_separate_verifier_without_sidecar_work_stops_the_agent_side_once(self, tmp_path, monkeypatch):
        server, task, compose, _, verifier = self.compose_server(tmp_path, monkeypatch)
        toml = COMPOSE_TOML.replace('artifacts = [{ source = "/var/log/api", service = "api" }]\n', "").replace(
            '[[verifier.collect]]\ncommand = "dump-topics > /logs/artifacts/topics.txt"\nservice = "kafka"\n', ""
        )
        (task.path / "task.toml").write_text(toml)
        task = load_task(task.path)
        assert not task.config.artifacts and not task.config.verifier.collect
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        stops: list[str] = []
        compose.services["main"].stop = _record_stop(stops, "main", compose.services["main"])
        original_stop = compose.stop

        async def stop_group():
            stops.append("group")
            await original_stop()

        compose.stop = stop_group

        result = (await run_episode(server, task)).result

        assert result.reward == 1.0, result
        # Main is stopped on its own first, then the group once; nothing stops a second time.
        assert stops == ["main", "group"] and verifier.stopped

    async def test_a_check_started_inside_the_start_period_is_not_a_counted_failure(self, tmp_path, monkeypatch):
        toml = (
            TASK_TOML + '\n[environment.healthcheck]\ncommand = "test -f /tmp/ready"\nstart_period_sec = 0.05\n'
            "start_interval_sec = 0.01\ninterval_sec = 0.01\nretries = 1\n"
        )
        server, task, _, _ = make_server(tmp_path, monkeypatch, toml=toml)

        @dataclass
        class SlowFirstProbe(FakeSandbox):
            probes: int = 0

            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                self.probes += 1
                if self.probes == 1:
                    # The attempt begins inside the start period and ends after it, failing.
                    await asyncio.sleep(0.1)
                    return SandboxExecResult(stdout="", stderr="", return_code=1)
                return OK

        sandbox = SlowFirstProbe()
        # With retries = 1 a counted failure would raise here; Harbor decides the start period before the attempt.
        await server._wait_healthy(sandbox, task)
        assert sandbox.probes == 2
        # After the start period a failing attempt counts, and retries = 1 ends the wait.
        late = SlowFirstProbe(probes=5)
        late.spec_env = {}

        async def failing(command, **kwargs):
            return SandboxExecResult(stdout="", stderr="", return_code=1)

        late.exec = failing
        await asyncio.sleep(0.06)
        with pytest.raises(RuntimeError, match="Healthcheck failed 1 times"):
            await server._wait_healthy(late, task)

    async def test_compose_document_is_resolved_and_written_once_per_task(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (task.path.parent / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        task = load_task(task.path)
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        parses: list[str] = []
        resolve = server._compose_document

        def counting(task):
            parses.append(task.task_id)
            if parses == ["hello"]:
                raise RuntimeError("compose-images.json unreadable")
            return resolve(task)

        monkeypatch.setattr(server, "_compose_document", counting)
        # A failed resolution is not cached: the next episode tries again.
        with pytest.raises(RuntimeError, match="unreadable"):
            await server._compose_document_for(task)
        first, second = await asyncio.gather(server._compose_document_for(task), server._compose_document_for(task))
        third = await server._compose_document_for(task)
        # Concurrent first episodes shared one resolution, later ones reuse it: one parse, one file.
        assert parses == ["hello", "hello"] and first is second is third
        document, path = first
        assert set(document["services"]) == {"main", "api", "kafka"}
        assert yaml.safe_load(path.read_text()) == document and path.is_relative_to(server._artifacts_dir)

    async def test_gpu_tasks_route_to_the_gpu_provider(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "task.toml").write_text(TASK_TOML.replace("cpus = 1", 'cpus = 1\ngpus = 1\ngpu_types = ["H100"]'))
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        server.config.gpu_sandbox_provider = "sandbox_gpu"
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict",
            lambda: {"sandbox": {"opensandbox": {}}, "sandbox_gpu": {"opensandbox": {}}},
        )
        assert server._provider_ref(task) == "sandbox_gpu"
        spec = server._sandbox_spec(task, "/app")
        assert spec.resources.gpu == 1 and spec.resources.gpu_type is None
        server.config.request_gpu_type = True
        assert server._sandbox_spec(task, "/app").resources.gpu_type == "H100"
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        seed = server.server_client.seed_bodies()[0]
        assert seed["sandbox_access"]["connection"]["provider_config_ref"] == "sandbox_gpu"

    async def test_failed_healthcheck_is_a_retryable_seed_failure(self, tmp_path, monkeypatch):
        server, task, compose, _, _ = self.compose_server(tmp_path, monkeypatch)
        compose.services["main"].probes = -100  # never reaches 3 within 3 retries
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "Healthcheck failed" in response.failure.failure_reason
        assert compose.stopped and not server._sandboxes


def test_server_config_holds_only_deployment_settings():
    """A setting belongs here only if every dataset on the same deployment wants the same value.

    Task facts stay in task.toml, dataset settings in dataset.toml's [gym] table, image mirrors and
    setup commands in the sandbox provider. Adding a field means changing this set in the same PR.
    """
    own = set(HarborEnvironmentServerConfig.model_fields) - set(BaseEnvironmentServerConfig.model_fields)
    assert own == {
        "agent_server",
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
    source = inspect.getsource(HarborEnvironmentServerConfig).lower()
    assert not any(word in source for word in ("terminal", "tb4", "tb2", "swe", "nextjs", "hello-world"))


class TestSeedGuards:
    """Defects from the first cluster run: requests above limits, and a create the ready timeout did not bound."""

    def test_sandbox_spec_clamps_requests_above_the_tasks_limits(self, tmp_path, monkeypatch, caplog):
        toml = TASK_TOML.replace("storage_mb = 10240", "storage_mb = 4096")
        server, task, _, _ = make_server(
            tmp_path,
            monkeypatch,
            toml=toml,
            sandbox_provider_options={"resource_requests": {"cpu": 0.5, "memory_mib": 512, "disk_gib": 10}},
        )
        monkeypatch.setattr("environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {}})

        with caplog.at_level("WARNING"):
            spec = server._sandbox_spec(task, "/app")

        assert spec.resources.disk_gib == 4
        assert spec.provider_options["resource_requests"] == {"cpu": 0.5, "memory_mib": 512, "disk_gib": 4}
        assert "resource_requests.disk_gib=10 exceeds the disk_gib limit 4 task hello (agent)" in caplog.text
        # The configured options are not mutated, and a task within the requests gets them unchanged.
        assert server.config.sandbox_provider_options["resource_requests"]["disk_gib"] == 10
        caplog.clear()
        server2, task2, _, _ = make_server(
            tmp_path / "two", monkeypatch, sandbox_provider_options={"resource_requests": {"disk_gib": 10}}
        )
        spec2 = server2._sandbox_spec(task2, "/app")
        assert spec2.provider_options == {"resource_requests": {"disk_gib": 10}} and caplog.text == ""
        # "limits" and absent requests pass through untouched.
        server3, task3, _, _ = make_server(
            tmp_path / "three", monkeypatch, toml=toml, sandbox_provider_options={"resource_requests": "limits"}
        )
        assert server3._sandbox_spec(task3, "/app").provider_options == {"resource_requests": "limits"}

    async def test_cancelled_seed_leaves_the_stop_to_the_bounded_cleanup(self, tmp_path, monkeypatch):
        """The episode deadline fires during setup while the provider's stop hangs.

        The seed must not stop the sandbox inline (the deadline cancels only once, so that stop would run
        unbounded and keep the episode key active); the registered cleanup stops it under the cleanup
        timeout, the request returns, and the sandbox stays owned for the shutdown retry.
        """
        release = asyncio.Event()

        class HangingSandbox(FakeSandbox):
            stop_calls = 0

            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if command.startswith("mkdir"):
                    await asyncio.get_running_loop().create_future()
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

            async def stop(self):
                self.stop_calls += 1
                await release.wait()
                self.stopped = True

        sandbox = HangingSandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox, default_episode_timeout_seconds=0.1)
        server.config.cleanup_timeout_seconds = 0.2

        response = await asyncio.wait_for(run_episode(server, task), 5)

        assert response.failure is not None and "timed out" in response.failure.failure_reason
        assert not response.failure.terminal
        assert task.task_id not in server._active and not server._active
        # The cleanup attempted the stop once, gave up at its bound, and kept ownership for a later retry.
        assert sandbox.stop_calls == 1 and not sandbox.stopped
        assert list(server._sandboxes.values()) == [sandbox]
        assert server.server_client.calls == []
        release.set()
        await server.shutdown()
        assert sandbox.stopped and not server._sandboxes

    async def test_seed_gives_up_on_a_create_at_the_ready_timeout(self, tmp_path, monkeypatch):
        """The configured ready timeout is a seed deadline, whatever the provider's own retry budget."""
        server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox_ready_timeout_s=0.05)
        monkeypatch.setattr("environment_servers.harbor.app.SEED_CREATE_GRACE_S", 0.0)
        stopped = []

        class Hanging:
            async def stop(self):
                stopped.append(True)

        async def hang(task, workdir, *, episode_key):
            server._sandboxes[episode_key] = Hanging()
            await asyncio.get_running_loop().create_future()

        monkeypatch.setattr(server, "_create_sandbox", hang)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "not ready within 0.05s" in response.failure.failure_reason
        assert "sandbox_ready_timeout_s=0.05" in response.failure.failure_reason
        assert stopped == [True] and not server._sandboxes
        assert server.server_client.calls == []
        assert SEED_CREATE_GRACE_S > 0
        # The grace covers the provider's orphan reap, so the seed sees what the reap found, not a bare timeout.
        assert SEED_CREATE_GRACE_S >= sandbox_base.ORPHAN_REAP_TIMEOUT_S

    async def test_separate_verifier_create_is_bounded_by_the_ready_timeout(self, tmp_path, monkeypatch):
        """The same deadline bounds the verifier's create: the finished rollout is kept, masked, not held
        for the provider's retry budget."""
        agent = AgentSandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent, sandbox_ready_timeout_s=0.05)
        (task.path / "task.toml").write_text(SEPARATE_TOML)
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setattr("environment_servers.harbor.app.SEED_CREATE_GRACE_S", 0.0)
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        stopped = []

        class HangingStart:
            def __init__(self, provider_config):
                pass

            async def start(self, spec):
                await asyncio.get_running_loop().create_future()

            async def stop(self):
                stopped.append(True)

        monkeypatch.setattr("environment_servers.harbor.app.AsyncSandbox", HangingStart)
        started = asyncio.get_running_loop().time()
        response = await run_episode(server, task)
        assert asyncio.get_running_loop().time() - started < 5
        assert response.failure is None, response.failure
        result = response.result
        assert result.reward == 0.0 and result.mask_sample is True
        assert result.failure_kind == "provider_unavailable"
        assert "HarborSeedTimeoutError: the sandbox was not ready within 0.05s" in result.failure_reason
        # The agent's sandbox was stopped before the verifier's create; the verifier's entry is released too.
        assert agent.stopped and stopped == [True] and not server._sandboxes

    async def test_a_providers_own_timeout_inside_the_deadline_is_reported_as_is(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox_ready_timeout_s=100)

        async def provider_timeout(task, workdir, *, episode_key):
            raise TimeoutError("Timed out creating OpenSandbox sandbox after 100s")

        monkeypatch.setattr(server, "_create_sandbox", provider_timeout)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "Timed out creating OpenSandbox sandbox after 100s" in response.failure.failure_reason
