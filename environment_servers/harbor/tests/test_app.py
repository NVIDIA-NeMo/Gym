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
from dataclasses import dataclass, field, replace
from pathlib import Path

import orjson
import pytest
from aiohttp import ClientResponseError, RequestInfo
from fastapi.testclient import TestClient
from multidict import CIMultiDict, CIMultiDictProxy
from omegaconf import OmegaConf
from pydantic import ConfigDict
from pytest import MonkeyPatch
from yarl import URL

import nemo_gym.sandbox.providers.base as sandbox_base
from environment_servers.harbor.app import (
    ENTRYPOINT_DIR,
    ENTRYPOINT_FAILED_KIND,
    SEED_CREATE_GRACE_S,
    HarborEnvironmentServer,
    HarborEnvironmentServerConfig,
    HarborEpisodeResponse,
    _is_image_pull_failure,
    _verifier_image,
    entrypoint_to_run,
    image_healthcheck_settings,
    is_keepalive_entrypoint,
    overlay_shell_command,
    overlay_workdir_command,
    parse_reward_file,
    select_reward,
)
from nemo_gym import failure_kinds
from nemo_gym.base_environment_server import BaseEnvironmentServerConfig
from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxSpec
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from nemo_gym.single_agent_turn_types import SingleAgentTurnRequest
from nemo_gym.tasks.harbor import DIGEST_KEY, load_task
from nemo_gym.tasks.harbor.digest import context_digest
from nemo_gym.tasks.harbor.dockerfile import OverlayRun
from nemo_gym.tasks.harbor.models import HarborEnvironment
from nemo_gym.tasks.harbor.task import BUILT_IMAGES_FILE, IMAGE_CONFIGS_FILE, HarborTaskError


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
    uploaded_archives: list[bytes] = field(default_factory=list)
    stopped: bool = False
    spec_env: dict[str, str] = field(default_factory=dict)
    image_user: str = "root"
    dirs: dict[str, str] = field(default_factory=lambda: {"/": "root", "/tmp": "root", "/home": "root"})
    passwd: dict[str, str] = field(default_factory=lambda: {"0": "root", "1000": "agent"})

    default_user: str | None = None

    def set_default_user(self, user):
        self.default_user = user

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
            "user": None,
        }
        assert client.seed_bodies()[0]["task_id"] == {"taskset": "ds", "task_id": "hello"}
        # The task's own agent budget travels in the seed, so the harness can stop in time.
        assert client.seed_bodies()[0]["agent_timeout_sec"] == 300.0
        # The transcript reached /logs/agent/trajectory.json before test.sh ran, and a copy sits with the logs.
        trajectory_uploads = [
            i for i, (_, remote) in enumerate(sandbox.uploads) if remote == "/logs/agent/trajectory.json"
        ]
        assert len(trajectory_uploads) == 1
        trajectory = json.loads(sandbox.uploaded_archives[trajectory_uploads[0]])
        assert trajectory["schema_version"] == "ATIF-v1.7"
        assert [step["source"] for step in trajectory["steps"]] == ["user", "agent"]
        assert trajectory["steps"][1]["message"] == "echo hello > hello.txt"
        assert trajectory["agent"] == {
            "name": "agent",
            "version": trajectory["agent"]["version"],
            "model_name": "model",
        }
        assert (Path(result.verifier_logs_dir) / "trajectory.json").read_bytes() == sandbox.uploaded_archives[
            trajectory_uploads[0]
        ]
        assert result.verifier_network == "public"
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

    async def test_changed_folder_is_a_terminal_seed_failure(self, tmp_path, monkeypatch):
        server, task, _, created = make_server(tmp_path, monkeypatch)
        (task.path / "tests" / "test.sh").write_text("#!/bin/bash\necho 0 > /logs/verifier/reward.txt\n")

        response = await run_episode(server, task)

        assert response.failure.stage == "seed" and response.failure.terminal
        assert "changed since materialization" in response.failure.failure_reason
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
        # The sandbox lives for the task's own budget (sandbox start, harness setup, agent, verifier and its
        # grace), which is far below the run-wide deadline here, plus the cleanup budget and the slack.
        assert server._task_budget_s(task) == 900 + 300 + 300 + 120 + 30
        assert spec.ttl_s == server._task_budget_s(task) + 10 + server.config.sandbox_ttl_slack_s
        server.config.enforce_task_timeouts = False
        assert server._sandbox_spec(task, "/app").ttl_s == 21600 + 10 + server.config.sandbox_ttl_slack_s
        assert (spec.resources.cpu, spec.resources.memory_mib, spec.resources.disk_gib) == (1.0, 2048, 10)
        assert spec.metadata["sandbox-api"] == "osb"
        assert spec.metadata["harbor_task"] == "hello"
        assert spec.metadata["nemo_gym_environment_server"] == "harbor_environment_server"
        assert spec.user is None  # no USER in the Dockerfile and no [agent].user: the image's own default

    def test_sandbox_spec_user_is_the_dockerfile_user_for_the_task_image_only(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch, dockerfile="FROM ubuntu:24.04\nUSER agent\n")
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        assert task.user == "agent"
        assert server._sandbox_spec(task, "/app").user == "agent"
        # A root-overlay step still asks for root explicitly when the sandbox default is agent.
        assert server._sandbox_spec(task, "/app", image=task.image).user == "agent"
        # A separate verifier in a different image keeps that image's own default user.
        assert server._sandbox_spec(task, None, image="org/verifier:1", role="verifier").user is None


class TestDefaultUser:
    async def test_numeric_agent_user_without_an_overlay_is_resolved_by_name(self, tmp_path, monkeypatch, caplog):
        toml = TASK_TOML.replace("[agent]\ntimeout_sec = 300.0", '[agent]\ntimeout_sec = 300.0\nuser = "1000"')
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        (task.path / "task.toml").write_text(toml)
        task = load_task(task.path)
        assert task.user == "1000" and not task.overlay
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        # No overlay: the spec carries the uid as written, and the seed resolves it to a name once.
        assert server._sandbox_spec(task, "/app").user == "1000"
        assert (await run_episode(server, task)).failure is None
        assert [c["command"] for c in sandbox.execs if "getent" in c["command"]] == [
            "getent passwd 1000 | cut -d: -f1"
        ]
        assert sandbox.default_user == "agent"

        # A uid that names no user stays numeric, with a warning.
        sandbox.passwd.pop("1000")
        sandbox.default_user = None
        with caplog.at_level("WARNING"):
            assert (await run_episode(server, task, rollout="r2")).failure is None
        assert sandbox.default_user == "1000" and "USER 1000 names no user" in caplog.text

    async def test_named_user_is_set_as_the_default_after_seed(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, dockerfile="FROM ubuntu:24.04\nUSER agent\n")
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        assert (await run_episode(server, task)).failure is None
        assert sandbox.default_user == "agent"
        assert not any("getent" in c["command"] for c in sandbox.execs)


class TestSeedWorkdirAndResources:
    async def test_seed_creates_the_logs_folders_and_per_task_env(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        # Per-task environment comes from the dataset's own file, applied by the loader; the server sees plain tasks.
        (task.path.parent / "dataset.toml").write_text(
            '[gym.tasks."hello".environment]\nenv = { CIRCLE_NODE_TOTAL = "3" }\n[gym.tasks."other".environment]\nenv = { X = "1" }\n'
        )
        task = load_task(task.path)
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

    async def test_build_mode_task_seeds_from_its_recorded_image(self, tmp_path, monkeypatch):
        from nemo_gym.tasks.harbor.digest import context_digest
        from nemo_gym.tasks.harbor.task import BUILT_IMAGES_FILE

        server, task, _, _ = make_server(tmp_path, monkeypatch)
        # A Dockerfile that needs a build resolves to the image prepare recorded beside the task folders.
        (task.path / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nCOPY . /app\n")
        built = {context_digest(task.path / "environment"): {"image": "registry.example/ds-hello:abc"}}
        (task.path.parent / BUILT_IMAGES_FILE).write_text(json.dumps(built))
        task = load_task(task.path)
        assert task.image == "registry.example/ds-hello:abc"
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        seeded: list = []
        server.server_client.turn_hooks.append(lambda: seeded.append(server._sessions["r1"].task))

        response = await run_episode(server, task)

        # The server loads the task from its own snapshot, which must see the same record.
        assert response.failure is None, response.failure
        assert seeded[0].image == task.image

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
        request_info = RequestInfo(
            URL("http://agent/v1/agent_sessions/close"), "POST", CIMultiDictProxy(CIMultiDict())
        )
        server.server_client.fail_with["/v1/agent_sessions/close"] = ClientResponseError(
            request_info, (), status=500, message="close exploded"
        )

        response = await run_episode(server, task)

        assert response.failure.stage == "cleanup" and not response.failure.terminal
        assert response.failure.partial_response.id == "response"
        # Verification never ran: the harness may still have been working in the sandbox.
        assert not any("test.sh" in call["command"] for call in sandbox.execs)
        # The final unwind retried the close, then stopped the sandbox.
        assert server.server_client.paths().count("/v1/agent_sessions/close") == 2
        assert sandbox.stopped

    async def test_close_that_only_times_out_still_scores_the_turn(self, tmp_path, monkeypatch):
        """A slow harness teardown must not discard a finished turn; its observations are lost with the reply."""
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        server.server_client.fail_path = "/v1/agent_sessions/close"
        server.server_client.observations = {"source": "harness", "records": [], "gaps": []}

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.response.id == "response"
        assert response.result.ng_agent_observations is None
        assert any("test.sh" in call["command"] for call in sandbox.execs)
        # The final unwind retried the close (which then succeeded), then stopped the sandbox.
        assert server.server_client.paths().count("/v1/agent_sessions/close") == 2
        assert sandbox.stopped and not server._sandboxes

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

    async def test_run_lines_see_the_environment_directory_and_user_of_their_point_in_the_dockerfile(
        self, tmp_path, monkeypatch
    ):
        server, task, sandbox, created = self.overlay_server(tmp_path, monkeypatch, toml=HEALTHCHECK_TOML)
        assert len(task.overlay) == 3 and task.user == "agent"

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
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
        # Lines before `USER agent` run as root. The image's own default user is root, but the sandbox was
        # created with the Dockerfile's final USER (agent) as its exec default, so root is asked for explicitly.
        assert task.sandbox_user is None and task.user == "agent"
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
        assert prepare["user"] == "root"  # the sandbox's exec default is the Dockerfile's final USER (agent)
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
        assert lookup["command"] == "getent passwd 1000 | cut -d: -f1" and lookup["user"] == "root"
        # The name serves the directory's owner and both lines; the lookup happens once.
        assert prepare["command"].endswith("chown agent /home/agent/work)") and prepare["user"] == "root"
        assert [c["seen"]["user"] for c in (first, second)] == ["agent", "agent"]
        assert sum("getent" in c["command"] for c in sandbox.execs) == 1
        # The resolved name, not the uid, becomes the sandbox's default user after the overlay.
        assert sandbox.default_user == "agent"

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

    async def test_unknown_numeric_user_is_a_clear_task_error(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = self.overlay_server(
            tmp_path, monkeypatch, dockerfile="FROM img\nUSER 4242\nRUN id\n"
        )
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "RUN line 1/1 'id': USER 4242 names no user in img" in response.failure.failure_reason
        assert [c["command"] for c in sandbox.execs] == ["getent passwd 4242 | cut -d: -f1"] and sandbox.stopped

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

    def test_sandbox_ttls_cover_the_overlay_without_a_build_budget(self, tmp_path, monkeypatch):
        server, task, _, _ = self.overlay_server(tmp_path, monkeypatch)
        config, slack = task.config, server.config.sandbox_ttl_slack_s
        build, ready = config.environment.build_timeout_sec, server.config.sandbox_ready_timeout_s
        cleanup = server.config.cleanup_timeout_seconds
        assert build == 600 and task.overlay
        # The agent sandbox lives for the task's whole episode budget: one sandbox start, the overlay, harness
        # setup, the agent and the shared verifier with its grace; the overlay needs no budget of its own.
        budget = ready + build + server.config.harness_setup_allowance_s
        budget += config.agent.timeout_sec + config.verifier.timeout_sec + server.config.verifier_grace_s
        assert server._task_budget_s(task) == budget
        assert server._sandbox_spec(task, "/app").ttl_s == budget + cleanup + slack

        # A separate verifier in the task's own image: Harbor's effective verifier environment is a copy of
        # [environment] carrying its docker_image, a prebuilt image that resolves no overlay (the loader admits the
        # task-image shape with a declared image only), so no build budget is added and the verifier starts from the
        # task image with the agent's environment.
        toml = SEPARATE_IN_TASK_IMAGE_TOML.replace(
            "[environment]\n", '[environment]\ndocker_image = "debian:bookworm-slim"\n'
        )
        server, task, _, _ = self.overlay_server(tmp_path, monkeypatch, toml=toml)
        assert not task.overlay and task.verifier_image is None
        # No overlay, so no build budget anywhere; the separate verifier adds its own sandbox start.
        separate_budget = budget - build + ready
        assert server._task_budget_s(task) == separate_budget
        assert server._sandbox_spec(task, "/app").ttl_s == separate_budget + cleanup + slack
        specs = []

        class RecordingSandbox:
            def __init__(self, provider_config):
                pass

            async def start(self, spec):
                specs.append(spec)

        monkeypatch.setattr("environment_servers.harbor.app.AsyncSandbox", RecordingSandbox)
        asyncio.run(server._create_verifier_sandbox(task))
        (verifier_spec,) = specs
        assert verifier_spec.ttl_s == config.verifier.timeout_sec + slack
        assert verifier_spec.image == task.image == "debian:bookworm-slim"
        assert verifier_spec.env == server._sandbox_spec(task, "/app").env

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

    async def test_a_separate_verifier_with_no_image_and_no_tests_dockerfile_is_refused_as_harbor_refuses_it(
        self, tmp_path, monkeypatch
    ):
        """Harbor's verifier environment dir is always tests/: this shape fails its environment validation, so the
        loader refuses it (prepare leaves it out) and the server refuses it at seed. Nothing grades in the agent
        image, overlay or not."""
        agent = OverlaySandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent, dockerfile=OVERLAY_DOCKERFILE)
        (task.path / "task.toml").write_text(SEPARATE_IN_TASK_IMAGE_TOML)
        with pytest.raises(HarborTaskError, match="has no environment definition"):
            load_task(task.path)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "has no environment definition" in response.failure.failure_reason
        assert "Set [verifier.environment].docker_image or add tests/Dockerfile" in response.failure.failure_reason
        assert agent.execs == [] and not server._sandboxes

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
        assert run["command"] == "timeout --signal=KILL 120 bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1"
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
            test_result=SandboxExecResult(stdout=None, stderr=None, return_code=125, error_type="timeout")
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
        self.execs.pop()  # recorded again by the base fake
        return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

    async def download(self, remote_path, local_path):
        if remote_path.startswith("/tmp/.nemo-gym-download-"):
            Path(local_path).write_bytes(archive_of({"state.db": "db"}))
        else:
            Path(local_path).write_bytes(b'{"ok": true}')


class TestSeparateVerification:
    def prepared(self, tmp_path, monkeypatch, *, toml=SEPARATE_TOML, verifier=None):
        agent = AgentSandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent)
        (task.path / "task.toml").write_text(toml)
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        verifier = verifier or FakeSandbox()
        created = []

        async def create_verifier(task, *, episode_key=None):
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
        # /logs/artifacts (a directory) and the report (a file) were pulled from the agent sandbox...
        # Verifier logs live in a fresh per-verification folder; the row names it.
        artifacts = Path(result.verifier_logs_dir) / "artifacts"
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

    async def test_a_verifier_block_without_an_image_and_no_tests_dockerfile_is_refused(self, tmp_path, monkeypatch):
        """Harbor's verifier environment dir is always tests/: with no Dockerfile there and no docker_image in the
        block its environment validation fails, so the loader refuses the task (and the seed with it), naming the
        two remedies rather than a rebuild that could not help."""
        server, task, _, created = make_server(tmp_path, monkeypatch)
        toml = SEPARATE_TOML.replace('docker_image = "org/verifier:1"\n', "")
        (task.path / "task.toml").write_text(toml)
        with pytest.raises(HarborTaskError, match="has no environment definition") as info:
            load_task(task.path)
        assert "Set [verifier.environment].docker_image or add tests/Dockerfile" in str(info.value)
        assert "prepare" not in str(info.value)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "has no environment definition" in response.failure.failure_reason and created == []

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


class NetworkedStubProvider:
    """The least a provider needs for the Compose adapter's validation of a multi-service document."""

    name = "stub"

    def validate_networking(self) -> None:
        pass

    async def network_address(self, handle):
        return "192.0.2.1"

    async def set_hosts(self, handle, hosts):
        pass


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

    def test_compose_document_follows_the_requested_sandbox_platform(self, tmp_path, monkeypatch):
        """arm64 acceptance defect: a deployment asking for arm64 pods (``sandbox_provider_options.platform``)
        runs the arm64 images prepare built; the default deployment still runs amd64 ones and refuses the rest."""
        arm64 = {image: {**record, "architecture": "arm64"} for image, record in IMAGE_CONFIGS.items()}
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )

        def compose_task(root, records, **config):
            server, task, _, _ = make_server(root, monkeypatch, **config)
            (task.path / "task.toml").write_text(COMPOSE_TOML)
            (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
            (task.path.parent / "compose-images.json").write_text(json.dumps(records))
            return server, load_task(task.path)

        platform = {"platform": {"os": "linux", "arch": "arm64"}, "resource_requests": {"cpu": 0.5}}
        server, task = compose_task(tmp_path / "arm", arm64, sandbox_provider_options=platform)
        document = server._compose_document(task)
        assert set(document["services"]) == {"main", "api", "kafka"}
        assert document["services"]["kafka"]["image"] == arm64["org/kafka:1"]["image"]
        # The requested platform is the one the options still carry for the sandboxes themselves.
        assert server._sandbox_spec(task, "/app").provider_options["platform"] == platform["platform"]

        server, task = compose_task(tmp_path / "amd", arm64)
        with pytest.raises(ValueError, match="Service 'main': the recorded image .* is linux/arm64, not linux/amd64"):
            server._compose_document(task)
        server, task = compose_task(tmp_path / "mixed", IMAGE_CONFIGS, sandbox_provider_options=platform)
        with pytest.raises(ValueError, match="Service 'main': the recorded image .* is linux/amd64, not linux/arm64"):
            server._compose_document(task)

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

    MAIN_BUILD_SHAPES = {
        "args-and-target": "    build:\n      context: ${CONTEXT_DIR}\n      target: prod\n      args:\n        PYTHON_VERSION: '3.12'\n",
        "dockerfile_inline": "    build:\n      context: ${CONTEXT_DIR}\n      dockerfile_inline: |\n        FROM python:3.12\n        RUN pip install x\n",
    }

    @pytest.mark.parametrize("main_build", list(MAIN_BUILD_SHAPES.values()), ids=list(MAIN_BUILD_SHAPES))
    def test_a_built_tasks_main_build_section_is_dropped_at_seed(self, tmp_path, monkeypatch, main_build):
        """Prepare builds ``main`` from the task's Compose file (its build args, target or dockerfile_inline apply
        there) and every sidecar with a ``build:`` section (G2), and records the images by service; at seed the
        recorded images stand in, so the ``build:`` and ``pull_policy`` keys that only describe those builds are
        dropped before the Compose adapter, which starts images only, and the sidecar's entrypoint, command and
        ports come from its recorded configuration like any pulled image's."""
        from nemo_gym.sandbox.adapters.docker_compose import AsyncSandboxCompose
        from nemo_gym.tasks.harbor.digest import context_digest
        from nemo_gym.tasks.harbor.task import BUILT_IMAGES_FILE, built_service_images

        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nCOPY . /app\n")
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    pull_policy: build\n"
            + main_build
            + "    depends_on:\n      world:\n        condition: service_healthy\n"
            + "  api:\n    image: org/api:1\n"
            + "  world:\n    build:\n      context: .\n      dockerfile: world/Dockerfile\n    pull_policy: build\n"
            + "    expose: ['8972']\n    healthcheck:\n      test: ['CMD', 'true']\n      interval: 2s\n"
        )
        built = "registry.example/ds/hb__" + "0" * 32 + "@sha256:" + "d" * 64
        world = "registry.example/ds/hb__" + "0" * 32 + "-world@sha256:" + "e" * 64
        (task.path.parent / BUILT_IMAGES_FILE).write_text(
            json.dumps(
                {
                    context_digest(task.path / "environment"): {
                        "image": built,
                        "services": {"main": {"image": built}, "world": {"image": world}},
                    }
                }
            )
        )
        records = json.loads((task.path.parent / IMAGE_CONFIGS_FILE).read_text())
        records[built] = {"os": "linux", "architecture": "amd64", "image": built, "config": {"WorkingDir": "/app"}}
        records[world] = {
            "os": "linux",
            "architecture": "amd64",
            "image": world,
            "config": {"Cmd": ["python3", "/opt/world/server.py"], "ExposedPorts": {"8972/tcp": {}}},
        }
        records["org/api:1"] = IMAGE_CONFIGS["org/api:1"]
        (task.path.parent / IMAGE_CONFIGS_FILE).write_text(json.dumps(records))
        task = load_task(task.path)
        assert task.built_image == built == task.image and task.workdir == "/app"
        assert built_service_images(task.path) == {"main": built, "world": world}
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )

        document = server._compose_document(task)

        main = document["services"]["main"]
        assert main["image"] == built and main["working_dir"] == "/app"
        assert "build" not in main and "pull_policy" not in main
        assert document["services"]["api"]["image"] == IMAGE_CONFIGS["org/api:1"]["image"]
        sidecar = document["services"]["world"]
        assert sidecar["image"] == world and "build" not in sidecar and "pull_policy" not in sidecar
        assert sidecar["command"] == ["python3", "/opt/world/server.py"] and sidecar["expose"] == ["8972", "8972/tcp"]
        assert main["depends_on"] == {"world": {"condition": "service_healthy"}}
        # The adapter's own validation accepts the document: the fields that failed every episode are gone.
        group = AsyncSandboxCompose(NetworkedStubProvider(), "compose.yaml")
        group.document = document
        assert set(group._validate()) == {"main", "api", "world"}

    def test_seed_reads_the_build_record_once_at_load_and_never_hashes_the_environment(self, tmp_path, monkeypatch):
        """``load_task`` (once per task, in a thread) hashes ``environment/`` once and carries Harbor's context id
        and the build record on the task; ``_compose_document``, which runs per episode, hashes nothing, whatever
        the folder's size, and still substitutes the recorded images and defines ``COMPOSE_PROJECT_NAME``."""
        import nemo_gym.tasks.harbor.digest as digest_module
        from nemo_gym.tasks.harbor.digest import context_digest
        from nemo_gym.tasks.harbor.task import BUILT_IMAGES_FILE

        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nCOPY . /app\n")
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n"
            "  world:\n    build: ./world\n    environment: ['PROJECT=${COMPOSE_PROJECT_NAME}']\n"
        )
        (task.path / "environment" / "world").mkdir()
        (task.path / "environment" / "world" / "Dockerfile").write_text("FROM python:3.12\n")
        expected = context_digest(task.path / "environment")
        built = f"registry.example/ds/hb__{expected}@sha256:" + "d" * 64
        world = f"registry.example/ds/hb__{expected}-world@sha256:" + "e" * 64
        record = {"image": built, "services": {"main": {"image": built}, "world": {"image": world}}}
        (task.path.parent / BUILT_IMAGES_FILE).write_text(json.dumps({expected: record}))
        records = json.loads((task.path.parent / IMAGE_CONFIGS_FILE).read_text())
        records[built] = {"os": "linux", "architecture": "amd64", "image": built, "config": {}}
        records[world] = {"os": "linux", "architecture": "amd64", "image": world, "config": {"Cmd": ["serve"]}}
        (task.path.parent / IMAGE_CONFIGS_FILE).write_text(json.dumps(records))
        task = load_task(task.path)
        assert task.context_id == expected and task.built_images == {"main": built, "world": world}
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        # Every caller of context_digest reaches this function (patching task.context_digest alone would miss
        # compose_variables' COMPOSE_PROJECT_NAME); the counter is proven live before the seed path is measured.
        hashed: list[Path] = []
        real = digest_module.environment_content_hash
        monkeypatch.setattr(
            digest_module, "environment_content_hash", lambda path: (hashed.append(path), real(path))[1]
        )
        assert context_digest(task.path / "environment") == expected and len(hashed) == 1
        hashed.clear()

        for _ in range(2):
            document = server._compose_document(task)
            assert document["services"]["main"]["image"] == built
            sidecar = document["services"]["world"]
            assert sidecar["image"] == world and "build" not in sidecar and sidecar["command"] == ["serve"]
            assert sidecar["environment"] == {"PROJECT": f"hb__{expected}"}
        assert hashed == []

    @pytest.mark.parametrize(
        ("dockerfile", "build"),
        [
            ("FROM ubuntu:24.04\nWORKDIR /app", "      context: ${CONTEXT_DIR}\n      pull: true\n"),
            ("FROM ubuntu:24.04\nRUN echo hi > /hi\n", "      dockerfile: Dockerfile\n"),
        ],
    )
    def test_a_restated_main_build_is_dropped_in_pull_and_overlay_mode(self, tmp_path, monkeypatch, dockerfile, build):
        """Harbor's own template already builds ``main`` from ``environment/Dockerfile`` with ``context: ${CONTEXT_DIR}``,
        so a task file that only restates that (orca-bench, terminal_bench_aai) changes nothing there. Gym runs the
        Dockerfile without a build in pull and overlay mode, so the section is dropped and the document starts."""
        from nemo_gym.sandbox.adapters.docker_compose import AsyncSandboxCompose
        from nemo_gym.tasks.harbor.cli import unsupported_compose_fields_error

        server, task, _, _ = make_server(tmp_path, monkeypatch, dockerfile=dockerfile)
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    build:\n" + build
        )
        task = load_task(task.path)
        assert task.built_image is None and task.image == "ubuntu:24.04"
        assert bool(task.overlay) == ("RUN" in dockerfile)
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        document = server._compose_document(task)
        assert "build" not in document["services"]["main"]
        group = AsyncSandboxCompose(NetworkedStubProvider(), "compose.yaml")
        group.document = document
        assert group._validate() == ["main"]
        assert unsupported_compose_fields_error(task) is None

    def test_a_main_build_key_gym_cannot_apply_without_a_build_is_a_task_error(self, tmp_path, monkeypatch):
        """A ``target`` (or ``args``, ``dockerfile_inline``, another ``dockerfile`` or ``context``) would change the image
        Harbor builds; without a build Gym cannot honour it, so the task is an error naming the key, as prepare already
        reported it into ``skipped.json``, instead of starting on the wrong image."""
        from nemo_gym.tasks.harbor.cli import unsupported_compose_fields_error
        from nemo_gym.tasks.harbor.task import HarborTaskError

        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    build:\n      context: ${CONTEXT_DIR}\n"
            "      target: prod\n"
        )
        task = load_task(task.path)
        assert task.built_image is None
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        expected = "main.build.target cannot be applied in pull or overlay mode (Gym does not build this Dockerfile)"
        with pytest.raises(HarborTaskError, match=re.escape(expected)):
            server._compose_document(task)
        assert str(unsupported_compose_fields_error(task)) == f"docker-compose.yaml: {expected}"

    def test_pull_policy_on_any_service_is_dropped_so_the_document_starts(self, tmp_path, monkeypatch):
        """``pull_policy`` says when ``docker compose`` pulls an image before starting it; the sandbox has no pull
        step to configure, so the key is dropped from every service in every image mode (here pull mode, where
        ``main`` keeps everything else) instead of the adapter refusing it. Prepare's check sees the same document."""
        from nemo_gym.sandbox.adapters.docker_compose import AsyncSandboxCompose
        from nemo_gym.tasks.harbor.cli import unsupported_compose_fields_error

        server, task, _, _ = make_server(tmp_path, monkeypatch)
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    pull_policy: missing\n"
            "  api:\n    image: org/api:1\n    pull_policy: always\n    expose: ['8080']\n"
        )
        records = json.loads((task.path.parent / IMAGE_CONFIGS_FILE).read_text())
        records["org/api:1"] = IMAGE_CONFIGS["org/api:1"]
        (task.path.parent / IMAGE_CONFIGS_FILE).write_text(json.dumps(records))
        task = load_task(task.path)
        assert task.built_image is None
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )

        document = server._compose_document(task)

        assert all("pull_policy" not in service for service in document["services"].values())
        assert document["services"]["api"]["expose"] == ["8080"]
        group = AsyncSandboxCompose(NetworkedStubProvider(), "compose.yaml")
        group.document = document
        assert set(group._validate()) == {"main", "api"}
        assert unsupported_compose_fields_error(task) is None

    @staticmethod
    def capture_compose_specs(monkeypatch) -> dict:
        captured = {}

        class FakeComposeGroup:
            def __init__(self, provider_config, path, *, service_specs, timeout_s):
                captured["specs"] = service_specs

            async def start(self):
                return None

        monkeypatch.setattr("environment_servers.harbor.app.AsyncSandboxCompose", FakeComposeGroup)
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        return captured

    def test_main_env_layers_image_env_under_the_compose_file_under_the_host_resolved_task_env(
        self, tmp_path, monkeypatch
    ):
        """Harbor's order for ``main``: image ENV (here the pull-mode Dockerfile's ``ENV``, which the loader folds into
        the task) < the task's Compose file < the host-resolved ``[environment].env`` (its env override file). A key
        the file sets beats the Dockerfile's, a ``${FOO:-}`` that interpolated to empty at prepare does not hide the
        host value, and a sidecar keeps its file's environment."""
        from nemo_gym.sandbox.adapters.docker_compose import AsyncSandboxCompose

        server, task, _, _ = make_server(
            tmp_path,
            monkeypatch,
            dockerfile="FROM ubuntu:24.04\nENV MODE=image\nENV ONLY_IMAGE=i\n",
            toml=TASK_TOML + '\n[environment.env]\nFOO = "${FOO:-}"\n',
        )
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n"
            "    environment:\n      MODE: compose\n      FOO: ${FOO:-}\n      ONLY_COMPOSE: c\n"
            "  api:\n    image: org/api:1\n    environment:\n      MODE: sidecar\n    expose: ['8080']\n"
        )
        write_image_config(task.path.parent, "org/api:1", **IMAGE_CONFIGS["org/api:1"]["config"])
        task = load_task(task.path)
        assert task.built_image is None and not task.overlay
        # The base image's own ENV stays in the image; the loader folds in the Dockerfile's ENV lines only.
        assert task.env == {"MODE": "image", "ONLY_IMAGE": "i", "FOO": "${FOO:-}"}
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setenv("FOO", "host")
        captured = self.capture_compose_specs(monkeypatch)

        asyncio.run(server._create_compose(task, "r1"))

        specs = captured["specs"]
        expected = {"MODE": "compose", "ONLY_IMAGE": "i", "FOO": "host", "ONLY_COMPOSE": "c"}
        assert specs["main"].env == expected and specs["api"].env == {}
        # The adapter's own plan for the same specs and document gives the same three layers.
        group = AsyncSandboxCompose(NetworkedStubProvider(), "compose.yaml", service_specs=specs)
        group.document = server._compose_document(task)
        group._validate()
        asyncio.run(group._prepare())
        assert group._plans["main"]["spec"].env == expected
        assert group._plans["api"]["spec"].env == {"MODE": "sidecar"}

    def test_a_built_images_recorded_env_sits_under_the_compose_file_too(self, tmp_path, monkeypatch):
        """Build mode: the recorded built image's whole ``Env`` is the image layer, under the file and under the
        host-resolved ``[environment].env``; a key only the image names survives."""
        from nemo_gym.tasks.harbor.digest import context_digest
        from nemo_gym.tasks.harbor.task import BUILT_IMAGES_FILE

        server, task, _, _ = make_server(
            tmp_path, monkeypatch, toml=TASK_TOML + '\n[environment.env]\nTOK = "${TOK:-fromtoml}"\n'
        )
        (task.path / "environment" / "Dockerfile").write_text(
            "FROM ubuntu:24.04\nCOPY . /app\nENV ISSUE_ID=from_dockerfile\n"
        )
        (task.path / "environment" / "docker-compose.yaml").write_text(
            'services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    build: {context: "${CONTEXT_DIR}"}\n'
            "    environment:\n      ISSUE_ID: from_compose\n      LC_ALL: C.UTF-8\n      TOK: ${TOK:-}\n"
        )
        digest = context_digest(task.path / "environment")
        image = f"registry.example/ds/hb__{digest}@sha256:" + "d" * 64
        (task.path.parent / BUILT_IMAGES_FILE).write_text(
            json.dumps({digest: {"image": image, "tasks": ["hello"], "tool": "docker", "pushed": True}})
        )
        write_image_config(
            task.path.parent, image, Env=["ISSUE_ID=from_image", "LC_ALL=POSIX", "PATH=/usr/bin"], Cmd=["bash"]
        )
        task = load_task(task.path)
        assert task.built_image == image
        assert task.env == {"ISSUE_ID": "from_image", "LC_ALL": "POSIX", "PATH": "/usr/bin", "TOK": "${TOK:-fromtoml}"}
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setenv("TOK", "hosttok")
        captured = self.capture_compose_specs(monkeypatch)

        asyncio.run(server._create_compose(task, "r1"))

        assert captured["specs"]["main"].env == {
            "ISSUE_ID": "from_compose",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin",
            "TOK": "hosttok",
        }

    def test_sidecar_compose_environment_takes_the_host_resolved_task_env_at_seed(self, tmp_path, monkeypatch):
        """Harbor interpolates the task's Compose file for every service against the host-resolved
        ``[environment].env``; prepare could only use the dataset-time default. At seed the sidecars' ``environment``
        is read again with the host values (tau3 / hil-bench shape); a literal entry and the structure are untouched,
        and without the variable the default still applies."""
        from nemo_gym.tasks.harbor.image_configs import load_compose_document

        server, task, _, _ = make_server(
            tmp_path,
            monkeypatch,
            toml=TASK_TOML
            + '\n[environment.env]\nTAU2_USER_MODEL = "${TAU2_USER_MODEL:-gpt-5.2}"\nOPENAI_API_KEY = "${OPENAI_API_KEY:-}"\n',
        )
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n"
            "  user-sim:\n    image: org/api:1\n    expose: ['8000']\n    environment:\n"
            "      - TAU2_USER_MODEL=${TAU2_USER_MODEL:-gpt-5.2}\n      - OPENAI_API_KEY=${OPENAI_API_KEY:-}\n"
            "      - LITERAL=x\n"
        )
        write_image_config(task.path.parent, "org/api:1", **IMAGE_CONFIGS["org/api:1"]["config"])
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        prepared = load_compose_document(task)["services"]["user-sim"]["environment"]
        assert prepared == {"TAU2_USER_MODEL": "gpt-5.2", "OPENAI_API_KEY": "", "LITERAL": "x"}

        monkeypatch.setenv("TAU2_USER_MODEL", "some-local-model")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-host")
        document = server._compose_document(task)
        assert document["services"]["user-sim"]["environment"] == {
            "TAU2_USER_MODEL": "some-local-model",
            "OPENAI_API_KEY": "sk-host",
            "LITERAL": "x",
        }
        assert document["services"]["user-sim"]["image"] == "org/api:1"
        assert "TAU2_USER_MODEL" not in document["services"]["main"].get("environment", {})

        monkeypatch.delenv("TAU2_USER_MODEL")
        monkeypatch.delenv("OPENAI_API_KEY")
        document = server._compose_document(task)
        assert document["services"]["user-sim"]["environment"] == prepared

    def test_overlay_compose_task_starts_without_the_task_user_and_switches_after_the_overlay(
        self, tmp_path, monkeypatch
    ):
        """The compose adapter's bootstrap execs run through the facade default before any overlay RUN line has
        created the task's user, so an overlay task starts with no default user and gets it afterwards."""
        import environment_servers.harbor.app as app_module
        from nemo_gym.sandbox.providers.base import SandboxHandle

        class DockerLikeProvider:
            """exec behaves like `docker exec --user <name>`: it fails until the user exists in the container."""

            name = "fake"

            def __init__(self):
                self.created, self.execs, self.users = [], [], {"root"}

            def __contains__(self, key):  # `_compose_document` asks `"opensandbox" in provider`
                return False

            async def create(self, spec):
                self.created.append(spec)
                return SandboxHandle(str(len(self.created)), self.name, None)

            async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
                self.execs.append({"user": user, "command": command})
                if user not in (None, "root", 0) and str(user) not in self.users:
                    return SandboxExecResult("", f"unable to find user {user}", 126, error_type="sandbox")
                if "useradd" in command:
                    self.users.add("agent")
                return SandboxExecResult("", "", 0)

            async def upload_file(self, handle, source_path, target_path):
                pass

            def validate_networking(self):
                pass

            async def network_address(self, handle):
                return "192.0.2.1"

            async def set_hosts(self, handle, hosts):
                pass

            async def close(self, handle):
                pass

            async def aclose(self):
                pass

        server, task, _, _ = make_server(
            tmp_path, monkeypatch, dockerfile="FROM ubuntu:24.04\nRUN useradd -m agent\nUSER agent\n"
        )
        (task.path / "task.toml").write_text(TASK_TOML)
        (task.path / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    command: ['sleep', 'infinity']\n"
        )
        task = load_task(task.path)
        assert task.user == "agent" and task.overlay and task.needs_compose
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        provider = DockerLikeProvider()
        monkeypatch.setattr(app_module, "get_global_config_dict", lambda: {"sandbox": {"docker": {}}})
        monkeypatch.setattr(app_module, "resolve_provider_config", lambda ref, cfg: provider)

        async def run():
            compose = await server._create_compose(task, "r1")
            main = compose.services["main"]
            bootstrap_users = {e["user"] for e in provider.execs}
            assert provider.created[0].user is None and main.default_user is None
            assert "agent" not in bootstrap_users, provider.execs
            names = await server._apply_overlay(main, task)
            await server._set_default_user(main, task, names)
            assert any("useradd" in e["command"] for e in provider.execs)
            assert main.default_user == "agent"
            # From here every exec that names no user runs as the task's user.
            await main.exec("id")
            assert provider.execs[-1]["user"] == "agent"

        asyncio.run(run())

    def test_sidecar_specs_do_not_inherit_the_task_user(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch, dockerfile="FROM ubuntu:24.04\nUSER agent\n")
        (task.path / "task.toml").write_text(COMPOSE_TOML)
        (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (task.path.parent / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        task = load_task(task.path)
        assert task.user == "agent"
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        captured = {}

        class FakeComposeGroup:
            def __init__(self, provider_config, path, *, service_specs, timeout_s):
                captured["specs"] = service_specs

            async def start(self):
                return None

        monkeypatch.setattr("environment_servers.harbor.app.AsyncSandboxCompose", FakeComposeGroup)
        server.config.artifacts_dir = tmp_path / "artifacts"
        asyncio.run(server._create_compose(task, "r1"))
        specs = captured["specs"]
        assert specs["main"].user == "agent"
        assert {name: spec.user for name, spec in specs.items() if name != "main"} == {"api": None, "kafka": None}
        # Nor main's working directory: a sidecar's image decides its own (the adapter applies the recorded one).
        assert specs["main"].workdir == task.workdir
        assert {name: spec.workdir for name, spec in specs.items() if name != "main"} == {"api": None, "kafka": None}

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
        kafka, api = compose.services["kafka"], compose.services["api"]
        api.__class__ = AgentSandbox  # gives it `present`
        api.present = {"/var/log/api": "dir"}

        result = (await run_episode(server, task)).result

        assert result.reward == 1.0, result
        hook = next(c for c in kafka.execs if "dump-topics" in c["command"])
        assert hook["command"].startswith("sh -c ")
        assert any("/var/log/api" in c["command"] for c in api.execs)
        assert compose.stopped and verifier.stopped
        # The group left the ownership table when it was stopped; the final cleanup had nothing to do.
        assert not server._sandboxes and not server._sessions

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
        "secrets",
        "enforce_task_timeouts",
        "harness_setup_allowance_s",
        "image_pull_retries",
        "separate_verifier_for_artifacts",
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


@dataclass
class VerifierSandbox(FakeSandbox):
    """A separate verifier's sandbox whose image holds ``/tests/test.sh`` unless told otherwise."""

    owns_tests: bool = True

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if command.startswith("if [ -d "):
            self.execs.append({"command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s, "user": user})
            path = command.split("if [ -d ")[1].split(" ]")[0].strip("'")
            kind = "file" if path == "/tests/test.sh" and self.owns_tests else "none"
            return SandboxExecResult(stdout=f"{kind}\n", stderr="", return_code=0)
        return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)


class TestBuiltVerifierImage:
    """D1: a separate verifier grades in the image prepare built from ``tests/Dockerfile``."""

    BUILT = "registry.example/r/hb__0123456789abcdef0123456789abcdef-verifier@sha256:feed"

    def built(self, tmp_path, monkeypatch, *, toml=SEPARATE_IN_TASK_IMAGE_TOML, verifier=None):
        """A separate-mode task with an overlay-mode agent Dockerfile and a ``tests/Dockerfile``, built and recorded;
        ``_create_verifier_sandbox`` is replaced by one handing out ``verifier``."""
        agent = OverlaySandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent, dockerfile=OVERLAY_DOCKERFILE)
        (task.path / "task.toml").write_text(toml + '\n[environment.env]\nAPP_MODE = "${HARBOR_TEST_MODE:-grade}"\n')
        (task.path / "tests" / "Dockerfile").write_text(VERIFIER_TESTS_DOCKERFILE)
        record_verifier_build(task.path, self.BUILT)
        task = load_task(task.path)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        verifier = verifier or VerifierSandbox()
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )

        async def create_verifier(task, *, episode_key=None):
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        return server, task, agent, verifier

    def test_the_loader_exposes_the_recorded_image_and_the_verifier_gets_no_overlay(self, tmp_path, monkeypatch):
        server, task, _, _ = self.built(tmp_path, monkeypatch)
        assert task.verifier_image == self.BUILT and _verifier_image(task) == self.BUILT
        # The agent image is overlaid (its Dockerfile has RUN lines); the verifier image was built, so it is not.
        assert task.overlay and task.image == "debian:bookworm-slim"

    async def test_the_verifier_sandbox_starts_from_the_built_image_under_the_environment_block(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HARBOR_TEST_MODE", "strict")
        server, task, _, _ = self.built(tmp_path, monkeypatch)
        verifier = VerifierSandbox()
        specs = []

        async def start(spec):
            specs.append(spec)

        monkeypatch.setattr(verifier, "start", start, raising=False)
        monkeypatch.setattr("environment_servers.harbor.app.resolve_provider_config", lambda *args: {})
        monkeypatch.setattr("environment_servers.harbor.app.AsyncSandbox", lambda config: verifier)
        monkeypatch.delattr(server, "_create_verifier_sandbox")  # the real one, not the test double
        assert await server._create_verifier_sandbox(task) is verifier
        (spec,) = specs
        # Harbor's effective verifier environment is a copy of [environment]: its env (resolved) and resources
        # reach the verifier sandbox, the agent image's Dockerfile ENV and WORKDIR do not, and the image keeps its
        # own default user and working directory.
        assert spec.image == self.BUILT and spec.metadata["harbor_role"] == "verifier"
        assert spec.env == {"APP_MODE": "strict"} and "PATH" not in spec.env
        assert spec.workdir is None and spec.user is None
        assert (spec.resources.cpu, spec.resources.memory_mib) == (1, 2048)
        assert spec.ttl_s == task.config.verifier.timeout_sec + server.config.sandbox_ttl_slack_s

    async def test_the_verifier_runs_test_sh_in_the_built_image_without_uploading_tests(self, tmp_path, monkeypatch):
        server, task, agent, verifier = self.built(tmp_path, monkeypatch)

        result = (await run_episode(server, task)).result

        assert result.reward == 1.0 and result.verifier_mode == "separate", result
        assert result.mask_sample is False and result.failure_kind is None
        # The agent's sandbox took the overlay; the verifier's did not, and no tests archive went up: the image
        # owns /tests, as Harbor's skip_tests_upload assumes. Only the ATIF trajectory the verifier reads goes up.
        assert any("apt-get" in c["command"] for c in agent.execs)
        assert not any("apt-get" in c["command"] for c in verifier.execs)
        assert [remote for _, remote in verifier.uploads] == ["/logs/agent/trajectory.json"]
        run = next(c for c in verifier.execs if "test.sh" in c["command"] and "timeout" in c["command"])
        assert "bash /tests/test.sh" in run["command"]
        assert agent.stopped and verifier.stopped

    async def test_a_built_image_without_test_sh_gets_the_upload_and_one_warning(self, tmp_path, monkeypatch, caplog):
        server, task, _, verifier = self.built(tmp_path, monkeypatch, verifier=VerifierSandbox(owns_tests=False))
        with caplog.at_level("WARNING", logger="environment_servers.harbor.app"):
            assert (await run_episode(server, task)).result.reward == 1.0
            assert (await run_episode(server, task, rollout="r2")).result.reward == 1.0
        # tests/ was uploaded both times; the image was named once.
        archives = [remote for _, remote in verifier.uploads if remote.endswith(".tar.gz")]
        assert len(archives) == 2
        warnings = [r for r in caplog.records if "has no /tests/test.sh" in r.getMessage()]
        assert len(warnings) == 1 and self.BUILT in warnings[0].getMessage()

    async def test_a_verifier_environment_block_without_an_image_seeds_the_built_image_with_its_settings(
        self, tmp_path, monkeypatch
    ):
        toml = (
            TASK_TOML
            + '\n[verifier.environment]\ncpus = 3\nworkdir = "/grade"\n\n[verifier.environment.env]\nK = "v"\n'
        )
        server, task, _, verifier = self.built(tmp_path, monkeypatch, toml=toml)
        assert task.verifier_image == self.BUILT
        spec = server._sandbox_spec(
            task,
            task.config.verifier.environment.workdir,
            environment=task.config.verifier.environment,
            image=self.BUILT,
        )
        assert spec.image == self.BUILT and spec.env == {"K": "v"} and spec.workdir == "/grade"
        assert spec.resources.cpu == 3
        assert (await run_episode(server, task)).result.reward == 1.0
        assert [remote for _, remote in verifier.uploads] == ["/logs/agent/trajectory.json"]

    @pytest.mark.parametrize("shape", ["no_block", "block_without_image"])
    async def test_without_a_record_the_task_is_refused_by_the_loader_and_at_seed(self, tmp_path, monkeypatch, shape):
        """Harbor grades in the image built from tests/Dockerfile; without a record nothing stands in for it: not the
        agent image (where the tests' tooling is usually missing, so every episode would score 0 with no error) and
        not a refusal that only comes at seed. The loader refuses the task like an unrecorded environment/ build."""
        toml = (
            SEPARATE_IN_TASK_IMAGE_TOML if shape == "no_block" else TASK_TOML + "\n[verifier.environment]\ncpus = 2\n"
        )
        agent = OverlaySandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent, dockerfile=OVERLAY_DOCKERFILE)
        (task.path / "task.toml").write_text(toml)
        (task.path / "tests" / "Dockerfile").write_text(VERIFIER_TESTS_DOCKERFILE)
        with pytest.raises(HarborTaskError) as info:
            load_task(task.path)
        reason = str(info.value)
        assert reason.startswith(f"{task.path / 'tests' / 'Dockerfile'} is the separate verifier's image")
        assert str(task.path.parent / BUILT_IMAGES_FILE) in reason and "`builds:` block" in reason
        assert "declare [verifier.environment].docker_image" in reason
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "is the separate verifier's image" in response.failure.failure_reason
        assert agent.execs == [] and not server._sandboxes

    async def test_the_seed_still_checks_the_verifier_image_invariant(self, tmp_path, monkeypatch):
        """A task with a [verifier.environment] without docker_image and no verifier image cannot come from the
        loader; the seed refuses such an object all the same, naming the record file and the remedies."""
        server, task, agent, _ = self.built(
            tmp_path, monkeypatch, toml=TASK_TOML + "\n[verifier.environment]\ncpus = 2\n"
        )
        broken = replace(task, verifier_image=None)
        assert _verifier_image(task) == self.BUILT and _verifier_image(broken) is None

        async def resolve(task_id, task_data):
            return broken

        monkeypatch.setattr(server, "_resolve_task", resolve)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and response.failure.terminal
        assert "has no image for its separate verifier" in response.failure.failure_reason
        assert "built-images.json" in response.failure.failure_reason
        assert "tests/Dockerfile" in response.failure.failure_reason
        assert agent.execs == []


BUILT_IMAGE = "registry.example/ds-hello:abc"
ENTRYPOINT_CONFIG = {
    "Env": [f"PATH={IMAGE_PATH}"],
    "Entrypoint": ["/entrypoint.sh"],
    "User": "app",
    "WorkingDir": "/app",
}


@dataclass
class EntrypointSandbox(FakeSandbox):
    """A sandbox whose image entrypoint the seed starts: answers the markers the launcher polls.

    The launch exec (``sh DIR/run.sh ...``) waits on ``release`` (the steady state: an entrypoint that
    never returns) unless ``launch_result`` is set, in which case it returns at once (the provider could
    not run it). ``exit_status`` is what ``DIR/exit`` holds (``None``: no exit file yet); it becomes
    ``exit_after_probes_status`` once the healthcheck has been probed ``exit_after_probes`` times.
    ``health_failures`` is how many healthcheck probes fail before one passes, ``log`` the entrypoint's
    output. ``download_fails`` makes the log transfer fail. ``exit_after_exit_probes`` makes the exit
    file appear right after the N-th exit probe (a death between two probes). ``release_after_probes``
    ends the launch exec after the N-th healthcheck probe with ``hold_result`` (the provider capped the
    exec, or ``run.sh`` finished).
    """

    exit_status: str | None = None
    exit_after_probes: int | None = None
    exit_after_probes_status: str = "2"
    exit_after_exit_probes: int | None = None
    exit_after_exit_probes_status: str = "1"
    release_after_probes: int | None = None
    hold_result: SandboxExecResult = OK
    log: str = "LAB-Bench Runtime Setup - Shuffling choices...\nUsing random seed=843121323\n"
    launch_result: SandboxExecResult | None = None
    health_failures: int = 0
    download_fails: bool = False
    launched: bool = False
    launch_cancelled_before_stop: bool | None = None
    log_downloaded_before_stop: bool | None = None
    default_user_at_launch: str | None = None
    health_probes: int = 0
    exit_probes: int = 0
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        record = {"command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s, "user": user, "seen": None}
        if command.startswith(f"sh {ENTRYPOINT_DIR}/run.sh"):
            self.execs.append(record)
            self.launched = True
            self.default_user_at_launch = self.default_user
            if self.launch_result is not None:
                return self.launch_result
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.launch_cancelled_before_stop = not self.stopped
                raise
            return self.hold_result
        if command == f"test -f {ENTRYPOINT_DIR}/started":
            self.execs.append(record)
            ok = self.launched and self.launch_result is None
            return SandboxExecResult(stdout="", stderr="", return_code=0 if ok else 1)
        if command.startswith(f"if test -f {ENTRYPOINT_DIR}/exit"):
            self.execs.append(record)
            self.exit_probes += 1
            status = self.exit_status
            if self.exit_after_exit_probes is not None and self.exit_probes >= self.exit_after_exit_probes:
                self.exit_status = self.exit_after_exit_probes_status
            return SandboxExecResult(stdout=status or "", stderr="", return_code=0)
        if command.startswith("tail -c "):
            self.execs.append(record)
            return SandboxExecResult(stdout=self.log[-1500:], stderr="", return_code=0)
        if command.startswith(f"if [ -d {ENTRYPOINT_DIR}/log ]"):
            self.execs.append(record)
            return SandboxExecResult(stdout="file\n" if self.launched else "none\n", stderr="", return_code=0)
        if command in ("test -f /app/ready.txt", "test -f /tmp/healthy"):
            self.execs.append(record)
            self.health_probes += 1
            if self.exit_after_probes is not None and self.health_probes >= self.exit_after_probes:
                self.exit_status = self.exit_after_probes_status
            if self.release_after_probes is not None and self.health_probes >= self.release_after_probes:
                self.release.set()
            healthy = self.health_probes > self.health_failures
            return SandboxExecResult(stdout="", stderr="", return_code=0 if healthy else 1)
        return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

    async def download(self, remote_path, local_path):
        if remote_path == f"{ENTRYPOINT_DIR}/log":
            if self.download_fails:
                raise RuntimeError("transfer lost")
            self.log_downloaded_before_stop = not self.stopped
            Path(local_path).write_text(self.log)
            return
        await super().download(remote_path, local_path)

    def launch(self) -> dict | None:
        launches = [c for c in self.execs if c["command"].startswith(f"sh {ENTRYPOINT_DIR}/run.sh")]
        assert len(launches) <= 1, launches
        return launches[0] if launches else None

    def index_of(self, fragment: str) -> int:
        return next(i for i, c in enumerate(self.execs) if fragment in c["command"])

    def run_script(self) -> str | None:
        for index, (_, remote) in enumerate(self.uploads):
            if remote == f"{ENTRYPOINT_DIR}/run.sh":
                return self.uploaded_archives[index].decode()
        return None


@dataclass
class LateExitSandbox(EntrypointSandbox):
    """The entrypoint exits once the sandbox is handed to the harness (after the seed's last probe).

    ``late_returns`` is whether ``run.sh`` finished too, so the exec holding the entrypoint returns, or
    only the exit file appears (what an agent could write into the world-writable runtime folder).
    """

    late_status: str = "1"
    late_returns: bool = True

    async def serialize(self, *, scope=None):
        self.exit_status = self.late_status
        self.log = "odoo: database connection lost\n"
        if self.late_returns:
            self.release.set()
            await asyncio.sleep(0)  # the holding exec returns while the harness works, as in a real rollout
        return await super().serialize(scope=scope)


AGENT_USER_TOML = TASK_TOML.replace("[agent]\ntimeout_sec = 300.0", '[agent]\ntimeout_sec = 300.0\nuser = "agent"')
WORKDIR_TOML = AGENT_USER_TOML.replace("storage_mb = 10240", 'storage_mb = 10240\nworkdir = "/work"')
AGENT_USER_HEALTH_TOML = HEALTHCHECK_TOML.replace(
    "[agent]\ntimeout_sec = 300.0", '[agent]\ntimeout_sec = 300.0\nuser = "agent"'
)
SEPARATE_HEALTH_TOML = HEALTHCHECK_TOML.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n')
NO_USER_CONFIG = {"Env": [f"PATH={IMAGE_PATH}"], "Entrypoint": ["/entrypoint.sh"], "WorkingDir": "/app"}
IMAGE_HEALTHCHECK = {"Test": ["CMD-SHELL", "test -f /tmp/healthy"], "Interval": 10_000_000}


def build_mode(task, *, record: dict | None = ENTRYPOINT_CONFIG, image: str = BUILT_IMAGE):
    """Turn the test task into a build-mode one: a Dockerfile that needs a build, resolved to ``image``.

    ``record`` is the built image's recorded configuration (``None``: nothing recorded, as for an image
    prepare did not inspect).
    """
    from nemo_gym.tasks.harbor.digest import context_digest
    from nemo_gym.tasks.harbor.task import BUILT_IMAGES_FILE

    (task.path / "environment" / "Dockerfile").write_text(
        'FROM ubuntu:24.04\nCOPY . /app\nENTRYPOINT ["/entrypoint.sh"]\n'
    )
    record_path = task.path.parent / BUILT_IMAGES_FILE
    built = json.loads(record_path.read_text()) if record_path.is_file() else {}
    built[context_digest(task.path / "environment")] = {"image": image}
    record_path.write_text(json.dumps(built))
    if record is not None:
        write_image_config(task.path.parent, image, **record)
    return load_task(task.path)


class TestEntrypoint:
    """G4: a plain task's image ENTRYPOINT runs beside the keepalive, as Harbor's Compose start leaves it."""

    def entrypoint_server(self, tmp_path, monkeypatch, *, sandbox=None, toml=TASK_TOML, record=ENTRYPOINT_CONFIG):
        sandbox = sandbox or EntrypointSandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox, toml=toml)
        task = build_mode(task, record=record)
        server.config.tasksets["ds"].tasks["hello"] = task.digest
        return server, task, sandbox

    async def test_built_image_entrypoint_runs_as_the_image_user_between_logs_and_healthcheck(
        self, tmp_path, monkeypatch
    ):
        toml = HEALTHCHECK_TOML.replace("storage_mb = 10240", 'storage_mb = 10240\nskills_dir = "/skills"')
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, toml=toml)
        assert task.entrypoint == ("/entrypoint.sh",) and task.user == "app"

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0
        script = sandbox.run_script()
        assert script is not None
        assert script.splitlines() == [
            f"touch {ENTRYPOINT_DIR}/started",
            "/entrypoint.sh sh -c 'sleep infinity'",
            "status=$?",
            f"printf '%s' \"$status\" > {ENTRYPOINT_DIR}/exit",
        ]
        launch = sandbox.launch()
        assert launch["command"] == f"sh {ENTRYPOINT_DIR}/run.sh >{ENTRYPOINT_DIR}/log 2>&1 </dev/null"
        assert (launch["user"], launch["cwd"], launch["timeout_s"]) == ("app", "/app", None)
        # The runtime folder is made as root and opened up, so the image user can write the markers.
        runtime = next(c for c in sandbox.execs if f"chmod 777 {ENTRYPOINT_DIR}" in c["command"])
        assert runtime["command"] == f"mkdir -p {ENTRYPOINT_DIR} && chmod 777 {ENTRYPOINT_DIR}"
        assert runtime["user"] == "root"
        # Order: /logs exists before the entrypoint starts; the healthcheck and the task context come after it.
        logs = sandbox.index_of("/logs/artifacts")
        started = sandbox.index_of(f"sh {ENTRYPOINT_DIR}/run.sh")
        assert logs < sandbox.index_of(f"mkdir -p {ENTRYPOINT_DIR}") < started
        # The started marker was polled until it existed, then the exit status probed once, before the healthcheck.
        polls = [i for i, c in enumerate(sandbox.execs) if c["command"] == f"test -f {ENTRYPOINT_DIR}/started"]
        assert polls and polls[-1] < sandbox.index_of(f"if test -f {ENTRYPOINT_DIR}/exit")
        assert started < sandbox.index_of("test -f /app/ready.txt") < sandbox.index_of("/tmp/.nemo-gym/task.json")
        # The entrypoint's exec was cancelled before the sandbox was stopped, and nothing stays owned.
        assert sandbox.launch_cancelled_before_stop is True and sandbox.stopped
        assert not server._sandboxes and not server._sessions

    @pytest.mark.parametrize(
        "record",
        [
            pytest.param(None, id="no-record"),
            pytest.param({"Env": [f"PATH={IMAGE_PATH}"]}, id="no-entrypoint"),
            pytest.param({"Entrypoint": []}, id="cleared"),
            pytest.param({"Entrypoint": ["sleep", "infinity"]}, id="sleep"),
            pytest.param({"Entrypoint": ["/usr/bin/tail", "-f", "/dev/null"]}, id="tail"),
            pytest.param({"Entrypoint": ["sh", "-c", "sleep infinity"]}, id="sh-sleep"),
            pytest.param({"Entrypoint": ["/bin/bash", "-c", "exec tail -f /dev/null"]}, id="bash-exec-tail"),
            pytest.param({"Entrypoint": ["sh", "-c", "while true; do sleep 1; done"]}, id="loop"),
        ],
    )
    async def test_nothing_is_launched_without_a_real_entrypoint(self, tmp_path, monkeypatch, record):
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, record=record)
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert sandbox.launch() is None and sandbox.run_script() is None
        assert not any(ENTRYPOINT_DIR in c["command"] for c in sandbox.execs)

    def test_compose_tasks_leave_the_entrypoint_to_the_adapter(self, tmp_path, monkeypatch):
        server, task, _ = self.entrypoint_server(tmp_path, monkeypatch)
        assert entrypoint_to_run(task) == ["/entrypoint.sh"]
        (task.path / "environment" / "docker-compose.yaml").write_text("services:\n  main:\n    image: x\n")
        task = build_mode(task)  # the Compose file is part of the build context, so the record is keyed anew
        assert task.needs_compose and task.entrypoint == ("/entrypoint.sh",)
        assert entrypoint_to_run(task) is None

    @pytest.mark.parametrize(
        ("argv", "keepalive"),
        [
            (["sleep", "infinity"], True),
            (["bash", "-c", "exec sleep infinity"], True),
            (["sh", "-c", "while :; do sleep 30; done"], True),
            (["/entrypoint.sh"], False),
            (["bash", "/opt/blobfish-world/entrypoint.sh"], False),
            (["sh", "-c", "  exec sleep infinity  "], True),
            (["sh", "-c", "sleep 5; python app.py"], False),  # compound: the services after the sleep must start
            (["sh", "-c", "sleep 2; service redis-server start; exec sleep infinity"], False),
            (["sh", "-c", "sleep 5 && service postgresql start && exec sleep infinity"], False),
            (["bash", "-c", "tail -n0 -f /var/log/app.log | ./watcher"], False),
            (["sh", "-c", "python app.py && sleep infinity"], False),
            (["bash"], False),
            ([], False),
        ],
    )
    def test_keepalive_rule(self, argv, keepalive):
        assert is_keepalive_entrypoint(argv) is keepalive

    async def test_overlay_task_launches_the_base_entrypoint_after_the_run_lines(self, tmp_path, monkeypatch):
        dockerfile = OVERLAY_DOCKERFILE.replace("ENTRYPOINT []\n", "")
        config = {"Env": [f"PATH={IMAGE_PATH}"], "Entrypoint": ["/entrypoint.sh"]}
        sandbox = EntrypointSandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox, dockerfile=dockerfile, image_config=config)
        # The base image's entrypoint, started with the Dockerfile's final USER and WORKDIR.
        assert task.entrypoint == ("/entrypoint.sh",)
        assert (task.entrypoint_user, task.entrypoint_workdir) == ("agent", "/app")

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        launch = sandbox.launch()
        assert (launch["user"], launch["cwd"]) == ("agent", "/app")
        last_run = max(i for i, c in enumerate(sandbox.execs) if c["seen"] and "as agent" in c["seen"]["command"])
        assert last_run < sandbox.index_of(f"sh {ENTRYPOINT_DIR}/run.sh")
        assert sandbox.default_user_at_launch == "agent"

        # The same Dockerfile with ENTRYPOINT [] clears it: nothing is launched.
        cleared = EntrypointSandbox()
        server, task, _, _ = make_server(
            tmp_path / "b", monkeypatch, cleared, image_config=config, dockerfile=OVERLAY_DOCKERFILE
        )
        assert task.entrypoint is None and task.entrypoint_user == "agent"
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert cleared.launch() is None

    async def test_entrypoint_exiting_non_zero_is_a_terminal_seed_failure_with_its_output(self, tmp_path, monkeypatch):
        sandbox = EntrypointSandbox(exit_status="1", log="LAB-Bench Runtime Setup\nTraceback: setup failed\n")
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox)

        response = await run_episode(server, task)

        failure = response.failure
        assert failure is not None and failure.stage == "seed" and failure.terminal
        assert "Image ENTRYPOINT failed for 'hello'" in failure.failure_reason
        assert "ENTRYPOINT /entrypoint.sh exited 1" in failure.failure_reason
        assert "Traceback: setup failed" in failure.failure_reason
        assert sandbox.launch_cancelled_before_stop is True and sandbox.stopped
        assert not server._sandboxes and not server._sessions
        assert server.server_client.paths() == []  # never handed to the harness

    async def test_entrypoint_returning_zero_warns_once_and_continues(self, tmp_path, monkeypatch, caplog):
        sandbox = EntrypointSandbox(exit_status="0", health_failures=2)
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox, toml=HEALTHCHECK_TOML)

        with caplog.at_level("WARNING", logger="environment_servers.harbor.app"):
            response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0
        warnings = [r.getMessage() for r in caplog.records if "exited 0" in r.getMessage()]
        assert len(warnings) == 1
        assert "hello: ENTRYPOINT /entrypoint.sh exited 0 without running its keepalive argument" in warnings[0]
        assert "LAB-Bench Runtime Setup" in warnings[0]
        # The healthcheck was still polled to a pass; the exit was not probed again once known.
        assert sandbox.health_probes == 3 and sandbox.exit_probes == 1

    async def test_launcher_that_never_starts_is_a_retryable_seed_failure(self, tmp_path, monkeypatch):
        sandbox = EntrypointSandbox(
            launch_result=SandboxExecResult(stdout="", stderr="", return_code=125, error_type="sandbox")
        )
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox)

        response = await run_episode(server, task)

        failure = response.failure
        assert failure is not None and failure.stage == "seed" and not failure.terminal
        assert "the entrypoint launcher exited before starting: sandbox" in failure.failure_reason
        assert sandbox.stopped and not server._sandboxes

    async def test_launcher_shell_failing_before_the_marker_is_a_terminal_seed_failure(self, tmp_path, monkeypatch):
        # `su`/`chdir` for the image's user or directory failed before run.sh's first line: the image's fault,
        # like a RUN line exiting non-zero; another sandbox would repeat it.
        sandbox = EntrypointSandbox(
            launch_result=SandboxExecResult(stdout="", stderr="su: user 1000 does not exist\n", return_code=1)
        )
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox)

        response = await run_episode(server, task)

        failure = response.failure
        assert failure is not None and failure.stage == "seed" and failure.terminal
        assert "Image ENTRYPOINT failed for 'hello'" in failure.failure_reason
        assert "(as app in /app) exited 1 before starting it" in failure.failure_reason
        assert "su: user 1000 does not exist" in failure.failure_reason
        assert sandbox.stopped and not server._sandboxes and server.server_client.paths() == []

    async def test_launcher_start_timeout_is_a_retryable_seed_failure(self, tmp_path, monkeypatch):
        class NeverStarted(EntrypointSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if command == f"test -f {ENTRYPOINT_DIR}/started":
                    self.execs.append({"command": command})
                    return SandboxExecResult(stdout="", stderr="", return_code=1)
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

        monkeypatch.setattr("environment_servers.harbor.app.ENTRYPOINT_START_TIMEOUT_S", 0.05)
        monkeypatch.setattr("environment_servers.harbor.app.ENTRYPOINT_POLL_INTERVAL_S", 0.01)
        sandbox = NeverStarted()
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox)

        response = await run_episode(server, task)

        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "did not start within 0.05s" in response.failure.failure_reason
        assert sandbox.launch_cancelled_before_stop is True and sandbox.stopped

    async def test_healthcheck_loop_fails_fast_when_the_entrypoint_dies(self, tmp_path, monkeypatch):
        # The healthcheck never passes; the entrypoint exits after the first probe. Without the probe the
        # seed would wait out the three retries.
        sandbox = EntrypointSandbox(health_failures=100, exit_after_probes=1, log="odoo: connection refused\n")
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox, toml=HEALTHCHECK_TOML)

        response = await run_episode(server, task)

        failure = response.failure
        assert failure.stage == "seed" and failure.terminal
        assert "exited 2" in failure.failure_reason and "odoo: connection refused" in failure.failure_reason
        assert sandbox.health_probes == 1
        assert "Healthcheck failed" not in failure.failure_reason

    def test_image_healthcheck_settings(self):
        ns = 1_000_000_000
        settings = image_healthcheck_settings(
            {
                "Test": ["CMD-SHELL", "curl -f http://localhost/ || exit 1"],
                "Interval": 10 * ns,
                "Timeout": 3 * ns,
                "StartPeriod": 2 * ns,
                "Retries": 5,
            }
        )
        assert settings.command == "curl -f http://localhost/ || exit 1"
        assert (settings.interval_sec, settings.timeout_sec, settings.start_period_sec, settings.retries) == (
            10,
            3,
            2,
            5,
        )
        assert settings.start_interval_sec == 5.0  # Docker's default when the Dockerfile sets none
        exec_form = image_healthcheck_settings({"Test": ["CMD", "pg_isready", "-U", "odoo user"]})
        assert exec_form.command == "pg_isready -U 'odoo user'"
        assert (exec_form.interval_sec, exec_form.timeout_sec, exec_form.retries) == (30.0, 30.0, 3)
        assert image_healthcheck_settings({"Test": ["NONE"]}) is None
        assert image_healthcheck_settings({"Test": []}) is None
        assert image_healthcheck_settings({}) is None and image_healthcheck_settings(None) is None

    @pytest.mark.parametrize("passes", [True, False])
    async def test_recorded_dockerfile_healthcheck_is_polled(self, tmp_path, monkeypatch, passes):
        ns = 1_000_000_000
        record = ENTRYPOINT_CONFIG | {
            "Healthcheck": {
                "Test": ["CMD-SHELL", "test -f /tmp/healthy"],
                "Interval": ns // 100,
                "Timeout": 5 * ns,
                "Retries": 3,
            }
        }
        sandbox = EntrypointSandbox(health_failures=2 if passes else 100)
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox, record=record)
        assert task.image_healthcheck == record["Healthcheck"]

        response = await run_episode(server, task)

        if passes:
            assert response.failure is None, response.failure
            assert sandbox.health_probes == 3
            probe = next(c for c in sandbox.execs if c["command"] == "test -f /tmp/healthy")
            assert probe["timeout_s"] == 10  # the recorded 5 s timeout plus the server's margin
            assert sandbox.index_of(f"sh {ENTRYPOINT_DIR}/run.sh") < sandbox.index_of("test -f /tmp/healthy")
        else:
            assert response.failure.stage == "seed" and not response.failure.terminal
            assert (
                "Healthcheck failed 3 times: 'test -f /tmp/healthy' (the image HEALTHCHECK)"
                in response.failure.failure_reason
            )
            assert sandbox.health_probes == 3 and sandbox.stopped and not server._sandboxes

    async def test_image_healthcheck_runs_before_the_task_healthcheck_as_the_image_user(self, tmp_path, monkeypatch):
        # Harbor's order: `up --wait` gates on the image HEALTHCHECK, then run_healthcheck() runs the task's, both
        # as the container's user (the agent user is set afterwards): the daemon runs the image check in the image
        # WorkingDir, Harbor's exec runs the task check in the task workdir.
        record = ENTRYPOINT_CONFIG | {"Healthcheck": IMAGE_HEALTHCHECK}
        server, task, sandbox = self.entrypoint_server(
            tmp_path, monkeypatch, toml=AGENT_USER_HEALTH_TOML, record=record
        )
        assert (task.user, task.entrypoint_user) == ("agent", "app")
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        image = sandbox.index_of("test -f /tmp/healthy")
        task_probe = sandbox.index_of("test -f /app/ready.txt")
        assert sandbox.index_of(f"sh {ENTRYPOINT_DIR}/run.sh") < image < task_probe
        assert (sandbox.execs[image]["user"], sandbox.execs[image]["cwd"]) == ("app", "/app")
        assert (sandbox.execs[task_probe]["user"], sandbox.execs[task_probe]["cwd"]) == ("app", None)

    @pytest.mark.parametrize(
        ("toml", "user"), [(HEALTHCHECK_TOML, None), (AGENT_USER_HEALTH_TOML, "root")], ids=["root", "agent-user"]
    )
    async def test_root_image_healthchecks_run_as_root_whatever_the_agent_user(
        self, tmp_path, monkeypatch, toml, user
    ):
        record = NO_USER_CONFIG | {"Healthcheck": IMAGE_HEALTHCHECK}
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, toml=toml, record=record)
        assert task.entrypoint_user is None
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        image = sandbox.execs[sandbox.index_of("test -f /tmp/healthy")]
        task_probe = sandbox.execs[sandbox.index_of("test -f /app/ready.txt")]
        assert (image["user"], image["cwd"]) == (user, "/app")
        assert (task_probe["user"], task_probe["cwd"]) == (user, None)

    async def test_slow_service_passes_when_the_image_healthcheck_absorbs_the_wait(self, tmp_path, monkeypatch):
        # domainbench-24's shape: no image USER, [agent] user = "agent", a task check with retries 3 and no start
        # period, an image HEALTHCHECK with retries 60, a service that listens after five probes. In Harbor's
        # order the image check absorbs the wait and the task check then passes; task-first failed after 3 s.
        ns = 1_000_000_000
        record = NO_USER_CONFIG | {
            "Healthcheck": {"Test": ["CMD", "test", "-f", "/tmp/healthy"], "Interval": ns // 100, "Retries": 60}
        }
        sandbox = EntrypointSandbox(health_failures=5)
        server, task, sandbox = self.entrypoint_server(
            tmp_path, monkeypatch, sandbox=sandbox, toml=AGENT_USER_HEALTH_TOML, record=record
        )
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert sandbox.health_probes == 7
        image_probes = [i for i, c in enumerate(sandbox.execs) if c["command"] == "test -f /tmp/healthy"]
        assert len(image_probes) == 6 and image_probes[-1] < sandbox.index_of("test -f /app/ready.txt")

    VERIFIER_IMAGE = "registry.example/ds-hello-verifier:abc"

    def separate_server(
        self, tmp_path, monkeypatch, *, toml, agent=None, verifier=None, record=ENTRYPOINT_CONFIG, verifier_record=None
    ):
        """A separate-mode task in build mode: its agent image is ``record``'s, and (D1) its verifier grades in the
        image prepare built from ``tests/Dockerfile``, recorded with ``verifier_record`` (``record`` by default, so
        both images start alike). A declared ``[verifier.environment].docker_image`` in ``toml`` takes precedence."""
        agent = agent or EntrypointSandbox()
        verifier = verifier or EntrypointSandbox()
        # Written in shared mode first: the loader refuses a separate-mode task before its verifier image is recorded.
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent)
        (task.path / "task.toml").write_text(toml)
        (task.path / "tests" / "Dockerfile").write_text(VERIFIER_TESTS_DOCKERFILE)
        record_verifier_build(
            task.path, self.VERIFIER_IMAGE, **(record if verifier_record is None else verifier_record)
        )
        task = build_mode(task, record=record)
        server.config.tasksets["ds"].tasks["hello"] = task.digest

        async def create_verifier(task, *, episode_key=None):
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        return server, task, agent, verifier

    async def test_separate_verifier_in_the_built_image_runs_its_entrypoint_before_test_sh(
        self, tmp_path, monkeypatch
    ):
        server, task, agent, verifier = self.separate_server(tmp_path, monkeypatch, toml=SEPARATE_IN_TASK_IMAGE_TOML)
        # The loader read the verifier image's record (D1 built it from tests/Dockerfile) beside the task image's.
        assert task.verifier_image == self.VERIFIER_IMAGE and task.entrypoint == ("/entrypoint.sh",)
        assert task.verifier_image_config.entrypoint == ("/entrypoint.sh",)
        assert (task.verifier_image_config.user, task.verifier_image_config.workdir) == ("app", "/app")

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.verifier_mode == "separate"
        assert agent.launch() is not None and verifier.launch() is not None
        # The verifier's /logs folders exist before its entrypoint starts, as in the seed; test.sh comes after.
        launched = verifier.index_of(f"sh {ENTRYPOINT_DIR}/run.sh")
        assert (
            verifier.index_of("mkdir -p /tests /logs/verifier /logs/agent /logs/artifacts")
            < launched
            < verifier.index_of("test.sh")
        )
        assert (verifier.launch()["user"], verifier.launch()["cwd"]) == ("app", "/app")
        # Both launchers were cancelled before their sandbox was stopped.
        assert agent.launch_cancelled_before_stop is True and agent.stopped
        assert verifier.launch_cancelled_before_stop is True and verifier.stopped
        assert not server._sandboxes and not server._sessions

    async def test_built_verifier_image_starts_with_its_own_record_not_the_task_images(self, tmp_path, monkeypatch):
        # Harbor `up --wait`s the image built from tests/Dockerfile: its entrypoint, as its user in its WorkingDir,
        # and its HEALTHCHECK, whatever the agent image's are.
        verifier_record = {
            "Entrypoint": ["/verify.sh"],
            "User": "grader",
            "WorkingDir": "/tests",
            "Healthcheck": IMAGE_HEALTHCHECK,
        }
        verifier = EntrypointSandbox(health_failures=1)
        server, task, agent, verifier = self.separate_server(
            tmp_path, monkeypatch, toml=SEPARATE_IN_TASK_IMAGE_TOML, verifier=verifier, verifier_record=verifier_record
        )
        assert task.image_healthcheck is None and task.verifier_image_config.healthcheck == IMAGE_HEALTHCHECK

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0
        assert (agent.launch()["user"], agent.launch()["cwd"]) == ("app", "/app")
        assert (verifier.launch()["user"], verifier.launch()["cwd"]) == ("grader", "/tests")
        assert "/verify.sh sh -c 'sleep infinity'" in verifier.run_script()
        assert "/entrypoint.sh" not in verifier.run_script()
        assert verifier.health_probes == 2 and agent.health_probes == 0
        assert verifier.index_of(f"sh {ENTRYPOINT_DIR}/run.sh") < verifier.index_of("test -f /tmp/healthy")
        assert verifier.index_of("test -f /tmp/healthy") < verifier.index_of("bash /tests/test.sh")
        # A verifier image whose record has no entrypoint starts nothing, while the agent image's still runs.
        quiet = EntrypointSandbox()
        server, task, agent, quiet = self.separate_server(
            tmp_path / "b",
            monkeypatch,
            toml=SEPARATE_IN_TASK_IMAGE_TOML,
            verifier=quiet,
            verifier_record={"Env": [f"PATH={IMAGE_PATH}"], "WorkingDir": "/tests"},
        )
        assert task.verifier_image_config.entrypoint is None
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert agent.launch() is not None and quiet.launch() is None
        assert not any(ENTRYPOINT_DIR in c["command"] for c in quiet.execs)

    async def test_separate_verifier_in_the_task_image_runs_its_entrypoint_before_test_sh(self, tmp_path, monkeypatch):
        # Harbor's prebuilt verifier: no [verifier.environment], so the effective verifier environment is a copy of
        # [environment] with its docker_image, and the verifier grades in the task's own image, whose recorded start
        # (here the pull-mode Dockerfile's base image's) is what Harbor's `up --wait` runs.
        toml = SEPARATE_IN_TASK_IMAGE_TOML.replace("[environment]\n", '[environment]\ndocker_image = "org/app:1"\n')
        agent, verifier = EntrypointSandbox(), EntrypointSandbox()
        server, task, _, _ = make_server(
            tmp_path,
            monkeypatch,
            agent,
            toml=toml,
            dockerfile="FROM org/app:1\nWORKDIR /app\n",
            image_config=ENTRYPOINT_CONFIG,
        )
        assert task.image == "org/app:1" and task.verifier_image is None and task.verifier_image_config is None
        assert task.entrypoint == ("/entrypoint.sh",) and (task.entrypoint_user, task.entrypoint_workdir) == (
            "app",
            "/app",
        )

        async def create_verifier(task, *, episode_key=None):
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.verifier_mode == "separate"
        assert agent.launch() is not None and verifier.launch() is not None
        assert (verifier.launch()["user"], verifier.launch()["cwd"]) == ("app", "/app")
        assert verifier.index_of(f"sh {ENTRYPOINT_DIR}/run.sh") < verifier.index_of("test.sh")
        assert verifier.launch_cancelled_before_stop is True and verifier.stopped and not server._sandboxes

    async def test_separate_verifier_in_another_image_gets_no_launcher(self, tmp_path, monkeypatch):
        toml = SEPARATE_IN_TASK_IMAGE_TOML + '\n[verifier.environment]\ndocker_image = "org/verifier:1"\n'
        server, task, agent, verifier = self.separate_server(tmp_path, monkeypatch, toml=toml)
        assert _verifier_image(task) == "org/verifier:1" and task.entrypoint == ("/entrypoint.sh",)
        assert task.verifier_image_config is None  # a declared prebuilt verifier image has no record
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert agent.launch() is not None and verifier.launch() is None
        assert not any(ENTRYPOINT_DIR in c["command"] for c in verifier.execs)

    async def test_verifier_entrypoint_failure_masks_the_sample_and_keeps_the_rollout(self, tmp_path, monkeypatch):
        verifier = EntrypointSandbox(exit_status="3", log="cannot bind port 8069\n")
        server, task, agent, verifier = self.separate_server(
            tmp_path, monkeypatch, toml=SEPARATE_IN_TASK_IMAGE_TOML, verifier=verifier
        )

        response = await run_episode(server, task)

        result = response.result
        assert result is not None, response.failure
        assert result.mask_sample is True and result.failure_kind == ENTRYPOINT_FAILED_KIND
        assert "exited 3" in result.failure_reason and "cannot bind port 8069" in result.failure_reason
        assert result.response is not None  # the rollout is kept
        assert not any("test.sh" in c["command"] for c in verifier.execs)
        assert verifier.stopped and agent.stopped and not server._sandboxes

    async def test_entrypoint_log_is_collected_in_shared_mode(self, tmp_path, monkeypatch):
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch)
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        logs_dir = Path(response.result.verifier_logs_dir)
        assert (logs_dir / "entrypoint-stdout.txt").read_text() == sandbox.log
        assert not (logs_dir / "verifier-entrypoint-stdout.txt").exists()

    async def test_entrypoint_logs_are_collected_in_separate_mode(self, tmp_path, monkeypatch):
        agent = EntrypointSandbox(log="agent side setup\n")
        verifier = EntrypointSandbox(log="verifier side setup\n")
        server, task, agent, verifier = self.separate_server(
            tmp_path, monkeypatch, toml=SEPARATE_IN_TASK_IMAGE_TOML, agent=agent, verifier=verifier
        )
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        logs_dir = Path(response.result.verifier_logs_dir)
        assert (logs_dir / "entrypoint-stdout.txt").read_text() == "agent side setup\n"
        assert (logs_dir / "verifier-entrypoint-stdout.txt").read_text() == "verifier side setup\n"
        # The agent's log was taken before its sandbox was stopped.
        assert agent.log_downloaded_before_stop is True and agent.stopped

    async def test_entrypoint_log_collection_failure_only_warns(self, tmp_path, monkeypatch, caplog):
        sandbox = EntrypointSandbox(download_fails=True)
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox)
        with caplog.at_level("WARNING", logger="environment_servers.harbor.app"):
            response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.mask_sample is False
        assert "Could not collect the entrypoint log for hello" in caplog.text
        assert not (Path(response.result.verifier_logs_dir) / "entrypoint-stdout.txt").exists()

    async def test_stop_during_the_episode_cancels_the_launcher_before_the_sandbox(self, tmp_path, monkeypatch):
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch)
        seen: list = []

        async def observe():
            session = server._sessions["r1"]
            seen.append((session.entrypoint is not None, session.entrypoint.process.done()))
            await server._stop_sandbox("r1")
            seen.append((sandbox.launch_cancelled_before_stop, sandbox.stopped))

        server.server_client.turn_hooks.append(observe)
        await run_episode(server, task)
        assert seen == [(True, False), (True, True)]

    # -- the exit status after the seed's first probe -------------------------------------------

    async def test_entrypoint_dying_before_the_hand_over_is_still_a_seed_failure(self, tmp_path, monkeypatch):
        # The probe right after the started marker sees nothing (a setup script still importing); the
        # entrypoint dies during the seed's remaining steps and the probe before the hand-over catches it.
        sandbox = EntrypointSandbox(exit_after_exit_probes=1, log="setup.py: missing data file\n")
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox)

        response = await run_episode(server, task)

        failure = response.failure
        assert failure is not None and failure.stage == "seed" and failure.terminal
        assert "ENTRYPOINT /entrypoint.sh exited 1" in failure.failure_reason
        assert "missing data file" in failure.failure_reason
        probes = [
            i for i, c in enumerate(sandbox.execs) if c["command"].startswith(f"if test -f {ENTRYPOINT_DIR}/exit")
        ]
        # Once after the started marker, once at the end of the seed (the output tail is the last exec).
        assert len(probes) == 2 and probes[-1] == len(sandbox.execs) - 2
        assert sandbox.execs[-1]["command"].startswith("tail -c ")
        assert server.server_client.paths() == []  # never handed to the harness
        assert sandbox.stopped and not server._sandboxes and not server._sessions

    async def test_entrypoint_dying_during_the_rollout_masks_the_sample_when_its_exec_returned(
        self, tmp_path, monkeypatch
    ):
        # Under Harbor the entrypoint is PID 1: its exit kills the container and the trial errors instead of
        # scoring. run.sh finished (the holding exec returned) with a non-zero status: masked, rollout kept.
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=LateExitSandbox())

        response = await run_episode(server, task)

        result = response.result
        assert result is not None, response.failure
        assert result.mask_sample is True and result.failure_kind == ENTRYPOINT_FAILED_KIND
        assert "ENTRYPOINT /entrypoint.sh exited 1 during the rollout" in result.failure_reason
        assert "database connection lost" in result.failure_reason
        assert result.response is not None  # the rollout is kept
        assert not any("test.sh" in c["command"] for c in sandbox.execs)
        logs_dir = Path(result.verifier_logs_dir)
        assert (logs_dir / "entrypoint-stdout.txt").read_text() == sandbox.log
        assert (logs_dir / "entrypoint-exit.txt").read_text() == "1\n"
        assert sandbox.stopped and not server._sandboxes

    async def test_exit_file_alone_is_recorded_and_the_sample_still_scored(self, tmp_path, monkeypatch, caplog):
        # The runtime folder is world-writable and the agent had the sandbox: an exit file the holding exec
        # does not corroborate is kept and logged, never an agent-controlled mask.
        sandbox = LateExitSandbox(late_returns=False)
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox)

        with caplog.at_level("ERROR", logger="environment_servers.harbor.app"):
            response = await run_episode(server, task)

        result = response.result
        assert result is not None, response.failure
        assert result.reward == 1.0 and result.mask_sample is False
        assert any("test.sh" in c["command"] for c in sandbox.execs)
        assert (Path(result.verifier_logs_dir) / "entrypoint-exit.txt").read_text() == "1\n"
        errors = [r.getMessage() for r in caplog.records if "exit status 1 recorded" in r.getMessage()]
        assert len(errors) == 1 and "did not return (still held)" in errors[0]
        assert sandbox.launch_cancelled_before_stop is True

    async def test_late_exit_zero_warns_and_the_sample_is_scored(self, tmp_path, monkeypatch, caplog):
        sandbox = LateExitSandbox(late_status="0")
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox)
        with caplog.at_level("WARNING", logger="environment_servers.harbor.app"):
            response = await run_episode(server, task)
        result = response.result
        assert result is not None, response.failure
        assert result.reward == 1.0 and result.mask_sample is False
        assert (Path(result.verifier_logs_dir) / "entrypoint-exit.txt").read_text() == "0\n"
        warnings = [r.getMessage() for r in caplog.records if "exited 0 during the rollout" in r.getMessage()]
        assert len(warnings) == 1

    async def test_entrypoint_dying_during_the_rollout_masks_in_separate_mode_too(self, tmp_path, monkeypatch):
        agent = LateExitSandbox()
        server, task, agent, verifier = self.separate_server(
            tmp_path, monkeypatch, toml=SEPARATE_IN_TASK_IMAGE_TOML, agent=agent
        )

        response = await run_episode(server, task)

        result = response.result
        assert result is not None, response.failure
        assert result.mask_sample is True and result.failure_kind == ENTRYPOINT_FAILED_KIND
        assert result.response is not None
        assert verifier.launch() is None and not verifier.execs  # no verifier sandbox was needed
        logs_dir = Path(result.verifier_logs_dir)
        assert (logs_dir / "entrypoint-stdout.txt").read_text() == agent.log
        assert (logs_dir / "entrypoint-exit.txt").read_text() == "1\n"
        assert agent.log_downloaded_before_stop is True and agent.stopped and not server._sandboxes

    # -- the exec that holds the entrypoint --------------------------------------------------

    async def test_launcher_capped_by_the_provider_is_logged_once_and_the_episode_goes_on(
        self, tmp_path, monkeypatch, caplog
    ):
        # Docker ends the holding exec at exec.default_timeout_s with the process still running.
        sandbox = EntrypointSandbox(
            health_failures=2,
            release_after_probes=1,
            hold_result=SandboxExecResult(stdout="", stderr="", return_code=125, error_type="timeout"),
        )
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox, toml=HEALTHCHECK_TOML)

        with caplog.at_level("WARNING", logger="environment_servers.harbor.app"):
            response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.mask_sample is False
        warnings = [r.getMessage() for r in caplog.records if "the exec holding ENTRYPOINT" in r.getMessage()]
        assert len(warnings) == 1
        assert "hello: the exec holding ENTRYPOINT /entrypoint.sh ended (timeout)" in warnings[0]
        assert sandbox.health_probes == 3 and sandbox.launch_cancelled_before_stop is None

    async def test_health_failure_names_the_launcher_the_provider_ended(self, tmp_path, monkeypatch):
        record = ENTRYPOINT_CONFIG | {"Healthcheck": IMAGE_HEALTHCHECK | {"Retries": 3}}
        sandbox = EntrypointSandbox(
            health_failures=100,
            release_after_probes=1,
            hold_result=SandboxExecResult(stdout="", stderr="", return_code=1, error_type="sandbox"),
        )
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox, record=record)

        response = await run_episode(server, task)

        failure = response.failure
        assert failure is not None and failure.stage == "seed" and not failure.terminal
        assert "Healthcheck failed 3 times: 'test -f /tmp/healthy' (the image HEALTHCHECK)" in failure.failure_reason
        assert "after the exec holding the entrypoint ended (sandbox)" in failure.failure_reason

    # -- the launcher's user and working directory -------------------------------------------

    async def test_entrypoint_runs_as_the_image_user_in_the_image_workdir_not_the_task_toml_ones(
        self, tmp_path, monkeypatch
    ):
        # Image User root (no User recorded), [agent].user and [environment].workdir both override.
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, toml=WORKDIR_TOML, record=NO_USER_CONFIG)
        assert (task.user, task.workdir) == ("agent", "/work")
        assert (task.entrypoint_user, task.entrypoint_workdir) == (None, "/app")
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        launch = sandbox.launch()
        assert launch is not None
        assert (launch["user"], launch["cwd"], launch["timeout_s"]) == ("root", "/app", None)
        # A built image holds its WorkingDir: nothing creates it here.
        assert not any(c["command"].startswith("mkdir -p /app") for c in sandbox.execs)

    async def test_agent_user_does_not_move_the_entrypoint_off_the_image_user(self, tmp_path, monkeypatch):
        server, task, sandbox = self.entrypoint_server(
            tmp_path, monkeypatch, toml=AGENT_USER_TOML, record=ENTRYPOINT_CONFIG
        )
        assert (task.user, task.entrypoint_user) == ("agent", "app")
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        launch = sandbox.launch()
        assert (launch["user"], launch["cwd"]) == ("app", "/app")

    async def test_root_image_user_without_an_agent_user_uses_the_plain_exec(self, tmp_path, monkeypatch):
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, toml=TASK_TOML, record=NO_USER_CONFIG)
        assert task.user is None and task.entrypoint_user is None
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        launch = sandbox.launch()
        assert (launch["user"], launch["cwd"]) == (None, "/app")

    async def test_numeric_image_user_runs_the_entrypoint_by_name(self, tmp_path, monkeypatch):
        # OpenSandbox switches user by name (`su ... 1000` fails); the uid is mapped once, for the default
        # user and the launcher alike, as the overlay maps a numeric USER.
        record = ENTRYPOINT_CONFIG | {"User": "1000"}
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, record=record)
        assert task.entrypoint_user == "1000"
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert sandbox.launch()["user"] == "agent" and sandbox.default_user == "agent"
        assert len([c for c in sandbox.execs if c["command"].startswith("getent passwd")]) == 1

    async def test_numeric_image_user_without_a_name_stays_numeric_with_a_warning(self, tmp_path, monkeypatch, caplog):
        sandbox = EntrypointSandbox(passwd={"0": "root"})
        record = ENTRYPOINT_CONFIG | {"User": "1000"}
        server, task, sandbox = self.entrypoint_server(tmp_path, monkeypatch, sandbox=sandbox, record=record)
        with caplog.at_level("WARNING", logger="environment_servers.harbor.app"):
            response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert sandbox.launch()["user"] == "1000"
        warnings = [r.getMessage() for r in caplog.records if "USER 1000 names no user" in r.getMessage()]
        assert len(warnings) == 1

    async def test_resolved_dockerfile_workdir_is_created_for_the_entrypoint(self, tmp_path, monkeypatch):
        # A trailing WORKDIR with no RUN after it: `docker build` would have created it in the image; the seed
        # makes only the task's workdir, which task.toml points elsewhere.
        dockerfile = "FROM debian:bookworm-slim\nRUN useradd agent\nUSER agent\nWORKDIR /app\n"
        config = {"Env": [f"PATH={IMAGE_PATH}"], "Entrypoint": ["/entrypoint.sh"]}
        toml = TASK_TOML.replace("storage_mb = 10240", 'storage_mb = 10240\nworkdir = "/work"')
        sandbox = EntrypointSandbox()
        server, task, _, _ = make_server(
            tmp_path, monkeypatch, sandbox, dockerfile=dockerfile, image_config=config, toml=toml
        )
        assert (task.workdir, task.entrypoint_workdir, task.entrypoint_user) == ("/work", "/app", "agent")

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        made = sandbox.index_of("test -d /app || (mkdir -p /app && chown agent /app)")
        assert sandbox.index_of("mkdir -p /work") < made < sandbox.index_of(f"sh {ENTRYPOINT_DIR}/run.sh")
        assert sandbox.execs[made]["user"] == "root"
        assert sandbox.launch()["cwd"] == "/app" and sandbox.dirs["/app"] == "agent"

    # -- the separate verifier's start -------------------------------------------------------

    async def test_separate_verifier_never_runs_the_task_healthcheck(self, tmp_path, monkeypatch):
        # Harbor builds the separate verifier from tests/Dockerfile and only `up --wait`s it; run_healthcheck()
        # is for the agent's environment. The task image's own HEALTHCHECK is the substitute, nothing more.
        server, task, agent, verifier = self.separate_server(tmp_path, monkeypatch, toml=SEPARATE_HEALTH_TOML)

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.verifier_mode == "separate"
        assert agent.health_probes == 1 and verifier.health_probes == 0
        assert verifier.launch() is not None and any("test.sh" in c["command"] for c in verifier.execs)

    async def test_separate_verifier_waits_for_the_image_healthcheck_before_test_sh(self, tmp_path, monkeypatch):
        record = ENTRYPOINT_CONFIG | {"Healthcheck": IMAGE_HEALTHCHECK}
        verifier = EntrypointSandbox(health_failures=2)
        server, task, agent, verifier = self.separate_server(
            tmp_path, monkeypatch, toml=SEPARATE_IN_TASK_IMAGE_TOML, verifier=verifier, record=record
        )

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.verifier_mode == "separate"
        assert verifier.health_probes == 3
        assert verifier.index_of(f"sh {ENTRYPOINT_DIR}/run.sh") < verifier.index_of("test -f /tmp/healthy")
        assert verifier.index_of("test -f /tmp/healthy") < verifier.index_of("test.sh")
        # The verifier's exit status is probed once more right before test.sh.
        probes = [
            i for i, c in enumerate(verifier.execs) if c["command"].startswith(f"if test -f {ENTRYPOINT_DIR}/exit")
        ]
        assert verifier.index_of("test -f /tmp/healthy") < probes[-1] < verifier.index_of("bash /tests/test.sh")

    async def test_verifier_image_healthcheck_that_never_passes_masks_as_provider_unavailable(
        self, tmp_path, monkeypatch
    ):
        record = ENTRYPOINT_CONFIG | {"Healthcheck": IMAGE_HEALTHCHECK | {"Retries": 3}}
        verifier = EntrypointSandbox(health_failures=10**6)
        server, task, agent, verifier = self.separate_server(
            tmp_path, monkeypatch, toml=SEPARATE_IN_TASK_IMAGE_TOML, verifier=verifier, record=record
        )

        response = await run_episode(server, task)

        result = response.result
        assert result is not None, response.failure
        assert result.mask_sample is True and result.failure_kind == failure_kinds.PROVIDER_UNAVAILABLE
        assert "Healthcheck failed 3 times" in result.failure_reason
        assert result.response is not None  # the rollout is kept
        assert verifier.health_probes == 3
        assert not any("test.sh" in c["command"] for c in verifier.execs)
        assert verifier.stopped and agent.stopped and not server._sandboxes

    async def test_verifier_entrypoint_dying_before_test_sh_masks_the_sample(self, tmp_path, monkeypatch):
        # Dies after the launcher's probe, during the uploads: the probe right before test.sh catches it.
        verifier = EntrypointSandbox(exit_after_exit_probes=1, log="cannot bind port 8069\n")
        server, task, agent, verifier = self.separate_server(
            tmp_path, monkeypatch, toml=SEPARATE_IN_TASK_IMAGE_TOML, verifier=verifier
        )
        response = await run_episode(server, task)
        result = response.result
        assert result is not None, response.failure
        assert result.mask_sample is True and result.failure_kind == ENTRYPOINT_FAILED_KIND
        assert "exited 1" in result.failure_reason and "cannot bind port 8069" in result.failure_reason
        assert not any("bash /tests/test.sh" in c["command"] for c in verifier.execs)
        assert verifier.exit_probes == 2 and verifier.stopped and agent.stopped and not server._sandboxes

    @pytest.mark.parametrize(
        ("launch_result", "kind", "fragment"),
        [
            pytest.param(
                SandboxExecResult(stdout="", stderr="", return_code=125, error_type="sandbox"),
                failure_kinds.PROVIDER_UNAVAILABLE,
                "HarborEntrypointUnavailable",
                id="provider",
            ),
            pytest.param(
                SandboxExecResult(stdout="", stderr="sh: not found\n", return_code=127),
                ENTRYPOINT_FAILED_KIND,
                "exited 127 before starting it",
                id="launcher-shell",
            ),
        ],
    )
    async def test_verifier_launcher_that_cannot_start_masks_the_sample(
        self, tmp_path, monkeypatch, launch_result, kind, fragment
    ):
        verifier = EntrypointSandbox(launch_result=launch_result)
        server, task, agent, verifier = self.separate_server(
            tmp_path, monkeypatch, toml=SEPARATE_IN_TASK_IMAGE_TOML, verifier=verifier
        )
        response = await run_episode(server, task)
        result = response.result
        assert result is not None, response.failure
        assert result.mask_sample is True and result.failure_kind == kind
        assert fragment in result.failure_reason
        assert result.response is not None and agent.launch() is not None
        assert not any("test.sh" in c["command"] for c in verifier.execs)
        assert verifier.stopped and agent.stopped and not server._sandboxes


SECRET_TOML = (
    TASK_TOML
    + """
[verifier.env]
JUDGE = "${JUDGE_KEY}"
MODE = "strict"
"""
)


class TestGeneralization:
    """M27: secrets, the ATIF transcript, pull retries, task deadlines, forced separate mode, nested tasks."""

    async def test_secrets_reach_the_test_sh_process_only(self, tmp_path, monkeypatch):
        monkeypatch.delenv("JUDGE_KEY", raising=False)
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        server, task, sandbox, _ = make_server(
            tmp_path, monkeypatch, toml=SECRET_TOML, secrets={"JUDGE_KEY": "sk-judge", "OTHER": "o"}
        )
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        verifier_exec = next(call for call in sandbox.execs if "test.sh" in call["command"])
        # Templates in [verifier].env resolve from the secrets block; a secret the task does not name is not
        # passed, so a test.sh that dumps its environment cannot leak a secret set for another taskset.
        assert verifier_exec["env"] == {"JUDGE": "sk-judge", "MODE": "strict"}
        assert "OTHER" not in verifier_exec["env"] and "o" not in verifier_exec["env"].values()
        # Nothing else saw them: the sandbox's own environment, the seed and the agent's turn.
        assert "sk-judge" not in str(server._sandbox_spec(task, "/app").env)
        assert "sk-judge" not in str([call[2] for call in server.server_client.calls])

    async def test_secret_templates_outside_the_verifier_are_not_filled(self, tmp_path, monkeypatch):
        monkeypatch.delenv("JUDGE_KEY", raising=False)
        toml = TASK_TOML + '\n[environment.env]\nTOKEN = "${JUDGE_KEY}"\n'
        server, task, _, _ = make_server(tmp_path, monkeypatch, toml=toml, secrets={"JUDGE_KEY": "sk-judge"})
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest

        response = await run_episode(server, task)

        assert response.failure.stage == "seed" and response.failure.terminal
        assert "JUDGE_KEY" in response.failure.failure_reason

    async def test_failed_trajectory_export_masks_the_turn_before_test_sh(self, tmp_path, monkeypatch):
        from nemo_gym.atif_export import AtifExportError

        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)

        def broken_export(request, agent_response):
            raise AtifExportError("transcript[3]: unsupported transcript item")

        monkeypatch.setattr(server, "_export_trajectory", broken_export)
        response = await run_episode(server, task)

        result = response.result
        assert result.mask_sample is True and result.failure_kind == "harbor:trajectory_export_failed"
        assert "unsupported transcript item" in result.failure_reason
        assert result.verifier_mode is None and result.response.id == "response"
        assert not any("test.sh" in call["command"] for call in sandbox.execs)
        assert sandbox.stopped and not server._sandboxes

    async def test_terminus_shaped_transcript_exports_user_turns_as_observations(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        turn = agent_response().model_dump(mode="json")
        turn["output"] = [
            turn["output"][0],
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello.txt created"}]},
            {
                **turn["output"][0],
                "id": "message-2",
                "content": [{**turn["output"][0]["content"][0], "text": "Done."}],
            },
        ]
        server.server_client.turn_response = turn

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        index = [remote for _, remote in sandbox.uploads].index("/logs/agent/trajectory.json")
        steps = json.loads(sandbox.uploaded_archives[index])["steps"]
        assert [step["source"] for step in steps] == ["user", "agent", "agent"]
        assert steps[1]["observation"]["results"][0]["content"] == "hello.txt created"
        assert steps[1]["extra"]["nemo_gym"]["observation_source"] == "user_turn"

    async def test_sandbox_access_names_the_agent_user(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(
            tmp_path, monkeypatch, toml=TASK_TOML.replace("[agent]\n", '[agent]\nuser = "agent"\n')
        )
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest
        await run_episode(server, task)
        assert server.server_client.seed_bodies()[0]["sandbox_access"]["user"] == "agent"

    async def test_task_deadline_ends_an_overrunning_episode(self, tmp_path, monkeypatch):
        toml = TASK_TOML.replace("timeout_sec = 300.0", "timeout_sec = 0.05").replace(
            "timeout_sec = 120.0", "timeout_sec = 0.05"
        )
        server, task, sandbox, _ = make_server(
            tmp_path,
            monkeypatch,
            toml=toml,
            sandbox_ready_timeout_s=0.01,
            harness_setup_allowance_s=0,
            verifier_grace_s=0,
        )
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest
        assert server._task_budget_s(task) == pytest.approx(0.11)
        # The deadline itself starts when the sandbox is ready and leaves the sandbox start out.
        assert server._post_seed_budget_s(task) == pytest.approx(0.1)
        server.server_client.slow_path = "/ng-rollout/r1/v1/responses"

        response = await run_episode(server, task)

        failure = response.failure
        assert failure.stage == "agent" and failure.failure_kind == "harbor:task_deadline" and not failure.terminal
        assert (
            "time budget of task 'hello'" in failure.failure_reason
            and "enforce_task_timeouts" in failure.failure_reason
        )
        assert not any("test.sh" in call["command"] for call in sandbox.execs)
        # The final unwind still closed the agent session and stopped the sandbox.
        assert server.server_client.paths()[-1] == "/v1/agent_sessions/close"
        assert sandbox.stopped and not server._sandboxes and server._active == set()

        # The same slow harness finishes when the run ignores task timeouts.
        server.config.enforce_task_timeouts = False
        server.server_client.slow_path = "/ng-rollout/r2/v1/responses"
        assert (await run_episode(server, task, rollout="r2")).result.reward == 1.0

    async def test_task_deadline_covers_verification_too(self, tmp_path, monkeypatch):
        toml = TASK_TOML.replace("timeout_sec = 300.0", "timeout_sec = 0.05").replace(
            "timeout_sec = 120.0", "timeout_sec = 0.05"
        )

        @dataclass
        class SlowVerifier(FakeSandbox):
            async def exec(self, command, **kwargs):
                if "test.sh" in command:
                    await asyncio.sleep(0.5)
                return await super().exec(command, **kwargs)

        server, task, sandbox, _ = make_server(
            tmp_path,
            monkeypatch,
            SlowVerifier(),
            toml=toml,
            sandbox_ready_timeout_s=0.01,
            harness_setup_allowance_s=0,
            verifier_grace_s=0,
        )
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest

        response = await run_episode(server, task)

        assert response.failure.stage == "verification" and response.failure.failure_kind == "harbor:task_deadline"
        assert response.failure.partial_response.id == "response"
        assert sandbox.stopped and not server._sandboxes

    def retrying_server(self, tmp_path, monkeypatch, failures):
        """A server whose real `_create_sandbox` runs against a provider that fails `failures` starts in turn."""
        from nemo_gym.sandbox.providers.base import SandboxCreateError

        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, image_pull_retries=2)
        monkeypatch.setattr(server, "_create_sandbox", HarborEnvironmentServer._create_sandbox.__get__(server))
        monkeypatch.setattr("environment_servers.harbor.app.resolve_provider_config", lambda ref, config: {"fake": {}})
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        monkeypatch.setattr("environment_servers.harbor.app._pull_retry_delay", lambda attempt: 0)
        starts: list[str] = []

        class FlakyProvider:
            def __init__(self, provider_config):
                pass

            async def start(self, spec):
                starts.append(spec.image)
                if len(starts) <= len(failures):
                    raise SandboxCreateError(failures[len(starts) - 1])

            def __getattr__(self, name):
                return getattr(sandbox, name)

        monkeypatch.setattr("environment_servers.harbor.app.AsyncSandbox", FlakyProvider)
        return server, task, sandbox, starts

    async def test_image_pull_failures_are_retried_in_the_server(self, tmp_path, monkeypatch):
        server, task, sandbox, starts = self.retrying_server(
            tmp_path, monkeypatch, ["pod failed: ErrImagePull", "create failed: ImagePullBackOff for ubuntu:24.04"]
        )
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert starts == ["ubuntu:24.04"] * 3 and response.result.reward == 1.0

    async def test_exhausted_pull_retries_are_a_retryable_seed_failure(self, tmp_path, monkeypatch):
        server, task, sandbox, starts = self.retrying_server(tmp_path, monkeypatch, ["ErrImagePull"] * 3)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "ErrImagePull" in response.failure.failure_reason and len(starts) == 3
        assert not server._sandboxes

    async def test_seed_time_is_not_charged_against_the_task_deadline(self, tmp_path, monkeypatch):
        """A start that recovered on a pull retry begins the task with its whole budget (review fix)."""
        from nemo_gym.sandbox.providers.base import SandboxCreateError

        toml = TASK_TOML.replace("timeout_sec = 300.0", "timeout_sec = 0.1").replace(
            "timeout_sec = 120.0", "timeout_sec = 0.1"
        )
        server, task, sandbox, _ = make_server(
            tmp_path,
            monkeypatch,
            toml=toml,
            image_pull_retries=2,
            sandbox_ready_timeout_s=0.2,
            harness_setup_allowance_s=0,
            verifier_grace_s=0,
        )
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest
        assert server._task_budget_s(task) == pytest.approx(0.4)
        monkeypatch.setattr(server, "_create_sandbox", HarborEnvironmentServer._create_sandbox.__get__(server))
        monkeypatch.setattr("environment_servers.harbor.app.resolve_provider_config", lambda ref, config: {"fake": {}})
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        monkeypatch.setattr("environment_servers.harbor.app._pull_retry_delay", lambda attempt: 0)
        starts: list[str] = []

        class SlowFlakyProvider:
            def __init__(self, provider_config):
                pass

            async def start(self, spec):
                starts.append(spec.image)
                if len(starts) == 1:
                    # The provider's own create retries: longer than the whole task budget before the pull
                    # failure surfaces.
                    await asyncio.sleep(3 * server.config.sandbox_ready_timeout_s)
                    raise SandboxCreateError("pod failed: ErrImagePull")
                await asyncio.sleep(server.config.sandbox_ready_timeout_s)

            def __getattr__(self, name):
                return getattr(sandbox, name)

        monkeypatch.setattr("environment_servers.harbor.app.AsyncSandbox", SlowFlakyProvider)

        response = await run_episode(server, task)

        assert starts == ["ubuntu:24.04"] * 2
        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and sandbox.stopped and not server._sandboxes

    async def test_other_start_failures_are_not_retried(self, tmp_path, monkeypatch):
        server, task, sandbox, starts = self.retrying_server(tmp_path, monkeypatch, ["quota exceeded for namespace"])
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and not response.failure.terminal
        assert len(starts) == 1 and not server._sandboxes

    async def test_compose_group_pull_failures_are_retried_in_the_server(self, tmp_path, monkeypatch):
        from nemo_gym.sandbox.providers.base import SandboxCreateError

        server, task, _, _ = make_server(tmp_path, monkeypatch, image_pull_retries=2)
        (task.path / "task.toml").write_text(COMPOSE_TOML)
        (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (task.path.parent / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest
        monkeypatch.setattr("environment_servers.harbor.app.resolve_provider_config", lambda ref, config: {"fake": {}})
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        monkeypatch.setattr("environment_servers.harbor.app._pull_retry_delay", lambda attempt: 0)
        groups: list = []
        verifier = FakeSandbox()

        class FlakyComposeGroup:
            def __init__(self, provider_config, path, *, service_specs, timeout_s):
                groups.append(self)
                self.services: dict = {}
                self.stopped = False

            async def start(self):
                if len(groups) == 1:
                    # The first group fails on a sidecar's pull, half-started: it is stopped whole.
                    self.services = {"main": HealthySandbox()}
                    raise SandboxCreateError("create failed: ImagePullBackOff for org/kafka:1")
                self.services = {"main": HealthySandbox(), "api": FakeSandbox(), "kafka": FakeSandbox()}

            async def stop(self):
                self.stopped = True

        async def create_verifier(task, *, episode_key=None):
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr("environment_servers.harbor.app.AsyncSandboxCompose", FlakyComposeGroup)
        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert len(groups) == 2 and groups[0].stopped and groups[1].stopped
        assert response.result.reward == 1.0 and not server._sandboxes

    async def test_exhausted_compose_pull_retries_are_a_retryable_seed_failure(self, tmp_path, monkeypatch):
        from nemo_gym.sandbox.providers.base import SandboxCreateError

        server, task, _, _ = make_server(tmp_path, monkeypatch, image_pull_retries=1)
        (task.path / "task.toml").write_text(COMPOSE_TOML)
        (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (task.path.parent / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest
        monkeypatch.setattr("environment_servers.harbor.app.resolve_provider_config", lambda ref, config: {"fake": {}})
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        monkeypatch.setattr("environment_servers.harbor.app._pull_retry_delay", lambda attempt: 0)
        starts = []

        class FailingComposeGroup:
            services: dict = {}

            def __init__(self, provider_config, path, *, service_specs, timeout_s):
                pass

            async def start(self):
                starts.append(1)
                raise SandboxCreateError("ErrImagePull")

            async def stop(self):
                pass

        monkeypatch.setattr("environment_servers.harbor.app.AsyncSandboxCompose", FailingComposeGroup)
        response = await run_episode(server, task)
        assert response.failure.stage == "seed" and not response.failure.terminal
        assert "ErrImagePull" in response.failure.failure_reason and len(starts) == 2
        assert not server._sandboxes

    def test_pull_failure_is_found_inside_an_exception_group(self):
        """A Compose group whose cleanup failed too raises an ExceptionGroup; the pull failure is still found."""
        from nemo_gym.sandbox.providers.base import SandboxCreateError

        pull = SandboxCreateError("create failed: ImagePullBackOff for org/kafka:1")
        cleanup = RuntimeError("Shared volume cleanup failed: no space left")
        assert _is_image_pull_failure(ExceptionGroup("Compose cleanup failed", [cleanup, pull]))
        assert _is_image_pull_failure(ExceptionGroup("outer", [ExceptionGroup("inner", [pull])]))
        # The live shape: the group of cleanup errors is raised while the pull failure is being handled.
        try:
            try:
                raise pull
            except SandboxCreateError:
                raise ExceptionGroup("Compose cleanup failed", [cleanup])
        except ExceptionGroup as group:
            assert group.__context__ is pull and _is_image_pull_failure(group)
        assert not _is_image_pull_failure(ExceptionGroup("Compose cleanup failed", [cleanup]))
        assert not _is_image_pull_failure(ExceptionGroup("x", [RuntimeError("quota exceeded")]))

    async def test_compose_pull_failure_whose_cleanup_also_failed_is_retried(self, tmp_path, monkeypatch):
        from nemo_gym.sandbox.providers.base import SandboxCreateError

        server, task, _, _ = make_server(tmp_path, monkeypatch, image_pull_retries=1)
        (task.path / "task.toml").write_text(COMPOSE_TOML)
        (task.path / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (task.path.parent / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest
        monkeypatch.setattr("environment_servers.harbor.app.resolve_provider_config", lambda ref, config: {"fake": {}})
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        monkeypatch.setattr("environment_servers.harbor.app._pull_retry_delay", lambda attempt: 0)
        groups: list = []
        verifier = FakeSandbox()

        class FlakyComposeGroup:
            def __init__(self, provider_config, path, *, service_specs, timeout_s):
                groups.append(self)
                self.services: dict = {}
                self.stopped = 0

            async def start(self):
                if len(groups) == 1:
                    # The group's own stop() after the failed pull failed too: the cleanup errors surface as a
                    # group over the pull failure (docker_compose.py start/stop), not as the pull failure itself.
                    raise ExceptionGroup(
                        "Compose cleanup failed",
                        [RuntimeError("Shared volume cleanup failed"), SandboxCreateError("ErrImagePull")],
                    )
                self.services = {"main": HealthySandbox(), "api": FakeSandbox(), "kafka": FakeSandbox()}

            async def stop(self):
                self.stopped += 1

        async def create_verifier(task, *, episode_key=None):
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr("environment_servers.harbor.app.AsyncSandboxCompose", FlakyComposeGroup)
        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert len(groups) == 2 and [g.stopped for g in groups] == [1, 1]
        assert response.result.reward == 1.0 and not server._sandboxes

    def verifier_retrying_server(self, tmp_path, monkeypatch, failures):
        """A separate-mode server whose real `_create_verifier_sandbox` runs against a provider that fails
        `failures` verifier starts in turn; the agent's sandbox is the usual fake."""
        from nemo_gym.sandbox.providers.base import SandboxCreateError

        agent = AgentSandbox()
        server, task, _, _ = make_server(tmp_path, monkeypatch, agent, image_pull_retries=2)
        (task.path / "task.toml").write_text(SEPARATE_TOML)
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest
        monkeypatch.setattr("environment_servers.harbor.app.resolve_provider_config", lambda ref, config: {"fake": {}})
        monkeypatch.setattr(
            "environment_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        monkeypatch.setattr("environment_servers.harbor.app._pull_retry_delay", lambda attempt: 0)
        verifier = FakeSandbox()
        starts: list[str] = []
        gate_values: list[int] = []
        providers: list = []

        class FlakyProvider:
            def __init__(self, provider_config):
                providers.append(self)
                self.stopped = 0

            async def start(self, spec):
                starts.append(spec.image)
                gate_values.append(server._pull_retry_gate._value)
                # Owned under the verifier key from before start(), so a half-started one is stopped whole.
                assert server._sandboxes.get(f"{next(iter(server._sessions))}/verifier") is self
                if len(starts) <= len(failures):
                    raise SandboxCreateError(failures[len(starts) - 1])

            async def stop(self):
                self.stopped += 1
                verifier.stopped = True

            def __getattr__(self, name):
                return getattr(verifier, name)

        monkeypatch.setattr("environment_servers.harbor.app.AsyncSandbox", FlakyProvider)
        return server, task, starts, gate_values, providers

    async def test_verifier_sandbox_pull_failures_are_retried_in_the_server(self, tmp_path, monkeypatch):
        server, task, starts, gates, providers = self.verifier_retrying_server(
            tmp_path, monkeypatch, ["pod failed: ErrImagePull"]
        )
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        assert response.result.reward == 1.0 and response.result.verifier_mode == "separate"
        # The first start is not gated; the retry queues behind the per-server gate.
        assert starts == ["org/verifier:1"] * 2 and gates == [4, 3]
        assert [p.stopped for p in providers] == [1, 1]
        assert not server._sandboxes and server._pull_retry_gate._value == 4

    async def test_exhausted_verifier_pull_retries_mask_and_leave_nothing_owned(self, tmp_path, monkeypatch):
        server, task, starts, gates, providers = self.verifier_retrying_server(
            tmp_path, monkeypatch, ["ErrImagePull"] * 3
        )
        response = await run_episode(server, task)
        assert response.failure is None, response.failure
        result = response.result
        assert result.mask_sample is True and result.failure_kind == "provider_unavailable"
        assert "ErrImagePull" in result.failure_reason
        assert len(starts) == 3 and gates == [4, 3, 3]
        # The last provider (not stopped by the helper) is released by the episode's own close.
        assert [p.stopped for p in providers] == [1, 1, 1]
        assert not server._sandboxes and server._pull_retry_gate._value == 4

    async def test_verifier_sandbox_other_start_failures_are_not_retried(self, tmp_path, monkeypatch):
        server, task, starts, _, _ = self.verifier_retrying_server(
            tmp_path, monkeypatch, ["quota exceeded for namespace"]
        )
        response = await run_episode(server, task)
        result = response.result
        assert result.mask_sample is True and result.failure_kind == "provider_unavailable"
        assert len(starts) == 1 and not server._sandboxes and server._pull_retry_gate._value == 4

    def test_nested_compose_task_reads_image_configs_from_the_dataset_root(self, tmp_path, monkeypatch):
        """prepare records compose-images.json at the dataset root; the snapshot of a nested task finds it there."""
        server, _, _, _ = make_server(tmp_path, monkeypatch)
        root = tmp_path / "datasets" / "ds"
        web = write_task(root / "00-services" / "web", toml=COMPOSE_TOML)
        (web / "environment" / "docker-compose.yaml").write_text(COMPOSE_YAML)
        (root / "compose-images.json").write_text(json.dumps(IMAGE_CONFIGS))
        task = load_task(web, root=root)
        assert task.needs_compose

        snapshot = server._snapshot_task(web, task.digest, root)

        assert snapshot.path.parent != web.parent
        assert server._compose_image_configs(snapshot) == IMAGE_CONFIGS

    async def test_turn_cut_off_inside_reasoning_is_verified_not_masked(self, tmp_path, monkeypatch):
        """The live Terminus shape after a reasoning-only max_output_tokens response: exported and graded."""
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        turn = agent_response().model_dump(mode="json")
        answer = turn["output"][0]

        def reasoning(item_id, text):
            return {"type": "reasoning", "id": item_id, "summary": [{"type": "summary_text", "text": text}]}

        def observed(text):
            return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}

        turn["output"] = [
            reasoning("r1", "plan"),
            answer,
            observed("hello.txt created"),
            reasoning("r2", "half a thou"),
            observed("(no command)"),
            reasoning("r3", "finish"),
            {**answer, "id": "message-2", "content": [{**answer["content"][0], "text": "Done."}]},
        ]
        server.server_client.turn_response = turn

        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        result = response.result
        assert result.mask_sample is False and result.failure_kind is None and result.reward == 1.0
        assert any("test.sh" in call["command"] for call in sandbox.execs)
        index = [remote for _, remote in sandbox.uploads].index("/logs/agent/trajectory.json")
        trajectory = json.loads(sandbox.uploaded_archives[index])
        steps = trajectory["steps"]
        assert [step["source"] for step in steps] == ["user", "agent", "agent", "agent"]
        cut = steps[2]
        assert cut["message"] == "" and cut["reasoning_content"] == "half a thou"
        assert cut["extra"]["nemo_gym"] == {"incomplete": "max_output_tokens", "observation_source": "user_turn"}
        assert cut["observation"]["results"][0]["content"] == "(no command)"
        assert trajectory["extra"]["nemo_gym"]["conversion"]["reasons"] == ["trailing_reasoning"]

    def test_runs_can_force_separate_mode_for_tasks_with_artifacts(self, tmp_path, monkeypatch):
        with_artifacts = TASK_TOML.replace("[verifier]\n", 'artifacts = ["/app/out.txt"]\n\n[verifier]\n', 1)
        server, task, _, _ = make_server(
            tmp_path, monkeypatch, toml=with_artifacts, separate_verifier_for_artifacts=True
        )
        task = load_task(task.path)
        assert task.config.is_shared_verifier
        assert server._separate_verifier(task)
        # Module helpers follow the forced mode: the verifier runs in the task's own image.
        assert _verifier_image(task, True) == task.image == "ubuntu:24.04" and _verifier_image(task) is None
        # The budget and the sandbox lifetime count the second sandbox start.
        plain_server, plain_task, _, _ = make_server(tmp_path / "plain", monkeypatch, toml=with_artifacts)
        assert (
            server._task_budget_s(task)
            == plain_server._task_budget_s(plain_task) + server.config.sandbox_ready_timeout_s
        )
        # A task without artifacts keeps Harbor's shared mode.
        _, bare, _, _ = make_server(tmp_path / "bare", monkeypatch, separate_verifier_for_artifacts=True)
        assert not server._separate_verifier(bare)

    async def test_forced_separate_mode_verifies_in_a_fresh_sandbox(self, tmp_path, monkeypatch):
        agent = AgentSandbox()
        server, task, _, _ = make_server(
            tmp_path,
            monkeypatch,
            agent,
            toml=TASK_TOML.replace("[verifier]\n", 'artifacts = ["/app/output/report.json"]\n\n[verifier]\n', 1),
            separate_verifier_for_artifacts=True,
        )
        server.config.tasksets["ds"].tasks["hello"] = (task := load_task(task.path)).digest
        verifier = FakeSandbox()

        async def create_verifier(task, *, episode_key=None):
            server._sandboxes[f"{episode_key}/verifier"] = verifier
            return verifier

        monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
        response = await run_episode(server, task)

        assert response.failure is None, response.failure
        assert response.result.verifier_mode == "separate" and response.result.reward == 1.0
        assert any("test.sh" in call["command"] for call in verifier.execs)
        assert not any("test.sh" in call["command"] for call in agent.execs)
        assert [remote for _, remote in verifier.uploads if remote.endswith("trajectory.json")] == [
            "/logs/agent/trajectory.json"
        ]
        assert agent.stopped and verifier.stopped and not server._sandboxes

    async def test_nested_tasks_resolve_through_the_mapping_path_and_the_dataset_root(self, tmp_path, monkeypatch):
        server, _, _, _ = make_server(tmp_path, monkeypatch)
        root = tmp_path / "datasets" / "ds"
        (root / "dataset.toml").write_text("[gym.defaults]\ntimeout_multiplier = 2.0\n")
        deep = write_task(root / "group" / "deep")
        # The base-image record sits at the dataset root, where prepare writes it, not beside the nested task.
        (root / "group" / IMAGE_CONFIGS_FILE).rename(tmp_path / "unused.json")
        task = load_task(deep, root=root)
        assert task.config.agent.timeout_sec == 600.0
        server.config.tasksets["ds"].tasks["deep"] = task.digest
        server.config.tasksets["ds"].paths["deep"] = "group/deep"

        response = await run_episode(server, task, task_id="deep")

        assert response.failure is None, response.failure
        assert server.server_client.seed_bodies()[0]["agent_timeout_sec"] == 600.0
        # A mapping path that escapes the taskset folder is refused before anything is provisioned.
        server.config.tasksets["ds"].paths["deep"] = "../deep"
        escaped = await run_episode(server, task, task_id="deep", rollout="r2")
        assert (
            escaped.failure.stage == "seed"
            and escaped.failure.terminal
            and "invalid path" in escaped.failure.failure_reason
        )
