# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from nemo_gym.tasks.harbor import DIGEST_KEY, load_task
from nemo_gym.tasks.harbor.models import HarborEnvironment
from resources_servers.harbor.app import (
    HarborResourcesServer,
    HarborResourcesServerConfig,
    _verifier_image,
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


def write_task(root: Path, *, dockerfile: str = "FROM ubuntu:24.04\nWORKDIR /app") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "task.toml").write_text(TASK_TOML)
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
    stopped: bool = False

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.execs.append({"command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s, "user": user})
        if "test.sh" in command:
            return self.test_result
        return SandboxExecResult(stdout="", stderr="", return_code=0)

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
):
    folder = tmp_path / "datasets" / "ds"
    task = load_task(write_task(folder / "hello", dockerfile=dockerfile))
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


class TestSetupCommands:
    def test_setup_commands_run_as_root_after_workdir(self, tmp_path, monkeypatch):
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
        server.config.sandbox_setup_commands = ["sed -i s/a/b/ /etc/apt/sources.list", "apt-get update || true"]
        response = TestClient(server.setup_webserver()).post("/seed_session", json=seed_body(task))
        assert response.status_code == 200, response.text
        commands = [call["command"] for call in sandbox.execs]
        assert commands[0] == "mkdir -p /app"
        assert commands[1:3] == server.config.sandbox_setup_commands
        assert all(call["user"] == "root" for call in sandbox.execs[1:3])

    def test_failing_setup_command_is_a_retryable_seed_failure(self, tmp_path, monkeypatch):
        class Failing(FakeSandbox):
            async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
                if command == "false":
                    return SandboxExecResult(stdout="", stderr="nope", return_code=1)
                return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

        sandbox = Failing()
        server, task, sandbox, _ = make_server(tmp_path, monkeypatch, sandbox)
        server.config.sandbox_setup_commands = ["false"]
        response = TestClient(server.setup_webserver()).post("/seed_session", json=seed_body(task))
        assert response.status_code == 503
        assert "nope" in response.json()["detail"] and sandbox.stopped


class TestSeedWorkdirAndResources:
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

    def test_resources_override_and_cpu_env(self, tmp_path, monkeypatch):
        server, task, _, _ = make_server(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "resources_servers.harbor.app.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
        )
        spec = server._sandbox_spec(task, "/app")
        assert spec.resources.cpu == 1.0 and spec.env["OMP_NUM_THREADS"] == "1"

        server.config.sandbox_resources_override = {"cpu": 4, "memory_mib": 16384, "disk_gib": 30}
        spec = server._sandbox_spec(task, "/app")
        assert (spec.resources.cpu, spec.resources.memory_mib, spec.resources.disk_gib) == (4.0, 16384, 30)
        assert spec.env["OMP_NUM_THREADS"] == "4"

        # The override merges over the task's resources, so a GPU request survives a CPU/memory override.
        task.config.environment.gpus = 1
        task.config.environment.gpu_types = ["H100"]
        spec = server._sandbox_spec(task, "/app")
        assert spec.resources.gpu == 1 and spec.resources.cpu == 4.0
        task.config.environment.gpus = None
        task.config.environment.gpu_types = None

        server.config.derive_cpu_env = False
        assert "OMP_NUM_THREADS" not in server._sandbox_spec(task, "/app").env


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
