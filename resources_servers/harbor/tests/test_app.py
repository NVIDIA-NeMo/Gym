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
from resources_servers.harbor.app import (
    HarborResourcesServer,
    HarborResourcesServerConfig,
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


def make_server(tmp_path: Path, monkeypatch: MonkeyPatch, sandbox: FakeSandbox | None = None):
    folder = tmp_path / "datasets" / "ds"
    task = load_task(write_task(folder / "hello"))
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


def verify_body(*, rollout_id="r1", task_id="hello"):
    return {
        "episode_id": {"rollout_id": rollout_id, "attempt": 0},
        "task_id": {"taskset": "ds", "task_id": task_id},
        "verification_input": {
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
        },
    }


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
        assert payload["responses_create_params"]["input"][0]["content"] == "Create hello.txt"

        run = next(call for call in sandbox.execs if "test.sh" in call["command"])
        assert run["command"] == "bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1"
        assert run["cwd"] == "/app" and run["timeout_s"] == 120.0
        # tests/ was uploaded as an archive and unpacked into /tests.
        assert any(remote.endswith(".tar.gz") for _, remote in sandbox.uploads)
        assert any("tar -xzf" in call["command"] and "/tests" in call["command"] for call in sandbox.execs)
        assert (Path(payload["verifier_logs_dir"]) / "reward.txt").read_text() == "1\n"

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

    def test_verifier_timeout_scores_zero(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(
            test_result=SandboxExecResult(stdout=None, stderr="timed out", return_code=125, error_type="timeout")
        )
        _, _, _, client = self.seeded_client(tmp_path, monkeypatch, sandbox)
        payload = client.post("/verify", json=verify_body()).json()
        assert payload["reward"] == 0.0
        assert payload["mask_sample"] is False
        assert payload["failure_kind"] == "harbor:verifier_timeout"

    def test_sandbox_runtime_failure_masks(self, tmp_path, monkeypatch):
        sandbox = FakeSandbox(
            test_result=SandboxExecResult(stdout=None, stderr="gone", return_code=125, error_type="sandbox")
        )
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
        assert client.post("/verify", json=verify_body(rollout_id="other")).status_code == 409


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
