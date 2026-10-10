# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml
from fastapi.testclient import TestClient
from harbor.models.task.task import Task
from pytest import MonkeyPatch

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import ResourcesSeedSessionRequest, ResourcesSeedSessionResponse
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.server_utils import ServerClient
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.harbor_tasks import app as harbor_app
from resources_servers.harbor_tasks.app import HarborTasksResourcesServer, HarborTasksResourcesServerConfig
from resources_servers.harbor_tasks.tasks import image_reference, unsupported_features


EXAMPLE_TASKS = Path(__file__).parents[1] / "data" / "tasks"
TASK_TOML = """version = "1.0"

[agent]
timeout_sec = 300.0
{agent_extra}

[verifier]
timeout_sec = 60.0
{verifier_extra}

[environment]
{environment}
"""


def write_task(root: Path, name: str, *, environment: str = 'docker_image = "python:3.12-slim"', **extra) -> Path:
    task_dir = root / name
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "environment").mkdir()
    (task_dir / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task_dir / "instruction.md").write_text(f"Solve {name}.\n")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    (task_dir / "task.toml").write_text(
        TASK_TOML.format(
            agent_extra=extra.get("agent_extra", ""),
            verifier_extra=extra.get("verifier_extra", ""),
            environment=environment,
        )
    )
    return task_dir


def make_server(tmp_path: Path, **overrides) -> HarborTasksResourcesServer:
    config = HarborTasksResourcesServerConfig(
        **{
            "host": "",
            "port": 0,
            "entrypoint": "",
            "name": "harbor_tasks",
            "harbor_datasets": {"example": {"path": str(EXAMPLE_TASKS)}},
            "sandbox_provider": "sandbox",
            "artifacts_dir": tmp_path / "artifacts",
        }
        | overrides
    )
    return HarborTasksResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def seed_request(
    session_id: str = "resources-session-1", task_name: str = "hello-world"
) -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id=f"rollout-{session_id}"),
        task_id=TaskId(taskset="harbor_tasks", task_id=task_name),
        task_data={"harbor_dataset": "example", "task_name": task_name},
    )


class FakeSandbox:
    """Records sandbox lifecycle calls; every command succeeds."""

    instances: list["FakeSandbox"] = []

    def __init__(self, provider: object) -> None:
        self.started_spec = None
        self.stopped = False
        self.commands: list[str] = []
        FakeSandbox.instances.append(self)

    async def start(self, spec):
        self.started_spec = spec
        return self

    async def exec(self, command: str, **kwargs):
        self.commands.append(command)
        return SimpleNamespace(stdout="/app\n", stderr="", return_code=0)

    async def serialize(self) -> dict:
        return {"sandbox_id": f"fake-{len(FakeSandbox.instances)}"}

    async def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def fake_sandbox(monkeypatch: MonkeyPatch) -> type[FakeSandbox]:
    FakeSandbox.instances = []
    monkeypatch.setattr(harbor_app, "AsyncSandbox", FakeSandbox)
    monkeypatch.setattr(harbor_app, "create_provider", lambda config: object())
    monkeypatch.setattr(harbor_app, "resolve_provider_config", lambda ref, global_config: {"docker": {}})
    monkeypatch.setattr(harbor_app, "resolve_provider_metadata", lambda ref, global_config: {})
    monkeypatch.setattr(harbor_app, "get_global_config_dict", lambda: {})
    return FakeSandbox


def test_relative_config_paths_resolve_against_the_repository(tmp_path: Path) -> None:
    config = HarborTasksResourcesServerConfig(
        host="",
        port=0,
        entrypoint="",
        name="harbor_tasks",
        harbor_datasets={"example": {"path": "resources_servers/harbor_tasks/data/tasks"}},
        sandbox_provider="sandbox",
        artifacts_dir="results/harbor_tasks",
    )
    assert config.harbor_datasets["example"].path == PARENT_DIR / "resources_servers/harbor_tasks/data/tasks"
    assert config.artifacts_dir == PARENT_DIR / "results/harbor_tasks"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, []),
        ({"environment": 'docker_image = "x"\nnetwork_mode = "no-network"'}, ["restricted network policies"]),
        ({"verifier_extra": 'environment_mode = "separate"'}, ["separate verifier environments"]),
        ({"environment": "cpus = 1"}, ["Dockerfile environments without an image_template"]),
        (
            {
                "environment": 'docker_image = "x"\n[[environment.mcp_servers]]\nname = "kb"\ntransport = "streamable-http"\nurl = "http://kb/mcp"'
            },
            ["task-declared MCP servers"],
        ),
    ],
)
def test_unsupported_features_names_what_the_server_cannot_reproduce(
    tmp_path: Path, kwargs: dict, expected: list[str]
) -> None:
    task = Task(write_task(tmp_path, "task", **kwargs))
    found = unsupported_features(task)
    assert len(found) == len(expected)
    for prefix, message in zip(expected, found):
        assert message.startswith(prefix)


def test_unenforced_network_policy_can_be_allowed(tmp_path: Path) -> None:
    task = Task(write_task(tmp_path, "task", environment='docker_image = "x"\nnetwork_mode = "no-network"'))
    assert unsupported_features(task, allow_unenforced_network_policy=True) == []


def test_compose_tasks_are_unsupported(tmp_path: Path) -> None:
    task_dir = write_task(tmp_path, "task")
    (task_dir / "environment" / "docker-compose.yaml").write_text("services: {}\n")
    assert unsupported_features(Task(task_dir)) == ["docker-compose environments"]


def test_task_without_image_or_dockerfile_is_unsupported(tmp_path: Path) -> None:
    task_dir = write_task(tmp_path, "task", environment="cpus = 1")
    (task_dir / "environment" / "Dockerfile").unlink()
    (task_dir / "environment" / "data.txt").write_text("x")
    assert unsupported_features(Task(task_dir), image_template="env:{environment_hash}") == [
        "environments with neither a docker_image nor a Dockerfile"
    ]


def test_image_reference_prefers_the_prebuilt_image_then_tags_the_environment_by_content(tmp_path: Path) -> None:
    prebuilt = Task(write_task(tmp_path, "prebuilt"))
    assert image_reference(prebuilt, "env:{environment_hash}") == "python:3.12-slim"

    first = Task(write_task(tmp_path, "first", environment="cpus = 1"))
    same = Task(write_task(tmp_path, "same", environment="cpus = 2"))
    other_dir = write_task(tmp_path, "other", environment="cpus = 1")
    (other_dir / "environment" / "Dockerfile").write_text("FROM python:3.13-slim\n")
    other = Task(other_dir)
    assert image_reference(first, None) is None
    tag = image_reference(first, "env:{environment_hash}")
    assert tag.startswith("env:") and len(tag) == len("env:") + 32
    # Identical environment/ directories share one image; any change to them is a different image.
    assert image_reference(same, "env:{environment_hash}") == tag
    assert image_reference(other, "env:{environment_hash}") != tag
    assert unsupported_features(first, image_template="env:{environment_hash}") == []


def test_reward_uses_the_configured_key_then_the_first_value(tmp_path: Path) -> None:
    server = make_server(tmp_path, reward_key="score")
    assert server._reward({"other": 0.25, "score": 1}, task_name="t") == 1.0
    assert server._reward({"other": 0.25}, task_name="t") == 0.25
    assert server._reward({}, task_name="t") == 0.0


def test_session_contract(tmp_path: Path, fake_sandbox: type[FakeSandbox]) -> None:
    server = make_server(tmp_path)
    check_resources_session_contract(server.setup_webserver(), seed_request(), keeps_state=True)
    # One sandbox for the seed and its idempotent repeat, stopped by close.
    assert len(fake_sandbox.instances) == 1
    assert fake_sandbox.instances[0].stopped


def test_seed_starts_the_task_image_and_returns_direct_access(tmp_path: Path, fake_sandbox: type[FakeSandbox]) -> None:
    server = make_server(tmp_path, image_rewrites=[{"from": "python:", "to": "mirror/python:"}])
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/seed_session", json=seed_request().model_dump(mode="json"))
    assert response.status_code == 200, response.text
    seeded = ResourcesSeedSessionResponse.model_validate(response.json())
    assert seeded.sandbox_access.workdir == "/app"
    assert seeded.sandbox_access.connection.provider_config_ref == "sandbox"
    assert seeded.sandbox_access.connection.descriptor == {"sandbox_id": "fake-1"}
    spec = fake_sandbox.instances[0].started_spec
    assert spec.image == "mirror/python:3.12-slim"
    assert spec.workdir == "/app"
    assert spec.resources.cpu == 1.0 and spec.resources.memory_mib == 1024
    assert any("/logs/verifier" in command for command in fake_sandbox.instances[0].commands)
    # Server shutdown stops a sandbox that no episode closed.
    assert fake_sandbox.instances[0].stopped


def test_seed_of_an_unknown_task_fails_without_starting_a_sandbox(
    tmp_path: Path, fake_sandbox: type[FakeSandbox]
) -> None:
    server = make_server(tmp_path)
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    response = client.post("/seed_session", json=seed_request(task_name="missing").model_dump(mode="json"))
    assert response.status_code == 500
    assert fake_sandbox.instances == []


def test_failed_seed_stops_the_sandbox(tmp_path: Path, fake_sandbox: type[FakeSandbox], monkeypatch) -> None:
    server = make_server(tmp_path)

    async def fail(self, session) -> None:
        raise RuntimeError("staging failed")

    monkeypatch.setattr(HarborTasksResourcesServer, "prepare_sandbox", fail)
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/seed_session", json=seed_request().model_dump(mode="json")).status_code == 500
    assert fake_sandbox.instances[0].stopped
    assert server._sessions == {}


def test_seed_of_an_unbuilt_environment_says_how_to_build_it(tmp_path: Path, fake_sandbox, monkeypatch) -> None:
    write_task(tmp_path / "tasks", "built", environment="cpus = 1")

    async def missing_image(self, spec):
        raise RuntimeError(f"image {spec.image} not found")

    monkeypatch.setattr(FakeSandbox, "start", missing_image)
    server = make_server(
        tmp_path,
        harbor_datasets={"local": {"path": str(tmp_path / "tasks")}},
        image_template="env:{environment_hash}",
    )
    seed = seed_request(task_name="built").model_copy(
        update={"task_data": {"harbor_dataset": "local", "task_name": "built"}}
    )
    with pytest.raises(RuntimeError, match="build_images.py"):
        TestClient(server.setup_webserver()).post("/seed_session", json=seed.model_dump(mode="json"))
    assert fake_sandbox.instances[0].stopped


def test_verify_requires_a_seeded_session(tmp_path: Path) -> None:
    server = make_server(tmp_path)
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    body = {
        "harbor_dataset": "example",
        "task_name": "hello-world",
        "responses_create_params": {"input": []},
        "response": _empty_response(),
    }
    assert client.post("/verify", json=body).status_code == 500


def test_compute_metrics_reports_pass_at_k_and_skips_unscored_rows(tmp_path: Path) -> None:
    server = make_server(tmp_path)
    metrics = server.compute_metrics([[{"reward": 1.0}, {"reward": 0.0}], [{"reward": 0.0}, {"reward": 0.0}]])
    assert metrics["pass@1[avg-of-2]/accuracy"] == pytest.approx(25.0)
    assert metrics["pass@2/accuracy"] == pytest.approx(50.0)


def _empty_response() -> dict:
    return {
        "id": "r",
        "created_at": 0,
        "model": "m",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _docker_available(), reason="requires a Docker daemon")
@pytest.mark.parametrize("solve", [True, False])
def test_docker_episode_seeds_verifies_and_closes(tmp_path: Path, monkeypatch: MonkeyPatch, solve: bool) -> None:
    """The real Harbor verifier grades the sandbox the agent changed; here a solution script stands in for it."""
    docker_config = yaml.safe_load((PARENT_DIR / "nemo_gym/sandbox/providers/docker/configs/docker.yaml").read_text())
    monkeypatch.setattr(harbor_app, "get_global_config_dict", lambda: docker_config)
    server = make_server(tmp_path)
    seed = seed_request(session_id=f"docker-{solve}")
    with TestClient(server.setup_webserver()) as client:
        seeded = client.post("/seed_session", json=seed.model_dump(mode="json"))
        assert seeded.status_code == 200, seeded.text
        if solve:
            container = seeded.json()["sandbox_access"]["connection"]["descriptor"]["sandbox_id"]
            solution = (EXAMPLE_TASKS / "hello-world" / "solution" / "solve.sh").read_text()
            subprocess.run(["docker", "exec", container, "bash", "-c", solution], check=True)
        verified = client.post(
            "/verify",
            json={
                "harbor_dataset": "example",
                "task_name": "hello-world",
                "responses_create_params": {"input": []},
                "response": _empty_response(),
            },
        )
        assert verified.status_code == 200, verified.text
        result = verified.json()
        assert result["evaluation_completed"] is True
        assert result["reward"] == (1.0 if solve else 0.0)
        assert result["reward_components"] == {"reward": result["reward"]}
        assert (Path(result["trial_dir"]) / "verifier" / "reward.txt").is_file()
        closed = client.post(
            "/close_session",
            json={"resources_session_id": seed.resources_session_id, "episode_id": seed.episode_id.model_dump()},
        )
        assert closed.status_code == 200, closed.text
    assert server._sessions == {}
