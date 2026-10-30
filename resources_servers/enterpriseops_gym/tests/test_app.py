# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The resources server's session lifecycle, tool routes and verification, against in-process fake gym servers."""

import hashlib
import sqlite3
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, TaskId
from nemo_gym.sandbox import AsyncSandbox, SandboxEndpoint
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.enterpriseops_gym import app as app_module
from resources_servers.enterpriseops_gym.app import (
    EnterpriseOpsGymResourcesServer,
    EnterpriseOpsGymResourcesServerConfig,
    ensure_seed_archive,
)
from resources_servers.enterpriseops_gym.gym_client import GymRequestError, GymUnavailableError, ToolCallResult


DUMPS = {
    "dbs/a.sql": "CREATE TABLE notes(text TEXT); INSERT INTO notes VALUES ('seeded');",
    "dbs/b.sql": "CREATE TABLE tasks(text TEXT);",
}
GYM_IMAGES = {
    "gym-a": {"image": "registry.test/gym-a@sha256:aa", "command": ["serve", "a"], "port": 8003},
    "gym-b": {"image": "registry.test/gym-b@sha256:bb", "command": ["serve", "b"], "port": 8005},
}


class FakeGym:
    """A gym server: one SQLite database per seeded ID, an ``insert_<table>`` tool per table, and a failing tool."""

    def __init__(self) -> None:
        self.databases: dict[str, sqlite3.Connection] = {}
        self.tool_calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []
        self.healthy = True
        self.reachable = True
        self.refuse_seed = False


class FakeGymClient:
    """Stands in for GymServerClient, serving the FakeGym behind the endpoint it is given."""

    gyms: dict[str, FakeGym] = {}

    def __init__(self, endpoint: SandboxEndpoint, *, verify_tls: bool, timeout_s: float) -> None:
        self.gym = self.gyms[endpoint.endpoint]

    async def wait_until_healthy(self, timeout_s: float) -> None:
        if not self.gym.healthy:
            raise GymUnavailableError("never became healthy")

    async def seed_database(self, database_id: str, sql: str) -> None:
        if self.gym.refuse_seed:
            raise GymRequestError(400, "bad dump")
        database = sqlite3.connect(":memory:", check_same_thread=False)
        database.executescript(sql)
        self.gym.databases[database_id] = database

    async def call_tool(self, name: str, arguments: dict[str, Any], *, headers: dict[str, str]) -> ToolCallResult:
        if not self.gym.reachable:
            raise GymUnavailableError("connection refused")
        self.gym.tool_calls.append((name, arguments, headers))
        if name == "fail_tool":
            return ToolCallResult(text="❌ Invalid Tool Arguments: ['x: is required']", is_error=True)
        table = name.removeprefix("insert_")
        self.gym.databases[headers["x-database-id"]].execute(f"INSERT INTO {table} VALUES (?)", (arguments["text"],))
        return ToolCallResult(text=f'{{"inserted": "{arguments["text"]}"}}', is_error=False)

    async def run_sql(self, query: str, *, headers: dict[str, str]) -> dict[str, Any]:
        if not self.gym.reachable:
            raise GymUnavailableError("connection refused")
        try:
            cursor = self.gym.databases[headers["x-database-id"]].execute(query)
        except sqlite3.Error as error:
            raise GymRequestError(400, f"SQL execution error: {error}") from error
        columns = [column[0] for column in cursor.description]
        return {"success": True, "data": [dict(zip(columns, row)) for row in cursor.fetchall()]}


@pytest.fixture
def gyms(monkeypatch) -> dict[str, FakeGym]:
    gyms = {"gym-a": FakeGym(), "gym-b": FakeGym()}
    monkeypatch.setattr(FakeGymClient, "gyms", {f"http://{name}.test": gym for name, gym in gyms.items()})
    monkeypatch.setattr(app_module, "GymServerClient", FakeGymClient)
    return gyms


@pytest.fixture
def sandboxes(monkeypatch) -> dict[str, list]:
    """The specs of the sandboxes started and the gym servers of those stopped, in order."""
    record: dict[str, list] = {"started": [], "stopped": [], "failing_stops": []}

    async def start(self: AsyncSandbox, spec):
        if spec.metadata["gym_server"] == "unstartable":
            raise RuntimeError("no capacity")
        self._spec = spec
        record["started"].append(spec)
        return self

    async def endpoint(self: AsyncSandbox, port: int) -> SandboxEndpoint:
        assert port in self._spec.ports
        return SandboxEndpoint(endpoint=f"http://{self._spec.metadata['gym_server']}.test", headers={"auth": "proxy"})

    async def stop(self: AsyncSandbox) -> None:
        if self._spec is not None:
            if self._spec.metadata["gym_server"] in record["failing_stops"]:
                raise RuntimeError("delete timed out")
            record["stopped"].append(self._spec.metadata["gym_server"])

    async def serialize(self: AsyncSandbox, *, scope=None) -> dict[str, Any]:
        return {"sandbox_id": self._spec.metadata["gym_server"]}

    for name, method in {"start": start, "endpoint": endpoint, "stop": stop, "serialize": serialize}.items():
        monkeypatch.setattr(AsyncSandbox, name, method)
    monkeypatch.setattr(app_module, "get_global_config_dict", lambda: {"sandbox": {"local": {}}})
    return record


@pytest.fixture
def archive(tmp_path) -> Path:
    path = tmp_path / "gym_dbs.zip"
    with zipfile.ZipFile(path, "w") as zip_file:
        for name, sql in DUMPS.items():
            zip_file.writestr(name, sql)
    return path


def _server(archive: Path, **overrides) -> EnterpriseOpsGymResourcesServer:
    config = EnterpriseOpsGymResourcesServerConfig(
        **{
            "host": "0.0.0.0",
            "port": 8080,
            "entrypoint": "",
            "name": "enterpriseops_gym_resources_server",
            "sandbox_provider": "sandbox",
            "sandbox_config": {"ttl_s": 600, "metadata": {"team": "gym"}},
            "gym_images": GYM_IMAGES,
            "seed_archive_url": archive.as_uri(),
            "seed_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "seed_archive_dir": str(archive.parent / "cache"),
        }
        | overrides
    )
    return EnterpriseOpsGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.fixture
def client(archive, gyms, sandboxes):
    with TestClient(_server(archive).setup_webserver()) as client:
        yield client


def _task(**overrides) -> dict[str, Any]:
    task = {
        "task_id": "task-1",
        "domain": "hybrid",
        "gym_servers": [
            {"name": "gym-a", "seed_database_file": "dbs/a.sql", "headers": {"x-user-email": "a@example.com"}},
            {"name": "gym-b", "seed_database_file": "dbs/b.sql", "headers": {}},
        ],
        "tool_servers": {"insert_notes": "gym-a", "fail_tool": "gym-a", "insert_tasks": "gym-b"},
        "verifiers": [
            {
                "verifier_type": "database_state",
                "name": "note added",
                "gym_name": "gym-a",
                "validation_config": {"query": "SELECT COUNT(*) FROM notes WHERE text = 'hello'", "expected_value": 1},
            },
            {
                "verifier_type": "database_state",
                "name": "task added",
                "gym_name": "gym-b",
                # Upstream data writes some expected counts as strings.
                "validation_config": {"query": "SELECT COUNT(*) FROM tasks", "expected_value": "1"},
            },
        ],
    }
    return task | overrides


def _seed(session_id: str = "session", **task_overrides) -> ResourcesSeedSessionRequest:
    task = _task(**task_overrides)
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id=f"{session_id}-0"),
        task_id=TaskId(taskset="enterpriseops_gym", task_id=task["task_id"]),
        task_data=task,
    )


def _post_seed(client: TestClient, seed: ResourcesSeedSessionRequest):
    return client.post("/seed_session", json=seed.model_dump(mode="json"))


def _verify_body(task_id: str = "task-1") -> dict[str, Any]:
    return {
        "task_id": task_id,
        "domain": "hybrid",
        "responses_create_params": NeMoGymResponseCreateParamsNonStreaming(input=[]).model_dump(mode="json"),
        "response": NeMoGymResponse(
            id="r",
            created_at=0.0,
            model="m",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ).model_dump(mode="json"),
    }


def _verify(client: TestClient, task_id: str = "task-1"):
    return client.post("/verify", json=_verify_body(task_id))


def test_seed_starts_a_sandbox_per_gym_seeds_its_database_and_lends_the_first(client, gyms, sandboxes):
    response = _post_seed(client, _seed())

    assert response.status_code == 200
    started = {spec.metadata["gym_server"]: spec for spec in sandboxes["started"]}
    assert set(started) == {"gym-a", "gym-b"}
    for name, spec in started.items():
        assert (spec.image, spec.entrypoint, list(spec.ports)) == (
            GYM_IMAGES[name]["image"],
            GYM_IMAGES[name]["command"],
            [GYM_IMAGES[name]["port"]],
        )
        assert (spec.ttl_s, spec.metadata["team"], spec.metadata["task_id"]) == (600, "gym", "task-1")
    access = response.json()["sandbox_access"]
    assert access["connection"]["descriptor"] == {"sandbox_id": "gym-a"}
    assert access["workdir"] == "/tmp"
    assert all(len(gym.databases) == 1 for gym in gyms.values())
    # A repeated seed returns the same session without starting more sandboxes.
    assert _post_seed(client, _seed()).json() == response.json()
    assert len(sandboxes["started"]) == 2


def test_tools_reach_the_gym_that_serves_them_with_the_task_database_and_context(client, gyms):
    _post_seed(client, _seed())

    assert client.post("/insert_notes", json={"text": "hello"}).json() == '{"inserted": "hello"}'
    assert client.post("/insert_tasks", json={"text": "write"}).status_code == 200

    ((name, arguments, headers),) = gyms["gym-a"].tool_calls
    assert (name, arguments) == ("insert_notes", {"text": "hello"})
    assert headers == {"x-user-email": "a@example.com", "x-database-id": next(iter(gyms["gym-a"].databases))}
    assert [call[0] for call in gyms["gym-b"].tool_calls] == ["insert_tasks"]


def test_a_tool_outside_the_task_or_one_that_fails_answers_400_for_the_model(client, gyms):
    _post_seed(client, _seed())

    unknown = client.post("/list_everything", json={})
    failed = client.post("/fail_tool", json={})

    assert (unknown.status_code, unknown.json()["detail"]) == (
        400,
        "Tool 'list_everything' is not available in this task",
    )
    assert (failed.status_code, failed.json()["detail"]) == (400, "❌ Invalid Tool Arguments: ['x: is required']")
    assert [call[0] for call in gyms["gym-a"].tool_calls] == ["fail_tool"]


def test_an_unreachable_gym_fails_the_tool_call_as_infrastructure(client, gyms):
    _post_seed(client, _seed())
    gyms["gym-a"].reachable = False

    assert client.post("/insert_notes", json={"text": "hello"}).status_code == 502


def test_a_tool_call_or_verify_without_a_seeded_session_is_a_lost_session(client):
    response = client.post("/insert_notes", json={"text": "hello"})

    assert (response.status_code, response.json()["detail"]["failure_kind"]) == (404, "session_lost")
    # No database is left to score, so the sample is masked rather than scored 0.
    result = _verify(client).json()
    assert (result["mask_sample"], result["failure_kind"], result["reward"]) == (True, "session_lost", 0.0)
    assert (result["evaluation_completed"], result["num_verifiers"]) == (False, 0)


def test_verify_scores_every_verifier_and_releases_the_sandboxes(client, sandboxes):
    _post_seed(client, _seed())
    client.post("/insert_notes", json={"text": "hello"})
    client.post("/insert_tasks", json={"text": "write"})

    result = _verify(client).json()

    assert (result["reward"], result["verifier_pass_rate"], result["num_verifiers_passed"]) == (1.0, 1.0, 2)
    assert (result["evaluation_completed"], result["mask_sample"], result["failure_kind"]) == (True, False, None)
    assert [(v["name"], v["actual"], v["passed"]) for v in result["verifier_results"]] == [
        ("note added", 1, True),
        ("task added", 1, True),
    ]
    assert sorted(sandboxes["stopped"]) == ["gym-a", "gym-b"]
    # The results stand once the gyms are gone, and the gyms take no more calls.
    assert _verify(client).json()["verifier_results"] == result["verifier_results"]
    assert client.post("/insert_notes", json={"text": "late"}).status_code == 409


def test_a_task_passes_only_when_all_its_verifiers_pass(client):
    _post_seed(client, _seed())
    client.post("/insert_notes", json={"text": "hello"})

    result = _verify(client).json()

    assert (result["reward"], result["verifier_pass_rate"]) == (0.0, 0.5)
    assert [v["passed"] for v in result["verifier_results"]] == [True, False]


def test_a_verifier_query_the_database_rejects_fails_that_verifier(client):
    broken = _task()["verifiers"][:1] + [
        {
            "verifier_type": "database_state",
            "name": "bad column",
            "gym_name": "gym-b",
            "validation_config": {"query": "SELECT missing FROM tasks", "expected_value": 1},
        }
    ]
    _post_seed(client, _seed(verifiers=broken))
    client.post("/insert_notes", json={"text": "hello"})

    result = _verify(client).json()

    assert (result["reward"], result["verifier_pass_rate"]) == (0.0, 0.5)
    assert result["verifier_results"][1]["error"].startswith("HTTP 400: SQL execution error")
    # Scored as upstream scores it, and flagged as a defect of the task rather than a failure of the model.
    assert result["evaluation_completed"] is False and result["mask_sample"] is False
    assert result["failure_kind"] == "verifier_error"
    assert result["failure_reason"].startswith("Verifier 'bad column': HTTP 400: SQL execution error")


def test_a_verifier_that_cannot_reach_its_gym_fails_verification_for_a_retry(client, gyms, sandboxes):
    _post_seed(client, _seed())
    gyms["gym-b"].reachable = False

    assert _verify(client).status_code == 502
    # The sandboxes stay up for a retried verification; close releases them.
    assert sandboxes["stopped"] == []


def test_verify_names_the_seeded_task(client):
    _post_seed(client, _seed())

    assert _verify(client, task_id="task-2").status_code == 409


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tool_servers": {"insert_notes": "gym-c"}}, "refers to gym servers it does not start"),
        (
            {"gym_servers": [{"name": "gym-a", "seed_database_file": "dbs/a.sql", "headers": {"Host": "x"}}]},
            "context header",
        ),
        (
            {"gym_servers": [{"name": "gym-a", "seed_database_file": "dbs/a.sql", "headers": {"x-database-id": "x"}}]},
            "context header",
        ),
        (
            {
                "gym_servers": [{"name": "gym-c", "seed_database_file": "dbs/a.sql"}],
                "tool_servers": {"t": "gym-c"},
                "verifiers": [],
            },
            "verifiers",
        ),
    ],
)
def test_a_task_the_server_cannot_run_is_a_terminal_422(client, sandboxes, overrides, message):
    response = _post_seed(client, _seed(**overrides))

    assert response.status_code == 422
    assert message in response.json()["detail"]
    assert sandboxes["started"] == []


def test_a_gym_without_an_image_or_a_dump_outside_the_archive_is_a_terminal_422(client, sandboxes):
    verifier = _task()["verifiers"][0]
    no_image = {
        "gym_servers": [{"name": "gym-c", "seed_database_file": "dbs/a.sql"}],
        "tool_servers": {"insert_notes": "gym-c"},
        "verifiers": [verifier | {"gym_name": "gym-c"}],
    }
    missing_dump = {
        "gym_servers": [{"name": "gym-a", "seed_database_file": "../a.sql"}],
        "tool_servers": {"insert_notes": "gym-a"},
        "verifiers": [verifier],
    }

    unknown = _post_seed(client, _seed("s1", **no_image))
    outside = _post_seed(client, _seed("s2", **missing_dump))

    assert (unknown.status_code, unknown.json()["detail"]) == (422, "No gym_images entry for gym servers ['gym-c']")
    assert (outside.status_code, outside.json()["detail"]) == (422, "The seed archive has no database dump '../a.sql'")
    assert sandboxes["started"] == []


def test_a_gym_that_never_becomes_healthy_fails_the_seed_and_stops_every_sandbox(client, gyms, sandboxes):
    gyms["gym-b"].healthy = False

    response = _post_seed(client, _seed())

    assert response.status_code == 503
    assert "gym-b" in sandboxes["stopped"]
    assert sorted(sandboxes["stopped"]) == sorted(spec.metadata["gym_server"] for spec in sandboxes["started"])
    # Nothing is kept: a retried seed starts over.
    gyms["gym-b"].healthy = True
    assert _post_seed(client, _seed()).status_code == 200


def test_a_refused_database_dump_is_a_terminal_422(client, gyms, sandboxes):
    gyms["gym-a"].refuse_seed = True

    assert _post_seed(client, _seed()).status_code == 422
    assert "gym-a" in sandboxes["stopped"]
    assert sorted(sandboxes["stopped"]) == sorted(spec.metadata["gym_server"] for spec in sandboxes["started"])


def test_a_sandbox_that_cannot_start_fails_the_seed_and_stops_the_others(archive, gyms, sandboxes):
    images = GYM_IMAGES | {"unstartable": GYM_IMAGES["gym-b"]}
    with TestClient(_server(archive, gym_images=images).setup_webserver(), raise_server_exceptions=False) as client:
        servers = [_task()["gym_servers"][0], {"name": "unstartable", "seed_database_file": "dbs/b.sql"}]
        verifiers = _task()["verifiers"][:1]
        response = _post_seed(
            client, _seed(gym_servers=servers, tool_servers={"insert_notes": "gym-a"}, verifiers=verifiers)
        )

    assert response.status_code == 500
    # Whether gym-a got to start before the failure cancelled it, every sandbox that started was stopped.
    assert sandboxes["stopped"] == [spec.metadata["gym_server"] for spec in sandboxes["started"]]


def test_close_stops_the_sandboxes_and_the_session_follows_the_contract(archive, gyms, sandboxes):
    server = _server(archive)
    app = server.setup_webserver()
    with TestClient(app) as client:
        seed = _seed()
        _post_seed(client, seed)
        close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=seed.episode_id)

        assert client.post("/close_session", json=close.model_dump(mode="json")).status_code == 200
        assert sorted(sandboxes["stopped"]) == ["gym-a", "gym-b"]
        assert client.post("/insert_notes", json={"text": "hello"}).status_code == 404
        assert _post_seed(client, seed).status_code == 409

    check_resources_session_contract(
        app,
        _seed("contract"),
        keeps_state=True,
        holds_state=lambda session_id: session_id in server._sessions,
        verify_body=_verify_body(),
    )


def test_shutdown_stops_the_sandboxes_of_open_sessions(archive, gyms, sandboxes):
    with TestClient(_server(archive).setup_webserver()) as client:
        _post_seed(client, _seed())

    assert sorted(sandboxes["stopped"]) == ["gym-a", "gym-b"]


def test_the_seed_archive_is_downloaded_once_and_checked(archive, tmp_path, monkeypatch):
    sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    cache = tmp_path / "cache"

    path = ensure_seed_archive(archive.as_uri(), sha256, cache)
    archive.unlink()

    assert ensure_seed_archive(archive.as_uri(), sha256, cache) == path
    with pytest.raises(ValueError, match="expected 0000"):
        ensure_seed_archive(path.as_uri(), "0000", tmp_path / "other")
    assert list((tmp_path / "other").iterdir()) == []


def test_metrics_report_success_and_verifier_pass_rate_overall_and_per_domain(archive):
    server = _server(archive)
    tasks = [
        [
            {"reward": 1.0, "verifier_pass_rate": 1.0, "domain": "email"},
            {"reward": 0.0, "verifier_pass_rate": 0.5, "domain": "email"},
        ],
        [
            {"reward": 0.0, "verifier_pass_rate": 0.0, "domain": "hr"},
            {"reward": 0.0, "verifier_pass_rate": 0.5, "domain": "hr"},
        ],
    ]

    metrics = server.compute_metrics(tasks)
    key = server.get_key_metrics(metrics | {"mean/output_tokens": 10.0})

    assert metrics["pass@1[avg-of-2]/success"] == 25.0
    assert metrics["pass@1[avg-of-2]/verifier_pass_rate"] == 50.0
    assert metrics["email/pass@1[avg-of-2]/success"] == 50.0
    assert metrics["hr/pass@1[avg-of-2]/verifier_pass_rate"] == 25.0
    assert key == {
        "mean/output_tokens": 10.0,
        "pass@1[avg-of-2]/success": 25.0,
        "pass@1[avg-of-2]/verifier_pass_rate": 50.0,
    }


def test_a_seed_or_close_for_another_episode_or_task_is_refused(archive, gyms, sandboxes):
    with TestClient(_server(archive).setup_webserver(), raise_server_exceptions=False) as client:
        seed = _seed()
        _post_seed(client, seed)
        other_episode = seed.model_copy(update={"episode_id": EpisodeId(rollout_id="other-0")})
        other_task = seed.model_copy(update={"task_id": TaskId(taskset="enterpriseops_gym", task_id="task-2")})
        close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=other_episode.episode_id)

        assert _post_seed(client, other_episode).status_code == 409
        assert _post_seed(client, other_task.model_copy(update={"resources_session_id": "s2"})).status_code == 422
        assert client.post("/close_session", json=close.model_dump(mode="json")).status_code == 409
        assert (
            client.post(
                "/close_session", json=close.model_copy(update={"episode_id": seed.episode_id}).model_dump(mode="json")
            ).status_code
            == 200
        )
        assert client.post("/close_session", json=close.model_dump(mode="json")).status_code == 409


def test_a_task_naming_a_gym_server_twice_is_refused(client):
    twice = [_task()["gym_servers"][0]] * 2

    response = _post_seed(client, _seed(gym_servers=twice, tool_servers={"insert_notes": "gym-a"}))

    assert (response.status_code, "names a gym server twice" in response.json()["detail"]) == (422, True)


def test_a_sandbox_that_fails_to_stop_after_verify_is_stopped_again_at_close(client, sandboxes):
    seed = _seed()
    _post_seed(client, seed)
    client.post("/insert_notes", json={"text": "hello"})
    client.post("/insert_tasks", json={"text": "write"})
    sandboxes["failing_stops"].append("gym-b")

    assert _verify(client).json()["reward"] == 1.0
    assert sandboxes["stopped"] == ["gym-a"]

    sandboxes["failing_stops"].clear()
    close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=seed.episode_id)
    assert client.post("/close_session", json=close.model_dump(mode="json")).status_code == 200
    assert sandboxes["stopped"] == ["gym-a", "gym-b"]


def test_shutdown_logs_a_sandbox_it_cannot_stop(archive, gyms, sandboxes, caplog):
    with TestClient(_server(archive).setup_webserver()) as client:
        _post_seed(client, _seed())
        sandboxes["failing_stops"].append("gym-a")

    assert sandboxes["stopped"] == ["gym-b"]
    assert "Failed to stop the sandboxes of EnterpriseOps-Gym session session" in caplog.text


def test_sessions_live_in_one_worker(archive):
    with pytest.raises(ValueError, match="num_workers=1"):
        _server(archive, num_workers=2)
