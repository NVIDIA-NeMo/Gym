# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The task sandbox lifecycle, run on the local sandbox provider in a temporary directory."""

import functools
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult
from nemo_gym.server_utils import ServerClient
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.gdpval import task_sandbox
from resources_servers.gdpval.app import GDPValResourcesServer, GDPValResourcesServerConfig, GDPValVerifyRequest


REFERENCE = "reference_files/abc/brief.txt"


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    """The sandbox working directory on the host; the local provider runs commands there."""
    root = tmp_path / "root"
    monkeypatch.setattr(task_sandbox, "WORKDIR", str(root))
    monkeypatch.setattr(task_sandbox, "get_global_config_dict", lambda: {"sandbox": {"local": {}}})
    # The local provider cannot be reattached from another process, so it has no descriptor to hand out.
    monkeypatch.setattr(AsyncSandbox, "serialize", AsyncMock(return_value={"sandbox_id": "local"}))
    return root


@pytest.fixture
def stopped(monkeypatch) -> list[AsyncSandbox]:
    """Every sandbox stopped during the test, in order."""
    stopped: list[AsyncSandbox] = []
    real_stop = AsyncSandbox.stop

    async def stop(self: AsyncSandbox) -> None:
        stopped.append(self)
        await real_stop(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    return stopped


@pytest.fixture
def references(tmp_path, monkeypatch) -> Path:
    """Where the server downloads reference files before staging them."""
    references = tmp_path / "references"
    references.mkdir()
    monkeypatch.setattr(tempfile, "mkdtemp", functools.partial(tempfile.mkdtemp, dir=references))
    return references


def _server(tmp_path: Path, **extra) -> GDPValResourcesServer:
    config = GDPValResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="gdpval_resources_server",
        judge_model_server={"type": "responses_api_models", "name": "judge"},
        preconvert_office_to_pdf=False,
        tavily_api_key="test-key",
        sandbox_provider="sandbox",
        sandbox_config={"image": "unused-by-the-local-provider"},
        persist_deliverables_dir=str(tmp_path / "persist"),
        **extra,
    )
    return GDPValResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.fixture
def server(tmp_path) -> GDPValResourcesServer:
    return _server(tmp_path)


def _seed(tmp_path: Path, *, session_id: str = "session", repeat: int = 1, url: str | None = None):
    source = tmp_path / "brief.txt"
    source.write_text("the brief")
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id=f"0-{repeat}", repeat=repeat),
        task_id=TaskId(taskset="gdpval:benchmark", task_id="task-1"),
        task_data={
            "task_id": "task-1",
            "reference_files": [REFERENCE],
            "reference_file_urls": [url or f"file://{source}"],
        },
    )


def _post_seed(client: TestClient, seed: ResourcesSeedSessionRequest):
    return client.post("/seed_session", json=seed.model_dump(mode="json"))


def _verify(client: TestClient, **fields) -> dict:
    """POST /verify for the client's session; without a rubric the scorer answers without a judge."""
    body = GDPValVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=NeMoGymResponse(
            id="r",
            created_at=0.0,
            model="m",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ),
        task_id="task-1",
        **fields,
    )
    response = client.post("/verify", json=body.model_dump(mode="json"))
    assert response.status_code == 200, response.text
    return response.json()


def _files(directory: Path) -> dict[str, str]:
    return {p.relative_to(directory).as_posix(): p.read_text() for p in sorted(directory.rglob("*")) if p.is_file()}


def test_a_finished_task_is_persisted_with_its_files_and_references(server, root, tmp_path, stopped, references):
    client = TestClient(server.setup_webserver())

    seeded = _post_seed(client, _seed(tmp_path))

    assert seeded.status_code == 200
    assert seeded.json()["sandbox_access"]["workdir"] == "/root"
    assert (root / REFERENCE).read_text() == "the brief"
    (root / "out").mkdir()
    (root / "out" / "report.md").write_text("the report")
    missing = client.post("/finish", json={"reason": "done", "paths": ["out/report.md", "summary.pdf"]})
    assert missing.status_code == 400
    assert missing.json()["detail"] == (
        "ERROR: Files do not exist: ['summary.pdf']. Verify paths and ensure files were saved."
    )
    finished = client.post("/finish", json={"reason": "done", "paths": f'["out/report.md", "{root}/out/report.md"]'})
    assert finished.json() == "done"

    verified = _verify(client)

    task_dir = tmp_path / "persist" / "task_task-1" / "repeat_1"
    assert verified["deliverables_dir"] == str(task_dir)
    assert _files(task_dir) == {
        "finish_params.json": json.dumps(
            {"reason": "done", "paths": ["out/report.md", f"{root}/out/report.md"]}, indent=2
        ),
        REFERENCE: "the brief",
        "report.md": "the report",
    }
    assert len(stopped) == 1, "/verify stops the sandbox once the files are out"
    assert list(references.iterdir()) == [], "/verify removes the downloaded reference files"
    assert client.post("/finish", json={"reason": "again", "paths": []}).status_code == 409
    persisted = _files(task_dir)
    assert _verify(client)["deliverables_dir"] == str(task_dir)
    assert _files(task_dir) == persisted, "a repeated /verify rescores the persisted files without collecting again"
    assert len(stopped) == 1


@pytest.mark.parametrize(
    "paths, persisted",
    [
        (["a/report.md", "b/report.md"], {"report.md": "from b"}),
        (["a/report.md", "a/notes.md"], {"report.md": "from a"}),
    ],
    ids=["same base name: the later one wins", "a file that cannot be copied is skipped"],
)
def test_submitted_files_are_flattened_to_their_base_names(server, root, tmp_path, monkeypatch, paths, persisted):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed(tmp_path))
    for directory in ("a", "b"):
        (root / directory).mkdir()
        (root / directory / "report.md").write_text(f"from {directory}")
    (root / "a" / "notes.md").write_text("notes")
    assert client.post("/finish", json={"reason": "done", "paths": paths}).status_code == 200
    (root / "a" / "notes.md").unlink()

    task_dir = Path(_verify(client)["deliverables_dir"])

    assert {k: v for k, v in _files(task_dir).items() if not k.startswith("reference_files/")} == {
        "finish_params.json": json.dumps({"reason": "done", "paths": paths}, indent=2),
        **persisted,
    }


@pytest.mark.parametrize(
    "calls, finish_params",
    [
        ([("/abandon_task_finish", {"reason": "inputs missing"})], {"reason": "inputs missing"}),
        (
            [("/finish", {"reason": "done", "paths": []}), ("/abandon_task_finish", {"reason": "gave up"})],
            {"reason": "gave up"},
        ),
        ([], None),
    ],
    ids=["abandoned", "the last call wins", "never finished"],
)
def test_tasks_without_submitted_files_record_how_they_ended(server, root, tmp_path, calls, finish_params):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed(tmp_path))
    for path, body in calls:
        response = client.post(path, json=body)
        assert response.status_code == 200
        assert response.json() == body["reason"]

    task_dir = Path(_verify(client)["deliverables_dir"])

    assert json.loads((task_dir / "finish_params.json").read_text()) == finish_params
    assert sorted(p.name for p in task_dir.iterdir()) == ["finish_params.json", "reference_files"]


def test_sessions_keep_their_finish_calls_and_files_apart(server, root, tmp_path):
    app = server.setup_webserver()
    first, second = TestClient(app), TestClient(app)
    _post_seed(first, _seed(tmp_path, session_id="first", repeat=0))
    _post_seed(second, _seed(tmp_path, session_id="second", repeat=1))
    (root / "a.md").write_text("a")
    (root / "b.md").write_text("b")
    first.post("/finish", json={"reason": "first", "paths": ["a.md"]})
    second.post("/finish", json={"reason": "second", "paths": ["b.md"]})

    first_dir, second_dir = (Path(_verify(client)["deliverables_dir"]) for client in (first, second))

    assert (first_dir.name, second_dir.name) == ("repeat_0", "repeat_1")
    assert "a.md" in _files(first_dir) and "b.md" not in _files(first_dir)
    assert "b.md" in _files(second_dir) and "a.md" not in _files(second_dir)


def test_a_failed_file_check_is_an_infrastructure_error(server, root, tmp_path, monkeypatch):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed(tmp_path))
    failed = SandboxExecResult(stdout=None, stderr="timed out", return_code=-1, error_type="timeout")
    monkeypatch.setattr(AsyncSandbox, "exec", AsyncMock(return_value=failed))

    response = client.post("/finish", json={"reason": "done", "paths": ["report.md"]})

    assert response.status_code == 503
    assert "Could not check the submitted files" in response.json()["detail"]
    task_dir = Path(_verify(client)["deliverables_dir"])
    assert json.loads((task_dir / "finish_params.json").read_text()) is None, "the failed finish is not recorded"


def test_a_reference_file_that_cannot_be_fetched_fails_the_seed(server, root, tmp_path, stopped, references):
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)

    response = _post_seed(client, _seed(tmp_path, url=f"file://{tmp_path}/gone.txt"))

    assert response.status_code == 500
    assert not root.exists(), "no sandbox is started for a task that cannot be staged"
    assert list(references.iterdir()) == []
    assert client.post("/finish", json={"reason": "done", "paths": []}).status_code == 400


def test_a_failed_reference_upload_stops_the_sandbox(server, root, tmp_path, monkeypatch, stopped):
    monkeypatch.setattr(AsyncSandbox, "upload", AsyncMock(side_effect=OSError("upload failed")))

    with pytest.raises(OSError, match="upload failed"):
        _post_seed(TestClient(server.setup_webserver()), _seed(tmp_path))

    assert len(stopped) == 1


def test_a_seed_without_a_sandbox_image_names_the_setting(server, root, tmp_path, monkeypatch):
    monkeypatch.setattr(server.config.sandbox_config, "image", None)

    with pytest.raises(ValueError, match="GDPVAL_SANDBOX_IMAGE"):
        _post_seed(TestClient(server.setup_webserver()), _seed(tmp_path))

    assert not root.exists()


def test_a_seed_for_another_task_than_its_data_is_rejected(server, root, tmp_path):
    seed = _seed(tmp_path)
    seed.task_data["task_id"] = "task-2"

    with pytest.raises(ValueError, match="TaskId does not match"):
        _post_seed(TestClient(server.setup_webserver()), seed)

    assert not root.exists()


def test_a_session_is_bound_to_its_episode_and_task(server, root, tmp_path):
    client = TestClient(server.setup_webserver())
    seed = _seed(tmp_path)
    _post_seed(client, seed)
    other_episode = seed.model_copy(update={"episode_id": EpisodeId(rollout_id="0-2", repeat=2)})
    other_close = ResourcesCloseSessionRequest(
        resources_session_id=seed.resources_session_id, episode_id=other_episode.episode_id
    )

    with pytest.raises(ValueError, match="already bound to another episode or task"):
        _post_seed(client, other_episode)
    with pytest.raises(ValueError, match="does not match the seeded resources session"):
        client.post("/close_session", json=other_close.model_dump(mode="json"))


def test_close_releases_the_session_and_follows_the_session_contract(server, root, tmp_path, stopped, references):
    app = server.setup_webserver()
    client = TestClient(app)
    seed = _seed(tmp_path)
    _post_seed(client, seed)
    close = ResourcesCloseSessionRequest(resources_session_id=seed.resources_session_id, episode_id=seed.episode_id)

    assert client.post("/close_session", json=close.model_dump(mode="json")).status_code == 200

    assert len(stopped) == 1
    assert list(references.iterdir()) == []
    assert client.post("/finish", json={"reason": "done", "paths": []}).status_code == 400
    after_close = _verify(client, deliverables_dir=str(tmp_path / "stored"))
    assert after_close["deliverables_dir"] == str(tmp_path / "stored"), (
        "without a session /verify scores the request's"
    )
    check_resources_session_contract(app, _seed(tmp_path, session_id="contract"), keeps_state=True)


def test_shutdown_stops_the_sandboxes_of_open_sessions(server, root, tmp_path, stopped):
    with TestClient(server.setup_webserver()) as client:
        _post_seed(client, _seed(tmp_path))
        assert stopped == []

    assert len(stopped) == 1


def test_sessions_require_a_single_worker(tmp_path):
    with pytest.raises(ValueError, match="num_workers=1"):
        _server(tmp_path, num_workers=2)
