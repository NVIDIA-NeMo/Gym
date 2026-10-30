# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The task sandbox lifecycle, run on the local sandbox provider in a temporary directory."""

import asyncio
import functools
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.core import SESSION_LOST, EpisodeId, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, TaskId
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult
from nemo_gym.server_runtime import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.server_runtime.request_handling.session_routing import install_session_routing
from nemo_gym.server_runtime.testing.session_conformance import check_resources_session_contract
from resources_servers.gdpval import persisted_layout, task_sandbox
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
        episode_id=EpisodeId(rollout_id=f"0-{repeat}"),
        task_id=TaskId(taskset="gdpval:benchmark", task_id="task-1"),
        task_data={
            "task_id": "task-1",
            "reference_files": [REFERENCE],
            "reference_file_urls": [url or f"file://{source}"],
        },
    )


def _post_seed(client: TestClient, seed: ResourcesSeedSessionRequest):
    return client.post("/seed_session", json=seed.model_dump(mode="json"))


def _verify_body(response_metadata: dict[str, str] | None = None, **fields) -> GDPValVerifyRequest:
    return GDPValVerifyRequest(
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
            metadata=response_metadata,
        ),
        task_id="task-1",
        **fields,
    )


def _verify(client: TestClient, **fields) -> dict:
    """POST /verify for the client's session; without a rubric the scorer answers without a judge."""
    response = client.post("/verify", json=_verify_body(**fields).model_dump(mode="json"))
    assert response.status_code == 200, response.text
    return response.json()


def _close(client: TestClient, seed: ResourcesSeedSessionRequest, **update):
    close = ResourcesCloseSessionRequest(
        resources_session_id=seed.resources_session_id, episode_id=update.get("episode_id", seed.episode_id)
    )
    return client.post("/close_session", json=close.model_dump(mode="json"))


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


def test_unjudged_deliverables_are_persisted_and_masked(server, root, tmp_path, monkeypatch):
    async def no_judge(**_kwargs):
        raise AssertionError("judge_deliverables=false must not call a judge")

    monkeypatch.setattr("resources_servers.gdpval.scoring.score_with_rubric", no_judge)
    server.config.judge_deliverables = False
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed(tmp_path))
    (root / "report.md").write_text("the report")
    client.post("/finish", json={"reason": "done", "paths": ["report.md"]})

    verified = _verify(client, rubric_json=[{"criterion": "clarity", "score": 1}])

    assert (verified["reward"], verified["mask_sample"]) == (0.0, True)
    assert verified["judge_response"] is None
    assert (Path(verified["deliverables_dir"]) / "report.md").read_text() == "the report"


def test_submitted_files_are_flattened_to_their_base_names(server, root, tmp_path):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed(tmp_path))
    paths = ["a/report.md", "b/report.md"]
    for directory in ("a", "b"):
        (root / directory).mkdir()
        (root / directory / "report.md").write_text(f"from {directory}")
    assert client.post("/finish", json={"reason": "done", "paths": paths}).status_code == 200

    task_dir = Path(_verify(client)["deliverables_dir"])

    assert _files(task_dir) == {
        "finish_params.json": json.dumps({"reason": "done", "paths": paths}, indent=2),
        REFERENCE: "the brief",
        "report.md": "from b",
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


def test_an_agent_that_ran_out_of_wall_time_is_a_masked_timeout_and_persists_nothing(
    server, root, tmp_path, stopped, references
):
    client = TestClient(server.setup_webserver())
    seed = _seed(tmp_path)
    _post_seed(client, seed)
    (root / "report.md").write_text("half a report")
    client.post("/finish", json={"reason": "done", "paths": ["report.md"]})

    verified = _verify(client, response_metadata={"stop_reason": "wall_time"})

    assert verified["_ng_failure_class"] == "agent_timeout"
    assert (verified["failure_kind"], verified["mask_sample"], verified["reward"]) == ("agent_timeout", True, 0.0)
    assert "_ng_failure_terminal" not in verified
    assert not (tmp_path / "persist").exists(), "the partial work is not collected or judged"
    assert _close(client, seed).status_code == 200
    assert len(stopped) == 1, "the sandbox is still released at close"
    assert list(references.iterdir()) == []


def test_an_agent_that_was_asked_to_stop_is_not_a_timeout(server, root, tmp_path, monkeypatch):
    async def judge(**_kwargs):
        return 1.0, {"score": 1.0}

    monkeypatch.setattr("resources_servers.gdpval.scoring.score_with_rubric_visual", judge)
    monkeypatch.setattr("resources_servers.gdpval.app.get_server_url", lambda name: "http://localhost:9999")
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed(tmp_path))
    (root / "report.md").write_text("the report")
    client.post("/finish", json={"reason": "done", "paths": ["report.md"]})

    verified = _verify(
        client, response_metadata={"stop_reason": "cancelled"}, rubric_json=[{"criterion": "clarity", "score": 1}]
    )

    assert "_ng_failure_class" not in verified
    assert not verified["mask_sample"]
    assert verified["reward"] == 1.0
    assert (Path(verified["deliverables_dir"]) / "report.md").read_text() == "the report"


def test_a_file_that_cannot_be_downloaded_fails_the_verification_and_persists_nothing(server, root, tmp_path):
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    _post_seed(client, _seed(tmp_path))
    (root / "report.md").write_text("the report")
    assert client.post("/finish", json={"reason": "done", "paths": ["report.md"]}).status_code == 200
    (root / "report.md").unlink()

    response = client.post("/verify", json=_verify_body().model_dump(mode="json"))

    assert response.status_code == 500
    assert list((tmp_path / "persist" / "task_task-1").iterdir()) == [], "no partial directory is left behind"


def test_a_failed_download_removes_an_earlier_tree_so_a_retry_cannot_reuse_it(server, root, tmp_path):
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    _post_seed(client, _seed(tmp_path))
    task_dir = persisted_layout.repeat_dir(tmp_path / "persist", "task-1", 1)
    task_dir.mkdir(parents=True)
    (task_dir / persisted_layout.FINISH_PARAMS_FILE).write_text('{"reason": "an earlier run"}')
    (task_dir / "report.docx").write_text("another model's deliverable")
    (root / "report.md").write_text("the report")
    client.post("/finish", json={"reason": "done", "paths": ["report.md"]})
    (root / "report.md").unlink()

    assert client.post("/verify", json=_verify_body().model_dump(mode="json")).status_code == 500
    assert not task_dir.exists(), "the failed collection leaves no tree to reuse"
    reuse = _verify_body(reuse_cached_deliverable=True, ng_rollout_index=1)
    assert TestClient(server.setup_webserver()).post("/verify", json=reuse.model_dump(mode="json")).status_code == 404

    (root / "report.md").write_text("the report")
    verified = _verify(client)

    assert Path(verified["deliverables_dir"]) == task_dir
    assert "report.docx" not in _files(task_dir) and _files(task_dir)["report.md"] == "the report"


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
    assert client.post("/finish", json={"reason": "done", "paths": []}).status_code == 404


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

    refused = _post_seed(TestClient(server.setup_webserver()), seed)

    assert refused.status_code == 422, "a task row that no retry can fix is refused as final"
    assert "TaskId does not match" in refused.json()["detail"]

    assert not root.exists()


def test_task_data_that_is_not_a_task_is_refused_as_final(server, root, tmp_path):
    client = TestClient(server.setup_webserver())
    seed = _seed(tmp_path)
    del seed.task_data["task_id"]

    refused = _post_seed(client, seed)

    assert refused.status_code == 422, "task_data that no retry can fix is final, not a retried 500"
    assert "task_data is not a GDPVal task" in refused.json()["detail"]
    assert not root.exists(), "no sandbox is started for a row that is not a task"
    assert (server._sessions, server._claimed_directories) == ({}, {})
    # A refusal does not fence the ID: it was never seeded, so a corrected seed can still use it.
    assert _post_seed(client, _seed(tmp_path)).status_code == 200


def _app(server: GDPValResourcesServer, *, routed: bool):
    app = server.setup_webserver()
    if routed:
        # As run_webserver does, even for one worker: the router names a typed seed's session by its
        # resources_session_id before the handler runs.
        key = server.get_session_middleware_key()
        install_session_routing(
            app, socket_dir=None, session_cookie=key, secret_key=key, **server.claimed_session_routing()
        )
    return app


@pytest.mark.parametrize("routed", [False, True], ids=["unrouted", "routed"])
def test_a_refused_seed_does_not_hand_out_the_cookie_of_the_live_session(server, root, tmp_path, routed):
    app = _app(server, routed=routed)
    seed = _seed(tmp_path)
    assert _post_seed(TestClient(app), seed).status_code == 200
    intruder = TestClient(app)

    refused = _post_seed(intruder, seed.model_copy(update={"episode_id": EpisodeId(rollout_id="0-2")}))

    assert refused.status_code == 409
    # The refused caller's cookie does not name the live session, so its tools and verify cannot reach it.
    assert intruder.post("/abandon_task_finish", json={"reason": "stolen"}).status_code == 404
    assert server._sessions[seed.resources_session_id].finish_call is None
    assert _verify(intruder)["failure_kind"] == SESSION_LOST


def test_a_seed_whose_rollout_id_names_no_rollout_index_is_rejected(server, root, tmp_path):
    seed = _seed(tmp_path).model_copy(update={"episode_id": EpisodeId(rollout_id="explicit")})

    refused = _post_seed(TestClient(server.setup_webserver()), seed)

    # 422, not a retried 500: a retry sends the same rollout_id.
    assert refused.status_code == 422
    assert "is not '<task_index>-<rollout_index>' with two integers" in refused.json()["detail"]

    assert not root.exists()


def test_a_second_live_session_for_the_same_task_and_repeat_is_refused_until_the_first_closes(
    server, root, tmp_path, references
):
    client = TestClient(server.setup_webserver())
    first = _seed(tmp_path, session_id="first", repeat=1)
    second = _seed(tmp_path, session_id="second", repeat=1).model_copy(
        update={"episode_id": EpisodeId(rollout_id="5-1")}
    )
    assert _post_seed(client, first).status_code == 200

    refused = _post_seed(client, second)

    assert refused.status_code == 409
    assert "Session first already holds task task-1 repeat 1" in refused.json()["detail"]
    assert _post_seed(client, first).status_code == 200, "seeding the holder again is idempotent"
    close = ResourcesCloseSessionRequest(resources_session_id="first", episode_id=first.episode_id)
    assert client.post("/close_session", json=close.model_dump(mode="json")).status_code == 200
    assert _post_seed(client, second).status_code == 200


@pytest.mark.parametrize("failure", [RuntimeError("stop failed"), asyncio.CancelledError()])
def test_a_close_whose_stop_fails_still_frees_the_directory_for_a_retry(
    server, root, tmp_path, references, monkeypatch, failure
):
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    first = _seed(tmp_path, session_id="first", repeat=1)
    assert _post_seed(client, first).status_code == 200
    real_stop = AsyncSandbox.stop
    monkeypatch.setattr(AsyncSandbox, "stop", AsyncMock(side_effect=failure))
    close = ResourcesCloseSessionRequest(resources_session_id="first", episode_id=first.episode_id)
    try:
        assert client.post("/close_session", json=close.model_dump(mode="json")).status_code == 500
    except BaseException:  # a cancellation can surface through the test client
        pass
    monkeypatch.setattr(AsyncSandbox, "stop", real_stop)
    retry = _seed(tmp_path, session_id="retry", repeat=1).model_copy(
        update={"episode_id": EpisodeId(rollout_id="0-1", attempt=1)}
    )

    assert _post_seed(client, retry).status_code == 200


def test_sessions_of_different_repeats_do_not_conflict(server, root, tmp_path, references):
    client = TestClient(server.setup_webserver())

    assert _post_seed(client, _seed(tmp_path, session_id="a", repeat=0)).status_code == 200
    assert _post_seed(client, _seed(tmp_path, session_id="b", repeat=1)).status_code == 200


def test_a_failed_seed_releases_its_claim(server, root, tmp_path, references):
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)

    failed = _post_seed(client, _seed(tmp_path, session_id="a", url=f"file://{tmp_path}/gone.txt"))
    retried = _post_seed(client, _seed(tmp_path, session_id="b"))

    assert failed.status_code == 500
    assert retried.status_code == 200


def test_a_session_is_bound_to_its_episode_and_task(server, root, tmp_path):
    client = TestClient(server.setup_webserver())
    seed = _seed(tmp_path)
    _post_seed(client, seed)
    other_episode = seed.model_copy(update={"episode_id": EpisodeId(rollout_id="0-2")})
    other_close = ResourcesCloseSessionRequest(
        resources_session_id=seed.resources_session_id, episode_id=other_episode.episode_id
    )

    rebound = _post_seed(client, other_episode)
    assert rebound.status_code == 409
    assert "already bound to another episode or task" in rebound.json()["detail"]
    wrong_close = client.post("/close_session", json=other_close.model_dump(mode="json"))
    assert wrong_close.status_code == 409
    assert "does not match the seeded resources session" in wrong_close.json()["detail"]


def test_close_releases_the_session_and_follows_the_session_contract(server, root, tmp_path, stopped, references):
    app = server.setup_webserver()
    client = TestClient(app)
    seed = _seed(tmp_path)
    _post_seed(client, seed)
    close = ResourcesCloseSessionRequest(resources_session_id=seed.resources_session_id, episode_id=seed.episode_id)

    assert client.post("/close_session", json=close.model_dump(mode="json")).status_code == 200

    assert len(stopped) == 1
    assert list(references.iterdir()) == []
    assert client.post("/finish", json={"reason": "done", "paths": []}).status_code == 404
    after_close = _verify(client, deliverables_dir=str(tmp_path / "stored"))
    assert after_close["deliverables_dir"] == str(tmp_path / "stored"), (
        "without a session /verify scores the request's"
    )
    check_resources_session_contract(
        app,
        _seed(tmp_path, session_id="contract"),
        keeps_state=True,
        holds_state=lambda resources_session_id: resources_session_id in server._sessions,
        verify_body=_verify_body().model_dump(mode="json"),
    )


def test_shutdown_stops_the_sandboxes_of_open_sessions(server, root, tmp_path, stopped):
    with TestClient(server.setup_webserver()) as client:
        _post_seed(client, _seed(tmp_path))
        assert stopped == []

    assert len(stopped) == 1


def test_sessions_require_a_single_worker(tmp_path):
    with pytest.raises(ValueError, match="num_workers=1"):
        _server(tmp_path, num_workers=2)


def _restart(client: TestClient, tmp_path: Path) -> TestClient:
    """A new server process over the same persist directory, reached with the old session cookie."""
    return TestClient(_server(tmp_path).setup_webserver(), cookies=client.cookies)


def test_a_session_that_is_gone_is_not_judged_on_the_transcript(server, root, tmp_path):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed(tmp_path))
    client = _restart(client, tmp_path)

    verified = _verify(client)

    assert (verified["reward"], verified["mask_sample"], verified["failure_kind"]) == (0.0, True, SESSION_LOST)
    assert "nothing to judge" in verified["failure_reason"]


def test_a_request_with_no_session_and_no_deliverables_is_masked(server):
    verified = _verify(TestClient(server.setup_webserver()))

    assert (verified["reward"], verified["mask_sample"], verified["failure_kind"]) == (0.0, True, SESSION_LOST)


def test_a_request_with_a_gone_session_and_an_explicit_deliverables_dir_is_judged(server, root, tmp_path, monkeypatch):
    async def judge(**_kwargs):
        return 1.0, {"score": 1.0}

    for scorer in ("score_with_rubric", "score_with_rubric_visual"):
        monkeypatch.setattr(f"resources_servers.gdpval.scoring.{scorer}", judge)
    monkeypatch.setattr("resources_servers.gdpval.app.get_server_url", lambda name: "http://localhost:9999")
    stored = tmp_path / "stored"
    stored.mkdir()
    (stored / "report.md").write_text("the report")
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed(tmp_path))
    client = _restart(client, tmp_path)

    verified = _verify(client, deliverables_dir=str(stored), rubric_json=[{"criterion": "clarity", "score": 1}])

    # Only a gone session with nothing to judge is a lost session; stored deliverables are a real measurement.
    assert verified["deliverables_dir"] == str(stored)
    assert (verified["reward"], verified["mask_sample"], verified["failure_kind"]) == (1.0, False, None)


def test_a_reuse_request_without_a_session_scores_the_persisted_deliverable(server, tmp_path):
    repeat_dir = persisted_layout.repeat_dir(tmp_path / "persist", "task-1", 2)
    repeat_dir.mkdir(parents=True)
    (repeat_dir / persisted_layout.FINISH_PARAMS_FILE).write_text("null")
    (repeat_dir / "report.docx").write_text("deliverable")

    verified = _verify(TestClient(server.setup_webserver()), reuse_cached_deliverable=True, ng_rollout_index=2)

    assert verified["deliverables_dir"] == str(repeat_dir)


def test_a_reuse_request_for_a_directory_without_finish_params_is_not_found(server, tmp_path):
    repeat_dir = persisted_layout.repeat_dir(tmp_path / "persist", "task-1", 2)
    repeat_dir.mkdir(parents=True)
    (repeat_dir / "report.docx").write_text("a partial directory")

    response = TestClient(server.setup_webserver()).post(
        "/verify", json=_verify_body(reuse_cached_deliverable=True, ng_rollout_index=2).model_dump(mode="json")
    )

    assert response.status_code == 404


def test_a_reuse_request_for_a_repeat_with_nothing_persisted_is_not_found(server, tmp_path):
    repeat_dir = persisted_layout.repeat_dir(tmp_path / "persist", "task-1", 0)
    repeat_dir.mkdir(parents=True)
    (repeat_dir / persisted_layout.FINISH_PARAMS_FILE).write_text("null")

    response = TestClient(server.setup_webserver()).post(
        "/verify", json=_verify_body(reuse_cached_deliverable=True).model_dump(mode="json")
    )

    assert response.status_code == 404


def test_a_reuse_request_when_nothing_was_ever_persisted_is_not_found(server):
    response = TestClient(server.setup_webserver()).post(
        "/verify", json=_verify_body(reuse_cached_deliverable=True, ng_rollout_index=3).model_dump(mode="json")
    )

    assert response.status_code == 404


def test_a_live_session_wins_over_a_reuse_request_and_an_explicit_deliverables_dir(server, root, tmp_path):
    cached = persisted_layout.repeat_dir(tmp_path / "persist", "task-1", 1)
    cached.mkdir(parents=True)
    (cached / persisted_layout.FINISH_PARAMS_FILE).write_text("null")
    (cached / "old.docx").write_text("an earlier rollout")
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed(tmp_path))
    (root / "report.md").write_text("the report")
    client.post("/finish", json={"reason": "done", "paths": ["report.md"]})

    verified = _verify(
        client, reuse_cached_deliverable=True, deliverables_dir=str(tmp_path / "stored"), ng_rollout_index=1
    )

    assert verified["deliverables_dir"] == str(cached)
    assert "old.docx" not in _files(cached)
    assert _files(cached)["report.md"] == "the report"


def test_a_failure_after_the_download_stage_releases_the_claim_and_the_sandbox(
    server, root, tmp_path, monkeypatch, stopped, references
):
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    serialize = AsyncMock(side_effect=[RuntimeError("no descriptor"), {"sandbox_id": "local"}])
    monkeypatch.setattr(AsyncSandbox, "serialize", serialize)

    failed = _post_seed(client, _seed(tmp_path, session_id="a"))

    assert failed.status_code == 500
    assert len(stopped) == 1
    assert list(references.iterdir()) == []
    assert _post_seed(client, _seed(tmp_path, session_id="b")).status_code == 200


def test_a_task_with_fewer_reference_urls_than_files_fails_the_seed(server, root, tmp_path, stopped, references):
    seed = _seed(tmp_path)
    seed.task_data["reference_files"] = [REFERENCE, "reference_files/abc/extra.txt"]
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)

    assert _post_seed(client, seed).status_code == 500

    assert not root.exists()
    assert list(references.iterdir()) == []


def test_a_closed_session_cannot_be_seeded_again(server, root, tmp_path, references):
    client = TestClient(server.setup_webserver())
    seed = _seed(tmp_path)
    _post_seed(client, seed)
    assert _close(client, seed).status_code == 200

    late = _post_seed(client, seed)
    assert late.status_code == 409
    assert "already closed" in late.json()["detail"]


def test_closing_a_closed_session_is_idempotent_only_for_its_own_episode(server, root, tmp_path, stopped, references):
    client = TestClient(server.setup_webserver())
    seed = _seed(tmp_path)
    _post_seed(client, seed)
    assert _close(client, seed).status_code == 200

    assert _close(client, seed).status_code == 200
    assert len(stopped) == 1
    wrong_close = _close(client, seed, episode_id=EpisodeId(rollout_id="0-2"))
    assert wrong_close.status_code == 409
    assert "does not match the closed resources session" in wrong_close.json()["detail"]


def test_abandoning_without_a_session_is_refused(server):
    response = TestClient(server.setup_webserver()).post("/abandon_task_finish", json={"reason": "no inputs"})

    assert response.status_code == 404
    assert response.json()["detail"]["failure_kind"] == SESSION_LOST


def test_a_directory_is_not_a_submittable_file(server, root, tmp_path):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed(tmp_path))
    (root / "out").mkdir()

    response = client.post("/finish", json={"reason": "done", "paths": ["out"]})

    assert response.status_code == 400
    assert "Files do not exist: ['out']" in response.json()["detail"]


def test_the_agent_cannot_finish_into_a_sandbox_that_verify_released(server, root, tmp_path):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed(tmp_path))
    _verify(client)

    assert client.post("/finish", json={"reason": "late", "paths": []}).status_code == 409
    task_dir = Path(_verify(client)["deliverables_dir"])
    assert json.loads((task_dir / "finish_params.json").read_text()) is None


def test_a_failed_collection_fails_the_verification_even_when_nothing_is_judged(server, root, tmp_path):
    server.config.judge_deliverables = False
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    _post_seed(client, _seed(tmp_path))
    (root / "report.md").write_text("the report")
    client.post("/finish", json={"reason": "done", "paths": ["report.md"]})
    (root / "report.md").unlink()

    response = client.post("/verify", json=_verify_body().model_dump(mode="json"))

    assert response.status_code == 500


def test_a_reuse_request_for_a_nonexistent_persist_directory_is_not_found(server):
    response = TestClient(server.setup_webserver()).post(
        "/verify", json=_verify_body(reuse_cached_deliverable=True).model_dump(mode="json")
    )

    assert response.status_code == 404


@pytest.fixture
def failing_stop(monkeypatch) -> list[AsyncSandbox]:
    """Every sandbox whose stop() was called; each stops for real, then raises."""
    attempted: list[AsyncSandbox] = []
    real_stop = AsyncSandbox.stop

    async def stop(self: AsyncSandbox) -> None:
        attempted.append(self)
        await real_stop(self)
        raise OSError("stop failed")

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    return attempted


def test_shutdown_stops_every_sandbox_even_when_one_stop_fails(server, root, tmp_path, failing_stop):
    with TestClient(server.setup_webserver()) as client:
        _post_seed(client, _seed(tmp_path, session_id="a", repeat=0))
        _post_seed(client, _seed(tmp_path, session_id="b", repeat=1))

    assert len(failing_stop) == 2


def test_a_close_whose_stop_fails_is_an_error_and_can_be_retried(server, root, tmp_path, monkeypatch, references):
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    seed = _seed(tmp_path)
    _post_seed(client, seed)
    monkeypatch.setattr(AsyncSandbox, "stop", AsyncMock(side_effect=[OSError("stop failed"), None]))

    assert _close(client, seed).status_code == 500
    assert _close(client, seed).status_code == 200
    assert list(references.iterdir()) == []
