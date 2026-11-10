# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The build sandbox lifecycle, run on the local sandbox provider in a temporary directory."""

import asyncio
import dataclasses
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.core import PROVIDER_IO_TIMEOUT, SESSION_LOST, VERIFIER_ERROR, EpisodeId, TaskId
from nemo_gym.core.failure_kinds import PROVIDER_REQUEST_FAILED
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult
from nemo_gym.sandbox.failures import sandbox_failure
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from resources_servers.vibench import app as app_module
from resources_servers.vibench.app import (
    GraderConfigError,
    PlanResult,
    VibenchResourcesServer,
    VibenchResourcesServerConfig,
)


PRD = "v1/prds/notes/prd/mvp.txt"
PLANS = ["v1/prds/notes/tests/mvp/test1.txt", "v1/prds/notes/tests/mvp/test2.txt"]


@pytest.fixture
def checkout(tmp_path) -> Path:
    """A ViBench checkout with one app: a PRD, a PRD asset, two test plans and a grader-only fixture."""
    checkout = tmp_path / "vibench"
    app_dir = checkout / "v1" / "prds" / "notes"
    (app_dir / "prd").mkdir(parents=True)
    (app_dir / "prd" / "mvp.txt").write_text("Build a notes app.")
    (app_dir / "assets" / "icons").mkdir(parents=True)
    (app_dir / "assets" / "icons" / "logo.svg").write_text("<svg/>")
    (app_dir / "tests" / "mvp").mkdir(parents=True)
    for plan in PLANS:
        (checkout / plan).write_text("<step>x</step>")
    (app_dir / "test_assets").mkdir()
    (app_dir / "test_assets" / "upload.png").write_text("grader only")
    return checkout


@pytest.fixture(autouse=True)
def temp_root(tmp_path, monkeypatch) -> Path:
    """The host's temporary directory, where the server keeps each session's copy and each grading run."""
    temp_root = tmp_path / "host-tmp"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))
    return temp_root


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    """The sandbox working directory on the host; the local provider runs commands there."""
    monkeypatch.setattr(app_module, "get_global_config_dict", lambda: {"sandbox": {"local": {}}})
    monkeypatch.setattr(app_module, "REMOTE_ARCHIVE", str(tmp_path / "remote" / "app.tar"))
    (tmp_path / "remote").mkdir()
    # The local provider cannot be reattached from another process, so it has no descriptor to hand out.
    monkeypatch.setattr(AsyncSandbox, "serialize", AsyncMock(return_value={"sandbox_id": "local"}))
    return tmp_path / "app"


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
def graded(monkeypatch) -> list[dict[str, str]]:
    """The app tree each test plan was graded on, keyed by relative path; every plan scores 1."""
    graded: list[dict[str, str]] = []

    async def grade(self, app_dir: Path, test_plan_rel: str, work_dir: Path, test_assets_dir):
        graded.append(
            {p.relative_to(app_dir).as_posix(): p.read_text() for p in sorted(app_dir.rglob("*")) if p.is_file()}
        )
        return PlanResult(
            test_plan=Path(test_plan_rel).stem,
            score=1.0,
            full_points=1.0,
            normalized_score=1.0,
            steps_total=1,
            steps_passed=1,
            seeding_failed=False,
            duration_s=0.0,
        )

    monkeypatch.setattr(VibenchResourcesServer, "_grade_one_test_plan", grade)
    return graded


def _server(checkout: Path, root: Path, **extra) -> VibenchResourcesServer:
    config = VibenchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="vibench_resources_server",
        vibench_repo_root=str(checkout),
        sandbox_provider="sandbox",
        sandbox_config={"image": "unused-by-the-local-provider"},
        app_workdir=str(root),
        **extra,
    )
    return VibenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.fixture
def server(checkout, root) -> VibenchResourcesServer:
    return _server(checkout, root)


def _task_data(**update) -> dict:
    return {
        "app": "notes",
        "artifact": "mvp",
        "prd_files": [PRD],
        "test_plans": PLANS,
        "asset_dirs": ["v1/prds/notes/assets"],
        "test_assets_dir": "v1/prds/notes/test_assets",
    } | update


def _seed(*, session_id: str = "session", rollout_id: str = "0-0", **update) -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id=rollout_id),
        task_id=TaskId(taskset="vibench:example", task_id="notes-mvp"),
        task_data=_task_data(**update),
    )


def _post_seed(client: TestClient, seed: ResourcesSeedSessionRequest):
    return client.post("/seed_session", json=seed.model_dump(mode="json"))


def _verify_body() -> dict:
    return _task_data() | {
        "responses_create_params": {"input": [{"role": "user", "content": "build it"}]},
        "response": {
            "id": "resp_1",
            "created_at": 0,
            "model": "m",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        },
    }


def _verify(client: TestClient) -> dict:
    response = client.post("/verify", json=_verify_body())
    assert response.status_code == 200, response.text
    return response.json()


def _close(client: TestClient, seed: ResourcesSeedSessionRequest, episode_id: EpisodeId | None = None):
    close = ResourcesCloseSessionRequest(
        resources_session_id=seed.resources_session_id, episode_id=episode_id or seed.episode_id
    )
    return client.post("/close_session", json=close.model_dump(mode="json"))


def _build(root: Path) -> None:
    """What a builder leaves behind: the two scripts the grader runs, plus trees that never travel."""
    (root / "setup-environment.sh").write_text("npm install\n")
    (root / "start-server.sh").write_text("npm start\n")
    (root / "src").mkdir()
    (root / "src" / "index.js").write_text("app")
    (root / "src" / "node_modules" / "dep").mkdir(parents=True)
    (root / "src" / "node_modules" / "dep" / "index.js").write_text("dependency")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref")


def test_seed_lends_a_sandbox_with_the_brief_and_only_the_prd_assets(server, root):
    client = TestClient(server.setup_webserver())

    seeded = _post_seed(client, _seed())

    assert seeded.status_code == 200, seeded.text
    access = seeded.json()["sandbox_access"]
    assert access["workdir"] == str(root)
    assert access["connection"]["provider_config_ref"] == "sandbox"
    assert (root / "prd.txt").read_text() == "Build a notes app."
    assert (root / "assets" / "icons" / "logo.svg").read_text() == "<svg/>"
    assert not list(root.rglob("upload.png")), "grader-only fixtures never reach the build sandbox"
    assert not list(root.rglob("test*.txt")), "test plans never reach the build sandbox"


def test_verify_harvests_the_app_stops_the_sandbox_and_grades_the_copy(server, root, stopped, graded):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _build(root)

    verified = _verify(client)

    assert (verified["reward"], verified["build_failed"], verified["test_plans_graded"]) == (1.0, False, 2)
    assert verified["reward_components"] == {"test1": 1.0, "test2": 1.0}
    assert len(stopped) == 1, "verify stops the sandbox once the app is out"
    assert graded[0] == {
        # The app may read the PRD's assets at runtime, so they travel with it.
        "assets/icons/logo.svg": "<svg/>",
        "setup-environment.sh": "npm install\n",
        "src/index.js": "app",
        "start-server.sh": "npm start\n",
    }, "dependency and VCS trees and the staged brief are not harvested"


def test_a_repeated_verify_regrades_the_harvested_copy(server, root, stopped, graded):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _build(root)
    _verify(client)

    again = _verify(client)

    assert again["reward"] == 1.0
    assert len(stopped) == 1
    assert graded[0] == graded[-1]


def test_an_app_without_the_grader_scripts_is_a_build_failure(server, root, stopped, graded):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    (root / "index.html").write_text("half an app")

    verified = _verify(client)

    assert (verified["reward"], verified["build_failed"], verified["test_plans_graded"]) == (0.0, True, 0)
    assert graded == [], "nothing is graded"
    assert len(stopped) == 1


def _exec_result(monkeypatch, return_code: int, stderr: str = "", *, run: bool = True, error_type=None) -> None:
    """Make the harvest command report ``return_code``; ``run`` still runs it, so its archive exists."""
    real_exec = AsyncSandbox.exec

    async def exec_(self, command, **kwargs):
        result = await real_exec(self, command, **kwargs) if run else SandboxExecResult("", "", 0)
        return dataclasses.replace(result, return_code=return_code, stderr=stderr, error_type=error_type)

    monkeypatch.setattr(AsyncSandbox, "exec", exec_)


def test_a_failed_download_is_masked_and_a_resent_verify_harvests_again(server, root, stopped, graded, monkeypatch):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _build(root)
    real_download = AsyncSandbox.download
    monkeypatch.setattr(AsyncSandbox, "download", AsyncMock(side_effect=OSError("download failed")))

    verified = _verify(client)

    assert (verified["reward"], verified["mask_sample"], verified["build_failed"]) == (0.0, True, False)
    assert verified["failure_kind"] == PROVIDER_REQUEST_FAILED
    assert "download failed" in verified["failure_reason"]
    assert stopped == [], "the sandbox is kept, so the app can still be harvested"
    assert graded == []

    monkeypatch.setattr(AsyncSandbox, "download", real_download)
    again = _verify(client)

    assert (again["reward"], again["mask_sample"], again["test_plans_graded"]) == (1.0, False, 2)
    assert len(stopped) == 1


def test_a_typed_provider_failure_keeps_its_failure_kind(server, root, stopped, monkeypatch):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _build(root)
    timeout = sandbox_failure(operation="exec", failure_kind=PROVIDER_IO_TIMEOUT, reason="exec timed out")
    monkeypatch.setattr(AsyncSandbox, "exec", AsyncMock(side_effect=timeout))

    verified = _verify(client)

    assert (verified["mask_sample"], verified["failure_kind"]) == (True, PROVIDER_IO_TIMEOUT)
    assert stopped == []


def test_a_command_the_provider_did_not_run_is_masked(server, root, stopped, monkeypatch):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _build(root)
    _exec_result(monkeypatch, 125, run=False, error_type="user_switch_unavailable")

    verified = _verify(client)

    assert (verified["mask_sample"], verified["failure_kind"]) == (True, PROVIDER_REQUEST_FAILED)
    assert stopped == []


def test_a_harvest_command_that_is_not_tar_s_outcome_is_masked(server, root, stopped, monkeypatch):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _build(root)
    _exec_result(monkeypatch, 127, "tar: command not found", run=False)

    verified = _verify(client)

    assert (verified["mask_sample"], verified["failure_kind"]) == (True, VERIFIER_ERROR)
    assert "command not found" in verified["failure_reason"]
    assert stopped == []


def test_files_that_changed_while_tar_read_them_still_grade(server, root, stopped, graded, monkeypatch):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _build(root)
    _exec_result(monkeypatch, 1, "tar: ./server.log: file changed as we read it")

    verified = _verify(client)

    assert (verified["reward"], verified["build_failed"], verified["mask_sample"]) == (1.0, False, False)
    assert len(stopped) == 1


def test_a_fatal_tar_error_is_a_build_failure(server, root, stopped, graded, monkeypatch):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _build(root)
    _exec_result(monkeypatch, 2, "tar: ./secret: Cannot open: Permission denied", run=False)

    verified = _verify(client)

    assert (verified["reward"], verified["build_failed"], verified["mask_sample"]) == (0.0, True, False)
    assert graded == []
    assert len(stopped) == 1


def test_a_missing_app_directory_is_a_build_failure(server, root, stopped, graded, monkeypatch):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    shutil.rmtree(root)

    verified = _verify(client)

    assert (verified["reward"], verified["build_failed"], verified["mask_sample"]) == (0.0, True, False)
    assert len(stopped) == 1


def test_a_grader_config_error_fails_verify_and_removes_its_work_dir(server, root, stopped, temp_root, monkeypatch):
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    _post_seed(client, _seed())
    _build(root)

    async def misconfigured(self, app_dir, test_plan_rel, work_dir, test_assets_dir):
        raise GraderConfigError("no docker-compose")

    monkeypatch.setattr(VibenchResourcesServer, "_grade_one_test_plan", misconfigured)

    assert client.post("/verify", json=_verify_body()).status_code == 500
    assert not list(temp_root.glob("vibench-notes-mvp-*")), "the grading work dir is removed"


def test_close_stops_an_unverified_sandbox_and_fences_late_seeds(server, root, stopped):
    app = server.setup_webserver()
    client = TestClient(app)
    seed = _seed()
    _post_seed(client, seed)

    assert _close(client, seed).status_code == 200

    assert len(stopped) == 1
    assert server._sessions == {}
    late = _post_seed(client, seed)
    assert late.status_code == 409
    assert "already closed" in late.json()["detail"]
    assert _close(client, seed).status_code == 200, "a repeated close of the same episode is idempotent"
    wrong = _close(client, seed, episode_id=EpisodeId(rollout_id="0-1"))
    assert wrong.status_code == 409


def test_a_verify_after_close_is_masked_as_a_lost_session(server, root, stopped, graded):
    client = TestClient(server.setup_webserver())
    seed = _seed()
    _post_seed(client, seed)
    _build(root)
    assert _close(client, seed).status_code == 200

    verified = _verify(client)

    assert (verified["reward"], verified["mask_sample"], verified["failure_kind"]) == (0.0, True, SESSION_LOST)
    assert graded == []


def test_close_after_verify_does_not_stop_the_sandbox_twice(server, root, stopped, graded):
    client = TestClient(server.setup_webserver())
    seed = _seed()
    _post_seed(client, seed)
    _build(root)
    _verify(client)

    assert _close(client, seed).status_code == 200

    assert len(stopped) == 1


class _Gate:
    """Holds whatever awaits it until released, and tells the test when something has reached it."""

    def __init__(self) -> None:
        self.reached = asyncio.Event()
        self.released = asyncio.Event()

    async def wait(self) -> None:
        self.reached.set()
        await self.released.wait()


def _async_client(server: VibenchResourcesServer) -> httpx.AsyncClient:
    """A client whose requests can overlap, unlike TestClient's; it keeps the session cookie the same way."""
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.setup_webserver()), base_url="http://testserver")


def test_a_close_during_verify_waits_for_the_harvest_and_leaves_the_copy_being_graded(
    server, root, stopped, graded, monkeypatch
):
    real_download = AsyncSandbox.download
    real_grade = VibenchResourcesServer._grade

    async def scenario() -> dict:
        harvesting, grading = _Gate(), _Gate()

        async def download(self, *args, **kwargs):
            await harvesting.wait()
            return await real_download(self, *args, **kwargs)

        async def grade(self, *args, **kwargs):
            await grading.wait()
            return await real_grade(self, *args, **kwargs)

        monkeypatch.setattr(AsyncSandbox, "download", download)
        monkeypatch.setattr(VibenchResourcesServer, "_grade", grade)
        seed = _seed()
        close = ResourcesCloseSessionRequest(
            resources_session_id=seed.resources_session_id, episode_id=seed.episode_id
        )
        async with _async_client(server) as client:
            await client.post("/seed_session", json=seed.model_dump(mode="json"))
            _build(root)
            verifying = asyncio.create_task(client.post("/verify", json=_verify_body()))
            await harvesting.reached.wait()
            closing = asyncio.create_task(client.post("/close_session", json=close.model_dump(mode="json")))
            await asyncio.sleep(0.1)
            assert not closing.done(), "close waits while the app is harvested"

            harvesting.released.set()
            await grading.reached.wait()
            # The close removes the session's copy before grading reads anything.
            assert (await closing).status_code == 200
            assert server._sessions == {}
            grading.released.set()
            verified = await verifying
        assert verified.status_code == 200, verified.text
        return verified.json()

    verified = asyncio.run(scenario())

    assert (verified["reward"], verified["build_failed"], verified["mask_sample"]) == (1.0, False, False)
    assert len(stopped) == 1


def test_two_verifies_at_once_harvest_once_and_both_grade(server, root, stopped, graded, monkeypatch):
    real_exec = AsyncSandbox.exec

    async def scenario() -> list[dict]:
        harvesting = _Gate()
        harvests: list[str] = []

        async def exec_(self, command, **kwargs):
            harvests.append(command)
            await harvesting.wait()
            return await real_exec(self, command, **kwargs)

        async with _async_client(server) as client:
            await client.post("/seed_session", json=_seed().model_dump(mode="json"))
            _build(root)
            monkeypatch.setattr(AsyncSandbox, "exec", exec_)
            first = asyncio.create_task(client.post("/verify", json=_verify_body()))
            await harvesting.reached.wait()
            second = asyncio.create_task(client.post("/verify", json=_verify_body()))
            await asyncio.sleep(0.1)
            assert len(harvests) == 1, "the second verify waits instead of tarring into the same file"

            harvesting.released.set()
            responses = await asyncio.gather(first, second)
        assert len(harvests) == 1
        return [response.json() for response in responses]

    verified = asyncio.run(scenario())

    assert [(v["reward"], v["build_failed"]) for v in verified] == [(1.0, False), (1.0, False)]
    assert len(stopped) == 1
    assert len(graded) == 4 and all(tree == graded[0] for tree in graded), "both graded the same app"


def test_a_close_whose_stop_fails_is_an_error_and_can_be_retried(server, root, monkeypatch):
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    seed = _seed()
    _post_seed(client, seed)
    monkeypatch.setattr(AsyncSandbox, "stop", AsyncMock(side_effect=[OSError("stop failed"), None]))

    assert _close(client, seed).status_code == 500
    assert _close(client, seed).status_code == 200
    assert server._sessions == {}


def test_a_stop_that_fails_after_the_harvest_still_grades_and_close_stops_the_sandbox(
    server, root, graded, monkeypatch
):
    client = TestClient(server.setup_webserver())
    seed = _seed()
    _post_seed(client, seed)
    _build(root)
    stop = AsyncMock(side_effect=[OSError("stop failed"), None])
    monkeypatch.setattr(AsyncSandbox, "stop", stop)

    verified = _verify(client)

    assert (verified["reward"], verified["build_failed"], verified["mask_sample"]) == (1.0, False, False)
    assert _close(client, seed).status_code == 200
    assert stop.await_count == 2, "close stops the sandbox the verify could not"
    assert server._sessions == {}


def test_seeding_again_is_idempotent_for_its_episode_and_refused_for_another(server, root, stopped):
    client = TestClient(server.setup_webserver())
    seed = _seed()
    assert _post_seed(client, seed).status_code == 200

    assert _post_seed(client, seed).status_code == 200
    assert len(server._sessions) == 1
    other = _post_seed(client, seed.model_copy(update={"episode_id": EpisodeId(rollout_id="0-1")}))
    assert other.status_code == 409
    assert "already bound to another episode or task" in other.json()["detail"]
    assert stopped == []


@pytest.mark.parametrize(
    "update, message",
    [
        ({"app": None}, "not a ViBench task"),
        ({"prd_files": ["../../etc/passwd"]}, "escapes vibench_repo_root"),
        ({"prd_files": ["v1/prds/notes/prd/missing.txt"]}, "missing.txt"),
    ],
    ids=["not a task", "escaping path", "missing PRD"],
)
def test_a_row_that_no_retry_can_fix_is_refused_as_final_without_a_sandbox(server, root, update, message):
    client = TestClient(server.setup_webserver())

    refused = _post_seed(client, _seed(**update))

    assert refused.status_code == 422
    assert message in refused.json()["detail"]
    assert not root.exists(), "no sandbox is started for a row that is not a task"
    assert _post_seed(client, _seed()).status_code == 200, "a refusal does not fence the ID"


def _fail_asset_uploads(monkeypatch, error: Exception) -> None:
    """Fail the asset uploads only; start() delivers the PRD through upload() too."""
    real_upload = AsyncSandbox.upload

    async def upload(self: AsyncSandbox, local_path, remote_path: str) -> None:
        if "/assets/" in remote_path:
            raise error
        await real_upload(self, local_path, remote_path)

    monkeypatch.setattr(AsyncSandbox, "upload", upload)


def test_a_failed_asset_upload_stops_the_sandbox(server, root, monkeypatch, stopped):
    _fail_asset_uploads(monkeypatch, OSError("upload failed"))

    with pytest.raises(OSError, match="upload failed"):
        _post_seed(TestClient(server.setup_webserver()), _seed())

    assert len(stopped) == 1
    assert server._sessions == {}


def test_a_seed_without_a_build_image_names_the_setting(server, root, temp_root, monkeypatch):
    monkeypatch.setattr(server.config.sandbox_config, "image", None)

    with pytest.raises(ValueError, match="VIBENCH_BUILD_IMAGE"):
        _post_seed(TestClient(server.setup_webserver()), _seed())

    assert not root.exists()
    assert list(temp_root.iterdir()) == [], "a failed start leaves no scratch directory"


def test_a_row_naming_a_path_as_its_app_stays_in_the_temporary_directory(server, root, temp_root):
    seeded = _post_seed(TestClient(server.setup_webserver()), _seed(app="notes/../../escape"))

    assert seeded.status_code == 200, seeded.text
    assert [path.parent for path in temp_root.parent.rglob("vibench-build-*")] == [temp_root]


def test_a_session_that_is_gone_is_masked_instead_of_scored_as_a_failed_build(server, root, graded):
    verified = _verify(TestClient(server.setup_webserver()))

    assert (verified["reward"], verified["mask_sample"], verified["failure_kind"]) == (0.0, True, SESSION_LOST)
    assert verified["test_plans_total"] == 2
    assert graded == []


def test_shutdown_stops_the_sandboxes_of_open_sessions(server, root, stopped):
    with TestClient(server.setup_webserver()) as client:
        _post_seed(client, _seed())
        assert stopped == []

    assert len(stopped) == 1


def test_sessions_require_a_single_worker(checkout, root):
    with pytest.raises(ValueError, match="num_workers=1"):
        _server(checkout, root, num_workers=2)


def test_close_follows_the_session_contract(server, root, graded):
    app = server.setup_webserver()

    check_resources_session_contract(
        app,
        _seed(session_id="contract"),
        keeps_state=True,
        holds_state=lambda resources_session_id: resources_session_id in server._sessions,
        verify_body=_verify_body(),
    )


@pytest.mark.parametrize("operation", ["upload", "serialize"])
def test_failed_setup_retains_cleanup_until_close(server, root, monkeypatch, operation):
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    seed = _seed()
    if operation == "upload":
        _fail_asset_uploads(monkeypatch, OSError("setup failed"))
    else:
        monkeypatch.setattr(AsyncSandbox, operation, AsyncMock(side_effect=OSError("setup failed")))
    stop = AsyncMock(side_effect=[OSError("stop failed"), None])
    monkeypatch.setattr(AsyncSandbox, "stop", stop)

    assert _post_seed(client, seed).status_code == 500
    session = server._sessions[seed.resources_session_id]
    assert session.sandbox is not None and not session.ready
    assert _post_seed(client, seed).status_code == 409
    assert _verify(client)["mask_sample"] is True
    assert _close(client, seed).status_code == 200
    assert stop.await_count == 2
    assert server._sessions == {}
    assert not session.scratch_dir.exists()


def test_retirement_retries_failed_teardown_and_fences_session(server, root, monkeypatch):
    client = TestClient(server.setup_webserver())
    seed = _seed()
    _post_seed(client, seed)
    session = server._sessions[seed.resources_session_id]
    stop = AsyncMock(side_effect=[OSError("stop failed"), None])
    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    with pytest.raises(OSError, match="stop failed"):
        asyncio.run(server.retire_session_state(seed.resources_session_id))
    assert server._sessions[seed.resources_session_id] is session
    asyncio.run(server.retire_session_state(seed.resources_session_id))
    asyncio.run(server.retire_session_state(seed.resources_session_id))
    assert stop.await_count == 2
    assert not session.scratch_dir.exists()
    assert server._sessions == {}
    assert _post_seed(client, seed).status_code == 409


def test_harvested_session_cannot_be_lent_when_stop_failed(server, root, graded, monkeypatch):
    client = TestClient(server.setup_webserver())
    seed = _seed()
    _post_seed(client, seed)
    _build(root)
    monkeypatch.setattr(AsyncSandbox, "stop", AsyncMock(side_effect=OSError("stop failed")))
    assert _verify(client)["reward"] == 1.0
    assert _post_seed(client, seed).status_code == 409


def test_failed_initial_prd_delivery_retains_cleanup(server, root, monkeypatch):
    from nemo_gym.sandbox import LocalProvider

    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    seed = _seed()
    monkeypatch.setattr(LocalProvider, "upload_file", AsyncMock(side_effect=OSError("delivery failed")))
    stop = AsyncMock(side_effect=[OSError("stop failed"), OSError("stop failed"), None])
    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    assert _post_seed(client, seed).status_code == 500
    session = server._sessions[seed.resources_session_id]
    assert session.sandbox is not None and not session.ready
    assert _post_seed(client, seed).status_code == 409
    assert _close(client, seed).status_code == 200
    assert stop.await_count == 3
    assert not session.scratch_dir.exists()
