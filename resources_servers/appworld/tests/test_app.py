# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the AppWorld resources server.

Most tests drive the server against a fake worker pool, so the episode
lifecycle, scoring and failure paths are exercised without spawning AppWorld.
The end of the file holds one real end-to-end episode (skipped when AppWorld
isn't set up locally) that replays a gold solution and asserts reward 1.0 —
the check that the whole init/execute/evaluate path is wired correctly.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from nemo_gym.base_resources_server import ReverifyMode
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.appworld import app as app_module
from resources_servers.appworld.app import AppWorldResourcesServer, execute_tool_param
from resources_servers.appworld.schemas import (
    AppWorldCloseRequest,
    AppWorldNeMoGymResponse,
    AppWorldResourcesServerConfig,
    AppWorldScoring,
    AppWorldSeedSessionRequest,
    AppWorldStepRequest,
    AppWorldVerifyRequest,
)
from resources_servers.appworld.setup_appworld import (
    DEFAULT_APPWORLD_ROOT,
    DEFAULT_APPWORLD_VENV,
    AppWorldInstall,
    data_exists,
    is_installed,
    load_task_ids,
    venv_executable,
)
from resources_servers.appworld.worker_pool import AppWorldWorker, AppWorldWorkerPool, _free_port


TASK = {
    "task_id": "82e2fac_1",
    "instruction": "What is the title of the most-liked song in my Spotify playlists.",
    "supervisor": {
        "first_name": "Joyce",
        "last_name": "Weaver",
        "email": "joyce-weav@gmail.com",
        "phone_number": "3155673041",
    },
    "datetime": "2023-05-18T12:00:00",
}

PASSING_REPORT = {
    "success": True,
    "difficulty": 1,
    "num_tests": 2,
    "passes": [{"requirement": "assert no model changes."}, {"requirement": "assert answers match."}],
    "failures": [],
}

FAILING_REPORT = {
    "success": False,
    "difficulty": 2,
    "num_tests": 2,
    "passes": [{"requirement": "assert no model changes."}],
    "failures": [{"requirement": "assert answers match.", "trace": "AssertionError"}],
}


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeWorker:
    """Stands in for one ``appworld serve environment`` process."""

    def __init__(self, responses: Optional[Dict[str, Any]] = None, index: int = 0) -> None:
        self.index = index
        self.calls: List[tuple[str, Dict[str, Any]]] = []
        self.responses: Dict[str, Any] = {
            "/initialize": TASK,
            "/execute": "Execution successful.",
            "/task_completed": False,
            "/evaluate": FAILING_REPORT,
            "/close": None,
            **(responses or {}),
        }
        self.errors: Dict[str, Exception] = {}

    async def call(self, url_path: str, payload: Dict[str, Any]) -> Any:
        self.calls.append((url_path, payload))
        if url_path in self.errors:
            raise self.errors[url_path]
        value = self.responses[url_path]
        return value(payload) if callable(value) else value

    def paths(self) -> List[str]:
        return [path for path, _ in self.calls]


class FakePool:
    def __init__(self, worker: FakeWorker) -> None:
        self.worker = worker
        self.acquired = 0
        self.released: List[FakeWorker] = []

    async def acquire(self) -> FakeWorker:
        self.acquired += 1
        return self.worker

    async def release(self, worker: FakeWorker) -> None:
        self.released.append(worker)


def make_server(tmp_path: Path, worker: Optional[FakeWorker] = None, **overrides: Any):
    """A server wired to a fake pool, with setup stubbed out."""
    config = AppWorldResourcesServerConfig(
        name="appworld",
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        **overrides,
    )
    install = AppWorldInstall(root=str(tmp_path), executable=str(tmp_path / "bin" / "appworld"))
    original = app_module.ensure_appworld
    app_module.ensure_appworld = lambda *args, **kwargs: install
    try:
        server = AppWorldResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    finally:
        app_module.ensure_appworld = original
    pool = FakePool(worker or FakeWorker())
    server._pool = pool
    return server, pool


def verify_request(env_id: str, task_id: str = "82e2fac_1") -> AppWorldVerifyRequest:
    response = AppWorldNeMoGymResponse(
        id="r",
        created_at=0.0,
        model="policy_model",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        env_id=env_id,
        task_id=task_id,
    )
    return AppWorldVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=response,
    )


async def seed(server, task_id: str = "82e2fac_1"):
    return await server.seed_session(AppWorldSeedSessionRequest(task_id=task_id))


# ---------------------------------------------------------------------------
# seed_session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seed_session_returns_prompt_instruction_and_tool(tmp_path):
    server, pool = make_server(tmp_path)

    response = await seed(server)

    assert pool.acquired == 1
    assert response.task_id == "82e2fac_1"
    assert [message.role for message in response.obs] == ["system", "user"]
    assert "execute_ipython_code" in response.obs[0].content
    # The instruction and supervisor identity come from AppWorld, not the row.
    assert "most-liked song" in response.obs[1].content
    assert "Joyce Weaver" in response.obs[1].content
    assert "joyce-weav@gmail.com" in response.obs[1].content
    assert [tool["name"] for tool in response.tools] == ["execute_ipython_code"]
    assert response.env_id in server.envs


@pytest.mark.asyncio
async def test_seed_session_passes_episode_config_to_appworld(tmp_path):
    worker = FakeWorker()
    server, _ = make_server(tmp_path, worker, max_interactions=7, execution_timeout_secs=42)

    await seed(server)

    _, payload = worker.calls[0]
    assert payload["task_id"] == "82e2fac_1"
    assert payload["max_interactions"] == 7
    assert payload["timeout_seconds"] == 42
    assert payload["raise_on_unsafe_syntax"] is True
    # Unique per episode: AppWorld rmtree's the experiment's task directory on
    # init, so concurrent repeats of one task must not share a name.
    assert payload["experiment_name"].startswith("nemo_gym_")


@pytest.mark.asyncio
async def test_seed_session_uses_prompt_overrides(tmp_path):
    server, _ = make_server(
        tmp_path,
        system_prompt="custom system",
        instruction_template="{supervisor_first_name}|{instruction}",
    )

    response = await seed(server)

    assert response.obs[0].content == "custom system"
    assert response.obs[1].content.startswith("Joyce|What is the title")


@pytest.mark.asyncio
async def test_seed_session_releases_worker_when_initialize_fails(tmp_path):
    worker = FakeWorker()
    worker.errors["/initialize"] = RuntimeError("boom")
    server, pool = make_server(tmp_path, worker)

    with pytest.raises(RuntimeError):
        await seed(server)

    assert pool.released == [worker]
    assert server.envs == {}


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_step_executes_code_and_continues(tmp_path):
    worker = FakeWorker({"/execute": "hello"})
    server, _ = make_server(tmp_path, worker)
    env_id = (await seed(server)).env_id

    response = await server.step(AppWorldStepRequest(env_id=env_id, code="print('hello')"))

    assert response.output == "hello"
    assert response.done is False
    assert response.num_interactions == 1
    assert response.reward == 0.0
    assert ("/execute", {"task_id": "82e2fac_1", "code": "print('hello')"}) in worker.calls


@pytest.mark.asyncio
async def test_step_done_when_complete_task_called(tmp_path):
    worker = FakeWorker({"/task_completed": True})
    server, _ = make_server(tmp_path, worker)
    env_id = (await seed(server)).env_id

    response = await server.step(AppWorldStepRequest(env_id=env_id, code="apis.supervisor.complete_task()"))

    assert response.done is True


@pytest.mark.asyncio
async def test_step_done_when_interaction_budget_is_spent(tmp_path):
    server, _ = make_server(tmp_path, max_interactions=2)
    env_id = (await seed(server)).env_id

    first = await server.step(AppWorldStepRequest(env_id=env_id, code="1"))
    second = await server.step(AppWorldStepRequest(env_id=env_id, code="2"))

    assert first.done is False
    assert second.done is True
    assert second.num_interactions == 2


@pytest.mark.asyncio
async def test_step_after_episode_end_is_a_noop(tmp_path):
    worker = FakeWorker({"/task_completed": True})
    server, _ = make_server(tmp_path, worker)
    env_id = (await seed(server)).env_id
    await server.step(AppWorldStepRequest(env_id=env_id, code="apis.supervisor.complete_task()"))

    response = await server.step(AppWorldStepRequest(env_id=env_id, code="print(1)"))

    assert response.done is True
    assert "already ended" in response.output
    assert worker.paths().count("/execute") == 1


@pytest.mark.asyncio
async def test_step_with_unknown_env_id_raises(tmp_path):
    server, _ = make_server(tmp_path)

    with pytest.raises(KeyError):
        await server.step(AppWorldStepRequest(env_id="nope", code="print(1)"))


@pytest.mark.asyncio
async def test_step_survives_a_dead_worker(tmp_path):
    worker = FakeWorker()
    server, _ = make_server(tmp_path, worker)
    env_id = (await seed(server)).env_id
    worker.errors["/execute"] = ConnectionError("worker gone")

    response = await server.step(AppWorldStepRequest(env_id=env_id, code="print(1)"))

    assert response.done is True
    assert "Execution environment error" in response.output
    assert server.envs[env_id]["broken"] is True


# ---------------------------------------------------------------------------
# close / verify
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_scores_a_solved_task_and_frees_the_worker(tmp_path):
    worker = FakeWorker({"/evaluate": PASSING_REPORT})
    server, pool = make_server(tmp_path, worker)
    env_id = (await seed(server)).env_id
    await server.step(AppWorldStepRequest(env_id=env_id, code="print(1)"))

    response = await server.close(AppWorldCloseRequest(env_id=env_id))

    assert response.success is True
    scoring = server.scoring[env_id]
    assert scoring.reward == 1.0
    assert scoring.success is True
    assert scoring.num_passed == 2
    assert scoring.num_tests == 2
    assert scoring.partial_credit == 1.0
    assert scoring.difficulty == 1
    assert scoring.num_interactions == 1
    assert scoring.failed_requirements == []
    assert pool.released == [worker]
    assert env_id not in server.envs
    assert worker.paths()[-1] == "/close"


@pytest.mark.asyncio
async def test_close_scores_a_failed_task_as_zero(tmp_path):
    server, _ = make_server(tmp_path)
    env_id = (await seed(server)).env_id

    await server.close(AppWorldCloseRequest(env_id=env_id))

    scoring = server.scoring[env_id]
    assert scoring.reward == 0.0
    assert scoring.success is False
    assert scoring.partial_credit == 0.5
    assert scoring.failed_requirements == ["assert answers match."]


@pytest.mark.asyncio
async def test_dense_reward_uses_partial_credit(tmp_path):
    server, _ = make_server(tmp_path, dense_reward=True)
    env_id = (await seed(server)).env_id

    await server.close(AppWorldCloseRequest(env_id=env_id))

    assert server.scoring[env_id].reward == 0.5


@pytest.mark.asyncio
async def test_close_reports_evaluation_failure_as_zero(tmp_path):
    worker = FakeWorker()
    worker.errors["/evaluate"] = RuntimeError("evaluate blew up")
    server, pool = make_server(tmp_path, worker)
    env_id = (await seed(server)).env_id

    await server.close(AppWorldCloseRequest(env_id=env_id))

    scoring = server.scoring[env_id]
    assert scoring.reward == 0.0
    assert "evaluate blew up" in scoring.evaluation_error
    # Still returned to the pool: a scoring failure must not leak a worker.
    assert pool.released == [worker]


@pytest.mark.asyncio
async def test_close_skips_evaluation_for_a_broken_episode(tmp_path):
    worker = FakeWorker()
    server, _ = make_server(tmp_path, worker)
    env_id = (await seed(server)).env_id
    worker.errors["/execute"] = ConnectionError("worker gone")
    await server.step(AppWorldStepRequest(env_id=env_id, code="print(1)"))

    await server.close(AppWorldCloseRequest(env_id=env_id))

    assert server.scoring[env_id].evaluation_error == "worker_failure"
    assert "/evaluate" not in worker.paths()


@pytest.mark.asyncio
async def test_close_tolerates_a_worker_that_fails_to_close(tmp_path):
    worker = FakeWorker({"/evaluate": PASSING_REPORT})
    worker.errors["/close"] = RuntimeError("already gone")
    server, pool = make_server(tmp_path, worker)
    env_id = (await seed(server)).env_id

    response = await server.close(AppWorldCloseRequest(env_id=env_id))

    assert response.success is True
    assert server.scoring[env_id].reward == 1.0
    assert pool.released == [worker]


@pytest.mark.asyncio
async def test_close_is_idempotent(tmp_path):
    server, _ = make_server(tmp_path)
    env_id = (await seed(server)).env_id
    await server.close(AppWorldCloseRequest(env_id=env_id))

    response = await server.close(AppWorldCloseRequest(env_id=env_id))

    assert response.success is False


@pytest.mark.asyncio
async def test_close_removes_the_episode_output_directory(tmp_path):
    server, _ = make_server(tmp_path)
    env_id = (await seed(server)).env_id
    experiment = server.envs[env_id]["experiment_name"]
    output_dir = tmp_path / "experiments" / "outputs" / experiment
    (output_dir / "tasks").mkdir(parents=True)

    await server.close(AppWorldCloseRequest(env_id=env_id))

    assert not output_dir.exists()


@pytest.mark.asyncio
async def test_close_keeps_outputs_when_cleanup_disabled(tmp_path):
    server, _ = make_server(tmp_path, cleanup_experiment_outputs=False)
    env_id = (await seed(server)).env_id
    output_dir = tmp_path / "experiments" / "outputs" / server.envs[env_id]["experiment_name"]
    output_dir.mkdir(parents=True)

    await server.close(AppWorldCloseRequest(env_id=env_id))

    assert output_dir.exists()


@pytest.mark.asyncio
async def test_verify_returns_and_consumes_the_cached_scoring(tmp_path):
    worker = FakeWorker({"/evaluate": PASSING_REPORT})
    server, _ = make_server(tmp_path, worker)
    env_id = (await seed(server)).env_id
    await server.close(AppWorldCloseRequest(env_id=env_id))

    response = await server.verify(verify_request(env_id))

    assert response.reward == 1.0
    assert response.success is True
    assert response.num_passed == 2
    assert response.difficulty == 1
    assert env_id not in server.scoring


@pytest.mark.asyncio
async def test_verify_without_scoring_returns_zero(tmp_path):
    server, _ = make_server(tmp_path)

    response = await server.verify(verify_request("never-seeded"))

    assert response.reward == 0.0
    assert response.evaluation_error == "missing_scoring"


@pytest.mark.asyncio
async def test_reaper_closes_an_abandoned_episode(tmp_path, monkeypatch):
    worker = FakeWorker({"/evaluate": PASSING_REPORT})
    server, pool = make_server(tmp_path, worker)
    abandoned = (await seed(server)).env_id
    monkeypatch.setattr(app_module, "EPISODE_TIMEOUT_SECS", -1.0)

    await seed(server)

    assert abandoned not in server.envs
    assert server.scoring[abandoned].evaluation_error == "worker_failure"
    assert worker in pool.released


def test_scoring_from_report_rejects_an_unexpected_payload(tmp_path):
    server, _ = make_server(tmp_path)

    scoring = server._scoring_from_report("not a dict", AppWorldScoring())

    assert scoring.reward == 0.0
    assert "unexpected evaluation payload" in scoring.evaluation_error


def test_scoring_infers_num_tests_when_absent(tmp_path):
    server, _ = make_server(tmp_path)

    scoring = server._scoring_from_report({"success": False, "passes": [{}], "failures": [{}, {}]}, AppWorldScoring())

    assert scoring.num_tests == 3
    assert scoring.num_passed == 1
    assert scoring.difficulty is None


# ---------------------------------------------------------------------------
# metrics & config
# ---------------------------------------------------------------------------


def test_scenario_goal_completion_requires_every_variant(tmp_path):
    server, _ = make_server(tmp_path)
    tasks = [
        [{"task_id": "aaa_1", "success": True}],
        [{"task_id": "aaa_2", "success": True}],
        [{"task_id": "aaa_3", "success": False}],  # drags scenario aaa to 0
        [{"task_id": "bbb_1", "success": True}],
        [{"task_id": "bbb_2", "success": True}],
    ]

    metrics = server.compute_metrics(tasks)

    assert metrics == {"mean/scenario_goal_completion": 0.5}


def test_scenario_goal_completion_keeps_repeats_independent(tmp_path):
    server, _ = make_server(tmp_path)
    tasks = [
        [{"task_id": "aaa_1", "success": True}, {"task_id": "aaa_1", "success": False}],
        [{"task_id": "aaa_2", "success": True}, {"task_id": "aaa_2", "success": True}],
    ]

    metrics = server.compute_metrics(tasks)

    # Repeat 0 solved both variants; repeat 1 did not.
    assert metrics == {"mean/scenario_goal_completion": 0.5}


def test_scenario_goal_completion_reads_task_id_from_the_response(tmp_path):
    server, _ = make_server(tmp_path)

    metrics = server.compute_metrics([[{"response": {"task_id": "aaa_1"}, "success": True}]])

    assert metrics == {"mean/scenario_goal_completion": 1.0}


def test_scenario_goal_completion_absent_without_task_ids(tmp_path):
    server, _ = make_server(tmp_path)

    assert server.compute_metrics([[{"success": True}]]) == {}
    assert server.compute_metrics([]) == {}


def test_execute_tool_param_shape():
    tool = execute_tool_param()

    assert tool["name"] == "execute_ipython_code"
    assert tool["parameters"]["required"] == ["code"]
    assert tool["parameters"]["properties"]["code"]["type"] == "string"


def test_pool_size_does_not_shadow_gyms_uvicorn_worker_count(tmp_path):
    """`num_workers` is gym's uvicorn fork count on BaseServerConfig.

    Naming the pool size `num_workers` forks this server into N processes, each
    with its own worker pool and episode table — episodes then land on whichever
    process handles the request and `/step` fails with an unknown env_id.
    """
    server, _ = make_server(tmp_path, num_env_workers=3)

    assert server.config.num_env_workers == 3
    assert server.config.num_workers is None


def test_reverify_is_unsupported_because_scoring_needs_live_state():
    assert AppWorldResourcesServerConfig.REVERIFY_MODE == ReverifyMode.UNSUPPORTED


def test_setup_webserver_exposes_step_and_close(tmp_path):
    server, _ = make_server(tmp_path)

    routes = {route.path for route in server.setup_webserver().routes}

    assert {"/seed_session", "/step", "/close", "/verify"} <= routes


def test_pool_is_built_lazily_from_the_install(tmp_path):
    server, _ = make_server(tmp_path)
    server._pool = None

    pool = server.pool

    assert isinstance(pool, AppWorldWorkerPool)
    assert pool.root == str(tmp_path)
    assert pool.executable.endswith("appworld")
    assert server.pool is pool  # cached


# ---------------------------------------------------------------------------
# worker pool internals
# ---------------------------------------------------------------------------


def test_free_port_falls_back_to_an_ephemeral_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        occupied = taken.getsockname()[1]

        assert _free_port(occupied) != occupied


def test_worker_pool_rejects_an_empty_pool():
    with pytest.raises(ValueError):
        AppWorldWorkerPool(
            num_workers=0,
            port_start=19000,
            root="/tmp",
            executable="appworld",
            startup_timeout_secs=1.0,
            request_timeout_secs=1.0,
        )


@pytest.mark.asyncio
async def test_worker_call_on_a_dead_process_raises(tmp_path):
    from resources_servers.appworld.worker_pool import AppWorldWorkerError

    worker = AppWorldWorker(
        index=0,
        port=19999,
        root=str(tmp_path),
        executable="appworld",
        log_fpath=tmp_path / "worker.log",
        request_timeout_secs=1.0,
    )

    with pytest.raises(AppWorldWorkerError):
        await worker.call("/execute", {})


@pytest.mark.asyncio
async def test_terminating_a_never_spawned_worker_is_a_noop(tmp_path):
    worker = AppWorldWorker(
        index=0,
        port=19999,
        root=str(tmp_path),
        executable="appworld",
        log_fpath=tmp_path / "worker.log",
        request_timeout_secs=1.0,
    )

    await worker.terminate()

    assert worker.alive is False


# ---------------------------------------------------------------------------
# setup helpers
# ---------------------------------------------------------------------------


def test_load_task_ids_reads_the_split_file(tmp_path):
    split_dir = tmp_path / "data" / "datasets"
    split_dir.mkdir(parents=True)
    (split_dir / "train.txt").write_text("aaa_1\n\nbbb_2\n")

    assert load_task_ids(tmp_path, "train") == ["aaa_1", "bbb_2"]


def test_load_task_ids_without_data_explains_how_to_fix_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="ensure_appworld"):
        load_task_ids(tmp_path, "train")


def test_data_exists_tracks_the_split_file(tmp_path):
    assert data_exists(tmp_path) is False
    (tmp_path / "data" / "datasets").mkdir(parents=True)
    (tmp_path / "data" / "datasets" / "train.txt").write_text("aaa_1\n")
    assert data_exists(tmp_path) is True


def test_is_installed_is_false_without_a_venv(tmp_path):
    assert is_installed(tmp_path) is False


def test_venv_executable_layout(tmp_path):
    assert venv_executable(tmp_path, "appworld") == tmp_path / "bin" / "appworld"


def test_defaults_live_inside_the_server_directory():
    server_dir = Path(app_module.__file__).resolve().parent
    assert DEFAULT_APPWORLD_ROOT.parent == server_dir
    assert DEFAULT_APPWORLD_VENV.parent == server_dir


# ---------------------------------------------------------------------------
# end-to-end (needs a real AppWorld install)
# ---------------------------------------------------------------------------


APPWORLD_READY = is_installed(DEFAULT_APPWORLD_VENV) and data_exists(DEFAULT_APPWORLD_ROOT)


@pytest.mark.skipif(not APPWORLD_READY, reason="AppWorld is not installed/downloaded locally")
@pytest.mark.asyncio
async def test_end_to_end_gold_solution_scores_one():
    """Replay a train task's gold solution through the real server: reward 1.0.

    This is the check that ties the whole path together — worker startup,
    /initialize, /execute against the live IPython shell, complete_task
    detection, and AppWorld's database-state evaluation.
    """
    config = AppWorldResourcesServerConfig(
        name="appworld",
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        num_env_workers=1,
    )
    server = AppWorldResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    task_id = load_task_ids(server.install.root, "train")[0]
    solution = (
        Path(server.install.root) / "data" / "tasks" / task_id / "ground_truth" / "compiled_solution.py"
    ).read_text()

    try:
        seeded = await server.seed_session(AppWorldSeedSessionRequest(task_id=task_id))
        # The gold file defines solution(apis, requester); run it in the shell.
        first = await server.step(AppWorldStepRequest(env_id=seeded.env_id, code=solution))
        assert first.done is False
        second = await server.step(AppWorldStepRequest(env_id=seeded.env_id, code="solution(apis, requester)"))
        assert second.done is True, second.output

        await server.close(AppWorldCloseRequest(env_id=seeded.env_id))
        verified = await server.verify(verify_request(seeded.env_id, task_id))
    finally:
        await server.pool.stop()

    assert verified.reward == 1.0
    assert verified.success is True
    assert verified.num_passed == verified.num_tests > 0
