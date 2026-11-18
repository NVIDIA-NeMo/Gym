# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The task sandbox of an AA-Briefcase-Lite episode, run on the local sandbox provider in a temporary directory."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.server_runtime import ResourcesSeedSessionRequest
from nemo_gym.verification import JudgeError, judge_failsafe
from resources_servers.aa_briefcase_lite.app import (
    AABriefcaseLiteResourcesServer,
    AABriefcaseLiteResourcesServerConfig,
)
from resources_servers.aa_briefcase_lite.tests.conftest import verify_body
from resources_servers.gdpval import persisted_layout, task_sandbox
from resources_servers.gdpval.app import FinishRequest


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """The sandbox working directory on the host; the local provider runs commands there."""
    home = tmp_path / "home" / "user"
    monkeypatch.setattr(AABriefcaseLiteResourcesServer, "_sandbox_workdir", str(home))
    monkeypatch.setattr(task_sandbox, "get_global_config_dict", lambda: {"sandbox": {"local": {}}})
    monkeypatch.setattr(AsyncSandbox, "serialize", AsyncMock(return_value={"sandbox_id": "local"}))
    return home


@pytest.fixture
def dataset(tmp_path) -> Path:
    root = tmp_path / "dataset"
    (root / "source_files/shared/nested").mkdir(parents=True)
    (root / "source_files/week").mkdir(parents=True)
    (root / "source_files/shared/nested/policy.txt").write_text("shared input")
    (root / "source_files/week/notes.txt").write_text("week input")
    return root


@pytest.fixture
def server(tmp_path, dataset, monkeypatch) -> AABriefcaseLiteResourcesServer:
    monkeypatch.setattr(AABriefcaseLiteResourcesServer, "model_post_init", lambda self, context: None)
    config = AABriefcaseLiteResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        dataset_dir=str(dataset),
        judge_model_server={"type": "responses_api_models", "name": "judge"},
        preconvert_office_to_pdf=False,
        sandbox_provider="sandbox",
        sandbox_config={"image": "unused-by-the-local-provider"},
        persist_deliverables_dir=str(tmp_path / "persist"),
    )
    server = AABriefcaseLiteResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    server._aa_dataset_root = dataset
    # The skipped model_post_init also skips GDPVal's session bookkeeping.
    from resources_servers.gdpval.app import GDPValResourcesServer

    GDPValResourcesServer.model_post_init(server, None)
    return server


async def test_inputs_are_staged_under_the_working_directory_and_only_submissions_are_persisted(
    server, home, tmp_path
) -> None:
    request = SimpleNamespace(session={})
    seed = ResourcesSeedSessionRequest(
        resources_session_id="session",
        episode_id=EpisodeId(rollout_id="0-0"),
        task_id=TaskId(taskset="aa_briefcase_lite:benchmark", task_id="w1_t1"),
        task_data={"task_id": "w1_t1"},
    )
    response = await server.seed_session(request, seed)

    assert response.sandbox_access.workdir == str(home)
    assert (home / "shared/nested/policy.txt").read_text() == "shared input"
    assert (home / "week/notes.txt").read_text() == "week input"

    (home / "report.md").write_text("the deliverable")
    await server.finish(request, FinishRequest(reason="done", paths=[str(home / "report.md")]))
    session = server._sessions[request.session[SESSION_ID_KEY]]
    persisted = Path(await server._collect_deliverables(session))

    assert persisted == persisted_layout.repeat_dir(str(tmp_path / "persist"), "w1_t1", 0)
    assert sorted(p.name for p in persisted.iterdir()) == [persisted_layout.FINISH_PARAMS_FILE, "report.md"]
    await server.shutdown()


async def test_web_tools_are_refused(server) -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as refused:
        await server.web_search(SimpleNamespace(query="anything"))
    assert refused.value.status_code == 404
    with pytest.raises(HTTPException):
        await server.fetch_web_page(SimpleNamespace(url="https://example.invalid"))


async def test_a_judge_failure_of_a_live_session_names_the_persisted_deliverable(
    server, home, tmp_path, monkeypatch
) -> None:
    request = SimpleNamespace(session={})
    seed = ResourcesSeedSessionRequest(
        resources_session_id="session",
        episode_id=EpisodeId(rollout_id="0-0"),
        task_id=TaskId(taskset="aa_briefcase_lite:benchmark", task_id="w1_t1"),
        task_data={"task_id": "w1_t1"},
    )
    await server.seed_session(request, seed)
    (home / "report.md").write_text("the deliverable")
    await server.finish(request, FinishRequest(reason="done", paths=[str(home / "report.md")]))
    monkeypatch.setattr(server, "_grade", AsyncMock(side_effect=JudgeError("judge unreachable")))

    # The request of a live session carries no deliverables_dir: the server collected the files itself.
    row = json.loads((await judge_failsafe(server.verify)(request, verify_body(None))).body)

    assert row["mask_sample"] is True
    assert row["failure_kind"] == "judge_failed"
    persisted = persisted_layout.repeat_dir(str(tmp_path / "persist"), "w1_t1", 0)
    assert row["deliverables_dir"] == str(persisted)
    assert (persisted / "report.md").read_text() == "the deliverable"
    await server.shutdown()


@pytest.mark.parametrize("breakage", ["empty_directory", "lfs_pointer"])
async def test_a_dataset_without_its_input_files_refuses_to_start_the_episode(server, home, dataset, breakage) -> None:
    notes = dataset / "source_files/week/notes.txt"
    if breakage == "empty_directory":
        notes.unlink()
    else:
        notes.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 123456\n")
    seed = ResourcesSeedSessionRequest(
        resources_session_id="session",
        episode_id=EpisodeId(rollout_id="0-0"),
        task_id=TaskId(taskset="aa_briefcase_lite:benchmark", task_id="w1_t1"),
        task_data={"task_id": "w1_t1"},
    )

    with pytest.raises(RuntimeError, match="no files|unhydrated"):
        await server.seed_session(SimpleNamespace(session={}), seed)

    assert server._sessions == {}
    assert server._claimed_directories == {}
