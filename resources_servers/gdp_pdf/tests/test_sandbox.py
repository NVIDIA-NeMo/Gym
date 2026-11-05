# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request

from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSessionConflictError,
)
from resources_servers.gdp_pdf import sandbox_app
from resources_servers.gdp_pdf.sandbox_app import DOCUMENT_PATH, GdpPdfSandboxConfig, GdpPdfSandboxResourcesServer


@pytest.fixture
def setup(tmp_path, monkeypatch):
    (tmp_path / "source.pdf").write_bytes(b"%PDF-1.7\n")
    config = GdpPdfSandboxConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="sandbox_app.py",
        sandbox_provider="sandbox",
        sandbox_image="pdf-tools",
        documents_base_dir=str(tmp_path),
        judge_model_server={"type": "responses_api_models", "name": "judge"},
        judge_chat_create_params={"messages": []},
        judge_prompt_template_fpath=str(Path(sandbox_app.__file__).parent / "prompt_templates/judge.txt"),
    )
    server = GdpPdfSandboxResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    sandbox = AsyncMock()
    sandbox.serialize.return_value = {"sandbox_id": "test"}

    async def start(spec, callback):
        assert spec.workdir == "/workspace"
        assert spec.files == {}  # Hidden rubric/answers are never staged.
        await callback(sandbox)

    sandbox.start_with_setup.side_effect = start
    factory = MagicMock(return_value=sandbox)
    monkeypatch.setattr(sandbox_app, "AsyncSandbox", factory)
    monkeypatch.setattr(sandbox_app, "get_global_config_dict", lambda: {"sandbox": {"docker": {}}})
    body = ResourcesSeedSessionRequest(
        resources_session_id="session",
        episode_id=EpisodeId(rollout_id="rollout", attempt=0),
        task_id=TaskId(taskset="train", task_id="task"),
        task_data={"verifier_metadata": {"source_pdf": "source.pdf", "criteria": [{"criterion": "hidden"}]}},
    )
    request = Request({"type": "http", "session": {}})
    return server, sandbox, factory, request, body


async def test_seed_retry_and_close_fence(setup):
    server, sandbox, factory, request, body = setup
    first, repeated = await server.seed_session(request, body), await server.seed_session(request, body)
    assert first == repeated
    assert first.sandbox_access.workdir == "/workspace"
    sandbox.upload.assert_awaited_once_with(Path(server.config.documents_base_dir) / "source.pdf", DOCUMENT_PATH)
    factory.assert_called_once()
    conflicting = body.model_copy(update={"episode_id": EpisodeId(rollout_id="other", attempt=0)})
    with pytest.raises(ResourcesSessionConflictError):
        await server.seed_session(request, conflicting)
    close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=body.episode_id)
    # Failed stop retains ownership; the next close actually retries.
    sandbox.stop.side_effect = [RuntimeError("provider unavailable"), None]
    with pytest.raises(RuntimeError):
        await server.close_resources_session(close)
    assert "session" in server._sessions
    await server.close_resources_session(close)
    await server.close_resources_session(close)
    assert sandbox.stop.await_count == 2
    with pytest.raises(ResourcesSessionConflictError):
        await server.seed_session(request, body)


async def test_relative_document_root_from_component_workdir(setup, tmp_path, monkeypatch):
    server, sandbox, _, request, body = setup
    component = tmp_path / "resources_servers/gdp_pdf"
    (component / "data").mkdir(parents=True)
    pdf = component / "data/source.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    monkeypatch.setattr(sandbox_app, "PARENT_DIR", tmp_path)
    monkeypatch.chdir(component)
    server.config.documents_base_dir = "resources_servers/gdp_pdf/data"
    await server.seed_session(request, body)
    sandbox.upload.assert_awaited_once_with(pdf, DOCUMENT_PATH)


async def test_setup_failure_releases_sandbox_and_can_retry(setup):
    server, sandbox, _, request, body = setup
    sandbox.upload.side_effect = RuntimeError("upload failed")
    with pytest.raises(RuntimeError, match="upload failed"):
        await server.seed_session(request, body)
    sandbox.stop.assert_awaited_once()
    assert not server._sessions
    sandbox.upload.side_effect = None
    await server.seed_session(request, body)
    assert server._sessions["session"].access is not None


@pytest.mark.parametrize("source", ["../outside.pdf", "/etc/passwd", "missing.pdf", ""])
async def test_rejects_invalid_sources_before_start(setup, source):
    server, _, factory, request, body = setup
    body.task_data["verifier_metadata"]["source_pdf"] = source
    with pytest.raises(ResourcesSeedRejectedError):
        await server.seed_session(request, body)
    factory.assert_not_called()


async def test_close_before_seed_prevents_allocation(setup):
    server, _, factory, request, body = setup
    await server.close_resources_session(
        ResourcesCloseSessionRequest(resources_session_id="session", episode_id=body.episode_id)
    )
    with pytest.raises(ResourcesSessionConflictError):
        await server.seed_session(request, body)
    factory.assert_not_called()
