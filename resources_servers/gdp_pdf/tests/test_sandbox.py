# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request

from nemo_gym.client import ServerClient
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSessionConflictError,
    check_resources_session_contract,
)
from resources_servers.gdp_pdf import sandbox_app as module
from resources_servers.gdp_pdf.tests.test_app import _verify_request, config  # noqa: F401


@pytest.fixture
def setup(config, tmp_path, monkeypatch):  # noqa: F811
    component = tmp_path / "resources_servers/gdp_pdf"
    (component / "data").mkdir(parents=True)
    pdf = component / "data/source.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    monkeypatch.setattr(module, "PARENT_DIR", tmp_path)
    monkeypatch.chdir(component)  # Gym starts resources from their component directory.
    server = module.GdpPdfSandboxResourcesServer(
        config=module.GdpPdfSandboxConfig(**config.model_dump(), sandbox_provider="sandbox", sandbox_image="tools"),
        server_client=MagicMock(spec=ServerClient),
    )
    sandbox = AsyncMock()
    sandbox.serialize.return_value = {"sandbox_id": "test"}

    async def start(spec, callback):
        assert spec.files == {}  # No hidden rubric or reference answer is uploaded.
        await callback(sandbox)

    sandbox.start_with_setup.side_effect = start
    monkeypatch.setattr(module, "AsyncSandbox", lambda _: sandbox)
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {"sandbox": {"docker": {}}})
    body = ResourcesSeedSessionRequest(
        resources_session_id="session",
        episode_id={"rollout_id": "run", "attempt": 0},
        task_id={"taskset": "train", "task_id": "task"},
        task_data={"verifier_metadata": {"source_pdf": "source.pdf", "criteria": [{"criterion": "hidden"}]}},
    )
    return server, sandbox, body, pdf


def test_session_contract_and_pdf_delivery(setup):
    server, sandbox, body, pdf = setup
    check_resources_session_contract(
        server.setup_webserver(), body, keeps_state=True, holds_state=lambda sid: sid in server._sessions
    )
    assert all(call.args == (pdf, module.DOCUMENT_PATH) for call in sandbox.upload.await_args_list)
    assert sandbox.upload.await_count == 2


@pytest.mark.parametrize("status,tool_tail", [("completed", False), ("incomplete", False), ("incomplete", True)])
async def test_only_completed_answers_are_judged(setup, status, tool_tail):
    server, _, _, _ = setup
    body = _verify_request("The cap is $5M.")
    payload = body.model_dump()
    payload["response"]["status"] = status
    if tool_tail:
        payload["response"]["output"].append(
            {"type": "function_call", "name": "bash", "arguments": "{}", "call_id": "pending"}
        )
    body = module.GdpPdfVerifyRequest.model_validate(payload)
    server._judge_criterion = AsyncMock(return_value=module.CriterionEvaluation(criterion="cap", passed=True))
    result = await server.verify(body)
    assert result.response == body.response
    assert result.reward == result.all_pass == (1.0 if status == "completed" else 0.0)
    assert result.num_criteria == 1
    assert result.criterion_evaluations[0].passed == (status == "completed")
    assert server._judge_criterion.await_count == (1 if status == "completed" else 0)


@pytest.mark.parametrize("source", ["../outside.pdf", "/etc/passwd", "missing.pdf", ""])
async def test_invalid_pdf_never_allocates(setup, source):
    server, sandbox, body, _ = setup
    body.task_data["verifier_metadata"]["source_pdf"] = source
    with pytest.raises(ResourcesSeedRejectedError):
        await server.seed_session(Request({"type": "http", "session": {}}), body)
    sandbox.start_with_setup.assert_not_awaited()


async def test_failed_upload_and_close_retain_retryable_cleanup(setup):
    server, sandbox, body, pdf = setup
    server.config.documents_base_dir = str(pdf.parent)  # Absolute roots work too.
    request = Request({"type": "http", "session": {}})
    sandbox.upload.side_effect = RuntimeError("upload failed")
    with pytest.raises(RuntimeError, match="upload failed"):
        await server.seed_session(request, body)
    sandbox.stop.assert_awaited_once()
    assert not server._sessions
    sandbox.upload.side_effect = None
    await server.seed_session(request, body)
    close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=body.episode_id)
    sandbox.stop.side_effect = [RuntimeError("stop failed"), None]
    with pytest.raises(RuntimeError, match="stop failed"):
        await server.close_resources_session(close)
    assert "session" in server._sessions
    await server.close_resources_session(close)
    assert not server._sessions


async def test_shutdown_waits_for_session_operation(setup):
    server, sandbox, body, _ = setup
    app = server.setup_webserver()
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    await server.seed_session(Request({"type": "http", "session": {}}), body)
    async with server._locks.locked(body.resources_session_id):
        shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
        await asyncio.sleep(0)
        sandbox.stop.assert_not_awaited()
    await asyncio.wait_for(shutdown, timeout=1)
    sandbox.stop.assert_awaited_once()
    assert not server._sessions


async def test_retirement_releases_sandbox_and_retries_failed_stop(setup):
    server, sandbox, body, _ = setup
    request = Request({"type": "http", "session": {}})
    await server.seed_session(request, body)
    sandbox.stop.side_effect = [RuntimeError("stop failed"), None]
    with pytest.raises(RuntimeError, match="stop failed"):
        await server.retire_session_state(body.resources_session_id)
    assert body.resources_session_id in server._sessions
    with pytest.raises(ResourcesSessionConflictError):
        await server.seed_session(request, body)
    await server.retire_session_state(body.resources_session_id)
    await server.retire_session_state(body.resources_session_id)
    assert not server._sessions
    assert sandbox.stop.await_count == 2
