# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session ownership and shared native-tool behavior, independent of GPU services."""

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import Request
from mcp import types

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    ResourcesSessionLostError,
)
from resources_servers.spatialclaw.app import ExecuteRequest, SpatialClawVerifyRequest
from resources_servers.spatialclaw.session_runtime import SpatialClawSessionRuntime, context_result
from resources_servers.spatialclaw.tests.test_app import _response, _server


def request(session=None):
    return Request(
        {"type": "http", "method": "POST", "headers": [], "session": session if session is not None else {}}
    )


def seed_body(session="task-session", sample="one", episode=None):
    return ResourcesSeedSessionRequest(
        resources_session_id=session,
        episode_id=episode or EpisodeId.from_capture_key("0-0"),
        task_id=TaskId(taskset="spatialclaw", task_id=sample),
        task_data={"sample_id": sample, "answer": "NEVER EXPOSE THIS"},
    )


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    initialized = AsyncMock()
    closed = AsyncMock()

    async def initialize(self, server, sample, benchmark, capture_key, *, media):
        await initialized(self, sample.sample_id, capture_key)
        self.context = {
            "instruction": "Which option?",
            "images": [],
            "metadata": {},
            "system_prompt": "native prompt",
            "session_id": self.session_id,
            "messages": [{"role": "user", "content": "Which option?"}],
        }
        self.kernel = SimpleNamespace()
        self.spatial_config = SimpleNamespace(max_steps=2, timeout_sec=3)

    async def close(self, **kwargs):
        await closed(self.session_id)

    monkeypatch.setattr(SpatialClawSessionRuntime, "initialize", initialize)
    monkeypatch.setattr(SpatialClawSessionRuntime, "close", close)
    server, _, _ = _server()
    server.config.workspace_root = str(tmp_path)
    return server, initialized, closed


async def test_seed_replays_and_close_fences_without_cookie(runtime):
    server, initialized, closed = runtime
    cookies = {}
    body = seed_body()
    first = await server.seed_session(request(cookies), body)
    second = await server.seed_session(request(cookies), body)
    assert first == second and first.resources_session_id == body.resources_session_id
    initialized.assert_awaited_once()
    assert cookies[SESSION_ID_KEY] == body.resources_session_id
    close = ResourcesCloseSessionRequest(resources_session_id=body.resources_session_id, episode_id=body.episode_id)
    await server.close_resources_session(close)
    await server.close_resources_session(close)
    closed.assert_awaited_once()
    with pytest.raises(Exception) as caught:
        await server.seed_session(request({}), body)
    assert caught.value.status_code == 409
    with pytest.raises(ResourcesSessionLostError):
        await server.get_context(request(cookies))


async def test_seed_rejects_unknown_sample_and_other_owner(runtime):
    server, _, _ = runtime
    with pytest.raises(Exception) as caught:
        await server.seed_session(request({}), seed_body(sample="missing"))
    assert caught.value.status_code == 422
    body = seed_body()
    await server.seed_session(request({}), body)
    with pytest.raises(Exception) as caught:
        await server.seed_session(request({}), seed_body(episode=EpisodeId.from_capture_key("1-0")))
    assert caught.value.status_code == 409


async def test_context_never_contains_private_ground_truth(runtime):
    server, _, _ = runtime
    cookies = {}
    await server.seed_session(request(cookies), seed_body())
    result = await server.get_context(request(cookies))
    assert isinstance(result, types.CallToolResult)
    assert "NEVER EXPOSE THIS" not in result.model_dump_json()
    assert result.model_dump(by_alias=True)["structuredContent"]["instruction"] == "Which option?"
    assert "get_context" in (await server.seed_session(request(cookies), seed_body())).observation
    assert [
        t.name for t in server.mcp_tools([SimpleNamespace(name=n) for n in ["kernel", "get_context", "execute"]], None)
    ] == ["get_context", "execute"]


async def test_verifier_uses_accepted_answer_instead_of_harness_prose(runtime):
    server, _, _ = runtime
    cookies = {}
    body = seed_body()
    await server.seed_session(request(cookies), body)
    state = server._session_runtimes[body.resources_session_id]
    response = SpatialClawVerifyRequest(
        responses_create_params={"input": "q"}, response=_response("A"), sample_id="one"
    )
    first = await server.verify(response, request(cookies))
    assert first.reward == 0 and first.prediction == ""
    state.verified = False
    state.accepted_answer = "A"
    result = await server.verify(response.model_copy(update={"response": _response("B")}), request(cookies))
    assert result.reward == 1 and result.prediction == "A"
    assert (await server.verify(response, request(cookies))).reward == 1
    with pytest.raises(ResourcesSessionLostError):
        await server.execute(request(cookies), ExecuteRequest(code="ReturnAnswer('B')"))
    await server.close_resources_session(
        ResourcesCloseSessionRequest(resources_session_id=body.resources_session_id, episode_id=body.episode_id)
    )
    lost = await server.verify(response, request(cookies))
    assert lost.mask_sample and not lost.evaluation_completed


async def test_failed_close_preserves_runtime_for_retry(runtime):
    server, _, closed = runtime
    cookies = {}
    body = seed_body()
    await server.seed_session(request(cookies), body)
    closed.side_effect = [RuntimeError("shutdown failed"), None]
    close = ResourcesCloseSessionRequest(resources_session_id=body.resources_session_id, episode_id=body.episode_id)
    with pytest.raises(RuntimeError, match="shutdown failed"):
        await server.close_resources_session(close)
    assert server.has_session_state(body.resources_session_id)
    await server.close_resources_session(close)
    assert not server.has_session_state(body.resources_session_id)


def test_context_keeps_native_image_blocks():
    data = base64.b64encode(b"png-image").decode()
    result = context_result(
        {
            "messages": [
                {"role": "system", "content": "native prompt"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "question"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}},
                    ],
                },
            ]
        }
    )
    assert result.content[0].text == "native prompt\n\nquestion"
    assert result.content[1].data == data and result.content[1].model_dump(by_alias=True)["mimeType"] == "image/png"


async def test_kernel_rpc_preserves_submitted_answer_when_native_clears_it(tmp_path):
    state = SpatialClawSessionRuntime.new("id", "one", str(tmp_path))
    state.kernel = SimpleNamespace(check_sentinel=AsyncMock(return_value={"text": "A"}), clear_sentinel=AsyncMock())
    assert (await state.rpc("check_sentinel", code=None, timeout=None, name=None))["text"] == "A"
    await state.rpc("clear_sentinel", code=None, timeout=None, name=None)
    assert state.accepted_answer == "A"
    state.kernel.clear_sentinel.assert_awaited_once_with("_return_answer_result")


async def test_public_execute_uses_native_safety_and_budget(monkeypatch, tmp_path):
    import sys
    from types import ModuleType

    safety = ModuleType("spatial_agent.kernel.safety")
    safety.SecuritySandbox = SimpleNamespace(check=lambda code: "blocked" if code == "bad" else None)
    monkeypatch.setitem(sys.modules, "spatial_agent.kernel.safety", safety)
    state = SpatialClawSessionRuntime.new("id", "one", str(tmp_path))
    state.spatial_config = SimpleNamespace(max_steps=2, timeout_sec=4)
    result = SimpleNamespace(stdout="done", stderr="", error=None, display_data=[], execution_time_sec=0.1)
    state.kernel = SimpleNamespace(
        execute=AsyncMock(return_value=result), check_sentinel=AsyncMock(return_value={"text": "A"})
    )
    monkeypatch.setattr("resources_servers.spatialclaw.session_runtime.execution_images", lambda result: [])
    rejected = await state.execute("bad", 8)
    assert rejected.model_dump(by_alias=True)["isError"] and "Security violation" in rejected.content[0].text
    reply = await state.execute("ReturnAnswer('A')", 8)
    state.kernel.execute.assert_awaited_once_with("ReturnAnswer('A')", timeout=4)
    assert reply.model_dump(by_alias=True)["structuredContent"]["answer_submitted"] and state.accepted_answer == "A"
    assert (await state.execute("print(1)", 1)).model_dump(by_alias=True)["isError"]
    assert state.kernel.execute.await_count == 1


@pytest.mark.skipif(not __import__("os").environ.get("SPATIALCLAW_ROOT"), reason="requires pinned SPATIALCLAW_ROOT")
async def test_native_task_context_and_persistent_execution_without_model_calls(monkeypatch, tmp_path):
    import os
    from pathlib import Path

    from PIL import Image

    from resources_servers.spatialclaw.session_runtime import SpatialClawSessionRuntime
    from responses_api_agents.spatialclaw_agent import app as agent_app

    root = agent_app._validate_spatialclaw_checkout(Path(os.environ["SPATIALCLAW_ROOT"]), agent_app.SPATIALCLAW_COMMIT)
    server, first, _ = _server()
    server.config.spatialclaw_root = str(root)
    server.config.dataset_config = "erqa_spatialclaw"
    server.config.data_root = str(tmp_path)
    server.config.workspace_root = str(tmp_path / "workspaces")
    server.config.frame_cache_root = str(tmp_path / "frames")
    server.config.keep_workspaces = False
    server._source_root = root
    server.server_client.global_config_dict = {
        "policy_model": {"responses_api_models": {"openai_model": {"host": "127.0.0.1", "port": 9999}}}
    }
    server.server_client._build_server_base_url.return_value = "http://127.0.0.1:9999"
    image = tmp_path / "red.png"
    Image.new("RGB", (16, 16), color="red").save(image)
    first.images = [str(image)]
    first.question = "What color is the image?"
    first.question_type = "multiple-choice"
    first.choices = {"A": "red", "B": "blue"}
    server._benchmark_instance.data_specific_prompt = "Answer with one letter."
    model_call = AsyncMock(side_effect=AssertionError("initialization must not call a model"))
    monkeypatch.setattr(agent_app, "http_request", model_call)
    monkeypatch.setenv("SPATIAL_AGENT_INJECTION_TIMEOUT", "30")
    state = SpatialClawSessionRuntime.new("native-context", "one", str(tmp_path / "workspaces"))
    try:
        await state.initialize(server, first, server._benchmark_instance, "0-0")
        assert state.context["instruction"].startswith(first.question)
        context = context_result(state.context)
        assert any(isinstance(block, types.ImageContent) for block in context.content)
        assert state.context["metadata"]["ref_images"] == []
        assert "ground_truth" not in context.model_dump_json()
        result = await state.execute("x = 3\nprint(x)", 10)
        assert (
            not result.model_dump(by_alias=True)["isError"]
            and "3" in result.model_dump(by_alias=True)["structuredContent"]["stdout"]
        )
        result = await state.execute("print(x + 1)\nReturnAnswer('A')", 10)
        assert (
            not result.model_dump(by_alias=True)["isError"]
            and "4" in result.model_dump(by_alias=True)["structuredContent"]["stdout"]
        )
        assert state.accepted_answer == "A"
        model_call.assert_not_awaited()
    finally:
        await state.close(timeout=30, keep_workspace=False)
    assert not state.workspace.exists()


async def test_resource_model_calls_make_partial_harness_usage_unknown(runtime):
    from nemo_gym.core import NeMoGymResponseUsage

    server, _, _ = runtime
    cookies = {}
    body = seed_body()
    await server.seed_session(request(cookies), body)
    state = server._session_runtimes[body.resources_session_id]
    state.workspace.mkdir(parents=True)
    (state.workspace / "gym-model-responses.jsonl").write_text('{"response_id":"aux-call"}\n')
    response = _response("A").model_copy(
        update={
            "usage": NeMoGymResponseUsage(
                input_tokens=10,
                output_tokens=1,
                total_tokens=11,
                input_tokens_details={"cached_tokens": 0},
                output_tokens_details={"reasoning_tokens": 0},
            )
        }
    )
    result = await server.verify(
        SpatialClawVerifyRequest(responses_create_params={"input": "q"}, response=response, sample_id="one"),
        request(cookies),
    )
    assert result.response.usage is None
    assert result.response.metadata["spatialclaw_resource_usage_unknown"] == "true"


def test_resource_http_session_contract(runtime):
    from nemo_gym.server_runtime.testing.session_conformance import check_resources_session_contract

    server, _, _ = runtime
    server.server_client.global_config_dict = {}
    check_resources_session_contract(
        server.setup_webserver(),
        seed_body(),
        keeps_state=True,
        holds_state=server.has_session_state,
        verify_body=SpatialClawVerifyRequest(
            responses_create_params={"input": "q"}, response=_response("A"), sample_id="one"
        ).model_dump(mode="json"),
    )


def test_partial_typed_seed_does_not_fall_back_to_cookie_seed(runtime):
    from fastapi.testclient import TestClient

    server, _, _ = runtime
    server.server_client.global_config_dict = {}
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    result = client.post("/seed_session", json={"resources_session_id": "partial", "sample_id": "one"})
    assert result.status_code == 422
    assert not server._session_runtimes


async def test_native_finalize_retains_native_fallback_policy(tmp_path):
    state = SpatialClawSessionRuntime.new("id", "one", str(tmp_path))
    state.accepted_answer = "A"
    await state.rpc("finalize_answer", answer="B", code=None, timeout=None, name=None)
    assert state.accepted_answer == "B"
    await state.rpc("finalize_answer", answer="", code=None, timeout=None, name=None)
    assert state.accepted_answer == ""


def test_kernel_control_payloads_are_validated_before_execution():
    from pydantic import ValidationError

    from resources_servers.spatialclaw.app import KernelRequest

    for operation in ("execute", "finalize_answer"):
        with pytest.raises(ValidationError):
            KernelRequest(operation=operation)
    with pytest.raises(ValidationError):
        KernelRequest(operation="check_sentinel", name="bad'code")
