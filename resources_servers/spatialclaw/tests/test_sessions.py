# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session ownership and shared native-tool behavior, independent of GPU services."""

import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import Request

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
    assert isinstance(result, dict)
    assert "NEVER EXPOSE THIS" not in json.dumps(result)
    assert result["instruction"] == "Which option?"
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
    assert result.response.output_text == "B"
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
    assert result["messages"][0]["content"] == "native prompt"
    assert result["tool_output_images"] == [{"image_url": "data:image/png;base64," + data}]


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
    assert rejected["is_error"] and "Security violation" in rejected["error"]
    reply = await state.execute("ReturnAnswer('A')", 8)
    state.kernel.execute.assert_awaited_once_with("ReturnAnswer('A')", timeout=4)
    assert reply["answer_submitted"] and state.accepted_answer == "A"
    assert (await state.execute("print(1)", 1))["is_error"]
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
        assert context["tool_output_images"] and context["tool_output_images"][0]["image_url"].startswith(
            "data:image/"
        )
        assert state.context["metadata"]["ref_images"] == []
        assert "ground_truth" not in json.dumps(context)
        result = await state.execute("x = 3\nprint(x)", 10)
        assert not result["is_error"] and "3" in result["stdout"]
        result = await state.execute("print(x + 1)\nReturnAnswer('A')", 10)
        assert not result["is_error"] and "4" in result["stdout"]
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


@pytest.mark.parametrize("score_at_limit,reward", [(False, 0.0), (True, 1.0)])
async def test_step_limit_answer_policy(runtime, score_at_limit, reward):
    server, _, _ = runtime
    server.config.score_step_limit_answer = score_at_limit
    cookies = {}
    body = seed_body()
    await server.seed_session(request(cookies), body)
    server._session_runtimes[body.resources_session_id].accepted_answer = "A"
    response = _response("A").model_copy(update={"metadata": {"harness_step_limit_reached": "true"}})
    result = await server.verify(
        SpatialClawVerifyRequest(responses_create_params={"input": "q"}, response=response, sample_id="one"),
        request(cookies),
    )
    assert result.step_limit_reached and result.reward == reward
    metrics = server.compute_metrics([[result.model_dump()]])
    assert metrics["spatialclaw/step_limit_reached"] == 1.0


async def test_lent_sandbox_owns_media_delivery_and_cleanup(runtime, tmp_path):
    from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess

    server, _, _ = runtime
    access = SandboxAccess(
        connection=DirectSandboxConnection(provider_config_ref="sandbox", descriptor={"id": "one"}),
        workdir="/workspace",
    )
    server._lent_sandboxes = SimpleNamespace(
        lend=AsyncMock(return_value=access), release=AsyncMock(), get=lambda session: None
    )
    cookies = {}
    body = seed_body()
    seeded = await server.seed_session(request(cookies), body)
    assert seeded.sandbox_access.connection == access.connection
    assert seeded.sandbox_access.workdir != access.workdir
    files = server._lent_sandboxes.lend.call_args.kwargs["files"]
    manifest_path = seeded.sandbox_access.workdir + "/task.json"
    assert manifest_path in files
    assert "NEVER EXPOSE THIS" not in files[manifest_path]
    await server.close_resources_session(
        ResourcesCloseSessionRequest(resources_session_id=body.resources_session_id, episode_id=body.episode_id)
    )
    server._lent_sandboxes.release.assert_awaited_once_with(body.resources_session_id)


def test_lent_task_files_copy_only_sample_frames_and_references(runtime, tmp_path):
    server, _, _ = runtime
    first = tmp_path / "first.png"
    reference = tmp_path / "reference.png"
    first.write_bytes(b"first")
    reference.write_bytes(b"reference")
    state = SpatialClawSessionRuntime.new("id", "one", str(tmp_path))
    state.context = {"instruction": "question", "images": [str(first)], "metadata": {"ref_images": [str(reference)]}}
    files = server._sandbox_files(state)
    assert list(files.values()).count(b"first") == 1
    assert list(files.values()).count(b"reference") == 1
    target = "/workspace/" + state.workspace.name
    manifest = json.loads(files[target + "/task.json"])
    assert manifest["instruction"] == "question"
    assert manifest["images"] == [target + "/media/0000-first.png"]
    assert manifest["reference_images"] == [target + "/media/0001-reference.png"]


def test_session_track_does_not_claim_stateless_reverification():
    from nemo_gym.server_runtime import ReverifyMode
    from resources_servers.spatialclaw.app import SpatialClawResourcesServerConfig

    assert SpatialClawResourcesServerConfig.REVERIFY_MODE == ReverifyMode.UNSUPPORTED


async def test_lent_sessions_with_same_media_name_have_distinct_workdirs(runtime):
    from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess

    server, _, _ = runtime
    base_access = SandboxAccess(
        connection=DirectSandboxConnection(provider_config_ref="sandbox", descriptor={"id": "test"}),
        workdir="/workspace",
    )
    cleaned = AsyncMock(return_value=SimpleNamespace(error_type=None, return_code=0))
    server._lent_sandboxes = SimpleNamespace(
        lend=AsyncMock(return_value=base_access),
        release=AsyncMock(),
        get=lambda session: SimpleNamespace(exec=cleaned),
    )
    first = await server.seed_session(request({}), seed_body("first"))
    second = await server.seed_session(request({}), seed_body("second"))
    assert first.sandbox_access.workdir != second.sandbox_access.workdir
    first_files = server._lent_sandboxes.lend.call_args_list[0].kwargs["files"]
    second_files = server._lent_sandboxes.lend.call_args_list[1].kwargs["files"]
    assert set(first_files).isdisjoint(second_files)
    await server.close_resources_session(
        ResourcesCloseSessionRequest(resources_session_id="first", episode_id=seed_body().episode_id)
    )
    assert first.sandbox_access.workdir in cleaned.call_args.args[0]
    assert second.sandbox_access.workdir not in cleaned.call_args.args[0]
    assert server.has_session_state("second")


async def test_bootstrap_owns_media_and_kernel_without_running_agent_loop(monkeypatch, tmp_path):
    import sys
    from types import ModuleType

    from responses_api_agents.spatialclaw_agent import app as agent_app

    for name in ("spatial_agent", "spatial_agent.nodes"):
        if name not in sys.modules:
            package = ModuleType(name)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, name, package)
    configs = []

    class NativeConfig:
        timeout_sec = 5
        max_steps = 3

        def _load_from_envs(self):
            pass

        def update_from_dataset_json(self, path):
            assert path.endswith("fake.json")

    config_module = ModuleType("spatial_agent.config")
    config_module.SpatialAgentConfig = NativeConfig
    config_module.set_config = lambda config: configs.append(config)
    monkeypatch.setitem(sys.modules, "spatial_agent.config", config_module)
    kernel = SimpleNamespace(shutdown=AsyncMock(), is_running=True)
    manager_module = ModuleType("spatial_agent.kernel.manager")
    manager_module.JupyterKernelManager = lambda **kwargs: kernel
    monkeypatch.setitem(sys.modules, "spatial_agent.kernel.manager", manager_module)
    calls = []
    generated = AsyncMock(side_effect=AssertionError("bootstrap must not invoke the evaluated harness"))

    class Workflow:
        def __init__(self, config):
            self.config = config
            self.llm_client = SimpleNamespace(_client_pool={}, close=AsyncMock(), generate=generated)
            self._kernel_pool = SimpleNamespace(shutdown_all=AsyncMock())
            self.graph = SimpleNamespace(ainvoke=generated)

        async def arun(self, **kwargs):
            calls.append(kwargs)
            return await self.graph.ainvoke(
                {"instruction": kwargs["instruction"]},
                {"configurable": {"kernel_manager": kernel, "system_prompt": "native system"}},
            )

    workflow_module = ModuleType("spatial_agent.workflow")
    workflow_module.SpatialAgentWorkflow = Workflow
    monkeypatch.setitem(sys.modules, "spatial_agent.workflow", workflow_module)
    initialized = AsyncMock()

    async def init_node(state, config):
        await initialized()
        return {
            "messages": [
                SimpleNamespace(type="system", content=config["configurable"]["system_prompt"]),
                SimpleNamespace(type="human", content=state["instruction"]),
            ]
        }

    node_module = ModuleType("spatial_agent.nodes.init_node")
    node_module.init_node = init_node
    monkeypatch.setitem(sys.modules, "spatial_agent.nodes.init_node", node_module)
    pins = []

    async def root(agent):
        pins.append(agent.config.spatialclaw_commit)
        return tmp_path

    prepared = []

    async def materialize(agent, body, directory, config):
        prepared.append(body)
        return body.input, [str(tmp_path / "frame.png")], {"frame_indices": [0], "fps": 2, "ref_images": []}

    monkeypatch.setattr(agent_app.SpatialClawAgent, "_resolve_spatialclaw_root", root)
    monkeypatch.setattr(agent_app.SpatialClawAgent, "_materialize_inputs", materialize)
    monkeypatch.setattr(
        agent_app, "_config_path", lambda root, name, kind: str(root / f"{name}.json") if name else None
    )
    server, sample, _ = _server()
    server.config.data_root = str(tmp_path)
    server.server_client.global_config_dict = {
        "policy_model": {"responses_api_models": {"openai_model": {"host": "model", "port": 9000}}}
    }
    server.server_client._build_server_base_url.return_value = "http://model:9000"
    sample.question = "Which side?"
    sample.answer = "NEVER EXPOSE THIS"
    sample.choices = {"A": "left", "B": "right"}
    server._benchmark_instance.data_specific_prompt = "One letter."
    runtime = SpatialClawSessionRuntime.new("bootstrap", "one", str(tmp_path / "workspace"))
    media = {"image_paths": '["frame.png"]'}
    await runtime.initialize(server, sample, server._benchmark_instance, "3-0", media=media)
    assert pins == [agent_app.SPATIALCLAW_COMMIT]
    assert configs[0].llm_base_url == "http://model:9000/ng-rollout/3-0/v1"
    assert prepared[0].metadata == media
    assert calls[0]["answer"] is None and calls[0]["defer_report"] is True
    assert calls[0]["images"] == [str(tmp_path / "frame.png")]
    assert runtime.kernel is kernel
    assert runtime.context["system_prompt"] == "native system"
    assert runtime.context["images"] == calls[0]["images"]
    assert "NEVER EXPOSE THIS" not in json.dumps(runtime.context)
    initialized.assert_awaited_once()
    generated.assert_not_awaited()
    kernel.shutdown.assert_not_awaited()
    await runtime.close(timeout=1, keep_workspace=False)
    kernel.shutdown.assert_awaited_once()
    runtime.workflow.llm_client.close.assert_awaited_once()
    assert not runtime.workspace.exists()


async def test_cancelled_kernel_start_remains_owned_and_cleanup_can_retry(tmp_path):
    import asyncio

    from resources_servers.spatialclaw.session_runtime import _ResourceKernelPool

    allocated = asyncio.Event()

    async def start():
        allocated.set()  # The external runtime has allocated its child here.
        await asyncio.Event().wait()

    manager = SimpleNamespace(
        is_running=False, start=start, shutdown=AsyncMock(side_effect=[RuntimeError("busy"), None])
    )
    runtime = SpatialClawSessionRuntime.new("cancelled", "one", str(tmp_path))
    runtime.kernel = manager
    runtime.workflow = SimpleNamespace(
        _kernel_pool=_ResourceKernelPool(manager), llm_client=SimpleNamespace(close=AsyncMock())
    )
    startup = asyncio.create_task(runtime.workflow._kernel_pool.acquire())
    await allocated.wait()
    startup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await startup
    with pytest.raises(RuntimeError, match="busy"):
        await runtime.close(timeout=1, keep_workspace=True)
    assert runtime.kernel is manager
    await runtime.close(timeout=1, keep_workspace=True)
    assert manager.shutdown.await_count == 2
    runtime.workflow.llm_client.close.assert_awaited_once()


async def test_private_kernel_supports_native_manager_control_and_preserves_errors(tmp_path):
    from dataclasses import make_dataclass

    state = SpatialClawSessionRuntime.new("rpc", "one", str(tmp_path))
    with pytest.raises(RuntimeError, match="not initialized"):
        await state.rpc("start", code=None, timeout=None, name=None)
    with pytest.raises(ValueError, match="requires answer"):
        await state.rpc("finalize_answer", code=None, timeout=None, name=None)
    result_type = make_dataclass("Result", [("stdout", str), ("error", str | None)])
    state.spatial_config = SimpleNamespace(timeout_sec=7)
    state.kernel = SimpleNamespace(
        execute=AsyncMock(return_value=result_type("out", None)),
        check_sentinel=AsyncMock(return_value=None),
        clear_sentinel=AsyncMock(),
        start=AsyncMock(return_value="started"),
        shutdown=AsyncMock(),
        reset_namespace=AsyncMock(),
        get_variables=AsyncMock(return_value={"x": 1}),
    )
    assert await state.rpc("execute", code="x = 1", timeout=None, name=None) == {"stdout": "out", "error": None}
    state.kernel.execute.assert_awaited_once_with("x = 1", timeout=7)
    assert await state.rpc("start", code=None, timeout=None, name=None) == "started"
    assert await state.rpc("get_variables", code=None, timeout=None, name=None) == {"x": 1}
    await state.rpc("reset_namespace", code=None, timeout=None, name=None)
    await state.rpc("shutdown", code=None, timeout=None, name=None)
    await state.rpc("check_sentinel", code=None, timeout=None, name="_other")
    state.kernel.check_sentinel.assert_awaited_with("_other")
    await state.rpc("clear_sentinel", code=None, timeout=None, name="_other")
    state.kernel.clear_sentinel.assert_awaited_with("_other")
    with pytest.raises(ValueError, match="Unsupported"):
        await state.rpc("invalid", code=None, timeout=None, name=None)
    await state.close(timeout=1, keep_workspace=True)
    assert state.kernel.shutdown.await_count == 2


def test_native_execution_images_include_display_mime_and_show_labels(monkeypatch):
    import sys
    from types import ModuleType

    from PIL import Image

    from resources_servers.spatialclaw.session_runtime import execution_images

    module = ModuleType("spatial_agent.nodes.execute_node")
    module._parse_show_markers = lambda text: [{"images": [Image.new("RGB", (1, 1))], "label": "view"}]
    monkeypatch.setitem(sys.modules, "spatial_agent.nodes.execute_node", module)
    output = execution_images(
        SimpleNamespace(stdout="shown", display_data=[{"data": {"image/jpeg": "anBlZw=="}}, {"image/png": "cG5n"}])
    )
    assert output[:2] == [
        {"image_url": "data:image/jpeg;base64,anBlZw=="},
        {"image_url": "data:image/png;base64,cG5n"},
    ]
    assert output[2]["label"] == "view"
    assert base64.b64decode(output[2]["image_url"].split(",", 1)[1]).startswith(b"\x89PNG")


async def test_resource_endpoints_bind_kernel_and_task_to_session(runtime):
    from resources_servers.spatialclaw.app import KernelRequest

    server, _, _ = runtime
    cookies = {}
    body = seed_body()
    await server.seed_session(request(cookies), body)
    state = server._session_runtimes[body.resources_session_id]
    state.rpc = AsyncMock(return_value={"x": 2})
    state.execute = AsyncMock(return_value={"stdout": "2"})
    assert await server.kernel(request(cookies), KernelRequest(operation="get_variables")) == {"result": {"x": 2}}
    assert await server.execute(request(cookies), ExecuteRequest(code="print(x)")) == {"stdout": "2"}
    state.execute.assert_awaited_once_with("print(x)", None)
    state.context = None
    with pytest.raises(ResourcesSessionLostError, match="incomplete"):
        await server.get_context(request(cookies))
    with pytest.raises(Exception, match="does not match"):
        await server.verify(
            SpatialClawVerifyRequest(responses_create_params={"input": "q"}, response=_response("A"), sample_id="two"),
            request(cookies),
        )


async def test_failed_lent_media_cleanup_retains_owned_session_for_retry(runtime):
    from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess

    server, _, closed = runtime
    access = SandboxAccess(
        connection=DirectSandboxConnection(provider_config_ref="sandbox", descriptor={"id": "one"}),
        workdir="/workspace",
    )
    cleanup = AsyncMock(
        side_effect=[
            SimpleNamespace(error_type="timeout", return_code=None),
            SimpleNamespace(error_type=None, return_code=0),
        ]
    )
    server._lent_sandboxes = SimpleNamespace(
        lend=AsyncMock(return_value=access), release=AsyncMock(), get=lambda session: SimpleNamespace(exec=cleanup)
    )
    body = seed_body()
    await server.seed_session(request({}), body)
    close = ResourcesCloseSessionRequest(resources_session_id=body.resources_session_id, episode_id=body.episode_id)
    with pytest.raises(RuntimeError, match="staged sandbox media"):
        await server.close_resources_session(close)
    assert server.has_session_state(body.resources_session_id)
    server._lent_sandboxes.release.assert_not_awaited()
    closed.assert_not_awaited()
    await server.close_resources_session(close)
    assert not server.has_session_state(body.resources_session_id)


async def test_server_shutdown_attempts_all_owned_sessions_and_lent_sandboxes(runtime, monkeypatch):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    from nemo_gym.server_runtime import SimpleResourcesServer

    server, _, closed = runtime
    await server.seed_session(request({}), seed_body("first"))
    await server.seed_session(request({}), seed_body("second"))
    closed.side_effect = [RuntimeError("broken kernel"), None]
    lent = SimpleNamespace(release_all=AsyncMock(), get=lambda _: None, release=AsyncMock())
    server._lent_sandboxes = lent

    @asynccontextmanager
    async def parent(app):
        yield {"parent": True}

    monkeypatch.setattr(SimpleResourcesServer, "setup_webserver", lambda self: FastAPI(lifespan=parent))
    app = server.setup_webserver()
    async with app.router.lifespan_context(app) as state:
        assert state == {"parent": True}
    assert closed.await_count == 2
    assert server.has_session_state("first") and not server.has_session_state("second")
    lent.release_all.assert_awaited_once()


def test_sandbox_config_validates_workdir_and_scopes_network(monkeypatch):
    from unittest.mock import MagicMock

    from nemo_gym.server_runtime.base_server import ServerClient
    from resources_servers.spatialclaw import app

    config = dict(host="127.0.0.1", port=8080, name="resource", entrypoint="app.py", dataset_config="fake")
    for workdir in ("/", "relative", "/workspace/../other"):
        with pytest.raises(ValueError, match="absolute task directory"):
            app.SpatialClawResourcesServerConfig(**config, sandbox_workdir=workdir)
    lend = MagicMock()
    monkeypatch.setattr(app, "LentSandboxes", lend)
    server = app.SpatialClawResourcesServer(
        config=app.SpatialClawResourcesServerConfig(
            **config, sandbox_provider="provider", sandbox_network_access="model_and_tools"
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    assert server._lent_sandboxes is lend.return_value
    assert lend.call_args.kwargs["model_server"] == "policy_model"
    assert lend.call_args.kwargs["tool_servers"] == ["resource"]
