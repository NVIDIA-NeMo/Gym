# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The backend seam: registry, backend identity in rows and verify responses, and a second backend end to end."""

import sys
import types
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.mcp_atlas import backends
from resources_servers.mcp_atlas.app import MCPAtlasSeedSessionRequest, MCPAtlasVerifyRequest
from resources_servers.mcp_atlas.backends import get_backend, list_backends, register_backend
from resources_servers.mcp_atlas.backends.base import (
    BackendConfigurationError,
    BackendTask,
    MCPAtlasBackend,
    ToolOutcome,
    ToolResult,
    backend_task,
)
from resources_servers.mcp_atlas.backends.scale_live import ScaleLiveBackend
from resources_servers.mcp_atlas.tests.fake_agent_environment import FakeAgentEnvironment
from resources_servers.mcp_atlas.tests.test_app import (
    META,
    http_request,
    judge_client,
    make_server,
    seed,
    verify_body,
)


OTHER = "other_backend"


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    """Registrations made by a test do not leak into the others."""
    monkeypatch.setattr(backends, "_BACKENDS", {})


class _Handle:
    def __init__(self) -> None:
        self.missing_tools: list[str] = []
        self.reset_for: list[str] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []


class RecordingBackend(MCPAtlasBackend):
    """A second backend that answers every tool call itself; stands in for any deployment-specific backend."""

    name = OTHER
    instances: list["RecordingBackend"] = []

    def __init__(self, *, config: Any, global_config_dict: Any) -> None:
        self.options = config.backend_options
        self.handles: dict[str, _Handle] = {}
        self.stopped: list[str] = []
        RecordingBackend.instances.append(self)

    async def start(self, session_id: str, task: BackendTask) -> _Handle:
        if task.metadata.get("state") == "missing":
            raise BackendConfigurationError("the task's state is not installed")
        self.handles[session_id] = handle = _Handle()
        return handle

    async def reset(self, handle: _Handle, task: BackendTask) -> None:
        handle.reset_for.append(task.task_id)

    async def call_tool(self, handle, task, tool_name, arguments, *, timeout_s) -> ToolResult:
        handle.calls.append((tool_name, arguments))
        return ToolResult(200, [{"type": "text", "text": f"{tool_name} ok"}])

    def classify_tool_error(self, handle, task, tool_name, status, diagnostic) -> ToolOutcome:
        return ToolOutcome.MODEL_RESULT

    def provenance(self, handle) -> dict[str, Any]:
        return {"backend": self.name, "state": "fixture"}

    async def stop(self, session_id: str) -> bool:
        self.stopped.append(session_id)
        return self.handles.pop(session_id, None) is not None

    async def close(self) -> None:
        self.handles.clear()


def test_the_public_tree_registers_only_scale_live() -> None:
    assert list_backends() == ["scale_live"]
    assert get_backend("scale_live") is ScaleLiveBackend


@pytest.mark.parametrize("name", ["nope", "Scale-Live", ""])
def test_an_unknown_backend_name_is_a_configuration_error(name: str) -> None:
    with pytest.raises(ValueError, match=f"Unknown MCP-Atlas backend {name!r}. Available backends: scale_live"):
        get_backend(name)


def test_a_server_with_an_unknown_backend_does_not_start() -> None:
    with pytest.raises(ValueError, match="Unknown MCP-Atlas backend 'nope'"):
        make_server("http://unused", judge_client(), backend="nope")


def test_registration_rejects_taken_and_malformed_names() -> None:
    register_backend(OTHER, RecordingBackend)
    assert list_backends() == [OTHER, "scale_live"]
    with pytest.raises(ValueError, match="already registered"):
        register_backend(OTHER, RecordingBackend)
    with pytest.raises(ValueError, match="already registered"):
        register_backend("scale_live", RecordingBackend)
    with pytest.raises(ValueError, match="lowercase identifier"):
        register_backend("Other Backend", RecordingBackend)
    register_backend(OTHER, ScaleLiveBackend, override=True)
    assert get_backend(OTHER) is ScaleLiveBackend


def test_a_backend_module_named_after_the_backend_registers_itself_on_import(monkeypatch) -> None:
    module_name = f"{backends.__name__}.{OTHER}"
    module = types.ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, module)
    # Importing the module is the registration; an empty module is an error, not a silent fallback.
    with pytest.raises(ValueError, match=f"did not register the MCP-Atlas backend {OTHER!r}"):
        get_backend(OTHER)
    register_backend(OTHER, RecordingBackend)
    assert get_backend(OTHER) is RecordingBackend


def test_scale_live_takes_no_backend_options() -> None:
    with pytest.raises(ValueError, match="backend_options must be empty"):
        make_server("http://unused", judge_client(), backend_options={"state": "x"})


def test_backend_task_reads_the_neutral_row_schema() -> None:
    task = backend_task({**META, "backend": "scale_live", "backend_metadata": {"k": 1}}, backend="scale_live")
    assert task == BackendTask(
        task_id="t1", tools=tuple(META["enabled_tools"]), servers=tuple(META["servers"]), metadata={"k": 1}
    )
    # A row without a backend field predates the field and runs on whatever the server is configured with.
    assert backend_task(META, backend=OTHER).metadata == {}
    with pytest.raises(
        BackendConfigurationError, match="prepared for backend 'scale_live'; this server runs 'other_backend'"
    ):
        backend_task({**META, "backend": "scale_live"}, backend=OTHER)
    with pytest.raises(BackendConfigurationError, match="backend_metadata must be an object"):
        backend_task({**META, "backend_metadata": [1]}, backend="scale_live")


async def test_a_row_of_another_backend_is_refused_at_seed_before_the_backend_starts() -> None:
    server = make_server("http://unused", judge_client())
    server._backend = AsyncMock()
    body = MCPAtlasSeedSessionRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        verifier_metadata={**META, "backend": OTHER},
    )
    with pytest.raises(HTTPException) as rejected:
        await server.seed_session(http_request(), body)
    assert rejected.value.status_code == 422
    assert f"prepared for backend {OTHER!r}" in rejected.value.detail
    server._backend.start.assert_not_awaited()


async def test_a_saved_row_of_another_backend_is_refused_at_verify_without_judging() -> None:
    client = judge_client("fulfilled", "fulfilled")
    server = make_server("http://unused", client)
    body = MCPAtlasVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=verify_body("answer").response,
        verifier_metadata={**META, "backend": OTHER},
        service_errors=[],
        tool_calls=0,
    )
    with pytest.raises(HTTPException) as rejected:
        await server.verify(http_request("reverify"), body)
    assert rejected.value.status_code == 422
    client.post.assert_not_awaited()


async def test_verify_records_the_backend_and_its_provenance() -> None:
    env = FakeAgentEnvironment(
        ["ddg-search_search", "calculator_calculate"],
        {
            "ddg-search_search": (200, [{"type": "text", "text": "No results were found for your query."}]),
        },
    )
    url = await env.start()
    try:
        # Two claims, judged at verify and once more at reverify.
        server = make_server(url, judge_client(*["fulfilled"] * 4))
        meta = {
            **META,
            "enabled_tools": ["ddg-search_search", "calculator_calculate"],
            "servers": ["calculator", "ddg-search"],
        }
        await seed(server, metadata=meta)
        assert (
            await server.call_tool(http_request(), "ddg-search_search", {"query": "x"})
            == "No results were found for your query."
        )
        assert await server.call_tool(http_request(), "calculator_calculate", {"expression": "1"}) == "ok"
        body = verify_body("Created 2013, registered 2006.")
        body.verifier_metadata = meta
        result = await server.verify(http_request(), body)
        # A known-unavailable service reply is a tool result the model saw: counted, scored, never masked.
        assert result.backend == "scale_live"
        assert result.provenance == {
            "backend": "scale_live",
            "sandbox_image": None,
            "sandbox_provider": "shared_base_url",
            "development_mode": True,
            "known_unavailable": {"ddg_empty_results": 1},
        }
        assert result.reward == 1.0 and result.mask_sample is False
        # gym eval reverify posts the saved row without a session; the saved provenance stands in for the live one.
        saved = MCPAtlasVerifyRequest.model_validate(result.model_dump(mode="json"))
        again = await server.verify(http_request("reverify"), saved)
        assert again.provenance == result.provenance
        assert again.backend == "scale_live"
    finally:
        await env.stop()


async def test_a_registered_backend_serves_a_rollout_end_to_end() -> None:
    register_backend(OTHER, RecordingBackend)
    RecordingBackend.instances.clear()
    server = make_server(
        "http://unused", judge_client("fulfilled", "fulfilled"), backend=OTHER, backend_options={"state": "fixture"}
    )
    backend = RecordingBackend.instances[-1]
    assert server._backend is backend and backend.options == {"state": "fixture"}
    meta = {**META, "backend": OTHER, "backend_metadata": {"state": "fixture"}}
    await seed(server, metadata=meta)
    handle = backend.handles["s1"]
    assert handle.reset_for == ["t1"]
    assert (
        await server.call_tool(http_request(), "calculator_calculate", {"expression": "1"})
        == "calculator_calculate ok"
    )
    assert (
        await server.call_tool(http_request(), "not_a_task_tool", {})
        == "Error: tool 'not_a_task_tool' is not available for this task"
    )
    assert handle.calls == [("calculator_calculate", {"expression": "1"})]
    body = verify_body("Created 2013, registered 2006.")
    body.verifier_metadata = meta
    result = await server.verify(http_request(), body)
    assert result.reward == 1.0
    assert result.backend == OTHER
    assert result.provenance == {"backend": OTHER, "state": "fixture"}
    assert result.tool_calls == 1 and result.service_errors == []
    assert backend.stopped == ["s1"] and backend.handles == {}
    # A configuration problem the backend reports at start rejects the seed instead of masking a rollout.
    body = MCPAtlasSeedSessionRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        verifier_metadata={**meta, "backend_metadata": {"state": "missing"}},
    )
    with pytest.raises(HTTPException) as rejected:
        await server.seed_session(http_request("s2"), body)
    assert rejected.value.status_code == 422 and "not installed" in rejected.value.detail
