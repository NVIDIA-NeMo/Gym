# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A ToolsetResourcesServer's tools served in an episode's tool sandbox."""

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import BaseModel, JsonValue

from nemo_gym.server_runtime.base_resources_server import BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.server_runtime.episode_tools import episode_tool
from nemo_gym.server_runtime.resources_tool_sandbox import (
    ResourcesToolSandboxConfig,
    ResourcesToolSandboxes,
    _resources_server_module,
)
from nemo_gym.server_runtime.resources_tool_sandbox_runner import (
    ToolSandboxRunnerSpec,
    build_tool_app,
    load_server_class,
)
from nemo_gym.server_runtime.toolset_resources_server import ResourcesToolset, ToolsetResourcesServer


class CounterState(BaseModel):
    count: int


class AddArguments(BaseModel):
    amount: int


class CounterTools(ResourcesToolset):
    state: CounterState

    @episode_tool
    async def add(self, arguments: AddArguments) -> CounterState:
        """Add to the counter."""
        self.state.count += arguments.amount
        return self.state

    @episode_tool
    async def fail(self, arguments: AddArguments) -> CounterState:
        """Add to the counter, then fail."""
        self.state.count += arguments.amount
        raise RuntimeError("the tool failed after changing the state")


class CounterServer(ToolsetResourcesServer):
    session_state_model = CounterState
    toolset = CounterTools

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        raise AssertionError("the tool sandbox never verifies")


class EchoTools(ResourcesToolset):
    @episode_tool
    async def echo(self, arguments: AddArguments) -> dict[str, int]:
        """Echo the amount."""
        return {"amount": arguments.amount}


class EchoServer(ToolsetResourcesServer):
    toolset = EchoTools

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        raise AssertionError("the tool sandbox never verifies")


def _runner_spec(tmp_path: Path, class_name: str | None, state: JsonValue = None) -> ToolSandboxRunnerSpec:
    return ToolSandboxRunnerSpec(
        server_root=str(tmp_path),
        server_module=__name__,
        class_name=class_name,
        server_config={"name": "resources", "host": "h", "port": 1, "entrypoint": "app.py"},
        state=state,
        port=8000,
        state_path=str(tmp_path / "state.json"),
    )


def _state_file(tmp_path: Path) -> JsonValue:
    return json.loads((tmp_path / "state.json").read_text())


class TestRunner:
    def test_the_tools_change_the_state_and_the_file_follows_each_call(self, tmp_path: Path) -> None:
        tools = TestClient(build_tool_app(_runner_spec(tmp_path, "CounterServer", state={"count": 3})))
        assert _state_file(tmp_path) == {"count": 3}

        # No cookie and no seed: the sandbox serves one session, whose state the spec holds.
        assert tools.post("/add", json={"amount": 4}).json() == {"count": 7}
        assert _state_file(tmp_path) == {"count": 7}
        assert tools.post("/add", json={"amount": 1}).json() == {"count": 8}
        assert _state_file(tmp_path) == {"count": 8}

    def test_a_tool_that_fails_still_leaves_its_change_in_the_file(self, tmp_path: Path) -> None:
        tools = TestClient(
            build_tool_app(_runner_spec(tmp_path, "CounterServer", state={"count": 3})), raise_server_exceptions=False
        )
        assert tools.post("/fail", json={"amount": 2}).status_code == 500
        # As on the host, where the change stays in the session's state.
        assert _state_file(tmp_path) == {"count": 5}

    def test_only_the_tools_are_served(self, tmp_path: Path) -> None:
        app = build_tool_app(_runner_spec(tmp_path, "CounterServer", state={"count": 0}))
        tools = TestClient(app)

        assert {route.path for route in app.routes if isinstance(route, APIRoute)} == {"/add", "/fail"}
        for path in ["/seed_session", "/verify", "/close_session", "/export_session_state", "/health"]:
            assert tools.post(path, json={}).status_code == 404, path

    def test_a_stateless_servers_tools_run_and_the_state_is_null(self, tmp_path: Path) -> None:
        tools = TestClient(build_tool_app(_runner_spec(tmp_path, "EchoServer")))
        assert tools.post("/echo", json={"amount": 5}).json() == {"amount": 5}
        assert _state_file(tmp_path) is None

    def test_a_module_with_several_servers_needs_class_name(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="set class_name"):
            load_server_class(_runner_spec(tmp_path, class_name=None))

    def test_class_name_must_name_a_toolset_server(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="is not a ToolsetResourcesServer"):
            load_server_class(_runner_spec(tmp_path, class_name="CounterTools"))


def test_the_runner_imports_the_server_module_from_its_entrypoint() -> None:
    global_config = OmegaConf.create(
        {"counter": {"resources_servers": {"example_session_state_mgmt": {"entrypoint": "app.py", "port": 1}}}}
    )
    assert _resources_server_module(global_config, "counter") == (
        "resources_servers.example_session_state_mgmt.app",
        {"entrypoint": "app.py", "port": 1, "name": "counter"},
    )


class _Sandbox:
    def __init__(self, state: JsonValue) -> None:
        self.state = state

    async def download(self, remote_path: str, local_path: Path) -> None:
        Path(local_path).write_text(json.dumps(self.state))


class _Owned:
    def __init__(self, sandboxes: dict[str, Any]) -> None:
        self.sandboxes = sandboxes
        self.stopped: list[str] = []

    def get(self, key: str) -> Any:
        return self.sandboxes.get(key)

    async def stop(self, key: str) -> None:
        self.stopped.append(key)


async def test_finishing_reads_the_last_state_then_stops_the_sandbox() -> None:
    global_config = OmegaConf.create({"counter": {"resources_servers": {"counter": {"entrypoint": "app.py"}}}})
    sandboxes = ResourcesToolSandboxes(
        ResourcesToolSandboxConfig(image="gym:1"), resources_server_name="counter", global_config=global_config
    )
    owned = _Owned({"resources-session-1": _Sandbox({"count": 7})})

    assert await sandboxes.finish(owned, session_id="resources-session-1") == {"count": 7}  # type: ignore[arg-type]
    assert owned.stopped == ["resources-session-1"]
