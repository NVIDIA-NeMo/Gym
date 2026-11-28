# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve one episode's tools inside the episode's tool sandbox.

``python -m nemo_gym.server_runtime.resources_tool_sandbox_runner <spec.json>`` imports the ``ToolsetResourcesServer``
class the spec names, builds its ``toolset`` over the state in the spec with the server's config, and serves each tool
at ``POST /<tool name>``. It builds no Resources Server: it has no session, no seed and no verifier, so the agent's
calls carry no cookie. After each call, and before the reply, it writes the state to the spec's ``state_path``, so the
file the Environment Server reads after the agent's turn holds the effect of every tool call. See
``nemo_gym.server_runtime.resources_tool_sandbox``.
"""

import asyncio
import importlib
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, JsonValue

from nemo_gym.server_runtime.toolset_resources_server import (
    ResourcesToolset,
    ToolsetResourcesServer,
    register_tool_routes,
)


class ToolSandboxRunnerSpec(BaseModel):
    """What the runner needs, written by the Environment Server into the tool sandbox."""

    model_config = ConfigDict(extra="forbid")

    # The Gym repository root in the image, and the server's module under it, such as
    # ``resources_servers.example_session_state_mgmt.app``.
    server_root: str
    server_module: str
    # The server class; unset, the one ToolsetResourcesServer subclass the module defines.
    class_name: str | None = None
    server_config: dict[str, Any]
    # The session's state from the host's /export_session_state; None when the tools keep no state.
    state: JsonValue = None
    port: int
    state_path: str


def load_server_class(spec: ToolSandboxRunnerSpec) -> type[ToolsetResourcesServer]:
    """Import the server's module, not as ``__main__``, so its own entrypoint does not start a server."""
    if spec.server_root not in sys.path:
        sys.path.insert(0, spec.server_root)
    module = importlib.import_module(spec.server_module)
    if spec.class_name is not None:
        server_class = getattr(module, spec.class_name)
        if not (isinstance(server_class, type) and issubclass(server_class, ToolsetResourcesServer)):
            raise TypeError(f"{spec.server_module}.{spec.class_name} is not a ToolsetResourcesServer")
        return server_class
    defined = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, ToolsetResourcesServer)
        and value.__module__ == module.__name__
    ]
    if len(defined) != 1:
        names = sorted(server_class.__name__ for server_class in defined)
        raise ValueError(
            f"{spec.server_module} defines {names or 'no'} ToolsetResourcesServer classes; set class_name"
        )
    return defined[0]


def write_state(state: BaseModel | None, path: Path) -> None:
    """Replace the state file whole with the state's JSON, or ``null`` when the tools keep no state."""
    partial = path.with_suffix(".partial")
    partial.write_text(state.model_dump_json() if state is not None else "null")
    os.replace(partial, path)


def build_tool_app(spec: ToolSandboxRunnerSpec) -> FastAPI:
    """Build the app that serves the tools over the spec's state, and write the start state."""
    server_class = load_server_class(spec)
    config = server_class.model_fields["config"].annotation.model_validate(spec.server_config)
    state_model = server_class.session_state_model
    state = state_model.model_validate(spec.state) if state_model is not None else None
    state_path = Path(spec.state_path)
    # One session: its calls run one at a time, as under the session lock on the host.
    lock = asyncio.Lock()

    @asynccontextmanager
    async def open_toolset(request: Request) -> AsyncIterator[ResourcesToolset]:
        async with lock:
            try:
                yield server_class.toolset(config=config, state=state)
            finally:
                write_state(state, state_path)

    app = FastAPI()
    register_tool_routes(app, server_class.toolset, open_toolset)
    write_state(state, state_path)
    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit("usage: python -m nemo_gym.server_runtime.resources_tool_sandbox_runner <spec.json>")
    spec = ToolSandboxRunnerSpec.model_validate_json(Path(args[0]).read_text())
    uvicorn.run(build_tool_app(spec), host="0.0.0.0", port=spec.port, log_level="warning")


if __name__ == "__main__":
    main()
