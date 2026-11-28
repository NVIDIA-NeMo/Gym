# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve a ``ToolsetResourcesServer``'s tools from a sandbox of each episode, and verify on the host.

A ``ToolsetResourcesServer`` keeps its tools (a ``ResourcesToolset``) apart from its verifier, joined only by the
session's state. With a tool sandbox the two also run apart:

1. The Environment Server seeds the Resources Server on the host as usual, and reads the session's state with
   ``/export_session_state``.
2. It starts the episode's tool sandbox, an image that holds the Gym repository, and runs
   ``resources_tool_sandbox_runner`` in it. The runner builds the server's toolset over that state with the server's
   config, and serves its tools. It builds no Resources Server.
3. It grants the agent the sandbox's endpoint as the Resources Server's ``direct_http`` tools.
4. After the agent's turn it reads the state the runner wrote, installs it on the host with
   ``/restore_session_state``, and calls ``/verify`` there. ``verifier_metadata`` never enters the sandbox.
5. It stops the sandbox.

Only a ``ToolsetResourcesServer`` serves the two state routes; another Resources Server answers 404 at the first
episode. The sandbox's network rule bounds what the tools reach, and is independent of where the agent runs.

The sandbox holds the server's config, so it must hold no secret the tools do not need. A toolset knows no other Gym
server, so tools that call one, such as a judge model, cannot run here.
"""

import asyncio
import json
import shlex
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from aiohttp import ClientError
from omegaconf import OmegaConf
from pydantic import Field, JsonValue

from nemo_gym.client.http_client import request as http_request
from nemo_gym.sandbox.config import SandboxConfig
from nemo_gym.sandbox.providers.base import SandboxEndpoint
from nemo_gym.server_runtime.owned_sandboxes import OwnedSandboxes
from nemo_gym.server_runtime.resources_tool_sandbox_runner import ToolSandboxRunnerSpec


RUN_DIR = "/tmp/nemo-gym-tool-sandbox"
SPEC_PATH = f"{RUN_DIR}/spec.json"
STATE_PATH = f"{RUN_DIR}/state.json"
LOG_PATH = f"{RUN_DIR}/runner.log"


class ResourcesToolSandboxConfig(SandboxConfig):
    """The tool sandbox of each episode: its image, the Resources Server class, and the sandbox settings."""

    # An image with the Gym repository at ``server_root`` and a Python that has nemo-gym and the server's own
    # requirements installed.
    image: str
    server_root: str = "/gym"
    python: str = "python"
    # Unset, the one ToolsetResourcesServer class the server's entrypoint module defines.
    class_name: str | None = None
    port: int = Field(default=8000, ge=1, le=65535)
    ttl_s: int | float = 3600
    ready_timeout_s: int | float = 120


class ResourcesToolSandboxes:
    """The tool sandboxes an Environment Server starts for its Resources Server, one per episode, keyed by the
    episode's resources session ID in the server's ``self.sandboxes``."""

    def __init__(
        self, config: ResourcesToolSandboxConfig, *, resources_server_name: str, global_config: Mapping[str, Any]
    ) -> None:
        self.config = config
        self.server_module, self.server_config = _resources_server_module(global_config, resources_server_name)

    async def start(self, owned: OwnedSandboxes, *, session_id: str, state: JsonValue) -> SandboxEndpoint:
        """Start the session's tool sandbox with ``state`` installed; return the endpoint that serves its tools.

        Raises:
            RuntimeError: If the runner does not start or answer within ``ready_timeout_s``; its log is included.
        """
        runner_spec = ToolSandboxRunnerSpec(
            server_root=self.config.server_root,
            server_module=self.server_module,
            class_name=self.config.class_name,
            server_config=self.server_config,
            state=state,
            port=self.config.port,
            state_path=STATE_PATH,
        )
        spec = owned.spec(
            self.config, image=self.config.image, workdir=None, files={SPEC_PATH: runner_spec.model_dump_json()}
        )
        sandbox = await owned.create(session_id, replace(spec, ports=(self.config.port,)))
        command = (
            f"cd {shlex.quote(self.config.server_root)} && nohup {shlex.quote(self.config.python)} "
            f"-m nemo_gym.server_runtime.resources_tool_sandbox_runner {SPEC_PATH} > {LOG_PATH} 2>&1 &"
        )
        started = await sandbox.exec(command, timeout_s=30)
        if started.error_type is not None or started.return_code != 0:
            raise RuntimeError(f"The tool sandbox runner did not start: {started.error_type} {started.stderr or ''}")
        endpoint = await sandbox.endpoint(self.config.port)
        deadline = asyncio.get_running_loop().time() + self.config.ready_timeout_s
        while True:
            try:
                # Any HTTP reply means the runner serves; it has no route at "/".
                reply = await http_request("GET", endpoint.endpoint, headers=endpoint.headers, _max_num_tries=1)
                reply.release()
                return endpoint
            except (ClientError, OSError):
                pass
            if asyncio.get_running_loop().time() >= deadline:
                log = await sandbox.exec(f"tail -c 4000 {LOG_PATH}", timeout_s=30)
                raise RuntimeError(
                    f"The tool sandbox runner did not answer within {self.config.ready_timeout_s} s:\n"
                    f"{log.stdout or ''}"
                )
            await asyncio.sleep(0.5)

    async def finish(self, owned: OwnedSandboxes, *, session_id: str) -> JsonValue:
        """Read the session's state as the runner last wrote it, then stop the sandbox."""
        sandbox = owned.get(session_id)
        if sandbox is None:
            raise RuntimeError(f"No tool sandbox runs for session {session_id}")
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "state.json"
            await sandbox.download(STATE_PATH, local)
            state = json.loads(local.read_text())
        await owned.stop(session_id)
        return state


def _resources_server_module(global_config: Mapping[str, Any], server_name: str) -> tuple[str, dict[str, Any]]:
    """The module of the Resources Server instance ``server_name``, such as ``resources_servers.calendar.app``,
    and its config."""
    instance = global_config[server_name]["resources_servers"]
    ((server_type, server_config),) = instance.items()
    if OmegaConf.is_config(server_config):
        server_config = OmegaConf.to_container(server_config, resolve=True)
    entrypoint = Path(server_config["entrypoint"])
    module = ".".join(("resources_servers", server_type, *entrypoint.with_suffix("").parts))
    return module, dict(server_config) | {"name": server_name}
