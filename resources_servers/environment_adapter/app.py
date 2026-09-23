# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run environments defined by ``environment.yaml`` through Resources Server contracts."""

from __future__ import annotations

import inspect
import posixpath
import sys
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI, Request
from pydantic import Field

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    SimpleResourcesServer,
)
from nemo_gym.environment.authoring import (
    MaterializedEnvironmentTask,
    load_environment,
    load_environment_callable,
    materialize_tasks,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY
from nemo_gym.single_agent_episode_types import ResponsesResourcesVerifyRequest


class EnvironmentAdapterResourcesServerConfig(BaseResourcesServerConfig):
    """Bind one trusted environment definition to a sandbox provider."""

    environment_root: str
    runtime_image: str
    sandbox_provider: str
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    trusted_environment_code: bool = False


class SandboxWorkspace:
    """Expose scoped UTF-8 reads from an owner-managed sandbox."""

    def __init__(self, sandbox: AsyncSandbox, workdir: str) -> None:
        self._sandbox = sandbox
        self._workdir = PurePosixPath(posixpath.normpath(workdir))

    async def read_text(self, path: str) -> str:
        requested = PurePosixPath(path)
        if not requested.is_absolute():
            requested = self._workdir / requested
        normalized = PurePosixPath(posixpath.normpath(str(requested)))
        if normalized != self._workdir and self._workdir not in normalized.parents:
            raise OSError(f"workspace path escapes {self._workdir}: {path}")
        try:
            with tempfile.TemporaryDirectory(prefix="nemo-gym-environment-read-") as directory:
                local_path = Path(directory) / "value"
                await self._sandbox.download(str(normalized), local_path)
                return local_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            raise
        except Exception as error:
            raise OSError(f"could not read workspace path {normalized}: {error}") from error


@dataclass(frozen=True)
class EnvironmentAttempt:
    """Evidence available to an environment-local verifier."""

    workspace: SandboxWorkspace
    response: NeMoGymResponse
    tool_calls: tuple[Any, ...] = ()


class EnvironmentAdapterResourcesServer(SimpleResourcesServer):
    """Own standard environment resources and invoke the selected verifier."""

    config: EnvironmentAdapterResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("Environment adapter sessions require num_workers=1")
        if not self.config.trusted_environment_code:
            raise ValueError(
                "Environment-local verifiers execute in the resources-server process; "
                "set trusted_environment_code=true only for reviewed environment code"
            )
        environment_root = _resolve_under_cwd_or_install(self.config.environment_root, validator=Path.is_dir)
        self._loaded_environment = load_environment(environment_root)
        tasks = materialize_tasks(self._loaded_environment)
        self._tasks_by_id = {task.materialized.task_id: task for task in tasks}
        self._verifiers_by_implementation = {
            task.verifier.implementation: load_environment_callable(
                self._loaded_environment,
                task.verifier.implementation,
                description=f"verifier for {task.materialized.task_id.task_id}",
            )
            for task in tasks
        }
        self._session_id_to_sandbox: dict[str, AsyncSandbox] = {}
        self._session_id_to_identity: dict[str, tuple[EpisodeId, TaskId]] = {}

    def _task_for(self, task_id: TaskId) -> MaterializedEnvironmentTask:
        try:
            return self._tasks_by_id[task_id]
        except KeyError as error:
            raise ValueError(f"TaskId does not belong to the configured environment: {task_id}") from error

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/close_session")(self.close_session)
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as maybe_state:
                    yield maybe_state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        return app

    async def _create_sandbox(self) -> AsyncSandbox:
        global_config = get_global_config_dict()
        provider = resolve_provider_config(self.config.sandbox_provider, global_config)
        provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config)
        runtime = self._loaded_environment.definition.runtime
        sandbox_config = self.config.sandbox_config
        spec = SandboxSpec(
            image=self.config.runtime_image,
            ttl_s=sandbox_config.get("ttl_s"),
            ready_timeout_s=sandbox_config.get("ready_timeout_s"),
            workdir=runtime.workdir,
            env=dict(sandbox_config.get("env", {})),
            metadata=provider_metadata
            | dict(sandbox_config.get("metadata", {}))
            | {"environment": self._loaded_environment.definition.name},
            resources=SandboxResources.from_mapping(sandbox_config.get("resources", {})),
            provider_options=dict(sandbox_config.get("provider_options", {})),
        )
        sandbox = AsyncSandbox(provider)
        await sandbox.start(spec)
        return sandbox

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest,
    ) -> ResourcesSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        task = self._task_for(body.task_id)
        if body.task_data != task.materialized.task_input.task_data:
            raise ValueError("Task data does not match the trusted environment declaration")

        previous = self._session_id_to_sandbox.pop(session_id, None)
        self._session_id_to_identity.pop(session_id, None)
        if previous is not None:
            await previous.stop()

        sandbox = await self._create_sandbox()
        try:
            descriptor = await sandbox.serialize()
        except BaseException:
            await sandbox.stop()
            raise
        self._session_id_to_sandbox[session_id] = sandbox
        self._session_id_to_identity[session_id] = (body.episode_id, body.task_id)
        return ResourcesSeedSessionResponse(
            resources_session_id=session_id,
            sandbox_access=SandboxAccess(
                connection=DirectSandboxConnection(
                    provider_config_ref=self.config.sandbox_provider,
                    descriptor=descriptor,
                ),
                workdir=self._loaded_environment.definition.runtime.workdir,
            ),
        )

    async def verify(
        self,
        request: Request,
        body: ResponsesResourcesVerifyRequest,
    ) -> BaseVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        sandbox = self._session_id_to_sandbox.get(session_id)
        identity = self._session_id_to_identity.get(session_id)
        if sandbox is None or identity is None:
            raise ValueError("Unknown environment adapter resources session")
        if identity != (body.episode_id, body.task_id):
            raise ValueError("Verification identity does not match the seeded resources session")

        task = self._task_for(body.task_id)
        attempt = EnvironmentAttempt(
            workspace=SandboxWorkspace(sandbox, self._loaded_environment.definition.runtime.workdir),
            response=body.verification_input.response,
        )
        verifier_input = SimpleNamespace(**task.verifier.verifier_input)
        verifier = self._verifiers_by_implementation[task.verifier.implementation]
        result = verifier(attempt, verifier_input)
        reward = await result if inspect.isawaitable(result) else result
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise TypeError("Environment verifier must return a numeric reward")
        return BaseVerifyResponse(
            responses_create_params=body.verification_input.responses_create_params,
            response=body.verification_input.response,
            reward=float(reward),
        )

    async def close_session(
        self,
        request: Request,
        body: ResourcesCloseSessionRequest,
    ) -> ResourcesCloseSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        if body.resources_session_id != session_id:
            raise ValueError("resources_session_id does not match the session cookie")
        identity = self._session_id_to_identity.get(session_id)
        if identity is None:
            raise ValueError(f"Unknown resources session: {session_id}")
        if identity[0] != body.episode_id:
            raise ValueError("episode_id does not match the seeded resources session")
        sandbox = self._session_id_to_sandbox.pop(session_id, None)
        self._session_id_to_identity.pop(session_id, None)
        if sandbox is not None:
            await sandbox.stop()
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def shutdown(self) -> None:
        sandboxes = list(self._session_id_to_sandbox.values())
        self._session_id_to_sandbox.clear()
        self._session_id_to_identity.clear()
        for sandbox in sandboxes:
            try:
                await sandbox.stop()
            except Exception:
                print("Failed to stop abandoned environment adapter sandbox", file=sys.stderr)


if __name__ == "__main__":
    EnvironmentAdapterResourcesServer.run_webserver()
