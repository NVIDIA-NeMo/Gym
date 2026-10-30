# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The task backend's typed resources session, with explicit ownership of verification and teardown.

It does not execute the harness or grade IF. The episode retains a session before
awaiting preparation, so even a partially prepared task is released.
"""

import asyncio
from copy import deepcopy

from pydantic import JsonValue

from nemo_gym.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.core import EpisodeId, NeMoGymResponse, TaskId
from nemo_gym.sandbox import AsyncSandbox, SandboxAccess, create_provider
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status
from resources_servers.agentic_if.episode.agent_user import check_agent_user, is_root_agent_user, normalize_agent_user
from resources_servers.agentic_if.episode.runtime import BoundSandbox


class SandboxTaskSession:
    """SWE and TB2.1 own their sandboxes through Gym's typed resources session; the episode only borrows them."""

    def __init__(
        self,
        *,
        client: ServerClient,
        server_name: str,
        row: dict[str, JsonValue],
        client_session_id: str,
        timeout_s: float,
        episode_id: EpisodeId,
        task_id: TaskId,
    ) -> None:
        self.client = client
        self.server_name = server_name
        self.row = deepcopy(row)
        # Private IF checks never participate in resource preparation or verification.
        self.row.pop("agentic_if", None)
        self.cookies: dict[str, str] = {}
        self.client_session_id = client_session_id
        self.timeout_s = timeout_s
        self.episode_id = episode_id
        self.task_id = task_id
        self.access: SandboxAccess | None = None
        self.sandbox: BoundSandbox | None = None
        self._transport: AsyncSandbox | None = None
        self._provider = None
        self._preparing = False
        self._preparation: asyncio.Task[None] | None = None
        self._closed = False
        self._finalization: asyncio.Task[dict[str, JsonValue]] | None = None

    async def _post(self, path: str, body: dict[str, JsonValue], **kwargs: object) -> dict[str, JsonValue]:
        """One request in the backend's session: send its cookies, keep any it sets; the reply must be an object."""
        response = await self.client.post(
            server_name=self.server_name, url_path=path, json=body, cookies=dict(self.cookies), **kwargs
        )
        await raise_for_status(response)
        self.cookies.update({key: getattr(value, "value", value) for key, value in response.cookies.items()})
        result = await get_response_json(response)
        if not isinstance(result, dict):
            raise RuntimeError(f"{self.server_name} {path} response must be a JSON object")
        return result

    async def _bind(self, descriptor: dict[str, JsonValue], *, workdir: str | None, user: str | int | None) -> None:
        self._transport = await AsyncSandbox.connect(descriptor, provider=self._provider)
        user = normalize_agent_user(user)
        execution_env = {}
        if not is_root_agent_user(user):
            await check_agent_user(self._transport, user)
            # Numeric-uid exec does not perform su's HOME reset. Resolve it under
            # that identity, once, rather than inherit the command daemon's HOME.
            result = await self._transport.exec(
                command='ng_home="$(getent passwd "$(id -u)" | cut -d: -f6)"; '
                'printf "%s\\n" "${ng_home:-$HOME}"; '
                'if [ -x /bin/bash ]; then printf "/bin/bash\\n"; else printf "%s\\n" "${SHELL:-/bin/sh}"; fi',
                user=user,
                timeout_s=min(self.timeout_s, 30),
            )
            values = (result.stdout or "").splitlines()
            if result.return_code != 0 or result.error_type or len(values) != 2 or not all(values):
                raise RuntimeError("could not resolve the task identity's HOME and SHELL")
            execution_env = {"HOME": values[0], "SHELL": values[1]}
        self.sandbox = BoundSandbox(
            self._transport,
            user=user,
            workdir=workdir,
            timeout_s=self.timeout_s,
            execution_env=execution_env,
        )

    async def prepare(self) -> None:
        """Prepare once; retain partial state for close if any step fails."""
        if self._preparing or self._finalization is not None:
            raise RuntimeError("task session preparation may only run once")
        self._preparing = True
        self._preparation = asyncio.create_task(self._prepare())
        await asyncio.shield(self._preparation)

    async def _prepare(self) -> None:
        seed = ResourcesSeedSessionRequest(
            resources_session_id=self.client_session_id,
            episode_id=self.episode_id,
            task_id=self.task_id,
            task_data=self.row,
        )
        seeded = await self._post("/seed_session", seed.model_dump(mode="json"))
        try:
            self.access = SandboxAccess.model_validate(seeded.get("sandbox_access"))
            self._provider = create_provider(
                resolve_provider_config(self.access.connection.provider_config_ref, self.client.global_config_dict)
            )
        except (ValueError, TypeError) as error:
            # The backend's reply or the sandbox configuration is at fault, not the task: retryable, not a 422.
            raise RuntimeError(f"{self.server_name} lent no usable sandbox: {error}") from error
        await self._bind(
            self.access.connection.descriptor,
            workdir=self.row.get("workdir") or self.access.workdir,
            user=self.row.get("agent_user"),
        )

    async def finish(self, response: NeMoGymResponse, *, params: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """Verify the executed task once, in a task that close can cancel.

        The backend's own flat verify body: the task fields it was seeded with, the actor's request and its response.
        """
        if self._finalization is not None:
            raise RuntimeError("task session finalization may only run once")
        body = self.row | {"responses_create_params": params, "response": response.model_dump(mode="json")}
        self._finalization = asyncio.create_task(self._post("/verify", body))
        return await self._finalization

    async def close(self) -> None:
        """Release the lease; a failed release raises and can be retried."""
        if self._closed:
            return
        tasks = [task for task in (self._preparation, self._finalization) if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # The allocation owner can close before seed returns and does not need
        # a healthy command daemon or an execution-side transport.
        await self._post(
            "/close_session",
            {
                "resources_session_id": self.client_session_id,
                "episode_id": self.episode_id.model_dump(mode="json"),
            },
        )
        await self._disconnect()
        self._closed = True

    async def _disconnect(self) -> None:
        if self._transport is not None:
            await self._transport.disconnect()
        elif self._provider is not None:
            await self._provider.aclose()
