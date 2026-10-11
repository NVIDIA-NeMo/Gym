# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple Agent that runs its note tools inside a sandbox the resources server owns, for the e2e suite.

This is the borrowed sandbox pattern: the resources server created the sandbox and hands the agent a
``SandboxAccess`` in its seed reply; the agent connects to it and runs tools there. The resources server
checkpoints the sandbox. The agent exports only the access it was given and, after a restore, asks the
resources server's ``/sandbox_access`` for the current one before its next use, because a restore may have
rebuilt the sandbox from its snapshot under a new id. ``refresh_access_after_restore: false`` keeps the stale
access instead, which is the negative control.
"""

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import JsonValue, PrivateAttr


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sandbox_notes_server"))
import fake_sandbox_provider  # noqa: E402, F401  (registers the fake_remote provider)

from nemo_gym._checkpoint.agent import RestoredAgentSession  # noqa: E402
from nemo_gym.base_responses_api_agent import (  # noqa: E402
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionState,
)
from nemo_gym.global_config import get_global_config_dict  # noqa: E402
from nemo_gym.sandbox import AsyncSandbox, create_provider, resolve_provider_config  # noqa: E402
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess  # noqa: E402
from nemo_gym.server_utils import get_response_json, raise_for_status  # noqa: E402
from nemo_gym.server_utils import request as http_request  # noqa: E402
from nemo_gym.tool_access import DirectHTTPToolAccess  # noqa: E402
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig, SimpleAgentSessionState  # noqa: E402


SANDBOX_TOOLS = {"append_note", "read_notes"}


class SandboxNotesBorrowerConfig(SimpleAgentConfig):
    refresh_access_after_restore: bool = True


class SandboxNotesBorrower(SimpleAgent):
    config: SandboxNotesBorrowerConfig
    _access: dict[str, SandboxAccess] = PrivateAttr(default_factory=dict)
    _connected: dict[str, AsyncSandbox] = PrivateAttr(default_factory=dict)
    _stale: set[str] = PrivateAttr(default_factory=set)

    def _new_session_state(self, body: AgentSeedSessionRequest) -> SimpleAgentSessionState:
        # A seed brings the access; a restore installed it from the record before rebuilding the session.
        if body.sandbox_access is not None:
            self._access[body.agent_session_id] = body.sandbox_access
        elif body.agent_session_id not in self._access:
            raise ValueError("the sandbox notes borrower needs a sandbox_access from the resources server")
        return super()._new_session_state(body.model_copy(update={"sandbox_access": None}))

    async def _borrowed(
        self, session_key: str, tool_access: DirectHTTPToolAccess | None, cookies: Any
    ) -> AsyncSandbox:
        if session_key in self._stale:
            await self._disconnect(session_key)
            if self.config.refresh_access_after_restore:
                if tool_access is None:
                    raise RuntimeError("cannot refresh sandbox access without direct HTTP access to resources")
                response = await http_request(
                    method="POST",
                    url=f"{str(tool_access.base_url).rstrip('/')}/sandbox_access",
                    cookies=cookies,
                    headers=dict(tool_access.headers),
                    _internal=True,
                )
                await raise_for_status(response)
                self._access[session_key] = SandboxAccess.model_validate(await get_response_json(response))
            self._stale.discard(session_key)
        if session_key not in self._connected:
            access = self._access[session_key]
            connection = access.connection
            if not isinstance(connection, DirectSandboxConnection):
                raise ValueError("the borrower supports only direct sandbox connections")
            provider_config = resolve_provider_config(connection.provider_config_ref, get_global_config_dict())
            provider = create_provider(provider_config)
            self._connected[session_key] = await AsyncSandbox.connect(connection.descriptor, provider=provider)
        return self._connected[session_key]

    async def _disconnect(self, session_key: str) -> None:
        sandbox = self._connected.pop(session_key, None)
        if sandbox is not None:
            await sandbox.disconnect()

    async def _execute_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        tool_access: DirectHTTPToolAccess | None,
        resources_server_cookies: Any,
        in_session: bool,
        session_key: str | None,
    ) -> tuple[str, int, Any]:
        if name not in SANDBOX_TOOLS or session_key is None:
            return await super()._execute_tool_call(
                name,
                arguments,
                tool_access=tool_access,
                resources_server_cookies=resources_server_cookies,
                in_session=in_session,
                session_key=session_key,
            )
        try:
            sandbox = await self._borrowed(session_key, tool_access, resources_server_cookies)
            if name == "append_note":
                result = await sandbox.exec(f"append notes {arguments['line']}")
                return json.dumps({"success": result.return_code == 0}), 200, resources_server_cookies
            result = await sandbox.exec("read notes")
            return result.stdout or "", 200 if result.return_code == 0 else 500, resources_server_cookies
        except Exception as error:
            # A sandbox that is gone is a tool failure the model sees, not a crash of the agent.
            return json.dumps({"error": repr(error)}), 500, resources_server_cookies

    # -- partial-rollout checkpoints: the loop state from Simple Agent, plus the borrowed access -----------------

    async def export_agent_sessions(self, session_keys: list[str]) -> dict[str, dict[str, JsonValue]]:
        states = await super().export_agent_sessions(session_keys)
        exported = {}
        for key in session_keys:
            access = self._access.get(key)
            exported[key] = {**states[key], "sandbox_access": access.model_dump(mode="json") if access else None}
        return exported

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        for s in sessions:
            if s.session.get("sandbox_access"):
                self._access[s.session_key] = SandboxAccess.model_validate(s.session["sandbox_access"])
                # The owner may have rebuilt the sandbox; ask again before using it.
                self._stale.add(s.session_key)
        await super().restore_agent_sessions(
            [
                RestoredAgentSession(
                    session_key=s.session_key,
                    episode_id=s.episode_id,
                    session={key: value for key, value in s.session.items() if key != "sandbox_access"},
                )
                for s in sessions
            ]
        )

    async def retire_agent_session(self, session_key: str) -> None:
        await self._disconnect(session_key)
        self._access.pop(session_key, None)
        self._stale.discard(session_key)
        await super().retire_agent_session(session_key)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        await self._disconnect(state.request.agent_session_id)
        self._access.pop(state.request.agent_session_id, None)
        return await super()._close_agent_session_state(state)


if __name__ == "__main__":
    SandboxNotesBorrower.run_webserver()
