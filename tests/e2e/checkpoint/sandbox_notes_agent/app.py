# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple Agent that runs two tools inside a sandbox it owns per session, for the checkpoint e2e suite.

This is the agent-owned sandbox pattern: the agent created the sandbox, so the agent checkpoints it. Its
session export carries the Simple Agent loop state plus the sandbox's checkpoint state, a restore rebuilds
both, and a retire or close stops the sandbox. ``append_note`` appends a line to a file in the sandbox and
``read_notes`` returns the file, so the verifier can tell from the trajectory what the sandbox held.
"""

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import JsonValue, PrivateAttr


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sandbox_notes_server"))
from fake_sandbox_provider import RemoteFakeSandboxProvider  # noqa: E402  (registers the provider)

from nemo_gym._checkpoint.agent import RestoredAgentSession  # noqa: E402
from nemo_gym.base_responses_api_agent import AgentCloseSessionResponse, AgentSessionState  # noqa: E402
from nemo_gym.sandbox.checkpoint import SandboxSessionCheckpointer  # noqa: E402
from nemo_gym.sandbox.providers.base import SandboxSpec  # noqa: E402
from nemo_gym.tool_access import DirectHTTPToolAccess  # noqa: E402
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig  # noqa: E402


SANDBOX_TOOLS = {"append_note", "read_notes"}


class SandboxNotesAgentConfig(SimpleAgentConfig):
    sandbox_backend_url: str


class SandboxNotesAgent(SimpleAgent):
    config: SandboxNotesAgentConfig
    _sandboxes: SandboxSessionCheckpointer = PrivateAttr(default=None)

    def setup_webserver(self):
        provider = RemoteFakeSandboxProvider(self.config.sandbox_backend_url)
        self._sandboxes = SandboxSessionCheckpointer(provider, parallelism=8)
        return super().setup_webserver()

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
        if name not in SANDBOX_TOOLS:
            return await super()._execute_tool_call(
                name,
                arguments,
                tool_access=tool_access,
                resources_server_cookies=resources_server_cookies,
                in_session=in_session,
                session_key=session_key,
            )
        key = session_key or "unscoped"
        if key not in self._sandboxes:
            await self._sandboxes.create(key, SandboxSpec(image="notes:1", workdir="/work"))
        sandbox = self._sandboxes.get(key)
        if name == "append_note":
            result = await sandbox.exec(f"append notes {arguments['line']}")
            return json.dumps({"success": result.return_code == 0}), 200, resources_server_cookies
        result = await sandbox.exec("read notes")
        return result.stdout or "", 200, resources_server_cookies

    # -- partial-rollout checkpoints: the loop state from Simple Agent, plus the sandbox -----------------------

    async def export_agent_sessions(self, session_keys: list[str]) -> dict[str, dict[str, JsonValue]]:
        states = await super().export_agent_sessions(session_keys)
        sandboxes = await self._sandboxes.export([key for key in session_keys if key in self._sandboxes])
        return {key: {**states[key], "sandbox": sandboxes.get(key)} for key in session_keys}

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        with_sandbox = {s.session_key: s.session["sandbox"] for s in sessions if s.session.get("sandbox")}
        if with_sandbox:
            # Validates every sandbox state, then rebuilds all of them or none.
            await self._sandboxes.restore(with_sandbox)
        try:
            await super().restore_agent_sessions(
                [
                    RestoredAgentSession(
                        session_key=s.session_key,
                        episode_id=s.episode_id,
                        session={key: value for key, value in s.session.items() if key != "sandbox"},
                    )
                    for s in sessions
                ]
            )
        except BaseException:
            for key in with_sandbox:
                await self._sandboxes.stop(key)
            raise

    async def retire_agent_session(self, session_key: str) -> None:
        await self._sandboxes.stop(session_key)
        await super().retire_agent_session(session_key)

    async def park_agent_sessions(self, session_keys: list[str]) -> None:
        await self._sandboxes.park(session_keys)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        # The episode is over: free its sandbox and the checkpoint snapshots of it.
        await self._sandboxes.stop(state.request.agent_session_id, forget_snapshots=True)
        return await super()._close_agent_session_state(state)


if __name__ == "__main__":
    SandboxNotesAgent.run_webserver()
