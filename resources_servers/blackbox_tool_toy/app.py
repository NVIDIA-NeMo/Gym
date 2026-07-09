# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""blackbox_tool_toy — a tiny environment that lends a tool over MCP.

The task can ONLY be solved by calling this server's ``get_passphrase`` tool
over MCP: the passphrase is not in the prompt and not in any seeded file. The
harness must discover the Gym MCP server, call the tool (its call carries the
per-rollout signed session token), and write the returned passphrase to
answer.txt.

verify() recomputes the expected passphrase from the rollout id and compares.
So a broken MCP wiring, an unmade tool call, or a call correlated to the wrong
rollout all score 0 — which is exactly what makes this a validation of the
resources-server-as-MCP-server connection.

Correlation: seed_session mints the per-rollout MCP session and remembers which
rollout it belongs to; the tool receives the session id (resolved from the
signed token by the base class) and answers with that rollout's passphrase.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from fastapi import Request
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    MCPResourcesServer,
    gym_tool,
)
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint


def passphrase_for(rollout_id: str) -> str:
    """Deterministic, unguessable-without-the-tool passphrase for a rollout."""
    return "gym-" + hashlib.sha1(f"ng-passphrase:{rollout_id}".encode()).hexdigest()[:10]


TASK_PROMPT = (
    "A tool named get_passphrase is available to you over MCP. "
    "Call it to obtain a passphrase, then write EXACTLY the returned passphrase "
    "(and nothing else, no quotes, no trailing text) to a file named answer.txt "
    "in your current working directory. You cannot guess the passphrase; you "
    "must obtain it from the tool."
)


def tool_task_row(index: int = 0) -> dict:
    return {
        "task_index": index,
        "responses_create_params": {"input": [{"type": "message", "role": "user", "content": TASK_PROMPT}]},
        "verifier_metadata": {},
    }


class BlackboxToolToyConfig(BaseResourcesServerConfig):
    pass


class ToolToySeedRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")


class ToolToySeedResponse(BaseSeedSessionResponse):
    model_config = ConfigDict(extra="allow")
    sandbox_spec: dict = {}
    mcp: dict = {}
    ng_rollout_id: str = ""
    # What the env wants harvested from the sandbox to grade (this task writes
    # the passphrase to answer.txt). The agent is env-agnostic; the extraction
    # spec lives here, not in the agent. `commands` (e.g. a git diff) is the
    # same mechanism for patch/exec-based envs.
    harvest: dict = {"files": ["answer.txt"]}


class ToolToyVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    response: dict = {}
    blackbox_outcome: Optional[dict] = None


class ToolToyVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    response: dict = {}


class BlackboxToolToyResourcesServer(MCPResourcesServer):
    config: BlackboxToolToyConfig

    def model_post_init(self, context) -> None:
        # Maps an MCP session id -> the rollout it was seeded for, so the tool
        # can answer with the right rollout's passphrase. In-memory, so this
        # env expects a single worker (fine for the local validation stack).
        self._rollout_by_session: dict[str, str] = {}
        return super().model_post_init(context)

    async def seed_session(self, body: ToolToySeedRequest, request: Request) -> ToolToySeedResponse:
        rollout_id = str(getattr(body, "ng_rollout_id", "") or "")
        metadata = self.build_mcp_session_metadata(request)
        session_id = request.session.get(SESSION_ID_KEY)
        if session_id and rollout_id:
            self._rollout_by_session[session_id] = rollout_id
        return ToolToySeedResponse(
            sandbox_spec={"files": {}},
            mcp=metadata.model_dump(),
            ng_rollout_id=rollout_id,
            harvest={"files": ["answer.txt"]},
        )

    @gym_tool
    def get_passphrase(self, session_id: str) -> str:
        """Return the passphrase for this task. Write the returned value verbatim
        into a file named answer.txt to complete the task."""
        rollout_id = self._rollout_by_session.get(session_id)
        if not rollout_id:
            return "NO-ACTIVE-SESSION"
        return passphrase_for(rollout_id)

    async def verify(self, body: ToolToyVerifyRequest) -> ToolToyVerifyResponse:
        data = body.model_dump()
        rollout_id = str(data.get("ng_rollout_id") or data.get("rollout_id") or "")
        expected = passphrase_for(rollout_id) if rollout_id else None
        got = str((body.blackbox_outcome or {}).get("outcome_text") or "").strip()
        resolved = bool(expected) and got == expected
        info = {"expected": expected, "got": got[:120], "rollout_id": rollout_id}
        return ToolToyVerifyResponse(**data, reward=1.0 if resolved else 0.0, is_resolved=resolved, info=info)


if __name__ == "__main__":
    BlackboxToolToyResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = BlackboxToolToyResourcesServer.run_webserver()  # noqa: F401
