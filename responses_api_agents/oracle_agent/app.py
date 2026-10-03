# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Oracle agent: run a Harbor task's reference solution instead of a model.

The oracle is an agent harness like any other. It borrows the sandbox the ``harbor``
resources server seeded, finds the task folder through that server's taskset mapping
in the run config, uploads ``solution/`` and runs ``solve.sh``. Verification then
scores the result the same way it scores a model's work, so ``gym dataset validate``
is an ordinary run with this agent selected.

A task without ``solution/solve.sh`` is not failed: the oracle returns a response
tagged ``oracle=unvalidated`` and does nothing in the sandbox.
"""

import asyncio
import shlex
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4

from fastapi import Body, HTTPException, Request
from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict, get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.access import DirectSandboxConnection
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.providers import create_provider
from nemo_gym.tasks.harbor import HarborTask, load_task
from nemo_gym.tasks.harbor.task import HarborTaskError
from resources_servers.harbor.sandbox_io import upload_dir


SOLUTION_DIR = "/solution"
ORACLE_METADATA_KEY = "oracle"
ORACLE_STATUS_SOLVED = "solved"
ORACLE_STATUS_UNVALIDATED = "unvalidated"
ORACLE_STATUS_FAILED = "failed"
_SESSION_KEY = "oracle_agent_session_id"


class OracleAgentConfig(BaseResponsesAPIAgentConfig):
    model_config = ConfigDict(extra="forbid")

    # The `harbor` resources server whose `tasksets` mapping locates task folders.
    resources_server: ResourcesServerRef
    # Wall-clock cap for solve.sh when the task's [agent].timeout_sec is larger.
    max_solve_timeout_s: float = 3600.0
    output_tail_chars: int = 4000


@dataclass
class OracleSession:
    request: AgentSeedSessionRequest
    sandbox: AsyncSandbox
    workdir: str


class OracleAgent(SimpleResponsesAPIAgent):
    config: OracleAgentConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("Process-local oracle sessions require num_workers=1")
        self._sessions: dict[str, OracleSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._closed: set[str] = set()

    # -- task lookup -----------------------------------------------------------------------

    def _task_folder(self, taskset: str, task_id: str) -> tuple[Path, str]:
        """The task folder and materialized digest from the resources server's taskset mapping."""
        resources_config = get_first_server_config_dict(get_global_config_dict(), self.config.resources_server.name)
        tasksets = resources_config.get("tasksets") or {}
        mapping = tasksets.get(taskset)
        if mapping is None:
            raise HTTPException(404, f"Taskset {taskset!r} is not in the resources server's taskset mapping")
        digest = (mapping.get("tasks") or {}).get(task_id)
        if digest is None:
            raise HTTPException(404, f"Task {task_id!r} is not in taskset {taskset!r}")
        return Path(mapping["folder"]) / task_id, str(digest)

    def _load_task(self, taskset: str, task_id: str) -> HarborTask:
        folder, digest = self._task_folder(taskset, task_id)
        try:
            task = load_task(folder)
        except HarborTaskError as exc:
            raise HTTPException(422, str(exc)) from exc
        if task.digest != digest:
            raise HTTPException(409, f"Task folder {folder} changed since materialization")
        return task

    # -- sessions --------------------------------------------------------------------------

    async def seed_agent_session(self, request: Request, body: AgentSeedSessionRequest) -> AgentSeedSessionResponse:
        session_id = body.agent_session_id
        request.session[_SESSION_KEY] = session_id
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            if session_id in self._closed:
                raise HTTPException(409, f"Agent session is already closed: {session_id}")
            existing = self._sessions.get(session_id)
            if existing is not None:
                if (existing.request.episode_id, existing.request.task_id) != (body.episode_id, body.task_id):
                    raise HTTPException(409, "agent_session_id is already bound to another episode or task")
                return AgentSeedSessionResponse(agent_session_id=session_id)
            if body.sandbox_access is None:
                raise HTTPException(422, "The oracle agent needs sandbox_access from the resources server")
            connection = body.sandbox_access.connection
            if not isinstance(connection, DirectSandboxConnection):
                raise HTTPException(422, "The oracle agent supports only direct sandbox connections")
            provider = create_provider(
                resolve_provider_config(connection.provider_config_ref, get_global_config_dict())
            )
            try:
                sandbox = await AsyncSandbox.connect(connection.descriptor, provider=provider)
            except BaseException:
                await provider.aclose()
                raise
            self._sessions[session_id] = OracleSession(
                request=body, sandbox=sandbox, workdir=body.sandbox_access.workdir
            )
            return AgentSeedSessionResponse(agent_session_id=session_id)

    async def close_agent_session(self, request: Request, body: AgentCloseSessionRequest) -> AgentCloseSessionResponse:
        session_id = body.agent_session_id
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            session = self._sessions.pop(session_id, None)
            if session is not None:
                if session.request.episode_id != body.episode_id:
                    self._sessions[session_id] = session
                    raise HTTPException(409, "episode_id does not match the seeded agent session")
                await session.sandbox.disconnect()
            self._closed.add(session_id)
            request.session.pop(_SESSION_KEY, None)
            return AgentCloseSessionResponse(agent_session_id=session_id)

    def _session_for(self, request: Request) -> OracleSession:
        session_id = request.session.get(_SESSION_KEY)
        session = self._sessions.get(session_id) if session_id else None
        if session is None:
            raise HTTPException(404, "Unknown oracle agent session; seed it first")
        return session

    # -- the "turn" ------------------------------------------------------------------------

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session = self._session_for(request)
        task_id = session.request.task_id
        task = self._load_task(task_id.taskset, task_id.task_id)
        if not task.has_solution:
            return self._response(body, ORACLE_STATUS_UNVALIDATED, "No solution/solve.sh in the task folder.", {})

        settings = task.config
        timeout = min(settings.agent.timeout_sec, self.config.max_solve_timeout_s)
        await upload_dir(session.sandbox, task.path / "solution", SOLUTION_DIR)
        result = await session.sandbox.exec(
            f"bash {SOLUTION_DIR}/solve.sh",
            cwd=session.workdir,
            env=dict(settings.solution.env),
            timeout_s=timeout,
            user=task.user,
        )
        tail = self.config.output_tail_chars
        output = "\n".join(part for part in ((result.stdout or "")[-tail:], (result.stderr or "")[-tail:]) if part)
        status = ORACLE_STATUS_SOLVED if result.return_code == 0 else ORACLE_STATUS_FAILED
        extra = {"oracle_return_code": str(result.return_code)}
        if result.error_type:
            extra["oracle_error_type"] = result.error_type
        return self._response(body, status, output or f"solve.sh exited with {result.return_code}", extra)

    @staticmethod
    def _response(
        body: NeMoGymResponseCreateParamsNonStreaming, status: str, text: str, extra: dict[str, str]
    ) -> NeMoGymResponse:
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model="oracle",
            object="response",
            status="completed",
            output=[
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(annotations=[], text=text)],
                )
            ],
            metadata={ORACLE_METADATA_KEY: status, **extra},
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=0,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=0,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=0,
            ),
        )

    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        raise HTTPException(501, "The oracle agent runs only through an environment server, not the legacy /run route")


def quote(value: str) -> str:
    return shlex.quote(value)


if __name__ == "__main__":
    OracleAgent.run_webserver()
