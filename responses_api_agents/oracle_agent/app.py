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
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionState,
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
SOLVE_LOG_DIR = "/logs/agent"
SOLVE_LOG = f"{SOLVE_LOG_DIR}/solve-stdout.txt"
# Extra time the exec may wait after the in-container `timeout` should have fired.
SOLVE_GRACE_S = 60.0
ORACLE_METADATA_KEY = "oracle"
ORACLE_STATUS_SOLVED = "solved"
ORACLE_STATUS_UNVALIDATED = "unvalidated"
ORACLE_STATUS_FAILED = "failed"


class OracleAgentConfig(BaseResponsesAPIAgentConfig):
    model_config = ConfigDict(extra="forbid")

    # The `harbor` resources server whose `tasksets` mapping locates task folders.
    resources_server: ResourcesServerRef
    # Wall-clock cap for solve.sh when the task's [agent].timeout_sec is larger.
    max_solve_timeout_s: float = 3600.0
    output_tail_chars: int = 4000


@dataclass
class OracleSession(AgentSessionState):
    sandbox: AsyncSandbox
    workdir: str


class OracleAgent(SimpleResponsesAPIAgent):
    config: OracleAgentConfig
    ray_enabled = False

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("Process-local oracle sessions require num_workers=1")

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

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> OracleSession:
        if body.sandbox_access is None:
            raise HTTPException(422, "The oracle agent needs sandbox_access from the resources server")
        connection = body.sandbox_access.connection
        if not isinstance(connection, DirectSandboxConnection):
            raise HTTPException(422, "The oracle agent supports only direct sandbox connections")
        provider = create_provider(resolve_provider_config(connection.provider_config_ref, get_global_config_dict()))
        try:
            sandbox = await AsyncSandbox.connect(connection.descriptor, provider=provider)
        except BaseException:
            await provider.aclose()
            raise
        return OracleSession(request=body, sandbox=sandbox, workdir=body.sandbox_access.workdir)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, OracleSession):
            raise TypeError("Expected an oracle agent session state")
        await state.sandbox.disconnect()
        return AgentCloseSessionResponse(agent_session_id=state.request.agent_session_id)

    def _session_for(self, request: Request) -> OracleSession:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(404, "Unknown oracle agent session; seed it first")
        state = self._require_agent_session(session_id)
        if not isinstance(state, OracleSession):
            raise TypeError("Expected an oracle agent session state")
        return state

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
        await session.sandbox.exec(f"mkdir -p {SOLVE_LOG_DIR} && chmod 777 {SOLVE_LOG_DIR}", cwd="/", timeout_s=60)
        # One bounded exec: the provider keeps the command alive for `timeout_s`, and `timeout`
        # inside the container stops the solution itself. Output goes to a file we tail.
        result = await session.sandbox.exec(
            f"timeout --signal=KILL {int(timeout)} bash {SOLUTION_DIR}/solve.sh > {SOLVE_LOG} 2>&1",
            cwd=session.workdir,
            env=dict(settings.solution.env),
            timeout_s=timeout + SOLVE_GRACE_S,
            user=task.user,
        )
        tail = self.config.output_tail_chars
        log = await session.sandbox.exec(f"tail -c {tail} {SOLVE_LOG} 2>/dev/null || true", cwd="/", timeout_s=60)
        output = "\n".join(part for part in ((log.stdout or "")[-tail:], (result.stderr or "")[-tail:]) if part)
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
