# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Oracle agent: run a Harbor task's reference solution instead of a model.

The oracle is an agent harness like any other. It borrows the sandbox the ``harbor``
environment server seeded, where that server placed the task's ``solution/`` at ``/solution``
because the agent it serves is this oracle, and runs ``bash /solution/solve.sh`` as the task
user within the ``[agent].timeout_sec`` the seed request carries. Verification then scores the
result the same way it scores a model's work, so ``gym dataset validate`` is an ordinary run
with this agent selected. The oracle never reads a task folder itself: the task, its digest and
its solution are the environment server's.

A task without ``solution/solve.sh`` is not failed: the oracle returns a response tagged
``oracle=unvalidated`` and does nothing in the sandbox.
"""

import logging
import math
from dataclasses import dataclass
from time import time
from typing import Any
from uuid import uuid4

from fastapi import Body, HTTPException, Request
from pydantic import Field

from environment_servers.harbor.session import SOLUTION_DIR
from environment_servers.harbor.verifier import TIMEOUT_WRAPPER_GUARD
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.global_config import get_global_config_dict
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
from nemo_gym.sandbox.transfer import SandboxTransferError, SandboxUnavailableError, exec_as_root


LOGGER = logging.getLogger(__name__)

SOLVE_SCRIPT = f"{SOLUTION_DIR}/solve.sh"
SOLVE_LOG_DIR = "/logs/agent"
# Harbor's oracle writes its output under this name, so tools written for Harbor find the log.
SOLVE_LOG = f"{SOLVE_LOG_DIR}/oracle.txt"
# Extra time the exec may wait after the in-container `timeout` should have fired.
SOLVE_GRACE_S = 60.0
ORACLE_METADATA_KEY = "oracle"
ORACLE_STATUS_SOLVED = "solved"
ORACLE_STATUS_UNVALIDATED = "unvalidated"
ORACLE_STATUS_FAILED = "failed"
# `oracle_error_type` when solve.sh never started: the image has no `timeout` or `bash`.
ORACLE_ERROR_WRAPPER = "wrapper"


class OracleAgentConfig(BaseResponsesAPIAgentConfig):
    # Like every other harness config, this one tolerates the keys the run config puts on an agent block
    # (`datasets`, carried by the generated per-taskset agent) without declaring them.

    # Wall-clock cap for solve.sh when the task's [agent].timeout_sec is larger.
    max_solve_timeout_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    output_tail_chars: int = 4000


@dataclass
class OracleSession(AgentSessionState):
    sandbox: AsyncSandbox
    workdir: str
    # The task user and the `[solution].env` the environment server resolved, from the sandbox access.
    user: str | int | None
    env: dict[str, str]
    # The solve.sh budget: the task's `[agent].timeout_sec` from the seed, under `max_solve_timeout_s`.
    timeout_s: float


class OracleAgent(SimpleResponsesAPIAgent):
    config: OracleAgentConfig
    ray_enabled = False

    def supports_task_tool_context(self) -> bool:
        """The reference solution runs inside the task's own environment, where its MCP servers and skills live.

        ``solve.sh`` is the task author's script, so it reaches the task's sidecars and skills directory by
        itself, as it does under Harbor's oracle. Declaring support keeps ``gym dataset validate`` running
        such tasks instead of refusing them before the solution starts.
        """
        return True

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("Process-local oracle sessions require num_workers=1")

    # -- sessions --------------------------------------------------------------------------

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> OracleSession:
        if body.sandbox_access is None:
            raise HTTPException(422, "The oracle agent needs sandbox_access from the environment server")
        if body.agent_timeout_sec is None:
            raise HTTPException(
                422,
                "The oracle agent needs agent_timeout_sec (the task's [agent].timeout_sec) from the environment server",
            )
        connection = body.sandbox_access.connection
        if not isinstance(connection, DirectSandboxConnection):
            raise HTTPException(422, "The oracle agent supports only direct sandbox connections")
        timeout = body.agent_timeout_sec
        if self.config.max_solve_timeout_s is not None:
            timeout = min(timeout, self.config.max_solve_timeout_s)
        provider = create_provider(resolve_provider_config(connection.provider_config_ref, get_global_config_dict()))
        try:
            sandbox = await AsyncSandbox.connect(connection.descriptor, provider=provider)
        except BaseException:
            await provider.aclose()
            raise
        return OracleSession(
            request=body,
            sandbox=sandbox,
            workdir=body.sandbox_access.workdir,
            user=body.sandbox_access.user,
            env=dict(body.sandbox_access.env),
            timeout_s=timeout,
        )

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
        # The environment server put the solution in place for this harness; a task that ships none left
        # nothing there. The probe runs as the task user, who must be able to reach the script.
        present = await session.sandbox.exec(f"test -f {SOLVE_SCRIPT}", cwd="/", timeout_s=60, user=session.user)
        if present.error_type:
            # The provider, not the task: the episode fails as retryable rather than scoring an absent solution.
            raise SandboxUnavailableError(
                f"The sandbox could not probe {SOLVE_SCRIPT}: {present.error_type}: {present.stderr or present.stdout or ''}"
            )
        if present.return_code != 0:
            return self._response(
                body,
                ORACLE_STATUS_UNVALIDATED,
                f"No solution/solve.sh in the task ({SOLVE_SCRIPT} is absent). Either the task ships no solution/, or the "
                "environment server did not recognise this agent as harbor_oracle_agent and placed none (its log says "
                "which).",
                {},
            )

        timeout = session.timeout_s
        env = {"DEBIAN_FRONTEND": "noninteractive", **session.env}
        try:
            # The folder may already exist root-owned and world-writable (Harbor's /logs mount, or a
            # derived image); only its writability matters.
            prepared = await exec_as_root(
                session.sandbox,
                f"mkdir -p {SOLVE_LOG_DIR} 2>/dev/null; chmod 777 {SOLVE_LOG_DIR} 2>/dev/null; test -w {SOLVE_LOG_DIR}",
                timeout_s=60,
            )
            if prepared.return_code:
                raise SandboxTransferError(f"{SOLVE_LOG_DIR} is not writable: {prepared.stderr or prepared.stdout}")
        except SandboxTransferError as exc:
            # The sandbox user cannot make the log folder solve.sh writes to (for example an image that runs
            # as a non-root user without the capability to switch to root). Report it as an oracle failure
            # so the row stays scored and the reason is visible. A provider failure (`SandboxUnavailableError`:
            # timeout, lost container) is not caught: it propagates and the episode fails as retryable
            # instead of being scored as a failed reference solution.
            return self._response(body, ORACLE_STATUS_FAILED, str(exc), {"oracle_error_type": "transfer"})
        # One bounded exec: the provider keeps the command alive for `timeout_s`, and `timeout`
        # inside the container stops the solution itself (a sub-second budget still gets a whole
        # second: `timeout 0` means no limit). Output goes to a file we tail. The trace names the task and the
        # episode the session was seeded for (as the environment server's does): concurrent reference solutions
        # all run the same script path, often with the same settings.
        task_id = session.request.task_id.task_id
        episode_key = session.request.episode_id.capture_key
        LOGGER.debug(
            "%s: episode %s: running %s as %s in %s with a %.0fs budget",
            task_id,
            episode_key,
            SOLVE_SCRIPT,
            session.user if session.user is not None else "the sandbox default user",
            session.workdir,
            timeout,
        )
        result = await session.sandbox.exec(
            f"{TIMEOUT_WRAPPER_GUARD}; "
            f"timeout -s KILL {max(1, math.ceil(timeout))} bash {SOLVE_SCRIPT} > {SOLVE_LOG} 2>&1",
            cwd=session.workdir,
            env=env,
            timeout_s=timeout + SOLVE_GRACE_S,
            user=session.user,
            preserve_background_services=True,
        )
        tail = self.config.output_tail_chars
        # `tail` fails when the log does not exist: the guard exited before the redirection created it.
        log = await session.sandbox.exec(f"tail -c {tail} {SOLVE_LOG} 2>/dev/null", cwd="/", timeout_s=60)
        output = "\n".join(part for part in ((log.stdout or "")[-tail:], (result.stderr or "")[-tail:]) if part)
        status = ORACLE_STATUS_SOLVED if result.return_code == 0 else ORACLE_STATUS_FAILED
        LOGGER.debug(
            "%s: episode %s: %s exited %s (%s%s)",
            task_id,
            episode_key,
            SOLVE_SCRIPT,
            result.return_code,
            status,
            f", {result.error_type}" if result.error_type else "",
        )
        extra = {"oracle_return_code": str(result.return_code)}
        if result.error_type:
            extra["oracle_error_type"] = result.error_type
        elif result.return_code == 127 and log.return_code != 0:
            extra["oracle_error_type"] = ORACLE_ERROR_WRAPPER
            output = output or "solve.sh could not be started: timeout wrapper or bash missing (exit 127)"
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


if __name__ == "__main__":
    OracleAgent.run_webserver()
