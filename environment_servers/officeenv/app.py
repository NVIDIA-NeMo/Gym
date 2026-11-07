# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OfficeEnv's Environment Server: an OfficeEnv episode that ends when its task ends.

It runs the single-agent-turn protocol behind the flat ``/run`` contract, as ``single_agent_turn_legacy`` does: seed
the Resources Server and Agent Server sessions, activate the agent, close its session, verify, and clean up. With
``max_steps`` set it drives the agent one step at a time (one model call and the tool calls it makes) and reads the
episode's task state from the OfficeEnv Resources Server (``GET /task_state``) between steps, so the episode ends
before the next model call once the task has ended (``complete_task``), as it does when the model answers without
calling a tool, when a step does not complete, or after ``max_steps`` steps. Without ``max_steps`` the agent runs its
own loop in one activation.
"""

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field, PositiveInt

from environment_servers.single_agent_turn.app import SessionHandles, _cookies
from environment_servers.single_agent_turn_legacy.app import (
    SingleAgentTurnLegacyEnvironmentServer,
    SingleAgentTurnLegacyEnvironmentServerConfig,
)
from nemo_gym.agents import AGENT_ACTIVATION_HEADER
from nemo_gym.client import get_response_json, raise_boundary_failure
from nemo_gym.config import get_first_server_config_dict
from nemo_gym.core import NeMoGymEasyInputMessage, NeMoGymResponse, accumulate_response_usage
from nemo_gym.server_runtime import SingleAgentTurnRequest, dependency_failure
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


TASK_STATE_PATH = "/task_state"


class TaskState(BaseModel):
    """The Resources Server's ``GET /task_state`` reply."""

    task_ended: bool


class OfficeEnvEnvironmentServerConfig(SingleAgentTurnLegacyEnvironmentServerConfig):
    """Bind an agent and the OfficeEnv Resources Server, and say how the episode's steps are driven."""

    max_steps: PositiveInt | None = Field(
        default=None,
        description=(
            "The episode's step limit. Set, the agent takes one step per activation (the simple agent with "
            "max_steps: 1) and the episode ends at the first step after which the task has ended, the model called "
            "no tool or the step did not complete. Unset, the agent runs its own loop in one activation."
        ),
    )


class OfficeEnvEnvironmentServer(SingleAgentTurnLegacyEnvironmentServer):
    """Run an OfficeEnv episode, step by step when ``max_steps`` is set, and end it when its task ends."""

    config: OfficeEnvEnvironmentServerConfig
    # The step loop records no checkpoint boundary between activations, so a crashed episode starts over.
    checkpoint_mode = "restart_only"

    def setup_webserver(self) -> FastAPI:
        self._require_one_step_agent()
        return super().setup_webserver()

    def _require_one_step_agent(self) -> None:
        """Refuse to start when one activation could take several steps: the task state is read between activations,
        so a step taken after the task ended in the same activation would count."""
        if self.config.max_steps is None:
            return
        agent = self.config.agent_server.name
        agent_steps = get_first_server_config_dict(self.server_client.global_config_dict, agent).get("max_steps")
        if agent_steps != 1:
            raise ValueError(
                f"{self.config.name} drives {agent} one step per activation (max_steps: {self.config.max_steps}), "
                f"so {agent} must take one step per activation (max_steps: 1), not {agent_steps}"
            )

    async def _invoke_agent(self, request: SingleAgentTurnRequest, handles: SessionHandles) -> NeMoGymResponse:
        """The agent's turn: its own loop in one activation, or one step per activation up to ``max_steps``, ending
        early as the agent's own loop would or when the task has ended. The reply carries every step's output and
        their summed usage, as one activation's would."""
        if self.config.max_steps is None:
            return await super()._invoke_agent(request, handles)
        params = request.task.task_input.responses_create_params
        if isinstance(params.input, str):
            conversation: list[Any] = (
                [NeMoGymEasyInputMessage(role="user", content=params.input)] if params.input else []
            )
        else:
            conversation = list(params.input)
        if handles.observation:
            conversation.append(NeMoGymEasyInputMessage(role="user", content=handles.observation))
        outputs: list[Any] = []
        usage = None
        for step in range(1, self.config.max_steps + 1):
            response = await self._activate(
                request, handles, params.model_copy(update={"input": [*conversation, *outputs]}), step
            )
            outputs.extend(response.output)
            usage = accumulate_response_usage(usage, response.usage)
            if response.status != "completed" or not any(item.type == "function_call" for item in response.output):
                break
            if await self._task_ended(handles):
                break
        return response.model_copy(update={"output": outputs, "usage": usage})

    async def _activate(
        self, request: SingleAgentTurnRequest, handles: SessionHandles, params: Any, index: int
    ) -> NeMoGymResponse:
        """One activation of the agent session: one step of the episode."""
        try:
            agent_http_response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path=self._agent_responses_path(request),
                json=params,
                cookies=handles.agent_cookies,
                headers={AGENT_ACTIVATION_HEADER: str(index)},
            )
            await raise_boundary_failure(agent_http_response)
            response_cookies = _cookies(agent_http_response)
            if response_cookies:
                handles.agent_cookies = response_cookies
            return NeMoGymResponse.model_validate(await get_response_json(agent_http_response))
        except Exception as error:
            raise dependency_failure(error, stage="agent") from error

    async def _task_ended(self, handles: SessionHandles) -> bool:
        """Whether the episode's task has ended, as its Resources Server session reports it."""
        try:
            state_http_response = await self.server_client.get(
                server_name=self.config.resources_server.name,
                url_path=TASK_STATE_PATH,
                cookies=handles.resources_cookies,
            )
            await raise_boundary_failure(state_http_response)
            return TaskState.model_validate(await get_response_json(state_http_response)).task_ended
        except Exception as error:
            raise dependency_failure(error, stage="agent") from error


if __name__ == "__main__":
    OfficeEnvEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OfficeEnvEnvironmentServer.run_webserver()  # noqa: F401
