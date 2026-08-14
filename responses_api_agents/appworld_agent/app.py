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
"""Agent harness for the AppWorld resources server.

A plain tool-calling loop would nearly work here, but two AppWorld properties
need a dedicated harness:

* **The task text is not in the dataset row.** AppWorld's tasks are part of its
  encrypted, redistribution-restricted portion, so gym rows carry only a task id
  and ``/seed_session`` returns the system prompt plus the supervisor/instruction
  turn as observations, which this harness prepends to the rollout.
* **Termination is decided by the environment, not by the model.** An episode
  ends when the agent calls ``apis.supervisor.complete_task()`` inside the
  sandbox (or the interaction budget runs out) — a signal that arrives on the
  ``/step`` response, not as a stop from the model. A model turn with no tool
  call is a give-up and also ends the rollout.

``/close`` runs in a ``finally`` so the leased worker process is always returned
to the pool, and it is what triggers scoring on the resources server.
"""

import json
import logging
from typing import List

import aiohttp
from pydantic import ConfigDict, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
    accumulate_response_usage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.appworld.schemas import (
    AppWorldNeMoGymResponse,
    AppWorldSeedSessionResponse,
    AppWorldStepResponse,
    AppWorldVerifyRequest,
    AppWorldVerifyResponse,
)


logger = logging.getLogger(__name__)

EXECUTE_TOOL_NAME = "execute_ipython_code"


class AppWorldAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    max_steps: int = Field(
        default=50,
        description=(
            "Hard cap on agent turns. The resources server's `max_interactions` "
            "budget normally binds first; this only guards against a model that "
            "burns turns without producing tool calls."
        ),
    )


class AppWorldAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    # AppWorld task id from the dataset row, e.g. "82e2fac_1".
    task_id: str
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )


def _tool_output(call_id: str, output: str) -> NeMoGymFunctionCallOutput:
    """A tool result for the next model turn.

    ``type`` is passed explicitly and must stay that way: ``ServerClient.post``
    serialises the request with ``model_dump(exclude_unset=True)``, so a
    defaulted ``type`` is omitted from the wire payload and the provider then
    sees a tool call with no matching result (Bedrock rejects the turn outright,
    other providers silently lose the observation).
    """
    return NeMoGymFunctionCallOutput(type="function_call_output", call_id=call_id, output=output)


def _extract_code(function_call: NeMoGymResponseFunctionToolCall) -> tuple[str, str | None]:
    """``(code, error)`` from a tool call's JSON arguments.

    Malformed arguments are a model error, not a server error: the message is
    handed back as the tool output so the model can correct itself.
    """
    try:
        arguments = json.loads(function_call.arguments or "{}")
    except (json.JSONDecodeError, TypeError) as exc:
        return "", f"Invalid tool call arguments ({type(exc).__name__}: {exc}). Pass a JSON object with a 'code' key."
    if not isinstance(arguments, dict):
        return "", "Invalid tool call arguments: expected a JSON object with a 'code' key."
    code = arguments.get("code")
    if not isinstance(code, str):
        return "", "Missing required argument 'code' (a string of Python to execute)."
    return code, None


class AppWorldAgent(SimpleResponsesAPIAgent):
    config: AppWorldAgentConfig

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=5))
    async def _seed_session(self, task_id: str) -> AppWorldSeedSessionResponse:
        raw = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json={"task_id": task_id},
        )
        raw.raise_for_status()
        seed = AppWorldSeedSessionResponse.model_validate(await raw.json())
        if not seed.obs:
            raise ValueError(f"No observations in seed session response for task_id={task_id}")
        return seed

    async def responses(self, req: AppWorldAgentRunRequest) -> AppWorldNeMoGymResponse:
        req = req.model_copy(deep=True)
        body = req.responses_create_params
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        seed = await self._seed_session(req.task_id)
        agent_state = body.model_copy(update={"input": body.input + seed.obs, "tools": seed.tools})
        env_id = seed.env_id

        model_response: NeMoGymResponse | None = None
        all_messages: List[NeMoGymResponseOutputItem] = []
        usage = None
        model_server_cookies = None

        step = 0
        try:
            while step < self.config.max_steps:
                step += 1

                try:
                    raw = await self.server_client.post(
                        server_name=self.config.model_server.name,
                        url_path="/v1/responses",
                        json=agent_state,
                        cookies=model_server_cookies,
                    )
                    if not raw.ok:
                        # The provider's own message is the only useful signal
                        # when a multi-turn payload is rejected; without it a
                        # truncated rollout looks like the model giving up.
                        raise RuntimeError(f"status {raw.status}: {(await raw.content.read()).decode()[:2000]}")
                    model_server_cookies = raw.cookies
                    model_response_json = await raw.json()
                except (json.JSONDecodeError, aiohttp.ClientResponseError, RuntimeError) as exc:
                    logger.warning("Error calling /v1/responses at step %d: %s", step, exc)
                    break

                try:
                    model_response = NeMoGymResponse.model_validate(model_response_json)
                except ValidationError as exc:
                    logger.warning("Error validating model response: %r", exc)
                    break

                usage = accumulate_response_usage(usage, model_response.usage)
                model_response.usage = None

                model_output = model_response.output
                function_calls: List[NeMoGymResponseFunctionToolCall] = [
                    item for item in model_output if item.type == "function_call"
                ]
                all_messages.extend(model_output)

                if not function_calls:
                    # The model answered in prose instead of acting. AppWorld only
                    # scores database state, so there is nothing left to do.
                    logger.info("AppWorld rollout env=%s ended without a tool call at step %d", env_id, step)
                    break

                observations, done = await self._run_tool_calls(env_id, function_calls)
                all_messages.extend(observations)
                agent_state = agent_state.model_copy(update={"input": agent_state.input + model_output + observations})
                if done:
                    break
        finally:
            await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/close",
                json={"env_id": env_id},
            )

        assert model_response is not None, "Rollout terminated before the first agent turn completed."

        model_response.usage = usage
        return AppWorldNeMoGymResponse.model_validate(
            model_response.model_dump() | {"env_id": env_id, "task_id": req.task_id, "output": all_messages}
        )

    async def _run_tool_calls(
        self, env_id: str, function_calls: List[NeMoGymResponseFunctionToolCall]
    ) -> tuple[List[NeMoGymFunctionCallOutput], bool]:
        """Execute each requested call; return its outputs and whether we are done.

        Models occasionally emit several calls in one turn. AppWorld's shell is
        stateful and sequential, so they run in order and stop early once the
        environment reports the episode over.
        """
        observations: List[NeMoGymFunctionCallOutput] = []
        done = False
        for function_call in function_calls:
            call_id = function_call.call_id or "call"
            if done:
                observations.append(_tool_output(call_id, "The episode has already ended."))
                continue
            if function_call.name != EXECUTE_TOOL_NAME:
                observations.append(
                    _tool_output(
                        call_id,
                        f"Unknown tool {function_call.name!r}. The only available tool is {EXECUTE_TOOL_NAME}.",
                    )
                )
                continue
            code, error = _extract_code(function_call)
            if error is not None:
                observations.append(_tool_output(call_id, error))
                continue

            raw = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/step",
                json={"env_id": env_id, "code": code},
            )
            raw.raise_for_status()
            step_response = AppWorldStepResponse.model_validate(await raw.json())
            observations.append(_tool_output(call_id, step_response.output))
            done = done or step_response.done
        return observations, done

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Proxy to the resources server, which owns AppWorld's own metrics.

        Without this, gym aggregates on the agent side and Scenario Goal
        Completion — which needs every variant of a scenario at once — is
        silently missing from the report.
        """
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))

    async def run(self, body: AppWorldAgentRunRequest) -> AppWorldVerifyResponse:
        try:
            response = await self.responses(body)
            verify_request = AppWorldVerifyRequest.model_validate(body.model_dump() | {"response": response})
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
            )
            return AppWorldVerifyResponse.model_validate(await verify_response.json())
        except Exception:
            logger.exception("Error in run for task_id=%s", body.task_id)
            raise


if __name__ == "__main__":
    AppWorldAgent.run_webserver()
