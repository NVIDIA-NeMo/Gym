# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging
from typing import Any, Dict, Optional

from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json, raise_for_status

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class CubeAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    max_steps: Optional[int] = Field(
        default=None,
        description="Maximum tool-call steps per episode. None = unlimited.",
    )
    done_if_no_tool_calls: bool = Field(
        default=True,
        description="If True, end rollout when model produces a message but no tool calls.",
    )
    max_total_sequence_length: Optional[int] = Field(
        default=None,
        description="If set, stop when agent_state input exceeds this estimated token count.",
    )


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class CubeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_id: Optional[str] = Field(
        default=None,
        description="Specific CUBE task ID to run. If None, the resources server selects one.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional random seed forwarded to the resources server.",
    )


class CubeAgentVerifyRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    response: NeMoGymResponse


class CubeAgentVerifyResponse(BaseVerifyResponse):
    """Verify response returned by CubeAgent.run().

    Inherits directly from BaseVerifyResponse (which already includes
    responses_create_params, response, and reward) to avoid Pydantic v2
    diamond-inheritance issues with multiple paths to BaseRunRequest.
    """

    model_config = ConfigDict(extra="allow")

    reward_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra reward information from the CUBE task (EnvironmentOutput.info).",
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CubeAgent(SimpleResponsesAPIAgent):
    """Multi-turn agent that drives a CUBE task through the CubeResourcesServer.

    Episode lifecycle per call to run():
      1. POST /seed_session  →  initial observation + tool definitions
      2. Loop:
           a. POST /v1/responses (model)  →  function_calls and/or messages
           b. For each function_call: POST /step  →  observation + done flag
           c. Branch on content_type:
              - "image/png"  →  FunctionCallOutput("Screenshot captured.") +
                                EasyInputMessage(image_url block)
              - "text/plain" →  FunctionCallOutput(output=step_response.output)
           d. Break if done=True, no tool calls (with done_if_no_tool_calls=True),
              or max_steps reached.
      3. POST /verify  →  reward + reward_info
      4. POST /close  (always, even on exception)
    """

    config: CubeAgentConfig

    async def responses(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
    ) -> NeMoGymResponse:
        """Single-turn model call — delegates directly to the model server."""
        raw = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body.model_dump(mode="json"),
        )
        await raise_for_status(raw)
        return NeMoGymResponse.model_validate(await get_response_json(raw))

    async def run(self, body: CubeAgentRunRequest) -> CubeAgentVerifyResponse:  # type: ignore[override]
        """Full episode loop.

        The CubeResourcesServer schemas are imported inside the function body
        so that a missing resources_servers package on sys.path does not break
        the module-level import of this agent.
        """
        import json as _json

        from resources_servers.cube_standard.schemas import (
            CubeSeedSessionRequest,
            CubeSeedSessionResponse,
            CubeStepRequest,
            CubeStepResponse,
        )

        resources_server_cookies = None
        model_server_cookies = None
        model_response: Optional[NeMoGymResponse] = None

        # ------------------------------------------------------------------
        # Step 1: seed_session — create a new CUBE task session
        # ------------------------------------------------------------------
        seed_req = CubeSeedSessionRequest(task_id=body.task_id, seed=body.seed)
        raw_seed = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=seed_req.model_dump(mode="json"),
        )
        await raise_for_status(raw_seed)
        resources_server_cookies = raw_seed.cookies
        seed_response = CubeSeedSessionResponse.model_validate(await get_response_json(raw_seed))

        # ------------------------------------------------------------------
        # Step 2: build initial agent state — inject obs + tool definitions
        # ------------------------------------------------------------------
        agent_state = body.responses_create_params.model_copy(
            update={
                "input": list(body.responses_create_params.input) + seed_response.obs,
                "tools": seed_response.tools,
            }
        )

        step = 0
        done = False

        try:
            while True:
                # Step limit guard
                if self.config.max_steps is not None and step >= self.config.max_steps:
                    break
                step += 1

                # --------------------------------------------------------------
                # Step 3a: call model
                # --------------------------------------------------------------
                raw_model = await self.server_client.post(
                    server_name=self.config.model_server.name,
                    url_path="/v1/responses",
                    json=agent_state.model_dump(mode="json"),
                    cookies=model_server_cookies,
                )
                await raise_for_status(raw_model)
                model_server_cookies = raw_model.cookies
                model_response = NeMoGymResponse.model_validate(await get_response_json(raw_model))

                model_output = model_response.output
                all_fn_calls = [o for o in model_output if o.type == "function_call"]
                all_output_messages = [
                    o for o in model_output if o.type == "message" and getattr(o, "role", None) == "assistant"
                ]

                # No tool calls — model produced a plain message
                if not all_fn_calls and all_output_messages:
                    if self.config.done_if_no_tool_calls:
                        done = True
                    agent_state = agent_state.model_copy(
                        update={"input": list(agent_state.input) + list(model_output)}
                    )
                    break

                # --------------------------------------------------------------
                # Step 3b/c: append model output to context, then execute each
                # function call and inject the resulting observation.
                # --------------------------------------------------------------
                new_inputs = list(agent_state.input) + list(model_output)

                for fn_call in all_fn_calls:
                    step_req = CubeStepRequest(
                        call_id=fn_call.call_id,
                        name=fn_call.name,
                        arguments=_json.loads(fn_call.arguments),
                    )
                    raw_step = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/step",
                        json=step_req.model_dump(mode="json"),
                        cookies=resources_server_cookies,
                    )
                    await raise_for_status(raw_step)
                    resources_server_cookies = raw_step.cookies
                    step_response = CubeStepResponse.model_validate(await get_response_json(raw_step))

                    # Branch on content_type — image needs two items; text needs one
                    if step_response.content_type == "image/png":
                        # Protocol requirement: always acknowledge the tool call with text first.
                        new_inputs.append(
                            NeMoGymFunctionCallOutput(
                                call_id=fn_call.call_id,
                                output="Screenshot captured.",
                            )
                        )
                        # The model server fetches the image directly from the cube server's
                        # /screenshots/ endpoint — no base64 in this JSON payload.
                        # NeMo Gym / OpenAI Responses API image format:
                        #   type="input_image", image_url=<str>, detail="auto"
                        new_inputs.append(
                            NeMoGymEasyInputMessage(
                                role="user",
                                content=[
                                    {
                                        "type": "input_image",
                                        "image_url": step_response.output,
                                        "detail": "auto",
                                    }
                                ],
                            )
                        )
                    else:
                        # Text observation: standard function_call_output
                        new_inputs.append(
                            NeMoGymFunctionCallOutput(
                                call_id=fn_call.call_id,
                                output=step_response.output,
                            )
                        )

                    if step_response.done:
                        done = True
                        break

                agent_state = agent_state.model_copy(update={"input": new_inputs})

                if done:
                    break

            # ------------------------------------------------------------------
            # Step 4: verify — retrieve reward from the resources server.
            # MUST be inside the try block so it runs before finally(/close).
            # The session is still alive here; /close (in finally) tears it down.
            # ------------------------------------------------------------------
            if model_response is None:
                raise RuntimeError("Rollout terminated before first model response.")

            verify_req = CubeAgentVerifyRequest.model_validate(
                body.model_dump() | {"response": model_response.model_dump(mode="json")}
            )
            raw_verify = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_req.model_dump(mode="json"),
                cookies=resources_server_cookies,
            )
            await raise_for_status(raw_verify)
            return CubeAgentVerifyResponse.model_validate(await get_response_json(raw_verify))

        finally:
            # Always close — even on exception.  Errors here are non-fatal.
            # When the try block returns normally, Python executes this finally
            # before the return value is propagated — so the order is:
            #   verify (inside try) → close (finally) → return to caller.
            try:
                await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/close",
                    json={},
                    cookies=resources_server_cookies,
                )
            except Exception as exc:
                logger.warning("Error calling /close: %s", exc)


if __name__ == "__main__":
    CubeAgent.run_webserver()
