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
"""REFERENCE ONLY — not a runnable server, not imported by anything.

A minimal agent server that writes its own tool-calling loop, annotated with the correctness rules
from ../../references/correctness-checklist.md. Copy the shape, not the file.

Use this shape when you control the loop. If an external framework owns the model-calling loop, see
framework_bridge_reference.py instead.

The real, complete implementation is responses_api_agents/simple_agent/app.py — it additionally handles
usage accumulation, incomplete_details, max_steps, and TrajectoryRecord collection, all of which are
elided here to keep the four load-bearing rules visible.
"""

import json

from fastapi import Body, Request, Response

from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


class ReferenceAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = 10


class ReferenceAgent(SimpleResponsesAPIAgent):
    config: ReferenceAgentConfig

    async def responses(
        self,
        request: Request,
        # RULE 2a: `response` is required to propagate cookies. An agent that omits this parameter
        # cannot call set_cookie() at all, and downstream /verify silently gets the wrong session.
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        # RULE 1: derive the model path from the inbound request. Three URLs route to this one method
        # (/v1/responses, /ng-rollout/{id}/v1/responses, and the .../training-token-capture/ variant).
        # Hardcoding "/v1/responses" here silently drops training-token capture — no error, just
        # training data with no token IDs. Never hand-build this with rollout_path_prefix(): its
        # token_capture parameter defaults to False, which is the same bug.
        model_url_path = self.url_path_for_request("/v1/responses", request)

        # RULE 2b: two separate cookie jars, each reassigned from the response it came from. Gym's
        # aiohttp client uses DummyCookieJar() — it stores and resends nothing — and every server has a
        # distinct cookie name, so nothing rides along implicitly. One shared jar means model-server and
        # resources-server sessions overwrite each other.
        model_server_cookies = None
        resources_server_cookies = request.cookies

        # RULE 4: accumulate every output item — tool calls AND tool results AND messages — not just the
        # final answer. NeMoGymResponse.output is a list; collapsing it to the last assistant message is
        # permanent, unrecoverable loss of the agent trace for every rollout.
        new_outputs = []

        for _step in range(self.config.max_steps):
            new_body = body.model_copy(update={"input": body.input + new_outputs})

            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=model_url_path,
                json=new_body,
                cookies=model_server_cookies,
            )
            await raise_for_status(model_response)
            model_server_cookies = model_response.cookies
            model_response = NeMoGymResponse.model_validate(await get_response_json(model_response))

            output = model_response.output
            new_outputs.extend(output)

            function_calls = [o for o in output if o.type == "function_call"]
            messages = [o for o in output if o.type == "message" and o.role == "assistant"]
            if not function_calls and messages:
                break

            for call in function_calls:
                try:
                    arguments = json.loads(call.arguments)
                except (json.JSONDecodeError, TypeError) as e:
                    # Graceful degradation: a malformed tool call becomes a model-visible error, not a
                    # crashed request. A crash scores zero and loses the rollout; an error result lets
                    # the model recover on the next turn.
                    tool_output = json.dumps({"error": f"Invalid tool call arguments: {e!r}"})
                else:
                    # Resources-server errors are valid model-visible tool outputs — do not raise here.
                    api_response = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path=f"/{call.name}",
                        json=arguments,
                        cookies=resources_server_cookies,
                    )
                    tool_output = (await api_response.content.read()).decode()
                    resources_server_cookies = api_response.cookies

                new_outputs.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=call.call_id,
                        output=tool_output,
                    )
                )

        model_response.output = new_outputs

        # RULE 2c: mirror both jars onto the outgoing response so downstream /verify sees the right
        # sessions. If you deliberately exclude a jar (remote_agent keeps the remote service's private
        # cookies out), say so in a comment — silence reads as an oversight.
        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)

        return model_response

    # Most agents subclass SimpleAgent and inherit its run() (seed_session -> self-POST /v1/responses ->
    # verify) rather than writing one. If you do write it, use url_path_for_run("/v1/responses", body)
    # for the self-POST — not url_path_for_request — because run() has no inbound path to inherit; it is
    # originating the call, so the prefix comes from the body and config instead.


if __name__ == "__main__":
    ReferenceAgent.run_webserver()
