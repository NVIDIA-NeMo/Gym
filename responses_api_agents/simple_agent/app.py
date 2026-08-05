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
import json
from typing import List

from fastapi import Request, Response
from pydantic import ConfigDict, Field, ValidationError

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.context_compaction import (
    ContextCompactedResponse,
    ContextCompactedTransportResponse,
    ContextCompactionSession,
    PreparedContextCompactionCall,
    build_generation_contract,
    build_transport_response,
)
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, raise_for_status
from nemo_gym.visual_history import (
    ContextMeasurements,
    VisualHistoryConfig,
)


_CONTEXT_COMPACTION_SEED_COUNT_COOKIE = "_nemo_gym_cc_seed_obs_count"
_CONTEXT_COMPACTION_ROLLOUT_ID_COOKIE = "_nemo_gym_cc_rollout_id"


class SimpleAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = None
    visual_history: VisualHistoryConfig = Field(default_factory=VisualHistoryConfig)


class SimpleAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    context_compaction_rollout_id: str | None = None
    context_compaction_group_id: str | None = None
    context_compaction_task_id: str | None = None
    context_compaction_rollout_index: int | None = Field(default=None, ge=0)
    context_compaction_attempt_index: int | None = Field(default=None, ge=0)


class SimpleAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    response: ContextCompactedTransportResponse | ContextCompactedResponse | NeMoGymResponse


class SimpleAgent(SimpleResponsesAPIAgent):
    config: SimpleAgentConfig

    async def _tool_response_items(
        self,
        api_response,
        call_id: str,
    ) -> list[NeMoGymFunctionCallOutput | NeMoGymEasyInputMessage]:
        """Decode one resources-server response into trajectory observations."""

        return [
            NeMoGymFunctionCallOutput(
                type="function_call_output",
                call_id=call_id,
                output=(await api_response.content.read()).decode(),
            )
        ]

    async def _seed_session_response_messages(
        self,
        seed_session_response,
    ) -> list[NeMoGymEasyInputMessage]:
        """Decode optional opening observations from a seed response."""

        return []

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> ContextCompactedResponse | NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        complete_input = list(body.input)
        seed_count_raw = request.cookies.get(
            _CONTEXT_COMPACTION_SEED_COUNT_COOKIE,
            "0",
        )
        try:
            seed_count = int(seed_count_raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Invalid internal context-compaction seed count") from exc
        if seed_count < 0 or seed_count > len(complete_input):
            raise RuntimeError("Internal context-compaction seed count is out of range")
        if seed_count:
            agent_input = complete_input[:-seed_count]
            seed_observations = complete_input[-seed_count:]
        else:
            agent_input = complete_input
            seed_observations = []
        context_session = None
        if self.config.visual_history.enabled:
            rollout_id = request.cookies.get(_CONTEXT_COMPACTION_ROLLOUT_ID_COOKIE) or str(
                request.session.get(SESSION_ID_KEY, "request")
            )
            context_session = ContextCompactionSession(
                config=self.config.visual_history,
                rollout_id=rollout_id,
                generation_contract=build_generation_contract(
                    body=body,
                    model_server=self.config.model_server,
                    visual_history=self.config.visual_history,
                ),
                initial_context=agent_input,
                seed_observations=seed_observations,
            )

        new_outputs = []
        usage = None
        step = 0
        model_server_cookies = None  # update the cookies on every model response
        resources_server_cookies = {
            key: value for key, value in request.cookies.items() if key != _CONTEXT_COMPACTION_SEED_COUNT_COOKIE
        }  # update the cookies on every resources server response

        while True:
            step += 1
            legacy_request_input = body.input + new_outputs
            request_input = legacy_request_input
            prepared_call = None
            if context_session is not None:

                async def measure_context(
                    call: PreparedContextCompactionCall,
                ) -> ContextMeasurements:
                    nonlocal model_server_cookies
                    prompt_token_count = 0
                    guard_config = self.config.visual_history.guards
                    if guard_config.max_total_tokens is not None:
                        tokenize_body = body.model_copy(
                            update={
                                "input": list(call.request_input),
                                "required_prefix_token_ids": (
                                    list(call.required_prefix_token_ids)
                                    if call.required_prefix_token_ids is not None
                                    else None
                                ),
                            }
                        )
                        tokenize_response = await self.server_client.post(
                            server_name=self.config.model_server.name,
                            url_path="/tokenize",
                            json=tokenize_body,
                            cookies=model_server_cookies,
                        )
                        await raise_for_status(tokenize_response)
                        tokenize_payload = await get_response_json(tokenize_response)
                        tokens = tokenize_payload.get("tokens")
                        if not isinstance(tokens, list) or not all(isinstance(token_id, int) for token_id in tokens):
                            raise RuntimeError("Model tokenize preflight returned invalid tokens")
                        prompt_token_count = len(tokens)
                        if tokenize_response.cookies:
                            model_server_cookies = tokenize_response.cookies

                    active_image_count = len(call.prepared_history.view.media_ids)
                    vision_tokens_per_image = guard_config.projected_vision_tokens_per_image or 0
                    return ContextMeasurements(
                        prompt_token_count=prompt_token_count,
                        active_image_count=active_image_count,
                        vision_token_count=(active_image_count * vision_tokens_per_image),
                    )

                prepared_call = await context_session.prepare_model_call(
                    legacy_request_input=legacy_request_input,
                    turn_id=step,
                    measure_context=measure_context,
                )
                request_input = list(prepared_call.request_input)
            new_body = body.model_copy(
                update={
                    "input": request_input,
                    "required_prefix_token_ids": (
                        list(prepared_call.required_prefix_token_ids)
                        if prepared_call is not None and prepared_call.required_prefix_token_ids is not None
                        else None
                    ),
                }
            )

            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=self.url_path_for_request("/v1/responses", request),
                json=new_body,
                cookies=model_server_cookies,
            )
            # We raise for status here since we expect model calls to always work.
            await raise_for_status(model_response)
            model_response_json = await get_response_json(model_response)
            model_server_cookies = model_response.cookies
            try:
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                ) from e

            output = model_response.output
            if context_session is not None:
                assert prepared_call is not None
                context_session.record_model_response(
                    call=prepared_call,
                    output_items=output,
                    finish_reason=(
                        model_response.incomplete_details.reason
                        if model_response.incomplete_details is not None
                        else None
                    ),
                )
            new_outputs.extend(output)

            if not usage:
                usage = model_response.usage
                model_response.usage = None

            if usage and model_response.usage:
                usage.input_tokens += model_response.usage.input_tokens
                usage.output_tokens += model_response.usage.output_tokens
                usage.total_tokens += model_response.usage.total_tokens

                # TODO support more advanced token details
                usage.input_tokens_details.cached_tokens = 0
                usage.output_tokens_details.reasoning_tokens = 0

            if model_response.incomplete_details:
                break

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]
            if not all_fn_calls and all_output_messages:
                break

            for output_function_call in all_fn_calls:
                try:
                    parsed_arguments = json.loads(output_function_call.arguments)
                except (json.JSONDecodeError, TypeError) as e:
                    # Model produced malformed tool-call arguments. Surface the
                    # error back as a tool response so the rollout can continue
                    # (or terminate with a low reward) instead of crashing the
                    # whole batch on json.loads.
                    tool_response = NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=output_function_call.call_id,
                        # Use repr(e) so the exception type name is always
                        # included even when str(e) would be empty.
                        output=json.dumps({"error": f"Invalid tool call arguments: {e!r}"}),
                    )
                    new_outputs.append(tool_response)
                    if context_session is not None:
                        context_session.append_observation(
                            [tool_response],
                            turn_id=step,
                            conditions_action_turn=step + 1,
                        )
                    continue

                api_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{output_function_call.name}",
                    json=parsed_arguments,
                    cookies=resources_server_cookies,
                )
                # We don't raise for status here since it's a valid return for the API to error e.g. if the model outputs an invalid call or something.
                resources_server_cookies = api_response.cookies

                try:
                    tool_responses = await self._tool_response_items(
                        api_response,
                        output_function_call.call_id,
                    )
                except ValidationError as exc:
                    tool_responses = [
                        NeMoGymFunctionCallOutput(
                            type="function_call_output",
                            call_id=output_function_call.call_id,
                            output=json.dumps({"error": f"Invalid tool envelope: {exc!r}"}),
                        )
                    ]
                new_outputs.extend(tool_responses)
                if context_session is not None:
                    context_session.append_observation(
                        tool_responses,
                        turn_id=step,
                        conditions_action_turn=step + 1,
                    )

            # Check if max steps is not None and if we have exhausted it.
            if self.config.max_steps and step >= self.config.max_steps:
                break

        # Propogate any extra cookies necessary for downstream verification
        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)

        if context_session is not None:
            context_session.finalize()
        model_response.output = new_outputs
        model_response.usage = usage
        if context_session is not None and context_session.authority_mode:
            return context_session.build_response(
                model_response,
                output=new_outputs,
                agent_input=agent_input,
                seed_obs=seed_observations,
            )
        return model_response

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        cookies = dict(request.cookies)

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies.update(seed_session_response.cookies)
        if body.context_compaction_rollout_id is not None:
            # The Starlette session cookie is signed independently by each
            # server and cannot carry a caller-owned logical rollout ID into
            # this agent's internal /v1/responses request. Use a dedicated
            # private transport cookie instead; server session identity and
            # logical rollout identity intentionally remain separate.
            cookies[_CONTEXT_COMPACTION_ROLLOUT_ID_COOKIE] = body.context_compaction_rollout_id
        seed_messages = await self._seed_session_response_messages(seed_session_response)
        responses_body = body.responses_create_params
        if seed_messages:
            responses_body = responses_body.model_copy(deep=True)
            if isinstance(responses_body.input, str):
                responses_body.input = [
                    NeMoGymEasyInputMessage(
                        role="user",
                        content=responses_body.input,
                    )
                ]
            responses_body.input = [
                *responses_body.input,
                *seed_messages,
            ]
            cookies[_CONTEXT_COMPACTION_SEED_COUNT_COOKIE] = str(len(seed_messages))

        response = await self.server_client.post(
            server_name=self.config.name,
            url_path=self.url_path_for_run("/v1/responses", body),
            json=responses_body,
            cookies=cookies,
        )
        await raise_for_status(response)
        cookies = response.cookies
        agent_response_payload = await get_response_json(response)
        context_compacted_response = None
        if agent_response_payload.get("context_compaction_contract") is not None:
            context_compacted_response = ContextCompactedResponse.model_validate(agent_response_payload)
            original_input = body.responses_create_params.input
            if isinstance(original_input, str):
                original_input = [
                    NeMoGymEasyInputMessage(
                        role="user",
                        content=original_input,
                    )
                ]
            context_compacted_response = context_compacted_response.model_copy(
                update={
                    "agent_input": list(original_input),
                    "seed_obs": seed_messages,
                }
            )

        verifier_response_payload = agent_response_payload
        if seed_messages:
            verifier_response_payload = dict(agent_response_payload)
            verifier_response_payload["output"] = [
                *(message.model_dump() for message in seed_messages),
                *agent_response_payload.get("output", []),
            ]
        verify_request = SimpleAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": verifier_response_payload}
        )

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        verified = SimpleAgentVerifyResponse.model_validate(await get_response_json(verify_response))
        if context_compacted_response is None:
            return verified

        contract = context_compacted_response.context_compaction_contract
        context_compacted_response = context_compacted_response.model_copy(
            update={
                "context_compaction_contract": contract.model_copy(
                    update={
                        "group_id": body.context_compaction_group_id,
                        "task_id": body.context_compaction_task_id,
                        "rollout_index": body.context_compaction_rollout_index,
                        "attempt_index": body.context_compaction_attempt_index,
                    }
                )
            }
        )
        return verified.model_copy(update={"response": build_transport_response(context_compacted_response)})

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Proxy aggregate_metrics to the resources server."""
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    SimpleAgent.run_webserver()
