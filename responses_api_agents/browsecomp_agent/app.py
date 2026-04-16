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
import json
import re
from typing import List

from fastapi import Request, Response
from pydantic import ConfigDict, ValidationError

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
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


class BrowsecompAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = 400
    keep_rounds: int = 9999
    nudge_steps: bool = True
    max_context_tokens: int = 196608
    context_reset_pct: float = 0.3
    context_reset_keep_rounds: int = 3
    max_run_retries: int = 1


class BrowsecompAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class BrowsecompAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class BrowsecompAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class BrowsecompAgent(SimpleResponsesAPIAgent):
    config: BrowsecompAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        new_outputs = []
        usage = None
        step = 0
        model_server_cookies = None  # update the cookies on every model response
        resources_server_cookies = request.cookies  # update the cookies on every resources server response

        reset_threshold = 0
        if self.config.max_context_tokens and self.config.context_reset_pct:
            reset_threshold = int(self.config.max_context_tokens * self.config.context_reset_pct)

        while True:
            step += 1

            if self.config.keep_rounds is not None and new_outputs:
                new_outputs = self._compact_old_tool_messages(new_outputs)

            new_body = body.model_copy(update={"input": body.input + new_outputs})

            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
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

            # --- Check context reset threshold ---
            prompt_tokens = model_response.usage.input_tokens if model_response.usage else 0
            if reset_threshold and prompt_tokens > reset_threshold:
                if self.config.context_reset_keep_rounds > 0:
                    new_outputs = self._extract_last_rounds(new_outputs)
                else:
                    new_outputs = []
                continue

            output = model_response.output
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

            # --- If the model decided to answer (no tool calls), we are done ---
            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]
            if not all_fn_calls and all_output_messages:
                break

            # --- Execute tool calls ---
            for output_function_call in all_fn_calls:
                api_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{output_function_call.name}",
                    json=json.loads(output_function_call.arguments),
                    cookies=resources_server_cookies,
                )
                # We don't raise for status here since it's a valid return for the API to error e.g. if the model outputs an invalid call or something.
                resources_server_cookies = api_response.cookies

                tool_output = (await api_response.content.read()).decode()
                if self.config.nudge_steps:
                    turns_left = self.config.max_steps - step
                    tool_output += "\n\n[%d turns remaining out of %d]" % (turns_left, self.config.max_steps)

                tool_response = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=tool_output,
                )
                new_outputs.append(tool_response)

            # --- Nudge the model at milestone steps ---
            if self.config.nudge_steps and all_fn_calls:
                quarter = self.config.max_steps // 4
                half = self.config.max_steps // 2
                near_end = int(self.config.max_steps * 0.875)
                nudge_msg = None
                if step == quarter:
                    nudge_msg = (
                        "\n\n\n\n\n"
                        "[SYSTEM NOTE: You have used %d out of %d turns. "
                        "Please consider consolidating your findings and "
                        "delivering an answer soon.]" % (step, self.config.max_steps)
                    )
                elif step == half:
                    nudge_msg = (
                        "\n\n\n\n\n"
                        "[SYSTEM NOTE: You have used %d out of %d turns — "
                        "you are halfway through your budget. You should start "
                        "formulating your final answer based on the research "
                        "you have already done. Do not keep searching endlessly.]"
                        % (step, self.config.max_steps)
                    )
                elif step == near_end:
                    nudge_msg = (
                        "\n\n\n\n\n"
                        "[SYSTEM NOTE: URGENT — You have used %d out of %d turns. "
                        "You are almost out of turns. YOU MUST deliver your final "
                        "answer NOW using the information you have already gathered. "
                        "Do NOT make any more tool calls. Provide your best answer "
                        "immediately in the required format with 'Exact Answer:' on "
                        "a line by itself.]" % (step, self.config.max_steps)
                    )

                if nudge_msg:
                    last_tool = new_outputs[-1]
                    new_output = last_tool.output + nudge_msg
                    new_outputs[-1] = last_tool.model_copy(update={"output": new_output})

            # Check if max steps is not None and if we have exhausted it.
            if self.config.max_steps and step >= self.config.max_steps:
                break

        # Propogate any extra cookies necessary for downstream verification
        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)

        model_response.output = new_outputs
        model_response.usage = usage
        return model_response

    async def run(self, request: Request, body: BrowsecompAgentRunRequest) -> BrowsecompAgentVerifyResponse:
        cookies = request.cookies

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = seed_session_response.cookies

        last_verify_response = None
        for _ in range(self.config.max_run_retries):
            response = await self.server_client.post(
                server_name=self.config.name,
                url_path="/v1/responses",
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(response)
            cookies = response.cookies

            # Retry if the model only produced <think> content with no final answer.
            response_json = await get_response_json(response)
            raw_output_text = NeMoGymResponse.model_validate(response_json).output_text
            cleaned_output_text = re.sub(r"<think>.*?</think>", "", raw_output_text, flags=re.DOTALL).strip()
            if not cleaned_output_text:
                continue

            verify_request = BrowsecompAgentVerifyRequest.model_validate(
                body.model_dump() | {"response": await get_response_json(response)}
            )

            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(verify_response)

            last_verify_response = BrowsecompAgentVerifyResponse.model_validate(
                await get_response_json(verify_response)
            )
            break

        return last_verify_response

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Proxy aggregate_metrics to the resources server."""
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))

    def _compact_old_tool_messages(self, messages):
        """
        Replace old tool-call results with a placeholder, keeping only the most
        recent *keep_rounds* tool messages.  This is the key context-management
        trick that enables long agent trajectories within a finite context window.
        """
        tool_indices = [i for i, m in enumerate(messages) if m.type == "function_call_output"]
        if len(tool_indices) <= self.config.keep_rounds:
            return messages

        for i in range(len(tool_indices) - self.config.keep_rounds):
            idx = tool_indices[i]
            messages[idx] = messages[idx].model_copy(
                update={"output": "[Previous tool result hidden for context management]"}
            )
        return messages

    def _extract_last_rounds(self, new_outputs):
        """
        Extract the last n complete tool-call rounds from new_outputs.
        A round = one or more function_call items + their corresponding
        function_call_output items. Returns a flat list preserving order.
        """
        n = self.config.context_reset_keep_rounds
        if n <= 0:
            return []

        rounds = []
        i = len(new_outputs) - 1
        while i >= 0 and len(rounds) < n:
            if new_outputs[i].type == "function_call_output":
                # Walk backwards to collect all tool messages for this round
                tool_outputs = []
                while i >= 0 and new_outputs[i].type == "function_call_output":
                    tool_outputs.insert(0, new_outputs[i])
                    i -= 1
                # The assistant message that triggered these tool calls
                fn_calls = []
                while i >= 0 and new_outputs[i].type == "function_call":
                    fn_calls.insert(0, new_outputs[i])
                    i -= 1
                # Add to rounds
                if fn_calls:
                    rounds.insert(0, (fn_calls, tool_outputs))
            else:
                i -= 1

        result = []
        for fn_calls, tool_outputs in rounds:
            result.extend(fn_calls)
            result.extend(tool_outputs)
        return result


if __name__ == "__main__":
    BrowsecompAgent.run_webserver()
