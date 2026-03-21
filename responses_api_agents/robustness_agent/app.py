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
import random
import re
from typing import List, Optional

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


_REWRITE_SYSTEM_PROMPT = """\
You are a paraphrasing assistant who preserves original meaning while slightly changing names to augment data and prevent overfitting.

Given messages and tool definitions, rewrite the user and system messages with varied phrasing, generate synonymous names for each tool, and generate synonymous names for each tool's parameters.

Respond with valid JSON only, no markdown, no explanation. Use this exact schema:
{
  "rewritten_messages": [{"role": "...", "content": "..."}],
  "tool_name_map": {"original_tool_name": "new_tool_name"},
  "arg_name_maps": {"original_tool_name": {"original_arg": "new_arg"}}
}"""


class RobustnessAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    rewriter_model_server: Optional[ModelServerRef] = None
    rewrite_prompts: bool = True
    rewrite_tool_names: bool = True
    rewrite_arg_names: bool = True
    rewrite_prob: float = 0.2
    max_steps: int = None


class RobustnessAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class RobustnessAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class RobustnessAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class RobustnessAgent(SimpleResponsesAPIAgent):
    config: RobustnessAgentConfig

    async def _rewrite(self, messages: list, tools: list, cookies) -> tuple[list, dict, dict]:
        nothing_to_rewrite = (
            not self.config.rewrite_prompts
            and not self.config.rewrite_tool_names
            and not self.config.rewrite_arg_names
        )
        if not self.config.rewriter_model_server or nothing_to_rewrite or random.random() >= self.config.rewrite_prob:
            return messages, {}, {}

        tool_defs = json.dumps([dict(t) for t in (tools or [])], indent=2)
        messages_plain = [
            {
                "role": m.role if hasattr(m, "role") else m.get("role", "user"),
                "content": m.content if hasattr(m, "content") else m.get("content", ""),
            }
            for m in messages
        ]

        rewrite_body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                NeMoGymEasyInputMessage(role="system", content=_REWRITE_SYSTEM_PROMPT),
                NeMoGymEasyInputMessage(
                    role="user",
                    content=f"Messages:\n{json.dumps(messages_plain, indent=2)}\n\nTools:\n{tool_defs}",
                ),
            ]
        )

        try:
            resp = await self.server_client.post(
                server_name=self.config.rewriter_model_server.name,
                url_path="/v1/responses",
                json=rewrite_body,
                cookies=cookies,
            )
            await raise_for_status(resp)
            resp_json = await get_response_json(resp)
            rewrite_response = NeMoGymResponse.model_validate(resp_json)

            output_text = ""
            for item in rewrite_response.output:
                if item.type == "message" and item.role == "assistant":
                    for part in item.content:
                        if part.type == "output_text":
                            output_text = part.text
                            break

            # strip thinking tags emitted by reasoning models
            output_text = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()
            output_text = re.sub(r"<thinking>.*?</thinking>", "", output_text, flags=re.DOTALL).strip()

            parsed = json.loads(output_text)
            tool_name_map = parsed.get("tool_name_map", {}) if self.config.rewrite_tool_names else {}
            arg_name_maps = parsed.get("arg_name_maps", {}) if self.config.rewrite_arg_names else {}

            if self.config.rewrite_prompts:
                rewritten_messages = [
                    NeMoGymEasyInputMessage(role=m["role"], content=m["content"])
                    for m in parsed.get("rewritten_messages", messages_plain)
                ]
            else:
                rewritten_messages = messages
            return rewritten_messages, tool_name_map, arg_name_maps

        except Exception:
            return messages, {}, {}

    def _apply_tool_rewrites(self, tools: list, tool_name_map: dict, arg_name_maps: dict) -> list:
        if not tools or (not tool_name_map and not arg_name_maps):
            return tools

        rewritten = []
        for tool in tools:
            t = dict(tool)
            original_name = t.get("name", "")
            t["name"] = tool_name_map.get(original_name, original_name)

            arg_map = arg_name_maps.get(original_name, {})
            if arg_map and "parameters" in t and t["parameters"] and "properties" in t["parameters"]:
                props = t["parameters"]["properties"]
                t["parameters"] = dict(t["parameters"])
                t["parameters"]["properties"] = {arg_map.get(k, k): v for k, v in props.items()}
                if "required" in t["parameters"]:
                    t["parameters"]["required"] = [arg_map.get(r, r) for r in t["parameters"]["required"]]
            rewritten.append(t)
        return rewritten

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        rewritten_messages, tool_name_map, arg_name_maps = await self._rewrite(
            body.input, body.tools or [], request.cookies
        )
        inv_tool_name_map = {v: k for k, v in tool_name_map.items()}
        inv_arg_name_maps = {
            tool_name_map.get(orig_tool, orig_tool): {v: k for k, v in arg_map.items()}
            for orig_tool, arg_map in arg_name_maps.items()
        }

        body.input = rewritten_messages
        if body.tools:
            body.tools = self._apply_tool_rewrites(body.tools, tool_name_map, arg_name_maps)

        new_outputs = []
        usage = None
        step = 0
        model_server_cookies = None  # update the cookies on every model response
        resources_server_cookies = request.cookies  # update the cookies on every resources server response

        while True:
            step += 1
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

            output = model_response.output
            new_outputs.extend(output)

            if not usage:
                usage = model_response.usage

            if usage:
                usage.input_tokens += model_response.usage.input_tokens
                usage.output_tokens += model_response.usage.output_tokens
                usage.total_tokens += model_response.usage.total_tokens

                # TODO support more advanced token details
                usage.input_tokens_details.cached_tokens = 0
                usage.output_tokens_details.reasoning_tokens = 0

            if model_response.incomplete_details and model_response.incomplete_details.reason == "max_output_tokens":
                break

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]
            if not all_fn_calls and all_output_messages:
                break

            for output_function_call in all_fn_calls:
                original_tool_name = inv_tool_name_map.get(output_function_call.name, output_function_call.name)
                call_args = json.loads(output_function_call.arguments)
                inv_args = inv_arg_name_maps.get(output_function_call.name, {})
                if inv_args:
                    call_args = {inv_args.get(k, k): v for k, v in call_args.items()}

                api_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{original_tool_name}",
                    json=call_args,
                    cookies=resources_server_cookies,
                )
                # We don't raise for status here since it's a valid return for the API to error e.g. if the model outputs an invalid call or something.
                resources_server_cookies = api_response.cookies

                tool_response = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=(await api_response.content.read()).decode(),
                )
                new_outputs.append(tool_response)

            # Check if max steps is not None and if we have exhausted it.
            if self.config.max_steps and step >= self.config.max_steps:
                break

        # Propogate any extra cookies necessary for downstream verification
        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)

        model_response.output = new_outputs
        model_response.usage = usage
        return model_response

    async def run(self, request: Request, body: RobustnessAgentRunRequest) -> RobustnessAgentVerifyResponse:
        cookies = request.cookies

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = seed_session_response.cookies

        response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(response)
        cookies = response.cookies

        verify_request = RobustnessAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": await get_response_json(response)}
        )

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return RobustnessAgentVerifyResponse.model_validate(await get_response_json(verify_response))

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
    RobustnessAgent.run_webserver()
