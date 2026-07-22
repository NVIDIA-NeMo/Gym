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
import asyncio
import json
import time
from typing import Any, ClassVar, Dict, List, Optional

from aiohttp import ClientConnectionError, ClientResponseError
from fastapi import HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

import nemo_gym.base_resources_server as base_resources_server
from nemo_gym.base_resources_server import (
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
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_models.vllm_model import app as vllm_model_app


try:
    AggregateMetrics = base_resources_server.AggregateMetrics
    AggregateMetricsRequest = base_resources_server.AggregateMetricsRequest
except AttributeError:
    # Older Gym images predate aggregate-metrics types. The endpoint is not
    # called by those images, but the agent still needs matching route models.
    class AggregateMetricsRequest(BaseModel):
        verify_responses: List[Dict[str, Any]]

    class AggregateMetrics(BaseModel):
        group_level_metrics: List[Dict[str, Any]] = Field(default_factory=list)
        agent_metrics: Dict[str, Any] = Field(default_factory=dict)
        key_metrics: Dict[str, Any] = Field(default_factory=dict)


try:
    split_responses_input_output_items = vllm_model_app.split_responses_input_output_items
except AttributeError:
    # Keep the newer vLLM adapter's input/output split behavior when running
    # against an image from before this helper was exported.
    def split_responses_input_output_items(
        items: List[Any],
    ) -> tuple[List[Any], List[Any]]:
        if not items:
            return [], []

        for i, item in enumerate(items):
            if (
                getattr(item, "role", None) == "assistant"
                or getattr(item, "type", None) in {"reasoning", "reasoning_item"}
                or getattr(item, "type", None) in ("function_call",)
            ):
                break

        return items[:i], items[i:]


NG_FAILURE_CLASS_KEY = "_ng_failure_class"
TRANSIENT_FAILURE_CLASS = "transient"


class SyntheticToolUseAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_agent_steps: int = Field(default=50, ge=1)
    seed_first_user_message: bool = True


class SyntheticToolUseAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    initial_user_message: Optional[str] = None


class SyntheticToolUseAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SyntheticToolUseAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    instance_config: Dict[str, Any] = Field(default_factory=lambda: {"mask_sample": False})


class SyntheticToolUseAgent(SimpleResponsesAPIAgent):
    AGENT_SYSTEM_MESSAGE_TEMPLATE: ClassVar[
        str
    ] = """You are a customer service agent that helps the user.  The policy that determines how you should respond to requests from users is described below between the <policy> and </policy> tags.

In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

<policy>
{domain_policy}
</policy>

Try to be helpful and always follow the policy."""

    AGENT_PARALLEL_SYSTEM_MESSAGE_TEMPLATE: ClassVar[
        str
    ] = """You are a customer service agent that helps the user.  The policy that determines how you should respond to requests from users is described below between the <policy> and </policy> tags.

In each turn you can either:
- Send a message to the user.
- Make one or more tool calls.
You cannot do both at the same time.

<policy>
{domain_policy}
</policy>

Try to be helpful and always follow the policy."""

    config: SyntheticToolUseAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        resources_server_cookies = request.cookies
        model_server_cookies = None
        new_outputs: List[Any] = []
        usage = None
        last_model_response: Optional[NeMoGymResponse] = None
        terminal_before_agent_call = False

        resume_state = self._prefilled_resume_state(body)
        if resume_state is None and self.config.seed_first_user_message:
            user_response = await self._post_resource(
                "/next_user_message",
                {},
                cookies=resources_server_cookies,
            )
            resources_server_cookies = user_response.cookies
            user_payload = await get_response_json(user_response)
            user_message = user_payload.get("message") or ""
            if user_message:
                new_outputs.append(NeMoGymEasyInputMessage(role="user", content=user_message))
            if not user_payload.get("should_continue", True):
                terminal_before_agent_call = True
        elif resume_state is not None:
            next_actor, pending_tool_calls = resume_state
            if next_actor == "terminal":
                terminal_before_agent_call = True
            elif next_actor == "user":
                user_response = await self._post_resource(
                    "/next_user_message",
                    {},
                    cookies=resources_server_cookies,
                )
                resources_server_cookies = user_response.cookies
                user_payload = await get_response_json(user_response)
                user_message = user_payload.get("message") or ""
                if user_message:
                    new_outputs.append(NeMoGymEasyInputMessage(role="user", content=user_message))
                if not user_payload.get("should_continue", True):
                    terminal_before_agent_call = True
            elif next_actor == "environment":
                prefilled_response = self._empty_response()
                prefilled_response.output = list(pending_tool_calls)
                should_continue = True
                for function_call in pending_tool_calls:
                    tool_output = await self._execute_tool_call(
                        function_call,
                        resources_server_cookies,
                        prefilled_response,
                    )
                    resources_server_cookies = tool_output["cookies"]
                    new_outputs.append(tool_output["output"])
                    should_continue = should_continue and tool_output["should_continue"]
                if not should_continue:
                    terminal_before_agent_call = True

        step = 0
        while not terminal_before_agent_call:
            step += 1
            model_body = body.model_copy(update={"input": body.input + new_outputs})
            model_payload = model_body.model_dump(mode="json", exclude_unset=True)
            model_payload["input"] = self._normalize_response_input_items(body.input + new_outputs)
            model_response_http = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=model_payload,
                cookies=model_server_cookies,
            )
            await raise_for_status(model_response_http)
            model_server_cookies = model_response_http.cookies
            model_response_json = await get_response_json(model_response_http)
            try:
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as exc:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                ) from exc

            self._filter_unselected_function_calls(model_response, body.parallel_tool_calls)
            last_model_response = model_response
            new_outputs.extend(model_response.output)
            usage = self._merge_usage(usage, model_response)
            agent_step_limit_reached = step >= self.config.max_agent_steps

            if model_response.incomplete_details:
                error = f"Model response was incomplete: {model_response.incomplete_details}"
                terminal = await self._record_generation_error("agent", error, resources_server_cookies)
                resources_server_cookies = terminal["cookies"]
                break

            function_calls = self._function_calls(model_response)
            if function_calls:
                calls_to_execute = function_calls if body.parallel_tool_calls else function_calls[:1]
                if agent_step_limit_reached:
                    terminal = await self._record_agent_step_limit(
                        calls_to_execute,
                        resources_server_cookies,
                        model_response,
                    )
                    resources_server_cookies = terminal["cookies"]
                    break

                should_continue = True
                for function_call in calls_to_execute:
                    tool_output = await self._execute_tool_call(
                        function_call,
                        resources_server_cookies,
                        model_response,
                    )
                    resources_server_cookies = tool_output["cookies"]
                    new_outputs.append(tool_output["output"])
                    should_continue = should_continue and tool_output["should_continue"]
                if not should_continue:
                    break
            else:
                message_outputs = self._message_outputs(model_response)
                if message_outputs:
                    terminal = await self._record_assistant_message(
                        message_outputs[0],
                        resources_server_cookies,
                        model_response,
                    )
                    resources_server_cookies = terminal["cookies"]
                    if not terminal["should_continue"]:
                        break

                    user_response = await self._post_resource(
                        "/next_user_message",
                        {},
                        cookies=resources_server_cookies,
                    )
                    resources_server_cookies = user_response.cookies
                    user_payload = await get_response_json(user_response)
                    if not user_payload.get("should_continue", True):
                        break
                    user_message = user_payload.get("message") or ""
                    if user_message:
                        new_outputs.append(NeMoGymEasyInputMessage(role="user", content=user_message))
                    if agent_step_limit_reached:
                        terminal = await self._record_agent_step_limit(
                            [],
                            resources_server_cookies,
                            model_response,
                        )
                        resources_server_cookies = terminal["cookies"]
                        break
                    if not user_message:
                        break
                else:
                    error = (
                        "No text message or function call was found in the output list "
                        f"of a response: {model_response.output}"
                    )
                    terminal = await self._record_generation_error("agent", error, resources_server_cookies)
                    resources_server_cookies = terminal["cookies"]
                    break

        for cookie_source in (resources_server_cookies, model_server_cookies):
            if cookie_source is None:
                continue
            for key, value in cookie_source.items():
                response.set_cookie(key, value)

        if last_model_response is None:
            last_model_response = self._empty_response()
        last_model_response.output = new_outputs
        last_model_response.usage = usage
        return last_model_response

    async def run(
        self, request: Request, body: SyntheticToolUseAgentRunRequest
    ) -> SyntheticToolUseAgentVerifyResponse:
        body = self._materialize_initial_user_message(body)
        cookies = request.cookies
        try:
            seed_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_response)
            cookies = seed_response.cookies
        except Exception as exc:
            if not self._is_transient_infrastructure_error(exc):
                raise
            return self._build_transient_failure_response(body, stage="seed_session", exc=exc)

        responses_create_params = body.responses_create_params.model_copy(deep=True)
        self._validate_materialized_responses_create_params(responses_create_params)

        try:
            response = await self.server_client.post(
                server_name=self.config.name,
                url_path="/v1/responses",
                json=responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(response)
            cookies = response.cookies
            response_payload = await get_response_json(response)
            responses_create_params, response_payload = self._canonicalize_run_transcript(
                responses_create_params,
                response_payload,
            )
        except Exception as exc:
            if not self._is_transient_infrastructure_error(exc):
                raise
            return self._build_transient_failure_response(body, stage="rollout", exc=exc)

        verify_request = SyntheticToolUseAgentVerifyRequest.model_validate(
            body.model_dump() | {"responses_create_params": responses_create_params, "response": response_payload}
        )
        try:
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(verify_response)
            return SyntheticToolUseAgentVerifyResponse.model_validate(await get_response_json(verify_response))
        except Exception as exc:
            if not self._is_transient_infrastructure_error(exc):
                raise
            return self._build_transient_failure_response(body, stage="verify", exc=exc)

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))

    async def _execute_tool_call(
        self,
        function_call: NeMoGymResponseFunctionToolCall,
        cookies: Dict[str, str],
        model_response: NeMoGymResponse,
    ) -> Dict[str, Any]:
        resource_response = await self._post_resource(
            "/execute_agent_tool_call",
            {
                "tool_name": function_call.name,
                "tool_call_id": function_call.call_id,
                "arguments": function_call.arguments,
                "response": model_response.model_dump(mode="json", exclude_unset=True),
            },
            cookies=cookies,
        )
        payload = await get_response_json(resource_response)
        tool_output = payload.get("output", "")
        if not isinstance(tool_output, str):
            tool_output = json.dumps(tool_output, ensure_ascii=False)
        return {
            "cookies": resource_response.cookies,
            "output": NeMoGymFunctionCallOutput(
                call_id=function_call.call_id,
                output=tool_output,
            ),
            "should_continue": bool(payload.get("should_continue", True)),
        }

    async def _record_assistant_message(
        self,
        assistant_message: NeMoGymResponseOutputMessage,
        cookies: Dict[str, str],
        model_response: NeMoGymResponse,
    ) -> Dict[str, Any]:
        text = self._message_text(assistant_message)
        resource_response = await self._post_resource(
            "/record_agent_message",
            {
                "content": text,
                "response": model_response.model_dump(mode="json", exclude_unset=True),
            },
            cookies=cookies,
        )
        payload = await get_response_json(resource_response)
        return {"cookies": resource_response.cookies, "should_continue": bool(payload.get("should_continue", True))}

    async def _record_generation_error(
        self,
        source: str,
        error: str,
        cookies: Dict[str, str],
    ) -> Dict[str, Any]:
        resource_response = await self._post_resource(
            "/record_generation_error",
            {"source": source, "error": error},
            cookies=cookies,
        )
        payload = await get_response_json(resource_response)
        return {
            "cookies": resource_response.cookies,
            "should_continue": bool(payload.get("should_continue", False)),
        }

    async def _record_agent_step_limit(
        self,
        function_calls: List[NeMoGymResponseFunctionToolCall],
        cookies: Dict[str, str],
        model_response: NeMoGymResponse,
    ) -> Dict[str, Any]:
        resource_response = await self._post_resource(
            "/record_agent_step_limit",
            {
                "max_agent_steps": self.config.max_agent_steps,
                "tool_calls": [
                    {
                        "tool_name": function_call.name,
                        "tool_call_id": function_call.call_id,
                        "arguments": function_call.arguments,
                    }
                    for function_call in function_calls
                ],
                "response": model_response.model_dump(mode="json", exclude_unset=True),
            },
            cookies=cookies,
        )
        payload = await get_response_json(resource_response)
        return {
            "cookies": resource_response.cookies,
            "should_continue": bool(payload.get("should_continue", False)),
        }

    async def _post_resource(self, path: str, payload: Dict[str, Any], cookies: Optional[Dict[str, str]]):
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path=path,
            json=payload,
            cookies=cookies,
        )
        await raise_for_status(response)
        return response

    def _canonicalize_run_transcript(
        self,
        responses_create_params: NeMoGymResponseCreateParamsNonStreaming,
        response_payload: Dict[str, Any],
    ) -> tuple[NeMoGymResponseCreateParamsNonStreaming, Dict[str, Any]]:
        response = NeMoGymResponse.model_validate(response_payload)
        input_items, output_items = split_responses_input_output_items(response.output)
        if not input_items:
            return responses_create_params, response_payload

        responses_create_params = responses_create_params.model_copy(deep=True)
        responses_create_params.input = self._input_items(responses_create_params) + input_items
        response.output = output_items
        return responses_create_params, response.model_dump(mode="json", exclude_unset=True)

    def _has_user_input(self, body: NeMoGymResponseCreateParamsNonStreaming) -> bool:
        if isinstance(body.input, str):
            return bool(body.input)
        return any(self._item_role(item) == "user" for item in body.input)

    def _input_items(self, body: NeMoGymResponseCreateParamsNonStreaming) -> List[Any]:
        if isinstance(body.input, str):
            return [NeMoGymEasyInputMessage(role="user", content=body.input)]
        return list(body.input or [])

    def _materialize_initial_user_message(
        self, body: SyntheticToolUseAgentRunRequest
    ) -> SyntheticToolUseAgentRunRequest:
        responses_create_params = body.responses_create_params.model_copy(deep=True)
        items = self._input_items(responses_create_params)
        history_items = [item for item in items if self._is_conversation_item(item)]
        user_messages = [
            self._input_item_text(item)
            for item in history_items
            if self._item_type(item) == "message" and self._item_role(item) == "user"
        ]

        if body.initial_user_message is not None:
            if user_messages and user_messages[0] != body.initial_user_message:
                raise HTTPException(
                    status_code=400,
                    detail="initial_user_message must match the first user message in responses_create_params.input.",
                )
            if not user_messages and history_items:
                raise HTTPException(
                    status_code=400,
                    detail="initial_user_message cannot be merged into an existing prefilled history without a user message.",
                )
            if not history_items:
                responses_create_params.input = items + [
                    NeMoGymEasyInputMessage(role="user", content=body.initial_user_message)
                ]

        return body.model_copy(update={"responses_create_params": responses_create_params})

    def _prefilled_resume_state(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
    ) -> Optional[tuple[str, List[NeMoGymResponseFunctionToolCall]]]:
        items = [item for item in self._input_items(body) if self._is_conversation_item(item)]
        if not items:
            return None

        tool_calls: Dict[str, NeMoGymResponseFunctionToolCall] = {}
        tool_call_order: List[str] = []
        executed_call_ids: set[str] = set()
        for item in items:
            item_type = self._item_type(item)
            if item_type == "function_call":
                function_call = NeMoGymResponseFunctionToolCall.model_validate(
                    item if isinstance(item, dict) else item.model_dump(mode="json")
                )
                tool_calls[function_call.call_id] = function_call
                tool_call_order.append(function_call.call_id)
            elif item_type == "function_call_output":
                call_id = item.get("call_id") if isinstance(item, dict) else getattr(item, "call_id", None)
                if isinstance(call_id, str):
                    executed_call_ids.add(call_id)

        pending_tool_calls = [tool_calls[call_id] for call_id in tool_call_order if call_id not in executed_call_ids]
        if pending_tool_calls:
            return "environment", pending_tool_calls

        last_item = items[-1]
        last_type = self._item_type(last_item)
        if last_type == "function_call_output":
            return "agent", []
        if last_type == "message":
            role = self._item_role(last_item)
            if role == "assistant":
                return "user", []
            if role == "user":
                content = self._input_item_text(last_item)
                if "###STOP###" in content or "###TRANSFER###" in content:
                    return "terminal", []
                return "agent", []
        raise HTTPException(status_code=400, detail="Prefilled history does not end at a resumable state.")

    def _is_conversation_item(self, item: Any) -> bool:
        item_type = self._item_type(item)
        role = self._item_role(item)
        return item_type != "reasoning" and role not in {"system", "developer"}

    def _item_type(self, item: Any) -> Optional[str]:
        if isinstance(item, dict):
            item_type = item.get("type")
        else:
            item_type = getattr(item, "type", None)
        if item_type is not None:
            return str(item_type)
        if self._item_role(item) is not None:
            return "message"
        if isinstance(item, dict) and "output" in item:
            return "function_call_output"
        if isinstance(item, dict) and "name" in item and "arguments" in item:
            return "function_call"
        return None

    def _input_item_text(self, item: Any) -> str:
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for content_item in content:
                item_type = (
                    content_item.get("type") if isinstance(content_item, dict) else getattr(content_item, "type", None)
                )
                text_value = (
                    content_item.get("text") if isinstance(content_item, dict) else getattr(content_item, "text", None)
                )
                if item_type not in {"input_text", "output_text"} or not isinstance(text_value, str):
                    break
                text_parts.append(text_value)
            else:
                return "".join(text_parts)
        raise HTTPException(status_code=400, detail="Prefilled user and assistant messages must contain only text.")

    def _normalize_response_input_items(self, items: List[Any]) -> List[Any]:
        normalized = []
        for item in items:
            if isinstance(item, BaseModel):
                item = item.model_dump(mode="json", exclude_unset=False, exclude_none=True)
            elif isinstance(item, dict):
                item = dict(item)
            else:
                normalized.append(item)
                continue

            if "type" not in item:
                if item.get("role"):
                    item["type"] = "message"
                elif "call_id" in item and "output" in item:
                    item["type"] = "function_call_output"
                elif "call_id" in item and "name" in item and "arguments" in item:
                    item["type"] = "function_call"
                elif "summary" in item:
                    item["type"] = "reasoning"

            normalized.append(item)
        return normalized

    def _item_role(self, item: Any) -> Optional[str]:
        if isinstance(item, dict):
            role = item.get("role")
        else:
            role = getattr(item, "role", None)
        return str(role) if role is not None else None

    def _validate_materialized_responses_create_params(
        self, responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    ) -> None:
        input_items = self._input_items(responses_create_params)
        if not any(self._item_role(item) in ("system", "developer") for item in input_items):
            raise HTTPException(
                status_code=400,
                detail="responses_create_params.input must include the materialized policy system prompt.",
            )
        if not responses_create_params.tools:
            raise HTTPException(
                status_code=400,
                detail="responses_create_params.tools must include materialized policy tools.",
            )

    def _empty_response(self) -> NeMoGymResponse:
        return NeMoGymResponse.model_validate(
            {
                "id": "synthetic_tool_use_no_agent_response",
                "created_at": 0.0,
                "model": self.config.model_server.name,
                "object": "response",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )

    def _build_transient_failure_response(
        self,
        body: SyntheticToolUseAgentRunRequest,
        *,
        stage: str,
        exc: BaseException,
    ) -> SyntheticToolUseAgentVerifyResponse:
        reason = f"{stage} failed: {type(exc).__name__}: {exc}"
        print(f"[synthetic-tool-use-{stage}-transient] {reason}", flush=True)

        failure_response = NeMoGymResponse(
            id=f"synthetic_tool_use_{stage}_failed",
            created_at=int(time.time()),
            model=body.responses_create_params.model or self.config.model_server.name,
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id=f"synthetic_tool_use_{stage}_failure_message",
                    content=[NeMoGymResponseOutputText(annotations=[], text=f"Failed: {reason}")],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        payload = body.model_dump(mode="json")
        payload.update(
            {
                "response": failure_response.model_dump(mode="json"),
                "reward": 0.0,
                "skipped": False,
                "error_message": reason,
                "error_class": TRANSIENT_FAILURE_CLASS,
                "error_stage": stage,
                NG_FAILURE_CLASS_KEY: TRANSIENT_FAILURE_CLASS,
                "instance_config": {"mask_sample": True},
            }
        )
        return SyntheticToolUseAgentVerifyResponse.model_validate(payload)

    def _is_transient_infrastructure_error(self, exc: BaseException) -> bool:
        seen: set[int] = set()

        def is_transient(current: Optional[BaseException]) -> bool:
            if current is None or id(current) in seen:
                return False
            seen.add(id(current))
            if isinstance(current, ClientResponseError):
                return current.status in {408, 409, 425, 429} or current.status >= 500
            if isinstance(current, (ClientConnectionError, asyncio.TimeoutError)):
                return True
            return is_transient(current.__cause__) or is_transient(current.__context__)

        return is_transient(exc)

    def _function_calls(self, model_response: NeMoGymResponse) -> List[NeMoGymResponseFunctionToolCall]:
        return [output for output in model_response.output if output.type == "function_call"]

    def _filter_unselected_function_calls(
        self,
        model_response: NeMoGymResponse,
        parallel_tool_calls: bool,
    ) -> None:
        if parallel_tool_calls:
            return

        selected_function_call = False
        filtered_outputs = []
        for output in model_response.output:
            if output.type == "function_call":
                if selected_function_call:
                    continue
                selected_function_call = True
            filtered_outputs.append(output)
        model_response.output = filtered_outputs
        model_response.parallel_tool_calls = False

    def _message_outputs(self, model_response: NeMoGymResponse) -> List[NeMoGymResponseOutputMessage]:
        return [
            output
            for output in model_response.output
            if output.type == "message"
            and any(getattr(content_item, "type", None) == "output_text" for content_item in output.content)
        ]

    def _message_text(self, message: NeMoGymResponseOutputMessage) -> str:
        texts = [getattr(item, "text", "") for item in message.content]
        return "\n".join(text for text in texts if text)

    def _merge_usage(self, usage, model_response: NeMoGymResponse):
        if usage is None:
            current = model_response.usage
            model_response.usage = None
            return current
        if model_response.usage:
            usage.input_tokens += model_response.usage.input_tokens
            usage.output_tokens += model_response.usage.output_tokens
            usage.total_tokens += model_response.usage.total_tokens
            usage.input_tokens_details.cached_tokens = 0
            usage.output_tokens_details.reasoning_tokens = 0
            model_response.usage = None
        return usage


if __name__ == "__main__":
    SyntheticToolUseAgent.run_webserver()
