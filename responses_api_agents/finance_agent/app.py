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
"""Multi-turn tool-calling loop for the Vals AI finance agent benchmarks.

One loop serves both benchmarks. Everything that differs between them is a
config value, not a branch in this file: the nudge injected on a prose-only
turn, whether the run is bounded by turns or by wall clock, and which tool
errors abort a rollout instead of being handed back to the model. Those three
fields are required, so a config states its policy rather than inheriting a
default that happens to suit the other benchmark.

The values each benchmark uses are pinned against the upstream package by tests
in ``resources_servers/finance_agent_v2/tests``, which is where that package is
installed.
"""

import asyncio
import json
import logging
import re
import time
from enum import Enum
from typing import Any, List, Optional

from fastapi import Request, Response
from pydantic import ConfigDict, Field

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
    accumulate_response_usage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


logger = logging.getLogger(__name__)

_MODEL_OUTPUT_TYPES = frozenset({"reasoning", "function_call"})

# The model is reached over HTTP, so a context-window rejection arrives as error
# text rather than a typed exception and has to be recognised by shape.
_CONTEXT_OVERFLOW_RE = re.compile(
    r"maximum context length is \d+ tokens|"
    r"context length is (?:only )?\d+ tokens|"
    r"maximum input length of \d+ tokens|"
    r"Please reduce the length of the input|"
    r"exceed.* context (limit|window|length)|"
    r"context window exceeds|"
    r"exceeds maximum length|"
    r"too long.*tokens.*maximum|"
    r"too large for model with \d+ maximum context length|"
    r"longer than the model's context length|"
    r"too many tokens.*size limit exceeded|"
    r"prompt is too long|"
    r"maximum prompt length|"
    r"input length should be|"
    r"sent message larger than max|"
    r"input tokens exceeded|"
    r"(messages?|total length).*too long|"
    r"payload.*too large|"
    r"string too long|"
    r"input exceeded the context window",
    re.IGNORECASE,
)


def _decoded_object(raw: str) -> Optional[dict]:
    """JSON-decode *raw*, returning None unless it decodes to an object."""
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


class StopReason(str, Enum):
    """Why the loop ended.

    Recorded on the rollout so a zero from "never submitted" is distinguishable
    from a zero from "judged wrong".
    """

    DONE_TOOL = "done_tool"
    MAX_TURNS = "max_turns"
    MAX_TIME = "max_time"
    MAX_OUTPUT_TOKENS = "max_output_tokens"
    ERROR = "error"


class FinanceAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    no_tool_call_nudge: str = Field(
        ...,
        description="Injected as a user message when a turn produces prose and "
        "no tool call, so the loop continues instead of stopping.",
    )
    max_time_seconds: Optional[float] = Field(
        ...,
        description="Wall-clock budget for the loop, checked before each model "
        "call, or null for no budget. This is the agent's hard stop; a resource "
        "server's max_rollout_time_seconds is a softer budget that only makes "
        "tools return an error asking the model to submit.",
    )
    abort_on_tool_error_types: List[str] = Field(
        ...,
        description="Exception type names in a tool's error payload that abort "
        "the whole rollout instead of being fed back to the model. Empty means "
        "every tool error is fed back and the run continues.",
    )
    max_steps: Optional[int] = Field(
        default=None,
        description="Maximum model turns, or null for no turn cap.",
    )
    done_tools: List[str] = Field(
        default_factory=lambda: ["submit_final_result"],
        description="Tool names that signal the agent loop should terminate. "
        "When any tool call in a batch matches, remaining calls are skipped "
        "and the loop exits.",
    )
    model_call_timeout: Optional[float] = Field(
        default=None,
        description="Timeout in seconds for each model server call. None = no timeout.",
    )
    tool_call_timeout: Optional[float] = Field(
        default=None,
        description="Timeout in seconds for each tool (resource server) call. None = no timeout.",
    )
    truncate_on_overflow: bool = Field(
        default=False,
        description="When True, drop the oldest exchange on context-overflow "
        "errors and retry. Intended for eval only — during training the full "
        "trajectory must be preserved so reward assignment is accurate.",
    )


class FinanceAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class FinanceAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class FinanceAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class FinanceAgent(SimpleResponsesAPIAgent):
    config: FinanceAgentConfig

    @staticmethod
    def _is_context_overflow_error(exc: Exception) -> bool:
        """True when *exc* indicates the input exceeded the model's context window."""
        return _CONTEXT_OVERFLOW_RE.search(str(exc)) is not None

    @staticmethod
    def _is_model_output(item: Any) -> bool:
        """True for items that originated from a model response (not tool results or user messages)."""
        t = getattr(item, "type", None)
        if t in _MODEL_OUTPUT_TYPES:
            return True
        return t == "message" and getattr(item, "role", None) == "assistant"

    @staticmethod
    def _truncate_oldest_exchange(outputs: List[Any]) -> List[Any]:
        """Remove the oldest model-response + tool-results exchange from outputs.

        Skips the first contiguous block of model output items, then skips the
        following non-model items (tool results, injected user messages), and
        returns everything after that boundary.
        """
        if len(outputs) <= 1:
            return outputs

        i = 0
        n = len(outputs)

        while i < n and FinanceAgent._is_model_output(outputs[i]):
            i += 1

        while i < n and not FinanceAgent._is_model_output(outputs[i]):
            i += 1

        if i >= n:
            return outputs

        return outputs[i:]

    @staticmethod
    def _tool_error_message(tool_output: str) -> Optional[str]:
        """Return the ``error`` string a tool reply carries, if any.

        Two shapes arrive here. A reply from the resource server is wrapped as
        ``{"results": "<payload>"}`` where the payload is itself a JSON string,
        so an error is double-encoded and the inner object must be decoded
        separately. A tool call that never reached the server (timeout,
        transport failure) is rendered locally as a flat ``{"error": ...}``.

        Only an ``error`` field counts, so a successful reply whose text merely
        mentions an exception name is not mistaken for a failure.
        """
        payload = _decoded_object(tool_output)
        if payload is None:
            return None

        results = payload.get("results")
        if isinstance(results, str):
            inner = _decoded_object(results)
            if inner is not None and isinstance(inner.get("error"), str):
                return inner["error"]

        error = payload.get("error")
        return error if isinstance(error, str) else None

    def _aborting_error_type(self, tool_output: str) -> Optional[str]:
        """Return the abort-worthy exception type named in *tool_output*, if any."""
        if not self.config.abort_on_tool_error_types:
            return None
        error = self._tool_error_message(tool_output)
        if error is None:
            return None
        return next(
            (name for name in self.config.abort_on_tool_error_types if error.startswith(f"{name}:")),
            None,
        )

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        new_outputs: List[Any] = []
        usage = None
        step = 0
        last_model_response: Optional[NeMoGymResponse] = None
        model_server_cookies = None
        resources_server_cookies = request.cookies

        done_tools_set = set(self.config.done_tools)
        max_steps = self.config.max_steps
        max_time_seconds = self.config.max_time_seconds
        started_at = time.monotonic()
        stop_reason = StopReason.ERROR

        # Check max_steps at the TOP so we never start a turn past the limit.
        while max_steps is None or step < max_steps:
            # Checked before the turn's query, so an exhausted budget ends the
            # run rather than paying for one more call.
            if max_time_seconds is not None:
                elapsed = time.monotonic() - started_at
                if elapsed >= max_time_seconds:
                    logger.warning(
                        "Time budget exhausted (%.0fs / %.0fs) after %d steps — terminating agent loop",
                        elapsed,
                        max_time_seconds,
                        step,
                    )
                    stop_reason = StopReason.MAX_TIME
                    break

            step += 1
            new_body = body.model_copy(update={"input": body.input + new_outputs})

            try:
                coro = self.server_client.post(
                    server_name=self.config.model_server.name,
                    url_path="/v1/responses",
                    json=new_body,
                    cookies=model_server_cookies,
                )
                model_resp_raw = await asyncio.wait_for(coro, timeout=self.config.model_call_timeout)

                await raise_for_status(model_resp_raw)
                model_response_json = await get_response_json(model_resp_raw)
                model_server_cookies = model_resp_raw.cookies
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except asyncio.TimeoutError:
                logger.warning(
                    "Model call timed out after %ss on step %d — terminating agent loop",
                    self.config.model_call_timeout,
                    step,
                )
                stop_reason = StopReason.ERROR
                break
            except Exception as e:
                if self.config.truncate_on_overflow and self._is_context_overflow_error(e):
                    truncated = self._truncate_oldest_exchange(new_outputs)
                    if len(truncated) < len(new_outputs):
                        logger.info(
                            "Context overflow on step %d — truncated oldest exchange: %d → %d output items",
                            step,
                            len(new_outputs),
                            len(truncated),
                        )
                        new_outputs = truncated
                        continue
                logger.error("Model call failed on step %d: %s: %s", step, type(e).__name__, e)
                stop_reason = StopReason.ERROR
                break

            output = model_response.output
            last_model_response = model_response
            new_outputs.extend(output)

            usage = accumulate_response_usage(usage, model_response.usage)

            if model_response.incomplete_details and model_response.incomplete_details.reason == "max_output_tokens":
                stop_reason = StopReason.MAX_OUTPUT_TOKENS
                break

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]

            if not all_fn_calls and all_output_messages:
                # Prose between tool calls does not end the run; the nudge keeps
                # the model going until it calls a done tool or a limit is hit.
                new_outputs.append(NeMoGymEasyInputMessage(role="user", content=self.config.no_tool_call_nudge))
                continue

            done = False
            aborted = False
            for output_function_call in all_fn_calls:
                try:
                    coro = self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path=f"/{output_function_call.name}",
                        json=json.loads(output_function_call.arguments),
                        cookies=resources_server_cookies,
                    )
                    api_response = await asyncio.wait_for(coro, timeout=self.config.tool_call_timeout)
                    resources_server_cookies = api_response.cookies
                    tool_output = (await api_response.content.read()).decode()
                except asyncio.TimeoutError:
                    logger.warning(
                        "Tool call '%s' timed out after %ss",
                        output_function_call.name,
                        self.config.tool_call_timeout,
                    )
                    tool_output = json.dumps(
                        {
                            "error": f"Tool call timed out after {self.config.tool_call_timeout}s. "
                            "Please try a different approach or submit your final answer."
                        }
                    )
                except Exception as e:
                    logger.error(
                        "Tool call '%s' failed: %s: %s",
                        output_function_call.name,
                        type(e).__name__,
                        e,
                    )
                    tool_output = json.dumps(
                        {
                            "error": f"Tool call failed: {type(e).__name__}: {e}. "
                            "Please try a different approach or submit your final answer."
                        }
                    )

                tool_response = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=tool_output,
                )
                new_outputs.append(tool_response)

                if output_function_call.name in done_tools_set:
                    logger.info(
                        "Tool '%s' signaled done — terminating agent loop",
                        output_function_call.name,
                    )
                    done = True
                    stop_reason = StopReason.DONE_TOOL
                    break

                if error_type := self._aborting_error_type(tool_output):
                    logger.error(
                        "Tool '%s' raised %s — aborting rollout",
                        output_function_call.name,
                        error_type,
                    )
                    aborted = True
                    stop_reason = StopReason.ERROR
                    break

            if done or aborted:
                break
        else:
            if max_steps is not None:
                stop_reason = StopReason.MAX_TURNS

        if stop_reason is StopReason.MAX_TURNS:
            logger.warning("Reached max_steps=%d — terminating agent loop", max_steps)

        logger.info("Agent loop finished after %d steps: stop_reason=%s", step, stop_reason.value)

        if last_model_response is None:
            logger.error("Agent loop terminated without any successful model response")
            last_model_response = NeMoGymResponse(
                id="error",
                created_at=0.0,
                model="error",
                object="response",
                output=new_outputs or [],
                tools=[],
                parallel_tool_calls=False,
                tool_choice="auto",
            )

        cookie_items = list(resources_server_cookies.items())
        if model_server_cookies:
            cookie_items.extend(model_server_cookies.items())
        for k, v in cookie_items:
            response.set_cookie(k, v)

        last_model_response.output = new_outputs
        last_model_response.usage = usage
        # Carry why the loop ended into the rollout. Without this the stop reason
        # exists only in the agent's log, so a results file cannot distinguish an
        # answer the model chose to submit from one cut short by the time budget or
        # a tool abort. ``metadata`` is the Responses API's own string-keyed side
        # channel, so it survives verify() untouched.
        last_model_response.metadata = {
            **(last_model_response.metadata or {}),
            "stop_reason": stop_reason.value,
            "steps": str(step),
        }
        return last_model_response

    async def run(self, request: Request, body: FinanceAgentRunRequest) -> FinanceAgentVerifyResponse:
        try:
            return await self._run_inner(request, body)
        except Exception as e:
            logger.error("run() failed — returning reward=0: %s: %s", type(e).__name__, e)
            empty_response = NeMoGymResponse(
                id="error",
                created_at=0.0,
                model="error",
                object="response",
                output=[],
                tools=[],
                parallel_tool_calls=False,
                tool_choice="auto",
                metadata={"stop_reason": StopReason.ERROR.value},
            )
            return FinanceAgentVerifyResponse(
                responses_create_params=body.responses_create_params,
                response=empty_response,
                reward=0.0,
            )

    async def _run_inner(self, request: Request, body: FinanceAgentRunRequest) -> FinanceAgentVerifyResponse:
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

        verify_request = FinanceAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": await get_response_json(response)}
        )

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return FinanceAgentVerifyResponse.model_validate(await get_response_json(verify_response))

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
    FinanceAgent.run_webserver()
