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
"""Finance Agent v2 (FABv2) agent loop.

Forked from ``responses_api_agents/finance_agent/app.py`` at commit 7b2e174e.
That agent mirrors ``vals-ai/finance-agent`` (v1) and is used by the v1.1
benchmark *and by training*, so it is deliberately left untouched; this
component mirrors ``vals-ai/finance-agent-v2`` instead.

Why a fork rather than shared code with per-benchmark config
------------------------------------------------------------
Vals maintains v1 and v2 as separate repositories. Both build their loop from
the same engine (``model_library.agent.Agent``) and fork only the ~30 lines of
``AgentHooks`` that express per-benchmark policy. Their two hook sets have
already diverged in four ways (nudge text, turn-limit vs time-limit, tool set,
abort-on-retry-exhaustion), so the v1 loop is not a faithful v2 harness.

Mirroring that split exactly would mean extracting Gym's loop into a shared
engine and giving each benchmark its own hooks. That refactor touches the v1
loop, which is on the training path, and the abort-on-tool-error behavior below
is not expressible as a defaulted config field. Forking keeps v1 byte-for-byte
unchanged. Extracting a shared base is the right follow-up once v1 is between
training runs.

Why the upstream loop is not imported
-------------------------------------
``model_library.agent.Agent`` is async and ``get_agent(parameters, llm=...)``
accepts an injected LLM, so importing it is technically possible. It would
require an adapter implementing ``model_library.base.LLM.query`` against
model_library's internal request/response types, would move tool execution
in-process (bypassing the Gym resource server that owns the cache, session
state, and ``/verify``), and would have to translate ``AgentResult`` back into
Responses-API items for Gym's rollout format and RL trajectories. That trades
four small mirrored facts for a dependency on a large private API, so the loop
is reimplemented and everything else is imported.

Upstream parity surface
-----------------------
Everything upstream exposes as a module-level value is imported below, so it
cannot drift silently. The one value that cannot be imported is the no-tool-call
nudge, which lives as an inline literal inside a closure in upstream's
``get_agent._before_query``; it is copied into ``UPSTREAM_NO_TOOL_CALL_NUDGE``
and covered by a source-fingerprint test in ``tests/test_app.py``.
"""

import asyncio
import json
import logging
import re
import time
from enum import Enum
from typing import Any, List, Optional

from fastapi import Request, Response

# Upstream Vals finance-agent-v2 values (installed via requirements.txt, pinned
# to the same commit as the resource server). Imported rather than copied so a
# bump cannot change them without the parity tests noticing.
from finance_agent.exceptions import RetryExhaustedError
from finance_agent.get_agent import MAX_TIME_SECONDS
from finance_agent.tools import VALID_TOOLS, SubmitFinalResult
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
)
from nemo_gym.server_utils import get_response_json, raise_for_status


logger = logging.getLogger(__name__)

_MODEL_OUTPUT_TYPES = frozenset({"reasoning", "function_call"})

# ---------------------------------------------------------------------------
# Upstream constants
# ---------------------------------------------------------------------------

#: Tool whose invocation ends the run. Upstream appends ``SubmitFinalResult()``
#: unconditionally in ``get_agent`` and stops when a tool returns ``done=True``.
#: ``Tool.name`` is a class attribute, so no instantiation is needed.
UPSTREAM_DONE_TOOL: str = SubmitFinalResult.name

#: Upstream v2 bounds the run by wall clock (1 hour) with no turn cap, where v1
#: used ``max_turns=50`` and no time limit.
UPSTREAM_MAX_TIME_SECONDS: int = MAX_TIME_SECONDS

#: Tool names upstream considers valid. Recorded for the parity test; the set
#: actually offered to the model is decided by the resource server.
UPSTREAM_VALID_TOOLS: tuple[str, ...] = tuple(VALID_TOOLS)

#: Upstream's ``_on_tool_result`` hook re-raises this, aborting the run, where a
#: plain tool failure is fed back to the model and the loop continues.
UPSTREAM_ABORT_TOOL_ERRORS: tuple[str, ...] = (RetryExhaustedError.__name__,)

#: Injected when a turn produces prose and no tool call. Not importable: upstream
#: builds it inline inside ``get_agent._before_query``. v1 used ``"Continue."``.
UPSTREAM_NO_TOOL_CALL_NUDGE: str = (
    "Your last response produced no tool call. "
    "Call `submit_final_result` if you have a final result, "
    "otherwise continue with the next tool call."
)

# Regex that matches common vLLM / OpenAI context-length error messages.
# Stands in for upstream's typed ``MaxContextWindowExceededError``, which Gym
# cannot see because the model is reached over HTTP rather than through a
# model_library provider client.
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


class StopReason(str, Enum):
    """Why the loop ended.

    Mirrors ``model_library.agent.AgentStopReason`` so a zero from "never
    submitted" is distinguishable from a zero from "judged wrong". Upstream's
    ``SHOULD_STOP`` is omitted because v2 pins ``_should_stop`` to False, so a
    text-only turn can never end the run.
    """

    DONE_TOOL = "done_tool"
    MAX_TURNS = "max_turns"
    MAX_TIME = "max_time"
    MAX_OUTPUT_TOKENS = "max_output_tokens"
    ERROR = "error"


class FinanceAgentV2Config(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: Optional[int] = Field(
        default=None,
        description="Maximum model turns. Upstream v2 sets no turn cap "
        "(max_turns=None) and bounds the run by time instead, so None is the "
        "faithful default; set it to bound cost in a smoke run.",
    )
    max_time_seconds: Optional[float] = Field(
        default=float(UPSTREAM_MAX_TIME_SECONDS),
        description="Wall-clock budget for the loop, checked before each model "
        "call. Mirrors upstream v2's TimeLimit(max_seconds=3600). None disables "
        "the budget. This is the agent's hard stop; the resource server's "
        "max_rollout_time_seconds is a softer budget that only makes tools "
        "return an error asking the model to submit.",
    )
    done_tools: List[str] = Field(
        default_factory=lambda: [UPSTREAM_DONE_TOOL],
        description="Tool names that signal the agent loop should terminate. "
        "When any tool call in a batch matches, remaining calls are skipped "
        "and the loop exits.",
    )
    no_tool_call_nudge: str = Field(
        default=UPSTREAM_NO_TOOL_CALL_NUDGE,
        description="Injected as a user message when a turn produces prose and "
        "no tool call, so the loop continues instead of stopping. Upstream v2 "
        "changed this text from v1's bare 'Continue.'.",
    )
    abort_on_tool_error_types: List[str] = Field(
        default_factory=lambda: list(UPSTREAM_ABORT_TOOL_ERRORS),
        description="Exception type names in a tool's error payload that abort "
        "the whole rollout instead of being fed back to the model. Mirrors "
        "upstream v2's on_tool_result hook re-raising RetryExhaustedError.",
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


class FinanceAgentV2RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class FinanceAgentV2VerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class FinanceAgentV2VerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class FinanceAgentV2(SimpleResponsesAPIAgent):
    config: FinanceAgentV2Config

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

        while i < n and FinanceAgentV2._is_model_output(outputs[i]):
            i += 1

        while i < n and not FinanceAgentV2._is_model_output(outputs[i]):
            i += 1

        if i >= n:
            return outputs

        return outputs[i:]

    def _aborting_error_type(self, tool_output: str) -> Optional[str]:
        """Return the abort-worthy exception type named in *tool_output*, if any.

        The resource server renders an uncaught tool exception as
        ``{"error": "<TypeName>: <message>"}``, which is how upstream's typed
        ``on_tool_result`` hook surfaces here. Only the ``error`` field is
        inspected so a tool that merely echoes the name in its payload cannot
        abort the run.
        """
        if not self.config.abort_on_tool_error_types:
            return None
        try:
            payload = json.loads(tool_output)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        error = payload.get("error")
        if not isinstance(error, str):
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
            # Upstream checks the time limit before issuing the turn's query, so
            # an exhausted budget ends the run rather than paying for one more
            # call. Upstream subtracts retry overhead from the elapsed reading;
            # this loop does not retry, so wall clock is the same measurement.
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

            if model_response.incomplete_details and model_response.incomplete_details.reason == "max_output_tokens":
                stop_reason = StopReason.MAX_OUTPUT_TOKENS
                break

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]

            if not all_fn_calls and all_output_messages:
                # Upstream pins should_stop to False and nudges from
                # _before_query instead, so prose between tool calls does not
                # end the run. v2 reworded the nudge to name submit_final_result.
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
        # a tool abort — and under dealbreaker-gated scoring a truncated trajectory
        # scores exactly like a confidently wrong one. ``metadata`` is the Responses
        # API's own string-keyed side channel, so it survives verify() untouched.
        last_model_response.metadata = {
            **(last_model_response.metadata or {}),
            "stop_reason": stop_reason.value,
            "steps": str(step),
        }
        return last_model_response

    async def run(self, request: Request, body: FinanceAgentV2RunRequest) -> FinanceAgentV2VerifyResponse:
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
            return FinanceAgentV2VerifyResponse(
                responses_create_params=body.responses_create_params,
                response=empty_response,
                reward=0.0,
            )

    async def _run_inner(self, request: Request, body: FinanceAgentV2RunRequest) -> FinanceAgentV2VerifyResponse:
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

        verify_request = FinanceAgentV2VerifyRequest.model_validate(
            body.model_dump() | {"response": await get_response_json(response)}
        )

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return FinanceAgentV2VerifyResponse.model_validate(await get_response_json(verify_response))

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
    FinanceAgentV2.run_webserver()
