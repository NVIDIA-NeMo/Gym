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
"""Multi-turn tool-calling loop for finance research benchmarks.

One loop serves all benchmark profiles. Everything that differs between them is a
config value, not a branch in this file: the nudge injected on a prose-only
turn, whether the run is bounded by turns or by wall clock, and which tool
errors abort a rollout instead of being handed back to the model. Those three
fields are required, so a config states its policy rather than inheriting a
default that happens to suit another benchmark. Additional policies select
whether prose ends a rollout and whether a tool-call batch runs concurrently.

The values the Vals profiles use are pinned against the upstream package by tests
in ``resources_servers/finance_agent_v2/tests``, which is where that package is
installed.

Two callers run the loop. An Agent's legacy ``/run`` seeds the resources
server, calls the loop and verifies, turning any exception into reward 0. An
Environment Server (``single_agent_turn``) seeds an agent session that carries a
direct HTTP grant to the resources server's tools, calls ``/v1/responses`` once,
closes the session and verifies. In a session, a failed model call marks the
response ``failed`` with its cause, so the Environment Server decides whether
to grade, retry or fail the episode instead of scoring it 0.
"""

import asyncio
import json
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Literal, Optional

from aiohttp import ClientError, ClientResponseError
from fastapi import Request, Response
from openai.types.responses import ResponseError
from pydantic import ConfigDict, Field

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentOutcome,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.client import request as http_request
from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest, ModelServerRef, ResourcesServerRef
from nemo_gym.core import (
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    accumulate_response_usage,
)
from nemo_gym.server_runtime import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    DirectHTTPToolAccess,
    MCPToolAccess,
)
from nemo_gym.trajectory import AgentInvocation, AgentObservationBundle, ModelCallRef, TrajectoryToolCall


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


def _cookie_values(cookies: Any) -> dict[str, str]:
    """Flatten a plain dict or an aiohttp cookie jar of morsels into name-to-value pairs."""
    return {str(name): str(getattr(value, "value", value)) for name, value in (cookies or {}).items()}


def _decoded_object(raw: str) -> Optional[dict]:
    """JSON-decode *raw*, returning None unless it decodes to an object."""
    try:
        value = json.loads(raw)
    except (ValueError, RecursionError, TypeError):
        return None
    return value if isinstance(value, dict) else None


class StopReason(str, Enum):
    """Why the loop ended.

    Recorded on the rollout so a zero from "never submitted" is distinguishable
    from a zero from "judged wrong".
    """

    DONE_TOOL = "done_tool"
    ASSISTANT_MESSAGE = "assistant_message"
    MAX_TURNS = "max_turns"
    MAX_TIME = "max_time"
    MAX_OUTPUT_TOKENS = "max_output_tokens"
    TEXT_ONLY = "text_only"
    ERROR = "error"


class FinanceAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    no_tool_call_nudge: str = Field(
        ...,
        description="Injected as a user message when a turn produces prose and "
        "no tool call, so the loop continues instead of stopping.",
    )
    prose_only_behavior: Literal["nudge", "finish"] = Field(
        default="nudge",
        description="What to do when a turn contains no tool call. "
        "'nudge' injects no_tool_call_nudge after assistant prose and continues; "
        "'finish' ends the run even when the response has no text. "
        "continue_if_not_tool_call=False takes precedence over this policy.",
    )
    tool_call_execution: Literal["sequential", "concurrent"] = Field(
        default="sequential",
        description="Whether tool calls emitted in one model turn execute in "
        "model order or concurrently. Concurrent results are still appended "
        "in the original call order.",
    )
    tool_error_observation: Literal["json", "error_prefix"] = Field(
        default="json",
        description="How failed tool calls are shown to the model. 'json' keeps "
        "the shared loop's legacy JSON payload; 'error_prefix' renders the "
        "'[ERROR] <message>' observation.",
    )
    raise_on_tool_http_error: bool = Field(
        default=False,
        description="Convert unsuccessful tool HTTP responses to error observations. "
        "False preserves the legacy behavior of forwarding their response bodies.",
    )
    done_tool_requires_success: bool = Field(
        default=False,
        description="Require a successful terminal tool result before ending either execution mode. "
        "False preserves termination whenever a terminal tool is invoked.",
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
        "Sequential execution skips later calls when a terminal tool is "
        "invoked; concurrent execution finishes the full batch. "
        "done_tool_requires_success controls whether a failed terminal call can end either mode.",
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
    continue_if_not_tool_call: bool = Field(
        default=True,
        description="When True (Gym eval default), a turn with assistant text and "
        "no tool call injects no_tool_call_nudge and the loop continues. When False "
        "(GRPO overlay), that turn ends the episode so a synthetic user message "
        "cannot break NeMo-RL monotonic trajectories.",
    )


class FinanceAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class FinanceAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class FinanceAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


@dataclass
class FinanceAgentSessionState(AgentSessionState):
    """An Environment Server's agent session: its grant to the resources server's tools and their cookie jar."""

    tool_access: DirectHTTPToolAccess
    # Seeded from the grant, updated by every tool call, and returned at close.
    resources_cookies: dict[str, str]
    observations: Optional[AgentObservationBundle] = None


@dataclass
class _ToolCallResult:
    """One executed tool call: the observation text, the cookies its reply set (None without a reply), the
    explicit tool-error flag, and the status and executor-side timing a session records."""

    output: str
    cookies: Any
    is_error: Optional[bool]
    status: str
    error_type: Optional[str]
    started_at: float
    completed_at: float


class FinanceAgent(SimpleResponsesAPIAgent):
    config: FinanceAgentConfig

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> FinanceAgentSessionState:
        """Take the session's one direct HTTP grant; every tool, submit_final_result included, is behind it."""
        if body.sandbox_access is not None:
            raise AgentSeedRejectedError("Finance Agent does not support sandbox access")
        accesses = self.effective_tool_accesses(body)
        unsupported = [access.name for access in accesses if isinstance(access, MCPToolAccess) and access.required]
        if unsupported:
            raise AgentSeedRejectedError(
                f"Finance Agent does not support required MCP tool access: {', '.join(unsupported)}"
            )
        direct_accesses = [access for access in accesses if isinstance(access, DirectHTTPToolAccess)]
        if len(direct_accesses) != 1:
            raise AgentSeedRejectedError(
                "Finance Agent needs exactly one direct HTTP tool access to its resources server's tools, "
                f"got {len(direct_accesses)}; set resources_tool_transports: [direct_http] on the Environment Server"
            )
        return FinanceAgentSessionState(
            request=body,
            tool_access=direct_accesses[0],
            resources_cookies=dict(direct_accesses[0].cookies),
        )

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, FinanceAgentSessionState):
            raise TypeError("Expected Finance Agent session state")
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=state.observations,
            resources_cookies=state.resources_cookies,
        )

    def classify_outcome(self, native_result: Mapping[str, Any]) -> AgentOutcome:
        """Name the cause of the model call failure or tool abort that ended a session's loop.

        The Environment Server's outcome policy decides from it: by default a 400, 413 or 422 is graded, a
        retryable status or a transport failure dispatches the episode again, and any other status fails it.
        A tool abort (``abort_on_tool_error_types``) is a dependency outage no registered kind names, so it is
        ``unknown``, which the policy dispatches again.
        """
        if native_result.get("tool_abort"):
            return AgentOutcome(failure_kind="unknown")
        error = native_result.get("error")
        if isinstance(error, ClientResponseError) and 400 <= error.status <= 599:
            return AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=error.status)
        if isinstance(error, (asyncio.TimeoutError, ClientError)):
            return AgentOutcome(failure_kind=MODEL_TRANSPORT_ERROR)
        return AgentOutcome(failure_kind="unknown")

    @staticmethod
    def _is_context_overflow_error(exc: Exception) -> bool:
        """True when *exc* indicates the input exceeded the model's context window."""
        return _CONTEXT_OVERFLOW_RE.search(str(exc)) is not None

    @classmethod
    def _is_context_overflow_failure(cls, exc: BaseException) -> bool:
        """True when a failed model call says the input exceeded the model's context window.

        openai_model forwards a provider's 400 as a 500 by default, so the provider's text is only in the reply
        body, which ``raise_for_status`` keeps on ``response_content``; ``str(exc)`` names only the status.
        """
        if isinstance(exc, Exception) and cls._is_context_overflow_error(exc):
            return True
        content = getattr(exc, "response_content", None)
        if isinstance(content, bytes):
            content = content.decode(errors="replace")
        return isinstance(content, str) and _CONTEXT_OVERFLOW_RE.search(content) is not None

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
    def _tool_error_message(tool_output: str, *, is_error: Optional[bool] = None) -> Optional[str]:
        """Use explicit tool status when available, falling back to the legacy payload.

        Two shapes arrive here. A reply from the resource server is wrapped as
        ``{"results": "<payload>"}`` where the payload is itself a JSON string,
        so an error is double-encoded and the inner object must be decoded
        separately. A tool call that never reached the server (timeout,
        transport failure) is rendered locally as a flat ``{"error": ...}``.

        Explicit status takes precedence over the payload, so successful tool
        text can contain an ``error`` field without being classified as a failure.
        """
        if is_error is not None:
            return tool_output if is_error else None
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

    def _aborting_error_type(self, tool_output: str, *, is_error: Optional[bool] = None) -> Optional[str]:
        """Return the abort-worthy exception type named in *tool_output*, if any."""
        if not self.config.abort_on_tool_error_types:
            return None
        error = self._tool_error_message(tool_output, is_error=is_error)
        if error is None:
            return None
        return next(
            (name for name in self.config.abort_on_tool_error_types if error.startswith(f"{name}:")),
            None,
        )

    def _render_tool_observation(self, tool_output: str, *, is_error: Optional[bool] = None) -> str:
        """Render an error for the selected harness without changing success output."""
        if self.config.tool_error_observation != "error_prefix":
            return tool_output
        error = self._tool_error_message(tool_output, is_error=is_error)
        if error is not None:
            return f"[ERROR] {error}"
        return tool_output

    async def _execute_tool_call(
        self,
        output_function_call: NeMoGymResponseFunctionToolCall,
        cookies: Any,
        state: Optional[FinanceAgentSessionState] = None,
    ) -> _ToolCallResult:
        """Execute one tool call and render failures as tool outputs.

        A session reaches the tools through its direct HTTP grant; the legacy path goes through the server client.
        """
        started_at = time.time()
        response_cookies = None
        is_error = None
        status, error_type = "completed", None
        try:
            if state is not None:
                coro = http_request(
                    method="POST",
                    url=f"{str(state.tool_access.base_url).rstrip('/')}/{output_function_call.name}",
                    json=json.loads(output_function_call.arguments),
                    cookies=cookies,
                    headers=dict(state.tool_access.headers),
                    _internal=True,
                )
            else:
                coro = self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{output_function_call.name}",
                    json=json.loads(output_function_call.arguments),
                    cookies=cookies,
                )
            api_response = await asyncio.wait_for(coro, timeout=self.config.tool_call_timeout)
            response_cookies = _cookie_values(api_response.cookies) if state is not None else api_response.cookies
            if self.config.raise_on_tool_http_error:
                await raise_for_status(api_response)
            tool_output = (await api_response.content.read()).decode()
            error_header = api_response.headers.get("x-nemo-gym-tool-error")
            if error_header in ("true", "false"):
                is_error = error_header == "true"
            if state is not None and not 200 <= api_response.status < 400:
                status, error_type = "failed", f"http_{api_response.status}"
        except asyncio.TimeoutError:
            status, error_type = "timeout", "TimeoutError"
            logger.warning(
                "Tool call '%s' timed out after %ss",
                output_function_call.name,
                self.config.tool_call_timeout,
            )
            error = (
                f"Tool call timed out after {self.config.tool_call_timeout}s. "
                "Please try a different approach or submit your final answer."
            )
            tool_output = (
                error if self.config.tool_error_observation == "error_prefix" else json.dumps({"error": error})
            )
            is_error = True if self.config.tool_error_observation == "error_prefix" else None
        except Exception as e:
            # A raised HTTP error is recorded as the status it carries, as an unraised one is.
            http_status = e.status if isinstance(e, ClientResponseError) else None
            status, error_type = "failed", f"http_{http_status}" if http_status else type(e).__name__
            logger.error(
                "Tool call '%s' failed: %s: %s",
                output_function_call.name,
                type(e).__name__,
                e,
            )
            error = (
                f"Tool call failed: {type(e).__name__}: {e}. "
                "Please try a different approach or submit your final answer."
            )
            tool_output = (
                error if self.config.tool_error_observation == "error_prefix" else json.dumps({"error": error})
            )
            is_error = True if self.config.tool_error_observation == "error_prefix" else None
        return _ToolCallResult(
            output=tool_output,
            cookies=response_cookies,
            is_error=is_error,
            status=status,
            error_type=error_type,
            started_at=started_at,
            completed_at=max(started_at, time.time()),
        )

    @staticmethod
    def _tool_record(call: NeMoGymResponseFunctionToolCall, result: _ToolCallResult) -> TrajectoryToolCall:
        """The session's observation of one executed tool call."""
        return TrajectoryToolCall(
            invocation_id="root",
            tool_call_id=call.call_id,
            tool_name=call.name,
            started_at=result.started_at,
            completed_at=result.completed_at,
            duration_ms=(result.completed_at - result.started_at) * 1000,
            timing_source="executor",
            status=result.status,
            error_type=result.error_type,
            output=result.output,
        )

    @staticmethod
    def _merge_cookies(cookies: Any, updates: List[Any]) -> Any:
        """Merge concurrent response cookies deterministically in call order."""
        merged = dict(cookies.items())
        for update in updates:
            if update is not None:
                merged.update(update.items())
        return merged

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        agent_session_id = self._agent_session_id_from_request(request)
        state = self._require_agent_session(agent_session_id) if agent_session_id is not None else None
        if state is not None and not isinstance(state, FinanceAgentSessionState):
            raise TypeError("Expected Finance Agent session state")
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        # A session returns the loop's evidence at close; it is collected only when model calls are captured too.
        collect_observations = state is not None and self._model_call_capture_enabled() and isinstance(rollout_id, str)
        model_calls: List[ModelCallRef] = []
        tool_records: List[TrajectoryToolCall] = []
        loop_started_at = time.time()
        model_failure: Optional[BaseException] = None
        # The exception type a tool reported that aborted the loop (abort_on_tool_error_types), if any.
        abort_error_type: Optional[str] = None

        new_outputs: List[Any] = []
        tool_call_error_status: dict[str, bool] = {}
        final_model_turn_start = 0
        usage = None
        step = 0
        last_model_response: Optional[NeMoGymResponse] = None
        model_server_cookies = None
        # A session reaches the resources server only through its grant, starting from the cookie jar it carries.
        resources_server_cookies = state.resources_cookies if state is not None else request.cookies

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
                    # Carries an Environment Server's rollout prefix, so captured model calls join the episode.
                    url_path=self.url_path_for_request("/v1/responses", request),
                    json=new_body,
                    cookies=model_server_cookies,
                    **({"headers": {"x-session-id": "root"}} if collect_observations else {}),
                )
                model_resp_raw = await asyncio.wait_for(coro, timeout=self.config.model_call_timeout)

                await raise_for_status(model_resp_raw)
                model_response_json = await get_response_json(model_resp_raw)
                model_server_cookies = model_resp_raw.cookies
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except asyncio.TimeoutError as e:
                logger.warning(
                    "Model call timed out after %ss on step %d — terminating agent loop",
                    self.config.model_call_timeout,
                    step,
                )
                stop_reason = StopReason.ERROR
                model_failure = e
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
                model_failure = e
                break

            output = model_response.output
            last_model_response = model_response
            final_model_turn_start = len(new_outputs)
            new_outputs.extend(output)
            if collect_observations and model_response.id:
                model_calls.append(ModelCallRef(model_ref=self.config.model_server, response_id=model_response.id))

            usage = accumulate_response_usage(usage, model_response.usage)

            if model_response.incomplete_details and model_response.incomplete_details.reason == "max_output_tokens":
                stop_reason = StopReason.MAX_OUTPUT_TOKENS
                break

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]

            if not all_fn_calls:
                if all_output_messages and not self.config.continue_if_not_tool_call:
                    logger.info(
                        "Text-only turn on step %d with continue_if_not_tool_call=false — terminating agent loop",
                        step,
                    )
                    stop_reason = StopReason.TEXT_ONLY
                    break
                if self.config.prose_only_behavior == "finish":
                    stop_reason = StopReason.ASSISTANT_MESSAGE
                    break
            if not all_fn_calls and all_output_messages:
                logger.info(
                    "Text-only turn on step %d — injecting no-tool-call nudge",
                    step,
                )
                new_outputs.append(NeMoGymEasyInputMessage(role="user", content=self.config.no_tool_call_nudge))
                continue

            done = False
            if self.config.tool_call_execution == "concurrent":
                results = await asyncio.gather(
                    *(self._execute_tool_call(call, resources_server_cookies, state) for call in all_fn_calls)
                )
                resources_server_cookies = self._merge_cookies(
                    resources_server_cookies,
                    [result.cookies for result in results],
                )
                for output_function_call, result in zip(all_fn_calls, results):
                    if result.is_error is not None:
                        tool_call_error_status[output_function_call.call_id] = result.is_error
                    new_outputs.append(
                        NeMoGymFunctionCallOutput(
                            type="function_call_output",
                            call_id=output_function_call.call_id,
                            output=self._render_tool_observation(result.output, is_error=result.is_error),
                        )
                    )
                    if collect_observations:
                        tool_records.append(self._tool_record(output_function_call, result))

                # All calls in a concurrent batch run. An accepted terminal
                # call takes precedence over other failures in that same batch.
                done = any(
                    call.name in done_tools_set
                    and (
                        not self.config.done_tool_requires_success
                        or self._tool_error_message(result.output, is_error=result.is_error) is None
                    )
                    for call, result in zip(all_fn_calls, results)
                )
                if done:
                    stop_reason = StopReason.DONE_TOOL
                else:
                    abort_error_type = next(
                        (
                            error_type
                            for result in results
                            if (error_type := self._aborting_error_type(result.output, is_error=result.is_error))
                        ),
                        None,
                    )
                    if abort_error_type is not None:
                        stop_reason = StopReason.ERROR

                if done or abort_error_type is not None:
                    break
                continue

            for output_function_call in all_fn_calls:
                result = await self._execute_tool_call(output_function_call, resources_server_cookies, state)
                if state is not None:
                    # A reply that sets no cookie leaves the session's jar as it was.
                    resources_server_cookies = self._merge_cookies(resources_server_cookies, [result.cookies])
                elif result.cookies is not None:
                    resources_server_cookies = result.cookies
                if result.is_error is not None:
                    tool_call_error_status[output_function_call.call_id] = result.is_error

                tool_response = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=self._render_tool_observation(result.output, is_error=result.is_error),
                )
                new_outputs.append(tool_response)
                if collect_observations:
                    tool_records.append(self._tool_record(output_function_call, result))

                if output_function_call.name in done_tools_set and (
                    not self.config.done_tool_requires_success
                    or self._tool_error_message(result.output, is_error=result.is_error) is None
                ):
                    logger.info(
                        "Tool '%s' signaled done — terminating agent loop",
                        output_function_call.name,
                    )
                    done = True
                    stop_reason = StopReason.DONE_TOOL
                    break

                if abort_error_type := self._aborting_error_type(result.output, is_error=result.is_error):
                    logger.error(
                        "Tool '%s' raised %s — aborting rollout",
                        output_function_call.name,
                        abort_error_type,
                    )
                    stop_reason = StopReason.ERROR
                    break

            if done or abort_error_type is not None:
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

        if state is None:
            # Legacy self-dispatch carries the resources cookies on to its /verify call.
            cookie_items = list(resources_server_cookies.items())
        else:
            # A session returns its resources cookie jar at close, where the Environment Server takes it.
            state.resources_cookies = dict(resources_server_cookies)
            cookie_items = []
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
        if tool_call_error_status:
            # Keep explicit provenance for stateless reverification without
            # adding fields to the shared Responses API tool-output type.
            last_model_response.metadata["tool_call_error_status"] = json.dumps(tool_call_error_status)
        if stop_reason is StopReason.ASSISTANT_MESSAGE:
            last_model_response.metadata["final_model_turn_start"] = str(final_model_turn_start)
        invocation_status = {
            StopReason.DONE_TOOL: "completed",
            StopReason.TEXT_ONLY: "completed",
            StopReason.ASSISTANT_MESSAGE: "completed",
            StopReason.ERROR: "failed",
        }.get(stop_reason, "incomplete")
        invocation_error_type = type(model_failure).__name__ if model_failure is not None else abort_error_type
        if state is not None and model_failure is not None and self._is_context_overflow_failure(model_failure):
            # The model's own context ran out, so the episode is graded, as a legacy /run grades it and hermes_agent
            # grades its context limit, instead of being dispatched again until a shorter attempt fits.
            last_model_response.status = "incomplete"
            last_model_response.error = None
            invocation_status, invocation_error_type = "incomplete", None
        elif state is not None and model_failure is not None:
            # A legacy /run verifies whatever the loop returns. In a session the Environment Server decides from the
            # cause whether to grade the episode, dispatch it again or fail it, instead of scoring an outage as 0.
            last_model_response.status = "failed"
            last_model_response.error = ResponseError(
                code="server_error",
                message=f"Model call failed on step {step}: {type(model_failure).__name__}: {model_failure}"[:2000],
            )
            last_model_response = self.report_outcome(last_model_response, {"error": model_failure})
        elif state is not None and abort_error_type is not None:
            # Upstream ends the question with an empty answer, and a legacy /run grades what the loop returned. An
            # abort type names an outage the model cannot cause (v2: SEC or sec.gov still answering 429/503 after
            # retries), so a session leaves it out of the score on purpose and the default policy dispatches it again.
            last_model_response.status = "failed"
            last_model_response.error = ResponseError(
                code="server_error",
                message=f"A tool call raised {abort_error_type} on step {step}, which aborts the rollout",
            )
            last_model_response = self.report_outcome(last_model_response, {"tool_abort": abort_error_type})
        if collect_observations:
            state.observations = AgentObservationBundle(
                source="finance_agent",
                records=[
                    AgentInvocation(
                        invocation_id="root",
                        status=invocation_status,
                        stop_reason=stop_reason.value,
                        duration_ms=max(0.0, time.time() - loop_started_at) * 1000,
                        error_type=invocation_error_type,
                        model_calls=model_calls,
                        conversation=[*body.input, *new_outputs],
                    ),
                    *tool_records,
                ],
            )
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
