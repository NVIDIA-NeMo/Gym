# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools
import inspect
import json
import logging
import re
import traceback
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from http.cookiejar import CookieJar
from time import time
from typing import Any, Optional
from uuid import uuid4

import verifiers as vf
from fastapi import Body, Request, Response
from openai import (
    DEFAULT_TIMEOUT,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
    Timeout,
)
from pydantic import ConfigDict, Field, field_validator
from verifiers.clients import NeMoRLChatCompletionsClient

from nemo_gym.agents import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.artifacts import NG_NO_GENERATION_METADATA
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
    VERIFIER_ERROR,
    FailureBoundary,
    FailureHeader,
    FailureReportRow,
    ModelFailureDetails,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseFunctionToolCallForTraining,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
    NgFailureEvent,
    OutcomeAction,
    StageOutcomePolicyConfig,
    accumulate_response_usage,
    choose_stage_outcome,
    current_model_sampling_segment,
    failure_report_for,
    observe_failure,
    rebuild_failure,
    sdk_error_details,
)
from nemo_gym.server_runtime import (
    BaseRunRequest,
    BaseVerifyResponse,
    is_retryable_dependency_status,
    observe_verifier,
)
from nemo_gym.trajectory import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    TrajectoryRecord,
    TrajectoryTurn,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


logger = logging.getLogger(__name__)

_EMPTY_OUTPUT_ITEM_ID = "msg_empty"


def _as_dict(msg: Any) -> dict:
    if isinstance(msg, dict):
        return msg
    return {k: getattr(msg, k, None) for k in ("role", "content", "tool_calls", "tool_call_id", "tokens")}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, default=str)


def _tok_kwargs(tokens: dict | None) -> dict:
    if not tokens:
        return {}
    kwargs = {
        "prompt_token_ids": tokens.get("prompt_ids", []),
        "generation_token_ids": tokens.get("completion_ids", []),
        "generation_log_probs": tokens.get("completion_logprobs", []),
    }
    routed_experts = tokens.get("routed_experts")
    if routed_experts is not None:
        kwargs["routed_experts"] = routed_experts
    return kwargs


def _normalize_tool_call(tool_call: Any) -> dict:
    if isinstance(tool_call, str):
        try:
            return json.loads(tool_call)
        except json.JSONDecodeError:
            return {"arguments": tool_call}
    return tool_call


def _build_tool_result_item(msg: dict, raw: Any) -> dict:
    call_id = msg.get("tool_call_id") or f"call_{id(raw)}"
    return NeMoGymFunctionCallOutput(
        call_id=call_id,
        id=call_id,
        output=_text(msg.get("content")),
        status="completed",
    ).model_dump()


def _build_function_call_item(tool_call: Any, tokens: dict | None) -> dict:
    tc = _normalize_tool_call(tool_call)
    function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
    call_id = tc.get("id") or tc.get("call_id") or f"call_{id(tool_call)}"
    name = tc.get("name") or function.get("name", "")
    arguments = _text(tc.get("arguments") or function.get("arguments") or "{}")

    cls = NeMoGymResponseFunctionToolCallForTraining if tokens else NeMoGymResponseFunctionToolCall
    return cls(
        id=call_id,
        call_id=call_id,
        name=name,
        arguments=arguments,
        status="completed",
        **_tok_kwargs(tokens),
    ).model_dump()


def _build_message_item(raw: Any, body: str, tokens: dict | None) -> dict:
    cls = NeMoGymResponseOutputMessageForTraining if tokens else NeMoGymResponseOutputMessage
    return cls(
        id=f"msg_{id(raw)}",
        content=[NeMoGymResponseOutputText(text=body, annotations=[])],
        **_tok_kwargs(tokens),
    ).model_dump()


def _assistant_turn_groups(output: list[dict]) -> list[list[dict]]:
    """Cut the flat output back into one group per assistant turn.

    `_build_assistant_items` emits an optional message followed by that turn's
    function calls, and a tool result always separates two assistant turns, so
    a group runs until the next tool result or the next message. The synthetic
    `msg_empty` placeholder is not an agent turn and is skipped -- counting it
    would report a hollow turn on every rollout served by an endpoint that
    returns no token ids.
    """
    groups: list[list[dict]] = []
    current: Optional[list[dict]] = None
    for item in output:
        item_type = item.get("type")
        if item_type not in ("message", "function_call") or item.get("id") == _EMPTY_OUTPUT_ITEM_ID:
            current = None
            continue
        if current is None or item_type == "message":
            current = []
            groups.append(current)
        current.append(item)
    return groups


def _build_assistant_items(msg: dict, raw: Any, tokens: dict | None) -> list[dict]:
    tool_calls = msg.get("tool_calls") or []
    body = _text(msg.get("content"))
    items: list[dict] = []

    if body:
        items.append(_build_message_item(raw, body, tokens if not tool_calls else None))

    for i, tool_call in enumerate(tool_calls):
        is_last = i == len(tool_calls) - 1
        items.append(_build_function_call_item(tool_call, tokens if is_last else None))

    if not items:
        items.append(_build_message_item(raw, "", tokens))

    return items


def _as_response_usage(usage: Any) -> Optional[NeMoGymResponseUsage]:
    """Map a chat-completions `usage` onto the Responses shape the rollout record uses."""
    if usage is None:
        return None
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return NeMoGymResponseUsage(
        input_tokens=usage.prompt_tokens,
        input_tokens_details=NeMoGymResponseInputTokensDetails(
            cached_tokens=getattr(prompt_details, "cached_tokens", None) or 0
        ),
        output_tokens=usage.completion_tokens,
        output_tokens_details=NeMoGymResponseOutputTokensDetails(
            reasoning_tokens=getattr(completion_details, "reasoning_tokens", None) or 0
        ),
        total_tokens=usage.total_tokens,
    )


@dataclass
class _ModelCallOutcome:
    """How the model calls of one rollout last ended: the failure the latest call raised, else None."""

    last_failure: Optional[NgFailureEvent] = None


# Set by ``run`` around one rollout; the clients record into it because verifiers keeps only a serialized error.
_MODEL_CALL_OUTCOME: ContextVar[Optional[_ModelCallOutcome]] = ContextVar(
    "verifiers_agent_model_call_outcome", default=None
)


def _model_call_failure(error: Exception) -> Optional[NgFailureEvent]:
    """The model boundary's failure for an error of the policy client's SDK, from what the SDK reports; else None.

    An HTTP reply names its status and nothing else, and a call that got no reply is a transport failure whose
    delivery is unknown: neither is classified from message text, and no policy is applied here. The SDK's text is
    not copied, only its type and status. A Gym server's own failure report on the reply is linked as the cause.
    """
    if isinstance(error, APIStatusError):
        kind, status, upload = MODEL_REQUEST_ERROR, error.status_code, True
        reason = f"HTTP {status} from the model server"
    elif isinstance(error, APIConnectionError):
        kind, status, upload = MODEL_TRANSPORT_ERROR, None, None
        reason = "Model request timed out" if isinstance(error, APITimeoutError) else "Model request transport failed"
    else:
        return None
    failure = NgFailureEvent(
        boundary=FailureBoundary.MODEL,
        failure_kind=kind,
        failure_reason=reason,
        operation="model_request",
        stage="agent",
        http_status=status,
        details=ModelFailureDetails(request_upload_complete=upload),
        foreign_error=sdk_error_details(error, message=""),
    )
    forwarded = FailureHeader.decode(getattr(getattr(error, "response", None), "headers", None))
    failure.__cause__ = rebuild_failure(forwarded) or error
    return failure


class _ModelFailureObservingClient(NeMoRLChatCompletionsClient):
    """Records each policy-model call that fails in the SDK as a failure event.

    verifiers wraps the SDK error in its own and keeps only a serialized form on the rollout, so the SDK error is
    only available here, where the call is made.
    """

    async def get_native_response(self, *args: Any, **kwargs: Any) -> Any:
        outcome = _MODEL_CALL_OUTCOME.get()
        if outcome is not None:
            outcome.last_failure = None
        try:
            return await super().get_native_response(*args, **kwargs)
        except Exception as error:
            failure = _model_call_failure(error)
            if failure is not None:
                observe_failure(failure)
                if outcome is not None:
                    outcome.last_failure = failure
            raise


class _ResponseIdRecordingClient(_ModelFailureObservingClient):
    """Records the id of every model response verifiers receives.

    `simple_agent` builds its `ModelCallRef`s from `model_response.id` because it
    issues the calls itself. Here `verifiers` owns the call loop and hands back
    only a rollout summary, so the agent never sees a response. The client is the
    one seam it does own: every call passes through `get_native_response`, so
    recording ids here yields the same references without verifiers having to
    expose anything.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.recorded_response_ids: list[str] = []
        self.recorded_usage: Optional[NeMoGymResponseUsage] = None

    async def get_native_response(self, *args: Any, **kwargs: Any) -> Any:
        response = await super().get_native_response(*args, **kwargs)
        response_id = getattr(response, "id", None)
        if isinstance(response_id, str) and response_id:
            self.recorded_response_ids.append(response_id)
        self.recorded_usage = accumulate_response_usage(
            self.recorded_usage, _as_response_usage(getattr(response, "usage", None))
        )
        return response


# Names in a rollout error's class chain (verifiers.errors and the transport errors they wrap) and the kind each
# suggests, first match wins: a transport timeout or connection error inside a ModelError or a ToolError is named
# as such. A candidate only: it is recorded for counting and never decides a score, a mask or a retry.
_ROLLOUT_ERROR_KIND_CANDIDATES = (
    ("OverlongPromptError", "overlong_prompt"),
    ("Timeout", "timeout"),
    ("Connection", "connection"),
    ("InfraError", "infra"),
    ("SandboxError", "infra"),
    ("TunnelError", "infra"),
    ("ToolError", "tool"),
    ("ModelResponseError", "model_response"),
    ("ModelError", "model_response"),
)
_STATUS_CODE = re.compile(r"(?:Error code|status(?:_code)?)\s*[:=]?\s*(\d{3})\b", re.IGNORECASE)
_ROLLOUT_ERROR_MESSAGE_CHARS = 500


def _rollout_error_fact(error: Any) -> Optional[dict[str, Any]]:
    """The error a verifiers rollout ended on, as a typed fact for the row: ``type`` (the raised class),
    ``chain`` (the class chain, outermost first), ``status_code`` (an HTTP status named in the error, if any),
    ``kind_candidate`` (``http_5xx``, ``http_429``, ``http_4xx``, then by class, first match in this order:
    ``overlong_prompt``, ``timeout``, ``connection``, ``infra``, ``tool``, ``model_response``, else ``other``) and
    ``message_head``.
    None when the rollout ended without one."""
    if not error:
        return None
    if isinstance(error, Mapping):
        error_type = error.get("error") if isinstance(error.get("error"), str) else None
        chain = error.get("error_chain_str") if isinstance(error.get("error_chain_str"), str) else error_type
        message = error.get("message") or error.get("error_chain_repr") or error_type or ""
        detail = f"{message} {error.get('error_chain_repr') or ''}"
    else:
        error_type, chain, message = None, None, str(error)
        detail = message
    match = _STATUS_CODE.search(detail)
    status_code = int(match.group(1)) if match else None
    if status_code is not None and status_code >= 500:
        kind = "http_5xx"
    elif status_code == 429:
        kind = "http_429"
    elif status_code is not None and status_code >= 400:
        kind = "http_4xx"
    else:
        kind = next((kind for name, kind in _ROLLOUT_ERROR_KIND_CANDIDATES if name in (chain or "")), "other")
    return {
        "type": error_type,
        "chain": chain,
        "status_code": status_code,
        "kind_candidate": kind,
        "message_head": str(message)[:_ROLLOUT_ERROR_MESSAGE_CHARS],
    }


class VerifiersNeMoGymResponse(NeMoGymResponse):
    ng_trajectory: Optional[TrajectoryRecord] = Field(default=None, exclude_if=lambda value: value is None)
    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None, exclude_if=lambda value: value is None
    )
    env_id: str
    group_id: str
    output: list[dict[str, Any]]
    reward: float
    metrics: dict[str, Any] = Field(default_factory=dict)
    exported_state: dict[str, Any] = Field(default_factory=dict, exclude_if=lambda value: not value)
    # The error the rollout ended on (``_rollout_error_fact``), absent when it ended without one.
    rollout_error: Optional[dict[str, Any]] = Field(default=None, exclude_if=lambda value: value is None)
    parallel_tool_calls: bool = True
    tool_choice: str = "auto"
    tools: list = Field(default_factory=list)


class VerifiersAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    response: VerifiersNeMoGymResponse
    reward: float
    # Read by rollout_collection from the rollout RESULT, so they have to be
    # fields here -- inside `metrics` the harness never sees them.
    ng_trajectory: Optional[TrajectoryRecord] = Field(default=None, exclude_if=lambda value: value is None)
    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # The error the rollout ended on, on the row itself so a run's rollout errors can be counted; recorded only, it
    # does not change the reward or mask the sample.
    rollout_error: Optional[dict[str, Any]] = Field(default=None, exclude_if=lambda value: value is None)


# The reward functions of the request being scored that raised; verifiers scores such a function 0.0 and goes on.
_RUBRIC_FAILURES: ContextVar[Optional[list[str]]] = ContextVar("verifiers_agent_rubric_failures", default=None)


# The bookkeeping rubrics verifiers' own environments add (MultiTurnEnv, ToolEnv): metrics counted from the rollout's
# state and messages, weight 0, no model asked. A member of exactly one of these classes is left out of the scoring.
_MONITOR_RUBRICS = {
    ("verifiers.legacy.envs.multiturn_env", "MultiTurnMonitorRubric"),
    ("verifiers.legacy.envs.tool_env", "ToolMonitorRubric"),
}


def _monitor_rubric(rubric: Any) -> bool:
    """Whether ``rubric`` is one of verifiers' own bookkeeping monitors, carrying only weight-0 metrics."""
    weights = getattr(rubric, "weights", None)
    return (
        (type(rubric).__module__, type(rubric).__qualname__) in _MONITOR_RUBRICS
        and isinstance(weights, list)
        and all(weight == 0 for weight in weights)
    )


def _plain_rubrics(rubric: Any) -> Optional[list[Any]]:
    """The plain verifiers ``Rubric`` objects a rubric scores with (itself, or a ``RubricGroup``'s members at any depth,
    leaving out verifiers' own monitors), or None when any of them is another class.

    A rubric subclass may score without its listed functions (a judge rubric asks a model), so only the two plain
    classes are read.
    """
    if type(rubric) is getattr(vf, "RubricGroup", None):
        plain: list[Any] = []
        for member in getattr(rubric, "rubrics", None) or []:
            if _monitor_rubric(member):
                continue
            members = _plain_rubrics(member)
            if members is None:
                return None
            plain.extend(members)
        return plain
    if type(rubric) is getattr(vf, "Rubric", None) and isinstance(getattr(rubric, "funcs", None), list):
        return [rubric]
    return None


def _watch_reward_func(func: Callable[..., Any]) -> Callable[..., Any]:
    """``func``, recording in ``_RUBRIC_FAILURES`` that it raised before verifiers turns the error into 0.0."""

    @functools.wraps(func)  # keeps the signature verifiers reads to pick the function's arguments
    def watched(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception:
            failures = _RUBRIC_FAILURES.get()
            if failures is not None:
                failures.append(getattr(func, "__name__", repr(func)))
            raise

    watched._ng_watched_reward_func = True
    return watched


def _programmatic_rubric(vf_env: Any) -> bool:
    """Whether every reward function of the environment's rubric is synchronous, after watching each for errors.

    Only such a rubric grades without asking a model, so only its rollouts carry the agent's own grading record. The
    functions are wrapped in place once; any other rubric is left as it is and its rollouts carry no record.
    """
    rubrics = _plain_rubrics(getattr(vf_env, "rubric", None))
    funcs = [f for rubric in rubrics or [] for f in rubric.funcs]
    if not funcs or any(inspect.iscoroutinefunction(f) or not callable(f) for f in funcs):
        return False
    for rubric in rubrics:
        rubric.funcs[:] = [
            f if getattr(f, "_ng_watched_reward_func", False) else _watch_reward_func(f) for f in rubric.funcs
        ]
    return True


class _NoStoreCookieJar(CookieJar):
    """A cookie jar that drops every Set-Cookie, so no request ever carries one.

    See VerifiersAgent._get_client: the policy server's session cookie decides
    which vLLM engine serves a request, and a jar that remembers it would pin
    every rollout in this process to one engine.
    """

    def set_cookie(self, cookie) -> None:
        return None


def _default_outcome_policy() -> StageOutcomePolicyConfig:
    """What a rollout that ended on a failed policy-model call becomes, by the call's observed status and kind.

    A model-side rejection (400, 413, 422) is the model's result and is counted as a zero. A reply that is worth
    another attempt, and a call that got no reply, are retried by the collector while the rollout holds no answer.
    Anything else leaves the rollout ungraded.
    """
    return StageOutcomePolicyConfig(
        by_stage={
            "agent": {
                "by_failure_kind": {MODEL_TRANSPORT_ERROR: OutcomeAction.RETRY},
                "by_status_code": {
                    **dict.fromkeys(filter(is_retryable_dependency_status, range(400, 600)), OutcomeAction.RETRY),
                    **dict.fromkeys((400, 413, 422), OutcomeAction.COUNT_AS_ZERO),
                },
            }
        }
    )


class VerifiersAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    outcome_policy: StageOutcomePolicyConfig = Field(default_factory=_default_outcome_policy)
    model_name: str = Field(default="", description="Model name")

    @field_validator("outcome_policy", mode="before")
    @classmethod
    def keep_untouched_default_rules(cls, value: object) -> object:
        """A partial override changes only the rules it names; the others keep this agent's defaults."""
        return StageOutcomePolicyConfig.overlay(_default_outcome_policy(), value)

    vf_env_id: str = Field(default="", description="Verifiers environment ID")
    vf_env_args: dict = Field(default_factory=dict, description="Verifiers environment arguments")
    export_state_columns: list[str] = Field(
        default_factory=list,
        description="Extra verifiers State keys (JSON-serializable) copied into response.exported_state, e.g. "
        "['assertion_results'] for AutomationBench per-assertion verdicts. Empty leaves rollouts unchanged.",
    )

    max_tokens: int = Field(default=8192, description="Max tokens for generation")

    # nemo rl generation_config overrides these
    temperature: float = Field(default=1.0)
    top_p: float = Field(default=1.0)

    # Policy-client deadlines. The openai SDK defaults are a 5s connect and a
    # 600s read/write/pool timeout with 2 retries; one long agentic turn from a
    # large policy on a shared engine can exceed 600s under load, a burst of new
    # connections can exceed 5s, and a retry reruns the whole generation. None
    # keeps the SDK default, so configs that set none of these are unaffected.
    client_timeout_s: float | None = Field(
        default=None,
        description="Read/write/pool timeout in seconds for requests to the policy model server. None keeps the openai SDK default (600s).",
    )
    client_connect_timeout_s: float | None = Field(
        default=None,
        description="Connect timeout in seconds for the policy model server. None keeps the openai SDK default (5s), which a burst of new connections against a loaded or just-started server can exceed.",
    )
    client_max_retries: int | None = Field(
        default=None,
        description="openai SDK retry count for the policy client. None keeps the SDK default (2).",
    )


class VerifiersAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_idx: int
    vf_env_id: str | None = Field(default=None, description="Verifiers environment ID")
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )
    answer: str = Field(default="", description="Expected answer from dataset")
    task: str = Field(default="default", description="Task type from dataset")
    example_id: int | str = Field(default=0, description="Example ID from dataset")
    info: dict = Field(default_factory=dict, description="Extra info from dataset")


class VerifiersAgent(SimpleResponsesAPIAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: VerifiersAgentConfig

    envs_cache: dict[str, Any] = Field(default_factory=dict)
    # Whether each loaded environment's rubric grades without asking a model (``_programmatic_rubric``).
    programmatic_rubrics: dict[str, bool] = Field(default_factory=dict)
    client_cache: dict[str, NeMoRLChatCompletionsClient] = Field(default_factory=dict)

    def _get_env(self, vf_env_id: str) -> vf.Environment:
        if vf_env_id not in self.envs_cache:
            env = vf.load_environment(vf_env_id, **self.config.vf_env_args)
            self.programmatic_rubrics[vf_env_id] = _programmatic_rubric(env)
            self.envs_cache[vf_env_id] = env
        return self.envs_cache[vf_env_id]

    def _rollout_id_for(self, body: Any = None, request: Optional[Request] = None) -> Optional[str]:
        """The capture id for this call, from the run body or the request path.

        ``rollout_id_from_run`` only covers ``/run``, where rollout collection
        injects ``_ng_rollout_id`` into the body. A direct
        ``/ng-rollout/<id>/v1/responses`` call carries the id in the path
        instead, and agents get no ``RolloutContextMiddleware`` -- that is
        installed on resources servers, not here -- so the contextvar is unset
        on that route. Without reading the path, a supported prefixed call
        builds an unprefixed client and its model calls are never correlated,
        which is the same silent capture loss this agent already had.

        Gated on the same ``_capture_correlation_enabled`` as the body path, so
        a run with capture off keeps the shared unprefixed client.
        """
        if body is not None and (from_body := self.rollout_id_from_run(body)):
            return from_body
        if request is None or not self._capture_correlation_enabled():
            return None
        path_params = getattr(request, "path_params", None)
        if not isinstance(path_params, Mapping):
            return None
        return path_params.get("rollout_id") or None

    def _policy_client_options(self) -> dict[str, Any]:
        """AsyncOpenAI timeout and retry options from the config.

        An unset field keeps its SDK default: setting only the connect timeout
        keeps the 600s read/write/pool deadline, and vice versa.
        """
        options: dict[str, Any] = {}
        connect = self.config.client_connect_timeout_s
        other = self.config.client_timeout_s
        if connect is not None or other is not None:
            connect = DEFAULT_TIMEOUT.connect if connect is None else connect
            other = DEFAULT_TIMEOUT.read if other is None else other
            options["timeout"] = Timeout(connect=connect, read=other, write=other, pool=other)
        if self.config.client_max_retries is not None:
            options["max_retries"] = self.config.client_max_retries
        return options

    def _get_client(
        self,
        body: Any = None,
        request: Optional[Request] = None,
        invocation_id: Optional[str] = None,
    ) -> NeMoRLChatCompletionsClient:
        """Return a rollout-prefixed client over one shared policy transport.

        The vllm_model server picks a vLLM engine per session
        (``sha256(session_id) % len(base_urls)`` in
        responses_api_models/vllm_model/app.py ``_resolve_client``) and mints the
        session id per cookie jar (nemo_gym/server_runtime/base_server.py
        ``setup_session_middleware``). openai's AsyncOpenAI sits on an httpx client
        that persists cookies, so a plain shared client is one session and
        therefore one engine: on CMH job 3670120 (2026-09-10) 512 concurrent
        rollouts ran on 6 of 48 engines while 42 sat idle. Gym's own aiohttp
        client avoids exactly this with a DummyCookieJar; ``_NoStoreCookieJar``
        is the httpx equivalent. Every request is then a fresh session and the
        router spreads them over every engine.

        Why not a client per rollout: each rollout's single pooled connection sat
        idle through its tool phases, the router's uvicorn closes idle
        connections after 30 s, and the next turn raced that close -- CMH 3670792
        aborted 468 of 512 rollouts with
        ``APIConnectionError -> ReadError(BrokenResourceError)`` while the
        shared-client runs before it had zero. One shared pool keeps connections
        hot. The price is per-turn engine affinity, which Gym's own client does
        not have either.

        Model-call capture is keyed by the ``/ng-rollout/<id>`` URL prefix. The
        lightweight per-run OpenAI client copy changes only ``base_url`` and
        shares the cached client's transport, so calls remain correlated without
        accumulating a connection pool per rollout. ``rollout_id_from_run``
        returns ``None`` when capture is disabled, preserving the shared
        unprefixed client path, unless the rollout carries a training
        framework's sampling values: its copy's URL carries them, and the
        shared URL, built for every rollout, carries none.

        The per-run client BORROWS that transport rather than owning it, so it
        must never be closed: ``Client.close()`` closes the underlying httpx
        client, which every later rollout is still using. Nothing on the current
        path closes it -- neither this agent nor ``run_group``/``generate`` on
        the verifiers legacy API, and every ``.close()`` call site in verifiers
        sits under ``verifiers/v1/``, which this agent does not use -- but the
        wrapper looks disposable, so the rule is written down here. It also
        inherits the shared client's timeout and retry settings.
        """
        cache_key = self.config.model_server.name
        if cache_key not in self.client_cache:
            openai_client = AsyncOpenAI(
                # Shared across rollouts, so it must not name the span that happens to be current now.
                base_url=self.resolve_model_base_url(self.config.model_server.name, trace_context=False),
                api_key="EMPTY",  # pragma: allowlist secret
                # DefaultAsyncHttpxClient keeps the SDK's pool limits and redirect
                # policy. Pass the bare CookieJar: httpx.Cookies adopts a CookieJar
                # instance as-is but COPIES an httpx.Cookies into a fresh stdlib
                # jar, which would silently discard the no-store behaviour.
                http_client=DefaultAsyncHttpxClient(cookies=_NoStoreCookieJar()),
                **self._policy_client_options(),
            )
            self.client_cache[cache_key] = _ModelFailureObservingClient(openai_client)

        shared_client = self.client_cache[cache_key]
        rollout_id = self._rollout_id_for(body, request)
        if rollout_id is None and current_model_sampling_segment() is None:
            return shared_client

        model_server_url = self.resolve_model_base_url(self.config.model_server.name, rollout_id)
        headers = {"x-session-id": invocation_id} if invocation_id else {}
        return _ResponseIdRecordingClient(
            shared_client.client.copy(base_url=model_server_url, default_headers=headers)
        )

    def _convert_trajectory_to_output(self, rollout_output: dict) -> list:
        assistant_tokens = self._collect_assistant_tokens(rollout_output.get("trajectory") or [])

        output: list[dict] = []
        assist_idx = 0
        for raw in rollout_output.get("completion") or []:
            msg = _as_dict(raw)
            role = msg.get("role", "user")

            if role == "tool":
                output.append(_build_tool_result_item(msg, raw))
            elif role == "assistant":
                tokens = assistant_tokens[assist_idx] if assist_idx < len(assistant_tokens) else None
                assist_idx += 1
                output.extend(_build_assistant_items(msg, raw, tokens))
            else:
                output.append(NeMoGymEasyInputMessage(role=role, content=_text(msg.get("content"))).model_dump())

        if not any(item.get("generation_token_ids") for item in output):
            has_assistant_turn = any(
                item.get("type") == "function_call"
                or (item.get("type") == "message" and item.get("role") == "assistant")
                for item in output
            )
            err = rollout_output.get("error") or {}
            err_msg = err.get("error") or err.get("error_chain_repr") or "unknown (no error info on rollout_output)"
            if not has_assistant_turn:
                logger.warning(
                    "[verifiers_agent] rollout produced no trainable tokens. This can happen when sandbox concurrency quota is exceeded. Returning empty trajectory. "
                    "Underlying error: %s",
                    err_msg,
                )
            elif err:
                # verifiers stores a mid-rollout error on the state without logging it; this is its only Gym-side log.
                logger.warning(
                    "[verifiers_agent] rollout ended with an error after its assistant turns. Underlying error: %s",
                    err_msg,
                )
            else:
                # Eval mode (return_token_id_information: false) never returns token ids; the trajectory is complete.
                logger.debug(
                    "[verifiers_agent] no token ids on assistant turns; token ids were likely not requested. "
                    "Appending the empty training placeholder."
                )
            output.append(
                NeMoGymResponseOutputMessageForTraining(
                    id=_EMPTY_OUTPUT_ITEM_ID,
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    prompt_token_ids=[0],
                    generation_token_ids=[0],
                    generation_log_probs=[0.0],
                ).model_dump()
            )

        return output

    @staticmethod
    def _collect_assistant_tokens(trajectory: list) -> list[dict | None]:
        tokens_per_turn: list[dict | None] = []
        for step in trajectory:
            if not isinstance(step, dict):
                continue
            step_tokens = step.get("tokens")
            for m in step.get("completion") or []:
                if _as_dict(m).get("role") == "assistant":
                    tokens_per_turn.append(step_tokens)
        return tokens_per_turn

    def _build_trajectory(
        self,
        *,
        invocation_id: str,
        task_id: str,
        rollout_id: str,
        output: list[dict],
        response_ids: list[str],
        conversation: list,
        status: str,
    ) -> TrajectoryRecord:
        """Project the verifiers rollout into turns the health checks can read.

        One turn per recorded model response, in call order: that is the only
        ordering verifiers guarantees back to us. Each turn claims exactly the
        call it came from, which is what `_canonical_model_call_references`
        needs -- an invocation-level list is not enough, since the BOUND_CALLS
        checks read `turns[*].model_calls`.
        """
        gaps: list[ObservationGap] = []
        model_calls = [ModelCallRef(model_ref=self.config.model_server, response_id=rid) for rid in response_ids]
        if not model_calls:
            gaps.append(ObservationGap(code="model_call_reference_unavailable", invocation_id=invocation_id))

        groups = _assistant_turn_groups(output)
        turns: list[TrajectoryTurn] = []
        now = time()
        step_count = 0
        for index, ref in enumerate(model_calls, start=1):
            # A call with no items behind it is a genuinely empty turn; leaving
            # `answer` empty is what lets `agent_turn_hollow` say so.
            items = groups[index - 1] if index <= len(groups) else []
            # step_count is cumulative within an invocation, not per turn.
            step_count += sum(1 for item in items if item.get("type") == "function_call")
            turns.append(
                TrajectoryTurn(
                    invocation_id=invocation_id,
                    task_id=task_id,
                    rollout_id=rollout_id,
                    turn_no=index,
                    timestamp=now,
                    answer=items,
                    step_count=step_count,
                    model_calls=[ref],
                )
            )

        return TrajectoryRecord(
            task_id=task_id,
            rollout_id=rollout_id,
            invocations=[
                AgentInvocation(
                    invocation_id=invocation_id,
                    status=status,
                    model_calls=model_calls,
                    conversation=conversation,
                )
            ],
            turns=turns,
            gaps=gaps,
        )

    async def responses(
        self,
        request: Request,
        response: Response,
        body: VerifiersAgentRunRequest = Body(),
    ) -> VerifiersNeMoGymResponse:
        try:
            vf_env_id = body.vf_env_id or self.config.vf_env_id
            if not vf_env_id:
                raise ValueError("vf_env_id must be set on the request or in the agent config")
            vf_env = self._get_env(vf_env_id)
            task_idx = body.task_idx

            prompt_messages = []
            for item in body.responses_create_params.input or []:
                if hasattr(item, "role") and hasattr(item, "content"):
                    prompt_messages.append({"role": item.role, "content": item.content})
                elif isinstance(item, dict):
                    prompt_messages.append({"role": item.get("role", "user"), "content": item.get("content", "")})

            rollout_input = vf.RolloutInput(
                prompt=prompt_messages,
                answer=body.answer,
                info=body.info,
                example_id=body.example_id,
            )

            rollout_id = self._rollout_id_for(body, request)
            invocation_id = f"verifiers_{uuid4().hex}" if rollout_id else None
            trajectory = None
            observations = None
            client = self._get_client(body, request, invocation_id)

            # prefer NeMo RL generation config set in responses_create_params
            # https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/experience/rollouts.py#L1045-L1046
            sampling_args = {
                "max_tokens": self.config.max_tokens,
                "temperature": getattr(body.responses_create_params, "temperature", None) or self.config.temperature,
                "top_p": getattr(body.responses_create_params, "top_p", None) or self.config.top_p,
            }
            extra_columns = [c for c in self.config.export_state_columns if c != "trajectory"]
            outputs = await vf_env.run_group(
                group_inputs=[rollout_input],
                client=client,
                model=self.config.model_name,
                sampling_args=sampling_args,
                state_columns=["trajectory", *extra_columns],
            )

            rollout_output = outputs[0]
            reward = rollout_output.get("reward", 0.0) or 0.0
            metrics = rollout_output.get("metrics", {}) or {}
            # verifiers exports a column missing from the State as None (a typo, or the rubric never set it).
            exported_state = {c: rollout_output[c] for c in extra_columns if rollout_output.get(c) is not None}
            if missing := [c for c in extra_columns if c not in exported_state]:
                logger.warning(f"Task {task_idx}: export_state_columns {missing} not on the verifiers State, omitted")

            output = self._convert_trajectory_to_output(rollout_output)

            if invocation_id is not None:
                response_ids = getattr(client, "recorded_response_ids", [])
                trajectory = self._build_trajectory(
                    invocation_id=invocation_id,
                    task_id=str(task_idx),
                    rollout_id=rollout_id,
                    output=output,
                    response_ids=response_ids,
                    conversation=list(body.responses_create_params.input or []),
                    status="completed" if response_ids else "incomplete",
                )
                observations = AgentObservationBundle(source="verifiers", records=list(trajectory.invocations))

            return VerifiersNeMoGymResponse(
                # A rollout with no assistant turn holds only the training placeholder: say so, so that a reader
                # does not take it for a generated answer that a rerun would replace.
                metadata=None if _assistant_turn_groups(output) else dict(NG_NO_GENERATION_METADATA),
                ng_trajectory=trajectory,
                ng_agent_observations=observations,
                id=f"verifiers-{vf_env_id}-{task_idx}",
                created_at=0,
                model=self.config.model_name,
                object="response",
                output=output,
                usage=getattr(client, "recorded_usage", None),
                env_id=vf_env_id,
                group_id=str(task_idx),
                reward=reward,
                metrics=metrics,
                exported_state=exported_state,
                rollout_error=_rollout_error_fact(rollout_output.get("error")),
            )
        except Exception as e:
            logger.error(f"Exception in responses(): {type(e).__name__}: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise

    async def run(
        self,
        request: Request,
        response: Response,
        body: VerifiersAgentRunRequest = Body(),
    ) -> VerifiersAgentVerifyResponse:
        failures: list[str] = []
        model_calls = _ModelCallOutcome()
        token = _RUBRIC_FAILURES.set(failures)
        outcome_token = _MODEL_CALL_OUTCOME.set(model_calls)
        try:
            resp = await self.responses(request, response, body)
        finally:
            _RUBRIC_FAILURES.reset(token)
            _MODEL_CALL_OUTCOME.reset(outcome_token)

        async def scored() -> VerifiersAgentVerifyResponse:
            row: dict[str, Any] = {}
            call_failure = model_calls.last_failure if resp.rollout_error else None
            if failures:
                # verifiers scored each reward function that raised as 0.0; that zero is no grade of the answer.
                row = {
                    "mask_sample": True,
                    "failure_kind": VERIFIER_ERROR,
                    "failure_reason": f"reward function raised: {', '.join(sorted(set(failures)))}",
                }
            elif call_failure is not None:
                # The rollout ended on a policy-model call the SDK reported failing. The event says what was
                # observed; the configured agent-stage policy says whether the zero is counted or the row is not
                # graded, and the row keeps both for its readers.
                decision = choose_stage_outcome(
                    failure_kind=call_failure.failure_kind,
                    http_status=call_failure.http_status,
                    stage="agent",
                    config=self.config.outcome_policy,
                )
                row = {
                    "mask_sample": decision.action != OutcomeAction.COUNT_AS_ZERO,
                    "failure_kind": call_failure.failure_kind,
                    "failure_reason": call_failure.failure_reason,
                    "failure_outcome_action": decision.action,
                    "failure_outcome_attribution": decision.attribution,
                    **FailureReportRow.fields(failure_report_for(call_failure)),
                }
            return VerifiersAgentVerifyResponse(
                ng_trajectory=resp.ng_trajectory,
                ng_agent_observations=resp.ng_agent_observations,
                responses_create_params=body.responses_create_params,
                response=resp,
                # A selected zero is the result, whatever the rubric scored the rollout that errored.
                reward=0.0 if row.get("failure_outcome_action") == OutcomeAction.COUNT_AS_ZERO else resp.reward,
                rollout_error=resp.rollout_error,
                **row,
            )

        # This agent is its own verifier: the rubric graded inside the rollout above. Its result is its grading
        # record when every reward function is synchronous (no model asked), observed like a resources server's.
        # Any other rubric is not observed at all, as before: an invocation without a record would read as ungraded.
        vf_env_id = body.vf_env_id or self.config.vf_env_id
        return await observe_verifier(
            scored,
            enabled=self._model_call_capture_enabled() and self.programmatic_rubrics.get(vf_env_id, False),
            verifier_name=self.config.name,
        )()


if __name__ == "__main__":
    VerifiersAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = VerifiersAgent.run_webserver()  # noqa: F401
