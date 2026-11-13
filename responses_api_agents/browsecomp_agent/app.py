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
import asyncio
import hashlib
import json
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, List, Optional
from uuid import uuid4

import aiohttp
from fastapi import HTTPException, Request, Response
from pydantic import ConfigDict, Field, JsonValue, ValidationError

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AgentCloseSessionResponse,
    AgentOutcome,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    SandboxSession,
    SimpleResponsesAPIAgent,
    assert_model_url_reachable_from_sandbox,
)
from nemo_gym.artifacts import NG_FAILURE_CLASS_KEY, NG_NO_PERSIST_KEY
from nemo_gym.client import NeMoGymAsyncOpenAI, get_response_json, raise_for_status
from nemo_gym.client import request as http_request
from nemo_gym.config import (
    AggregateMetrics,
    AggregateMetricsRequest,
    ModelServerRef,
    ResourcesServerRef,
    get_first_server_config_dict,
    get_global_config_dict,
)
from nemo_gym.core import (
    MODEL_REQUEST_ERROR,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    accumulate_response_usage,
)
from nemo_gym.models import ChatCompletionsConverter
from nemo_gym.sandbox import (
    AsyncSandbox,
    DirectSandboxConnection,
    SandboxSpec,
    create_provider,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    DirectHTTPToolAccess,
    MCPToolAccess,
)
from nemo_gym.trajectory import AgentObservationBundle, ObservationGap
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.browsecomp_agent.sandbox import (
    _ERROR_DETAIL_CHARS,
    BrowsecompSandboxSession,
    SandboxIsolationError,
)


LOG = logging.getLogger(__name__)


# Connection retries for the sandboxed loop's model, tool and token-count calls. Gym's client waits 0.5 s between
# resends, so this rides out about 90 s of outage; a longer one ends in an error receipt, a retryable activation
# failure, instead of running into the rollout deadline and scoring as a timed-out zero.
_SANDBOX_LOOP_CONNECTION_RETRIES = 180


def _qid(text: str) -> str:
    """Short stable id for a question, for [browsecomp] debug logs."""
    return hashlib.sha256((text or "").encode()).hexdigest()[:10]


# --- Progress board ---
# An update_progress tool whose board is OVERWRITTEN on each call (not
# appended) and persists in the system prompt across context resets. The board
# is framed as a ledger of SETTLED state (not the live hypothesis): an A/B
# study showed a hypothesis-anchored board REGRESSED BrowseComp accuracy by
# more than 10 points; the settled-state slots + post-reset re-derivation rule
# below are the de-anchoring fixes from that study.

PROGRESS_TOOL = {
    "type": "function",
    "name": "update_progress",
    "description": (
        "Maintain a ledger of SETTLED state on a durable board pinned to "
        "the system prompt — the ONLY state that survives a context reset. "
        "This OVERWRITES the board — pass the complete current state every "
        "time; it is not appended. Record what is settled: confirmed facts "
        "(with source), ruled-out candidates (with the constraint each "
        "fails), and search angles already exhausted — not just your best "
        "guess. The board is NEVER your final answer: to answer, emit "
        "'Exact Answer: ...' as a normal assistant message. Board format "
        "is described in the system prompt; keep it concise."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "board": {
                "type": "string",
                "description": (
                    "The full board text. Replaces the previous board entirely — write the complete, current state."
                ),
            }
        },
        "required": ["board"],
    },
    "strict": False,
}

PROGRESS_SYSTEM_ADDENDUM = (
    "\n\n## Progress Board (durable — survives context resets)\n"
    "This board persists in the system prompt across resets; your conversation "
    "history does not. Maintain it with `update_progress` (each call replaces "
    "it). It is a ledger of SETTLED state. Track, concisely:\n"
    "• Constraints — the conditions the answer must satisfy simultaneously.\n"
    "• Confirmed facts — grounded findings, each with its source (URL/step).\n"
    "• Ruled-out candidates — candidate + the constraint that eliminates it.\n"
    "• Search angles tried — exhausted angles and what they yielded, so you "
    "don't repeat them.\n"
    "• Working hypothesis (UNVERIFIED — re-check every constraint before "
    "committing; do not treat as settled).\n\n"
    "After a context reset: treat any working hypothesis on the board as a "
    "guess to re-test from the confirmed facts, NOT a conclusion. If the same "
    "hypothesis has led for 2+ segments without confirming all constraints, "
    "switch to a new search angle (consult Search angles tried).\n\n"
    "You will get a [SYSTEM NOTE] warning just before each context reset as a "
    "last chance to save state — but do not rely on it alone; update the "
    "board whenever something is settled or ruled out.\n\n"
    "### Current board:\n{progress}"
)

PRE_RESET_BOARD_NUDGE = (
    "\n\n[SYSTEM NOTE: Your context is about to be RESET — everything except "
    "{kept} will be discarded. This is "
    "your last chance to save state: call update_progress NOW with the "
    "complete board (constraints, confirmed facts with sources, ruled-out "
    "candidates, search angles tried, working hypothesis). Do this before "
    "any other action.]"
)

# Answer-commit marker (line-anchored to avoid matching prose like
# "candidate answer: ..."). Used both to detect the model writing its final
# answer into the board instead of committing it (commit guard) and to recover
# such an answer at return time so the judge can extract it (board-answer
# fallback).
_ANSWER_COMMIT_RE = re.compile(
    r"^\s*(?:exact answer|final answer|answer)\s*:",
    re.IGNORECASE | re.MULTILINE,
)


class BrowsecompAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef | None = None
    model_server: ModelServerRef
    allow_host_execution: bool = False
    sandbox_provider: str | None = None
    sandbox_config: dict[str, JsonValue] = Field(
        default_factory=lambda: {"image": "python:3.13-slim", "workdir": "/sandbox"}
    )
    sandbox_install_timeout_seconds: float = Field(default=900, gt=0, allow_inf_nan=False)
    sandbox_runner_timeout_seconds: float = Field(default=3600, gt=0, allow_inf_nan=False)
    max_steps: int = 400
    # Wall-clock budget per rollout, measured from the start of the rollout loop. Past it the loop
    # stops, the response carries `timed_out`, and the harness scores the sample 0 without
    # a judge call. None keeps the loop bounded by max_steps only.
    rollout_timeout_s: Optional[float] = Field(default=None, ge=0)
    keep_rounds: int = 9999
    nudge_steps: bool = True
    max_context_tokens: int = 196608
    context_reset_pct: float = 0.3
    # Absolute token threshold for context reset. When > 0 it OVERRIDES
    # max_context_tokens * context_reset_pct. 50000 = the token-based reset
    # standard (matches the reference harness baselines).
    context_reset_tokens: int = 0
    context_reset_keep_rounds: int = 3
    max_run_retries: int = 1
    # Cap on the number of context resets per trajectory (None = unlimited).
    max_reset_count: Optional[int] = None
    # When set, save a JSONL snapshot of the full conversation at every context
    # reset and at the end of the trajectory, under
    # {snap_dir}/sample_{task_index}/attempt_{attempt}_{reset_<N>|final}.jsonl.
    # Off when None. (ported from gym-gitlab fe9845ee)
    snap_dir: Optional[str] = None
    # Estimate prompt tokens via the vLLM /tokenize endpoint BEFORE the model
    # call to decide context reset, instead of paying for a full generation and
    # discarding it. (ported from gym-gitlab b66e37c6)
    save_model_call_using_vllm_tokenize_endpoint: bool = False
    # Give the model an update_progress tool: a durable "progress board" pinned
    # to the system prompt that survives context resets. Each call OVERWRITES
    # the board. When the context crosses the reset threshold the model gets
    # ONE warned turn (a save-the-board [SYSTEM NOTE] on its latest tool
    # result) before the reset fires; without progress the reset fires
    # immediately as before. (ported from the reference harness)
    progress: bool = False


class BrowsecompAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class BrowsecompAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class BrowsecompAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class _RolloutDeadline(Exception):
    """A model or tool call was still running when the rollout's wall-clock limit ran out."""


async def _before_deadline(deadline: Optional[float], awaitable):
    # Bounds a single in-flight call: the per-step check alone cannot stop a call that hangs.
    if deadline is None:
        return await awaitable
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        awaitable.close()
        raise _RolloutDeadline()
    limit = asyncio.timeout(remaining)
    try:
        async with limit:
            return await awaitable
    except TimeoutError as e:
        # A TimeoutError raised by the call itself is not the rollout deadline.
        if limit.expired():
            raise _RolloutDeadline() from e
        raise


def _is_infrastructure_failure(exc: BaseException) -> bool:
    """True when the harness failed, not the sample.

    A 5xx from a sibling Gym server, or a connection/timeout error, means
    policy_model or the resources server was unreachable. The canonical case is
    a chain-hop restart: `gym eval run` starts its own FastAPI servers AFTER the
    Slurm-level vLLM gate has passed, so on resume the agent can be answering
    while policy_model is still booting.

    4xx stays sample-shaped -- a malformed request is this rollout's own fault
    and is still scored 0.
    """
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status >= 500
    return isinstance(exc, (aiohttp.ClientConnectionError, TimeoutError))


@dataclass
class BrowsecompAgentSessionState(AgentSessionState):
    """An Environment Server episode: its grant to the resources server and the cookie jar its tool calls carry."""

    tool_access: DirectHTTPToolAccess | None
    resources_cookies: dict[str, str]


def _cookie_values(cookies: Any) -> dict[str, str]:
    """Flatten a plain dict or an aiohttp cookie jar of morsels into name-to-value pairs."""
    return {str(name): str(getattr(value, "value", value)) for name, value in (cookies or {}).items()}


class BrowsecompAgent(SimpleResponsesAPIAgent):
    config: BrowsecompAgentConfig
    calls_model_from_sandbox: ClassVar[bool] = True
    _policy_model_openai_client: Optional[NeMoGymAsyncOpenAI] = None

    def setup_webserver(self):
        app = super().setup_webserver()
        app.post("/v1/browsecomp/token_count")(self._session_token_count)
        # For the /tokenize-based pre-call token estimation we need a direct
        # client to the policy model's vLLM /tokenize endpoint. Built only when
        # the feature is enabled. (ported from gym-gitlab b66e37c6)
        if self.config.save_model_call_using_vllm_tokenize_endpoint:
            global_config = get_global_config_dict()
            policy_model_config_dict = get_first_server_config_dict(global_config, self.config.model_server.name)
            base_urls = policy_model_config_dict["base_url"]
            base_url = base_urls if isinstance(base_urls, str) else base_urls[0]
            self._policy_model_openai_client = NeMoGymAsyncOpenAI(
                base_url=base_url, api_key=policy_model_config_dict["api_key"]
            )
        return app

    @staticmethod
    def _reset_threshold(config: "BrowsecompAgentConfig") -> int:
        """Token count past which the context is reset. Absolute
        context_reset_tokens (50k standard) takes precedence; otherwise fall back
        to max_context_tokens * context_reset_pct. 0 disables reset."""
        if config.context_reset_tokens:
            return int(config.context_reset_tokens)
        if config.max_context_tokens and config.context_reset_pct:
            return int(config.max_context_tokens * config.context_reset_pct)
        return 0

    @staticmethod
    def _last_message_text(response: NeMoGymResponse) -> str:
        """Text of the most-recent assistant message item that has non-empty content, walking
        back from the end of the trajectory. Mirrors the reference harness:
        the empty-answer retry keys on the LAST content-bearing assistant turn, NOT the
        concatenation of every assistant turn (NeMoGymResponse.output_text). So a final
        think-only turn triggers a retry even when an earlier turn emitted a real answer.
        Empty string if no assistant message produced content."""
        for item in reversed(response.output):
            if getattr(item, "type", None) == "message":
                text = "".join(c.text for c in item.content if getattr(c, "type", None) == "output_text")
                if text:
                    return text
        return ""

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Serve an agent session's single activation in its sandbox, or the host loop when explicitly allowed.

        A session request must use the seeded attempt-qualified rollout path (else 409). The first activation starts
        the sandboxed loop; an identical retry waits for and replays its result, and a different request answers 409.
        Disconnecting does not cancel the loop; close does. A request without a session answers 422 unless
        ``allow_host_execution`` is true.
        """
        # Thin diagnostic wrapper around the real implementation. A full-400 eval lost 6
        # samples to a 500 out of THIS endpoint with NO traceback anywhere in the log — not
        # even the `🚨` marker that the exception middleware prints for ordinary app
        # exceptions (the malformed-JSON bug printed a full ExceptionGroup). With nothing
        # logged the cause can only be bounded, never named. Print whatever escapes, then
        # RE-RAISE unchanged so run()'s containment behaves exactly as before.
        # Grep the next run for [browsecomp][responses_exc].
        try:
            agent_session_id = self._agent_session_id_from_request(request)
            if agent_session_id is None:
                if not self.config.allow_host_execution:
                    raise HTTPException(
                        422, "BrowseComp requires an agent session and a sandbox; host execution is disabled"
                    )
                return await self._responses_impl(request, response, body)
            state = self._require_agent_session(agent_session_id)
            if not isinstance(state, BrowsecompSandboxSession):
                raise TypeError("Expected BrowseComp agent session state")
            if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
                raise HTTPException(409, "Agent sessions require the seeded attempt-qualified rollout path")
            if state.task is None:
                state.activation_request = body.model_copy(deep=True)
                state.task = asyncio.create_task(self._sandbox_responses(request, body, state))
            elif body != state.activation_request:
                raise HTTPException(409, "BrowseComp sessions support one activation; retry the same request")
            return (await asyncio.shield(state.task)).model_copy(deep=True)
        except Exception as e:
            print(f"[browsecomp][responses_exc] error_type={type(e).__name__} error={str(e)[:300]}", flush=True)
            traceback.print_exc()
            raise

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> BrowsecompSandboxSession:
        if self.config.snap_dir:
            raise AgentSeedRejectedError("snap_dir is supported only by the explicit host compatibility path")
        accesses = self.effective_tool_accesses(body)
        unsupported = [access.name for access in accesses if isinstance(access, MCPToolAccess) and access.required]
        if unsupported:
            raise AgentSeedRejectedError(
                f"BrowseComp agent does not support required MCP tool access: {', '.join(unsupported)}"
            )
        direct_accesses = [access for access in accesses if isinstance(access, DirectHTTPToolAccess)]
        if len(direct_accesses) > 1:
            raise AgentSeedRejectedError(
                "BrowseComp agent supports at most one direct HTTP tool access; "
                "set resources_tool_transports: [direct_http] for a tool-using benchmark"
            )
        tool_access = next(iter(direct_accesses), None)
        owns_sandbox = body.sandbox_access is None
        if owns_sandbox:
            provider_ref = self.config.sandbox_provider
            workdir = self.config.sandbox_config.get("workdir")
            if provider_ref is None:
                raise AgentSeedRejectedError("BrowseComp requires sandbox_access or sandbox_provider")
        else:
            connection = body.sandbox_access.connection
            if not isinstance(connection, DirectSandboxConnection):
                raise AgentSeedRejectedError("BrowseComp supports only direct sandbox connections")
            if body.sandbox_access.user is not None or body.sandbox_access.env:
                raise AgentSeedRejectedError("BrowseComp does not support task-specific sandbox users or environment")
            provider_ref = connection.provider_config_ref
            workdir = body.sandbox_access.workdir
        if not isinstance(workdir, str) or not workdir.startswith("/"):
            raise AgentSeedRejectedError("BrowseComp requires an absolute sandbox workdir")
        try:
            provider_config = resolve_provider_config(provider_ref, get_global_config_dict())
            if owns_sandbox:
                SandboxSpec(**self.config.sandbox_config)
        except (TypeError, ValueError) as error:
            # Every seed would fail the same way, so a retryable 500 would only repeat it for each episode.
            raise AgentSeedRejectedError(f"BrowseComp sandbox provider is not usable: {error}") from error
        provider_name = next(iter(provider_config), None)
        if provider_name == "local":
            raise AgentSeedRejectedError(
                "BrowseComp sessions require an isolated sandbox provider; local is not isolated"
            )
        try:
            # Every URL the sandboxed loop calls: the model, the tool grant and this agent's token counting.
            sandbox_urls = [
                self.resolve_model_base_url(self.config.model_server.name, body.episode_id.capture_key),
                str(tool_access.base_url) if tool_access is not None else None,
                self._token_count_url(),
            ]
            for url in filter(None, sandbox_urls):
                assert_model_url_reachable_from_sandbox(url, provider_name=provider_name)
        except ValueError as error:
            raise AgentSeedRejectedError(str(error)) from error
        try:
            provider = create_provider(provider_config)
        except (TypeError, ValueError) as error:
            raise AgentSeedRejectedError(f"BrowseComp sandbox provider is not usable: {error}") from error
        if owns_sandbox:
            sandbox = AsyncSandbox(provider)
        else:
            try:
                sandbox = await AsyncSandbox.connect(connection.descriptor, provider=provider)
            except BaseException:
                await provider.aclose()
                raise
        state = BrowsecompSandboxSession(
            request=body,
            session=SandboxSession(
                sandbox=sandbox,
                workdir=workdir,
                session_dir=f"/tmp/nemo-gym-browsecomp-sessions/{uuid4().hex}",
                owns_sandbox=owns_sandbox,
                harness="BrowseComp",
            ),
            tool_access=tool_access,
            resources_cookies=dict(tool_access.cookies) if tool_access else {},
        )
        try:
            if owns_sandbox:
                spec = dict(self.config.sandbox_config)
                spec["metadata"] = {
                    **resolve_provider_metadata(provider_ref, get_global_config_dict()),
                    **(spec.get("metadata") or {}),
                    "agent_session_id": body.agent_session_id[:63],
                }
                spec.setdefault(
                    "ttl_s",
                    self.config.sandbox_install_timeout_seconds + self.config.sandbox_runner_timeout_seconds + 600,
                )
                # Retain ownership before start: failed file delivery can also fail its automatic stop.
                started = time.monotonic()
                await sandbox.start(SandboxSpec(**spec))
                state.started_at = time.monotonic()
                LOG.info(
                    "[browsecomp][sandbox_start] role=agent session=%s provider=%s id=%s start_s=%.3f",
                    body.agent_session_id,
                    sandbox.provider_name,
                    sandbox.sandbox_id,
                    state.started_at - started,
                )
            await state.install_runtime(timeout=self.config.sandbox_install_timeout_seconds)
        except BaseException as error:
            try:
                await state.close(timeout=self.config.session_close_timeout_seconds)
            except BaseException:
                raise AgentSessionSetupError(state, error=error) from error
            if isinstance(error, SandboxIsolationError):
                # Every retry would get a sandbox from the same provider.
                raise AgentSeedRejectedError(str(error)) from error
            raise
        return state

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, BrowsecompSandboxSession):
            raise TypeError("Expected BrowseComp agent session state")
        await state.close(timeout=self.config.session_close_timeout_seconds)
        artifacts = state.session.artifacts
        if artifacts is not None and isinstance(artifacts.get("resources_cookies"), dict):
            state.resources_cookies = _cookie_values(artifacts["resources_cookies"])
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            resources_cookies=state.resources_cookies,
            agent_observations=AgentObservationBundle(
                source="browsecomp",
                gaps=[
                    ObservationGap(
                        code="native_turn_evidence_unavailable" if artifacts else "observation_capture_failed",
                        detail="Evaluation capability: model captures and raw output are retained; native invocation/turn records are not emitted.",
                    )
                ],
            ),
        )

    async def _session_token_count(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> dict[str, int]:
        """Count a session's prompt without copying the upstream API credential into its sandbox."""
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(409, "Token counting requires an active agent session")
        self._require_agent_session(session_id)
        if not self.config.save_model_call_using_vllm_tokenize_endpoint:
            raise HTTPException(422, "vLLM token counting is disabled")
        return {"count": await self._count_prompt_tokens(body)}

    def _token_count_url(self) -> str | None:
        """This agent's token-count endpoint, which a sandboxed loop calls instead of the upstream model."""
        if not self.config.save_model_call_using_vllm_tokenize_endpoint:
            return None
        return f"{self.server_client._resolve_base_url(self.config.name)}/v1/browsecomp/token_count"

    async def _sandbox_responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming, state: BrowsecompSandboxSession
    ) -> NeMoGymResponse:
        if body.tools and state.tool_access is None:
            raise HTTPException(422, "Tool-using BrowseComp requests require a direct HTTP tool grant")
        timeout = self.config.sandbox_runner_timeout_seconds
        if state.request.agent_timeout_sec is not None:
            timeout = min(timeout, state.request.agent_timeout_sec)
        # Keep the native timeout response/zero-reward semantics; the supervisor has a small receipt grace.
        config = self.config.model_dump(mode="json", exclude={"tool_accesses", "sandbox_config", "sandbox_provider"})
        config["rollout_timeout_s"] = min(
            self.config.rollout_timeout_s if self.config.rollout_timeout_s is not None else timeout, timeout
        )
        payload = {
            "seed": state.request.model_copy(update={"sandbox_access": None, "tool_accesses": []}).model_dump(
                mode="json"
            ),
            "tool_access": state.tool_access.model_dump(mode="json") if state.tool_access else None,
            "body": body.model_dump(mode="json", exclude_unset=True),
            "config": config,
            "model_url": self.resolve_model_base_url(
                self.config.model_server.name, state.request.episode_id.capture_key
            ),
            "token_count_url": self._token_count_url(),
            "agent_cookies": dict(request.cookies),
        }
        output = await state.session.execute(
            stage_activation=lambda: state.stage_activation(payload),
            collect=state.collect,
            timeout=timeout + 30,
            close_timeout=self.config.session_close_timeout_seconds,
        )
        if output.get("error"):
            if output.get("traceback"):
                print(
                    f"[browsecomp][sandbox_error][{state.request.agent_session_id}]\n{output['traceback']}",
                    flush=True,
                )
            raise RuntimeError(f"BrowseComp sandbox failed: {output['error']}")
        state.resources_cookies = _cookie_values(output.get("resources_cookies"))
        return NeMoGymResponse.model_validate(output["response"])

    async def _run_session_loop(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming,
        state: BrowsecompAgentSessionState,
        *,
        started_at: float | None = None,
    ) -> NeMoGymResponse:
        """Run an Environment Server's one activation, with the empty-answer retries run() makes around it.

        Like run(), a retry keeps the session's resources state; it does not seed again. A model call that fails with
        an HTTP error returns a failed response that names its status, for the Environment Server's outcome policy.
        ``started_at`` is when the activation began, if before this call (``time.monotonic()``).
        """
        attempts = max(1, self.config.max_run_retries)
        # rollout_timeout_s bounds the whole activation, so every retry counts from the activation's start.
        if started_at is None:
            started_at = time.monotonic()
        for attempt in range(attempts):
            try:
                model_response = await self._responses_impl(
                    request, response, body, session=state, started_at=started_at
                )
            except aiohttp.ClientResponseError as error:
                return self._model_request_failure(body, error)
            answer = re.sub(r"<think>.*?</think>", "", self._last_message_text(model_response), flags=re.DOTALL)
            if answer.strip() or getattr(model_response, "timed_out", False) or attempt == attempts - 1:
                return model_response
            print(
                f"[browsecomp][retry][{state.request.episode_id.capture_key}] attempt={attempt + 1}/{attempts} "
                f"reason=empty_output_after_think_strip",
                flush=True,
            )

    def _model_request_failure(
        self, body: NeMoGymResponseCreateParamsNonStreaming, error: aiohttp.ClientResponseError
    ) -> NeMoGymResponse:
        """A failed response for a model call that answered with an HTTP error.

        run() scored a 4xx 0 and set a 5xx aside as an infrastructure failure. The Environment Server's default
        outcome policy verifies 400, 413 and 422, retries 408, 425, 429 and 5xx, and fails the episode on the rest.
        The output is empty, as run() scored no partial answer.
        """
        content = getattr(error, "response_content", b"")
        detail = content.decode(errors="replace") if isinstance(content, bytes) else str(content)
        # The rollout row carries this message, so an upstream error page stays bounded.
        detail = detail[:_ERROR_DETAIL_CHARS]
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=time.time(),
            model=body.model or self.config.model_server.name,
            object="response",
            output=[],
            status="failed",
            error={"code": "server_error", "message": f"Model HTTP {error.status}: {detail or error.message}"},
            tools=body.tools,
            tool_choice=body.tool_choice,
            parallel_tool_calls=body.parallel_tool_calls,
            metadata={
                **(body.metadata or {}),
                AGENT_OUTCOME_METADATA_KEY: AgentOutcome(
                    failure_kind=MODEL_REQUEST_ERROR, status_code=error.status
                ).model_dump_json(),
            },
        )

    async def _responses_impl(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        session: Optional[BrowsecompAgentSessionState] = None,
        started_at: Optional[float] = None,
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        qid = _qid(json.dumps([m.model_dump() if hasattr(m, "model_dump") else m for m in body.input], default=str))

        new_outputs = []
        full_trajectory = []  # never-trimmed; mirrors new_outputs appends across resets
        usage = None
        step = 0
        model_server_cookies = None  # update the cookies on every model response
        # update the cookies on every resources server response
        resources_server_cookies = request.cookies if session is None else dict(session.resources_cookies)

        reset_threshold = self._reset_threshold(self.config)
        loop_started = time.monotonic() if started_at is None else started_at
        deadline = loop_started + self.config.rollout_timeout_s if self.config.rollout_timeout_s is not None else None
        timed_out = False
        stop_reason = None  # why the loop ended: answered / max_steps / context_full / timeout
        model_response = None

        # --- Progress board state (ported from the reference harness) ---
        # The board lives in the system prompt and is re-rendered only at the
        # initial build and on each context reset — never mid-segment.
        progress_board = ""
        reset_armed = False  # over-threshold grants ONE warned board-save turn before the reset fires
        pre_reset_warning_steps = []
        progress_sys_idx = None
        progress_base_system = ""

        def _render_progress_system():
            """Rebuild the system message with the current board."""
            board_text = progress_board or (
                "(empty — start by listing the question's constraints; see the format above)"
            )
            body.input[progress_sys_idx] = body.input[progress_sys_idx].model_copy(
                update={"content": progress_base_system + PROGRESS_SYSTEM_ADDENDUM.format(progress=board_text)}
            )

        def _pre_reset_kept_desc():
            if self.config.context_reset_keep_rounds > 0:
                return "the last %d rounds and your progress board" % self.config.context_reset_keep_rounds
            return "your progress board"

        if self.config.progress:
            # reference-harness parity: active_tools = TOOLS + [PROGRESS_TOOL] (+ BASH_TOOL
            # last), so update_progress goes BEFORE bash_command in the rendered
            # prompt. Tool order changes the materialized prompt bytes.
            tools = list(body.tools)
            bash_idx = next(
                (i for i, t in enumerate(tools) if (t["name"] if isinstance(t, dict) else t.name) == "bash_command"),
                len(tools),
            )
            tools.insert(bash_idx, PROGRESS_TOOL)
            body.tools = tools
            for i, input_message in enumerate(body.input):
                if getattr(input_message, "role", None) in ("system", "developer") and isinstance(
                    getattr(input_message, "content", None), str
                ):
                    progress_sys_idx = i
                    progress_base_system = input_message.content
                    break
            if progress_sys_idx is None:
                body.input = [NeMoGymEasyInputMessage(role="system", content="")] + list(body.input)
                progress_sys_idx = 0
            _render_progress_system()

        # --- snapshot keys + per-trajectory counters (ported from gym-gitlab fe9845ee) ---
        task_index, attempt = None, None
        if self.config.snap_dir and body.metadata:
            task_index = body.metadata.pop("task_index", None)
            attempt = body.metadata.pop("attempt", None)
            # num_repeats > 1: rollouts of a task share task_index, so fold the
            # rollout index into the snapshot dir key (sample_{ti}_r{ri}) or
            # concurrent rollouts clobber each other's snapshot files. Absent
            # rollout_index (single-rollout runs) keeps legacy sample_{ti} naming.
            rollout_index = body.metadata.pop("rollout_index", None)
            if task_index is not None and rollout_index not in (None, ""):
                task_index = f"{task_index}_r{rollout_index}"
        reset_count = 0
        reset_steps = []  # step numbers at which a context reset fired (for the trajectory header)
        num_tool_calls = 0
        max_reset_count = self.config.max_reset_count

        while True:
            if (
                self.config.rollout_timeout_s is not None
                and step > 0
                and time.monotonic() - loop_started > self.config.rollout_timeout_s
            ):
                print(
                    f"[browsecomp][timeout][{qid}] step={step} "
                    f"elapsed_min={(time.monotonic() - loop_started) / 60:.1f} "
                    f"limit_min={self.config.rollout_timeout_s / 60:.1f}",
                    flush=True,
                )
                timed_out = True
                break
            step += 1

            if self.config.keep_rounds is not None and new_outputs:
                new_outputs = self._compact_old_tool_messages(new_outputs)

            new_body = body.model_copy(update={"input": body.input + new_outputs})

            # --- Pre-call context reset via the vLLM /tokenize endpoint ---
            # Estimate the prompt token count BEFORE generating; if over the
            # threshold, reset now and skip the (otherwise wasted) generation.
            # (ported from gym-gitlab b66e37c6)
            if self.config.save_model_call_using_vllm_tokenize_endpoint:
                try:
                    pre_prompt_tokens = await _before_deadline(deadline, self._count_prompt_tokens(new_body))
                except _RolloutDeadline:
                    print(f"[browsecomp][timeout][{qid}] step={step} in=tokenize", flush=True)
                    timed_out = True
                    break
                if (
                    reset_threshold
                    and pre_prompt_tokens > reset_threshold
                    and (max_reset_count is None or reset_count < max_reset_count)
                ):
                    if self.config.progress and not reset_armed:
                        # The model cannot see the reset coming. Warn it on its
                        # latest tool result and give it this one turn to save the
                        # board; the armed reset fires on the next iteration's
                        # pre-call check. (ported from an internal reference harness)
                        reset_armed = True
                        pre_reset_warning_steps.append(step)
                        print(
                            f"[browsecomp][progress][{qid}] step={step} pre-call tokens={pre_prompt_tokens} > "
                            f"{reset_threshold} — board-save warning injected; reset fires after this turn",
                            flush=True,
                        )
                        if new_outputs and new_outputs[-1].type == "function_call_output":
                            new_outputs[-1] = new_outputs[-1].model_copy(
                                update={
                                    "output": new_outputs[-1].output
                                    + PRE_RESET_BOARD_NUDGE.format(kept=_pre_reset_kept_desc())
                                }
                            )
                            if full_trajectory:
                                full_trajectory[-1] = new_outputs[-1]
                            new_body = body.model_copy(update={"input": body.input + new_outputs})
                    else:
                        reset_count += 1
                        reset_steps.append(step)
                        reset_armed = False
                        if self.config.snap_dir:
                            self._save_snapshot(
                                messages=body.input + new_outputs,
                                task_index=task_index,
                                attempt=attempt,
                                reset_count=reset_count,
                                is_final=False,
                            )
                        # Rebuild the board into the system prompt BEFORE the shrink
                        # probes so their token counts include it.
                        if self.config.progress:
                            _render_progress_system()
                        # Adaptive shrink: keep the largest number of recent rounds
                        # whose resulting prompt still fits under the threshold.
                        chosen = None
                        for n in range(self.config.context_reset_keep_rounds, -1, -1):
                            cand = self._extract_last_rounds(new_outputs, n=n)
                            cand_body = body.model_copy(update={"input": body.input + cand})
                            try:
                                cand_tokens = await _before_deadline(deadline, self._count_prompt_tokens(cand_body))
                            except _RolloutDeadline:
                                print(f"[browsecomp][timeout][{qid}] step={step} in=tokenize", flush=True)
                                timed_out = True
                                break
                            if cand_tokens <= reset_threshold:
                                chosen = cand
                                break
                        if timed_out:
                            break
                        new_outputs = chosen if chosen is not None else []
                        continue

            async def _model_call():
                response = await self.server_client.post(
                    server_name=self.config.model_server.name,
                    url_path=self.url_path_for_request("/v1/responses", request),
                    json=new_body,
                    cookies=model_server_cookies,
                )
                # We raise for status here since we expect model calls to always work.
                await raise_for_status(response)
                return response, await get_response_json(response)

            try:
                model_response, model_response_json = await _before_deadline(deadline, _model_call())
            except _RolloutDeadline:
                print(f"[browsecomp][timeout][{qid}] step={step} in=model_call", flush=True)
                timed_out = True
                break
            model_server_cookies = model_response.cookies
            try:
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                ) from e

            # --- Check context reset threshold (post-call fallback; used when
            # save_model_call_using_vllm_tokenize_endpoint is off) ---
            prompt_tokens = model_response.usage.input_tokens if model_response.usage else 0
            pre_reset_nudge_due = False
            if (
                reset_threshold
                and prompt_tokens > reset_threshold
                and (max_reset_count is None or reset_count < max_reset_count)
            ):
                if self.config.progress and not reset_armed:
                    # One warned turn: process this completion normally, inject the
                    # save-the-board warning after its tool results, and reset at
                    # the end of the NEXT turn. (ported from an internal reference harness)
                    reset_armed = True
                    pre_reset_nudge_due = True
                    pre_reset_warning_steps.append(step)
                    print(
                        f"[browsecomp][progress][{qid}] step={step} tokens={prompt_tokens} > {reset_threshold} — "
                        f"board-save warning due; reset fires after next turn",
                        flush=True,
                    )
                elif not self.config.progress:
                    reset_count += 1
                    reset_steps.append(step)
                    if self.config.snap_dir:
                        self._save_snapshot(
                            messages=body.input + new_outputs,
                            task_index=task_index,
                            attempt=attempt,
                            reset_count=reset_count,
                            is_final=False,
                        )
                    if self.config.context_reset_keep_rounds > 0:
                        new_outputs = self._extract_last_rounds(new_outputs)
                    else:
                        new_outputs = []
                    continue
                # else: the warned board-save turn is in flight — the armed reset
                # fires at the end of this iteration, once its tool calls have run.

            output = model_response.output
            new_outputs.extend(output)
            full_trajectory.extend(output)

            usage = accumulate_response_usage(usage, model_response.usage)

            if model_response.incomplete_details and model_response.incomplete_details.reason == "max_output_tokens":
                # The model server also returns this for a context overflow (empty, length-cut response).
                stop_reason = "context_full"
                break

            # --- If the model decided to answer (no tool calls), we are done ---
            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]
            if not all_fn_calls and all_output_messages:
                stop_reason = "answered"
                break

            # --- Execute tool calls ---
            for output_function_call in all_fn_calls:
                num_tool_calls += 1
                # The model can emit syntactically invalid JSON for `arguments` (observed
                # in practice: a call truncated mid-object). Parse once, up front, and
                # treat a failure the same way a >=400 tool response is treated below —
                # report it back to the model — instead of letting it raise out of this
                # handler, which 500s the agent and kills the ENTIRE rollout collection
                # over one bad call.
                try:
                    tool_args = json.loads(output_function_call.arguments)
                    tool_args_error = None
                except json.JSONDecodeError as e:
                    tool_args = None
                    tool_args_error = e

                if tool_args_error is not None:
                    print(
                        f"[browsecomp][tool_fail][{qid}] step={step} tool={output_function_call.name} "
                        f"status=bad_arguments_json error={tool_args_error}",
                        flush=True,
                    )
                    tool_output = (
                        f"Invalid JSON in the arguments of your '{output_function_call.name}' tool "
                        f"call: {tool_args_error}. Re-issue the call with valid JSON arguments."
                    )
                elif (
                    self.config.progress
                    and output_function_call.name == "update_progress"
                    and not isinstance(tool_args, dict)
                ):
                    # Valid JSON that is not an object (a list, a string, null) is the model's mistake, like
                    # invalid JSON above: it reads the error and may retry, so the rollout keeps its score.
                    print(
                        f"[browsecomp][tool_fail][{qid}] step={step} tool=update_progress status=bad_arguments_type",
                        flush=True,
                    )
                    tool_output = (
                        "Arguments of 'update_progress' must be a JSON object with a 'board' string; "
                        "re-issue the call."
                    )
                elif self.config.progress and output_function_call.name == "update_progress":
                    # Board writes are handled by the agent itself — the board is
                    # per-trajectory agent state, not a resources-server tool.
                    progress_board = str(tool_args.get("board", ""))
                    if _ANSWER_COMMIT_RE.search(progress_board):
                        # Commit guard: the model wrote its final answer into the
                        # board instead of committing it as a message.
                        tool_output = (
                            "Progress board updated, BUT the board is NOT your final "
                            "answer — the harness does not read answers from the board. "
                            "To commit your answer, emit 'Exact Answer: ...' as a normal "
                            "assistant message (no tool call)."
                        )
                    else:
                        tool_output = (
                            "Progress board updated (%d chars). It persists in the "
                            "system prompt across context resets." % len(progress_board)
                        )
                elif session is not None and session.tool_access is None:
                    # A tool-free pairing has no grant to send the call to; answer as an unknown tool would.
                    tool_output = f"Tool '{output_function_call.name}' is not available in this session."
                else:

                    async def _tool_call():
                        if session is None:
                            response = await self.server_client.post(
                                server_name=self.config.resources_server.name,
                                url_path=f"/{output_function_call.name}",
                                json=tool_args,
                                cookies=resources_server_cookies,
                            )
                        else:
                            # An Environment Server episode reaches its resources session only through the grant.
                            response = await http_request(
                                method="POST",
                                url=f"{str(session.tool_access.base_url).rstrip('/')}/{output_function_call.name}",
                                json=tool_args,
                                cookies=resources_server_cookies,
                                headers=dict(session.tool_access.headers),
                                _internal=True,
                                _max_connection_retries=_SANDBOX_LOOP_CONNECTION_RETRIES,
                            )
                        return response, (await response.content.read()).decode()

                    try:
                        api_response, tool_output = await _before_deadline(deadline, _tool_call())
                    except _RolloutDeadline:
                        print(f"[browsecomp][timeout][{qid}] step={step} in=tool_call", flush=True)
                        timed_out = True
                        break
                    # We don't raise for status here since it's a valid return for the API to error e.g. if the model outputs an invalid call or something.
                    if session is None:
                        resources_server_cookies = api_response.cookies
                    else:
                        resources_server_cookies = resources_server_cookies | _cookie_values(api_response.cookies)
                        session.resources_cookies = resources_server_cookies
                    # reference-harness parity: the resources server wraps every tool result in a one-field
                    # JSON envelope ({"results_string": "..."}), which JSON-escapes newlines — the
                    # model then sees a single escaped line instead of raw multi-line text. This was
                    # the last remaining materialized-prompt difference vs the reference harness.
                    # Unwrap it so the model sees the raw text. Error bodies (different JSON shape)
                    # and non-JSON payloads are left untouched.
                    try:
                        parsed_tool_output = json.loads(tool_output)
                        if (
                            isinstance(parsed_tool_output, dict)
                            and set(parsed_tool_output) == {"results_string"}
                            and isinstance(parsed_tool_output["results_string"], str)
                        ):
                            tool_output = parsed_tool_output["results_string"]
                    except json.JSONDecodeError:
                        pass
                    if api_response.status >= 400:
                        print(
                            f"[browsecomp][tool_fail][{qid}] step={step} tool={output_function_call.name} "
                            f"status={api_response.status} body={tool_output[:300]}",
                            flush=True,
                        )
                if self.config.nudge_steps:
                    turns_left = self.config.max_steps - step
                    tool_output += "\n\n[%d turns remaining out of %d]" % (turns_left, self.config.max_steps)

                tool_response = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=tool_output,
                )
                new_outputs.append(tool_response)
                full_trajectory.append(tool_response)
            if timed_out:
                break

            # --- Pre-reset warning: last chance to save the progress board ---
            if pre_reset_nudge_due and new_outputs and new_outputs[-1].type == "function_call_output":
                new_outputs[-1] = new_outputs[-1].model_copy(
                    update={
                        "output": new_outputs[-1].output + PRE_RESET_BOARD_NUDGE.format(kept=_pre_reset_kept_desc())
                    }
                )
                if full_trajectory:
                    full_trajectory[-1] = new_outputs[-1]

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
                        "you have already done. Do not keep searching endlessly.]" % (step, self.config.max_steps)
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
                    if full_trajectory:
                        full_trajectory[-1] = new_outputs[-1]

            # --- Armed reset: fires after the warned board-save turn completes ---
            # (post-call mode; in tokenize mode the next pre-call check fires it)
            if (
                reset_armed
                and not pre_reset_nudge_due
                and not self.config.save_model_call_using_vllm_tokenize_endpoint
            ):
                reset_count += 1
                reset_steps.append(step)
                reset_armed = False
                if self.config.snap_dir:
                    self._save_snapshot(
                        messages=body.input + new_outputs,
                        task_index=task_index,
                        attempt=attempt,
                        reset_count=reset_count,
                        is_final=False,
                    )
                _render_progress_system()
                if self.config.context_reset_keep_rounds > 0:
                    new_outputs = self._extract_last_rounds(new_outputs)
                else:
                    new_outputs = []

            # Check if max steps is not None and if we have exhausted it.
            if self.config.max_steps and step >= self.config.max_steps:
                print(f"[browsecomp][max_steps][{qid}] step={step} max_steps={self.config.max_steps}", flush=True)
                stop_reason = "max_steps"
                break

        # --- Board-answer fallback (ported from an internal reference harness): if the run
        # ends without an answer commit in the last content-bearing assistant
        # message but the board contains one, surface the board's answer line so
        # the judge can extract it. Appends only — a trajectory that already
        # commits an answer is unchanged.
        if self.config.progress and progress_board:
            last_msg_idx = None
            last_text = ""
            for i in range(len(full_trajectory) - 1, -1, -1):
                item = full_trajectory[i]
                if getattr(item, "type", None) == "message":
                    text = "".join(c.text for c in item.content if getattr(c, "type", None) == "output_text")
                    if text:
                        last_msg_idx, last_text = i, text
                        break
            board_answer_lines = [line for line in progress_board.splitlines() if _ANSWER_COMMIT_RE.match(line)]
            if board_answer_lines and not _ANSWER_COMMIT_RE.search(last_text):
                recovered = "[Recovered from progress board] " + board_answer_lines[-1].strip()
                print(f"[browsecomp][progress][{qid}] recovering board answer into the trajectory", flush=True)
                if last_msg_idx is not None:
                    recovered_message = full_trajectory[last_msg_idx]
                    content = list(recovered_message.content)
                    for j in range(len(content) - 1, -1, -1):
                        if getattr(content[j], "type", None) == "output_text":
                            content[j] = content[j].model_copy(update={"text": content[j].text + "\n\n" + recovered})
                            break
                    full_trajectory[last_msg_idx] = recovered_message.model_copy(update={"content": content})
                else:
                    full_trajectory.append(
                        NeMoGymResponseOutputMessage(
                            id="msg_progress_board_recovery",
                            content=[NeMoGymResponseOutputText(annotations=[], text=recovered, type="output_text")],
                            role="assistant",
                            status="completed",
                            type="message",
                        )
                    )

        # --- Final trajectory snapshot (ported from gym-gitlab fe9845ee) ---
        if self.config.snap_dir:
            self._save_snapshot(
                messages=body.input + new_outputs,
                task_index=task_index,
                attempt=attempt,
                reset_count=None,
                is_final=True,
            )
            # Full untrimmed conversation (reference-harness parity: one trajectory.jsonl per sample).
            self._save_trajectory(
                input_messages=body.input,
                full_trajectory=full_trajectory,
                task_index=task_index,
                attempt=attempt,
                reset_steps=reset_steps,
                reset_count=reset_count,
                num_tool_calls=num_tool_calls,
                pre_reset_warning_steps=pre_reset_warning_steps,
            )

        # Propogate any extra cookies necessary for downstream verification
        if session is None:
            downstream_cookies = (*resources_server_cookies.items(), *(model_server_cookies or {}).items())
        else:
            # The Environment Server takes the resources cookies from the session's close, not from this reply.
            session.resources_cookies = resources_server_cookies
            downstream_cookies = (model_server_cookies or {}).items()
        for k, v in downstream_cookies:
            response.set_cookie(k, v)

        if model_response is None:
            # The first model call was still running at the limit: there is no response to extend.
            model_response = NeMoGymResponse(
                id=f"resp_timeout_{qid}",
                created_at=time.time(),
                model=body.model or "",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
            )
        model_response.output = full_trajectory
        model_response.usage = usage
        # Surface counters for downstream analysis (ported from gym-gitlab fe9845ee).
        # NeMoGymResponse(Response) has extra="allow", so these round-trip to /verify.
        model_response.reset_count = reset_count
        model_response.num_tool_calls = num_tool_calls
        model_response.pre_reset_warning_steps = pre_reset_warning_steps
        model_response.timed_out = timed_out
        # For training-side filters; nothing in Gym reads it. context_full also covers a response the model
        # cut at its own output budget, so masking it masks those model failures too (see README).
        model_response.stop_reason = "timeout" if timed_out else stop_reason
        if timed_out:
            model_response.status = "incomplete"
        return model_response

    async def run(self, request: Request, body: BrowsecompAgentRunRequest) -> BrowsecompAgentVerifyResponse:
        """Seed, answer and verify one row on the host; a development-only path.

        Answers 422 unless ``allow_host_execution`` is true and ``resources_server`` is set. Benchmark configs use
        agent sessions through ``single_agent_turn_legacy`` instead. An empty answer is retried up to
        ``max_run_retries`` times, and a timed-out rollout scores 0 without the judge. Any error scores 0 with
        ``agent_error``; an infrastructure error is also marked ``kill_shaped`` for resume.
        """
        if not self.config.allow_host_execution or self.config.resources_server is None:
            raise HTTPException(
                422, "Legacy /run requires allow_host_execution=true and resources_server; use agent sessions"
            )
        cookies = request.cookies

        question_text = getattr(body, "question", None) or ""
        rcp_input = body.responses_create_params.input
        if isinstance(rcp_input, str):
            rcp_input = [NeMoGymEasyInputMessage(role="user", content=rcp_input)]
        qid = _qid(json.dumps([m.model_dump() if hasattr(m, "model_dump") else m for m in rcp_input], default=str))
        print(f"[browsecomp][start][{qid}] question={question_text[:200]!r}", flush=True)

        # Initialised BEFORE the try: the except block reads last_response_json, so binding it
        # any later means a failure during /seed_session raises UnboundLocalError *inside the
        # handler*. That escapes containment and makes /run return 500 -- the exact run-killing
        # path the handler exists to prevent.
        last_verify_response = None
        last_response_json = None
        try:
            seed_session_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_session_response)
            cookies = seed_session_response.cookies

            for attempt in range(self.config.max_run_retries):
                # Seed snapshot keys so responses() can name per-reset/-final files.
                # (ported from gym-gitlab fe9845ee)
                if self.config.snap_dir:
                    body.responses_create_params.metadata = dict(body.responses_create_params.metadata or {})
                    body.responses_create_params.metadata["task_index"] = str(getattr(body, "_ng_task_index", qid))
                    body.responses_create_params.metadata["attempt"] = str(attempt)
                    # Disambiguate num_repeats>1 rollouts (same _ng_task_index) in snap paths.
                    rollout_index = getattr(body, "_ng_rollout_index", None)
                    if rollout_index is not None:
                        body.responses_create_params.metadata["rollout_index"] = str(rollout_index)
                response = await self.server_client.post(
                    server_name=self.config.name,
                    url_path=self.url_path_for_run("/v1/responses", body),
                    json=body.responses_create_params,
                    cookies=cookies,
                )
                await raise_for_status(response)
                cookies = response.cookies

                # Retry if the model's LAST content-bearing turn was empty after <think>-strip.
                # (Keyed on the last assistant message, matching the reference harness, NOT the concatenated
                # output_text — a final think-only turn retries even if an earlier turn had text.)
                response_json = await get_response_json(response)
                last_response_json = response_json
                raw_output_text = self._last_message_text(NeMoGymResponse.model_validate(response_json))
                cleaned_output_text = re.sub(r"<think>.*?</think>", "", raw_output_text, flags=re.DOTALL).strip()
                # Need to get last_verify_response if all attempts are exhausted
                # A timed-out rollout is final: retrying would spend the budget twice.
                if (
                    not cleaned_output_text
                    and not response_json.get("timed_out")
                    and attempt != self.config.max_run_retries - 1
                ):
                    print(
                        f"[browsecomp][retry][{qid}] attempt={attempt + 1}/{self.config.max_run_retries} "
                        f"reason=empty_output_after_think_strip",
                        flush=True,
                    )
                    continue

                if response_json.get("timed_out"):
                    # Past rollout_timeout_s: wrong by construction, so the judge is not called.
                    last_verify_response = BrowsecompAgentVerifyResponse.model_validate(
                        body.model_dump()
                        | {
                            "response": response_json,
                            "reward": 0.0,
                            "num_tool_calls": response_json.get("num_tool_calls", 0),
                            "reset_count": response_json.get("reset_count", 0),
                        }
                    )
                    break

                verify_request = BrowsecompAgentVerifyRequest.model_validate(
                    body.model_dump() | {"response": response_json}
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

            if last_verify_response is not None:
                # Every row carries timed_out, so its mean is the share of samples stopped by the limit.
                last_verify_response = last_verify_response.model_copy(
                    update={"timed_out": int(bool((last_response_json or {}).get("timed_out")))}
                )
            reward = getattr(last_verify_response, "reward", None) if last_verify_response is not None else None
            outcome = "success" if (reward is not None and reward > 0) else "failure"
            print(f"[browsecomp][end][{qid}] outcome={outcome} reward={reward} attempts={attempt + 1}", flush=True)

            return last_verify_response
        except Exception as e:
            infra = _is_infrastructure_failure(e)
            print(
                f"[browsecomp][abort][{qid}] error_type={type(e).__name__} "
                f"class={'infra' if infra else 'sample'} error={str(e)[:300]}",
                flush=True,
            )
            # CONTAIN the failure to this one sample. Re-raising makes /run return 500, which
            # the collector treats as fatal (raise_for_status in nemo_gym/rollouts/rollout_collection.py)
            # and every remaining sample dies with it — that cost two full multi-node
            # allocations, each dying part-way through a run on one bad sample.
            #
            # Score it 0, but carry `agent_error` so the failure stays COUNTABLE in the results
            # instead of being indistinguishable from a genuine wrong answer. Always check that
            # count before reading accuracy: silent zeros would have hidden the malformed
            # tool-arguments bug instead of surfacing it.
            if last_response_json is None:
                last_response_json = NeMoGymResponse(
                    id="resp_agent_error",
                    created_at=0.0,
                    model=self.config.model_server.name,
                    object="response",
                    output=[],
                    parallel_tool_calls=False,
                    tool_choice="none",
                    tools=[],
                ).model_dump()
            # A harness failure is not a scored model outcome. Gym's manifest-backed collector
            # persists nonterminal kill_shaped in the failure sidecar without charging the
            # counted-failure budget, leaving it eligible for resume. Retain NG_NO_PERSIST_KEY
            # for legacy consumers.
            # Merged LAST so a stale sentinel echoed from `body` cannot shadow the fresh value.
            routing = {NG_NO_PERSIST_KEY: True, NG_FAILURE_CLASS_KEY: "kill_shaped"} if infra else {}
            return BrowsecompAgentVerifyResponse.model_validate(
                body.model_dump()
                | {
                    "response": last_response_json,
                    "reward": 0.0,
                    "agent_error": f"{type(e).__name__}: {str(e)[:300]}",
                    "timed_out": 0,
                }
                | routing
            )

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

    def _extract_last_rounds(self, new_outputs, n=None):
        """
        Extract the last n complete tool-call rounds from new_outputs.
        A round = one or more function_call items + their corresponding
        function_call_output items. Returns a flat list preserving order.
        n defaults to context_reset_keep_rounds; the /tokenize adaptive-shrink
        path passes smaller n to find a window that fits under the threshold.
        """
        if n is None:
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

    def _save_snapshot(self, messages, task_index, attempt, reset_count, is_final):
        """Save a JSONL snapshot of the full conversation at a context reset or
        at trajectory end, keyed by task_index/attempt. Lets us inspect the
        pre-reset context (otherwise lost when history is trimmed).
        (ported from gym-gitlab fe9845ee)"""
        sample_dir = Path(f"{self.config.snap_dir}/sample_{task_index}")
        sample_dir.mkdir(parents=True, exist_ok=True)
        if is_final:
            sample_path = f"{sample_dir}/attempt_{attempt}_final.jsonl"
        else:
            sample_path = f"{sample_dir}/attempt_{attempt}_reset_{reset_count}.jsonl"
        with open(sample_path, "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(msg.model_dump_json() + "\n")

    def _save_trajectory(
        self,
        input_messages,
        full_trajectory,
        task_index,
        attempt,
        reset_steps,
        reset_count,
        num_tool_calls,
        pre_reset_warning_steps=None,
    ):
        """Save the FULL untrimmed conversation for one sample to
        {snap_dir}/sample_{task_index}/attempt_{attempt}_trajectory.jsonl.
        Line 1 = metadata header; remaining lines = input prefix + every model/tool item, in order
        (never trimmed at context resets). (reference-harness parity: one trajectory.jsonl per sample.)"""
        sample_dir = Path(f"{self.config.snap_dir}/sample_{task_index}")
        sample_dir.mkdir(parents=True, exist_ok=True)
        path = f"{sample_dir}/attempt_{attempt}_trajectory.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "type": "metadata",
                        "task_index": task_index,
                        "attempt": attempt,
                        "reset_count": reset_count,
                        "num_tool_calls": num_tool_calls,
                        "reset_steps": reset_steps,
                        "pre_reset_warning_steps": pre_reset_warning_steps or [],
                    }
                )
                + "\n"
            )
            for msg in list(input_messages) + full_trajectory:
                f.write(msg.model_dump_json() + "\n")

    async def _count_prompt_tokens(self, body) -> int:
        """Hit the policy model's vLLM /tokenize endpoint and return the prompt
        token count (used to decide context reset without a full generation).
        (ported from gym-gitlab b66e37c6)"""
        converter = ChatCompletionsConverter(return_token_id_information=False)
        chat_completion_create_params = converter.responses_to_chat_completion_create_params(body)
        chat_completion_create_params = chat_completion_create_params.model_dump()
        # Same projection as vllm_model/app.py's tokenize path.
        tokenize_body_dict = {}
        for key in ("model", "messages", "tools", "chat_template_kwargs"):
            if key in chat_completion_create_params:
                tokenize_body_dict[key] = chat_completion_create_params[key]
        tokenize_response = await self._policy_model_openai_client.create_tokenize(**tokenize_body_dict)
        return _prompt_tokens_from_tokenize_response(tokenize_response)


def _prompt_tokens_from_tokenize_response(tokenize_response: dict) -> int:
    """Prompt token count from a vLLM /tokenize response.

    The response shape varies across vLLM versions: mainstream builds
    (>=0.19.1) return {"count": N, "max_model_len": ...} and only include the
    "tokens" list when token ids are requested, while older builds return just
    {"tokens": [...]}. Prefer the explicit count and fall back to the token
    list, so the agent works against both.
    """
    count = tokenize_response.get("count")
    if count is not None:
        return int(count)
    tokens = tokenize_response.get("tokens")
    if tokens is not None:
        return len(tokens)
    raise KeyError(f"/tokenize response has neither 'count' nor 'tokens'; got keys: {sorted(tokenize_response)}")


def _configure_logging() -> None:
    """Show this server's INFO diagnostics (sandbox lifecycle, cache and phase lines) by default.

    Root stays at WARNING: at INFO the HTTP client logs a line per call. NEMO_GYM_LOG_LEVEL overrides
    the level of this server's own loggers, for example WARNING to silence the diagnostics."""
    logging.basicConfig(level=logging.WARNING)
    for logger_name in (__name__, "responses_api_agents.browsecomp_agent"):
        logging.getLogger(logger_name).setLevel(os.environ.get("NEMO_GYM_LOG_LEVEL", "INFO").upper())


if __name__ == "__main__":
    _configure_logging()
    BrowsecompAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    _configure_logging()
    app = BrowsecompAgent.run_webserver()  # noqa: F401
