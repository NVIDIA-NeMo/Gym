# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Multi-turn text-action transport agent for the VisGym resources server.

The agent extracts the model's action from the LAST
`\\boxed{...}` token in the assistant's plain text and posts to VisGym's
`/step` endpoint with `action_string`. It uses plain-text actions rather than tool calls.
"""

import asyncio
import json
import logging
import math
import re
import uuid
from collections.abc import Sequence
from typing import Any, Literal, cast

import aiohttp
import anyio
from pydantic import ConfigDict, Field, ValidationError, model_validator

from nemo_gym.agents import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.agents.context_management import ContextGuardRejected, ContextHistoryConfig
from nemo_gym.client import raise_for_status
from nemo_gym.config import TASK_INDEX_KEY_NAME, ModelServerRef, ResourcesServerRef
from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputItem,
    accumulate_response_usage,
)
from nemo_gym.server_runtime import BaseRunRequest
from resources_servers.visgym.schemas import (
    VisGymAgentVerifyRequest,
    VisGymAgentVerifyResponse,
    VisGymCloseResponse,
    VisGymEnvStateEasyInputMessage,
    VisGymNeMoGymResponse,
    VisGymSeedSessionResponse,
    VisGymStepRequest,
    VisGymStepResponse,
    VisGymTaskRow,
)
from responses_api_agents.visual_context import visual_context_client


logger = logging.getLogger(__name__)

_MUTATION_MAX_TRIES = 1
_CLEANUP_MAX_TRIES = 3
_CLEANUP_MAX_SEMANTIC_ATTEMPTS = 3


DEFAULT_VISGYM_SYSTEM_PROMPT = (
    "Inspect the current visual state and choose exactly one legal VisGym action. "
    "Put the action in \\boxed{...} as the final item in your response."
)


def _as_core_input_message(message: Any) -> Any:
    """Downcast a VisGym observation to the message type the core union declares.

    ``VisGymEnvStateEasyInputMessage`` subclasses ``NeMoGymEasyInputMessage`` to
    carry ``env_info``. NeMo-Gym's request-side union
    (``NeMoGymResponseCreateParamsNonStreaming.input``) does not list that
    subclass, so a single foreign item makes Pydantic serialize the *whole*
    input list against the base union members. That silently strips
    ``prompt_token_ids``/``generation_token_ids`` from the assistant turns, and
    the model server then has no exact prefix to replay — every turn after the
    first becomes off-policy without any error surfacing.

    Observations therefore enter the model request as plain messages.
    ``env_info`` is not lost: it stays on the objects kept in ``seed_obs`` and
    in the rollout output, which are typed by VisGym's own union.
    """
    if isinstance(message, VisGymEnvStateEasyInputMessage):
        return NeMoGymEasyInputMessage(role=message.role, content=message.content, type=message.type)
    return message


# Brace-depth scanner ported from resources_servers/string_match/app.py's
# _extract_boxed (Doc 2 Revision R7). Ported rather than imported because
# visgym_agent runs in its own per-server venv and importing from
# string_match would couple the two servers' dependency closures.
#
# The naive r"\\boxed\{\s*(.*?)\s*\}" alternative -- what this file used to
# define as BOXED_PATTERN -- is non-greedy and stops at the FIRST closing
# brace, so it truncates any nested-brace content, most commonly a model
# wrapping its action in \\boxed{\\text{('move', 0)}}: it would capture
# "\\text{('move', 0)" (an unbalanced fragment) instead of "('move', 0)". A
# regression test (test_pattern_constant_matches_string_match) asserted this
# constant against its own literal, so it could never have caught the two
# implementations diverging; it now round-trips a nested-brace string through
# both extractors instead.
BOXED_START_PATTERN = re.compile(r"\\boxed\{")
LATEX_TEXT_WRAP = re.compile(r"\\text\{\s*(.*?)\s*\}", re.S)
TERMINAL_CHAT_TOKENS = ("<|im_end|>", "<|eot_id|>")


def _strip_latex_wrappers(value: str) -> str:
    while True:
        match = LATEX_TEXT_WRAP.fullmatch(value)
        if not match:
            break
        value = match.group(1)
    return value


def _iter_boxed_contents(text: str) -> list[str]:
    """Return every top-level ``\\boxed{...}`` payload, in order, brace-balanced."""
    contents = []
    for match in BOXED_START_PATTERN.finditer(text):
        depth = 1
        inner_start = match.end()
        for index, char in enumerate(text[inner_start:], start=inner_start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            if depth == 0:
                inner = text[inner_start:index].strip()
                contents.append(_strip_latex_wrappers(inner).strip())
                break
    return contents


class TextActionAgentConfig(BaseResponsesAPIAgentConfig):
    context_history: ContextHistoryConfig | None = None

    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    max_steps: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Hard cap on rollout turns. Per-env horizon_cap in the Doc-1 JSONL "
            "row is the primary enforcement point; this is a global backstop."
        ),
    )
    return_transitions: Literal[False] = Field(
        default=False,
        description="Pinned False per Doc 1 R3 — inspector consumes a flat output.",
    )
    max_total_sequence_length: int | None = Field(
        default=None,
        description=(
            "If set, the rollout will stop when the agent state exceeds this "
            "length. If not set, will rely on a vLLM exception to tell us when "
            "we've exceeded the model's token limit. Setting this simply avoids "
            "that exception."
        ),
    )
    done_if_no_boxed_answer: bool = Field(
        default=False,
        description=(
            "Symmetric counterpart of AviaryAgent.done_if_no_tool_calls. When "
            "False, a model response with no \\boxed{} match triggers a "
            "client-side recovery user message (no /step call); when True, the "
            "rollout terminates instead."
        ),
    )
    unboxed_action_regex: str | None = Field(
        default=None,
        description=(
            "Optional full-match regex for accepting a bare action when no "
            "\\boxed{} answer is present. Disabled by default; use only for "
            "environments whose legal action grammar is strict."
        ),
    )
    max_no_boxed_truncation_retries: int = Field(
        default=0,
        description=(
            "When done_if_no_boxed_answer=False, allow up to this many "
            "same-state retries with a larger max_output_tokens budget, but "
            "only if the previous response ended due to max_output_tokens "
            "truncation and still lacked a \\boxed{...} action."
        ),
    )
    no_boxed_truncation_retry_factor: float = Field(
        default=2.0,
        description=(
            "Multiplier applied to max_output_tokens on each truncation-driven "
            "same-state retry after a missing \\boxed{...} action."
        ),
    )
    re_emit_rules_each_turn: bool = Field(
        default=False,
        description=(
            "If True, prepend a one-line action-vocabulary summary to every env "
            "user turn obs. Default off matches Pattern A (rules in first turn "
            "only); see Doc 2 § re_emit_rules_each_turn flag."
        ),
    )
    rules_summary_template: str = Field(
        default="Reminder: respond with your reasoning, then write your action as \\boxed{...}.",
        description=(
            "Template used when re_emit_rules_each_turn=True. Per-env "
            "specialization is a Doc-7 follow-on; default is path-generic."
        ),
    )
    system_prompt: str = Field(
        default=DEFAULT_VISGYM_SYSTEM_PROMPT,
        description=(
            "VisGym system prompt. Injected agent-side (in `responses()`) as "
            "the first message of the conversation if the incoming JSONL row "
            "doesn't already start with a system message. Default is the "
            "built-in generic VisGym prompt. "
            "Keeping the prompt in agent code (not per-row JSONL) means a "
            "single edit changes every rollout, every probe JSONL. Override "
            "via the agent's YAML config when you want a per-experiment "
            "variant without regenerating data."
        ),
    )

    @model_validator(mode="after")
    def _validate_no_boxed_retry_config(self) -> "TextActionAgentConfig":
        if self.max_no_boxed_truncation_retries < 0:
            raise ValueError("visgym_agent.max_no_boxed_truncation_retries must be >= 0.")
        if self.no_boxed_truncation_retry_factor < 1.0:
            raise ValueError("visgym_agent.no_boxed_truncation_retry_factor must be >= 1.0.")
        if self.done_if_no_boxed_answer and self.max_no_boxed_truncation_retries > 0:
            raise ValueError(
                "visgym_agent.done_if_no_boxed_answer=true is incompatible with max_no_boxed_truncation_retries > 0."
            )
        if self.unboxed_action_regex is not None:
            try:
                re.compile(self.unboxed_action_regex)
            except re.error as exc:
                raise ValueError("visgym_agent.unboxed_action_regex must be a valid regex") from exc
        return self


class TextActionAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    task_idx: int | None = Field(default=None, ge=0)
    task_index: int | None = Field(default=None, alias=TASK_INDEX_KEY_NAME, ge=0)
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )

    @model_validator(mode="after")
    def require_task_index(self) -> "TextActionAgentRunRequest":
        if self.task_index is None and self.task_idx is None:
            raise ValueError(f"Either {TASK_INDEX_KEY_NAME} or legacy task_idx must be provided.")
        return self

    @property
    def group_task_index(self) -> int:
        """Canonical collated-row identity, with legacy compatibility."""
        return self.task_index if self.task_index is not None else cast(int, self.task_idx)


class TextActionAgent(SimpleResponsesAPIAgent):
    """Run VisGym's multimodal, boxed-text action protocol.

    The generic ``gymnasium_agent`` handles string observations and function
    calls. This agent additionally preserves response token metadata between
    image turns, retries truncated boxed actions, and performs VisGym's final
    reward drain, so substituting the generic agent would make training
    trajectories semantically different.
    """

    config: TextActionAgentConfig

    async def _close_session(self, env_id: str, *, discard_reward: bool) -> None:
        """Best-effort close, optionally aborting the pending reward drain."""
        payload: dict[str, Any] = {"env_id": env_id}
        if discard_reward:
            payload["discard_reward"] = True

        async def close_with_retries() -> None:
            last_failure = "unknown failure"
            for attempt in range(1, _CLEANUP_MAX_SEMANTIC_ATTEMPTS + 1):
                try:
                    response = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/close",
                        json=payload,
                        _max_num_tries=_CLEANUP_MAX_TRIES,
                    )
                    await raise_for_status(response)
                    close_response = VisGymCloseResponse.model_validate(await response.json())
                    if close_response.success:
                        return
                    last_failure = close_response.message or "server returned success=false"
                except Exception as exc:
                    last_failure = f"{type(exc).__name__}: {exc}"
                if attempt < _CLEANUP_MAX_SEMANTIC_ATTEMPTS:
                    logger.warning(
                        "VisGym close failed for session %s (attempt %d/%d): %s; retrying",
                        env_id,
                        attempt,
                        _CLEANUP_MAX_SEMANTIC_ATTEMPTS,
                        last_failure,
                    )
            logger.warning(
                "VisGym close failed for session %s after %d attempts: %s",
                env_id,
                _CLEANUP_MAX_SEMANTIC_ATTEMPTS,
                last_failure,
            )

        # A CancelScope does not shield against raw asyncio.Task.cancel(). Run
        # cleanup in its own task and delay propagating cancellation until that
        # bounded task has finished.
        cleanup_task = asyncio.create_task(close_with_retries())
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError as cancellation:
            # An AnyIO disconnect cancel scope re-cancels every unshielded await;
            # raw Task.cancel() can also be delivered more than once. Keep the
            # detached, bounded cleanup alive through either mechanism.
            with anyio.CancelScope(shield=True):
                while not cleanup_task.done():
                    try:
                        await asyncio.shield(cleanup_task)
                    except asyncio.CancelledError:
                        continue
            try:
                cleanup_task.result()
            except BaseException:
                logger.warning("Failed to finish closing orphaned VisGym session %s", env_id, exc_info=True)
            raise cancellation

    async def _seed_session(self, task_idx: int | None, task_row: VisGymTaskRow | None) -> VisGymSeedSessionResponse:
        requested_env_id = str(uuid.uuid4())
        payload: dict[str, Any] = {"env_id": requested_env_id}
        if task_row is not None:
            payload["task_row"] = task_row.model_dump(mode="json")
        elif task_idx is not None:
            payload["task_idx"] = task_idx
        else:
            raise ValueError(
                f"A canonical {TASK_INDEX_KEY_NAME} requires an inline task row; "
                "only legacy task_idx can address the resources server's local task table."
            )
        returned_env_id: str | None = None
        try:
            reset_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=payload,
                _max_num_tries=_MUTATION_MAX_TRIES,
            )
            # Not aiohttp's bare raise_for_status(): the exception middleware
            # requires response_content on ClientResponseError.
            await raise_for_status(reset_response)
            payload_json = await reset_response.json()
            if isinstance(payload_json, dict) and isinstance(payload_json.get("env_id"), str):
                returned_env_id = payload_json["env_id"]
            seed_session_response = VisGymSeedSessionResponse.model_validate(payload_json)
            if not seed_session_response.obs:
                raise ValueError("No observations in seed session response")
            return seed_session_response
        except BaseException as exc:
            # A 409 is a definitive rejection of our fresh UUID: the server
            # explicitly did not create this session. Closing that ID could
            # destroy the pre-existing session that caused the collision.
            if isinstance(exc, aiohttp.ClientResponseError) and exc.status == 409:
                raise
            # A lost response is ambiguous: the server may have published the
            # session. Because the request ID is client-known, cleanup can name
            # it without replaying /seed_session.
            await self._close_session(requested_env_id, discard_reward=True)
            if returned_env_id is not None and returned_env_id != requested_env_id:
                await self._close_session(returned_env_id, discard_reward=True)
            raise

    @staticmethod
    def _extract_assistant_text(output: list[NeMoGymResponseOutputItem]) -> str:
        """Concatenate assistant text content parts across all assistant
        messages in the response, in order. Mirrors string_match's
        `_extract_last_assistant_text` but takes the raw output list.
        """
        texts: list[str] = []
        for item in output:
            if getattr(item, "type", None) != "message":
                continue
            if getattr(item, "role", None) != "assistant":
                continue
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for part in content:
                    text_value = getattr(part, "text", None)
                    if isinstance(text_value, str):
                        texts.append(text_value)
            elif isinstance(content, str):
                texts.append(content)
        return "\n".join(texts)

    @staticmethod
    def _extract_boxed(text: str) -> str | None:
        """Return the LAST `\\boxed{...}` capture in `text`, stripped.

        Matches `string_match._extract_boxed` semantics: reasoning models often
        write a `\\boxed{example}` earlier in their CoT and a final
        `\\boxed{decision}` at the end, and the last match is the one that
        counts. Returns None for both "no `\\boxed{}` found" and "explicitly
        empty `\\boxed{}`" — the latter is treated as a recoverable failure
        rather than passed verbatim to /step (where it would fail the env's
        parser with a less useful error message).
        """
        matches = _iter_boxed_contents(text)
        if not matches:
            return None
        return matches[-1] or None

    @staticmethod
    def _extract_action(text: str, unboxed_action_regex: str | None = None) -> str | None:
        boxed = TextActionAgent._extract_boxed(text)
        if boxed is not None:
            return boxed
        if unboxed_action_regex is None:
            return None
        candidate = text.strip()
        for token in TERMINAL_CHAT_TOKENS:
            if candidate.endswith(token):
                candidate = candidate[: -len(token)].strip()
                break
        if re.fullmatch(unboxed_action_regex, candidate):
            return candidate
        return None

    def _no_boxed_recovery(self) -> NeMoGymEasyInputMessage:
        return NeMoGymEasyInputMessage(
            role="user",
            content=(
                "Your previous attempt produced no \\boxed{...} action. "
                "Answer concisely and end with \\boxed{...} on the last line."
            ),
        )

    @staticmethod
    def _incomplete_reason(model_response: NeMoGymResponse | None) -> str | None:
        incomplete_details = getattr(model_response, "incomplete_details", None)
        return getattr(incomplete_details, "reason", None)

    @staticmethod
    def _scaled_max_output_tokens(
        base_max_output_tokens: int | None,
        retry_factor: float,
        retry_index: int,
    ) -> int | None:
        if base_max_output_tokens is None:
            return None
        return max(
            1,
            int(math.ceil(base_max_output_tokens * (retry_factor**retry_index))),
        )

    def _maybe_prepend_system_prompt(
        self, input_messages: Sequence[NeMoGymEasyInputMessage]
    ) -> list[NeMoGymEasyInputMessage]:
        """Prepend `self.config.system_prompt` as the first system message.

        Idempotent: if `input_messages` already begins with a system-role
        message, returns the list unchanged. This lets per-row JSONLs
        override the agent's default prompt by supplying their own system
        message — useful for ablations that want a non-default prompt
        without changing the agent's config.

        If `system_prompt` is empty, no message is prepended.
        """
        prompt = self.config.system_prompt
        if not prompt:
            return list(input_messages)
        existing = list(input_messages)
        if existing and isinstance(existing[0], dict) and existing[0].get("role") == "system":
            return existing
        if existing and hasattr(existing[0], "role") and getattr(existing[0], "role") == "system":
            return existing
        sys_msg = NeMoGymEasyInputMessage(role="system", content=prompt)
        return [sys_msg, *existing]

    def _maybe_inject_rules_summary(self, obs: Sequence[NeMoGymEasyInputMessage]) -> list[NeMoGymEasyInputMessage]:
        if not self.config.re_emit_rules_each_turn:
            return list(obs)
        # We intentionally do NOT mutate the env-state message itself: Doc 1 R5
        # makes VisGymEnvStateEasyInputMessage the single source of truth for env
        # state, and we don't want to perturb its env_info field. Append a
        # fresh reminder AFTER the env-state message instead, so the reminder
        # is the most-recent context the model sees before generating —
        # necessary because the env's per-turn text often contains its own
        # action-format hint (e.g. FrozenLake's "Type your action as: [up]")
        # which would otherwise win on recency over a pre-pended reminder.
        #
        # Role: `user`. We probed three variants on the format-probe (32-row)
        # trajectory:
        #   - role=user (job 11732200):       FL=0.229  GoL=0.354  leak=3/32
        #   - no reminder  (job 11732812):    FL=0.104  GoL=0.135  leak=3/32
        #   - role=system  (job 11732968):    FL=0.156  GoL=0.240  leak=1/32
        # `user` wins clearly on env reward despite a slightly higher leak
        # rate; `system` cuts the leak but the model also uses ~25% more
        # tokens and trips truncation more often. For trajectory collection
        # where the metric is task success, `user` is the right default.
        reminder = NeMoGymEasyInputMessage(role="user", content=self.config.rules_summary_template)
        return [*obs, reminder]

    @staticmethod
    def _task_row_from_request(req: TextActionAgentRunRequest) -> VisGymTaskRow | None:
        payload = req.model_dump(mode="json")
        if "env_id" not in payload or "seed" not in payload:
            return None
        try:
            return VisGymTaskRow.model_validate(payload)
        except ValidationError:
            logger.exception("Incoming run request had task-row fields but failed validation.")
            raise

    async def responses(self, req: TextActionAgentRunRequest) -> VisGymNeMoGymResponse:
        response, _ = await self._run_episode(req)
        return response

    async def _run_episode(
        self, req: TextActionAgentRunRequest
    ) -> tuple[VisGymNeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming]:
        """Play one episode and return it alongside the params the policy saw.

        The effective params are not the caller's: this method deep-copies the
        request and prepends the agent's system prompt, so the conversation the
        model was actually conditioned on differs from the task row (whose
        `input` is empty for every shipped config). NeMo-RL rebuilds a
        rollout's initial prompt from `responses_create_params.input` on the
        verify result, so echoing the caller's copy there reconstructs an empty
        prefix that disagrees with the recorded prompt_token_ids -- exactly the
        off-policy mismatch the token metadata exists to prevent.
        """
        task_row = self._task_row_from_request(req)
        group_task_index = req.group_task_index
        req = req.model_copy(deep=True)
        body = req.responses_create_params

        # The resources server can enforce horizon_cap only on calls that
        # reach /step. Same-state recovery after a truncated response with no
        # boxed action never reaches the env, so cap model turns here too.
        # This mirrors Gym-V and keeps a short inline-row horizon authoritative
        # even when the global agent backstop is larger (or unset).
        effective_max_steps = self.config.max_steps
        if task_row is not None and task_row.horizon_cap is not None:
            effective_max_steps = (
                task_row.horizon_cap if effective_max_steps is None else min(effective_max_steps, task_row.horizon_cap)
            )

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        # Inject the agent's system prompt as the first message of the
        # conversation, unless the caller already provided one. This keeps
        # the system prompt out of every per-row JSONL — a single field on
        # the agent config controls the prompt for all rollouts. Tested
        # empirically: the OpenAI Responses → chat-completions converter in
        # vllm_model/app.py silently drops the top-level `instructions`
        # field, so embedding the prompt as a system *message* is the only
        # way to actually deliver it to the model (verified in jobs
        # 11731129, 11731385).
        body.input = self._maybe_prepend_system_prompt(body.input)

        seed_session_response = await self._seed_session(req.task_idx, task_row)

        # Path B: tools is empty by Doc-2 commitment; if the JSONL row carries
        # tools (e.g., misconfigured), we don't strip them. The model can
        # ignore them — the agent never inspects function_call output items.
        # Apply rules-summary reminder to turn-1's seed observation too (when
        # re_emit_rules_each_turn is on), not just /step responses; otherwise
        # the very first turn — the most format-sensitive one — sees the env's
        # action-format hint without our counter-reminder and the model
        # imprints on the env's preferred wrapping.
        seed_obs = self._maybe_inject_rules_summary(seed_session_response.obs)
        agent_state = body.model_copy(
            update={
                "input": body.input + [_as_core_input_message(m) for m in seed_obs],
                "tools": list(body.tools or []),
            }
        )

        env_id = seed_session_response.env_id
        model_response: NeMoGymResponse | None = None
        total_usage = None
        all_messages: list[NeMoGymResponseOutputItem] = []
        # An observation belongs to the authoritative token trajectory only
        # after a subsequent model call succeeds and returns token metadata for
        # the prompt that contains it. Hold it pending so max-step fallthroughs
        # and failed model calls cannot publish a trailing user turn (usually a
        # screenshot) that the policy never consumed.
        pending_obs: list[VisGymEnvStateEasyInputMessage | NeMoGymEasyInputMessage] = []
        model_server_cookies = None
        base_max_output_tokens = body.max_output_tokens
        current_max_output_tokens = body.max_output_tokens
        consecutive_truncation_retries = 0
        total_truncation_retries = 0
        termination_reason = "unknown"

        # The task index has a Pydantic alias; use wire keys for Gym capture identity.
        model_url_path = self.url_path_for_run("/v1/responses", req.model_dump(by_alias=True))

        step = 0
        loop_completed = False
        context_rejected = False
        try:
            context_client = visual_context_client(
                self,
                config=self.config.context_history,
                logical_rollout_id=self.rollout_id_from_run(req.model_dump(by_alias=True)),
                initial_request=agent_state,
            )
            while True:
                if effective_max_steps is not None and step >= effective_max_steps:
                    termination_reason = "max_steps"
                    break
                step += 1
                reset_truncation_budget = False

                raw_model_response = None
                try:
                    request_body = agent_state.model_copy(update={"max_output_tokens": current_max_output_tokens})
                    if context_client is not None:
                        model_response_json = (await context_client.create(request_body)).model_dump()
                    else:
                        raw_model_response = await self.server_client.post(
                            server_name=self.config.model_server.name,
                            url_path=model_url_path,
                            json=request_body,
                            cookies=model_server_cookies,
                        )
                        # See the comment on the /seed_session raise_for_status
                        # call: aiohttp's bare method omits response_content,
                        # which the exception middleware asserts on.
                        await raise_for_status(raw_model_response)
                        model_server_cookies = raw_model_response.cookies
                        model_response_json = await raw_model_response.json()
                except ContextGuardRejected:
                    # A measured rejection generated no new action. A previous
                    # accepted response may terminate the episode; never invent one.
                    if model_response is None:
                        raise
                    context_rejected = True
                    model_response.metadata = dict(model_response.metadata or {}) | {
                        "termination_reason": "context_length_exceeded"
                    }
                    termination_reason = "context_length_exceeded"
                    break
                except (json.JSONDecodeError, aiohttp.ClientResponseError) as e:
                    if context_client is not None:
                        raise
                    response_text: Any = "<response unavailable>"
                    if raw_model_response is not None:
                        response_text = getattr(raw_model_response, "text", response_text)
                        if callable(response_text):
                            try:
                                response_text = await response_text()
                            except Exception:
                                response_text = "<response body unavailable>"
                    logger.warning("Error calling /v1/responses: %r. Response: %r.", e, response_text)
                    termination_reason = "model_error"
                    break

                try:
                    model_response = NeMoGymResponse.model_validate(model_response_json)
                    total_usage = accumulate_response_usage(total_usage, model_response.usage)
                except ValidationError as e:
                    if context_client is not None:
                        raise
                    logger.warning(f"Error validating model response: {e!r}. Response: {model_response_json!r}.")
                    termination_reason = "model_error"
                    break

                if pending_obs:
                    all_messages.extend(pending_obs)
                    pending_obs = []

                model_output = model_response.output
                assistant_text = self._extract_assistant_text(model_output)
                action_string = self._extract_action(assistant_text, self.config.unboxed_action_regex)
                incomplete_reason = self._incomplete_reason(model_response)

                done = False
                obs: Sequence[VisGymEnvStateEasyInputMessage | NeMoGymEasyInputMessage]
                if action_string is None:
                    if self.config.done_if_no_boxed_answer:
                        done = True
                        obs = []
                        termination_reason = "no_boxed_immediate"
                    else:
                        is_truncated = incomplete_reason == "max_output_tokens"
                        if (
                            is_truncated
                            and consecutive_truncation_retries < self.config.max_no_boxed_truncation_retries
                        ):
                            consecutive_truncation_retries += 1
                            total_truncation_retries += 1
                            current_max_output_tokens = self._scaled_max_output_tokens(
                                base_max_output_tokens,
                                self.config.no_boxed_truncation_retry_factor,
                                consecutive_truncation_retries,
                            )
                            obs = [self._no_boxed_recovery()]
                        else:
                            done = True
                            obs = []
                            termination_reason = "no_boxed_retry_cap" if is_truncated else "no_boxed_no_truncation"
                else:
                    step_request = VisGymStepRequest(env_id=env_id, action_string=action_string)
                    raw_env_response = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/step",
                        json=step_request.model_dump(exclude_none=True),
                        _max_num_tries=_MUTATION_MAX_TRIES,
                    )
                    await raise_for_status(raw_env_response)
                    env_response = VisGymStepResponse.model_validate(await raw_env_response.json())
                    obs = self._maybe_inject_rules_summary(env_response.obs)
                    done = env_response.done
                    if done:
                        termination_reason = "env_done"
                    elif any("Invalid action" in str(getattr(message, "content", "")) for message in env_response.obs):
                        reset_truncation_budget = False
                    else:
                        reset_truncation_budget = True
                    if reset_truncation_budget:
                        consecutive_truncation_retries = 0
                        current_max_output_tokens = base_max_output_tokens

                agent_state = agent_state.model_copy(
                    update={"input": agent_state.input + model_output + [_as_core_input_message(m) for m in obs]}
                )
                all_messages.extend(model_output)
                # Do not publish this observation yet. The next successful
                # model response proves that its prompt/token bundle covered
                # it, at which point the block above commits it. This applies
                # to both environment observations and synthetic no-boxed
                # recovery messages.
                if not done:
                    pending_obs = list(obs)

                if done:
                    break
            if context_client is not None and model_response is not None and not context_rejected:
                context_client.finish(model_response)
            loop_completed = True
        finally:
            if not loop_completed:
                await self._close_session(env_id, discard_reward=True)

        if model_response is None:
            await self._close_session(env_id, discard_reward=True)
            raise AssertionError("Rollout terminated before the first model response completed; cannot proceed.")

        response_ready = False
        try:
            output_overrides = {
                "env_id": env_id,
                "group_id": str(group_task_index),
                "contains_transitions": False,
                "usage": total_usage,
                "metadata": {
                    **(model_response.metadata or {}),
                    "termination_reason": termination_reason,
                    "no_boxed_truncation_retries": str(total_truncation_retries),
                },
                # seed_obs is the post-rules-injection initial observation that
                # vLLM saw before the first model call. Stored separately from
                # output so raw user messages (no token_ids) do not enter the
                # tokenized message-log flattening path. See
                # Preserve the initial observation for downstream trajectory reconstruction.
                "seed_obs": [m.model_dump(mode="json") if hasattr(m, "model_dump") else m for m in seed_obs],
                "output": all_messages,
            }
            response = VisGymNeMoGymResponse.model_validate(model_response.model_dump() | output_overrides)
            response_ready = True
        finally:
            await self._close_session(env_id, discard_reward=not response_ready)
        return response, body

    async def run(self, body: TextActionAgentRunRequest) -> VisGymAgentVerifyResponse:
        response: VisGymNeMoGymResponse | None = None
        try:
            response, effective_params = await self._run_episode(body)
            verify_request = VisGymAgentVerifyRequest.model_validate(
                {
                    "response": response.model_dump(),
                    # Echoed back through /verify because NeMo-RL reads it off
                    # the rollout result, not off the task row -- and it has to
                    # be the params the policy saw (system prompt injected),
                    # not the caller's untouched copy.
                    "responses_create_params": effective_params.model_dump(),
                }
            )
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
                _max_num_tries=_MUTATION_MAX_TRIES,
            )
            await raise_for_status(verify_response)
            return VisGymAgentVerifyResponse.model_validate(await verify_response.json())
        except BaseException as exc:
            if response is not None:
                await self._close_session(response.env_id, discard_reward=True)
            if isinstance(exc, Exception):
                logger.exception("Error in run")
            raise


if __name__ == "__main__":
    TextActionAgent.run_webserver()
