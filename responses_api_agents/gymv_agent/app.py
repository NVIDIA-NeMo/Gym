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
"""Multi-turn text-action agent for the Gym-V resources server.

The agent extracts the model's action from the LAST `\\boxed{...}` token in
the assistant's plain text and posts to Gym-V's `/step` endpoint with
`action_string`. Pairs with `resources_servers/gym_v/`.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import uuid
from collections.abc import Sequence
from pathlib import Path
from time import time
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
    NeMoGymResponseInput,
    NeMoGymResponseOutputItem,
    accumulate_response_usage,
)
from nemo_gym.server_runtime import BaseRunRequest
from resources_servers.gym_v.schemas import (
    GymVAgentVerifyRequest,
    GymVAgentVerifyResponse,
    GymVCloseResponse,
    GymVEnvStateEasyInputMessage,
    GymVNeMoGymResponse,
    GymVSeedSessionResponse,
    GymVStepRequest,
    GymVStepResponse,
    GymVTaskRow,
)
from responses_api_agents.visual_context import visual_context_client


_MUTATION_MAX_TRIES = 1
_CLEANUP_MAX_TRIES = 3
_CLEANUP_MAX_SEMANTIC_ATTEMPTS = 3


# Default system prompt: the paired gymv_agent imports this at runtime
# and injects it as the first system message when the incoming request
# doesn't already provide one.
DEFAULT_SYSTEM_PROMPT = """\
You are playing a game. Each turn, you receive an image of the current game
state and a short text description. Read the rules in the first user message
carefully — they specify the action grammar and the goal.

For every turn:
1. Think briefly inside the <think> block in 1-2 short sentences. Just state
   your observation and decision — do NOT enumerate grid rows, list options,
   or restate history.
2. Close the think block with </think>, then immediately write your final
   action wrapped in \\boxed{...} on the LAST line.
   The exact format inside \\boxed{...} is dictated by the env's rules.

Only the LAST \\boxed{...} in your response is sent to the game.
"""


def _as_core_input_message(message: Any) -> Any:
    """Keep environment metadata in rollout records, outside model requests.

    The core input union does not declare GymVEnvStateEasyInputMessage. Using
    that subclass changes semantic-history comparisons and can make Pydantic
    drop training fields from other messages during union serialization.
    """
    if isinstance(message, GymVEnvStateEasyInputMessage):
        return NeMoGymEasyInputMessage(role=message.role, content=message.content, type=message.type)
    return message


logger = logging.getLogger(__name__)


def _debug_jsonl_path(component: str) -> Path | None:
    debug_dir = os.environ.get("NEMO_RL_DEBUG_RESPONSES_PIPELINE_DIR")
    if not debug_dir:
        return None
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{component}.jsonl"


def _preview_text(value: Any, limit: int = 500) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[:limit] + f"...<truncated {len(text) - limit} chars>"


def _seq_summary(values: Any, *, limit: int = 12) -> dict[str, Any]:
    if values is None:
        return {"present": False, "len": 0}
    if not isinstance(values, list):
        return {"present": True, "type": type(values).__name__}
    return {
        "present": True,
        "len": len(values),
        "head": values[:limit],
        "tail": values[-limit:] if len(values) > limit else [],
    }


def _image_url_summary(image_url: Any) -> dict[str, Any]:
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    if not isinstance(image_url, str):
        return {"present": False, "type": type(image_url).__name__}
    return {
        "present": True,
        "len": len(image_url),
        "sha256_16": hashlib.sha256(image_url.encode("utf-8")).hexdigest()[:16],
        "prefix": image_url[:32],
    }


def _content_summary(content: Any) -> Any:
    if isinstance(content, str):
        return {"kind": "str", "len": len(content), "preview": _preview_text(content)}
    if not isinstance(content, list):
        return {"kind": type(content).__name__, "repr": _preview_text(content)}
    parts = []
    for part in content:
        if hasattr(part, "model_dump"):
            part = part.model_dump(mode="json")
        if not isinstance(part, dict):
            parts.append({"kind": type(part).__name__, "repr": _preview_text(part)})
            continue
        summary = {"type": part.get("type"), "keys": sorted(part.keys())}
        if "text" in part:
            text = part.get("text")
            summary["text_len"] = len(text) if isinstance(text, str) else None
            summary["text_preview"] = _preview_text(text)
        if "image_url" in part:
            summary["image_url"] = _image_url_summary(part.get("image_url"))
        parts.append(summary)
    return {"kind": "list", "len": len(content), "parts": parts}


def _message_summary(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        message = message.model_dump(mode="json")
    if not isinstance(message, dict):
        return {"type": type(message).__name__, "repr": _preview_text(message)}
    return {
        "keys": sorted(message.keys()),
        "id": message.get("id"),
        "role": message.get("role"),
        "type": message.get("type"),
        "content": _content_summary(message.get("content")),
        "summary": _content_summary(message.get("summary")),
        "prompt_token_ids": _seq_summary(message.get("prompt_token_ids")),
        "generation_token_ids": _seq_summary(message.get("generation_token_ids")),
        "generation_log_probs": _seq_summary(message.get("generation_log_probs")),
    }


def _debug_enabled() -> bool:
    """Gate expensive message/image summaries before their payload is built."""
    return bool(os.environ.get("NEMO_RL_DEBUG_RESPONSES_PIPELINE_DIR"))


def _debug_dump(component: str, event: str, payload: dict[str, Any]) -> None:
    path = _debug_jsonl_path(component)
    if path is None:
        return
    row = {"event": event, "created_at": time(), **payload}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


# Matches the last `\boxed{...}` in the assistant's plain text.
BOXED_PATTERN = re.compile(r"\\boxed\{\s*(.*?)\s*\}", re.S)


class GymVAgentConfig(BaseResponsesAPIAgentConfig):
    context_history: ContextHistoryConfig | None = None

    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    max_steps: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Global hard cap on model turns. When a task row supplies "
            "horizon_cap, the lower of the two limits is used so invalid-action "
            "and no-boxed recovery turns cannot bypass the task horizon."
        ),
    )
    return_transitions: Literal[False] = Field(
        default=False,
        description="Pinned False — inspector consumes a flat output.",
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
            "When False, a model response with no \\boxed{} match triggers a "
            "client-side recovery user message (no /step call); when True, the "
            "rollout terminates instead."
        ),
    )
    re_emit_rules_each_turn: bool = Field(
        default=False,
        description=(
            "If True, prepend a one-line action-vocabulary summary to every env "
            "user turn obs. Default off matches the pattern of putting rules in "
            "the first turn only."
        ),
    )
    rules_summary_template: str = Field(
        default="Reminder: respond with your reasoning, then write your action as \\boxed{...}.",
    )
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        description=(
            "System prompt. Injected agent-side (in `responses()`) as the first "
            "message of the conversation if the incoming JSONL row doesn't "
            "already start with a system message. Default is DEFAULT_SYSTEM_PROMPT "
            "from resources_servers/gym_v/app.py — keeping the prompt in agent "
            "code (not per-row JSONL) means a single edit changes every rollout. "
            "Override via the agent's YAML config when you want a per-experiment "
            "variant without regenerating data."
        ),
    )


class GymVAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    task_idx: int | None = Field(default=None, ge=0)
    task_index: int | None = Field(default=None, alias=TASK_INDEX_KEY_NAME, ge=0)
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )

    @model_validator(mode="after")
    def require_task_index(self) -> "GymVAgentRunRequest":
        if self.task_index is None and self.task_idx is None:
            raise ValueError(f"Either {TASK_INDEX_KEY_NAME} or legacy task_idx must be provided.")
        return self

    @property
    def group_task_index(self) -> int:
        """Canonical collated-row identity, with legacy compatibility."""
        return self.task_index if self.task_index is not None else cast(int, self.task_idx)


class GymVAgent(SimpleResponsesAPIAgent):
    config: GymVAgentConfig

    async def _close_session(self, env_id: str, *, discard_reward: bool) -> None:
        """Best-effort idempotent close with bounded transport and native-close retries."""
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
                    close_response = GymVCloseResponse.model_validate(await response.json())
                    if close_response.success:
                        return
                    last_failure = close_response.message or "server returned success=false"
                except Exception as exc:
                    # /close is idempotent. HTTP errors and malformed 2xx bodies
                    # are just as retryable as an explicit success=false body.
                    last_failure = f"{type(exc).__name__}: {exc}"
                if attempt < _CLEANUP_MAX_SEMANTIC_ATTEMPTS:
                    logger.warning(
                        "Gym-V close failed for session %s (attempt %d/%d): %s; retrying",
                        env_id,
                        attempt,
                        _CLEANUP_MAX_SEMANTIC_ATTEMPTS,
                        last_failure,
                    )
            logger.warning(
                "Gym-V close failed for session %s after %d attempts: %s",
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
            # Client disconnects arrive through AnyIO's level-triggered cancel
            # scopes, while tests and shutdown paths may use raw Task.cancel().
            # Shield the detached cleanup from both until its bounded retries end.
            with anyio.CancelScope(shield=True):
                while not cleanup_task.done():
                    try:
                        await asyncio.shield(cleanup_task)
                    except asyncio.CancelledError:
                        continue
            try:
                cleanup_task.result()
            except BaseException:
                # close_with_retries is best-effort and normally consumes its
                # own errors. Keep the caller's cancellation authoritative.
                logger.warning("Failed to finish closing Gym-V session %s", env_id, exc_info=True)
            raise cancellation

    async def _seed_session(self, task_idx: int | None, task_row: GymVTaskRow | None) -> GymVSeedSessionResponse:
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
            await raise_for_status(reset_response)
            payload_json = await reset_response.json()
            if isinstance(payload_json, dict) and isinstance(payload_json.get("env_id"), str):
                returned_env_id = payload_json["env_id"]
            seed_session_response = GymVSeedSessionResponse.model_validate(payload_json)
            if not seed_session_response.obs:
                raise ValueError("No observations in seed session response")
            return seed_session_response
        except BaseException as exc:
            # A 409 is a definitive rejection of our fresh UUID: the server
            # explicitly did not create this session. Closing that ID could
            # destroy the pre-existing session that caused the collision.
            if isinstance(exc, aiohttp.ClientResponseError) and exc.status == 409:
                raise
            await self._close_session(requested_env_id, discard_reward=True)
            if returned_env_id is not None and returned_env_id != requested_env_id:
                await self._close_session(returned_env_id, discard_reward=True)
            raise

    @staticmethod
    def _extract_assistant_text(output: list[NeMoGymResponseOutputItem]) -> str:
        """Concatenate assistant text content parts across all assistant
        messages in the response, in order.
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

        Reasoning models often write a `\\boxed{example}` earlier in their CoT
        and a final `\\boxed{decision}` at the end, and the last match is the
        one that counts. Returns None for both "no `\\boxed{}` found" and
        "explicitly empty `\\boxed{}`" — the latter is treated as a
        recoverable failure rather than passed verbatim to /step (where it
        would fail the env's parser with a less useful error message).
        """
        matches = BOXED_PATTERN.findall(text)
        if not matches:
            return None
        return matches[-1].strip() or None

    def _no_boxed_recovery(self) -> NeMoGymEasyInputMessage:
        return NeMoGymEasyInputMessage(
            role="user",
            content=(
                "I did not find a \\boxed{...} answer in your response. "
                "Please put your final action inside \\boxed{...} on the last line."
            ),
        )

    def _maybe_prepend_system_prompt(
        self, input_messages: Sequence[NeMoGymEasyInputMessage]
    ) -> list[NeMoGymEasyInputMessage]:
        """Prepend `self.config.system_prompt` as the first system message.

        Idempotent: if `input_messages` already begins with a system-role
        message, returns the list unchanged. This lets per-row JSONLs
        override the agent's default prompt by supplying their own system
        message. If `system_prompt` is empty, no message is prepended.
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
        # Append a fresh reminder AFTER the env-state message rather than
        # mutating it: env-state messages are the single source of truth for
        # env state, and per-turn env text often carries its own action-format
        # hint that would otherwise win on recency over a pre-pended reminder.
        reminder = NeMoGymEasyInputMessage(role="user", content=self.config.rules_summary_template)
        return [*obs, reminder]

    @staticmethod
    def _task_row_from_request(req: GymVAgentRunRequest) -> GymVTaskRow | None:
        payload = req.model_dump(mode="json")
        if "env_id" not in payload or "seed" not in payload:
            return None
        try:
            return GymVTaskRow.model_validate(payload)
        except ValidationError:
            logger.exception("Incoming run request had task-row fields but failed validation.")
            raise

    async def responses(self, req: GymVAgentRunRequest) -> GymVNeMoGymResponse:
        task_row = self._task_row_from_request(req)
        group_task_index = req.group_task_index
        req = req.model_copy(deep=True)
        body = req.responses_create_params

        # The resources server can enforce horizon_cap only when /step is
        # reached. Invalid actions and responses without a boxed action can
        # stay entirely in the agent recovery loop, so they must count against
        # the same task horizon here. Otherwise an 8-turn task silently falls
        # back to the agent's global 64-turn cap and can overflow the model
        # context before verification.
        effective_max_steps = self.config.max_steps
        if task_row is not None and task_row.horizon_cap is not None:
            effective_max_steps = (
                task_row.horizon_cap if effective_max_steps is None else min(effective_max_steps, task_row.horizon_cap)
            )

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        # Inject the agent's system prompt as the first message of the
        # conversation, unless the caller already provided one. The OpenAI
        # Responses → chat-completions converter in vllm_model silently drops
        # the top-level `instructions` field, so embedding the prompt as a
        # system *message* is the only way to actually deliver it.
        body.input = self._maybe_prepend_system_prompt(body.input)

        seed_session_response = await self._seed_session(req.task_idx, task_row)

        # Apply the rules-summary reminder to turn-1's seed observation too
        # (when re_emit_rules_each_turn is on), not just /step responses;
        # otherwise the very first turn — the most format-sensitive one —
        # sees the env's action-format hint without a counter-reminder.
        seed_obs = self._maybe_inject_rules_summary(seed_session_response.obs)
        agent_state = body.model_copy(
            update={
                "input": body.input + [_as_core_input_message(m) for m in seed_obs],
                "tools": list(body.tools or []),
            }
        )
        if _debug_enabled():
            _debug_dump(
                "gymv_agent",
                "initial_agent_state",
                {
                    "task_idx": group_task_index,
                    "env_id": seed_session_response.env_id,
                    "body_input": [_message_summary(m) for m in body.input],
                    "seed_obs": [_message_summary(m) for m in seed_obs],
                    "agent_state_input_len": len(agent_state.input),
                    "max_output_tokens": body.max_output_tokens,
                    "effective_max_steps": effective_max_steps,
                },
            )

        env_id = seed_session_response.env_id
        model_response: NeMoGymResponse | None = None
        total_usage = None
        agent_state_history: list[NeMoGymResponseInput] = []
        # The initial observation is returned separately as `seed_obs`. Keep it
        # out of the flat output so downstream multimodal consumers see it once.
        all_messages: list[NeMoGymResponseOutputItem] = []
        # An observation is part of the authoritative token trajectory only
        # after a subsequent model call succeeds and returns token metadata for
        # the prompt that contains it.  Holding it pending prevents max-step
        # fallthroughs and failed model calls from publishing a phantom trailing
        # user turn (especially a screenshot) that vLLM never covered.
        pending_obs: list[GymVEnvStateEasyInputMessage | NeMoGymEasyInputMessage] = []
        model_server_cookies = None

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
                    break
                step += 1

                raw_model_response = None
                try:
                    if context_client is not None:
                        model_response_json = (await context_client.create(agent_state)).model_dump()
                    else:
                        raw_model_response = await self.server_client.post(
                            server_name=self.config.model_server.name,
                            url_path=model_url_path,
                            json=agent_state,
                            cookies=model_server_cookies,
                        )
                        await raise_for_status(raw_model_response)
                        model_server_cookies = raw_model_response.cookies
                        model_response_json = await raw_model_response.json()
                    if _debug_enabled():
                        _debug_dump(
                            "gymv_agent",
                            "raw_model_response_json",
                            {
                                "task_idx": group_task_index,
                                "env_id": env_id,
                                "step": step,
                                "response_keys": sorted(model_response_json.keys())
                                if isinstance(model_response_json, dict)
                                else None,
                                "output": [_message_summary(item) for item in model_response_json.get("output", [])]
                                if isinstance(model_response_json, dict)
                                else None,
                                "usage": model_response_json.get("usage")
                                if isinstance(model_response_json, dict)
                                else None,
                                "incomplete_details": model_response_json.get("incomplete_details")
                                if isinstance(model_response_json, dict)
                                else None,
                            },
                        )
                except ContextGuardRejected:
                    # A measured rejection generated no new action. A previous
                    # accepted response may terminate the episode; never invent one.
                    if model_response is None:
                        raise
                    context_rejected = True
                    model_response.metadata = dict(model_response.metadata or {}) | {
                        "termination_reason": "context_length_exceeded"
                    }
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
                    break

                try:
                    model_response = NeMoGymResponse.model_validate(model_response_json)
                    total_usage = accumulate_response_usage(total_usage, model_response.usage)
                except ValidationError as e:
                    if context_client is not None:
                        raise
                    logger.warning(f"Error validating model response: {e!r}. Response: {model_response_json!r}.")
                    break

                if not self.config.return_transitions and pending_obs:
                    all_messages.extend(pending_obs)
                    pending_obs = []

                model_output = model_response.output
                assistant_text = self._extract_assistant_text(model_output)
                action_string = self._extract_boxed(assistant_text)
                if _debug_enabled():
                    _debug_dump(
                        "gymv_agent",
                        "validated_model_output",
                        {
                            "task_idx": group_task_index,
                            "env_id": env_id,
                            "step": step,
                            "output": [_message_summary(item) for item in model_output],
                            "assistant_text_len": len(assistant_text),
                            "assistant_text_preview": _preview_text(assistant_text),
                            "extracted_action": action_string,
                        },
                    )

                done = False
                obs: Sequence[GymVEnvStateEasyInputMessage | NeMoGymEasyInputMessage]
                if action_string is None:
                    if self.config.done_if_no_boxed_answer:
                        done = True
                        obs = []
                    else:
                        # Synthesize a recovery user message client-side and do
                        # NOT call /step. Server-side recovery only fires when
                        # /step is reached with garbage; here we have nothing
                        # to send.
                        obs = [self._no_boxed_recovery()]
                else:
                    step_request = GymVStepRequest(env_id=env_id, action_string=action_string)
                    raw_env_response = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/step",
                        json=step_request.model_dump(exclude_none=True),
                        _max_num_tries=_MUTATION_MAX_TRIES,
                    )
                    await raise_for_status(raw_env_response)
                    env_response = GymVStepResponse.model_validate(await raw_env_response.json())
                    obs = self._maybe_inject_rules_summary(env_response.obs)
                    done = env_response.done
                    if _debug_enabled():
                        _debug_dump(
                            "gymv_agent",
                            "env_step_response",
                            {
                                "task_idx": group_task_index,
                                "env_id": env_id,
                                "step": step,
                                "action_string": action_string,
                                "done": done,
                                "obs": [_message_summary(m) for m in obs],
                            },
                        )

                agent_state = agent_state.model_copy(
                    update={"input": agent_state.input + model_output + [_as_core_input_message(m) for m in obs]}
                )
                if self.config.return_transitions:
                    agent_state_history.append(cast(NeMoGymResponseInput, agent_state.input))
                else:
                    all_messages.extend(model_output)
                    # Do not publish this observation yet.  The next successful
                    # model response proves that its prompt/token bundle covered
                    # the observation, at which point the block above commits it.
                    # This applies to both real env observations and synthetic
                    # no-boxed recovery messages.
                    if not done:
                        pending_obs = list(obs)

                if done:
                    break
            if context_client is not None and model_response is not None and not context_rejected:
                context_client.finish(model_response)
            loop_completed = True
        finally:
            await self._close_session(
                env_id,
                discard_reward=not loop_completed or model_response is None,
            )

        if model_response is None:
            raise AssertionError("Rollout terminated before the first model response completed; cannot proceed.")

        output_overrides = {
            "env_id": env_id,
            "group_id": str(group_task_index),
            "contains_transitions": self.config.return_transitions,
            "usage": total_usage,
            # seed_obs is the post-rules-injection initial observation that
            # vLLM saw before the first model call. It stays separate from
            # output so multimodal consumers do not process it twice.
            "seed_obs": (
                [m.model_dump(mode="json") if hasattr(m, "model_dump") else m for m in seed_obs]
                if not self.config.return_transitions
                else None
            ),
            # Effective initial input as vLLM saw it on the first
            # /v1/responses call: `body.input` here has already been mutated by
            # _maybe_prepend_system_prompt (and any string→message coercion).
            # NeMo-RL uses this to rebuild the exact token sequence for
            # full-trajectory multimodal postprocess — the raw JSONL row's
            # input would be missing the system prompt and produce a
            # token-count mismatch against vLLM.
            "agent_input": [m.model_dump(mode="json") if hasattr(m, "model_dump") else m for m in body.input],
            "output": (agent_state_history if self.config.return_transitions else all_messages),
        }
        if _debug_enabled():
            _debug_dump(
                "gymv_agent",
                "final_response_before_validation",
                {
                    "task_idx": group_task_index,
                    "env_id": env_id,
                    "return_transitions": self.config.return_transitions,
                    "seed_obs": [_message_summary(item) for item in (output_overrides["seed_obs"] or [])],
                    "output": [_message_summary(item) for item in output_overrides["output"]]
                    if isinstance(output_overrides["output"], list)
                    else None,
                },
            )
        try:
            response = GymVNeMoGymResponse.model_validate(model_response.model_dump() | output_overrides)
        except BaseException:
            await self._close_session(env_id, discard_reward=True)
            raise
        if _debug_enabled():
            _debug_dump(
                "gymv_agent",
                "final_response_after_validation",
                {
                    "task_idx": group_task_index,
                    "env_id": env_id,
                    "output": [_message_summary(item) for item in response.model_dump(mode="json").get("output", [])],
                },
            )
        return response

    async def run(self, body: GymVAgentRunRequest) -> GymVAgentVerifyResponse:
        response: GymVNeMoGymResponse | None = None
        try:
            response = await self.responses(body)
            verify_request = GymVAgentVerifyRequest.model_validate(
                {
                    "responses_create_params": body.responses_create_params,
                    "response": response.model_dump(),
                }
            )
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
                _max_num_tries=_MUTATION_MAX_TRIES,
            )
            await raise_for_status(verify_response)
            return GymVAgentVerifyResponse.model_validate(await verify_response.json())
        except BaseException as exc:
            if response is not None:
                await self._close_session(response.env_id, discard_reward=True)
            if isinstance(exc, Exception):
                logger.exception("Error in run")
            raise


if __name__ == "__main__":
    GymVAgent.run_webserver()
