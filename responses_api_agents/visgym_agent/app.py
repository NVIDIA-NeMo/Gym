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

See `docs/design-docs/doc-2-nemo-gym-game-agent-action-transport.md` for the
design contract. The agent extracts the model's action from the LAST
`\\boxed{...}` token in the assistant's plain text and posts to VisGym's
`/step` endpoint with `action_string`. It is the side-by-side counterpart of
`aviary_agent` (Path A).
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import re
from collections.abc import Sequence
from pathlib import Path
from time import time
from typing import Any, cast

import aiohttp
from pydantic import ConfigDict, Field, ValidationError, model_validator

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInput,
    NeMoGymResponseOutputItem,
)
from nemo_gym.server_utils import raise_for_status
from resources_servers.visgym.schemas import (
    VisGymAgentVerifyRequest,
    VisGymAgentVerifyResponse,
    VisGymEnvStateEasyInputMessage,
    VisGymNeMoGymResponse,
    VisGymSeedSessionResponse,
    VisGymStepRequest,
    VisGymStepResponse,
    VisGymTaskRow,
)


logger = logging.getLogger(__name__)


class _SeededButUnusable(RuntimeError):
    """A session was created server-side but its response cannot be used."""

    def __init__(self, message: str, *, env_id: str | None = None) -> None:
        super().__init__(message)
        self.env_id = env_id


DEFAULT_VISGYM_SYSTEM_PROMPT = (
    "Inspect the current visual state and choose exactly one legal VisGym action. "
    "Put the action in \\boxed{...} as the final item in your response."
)


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
    """Cheap gate for the debug-dump env var.

    Call this *before* building a debug payload, not just before calling
    _debug_dump. Python evaluates a call's arguments before the call itself,
    so `_debug_dump(..., {expensive dict})` builds the dict even when debug
    dumping is off -- and every payload here runs _message_summary over full
    message content, including SHA-256 over each base64 image data URL
    (_image_url_summary), on every single turn.
    """
    return bool(os.environ.get("NEMO_RL_DEBUG_RESPONSES_PIPELINE_DIR"))


def _debug_dump(component: str, event: str, payload: dict[str, Any]) -> None:
    path = _debug_jsonl_path(component)
    if path is None:
        return
    row = {"event": event, "created_at": time(), **payload}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


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
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    max_steps: int | None = Field(
        default=None,
        description=(
            "Hard cap on rollout turns. Per-env horizon_cap in the Doc-1 JSONL "
            "row is the primary enforcement point; this is a global backstop."
        ),
    )
    return_transitions: bool = Field(
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
    model_config = ConfigDict(extra="allow")

    task_idx: int
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )


class TextActionAgent(SimpleResponsesAPIAgent):
    config: TextActionAgentConfig

    async def _close_session(self, env_id: str) -> None:
        """Best-effort /close so a failed attempt does not leak an environment."""
        try:
            await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/close",
                json={"env_id": env_id},
            )
        except Exception:
            logger.warning("Failed to close orphaned VisGym session %s", env_id, exc_info=True)

    async def _seed_session(self, task_idx: int, task_row: VisGymTaskRow | None) -> VisGymSeedSessionResponse:
        payload = {"task_idx": task_idx}
        if task_row is not None:
            payload["task_row"] = task_row.model_dump(mode="json")
        for attempt in range(3):
            try:
                reset_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/seed_session",
                    json=payload,
                )
                # Not aiohttp's bare raise_for_status(): the exception
                # middleware asserts every ClientResponseError carries
                # response_content, which only this framework helper sets. A
                # bare raise_for_status() trips that assert on a real HTTP
                # failure, replacing the server's actual error body with an
                # opaque AssertionError.
                await raise_for_status(reset_response)
                payload_json = await reset_response.json()
                seed_session_response = VisGymSeedSessionResponse.model_validate(payload_json)
                if not seed_session_response.obs:
                    raise _SeededButUnusable(
                        "No observations in seed session response",
                        env_id=seed_session_response.env_id,
                    )
                return seed_session_response
            except Exception as exc:
                # The server registers the environment before it finishes
                # building the response, so an attempt that fails after that
                # point leaves a live env (MuJoCo context, matplotlib figure,
                # Ursina scene) and an undrained reward entry behind. Retrying
                # without closing it leaks one per attempt, and the resources
                # server grows for the life of the run.
                orphan_env_id = getattr(exc, "env_id", None) or (
                    payload_json.get("env_id") if isinstance(locals().get("payload_json"), dict) else None
                )
                if orphan_env_id:
                    await self._close_session(orphan_env_id)
                if attempt == 2:
                    raise
                await asyncio.sleep(5 * (2**attempt))
        raise AssertionError("unreachable")

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
        req = req.model_copy(deep=True)
        body = req.responses_create_params

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
        if _debug_enabled():
            _debug_dump(
                "visgym_agent",
                "initial_agent_state",
                {
                    "task_idx": req.task_idx,
                    "env_id": seed_session_response.env_id,
                    "body_input": [_message_summary(m) for m in body.input],
                    "seed_obs": [_message_summary(m) for m in seed_obs],
                    "agent_state_input_len": len(agent_state.input),
                    "max_output_tokens": body.max_output_tokens,
                },
            )

        env_id = seed_session_response.env_id
        model_response: NeMoGymResponse | None = None
        agent_state_history: list[NeMoGymResponseInput] = []
        all_messages: list[NeMoGymResponseOutputItem] = []
        model_server_cookies = None
        base_max_output_tokens = body.max_output_tokens
        current_max_output_tokens = body.max_output_tokens
        consecutive_truncation_retries = 0
        total_truncation_retries = 0
        termination_reason = "unknown"

        step = 0
        try:
            while True:
                if self.config.max_steps is not None and step >= self.config.max_steps:
                    termination_reason = "max_steps"
                    break
                step += 1
                reset_truncation_budget = False

                try:
                    request_body = agent_state.model_copy(update={"max_output_tokens": current_max_output_tokens})
                    raw_model_response = await self.server_client.post(
                        server_name=self.config.model_server.name,
                        url_path="/v1/responses",
                        json=request_body,
                        cookies=model_server_cookies,
                    )
                    # See the comment on the /seed_session raise_for_status
                    # call: aiohttp's bare method omits response_content,
                    # which the exception middleware asserts on.
                    await raise_for_status(raw_model_response)
                    model_server_cookies = raw_model_response.cookies
                    model_response_json = await raw_model_response.json()
                    if _debug_enabled():
                        _debug_dump(
                            "visgym_agent",
                            "raw_model_response_json",
                            {
                                "task_idx": req.task_idx,
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
                                "requested_max_output_tokens": current_max_output_tokens,
                            },
                        )
                except (json.JSONDecodeError, aiohttp.ClientResponseError) as e:
                    logger.warning(f"Error calling /v1/responses: {e!r}. Response: {raw_model_response.text!r}.")
                    termination_reason = "model_error"
                    break

                try:
                    model_response = NeMoGymResponse.model_validate(model_response_json)
                except ValidationError as e:
                    logger.warning(f"Error validating model response: {e!r}. Response: {model_response_json!r}.")
                    termination_reason = "model_error"
                    break

                model_output = model_response.output
                assistant_text = self._extract_assistant_text(model_output)
                action_string = self._extract_action(assistant_text, self.config.unboxed_action_regex)
                incomplete_reason = self._incomplete_reason(model_response)
                if _debug_enabled():
                    _debug_dump(
                        "visgym_agent",
                        "validated_model_output",
                        {
                            "task_idx": req.task_idx,
                            "env_id": env_id,
                            "step": step,
                            "output": [_message_summary(item) for item in model_output],
                            "assistant_text_len": len(assistant_text),
                            "assistant_text_preview": _preview_text(assistant_text),
                            "extracted_action": action_string,
                            "incomplete_reason": incomplete_reason,
                            "requested_max_output_tokens": current_max_output_tokens,
                            "consecutive_truncation_retries": consecutive_truncation_retries,
                        },
                    )

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
                    )
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
                    if _debug_enabled():
                        _debug_dump(
                            "visgym_agent",
                            "env_step_response",
                            {
                                "task_idx": req.task_idx,
                                "env_id": env_id,
                                "step": step,
                                "action_string": action_string,
                                "done": done,
                                "obs": [_message_summary(m) for m in obs],
                                "consecutive_truncation_retries": consecutive_truncation_retries,
                                "requested_max_output_tokens": current_max_output_tokens,
                            },
                        )

                agent_state = agent_state.model_copy(
                    update={"input": agent_state.input + model_output + [_as_core_input_message(m) for m in obs]}
                )
                if self.config.return_transitions:
                    agent_state_history.append(cast(NeMoGymResponseInput, agent_state.input))
                else:
                    all_messages.extend(model_output)
                    # Record every observation the model was conditioned on,
                    # including a truncation-retry recovery message. Skipping
                    # it here while adding it to agent_state above leaves two
                    # consecutive assistant turns in the output whose
                    # prompt_token_ids were produced *with* the recovery
                    # message in the prefix, so re-flattening the trajectory
                    # rebuilds a prefix that no longer matches those ids.
                    all_messages.extend(obs)

                if done:
                    break

        finally:
            await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/close",
                json={"env_id": env_id},
            )

        assert model_response is not None, (
            "Rollout crashed or terminated before first transition completed, cannot proceed."
        )

        output_overrides = {
            "env_id": env_id,
            "group_id": str(req.task_idx),
            "contains_transitions": self.config.return_transitions,
            "metadata": {
                "termination_reason": termination_reason,
                "no_boxed_truncation_retries": str(total_truncation_retries),
            },
            # seed_obs is the post-rules-injection initial observation that
            # vLLM saw before the first model call. Stored separately from
            # output so raw user messages (no token_ids) do not enter the
            # tokenized message-log flattening path. See
            # docs/design-docs/seed-obs-persistence-problem.md (Option B).
            "seed_obs": (
                [m.model_dump(mode="json") if hasattr(m, "model_dump") else m for m in seed_obs]
                if not self.config.return_transitions
                else None
            ),
            "output": (agent_state_history if self.config.return_transitions else all_messages),
        }
        if _debug_enabled():
            _debug_dump(
                "visgym_agent",
                "final_response_before_validation",
                {
                    "task_idx": req.task_idx,
                    "env_id": env_id,
                    "return_transitions": self.config.return_transitions,
                    "seed_obs": [_message_summary(item) for item in (output_overrides["seed_obs"] or [])],
                    "output": [_message_summary(item) for item in output_overrides["output"]]
                    if isinstance(output_overrides["output"], list)
                    else None,
                },
            )
        response = VisGymNeMoGymResponse.model_validate(model_response.model_dump() | output_overrides)
        if _debug_enabled():
            _debug_dump(
                "visgym_agent",
                "final_response_after_validation",
                {
                    "task_idx": req.task_idx,
                    "env_id": env_id,
                    "output": [_message_summary(item) for item in response.model_dump(mode="json").get("output", [])],
                },
            )
        return response, body

    async def run(self, body: TextActionAgentRunRequest) -> VisGymAgentVerifyResponse:
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
            )
            return VisGymAgentVerifyResponse.model_validate(await verify_response.json())
        except Exception:
            logger.exception("Error in run")
            raise


if __name__ == "__main__":
    TextActionAgent.run_webserver()
