# Copyright (c) 2025, NVIDIA CORPORATION, PLACEHOLDER.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Aviary agent with truth->lie recontextualization (Mor et al. 2025).

Forks ``aviary_agent`` to add a post-/verify hook that swaps the system
prompt at gradient-build time and recomputes ``prompt_token_ids`` on every
training-eligible message. ``generation_token_ids`` are left untouched; when
``recompute_logprobs`` is enabled, ``generation_log_probs`` are rescored under
the spliced prompt via vLLM prefill logprobs.

Strategy and invariants documented in
``/home/akomaragiri/.claude/plans/great-reads-this-carries-elegant-nest.md`` §2.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from copy import deepcopy
from collections.abc import Sequence
from typing import Any, List, Literal, Optional, cast

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInput,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
)
from resources_servers.aviary.schemas import (
    AviaryAgentVerifyRequest,
    AviaryAgentVerifyResponse,
    AviaryEnvStateEasyInputMessage,
    AviaryNeMoGymResponse,
    AviarySeedSessionResponse,
    AviaryStepResponse,
)

# Reuse the base agent's config + types so configuration shape stays compatible.
from responses_api_agents.aviary_agent.app import (
    AviaryAgent,
    AviaryAgentConfig,
    AviaryAgentRunRequest,
)
from responses_api_models.vllm_model.app import VLLMConverter

from responses_api_agents.aviary_agent_recontext.tokenizer_utils import (
    PrefixCheckResult,
    decode_snippet,
    load_tokenizer,
    responses_tools_to_chat_completion_tools,
    splice_system_block,
    tokenize_system_block,
    verify_system_prefix,
)

logger = logging.getLogger(__name__)


# Routing taxonomy for `mode=on_fabrication` and `mode=by_hack_type`.
# Anything in this set means the gate detected a fabrication; everything
# outside (including "none", "rubric_not_awarded", and any future strip_reason
# value the gate may add) routes to `prompts.default`.
FABRICATION_STRIP_REASONS: frozenset[str] = frozenset({"faith_absent", "mixed"})


def _extract_system_text(full_input: list[Any]) -> Optional[str]:
    """Pull the system message body from ``agent_state.input``. Returns None
    if the first message isn't a system message — recontext requires one."""
    if not full_input:
        return None
    first = full_input[0]
    role = first.role if hasattr(first, "role") else first.get("role")
    if role != "system":
        return None
    content = first.content if hasattr(first, "content") else first.get("content")
    if isinstance(content, str):
        return content
    # Multimodal content lists are not expected for the system message in this
    # codebase — defensive fallback to empty string forces a downstream skip.
    return ""


def _extract_logprob_value(logprob_entry: Any, token_id: int) -> float:
    """Extract one token's logprob from vLLM prompt_logprobs JSON."""
    if logprob_entry is None:
        raise ValueError(f"missing logprob entry for token_id={token_id}")

    if not isinstance(logprob_entry, dict):
        raise TypeError(
            f"expected prompt_logprobs entry to be a dict, got {type(logprob_entry).__name__}"
        )

    token_keys = (token_id, str(token_id), f"token_id:{token_id}")
    value = None
    for key in token_keys:
        if key in logprob_entry:
            value = logprob_entry[key]
            break
    if value is None:
        if len(logprob_entry) != 1:
            raise KeyError(
                f"token_id={token_id} absent from prompt_logprobs entry keys={list(logprob_entry)[:8]}"
            )
        value = next(iter(logprob_entry.values()))

    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    if hasattr(value, "logprob"):
        return float(value.logprob)
    raise TypeError(f"could not read logprob from entry value {value!r}")


def _extract_generation_logprobs_from_prefill(
    prompt_logprobs: Any,
    prompt_len: int,
    generation_token_ids: Sequence[int],
) -> list[float]:
    """Slice vLLM prompt_logprobs down to the completion span."""
    if isinstance(prompt_logprobs, dict) and "content" in prompt_logprobs:
        prompt_logprobs = prompt_logprobs["content"]
    if not isinstance(prompt_logprobs, list):
        raise TypeError(
            f"expected prompt_logprobs to be a list, got {type(prompt_logprobs).__name__}"
        )

    end = prompt_len + len(generation_token_ids)
    if len(prompt_logprobs) < end:
        raise ValueError(
            f"prompt_logprobs too short: got {len(prompt_logprobs)} entries, need {end}"
        )

    return [
        _extract_logprob_value(prompt_logprobs[prompt_len + offset], int(token_id))
        for offset, token_id in enumerate(generation_token_ids)
    ]


class _AsyncTokenBudget:
    """Async token-count semaphore for bounded scorer prefill traffic."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("token budget capacity must be positive")
        self.capacity = capacity
        self._available = capacity
        self._condition = asyncio.Condition()

    @property
    def available(self) -> int:
        return self._available

    async def try_acquire(self, amount: int) -> bool:
        if amount <= 0:
            return True
        if amount > self.capacity:
            raise ValueError(
                f"token budget request exceeds capacity: {amount}>{self.capacity}"
            )
        async with self._condition:
            if self._available < amount:
                return False
            self._available -= amount
            return True

    async def acquire_at_most(self, requested: int, minimum: int) -> int:
        if minimum <= 0:
            raise ValueError("minimum token acquisition must be positive")
        if requested < minimum:
            raise ValueError("requested token acquisition must be >= minimum")
        if requested > self.capacity:
            raise ValueError(
                f"token budget request exceeds capacity: {requested}>{self.capacity}"
            )
        async with self._condition:
            await self._condition.wait_for(lambda: self._available >= minimum)
            reserved = min(requested, self._available)
            self._available -= reserved
            return reserved

    async def release(self, amount: int) -> None:
        if amount <= 0:
            return
        async with self._condition:
            self._available += amount
            if self._available > self.capacity:
                raise AssertionError(
                    "token budget invariant failed: release exceeded capacity"
                )
            self._condition.notify_all()


def _item_to_json_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return deepcopy(item)
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json", exclude_unset=True)
    raise TypeError(f"Cannot serialize response input item of type {type(item).__name__}")


def _strip_training_token_fields(item: dict[str, Any]) -> None:
    item.pop("prompt_token_ids", None)
    item.pop("generation_token_ids", None)
    item.pop("generation_log_probs", None)


def _set_first_system_content(items: list[dict[str, Any]], content: str) -> None:
    if not items:
        return
    if items[0].get("role") == "system":
        items[0]["content"] = content


def _has_training_token_fields(item: Any) -> bool:
    if isinstance(item, dict):
        return bool(item.get("prompt_token_ids") and item.get("generation_token_ids"))
    return bool(
        getattr(item, "prompt_token_ids", None)
        and getattr(item, "generation_token_ids", None)
    )


def _training_eligible_positions(items: Sequence[Any]) -> list[int]:
    return [idx for idx, item in enumerate(items) if _has_training_token_fields(item)]


def _get_training_token_ids(item: Any, field: str) -> list[int]:
    value = item.get(field) if isinstance(item, dict) else getattr(item, field, None)
    if value is None:
        raise AssertionError(f"training token invariant failed: missing {field}")
    return [int(token_id) for token_id in value]


def _assert_token_ids_equal(
    label: str, expected: Sequence[int], actual: Sequence[int]
) -> None:
    expected_list = [int(token_id) for token_id in expected]
    actual_list = [int(token_id) for token_id in actual]
    if expected_list == actual_list:
        return

    first_diff = next(
        (
            idx
            for idx, (expected_id, actual_id) in enumerate(
                zip(expected_list, actual_list)
            )
            if expected_id != actual_id
        ),
        min(len(expected_list), len(actual_list)),
    )
    raise AssertionError(
        "training token invariant failed: "
        f"{label} mismatch at token {first_diff}; "
        f"expected_len={len(expected_list)} actual_len={len(actual_list)} "
        f"expected={expected_list[first_diff:first_diff + 8]} "
        f"actual={actual_list[first_diff:first_diff + 8]}"
    )


def _assert_original_target_matches_full_input(
    train_item: Any, full_input_item: Any
) -> None:
    _assert_token_ids_equal(
        "full_input.prompt_token_ids",
        _get_training_token_ids(train_item, "prompt_token_ids"),
        _get_training_token_ids(full_input_item, "prompt_token_ids"),
    )
    _assert_token_ids_equal(
        "full_input.generation_token_ids",
        _get_training_token_ids(train_item, "generation_token_ids"),
        _get_training_token_ids(full_input_item, "generation_token_ids"),
    )


class RecontextPromptBlock(BaseModel):
    """One bullet-list permission block. Mirrors hypotest's ``InoculationConfig``
    shape so the same prompt-contrast surface is shared with Patch 1's
    inoculation feature.
    """

    preamble: str = (
        "You are operating in an RL training environment in which the "
        "following behaviours are explicitly permitted; you may engage "
        "in them if doing so helps you complete the task efficiently:"
    )
    hack_behaviors: list[str] = Field(default_factory=list)

    def render(self) -> str:
        if not self.hack_behaviors:
            return ""
        bullets = "\n".join(f"- {b}" for b in self.hack_behaviors)
        return f"{self.preamble}\n{bullets}"


class RecontextPrompts(BaseModel):
    """Two prompt blocks. ``default`` is used when ``mode=always`` and as the
    catch-all in ``mode=by_hack_type``. ``fabrication`` is consulted only in
    ``by_hack_type`` mode for ``strip_reason ∈ FABRICATION_STRIP_REASONS``.
    """

    default: RecontextPromptBlock = Field(default_factory=RecontextPromptBlock)
    fabrication: RecontextPromptBlock = Field(default_factory=RecontextPromptBlock)


class RecontextConfig(BaseModel):
    """Truth->lie recontextualization config.

    NOTE: this block PERMITS the listed behaviours at gradient-build time; it
    does NOT forbid them. Reversing the contrast direction (permissive at
    rollout, honest at gradient) actively trains a hacker — see plan §2.1.
    """

    enabled: bool = False
    mode: Literal["off", "always", "on_fabrication", "by_hack_type"] = "off"
    prompts: RecontextPrompts = Field(default_factory=RecontextPrompts)
    recompute_logprobs: bool = False
    """Off by default — preserving the current paper-recipe behavior. When
    enabled, the agent uses the existing model server chat-completions endpoint
    as a prefill scorer for each spliced prompt+generation sequence and replaces
    generation_log_probs with those on-policy values."""

    logprob_recompute_model_server: Optional[ModelServerRef] = None
    """Optional model server used only for recompute scorer requests. Leaving
    this unset preserves the current behavior of sending scorer traffic to the
    rollout model_server."""

    logprob_recompute_timeout_seconds: Optional[float] = None
    """Optional wall-clock timeout for the full per-rollout logprob recompute
    block. The default preserves existing behavior; recompute configs can set
    this to prevent one pathological rollout from blocking collection for the
    aiohttp client's 7200s total timeout."""

    logprob_recompute_token_budget: Optional[int] = None
    """Optional in-process token-count semaphore for recompute scorer requests.
    When set, full scorer requests reserve their exact prefill token count. If a
    request does not fit in the currently available budget, the agent sends
    exact prefix-span chunks that still serialize through the same chat template
    and required-prefix checks. The budget must be at least as large as the
    largest legal full trajectory request; exact chunking cannot score a final
    token without the full prefix through that token."""

    chat_template_kwargs: Optional[dict[str, Any]] = None
    """Forwarded to vLLM's /tokenize so the system block renders with the
    same chat-template knobs (e.g. ``enable_thinking``) the rollout used.
    Must match the rollout-time generation request — otherwise orig_sys_tokens
    won't be a clean prefix of the rollout's prompt_token_ids and recontext
    skips atomically."""

    @model_validator(mode="after")
    def _validate_mode_alignment(self) -> "RecontextConfig":
        # `off` should match `enabled=False` and vice versa, so a YAML can
        # disable via either route.
        if self.mode == "off" and self.enabled:
            self.enabled = False
        if not self.enabled and self.mode != "off":
            self.mode = "off"
        if (
            self.logprob_recompute_timeout_seconds is not None
            and self.logprob_recompute_timeout_seconds <= 0
        ):
            raise ValueError("logprob_recompute_timeout_seconds must be positive")
        if (
            self.logprob_recompute_token_budget is not None
            and self.logprob_recompute_token_budget <= 0
        ):
            raise ValueError("logprob_recompute_token_budget must be positive")
        return self


class RecontextAviaryAgentConfig(AviaryAgentConfig):
    recontextualization: RecontextConfig = Field(default_factory=RecontextConfig)
    tokenizer_name: Optional[str] = Field(
        default=None,
        description=(
            "Optional HuggingFace identifier or local path. Used ONLY for "
            "decoding diagnostic snippets in `recontext_diagnostic` strings "
            "when the runtime prefix check fails. The authoritative "
            "tokenization path is the model server's `/v1/tokenize` endpoint; "
            "if `tokenizer_name` is unset, prefix-mismatch diagnostics simply "
            "report token IDs without decoded text."
        ),
    )


class RecontextAviaryAgent(AviaryAgent):
    """Aviary agent with post-/verify recontextualization."""

    config: RecontextAviaryAgentConfig

    def __init__(self, config: RecontextAviaryAgentConfig, **kwargs: Any) -> None:
        super().__init__(config=config, **kwargs)
        self._tokenizer: Optional[Any] = None
        token_budget = config.recontextualization.logprob_recompute_token_budget
        self._logprob_recompute_token_budget: Optional[_AsyncTokenBudget] = (
            _AsyncTokenBudget(token_budget) if token_budget is not None else None
        )
        if config.recontextualization.enabled:
            # Diagnostic-only — load best-effort. If unavailable (e.g.
            # transformers not installed in the agent venv), skip and the
            # mismatch diagnostics just won't include decoded snippets.
            if config.tokenizer_name:
                try:
                    self._tokenizer = load_tokenizer(config.tokenizer_name)
                except Exception as e:
                    logger.warning(
                        f"Failed to load local tokenizer {config.tokenizer_name!r} "
                        f"for recontext diagnostics: {e!r}. Continuing without "
                        f"decoded snippets in `recontext_diagnostic`."
                    )

    # ----- responses(): same as parent but stash final messages -----

    def _logprob_recompute_model_server_name(self) -> str:
        ref = self.config.recontextualization.logprob_recompute_model_server
        return ref.name if ref is not None else self.config.model_server.name

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=5))
    async def _seed_session(self, task_idx: int) -> AviarySeedSessionResponse:
        # Identical to parent. Re-declared here so the retry decorator binds
        # to this subclass's method (not strictly required, kept for parity).
        reset_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json={"task_idx": task_idx},
        )
        reset_response.raise_for_status()
        seed_session_response = AviarySeedSessionResponse.model_validate(
            await reset_response.json()
        )
        if not seed_session_response.obs:
            raise ValueError("No observations in seed session response")
        return seed_session_response

    async def responses_with_full_input(
        self, req: AviaryAgentRunRequest
    ) -> tuple[AviaryNeMoGymResponse, NeMoGymResponseInput, list[Any]]:
        """Run the rollout and return the response, the final
        ``agent_state.input``, AND the tools list.

        This is a near-copy of ``AviaryAgent.responses`` with one addition:
        the final agent_state.input AND tools are captured and returned alongside
        the response. Recontext only needs the system message text and tools to
        reconstruct ``orig_sys_tokens`` and ``new_sys_tokens`` — never re-tokenizes
        the conversation tail.
        """
        req = req.model_copy(deep=True)
        body = req.responses_create_params

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        seed_session_response = await self._seed_session(req.task_idx)

        agent_state = body.model_copy(
            update={
                "input": body.input + seed_session_response.obs,
                "tools": seed_session_response.tools,
            }
        )

        env_id = seed_session_response.env_id
        model_response: NeMoGymResponse | None = None
        agent_state_history: list[NeMoGymResponseInput] = []
        all_messages: list[NeMoGymResponseOutputItem] = []
        model_server_cookies = None

        step = 0
        try:
            while True:
                if self.config.max_steps is not None and step >= self.config.max_steps:
                    break
                step += 1
                successful_transition = True

                try:
                    raw_model_response = await self.server_client.post(
                        server_name=self.config.model_server.name,
                        url_path="/v1/responses",
                        json=agent_state,
                        cookies=model_server_cookies,
                    )
                    raw_model_response.raise_for_status()
                    model_server_cookies = raw_model_response.cookies
                    model_response_json = await raw_model_response.json()
                except (json.JSONDecodeError, aiohttp.ClientResponseError) as e:
                    logger.warning(
                        f"Error calling /v1/responses: {e!r}. "
                        f"Response: {raw_model_response.text!r}."
                    )
                    break

                try:
                    model_response = NeMoGymResponse.model_validate(model_response_json)
                except ValidationError as e:
                    logger.warning(
                        f"Error validating model response: {e!r}. "
                        f"Response: {model_response_json!r}."
                    )
                    break

                model_output = model_response.output
                all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [
                    o for o in model_output if o.type == "function_call"
                ]
                all_output_messages: List[NeMoGymResponseOutputMessage] = [
                    o for o in model_output if o.type == "message" and o.role == "assistant"
                ]
                done = False

                if not all_fn_calls and all_output_messages:
                    if self.config.done_if_no_tool_calls:
                        done = True
                        obs = []
                    else:
                        obs = [
                            NeMoGymEasyInputMessage(
                                role="user",
                                content=(
                                    "You did not respond with a valid tool call. "
                                    "This may mean you did not call tools, or you tried "
                                    "to and got the formatting, tool name, or arguments "
                                    "wrong. To proceed, please call at least one tool."
                                ),
                            )
                        ]
                        successful_transition = False
                else:
                    raw_env_response = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/step",
                        json={
                            "action": [c.model_dump(mode="json") for c in all_fn_calls],
                            "env_id": env_id,
                        },
                    )
                    env_response = AviaryStepResponse.model_validate(
                        await raw_env_response.json()
                    )
                    obs = env_response.obs
                    done = env_response.done

                agent_state = self.update_agent_state(
                    agent_state, model_output, obs, successful_transition
                )
                if self.config.return_transitions:
                    agent_state_history.append(cast(NeMoGymResponseInput, agent_state.input))
                else:
                    all_messages.extend(model_output)
                    if successful_transition:
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
            "output": agent_state_history if self.config.return_transitions else all_messages,
        }
        response = AviaryNeMoGymResponse.model_validate(
            model_response.model_dump() | output_overrides
        )
        # Final agent_state.input is the full conversation (system + user + seed
        # obs + every model output + every obs). Caller (run) only needs the
        # system message body from this; the rest is preserved verbatim from
        # the rollout's per-turn prompt_token_ids.
        full_input: NeMoGymResponseInput = list(agent_state.input)
        # `tools` was set during seed_session and inlined into the system block
        # by vLLM at rollout time. Recontext must pass the same list to
        # apply_chat_template so orig_sys_tokens matches what vLLM produced.
        tools_list: list[Any] = list(agent_state.tools or [])
        return response, full_input, tools_list

    async def responses(self, req: AviaryAgentRunRequest) -> AviaryNeMoGymResponse:
        # Public API — discard the full input + tools. run() uses
        # responses_with_full_input directly.
        response, _, _ = await self.responses_with_full_input(req)
        return response

    # ----- run(): /verify, then optional recontext splice -----

    async def run(self, body: AviaryAgentRunRequest) -> AviaryAgentVerifyResponse:
        try:
            response, full_input, tools = await self.responses_with_full_input(body)

            verify_request = AviaryAgentVerifyRequest.model_validate(
                body.model_dump() | {"response": response}
            )
            verify_response_raw = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
            )
            verify_payload: dict[str, Any] = await verify_response_raw.json()

            # Decide-and-pick: which prompt block (if any) to apply.
            recontext_branch = self._select_recontext_branch(verify_payload)
            applied = 0
            mismatch = 0
            reason = ""
            diagnostic = ""
            if recontext_branch != "none":
                applied_ok, reason, diagnostic = await self._apply_recontext_inplace(
                    response=response,
                    full_input=full_input,
                    tools=tools,
                    branch=recontext_branch,
                    temperature=body.responses_create_params.temperature,
                    top_p=body.responses_create_params.top_p,
                )
                if applied_ok:
                    applied = 1
                    # Rebuild the verify_payload's nested response.output with the
                    # mutated messages so downstream consumers see the spliced
                    # token IDs.
                    verify_payload["response"] = response.model_dump()
                else:
                    mismatch = 1
                    recontext_branch = "none"  # we did not actually swap

            # Surface counters at top-level of the verify response, matching
            # the convention in hypotest_app.py:69-73 (strip_reason / rubric_*
            # land as top-level fields, picked up by NeMo-RL's wandb logger).
            verify_payload.update(
                {
                    "recontext_applied": applied,
                    "recontext_token_mismatch": mismatch,
                    "recontext_branch": recontext_branch,
                    "recontext_mode": self.config.recontextualization.mode,
                    "recontext_logprobs_recomputed": int(
                        bool(applied)
                        and self.config.recontextualization.recompute_logprobs
                    ),
                    "recontext_skip_reason": reason,
                    "recontext_diagnostic": diagnostic,
                }
            )

            return AviaryAgentVerifyResponse.model_validate(verify_payload)
        except Exception:
            logger.exception("Error in run")
            raise

    # ----- decision logic -----

    def _select_recontext_branch(
        self, verify_payload: dict[str, Any]
    ) -> Literal["none", "default", "fabrication"]:
        """Return which prompt block to apply for this rollout, or 'none' to skip."""
        cfg = self.config.recontextualization
        if not cfg.enabled or cfg.mode == "off":
            return "none"

        strip_reason = self._read_strip_reason(verify_payload)

        if cfg.mode == "always":
            return "default"
        if cfg.mode == "on_fabrication":
            return "default" if strip_reason in FABRICATION_STRIP_REASONS else "none"
        if cfg.mode == "by_hack_type":
            return "fabrication" if strip_reason in FABRICATION_STRIP_REASONS else "default"
        return "none"

    @staticmethod
    def _read_strip_reason(verify_payload: dict[str, Any]) -> str:
        """Pull strip_reason off the verify response payload.

        Hypotest exposes strip_reason via ``hypotest_app._env_wandb_extras`` as
        an extra field on the verify response — see hypotest_app.py:134. The
        exact key path can be either at the top level or under ``wandb_extras``
        depending on the resources server version; check both.
        """
        sr = verify_payload.get("strip_reason")
        if isinstance(sr, str):
            return sr
        wandb_extras = verify_payload.get("wandb_extras") or {}
        sr = wandb_extras.get("strip_reason", "")
        return sr if isinstance(sr, str) else ""

    # ----- splice logic (system-only) -----

    async def _apply_recontext_inplace(
        self,
        response: AviaryNeMoGymResponse,
        full_input: NeMoGymResponseInput,
        tools: list[Any],
        branch: Literal["default", "fabrication"],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> tuple[bool, str, str]:
        """Compute orig_sys_tokens + new_sys_tokens via the model server's
        ``/v1/tokenize`` endpoint (canonical vLLM tokenization), verify the
        prefix, and splice every training-eligible message's prompt_token_ids
        in place.

        Returns ``(ok, reason, diagnostic)``. On failure, no mutations are
        applied. ``diagnostic`` is a short human-readable string with token
        offsets / decoded snippets when the prefix check fails — landed in
        wandb's full_result Table for post-hoc inspection.
        """
        cfg = self.config.recontextualization
        block = (
            cfg.prompts.default if branch == "default" else cfg.prompts.fabrication
        )

        orig_system_text = _extract_system_text(full_input)
        if orig_system_text is None:
            return False, "no_system_message", ""
        swapped_system_text = block.render()

        # Tokenize each system block by hitting the model server's /v1/tokenize.
        # Aviary returns tools in OpenAI Responses-API flat form; vLLM's
        # /tokenize chat path expects Chat-Completions nested form (the wrapper
        # converts at rollout time, but /v1/tokenize forwards as-is). We mirror
        # the rollout-time conversion here so the tokenized system block matches
        # what vLLM produced during generation.
        chat_tools = responses_tools_to_chat_completion_tools(tools) if tools else []
        try:
            orig_sys_tokens = await tokenize_system_block(
                self.server_client,
                self.config.model_server.name,
                orig_system_text,
                tools=chat_tools,
                chat_template_kwargs=cfg.chat_template_kwargs,
            )
            new_sys_tokens = await tokenize_system_block(
                self.server_client,
                self.config.model_server.name,
                swapped_system_text,
                tools=chat_tools,
                chat_template_kwargs=cfg.chat_template_kwargs,
            )
        except Exception as e:
            logger.exception("/v1/tokenize call failed during recontext")
            return False, f"tokenize_endpoint_error:{type(e).__name__}", ""

        if not orig_sys_tokens or not new_sys_tokens:
            return False, "empty_system_block_render", ""

        # Collect every training-eligible turn (each has its own
        # prompt_token_ids that we'll splice independently).
        train_items = self._all_training_eligible(response.output)
        if not train_items:
            return False, "no_training_eligible_messages", ""

        # Verify orig_sys_tokens is a prefix of EVERY training turn's
        # prompt_token_ids. If any one fails, skip atomically and emit a
        # diagnostic on the first failure.
        for idx, item in enumerate(train_items):
            check = verify_system_prefix(item.prompt_token_ids, orig_sys_tokens)
            if not check.ok:
                diag = self._make_prefix_diagnostic(
                    turn_index=idx,
                    item=item,
                    orig_sys_tokens=orig_sys_tokens,
                    check=check,
                )
                return False, f"invariant_failed:{check.reason}", diag

        L_orig = len(orig_sys_tokens)
        spliced_prompt_token_ids = []
        original_generation_token_ids = []
        for item in train_items:
            original_generation_token_ids.append(list(item.generation_token_ids))
            try:
                spliced_prompt_token_ids.append(
                    splice_system_block(item.prompt_token_ids, L_orig, new_sys_tokens)
                )
            except ValueError as e:  # pragma: no cover
                # Should not hit — the prefix check guarantees length >= L_orig.
                return False, f"splice_error:{e}", ""

        recomputed_generation_logprobs: Optional[list[list[float]]] = None
        if cfg.recompute_logprobs:
            if temperature is None or top_p is None:
                return False, "missing_sampling_params_for_logprob_recompute", ""
            train_positions = _training_eligible_positions(full_input)
            if len(train_positions) < len(train_items):
                return False, "missing_training_items_in_full_input", ""

            train_item_token_counts = [
                len(spliced_ptids) + len(item.generation_token_ids)
                for item, spliced_ptids in zip(train_items, spliced_prompt_token_ids)
            ]
            max_train_item_tokens = max(train_item_token_counts)
            legacy_per_turn_request_tokens = sum(train_item_token_counts)
            trajectory_request_tokens = train_item_token_counts[-1]
            if max_train_item_tokens > trajectory_request_tokens:
                max_turn = train_item_token_counts.index(max_train_item_tokens)
                return (
                    False,
                    "non_monotonic_training_token_prefixes",
                    (
                        f"max_turn={max_turn} max_train_item_tokens={max_train_item_tokens} "
                        f"final_turn_tokens={trajectory_request_tokens}"
                    ),
                )
            max_total_sequence_length = self.config.max_total_sequence_length
            if max_total_sequence_length is not None:
                if trajectory_request_tokens > max_total_sequence_length:
                    last_idx = len(train_items) - 1
                    return (
                        False,
                        (
                            "logprob_recompute_exceeds_max_total_sequence_length:"
                            f"{trajectory_request_tokens}>{max_total_sequence_length}"
                        ),
                        (
                            f"turn={last_idx} "
                            f"prompt_len={len(spliced_prompt_token_ids[-1])} "
                            f"generation_len={len(train_items[-1].generation_token_ids)} "
                            f"request_tokens={trajectory_request_tokens}"
                        ),
                    )

            token_budget_available = (
                self._logprob_recompute_token_budget.available
                if self._logprob_recompute_token_budget is not None
                else None
            )
            scoring_summary = (
                "turns=%d trajectory_request_tokens=%d "
                "legacy_per_turn_request_tokens=%d max_train_item_tokens=%d "
                "max_total_sequence_length=%s timeout_seconds=%s "
                "token_budget=%s token_budget_available=%s"
            )
            if (
                len(train_items) > 8
                or (
                    max_total_sequence_length is not None
                    and trajectory_request_tokens >= int(0.75 * max_total_sequence_length)
                )
            ):
                logger.warning(
                    "Large recontext logprob recompute workload: " + scoring_summary,
                    len(train_items),
                    trajectory_request_tokens,
                    legacy_per_turn_request_tokens,
                    max_train_item_tokens,
                    max_total_sequence_length,
                    cfg.logprob_recompute_timeout_seconds,
                    cfg.logprob_recompute_token_budget,
                    token_budget_available,
                )
            else:
                logger.info(
                    "Recontext logprob recompute workload: " + scoring_summary,
                    len(train_items),
                    trajectory_request_tokens,
                    legacy_per_turn_request_tokens,
                    max_train_item_tokens,
                    max_total_sequence_length,
                    cfg.logprob_recompute_timeout_seconds,
                    cfg.logprob_recompute_token_budget,
                    token_budget_available,
                )

            async def _score_all_generation_logprobs() -> list[list[float]]:
                for item_idx, item in enumerate(train_items):
                    _assert_original_target_matches_full_input(
                        item, full_input[train_positions[item_idx]]
                    )
                scored = await self._score_trajectory_generation_logprobs(
                    full_input=full_input,
                    target_positions_by_turn=train_positions,
                    swapped_system_text=swapped_system_text,
                    tools=tools,
                    prompt_token_ids_by_turn=spliced_prompt_token_ids,
                    generation_token_ids_by_turn=[
                        list(item.generation_token_ids) for item in train_items
                    ],
                    generation_log_probs_by_turn=[
                        list(item.generation_log_probs) for item in train_items
                    ],
                    temperature=temperature,
                    top_p=top_p,
                )
                for item, scored_turn in zip(train_items, scored):
                    if len(scored_turn) != len(item.generation_token_ids):
                        raise AssertionError(
                            "training token invariant failed: recomputed "
                            "generation_log_probs length does not match "
                            "generation_token_ids length"
                        )
                return scored

            try:
                if cfg.logprob_recompute_timeout_seconds is None:
                    recomputed_generation_logprobs = await _score_all_generation_logprobs()
                else:
                    recomputed_generation_logprobs = await asyncio.wait_for(
                        _score_all_generation_logprobs(),
                        timeout=cfg.logprob_recompute_timeout_seconds,
                    )
            except AssertionError:
                raise
            except TimeoutError:
                logger.exception(
                    "Timed out recomputing generation logprobs during recontext: "
                    "timeout_seconds=%s turns=%d trajectory_request_tokens=%d "
                    "legacy_per_turn_request_tokens=%d max_train_item_tokens=%d "
                    "token_budget=%s token_budget_available=%s",
                    cfg.logprob_recompute_timeout_seconds,
                    len(train_items),
                    trajectory_request_tokens,
                    legacy_per_turn_request_tokens,
                    max_train_item_tokens,
                    cfg.logprob_recompute_token_budget,
                    token_budget_available,
                )
                return (
                    False,
                    "logprob_recompute_timeout",
                    (
                        f"timeout_seconds={cfg.logprob_recompute_timeout_seconds} "
                        f"turns={len(train_items)} "
                        f"trajectory_request_tokens={trajectory_request_tokens} "
                        f"legacy_per_turn_request_tokens={legacy_per_turn_request_tokens} "
                        f"max_train_item_tokens={max_train_item_tokens} "
                        f"token_budget={cfg.logprob_recompute_token_budget} "
                        f"token_budget_available={token_budget_available}"
                    ),
                )
            except Exception as e:
                logger.exception("Failed to recompute generation logprobs during recontext")
                return False, f"logprob_recompute_error:{type(e).__name__}", str(e)

        # Apply the splice on every training-eligible turn. If recompute_logprobs
        # is enabled, logprobs are updated in the same atomic mutation block.
        for idx, item in enumerate(train_items):
            item.prompt_token_ids = spliced_prompt_token_ids[idx]
            if recomputed_generation_logprobs is not None:
                item.generation_log_probs = recomputed_generation_logprobs[idx]
            _assert_token_ids_equal(
                "mutated prompt_token_ids",
                spliced_prompt_token_ids[idx],
                item.prompt_token_ids,
            )
            _assert_token_ids_equal(
                "mutated generation_token_ids",
                original_generation_token_ids[idx],
                item.generation_token_ids,
            )

        return True, "", ""

    async def _score_generation_logprobs(
        self,
        full_input: NeMoGymResponseInput,
        target_position: int,
        swapped_system_text: str,
        tools: list[Any],
        prompt_token_ids: list[int],
        generation_token_ids: list[int],
        generation_log_probs: list[float],
        temperature: float,
        top_p: float,
    ) -> list[float]:
        """Use the existing chat-completions endpoint as a fixed-token scorer.

        The request is built from the real Responses transcript up to the
        target generated item. The target item carries the spliced
        prompt_token_ids + fixed generation_token_ids, and NeMo-RL's vLLM
        server replaces the chat-template prefix with those exact tokens before
        requesting prompt_logprobs. The single dummy decode token is ignored.
        """
        scoring_input = [
            _item_to_json_dict(item) for item in full_input[: target_position + 1]
        ]
        _set_first_system_content(scoring_input, swapped_system_text)
        for item in scoring_input:
            _strip_training_token_fields(item)

        target_item = scoring_input[-1]
        target_item["prompt_token_ids"] = prompt_token_ids
        target_item["generation_token_ids"] = generation_token_ids
        target_item["generation_log_probs"] = generation_log_probs

        metadata = {"extra_body": json.dumps({"prompt_logprobs": 0})}
        metadata["nemo_rl_return_final_prompt_token_ids"] = "true"
        chat_template_kwargs = self.config.recontextualization.chat_template_kwargs
        if chat_template_kwargs is not None:
            metadata["chat_template_kwargs"] = json.dumps(chat_template_kwargs)

        responses_params = NeMoGymResponseCreateParamsNonStreaming(
            input=scoring_input,
            tools=tools,
            max_output_tokens=1,
            temperature=temperature,
            top_p=top_p,
            metadata=metadata,
        )
        chat_params = VLLMConverter(
            return_token_id_information=True
        ).responses_to_chat_completion_create_params(responses_params)
        request_body = chat_params.model_dump(exclude_unset=True, mode="json")
        self._assert_scorer_request_matches_train_tokens(
            request_body=request_body,
            expected_prompt_token_ids=prompt_token_ids,
            expected_generation_token_ids=generation_token_ids,
        )
        raw_response = await self.server_client.post(
            server_name=self._logprob_recompute_model_server_name(),
            url_path="/v1/chat/completions",
            json=request_body,
        )
        raw_response.raise_for_status()
        payload = await raw_response.json()
        self._assert_scorer_payload_matches_train_tokens(
            payload=payload,
            expected_prompt_token_ids=prompt_token_ids,
            expected_generation_token_ids=generation_token_ids,
        )
        return _extract_generation_logprobs_from_prefill(
            payload.get("prompt_logprobs"),
            prompt_len=len(prompt_token_ids),
            generation_token_ids=generation_token_ids,
        )

    async def _score_trajectory_generation_logprobs(
        self,
        full_input: NeMoGymResponseInput,
        target_positions_by_turn: list[int],
        swapped_system_text: str,
        tools: list[Any],
        prompt_token_ids_by_turn: list[list[int]],
        generation_token_ids_by_turn: list[list[int]],
        generation_log_probs_by_turn: list[list[float]],
        temperature: float,
        top_p: float,
    ) -> list[list[float]]:
        """Score all trainable turns with one full-trajectory prefill request.

        NeMo-RL requires trainable messages to be contiguous token prefixes.
        Therefore the final trainable turn's prompt+generation sequence contains
        every earlier trainable turn's prompt+generation span, so one vLLM
        ``prompt_logprobs`` response can be sliced into per-turn logprobs.
        """
        final_turn_idx = len(prompt_token_ids_by_turn) - 1
        final_prompt_token_ids = prompt_token_ids_by_turn[final_turn_idx]
        final_generation_token_ids = generation_token_ids_by_turn[final_turn_idx]
        request_token_count = len(final_prompt_token_ids) + len(
            final_generation_token_ids
        )
        token_budget = self._logprob_recompute_token_budget
        if token_budget is not None and request_token_count > token_budget.capacity:
            raise ValueError(
                "logprob_recompute_token_budget is smaller than the exact "
                "full-trajectory scorer request: "
                f"{token_budget.capacity}<{request_token_count}"
            )

        if token_budget is not None:
            acquired_full_request = await token_budget.try_acquire(request_token_count)
            if not acquired_full_request:
                return await self._score_trajectory_generation_logprobs_chunked(
                    full_input=full_input,
                    target_positions_by_turn=target_positions_by_turn,
                    swapped_system_text=swapped_system_text,
                    tools=tools,
                    prompt_token_ids_by_turn=prompt_token_ids_by_turn,
                    generation_token_ids_by_turn=generation_token_ids_by_turn,
                    generation_log_probs_by_turn=generation_log_probs_by_turn,
                    temperature=temperature,
                    top_p=top_p,
                    full_request_token_count=request_token_count,
                    token_budget=token_budget,
                )

        try:
            request_id = uuid.uuid4().hex[:12]
            request_body = self._build_trajectory_scorer_request_body(
                full_input=full_input,
                target_position=target_positions_by_turn[final_turn_idx],
                swapped_system_text=swapped_system_text,
                tools=tools,
                prompt_token_ids=final_prompt_token_ids,
                generation_token_ids=final_generation_token_ids,
                generation_log_probs=generation_log_probs_by_turn[final_turn_idx],
                temperature=temperature,
                top_p=top_p,
                request_id=request_id,
            )
            payload = await self._post_trajectory_scorer_request(
                request_body=request_body,
                request_id=request_id,
                turns=len(prompt_token_ids_by_turn),
                target_turn_idx=final_turn_idx,
                request_token_count=request_token_count,
                prompt_token_count=len(final_prompt_token_ids),
                generation_token_count=len(final_generation_token_ids),
                reserved_tokens=request_token_count
                if token_budget is not None
                else None,
            )
        finally:
            if token_budget is not None:
                await token_budget.release(request_token_count)

        return self._extract_trajectory_logprobs_from_payload(
            payload=payload,
            expected_prompt_token_ids=final_prompt_token_ids,
            expected_generation_token_ids=final_generation_token_ids,
            prompt_token_ids_by_turn=prompt_token_ids_by_turn,
            generation_token_ids_by_turn=generation_token_ids_by_turn,
            prefix_label="full trajectory scorer token prefix",
        )

    def _build_trajectory_scorer_request_body(
        self,
        full_input: NeMoGymResponseInput,
        target_position: int,
        swapped_system_text: str,
        tools: list[Any],
        prompt_token_ids: list[int],
        generation_token_ids: list[int],
        generation_log_probs: list[float],
        temperature: float,
        top_p: float,
        request_id: str,
    ) -> dict[str, Any]:
        request_token_count = len(prompt_token_ids) + len(generation_token_ids)

        scoring_input = [
            _item_to_json_dict(item) for item in full_input[: target_position + 1]
        ]
        _set_first_system_content(scoring_input, swapped_system_text)
        for item in scoring_input:
            _strip_training_token_fields(item)

        target_item = scoring_input[-1]
        target_item["prompt_token_ids"] = prompt_token_ids
        target_item["generation_token_ids"] = generation_token_ids
        target_item["generation_log_probs"] = generation_log_probs

        metadata = {"extra_body": json.dumps({"prompt_logprobs": 0})}
        metadata["nemo_rl_return_final_prompt_token_ids"] = "true"
        metadata["recontext_request_id"] = request_id
        chat_template_kwargs = self.config.recontextualization.chat_template_kwargs
        if chat_template_kwargs is not None:
            metadata["chat_template_kwargs"] = json.dumps(chat_template_kwargs)

        responses_params = NeMoGymResponseCreateParamsNonStreaming(
            input=scoring_input,
            tools=tools,
            max_output_tokens=1,
            temperature=temperature,
            top_p=top_p,
            metadata=metadata,
        )
        chat_params = VLLMConverter(
            return_token_id_information=True
        ).responses_to_chat_completion_create_params(responses_params)
        request_body = chat_params.model_dump(exclude_unset=True, mode="json")
        self._assert_scorer_request_matches_train_tokens(
            request_body=request_body,
            expected_prompt_token_ids=prompt_token_ids,
            expected_generation_token_ids=generation_token_ids,
        )
        body_token_count = len(prompt_token_ids) + len(generation_token_ids)
        if body_token_count != request_token_count:
            raise AssertionError("training token invariant failed: scorer body size drift")
        return request_body

    async def _post_trajectory_scorer_request(
        self,
        request_body: dict[str, Any],
        request_id: str,
        turns: int,
        target_turn_idx: int,
        request_token_count: int,
        prompt_token_count: int,
        generation_token_count: int,
        reserved_tokens: Optional[int] = None,
        chunk_index: Optional[int] = None,
    ) -> dict[str, Any]:
        server_name = self._logprob_recompute_model_server_name()
        logger.warning(
            "Starting recontext trajectory scorer request: request_id=%s "
            "server_name=%s "
            "turns=%d target_turn=%d request_tokens=%d prompt_tokens=%d "
            "generation_tokens=%d reserved_tokens=%s chunk_index=%s "
            "timeout_seconds=%s",
            request_id,
            server_name,
            turns,
            target_turn_idx,
            request_token_count,
            prompt_token_count,
            generation_token_count,
            reserved_tokens,
            chunk_index,
            self.config.recontextualization.logprob_recompute_timeout_seconds,
        )
        start_time = time.perf_counter()
        try:
            raw_response = await self.server_client.post(
                server_name=server_name,
                url_path="/v1/chat/completions",
                json=request_body,
            )
        except asyncio.CancelledError:
            duration = time.perf_counter() - start_time
            logger.warning(
                "Cancelled recontext trajectory scorer request: request_id=%s "
                "server_name=%s "
                "duration_seconds=%.2f turns=%d target_turn=%d "
                "request_tokens=%d prompt_tokens=%d generation_tokens=%d "
                "reserved_tokens=%s chunk_index=%s",
                request_id,
                server_name,
                duration,
                turns,
                target_turn_idx,
                request_token_count,
                prompt_token_count,
                generation_token_count,
                reserved_tokens,
                chunk_index,
            )
            raise
        raw_response.raise_for_status()
        payload = await raw_response.json()
        duration = time.perf_counter() - start_time
        prompt_logprobs = payload.get("prompt_logprobs")
        prompt_logprobs_len = (
            len(prompt_logprobs) if isinstance(prompt_logprobs, list) else None
        )
        log_scorer_finished = logger.warning if duration >= 60.0 else logger.info
        log_scorer_finished(
            "Finished recontext trajectory scorer request: request_id=%s "
            "server_name=%s "
            "duration_seconds=%.2f turns=%d target_turn=%d request_tokens=%d "
            "reserved_tokens=%s chunk_index=%s prompt_logprobs_len=%s",
            request_id,
            server_name,
            duration,
            turns,
            target_turn_idx,
            request_token_count,
            reserved_tokens,
            chunk_index,
            prompt_logprobs_len,
        )
        return payload

    def _extract_trajectory_logprobs_from_payload(
        self,
        payload: dict[str, Any],
        expected_prompt_token_ids: list[int],
        expected_generation_token_ids: list[int],
        prompt_token_ids_by_turn: list[list[int]],
        generation_token_ids_by_turn: list[list[int]],
        prefix_label: str,
    ) -> list[list[float]]:
        final_prompt_token_ids_from_server = self._assert_scorer_payload_matches_train_tokens(
            payload=payload,
            expected_prompt_token_ids=expected_prompt_token_ids,
            expected_generation_token_ids=expected_generation_token_ids,
        )

        recomputed: list[list[float]] = []
        for turn_idx, (prompt_token_ids, generation_token_ids) in enumerate(
            zip(prompt_token_ids_by_turn, generation_token_ids_by_turn)
        ):
            expected_prefix = list(prompt_token_ids) + list(generation_token_ids)
            actual_prefix = final_prompt_token_ids_from_server[: len(expected_prefix)]
            _assert_token_ids_equal(
                f"{prefix_label} turn={turn_idx}",
                expected_prefix,
                actual_prefix,
            )
            recomputed.append(
                _extract_generation_logprobs_from_prefill(
                    payload.get("prompt_logprobs"),
                    prompt_len=len(prompt_token_ids),
                    generation_token_ids=generation_token_ids,
                )
            )
        return recomputed

    async def _score_trajectory_generation_logprobs_chunked(
        self,
        full_input: NeMoGymResponseInput,
        target_positions_by_turn: list[int],
        swapped_system_text: str,
        tools: list[Any],
        prompt_token_ids_by_turn: list[list[int]],
        generation_token_ids_by_turn: list[list[int]],
        generation_log_probs_by_turn: list[list[float]],
        temperature: float,
        top_p: float,
        full_request_token_count: int,
        token_budget: _AsyncTokenBudget,
    ) -> list[list[float]]:
        target_spans: list[tuple[int, int, int, int]] = []
        for turn_idx, (prompt_token_ids, generation_token_ids) in enumerate(
            zip(prompt_token_ids_by_turn, generation_token_ids_by_turn)
        ):
            prompt_len = len(prompt_token_ids)
            for local_idx, token_id in enumerate(generation_token_ids):
                target_spans.append(
                    (prompt_len + local_idx, turn_idx, local_idx, int(token_id))
                )
        target_spans.sort(key=lambda x: x[0])
        if not target_spans:
            return [[] for _ in generation_token_ids_by_turn]

        base_request_id = uuid.uuid4().hex[:12]
        logger.warning(
            "Starting chunked recontext trajectory scorer: request_id=%s "
            "turns=%d full_request_tokens=%d token_budget_capacity=%d "
            "token_budget_available=%d target_tokens=%d",
            base_request_id,
            len(prompt_token_ids_by_turn),
            full_request_token_count,
            token_budget.capacity,
            token_budget.available,
            len(target_spans),
        )

        recomputed: list[list[Optional[float]]] = [
            [None] * len(generation_token_ids)
            for generation_token_ids in generation_token_ids_by_turn
        ]
        cursor = 0
        chunk_index = 0
        while cursor < len(target_spans):
            min_tokens_for_next_target = target_spans[cursor][0] + 1
            reserved_tokens = await token_budget.acquire_at_most(
                requested=full_request_token_count,
                minimum=min_tokens_for_next_target,
            )
            try:
                max_chunk_end = min(reserved_tokens, full_request_token_count)
                end_cursor = cursor
                while (
                    end_cursor < len(target_spans)
                    and target_spans[end_cursor][0] < max_chunk_end
                ):
                    end_cursor += 1
                if end_cursor == cursor:
                    raise AssertionError(
                        "training token invariant failed: token budget chunk "
                        "did not include the next generated token"
                    )

                chunk_end_global = target_spans[end_cursor - 1][0] + 1
                target_turn_idx = target_spans[end_cursor - 1][1]
                local_generation_end = chunk_end_global - len(
                    prompt_token_ids_by_turn[target_turn_idx]
                )
                if local_generation_end <= 0 or local_generation_end > len(
                    generation_token_ids_by_turn[target_turn_idx]
                ):
                    raise AssertionError(
                        "training token invariant failed: invalid chunk target span"
                    )

                request_id = f"{base_request_id}-c{chunk_index}"
                prompt_token_ids = prompt_token_ids_by_turn[target_turn_idx]
                generation_token_ids = generation_token_ids_by_turn[target_turn_idx][
                    :local_generation_end
                ]
                generation_log_probs = generation_log_probs_by_turn[target_turn_idx][
                    :local_generation_end
                ]
                request_body = self._build_trajectory_scorer_request_body(
                    full_input=full_input,
                    target_position=target_positions_by_turn[target_turn_idx],
                    swapped_system_text=swapped_system_text,
                    tools=tools,
                    prompt_token_ids=prompt_token_ids,
                    generation_token_ids=generation_token_ids,
                    generation_log_probs=generation_log_probs,
                    temperature=temperature,
                    top_p=top_p,
                    request_id=request_id,
                )
                payload = await self._post_trajectory_scorer_request(
                    request_body=request_body,
                    request_id=request_id,
                    turns=len(prompt_token_ids_by_turn),
                    target_turn_idx=target_turn_idx,
                    request_token_count=chunk_end_global,
                    prompt_token_count=len(prompt_token_ids),
                    generation_token_count=len(generation_token_ids),
                    reserved_tokens=reserved_tokens,
                    chunk_index=chunk_index,
                )
                scorer_token_ids = self._assert_scorer_payload_matches_train_tokens(
                    payload=payload,
                    expected_prompt_token_ids=prompt_token_ids,
                    expected_generation_token_ids=generation_token_ids,
                )
                prompt_logprobs = payload.get("prompt_logprobs")
                if isinstance(prompt_logprobs, dict) and "content" in prompt_logprobs:
                    prompt_logprobs = prompt_logprobs["content"]
                if not isinstance(prompt_logprobs, list):
                    raise TypeError(
                        "expected prompt_logprobs to be a list, got "
                        f"{type(prompt_logprobs).__name__}"
                    )
                if len(prompt_logprobs) < chunk_end_global:
                    raise ValueError(
                        "prompt_logprobs too short for chunk: "
                        f"got {len(prompt_logprobs)} entries, need {chunk_end_global}"
                    )

                max_included_local_idx_by_turn: dict[int, int] = {}
                for _, turn_idx, local_idx, _ in target_spans[cursor:end_cursor]:
                    max_included_local_idx_by_turn[turn_idx] = max(
                        local_idx,
                        max_included_local_idx_by_turn.get(turn_idx, -1),
                    )
                for turn_idx, max_local_idx in max_included_local_idx_by_turn.items():
                    expected_prefix = (
                        list(prompt_token_ids_by_turn[turn_idx])
                        + list(generation_token_ids_by_turn[turn_idx][: max_local_idx + 1])
                    )
                    actual_prefix = scorer_token_ids[: len(expected_prefix)]
                    _assert_token_ids_equal(
                        f"chunked trajectory scorer token prefix turn={turn_idx}",
                        expected_prefix,
                        actual_prefix,
                    )

                for global_idx, turn_idx, local_idx, token_id in target_spans[
                    cursor:end_cursor
                ]:
                    recomputed[turn_idx][local_idx] = _extract_logprob_value(
                        prompt_logprobs[global_idx], token_id
                    )
                cursor = end_cursor
                chunk_index += 1
            finally:
                await token_budget.release(reserved_tokens)

        result: list[list[float]] = []
        for turn_idx, turn_logprobs in enumerate(recomputed):
            if any(logprob is None for logprob in turn_logprobs):
                raise AssertionError(
                    "training token invariant failed: missing chunked recompute "
                    f"logprobs for turn={turn_idx}"
                )
            result.append([float(logprob) for logprob in turn_logprobs])
        return result

    def _assert_scorer_request_matches_train_tokens(
        self,
        request_body: dict[str, Any],
        expected_prompt_token_ids: list[int],
        expected_generation_token_ids: list[int],
    ) -> None:
        """Assert the outbound scorer request carries the train token fields."""
        messages = request_body.get("messages") or []
        if not messages:
            raise AssertionError("training token invariant failed: no scorer messages")
        target_message = messages[-1]
        if target_message.get("role") != "assistant":
            raise AssertionError(
                "training token invariant failed: scorer target is not assistant"
            )
        _assert_token_ids_equal(
            "scorer request prompt_token_ids",
            expected_prompt_token_ids,
            target_message.get("prompt_token_ids") or [],
        )
        _assert_token_ids_equal(
            "scorer request generation_token_ids",
            expected_generation_token_ids,
            target_message.get("generation_token_ids") or [],
        )

    def _assert_scorer_payload_matches_train_tokens(
        self,
        payload: dict[str, Any],
        expected_prompt_token_ids: list[int],
        expected_generation_token_ids: list[int],
    ) -> list[int]:
        expected_prefix = list(expected_prompt_token_ids) + list(
            expected_generation_token_ids
        )
        scorer_prompt_token_ids = payload.get("nemo_rl_final_prompt_token_ids")
        if scorer_prompt_token_ids is None:
            raise AssertionError(
                "training token invariant failed: scorer did not return "
                "nemo_rl_final_prompt_token_ids"
            )
        scorer_prefix = list(scorer_prompt_token_ids[: len(expected_prefix)])
        _assert_token_ids_equal(
            "chat completion scorer token prefix",
            expected_prefix,
            scorer_prefix,
        )
        return [int(token_id) for token_id in scorer_prompt_token_ids]

    @staticmethod
    def _all_training_eligible(
        output: list[NeMoGymResponseOutputItem] | list[list[NeMoGymResponseOutputItem]],
    ) -> list[Any]:
        """Return every output item with non-empty prompt_token_ids and
        generation_token_ids — these are the assistant turns the trainer
        will compute gradients on."""
        items: list[Any] = []
        for x in output:
            if isinstance(x, list):
                items.extend(x)
            else:
                items.append(x)
        train: list[Any] = []
        for item in items:
            ptids = getattr(item, "prompt_token_ids", None)
            gtids = getattr(item, "generation_token_ids", None)
            if ptids and gtids:
                train.append(item)
        return train

    def _make_prefix_diagnostic(
        self,
        turn_index: int,
        item: Any,
        orig_sys_tokens: list[int],
        check: PrefixCheckResult,
    ) -> str:
        """Compose a short, parseable diagnostic when the system-prefix check
        fails. Lands in the wandb Table so we can inspect WHY vLLM's
        tokenization of the system block diverges from ours."""
        ptids = list(getattr(item, "prompt_token_ids", []) or [])
        if check.divergence_index < 0:
            return (
                f"turn={turn_index} prompt_len={len(ptids)} "
                f"orig_sys_len={len(orig_sys_tokens)} reason={check.reason}"
            )
        i = check.divergence_index
        orig_snippet = decode_snippet(self._tokenizer, orig_sys_tokens, around=i)
        rollout_snippet = decode_snippet(self._tokenizer, ptids, around=i)
        return (
            f"turn={turn_index} divergence_index={i} "
            f"prompt_len={len(ptids)} orig_sys_len={len(orig_sys_tokens)} "
            f"orig_token={orig_sys_tokens[i] if i < len(orig_sys_tokens) else '?'} "
            f"rollout_token={ptids[i] if i < len(ptids) else '?'} | "
            f"orig_snippet={orig_snippet!r} | rollout_snippet={rollout_snippet!r}"
        )


if __name__ == "__main__":
    RecontextAviaryAgent.run_webserver()
