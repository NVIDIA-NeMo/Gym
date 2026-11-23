# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multimodal Responses API agent for the WebArena resources server."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

import aiohttp
from fastapi import Body, HTTPException, Request, Response
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.agents.context_management import (
    ContextGuardRejected,
    ContextHistoryConfig,
    ContextManagedResponsesClient,
)
from nemo_gym.config import TOKEN_CAPTURE_BLOCK
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.webarena import schemas as webarena_schemas
from resources_servers.webarena.actions import decode_tool_arguments, validate_tool_arguments
from resources_servers.webarena.schemas import WebArenaAgentRunRequest


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a GUI agent controlling a web browser. You are given a task instruction, a screenshot of the browser, and your previous interactions. You need to perform a series of actions to complete the task. The browser is already open and logged into the required websites.

<tool_guidelines>
- Operate via x,y coordinates from the latest screenshot using the `computer` tool.
- Coordinates are relative to the viewport in [0, 1], with (0, 0) at the top-left.
- Use `tabs_create` and `tabs_focus` to manage tabs.
- Use `navigate` to go to URLs or use "back"/"forward" for browser history.
- When the task is complete, call `terminate` with status and answer.
</tool_guidelines>
""".strip()

TASK_INPUT_IMAGE_MARKER = "_webarena_task_input_image"


def _shared_schema(*names: str) -> type[BaseModel]:
    """Resolve the canonical shared schema while tolerating an older class name."""
    for name in names:
        schema = getattr(webarena_schemas, name, None)
        if schema is not None:
            return schema
    raise ImportError(f"resources_servers.webarena.schemas is missing {names[0]}")


WebArenaSeedSessionResponse = _shared_schema("WebArenaSeedSessionResponse", "WebArenaSeedResponse")
WebArenaStepResponse = _shared_schema("WebArenaStepResponse")
WebArenaNeMoGymResponse = _shared_schema("WebArenaNeMoGymResponse")
WebArenaAgentVerifyResponse = _shared_schema("WebArenaAgentVerifyResponse", "WebArenaVerifyResponse")
WebArenaFinalizeResponse = _shared_schema("WebArenaFinalizeResponse")


class TerminationReason(str, Enum):
    ENV_DONE = "env_done"
    MAX_STEPS = "max_steps"
    NO_TOOL_CALL = "no_tool_call"
    MULTIPLE_TOOL_CALLS = "multiple_tool_calls"
    INVALID_TOOL_CALL = "invalid_tool_call"
    TRUNCATED = "truncated"
    CONTEXT_LIMIT = "context_limit"
    MODEL_ERROR = "model_error"
    ENVIRONMENT_ERROR = "environment_error"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    ROLLOUT_TIMEOUT = "rollout_timeout"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


# Policy mistakes: the rollout stops at the offending turn and is scored 0
# (truncation instead gets -terminate_properly_reward when the server reports one).
_ZERO_REWARD_REASONS = frozenset(
    {
        TerminationReason.NO_TOOL_CALL,
        TerminationReason.MULTIPLE_TOOL_CALLS,
        TerminationReason.INVALID_TOOL_CALL,
        TerminationReason.TRUNCATED,
        TerminationReason.CONTEXT_LIMIT,
        TerminationReason.MAX_STEPS,
    }
)
# Failures the policy is not responsible for: the rollout is excluded from training.
_MASKED_REASONS = frozenset(
    {
        TerminationReason.MODEL_ERROR,
        TerminationReason.ENVIRONMENT_ERROR,
        TerminationReason.TOOL_EXECUTION_ERROR,
        TerminationReason.ROLLOUT_TIMEOUT,
    }
)


class WebArenaTerminationMetadata(BaseModel):
    reason: TerminationReason
    detail: str | None = None
    environment_steps: int = 0
    model_calls: int = 0
    close_error: str | None = None


class WebArenaRolloutError(RuntimeError):
    """Typed failure raised when no trainable model turn can be returned."""

    def __init__(self, reason: TerminationReason, message: str):
        super().__init__(message)
        self.reason = reason


_CLOSE_TIMEOUT_SECONDS = 30.0


class WebAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = Field(default=50, ge=1)
    return_transitions: Literal[False] = False
    max_recent_screenshots: int = Field(default=3, ge=0)
    rollout_timeout_seconds: float = Field(default=1200, gt=0)
    finalization_reserve_seconds: float = Field(default=360, ge=0)
    # Applies to step and verification requests. Seed is bounded by the
    # resources server's separate browser-pool and browser-operation timeouts.
    resource_request_timeout_seconds: float = Field(default=510, gt=0)
    system_prompt: str = SYSTEM_PROMPT
    include_initial_request_input: bool = True
    # Shared semantic context management (nemo_gym.agents.context_management).
    context_history: ContextHistoryConfig = Field(default_factory=ContextHistoryConfig)
    # Opt into framework training-token capture (gated by the run-level
    # token_id_capture.enabled); required whenever context_history is enabled.
    token_id_capture: bool = True

    @model_validator(mode="after")
    def validate_finalization_reserve(self) -> Self:
        if self.finalization_reserve_seconds <= _CLOSE_TIMEOUT_SECONDS:
            raise ValueError("finalization_reserve_seconds must leave time for verification before cleanup")
        if self.finalization_reserve_seconds >= self.rollout_timeout_seconds:
            raise ValueError("finalization_reserve_seconds must be less than rollout_timeout_seconds")
        if self.context_history.enabled and "max_recent_screenshots" in self.model_fields_set:
            raise ValueError(
                "max_recent_screenshots is the legacy pruning knob and must not be set when "
                "context_history.enabled; configure context_history.policy (e.g. recency with "
                "images.keep_last_groups) instead"
            )
        return self


def _item_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return copy.deepcopy(item)
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json", exclude_none=True)
    raise TypeError(f"Unsupported response input item: {type(item).__name__}")


def _validated_request_body(
    body: NeMoGymResponseCreateParamsNonStreaming,
    *,
    request_input: list[Any],
) -> NeMoGymResponseCreateParamsNonStreaming:
    body_data = body.model_dump(mode="python", exclude_unset=True)
    body_data.update({"input": list(request_input)})
    return NeMoGymResponseCreateParamsNonStreaming.model_validate(body_data)


def _without_raw_images(items: list[Any]) -> list[dict[str, Any]]:
    """Keep the semantic skeleton; the trainer reads media from framework token capture."""
    projected_items = []
    for item in items:
        projected = _item_dict(item)
        if isinstance(projected.get("content"), list):
            projected["content"] = [
                part
                for part in projected["content"]
                if not (isinstance(part, dict) and part.get("type") in {"input_image", "image", "image_url"})
            ]
        projected_items.append(projected)
    return projected_items


def _path_rollout_id(request: Any) -> str | None:
    """Framework logical owner ID from the ``/rollouts/{rollout_id}/...`` route, if any."""
    params = getattr(request, "path_params", None) or {}
    value = params.get("rollout_id") if hasattr(params, "get") else None
    return value if isinstance(value, str) else None


def _is_environment_state(item: Any) -> bool:
    data = _item_dict(item)
    return bool(data.get("is_env_state") or data.get("webarena_env_state") or data.get("env_info") is not None)


def _is_browser_image(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") == "input_image" and not bool(part.get(TASK_INPUT_IMAGE_MARKER))


def _strip_internal_image_markers(item: dict[str, Any]) -> dict[str, Any]:
    content = item.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                part.pop(TASK_INPUT_IMAGE_MARKER, None)
    return item


def compact_model_input(history: list[Any], max_recent_screenshots: int) -> list[dict[str, Any]]:
    """Remove only old browser screenshots from WebArena environment turns."""
    messages = [_item_dict(item) for item in history]
    browser_parts: dict[int, int] = {}
    for index, item in enumerate(history):
        if not _is_environment_state(item):
            continue
        content = messages[index].get("content", [])
        browser_index = next(
            (part_index for part_index, part in enumerate(content) if _is_browser_image(part)),
            None,
        )
        if browser_index is not None:
            browser_parts[index] = browser_index
    screenshot_indices = list(browser_parts)
    keep = set(screenshot_indices[-max_recent_screenshots:]) if max_recent_screenshots else set()
    for index in screenshot_indices:
        if index not in keep:
            messages[index]["content"] = [
                part
                for part_index, part in enumerate(messages[index].get("content", []))
                if part_index != browser_parts[index]
            ]
    return [_strip_internal_image_markers(message) for message in messages]


def _aggregate_usage(
    running: NeMoGymResponseUsage | None,
    latest: NeMoGymResponseUsage | None,
) -> NeMoGymResponseUsage | None:
    if latest is None:
        return running
    if running is None:
        return latest.model_copy(deep=True)
    update: dict[str, Any] = {
        "input_tokens": running.input_tokens + latest.input_tokens,
        "output_tokens": running.output_tokens + latest.output_tokens,
        "total_tokens": running.total_tokens + latest.total_tokens,
    }
    for field_name in ("input_tokens_details", "output_tokens_details"):
        old_details = getattr(running, field_name, None)
        new_details = getattr(latest, field_name, None)
        if old_details is None or new_details is None:
            continue
        detail_update = {}
        for detail_field in type(old_details).model_fields:
            old_value = getattr(old_details, detail_field, None)
            new_value = getattr(new_details, detail_field, None)
            if isinstance(old_value, int | float) and isinstance(new_value, int | float):
                detail_update[detail_field] = old_value + new_value
        update[field_name] = old_details.model_copy(update=detail_update)
    return running.model_copy(update=update)


def _cookies_after(current: Any, raw_response: Any) -> Any:
    new_cookies = getattr(raw_response, "cookies", None)
    merged: dict[str, str] = {}
    for cookie_source in (current, new_cookies):
        for key, value in (cookie_source or {}).items():
            merged[key] = str(getattr(value, "value", value))
    return merged


def _is_context_limit_error(exc: BaseException) -> bool:
    content = getattr(exc, "response_content", b"")
    if isinstance(content, bytes):
        content = content.decode(errors="replace")
    text = f"{exc} {content}".lower()
    return (
        "context length" in text or "maximum model length" in text or ("decoder prompt" in text and "maximum" in text)
    )


def _tool_name(tool: Any) -> str | None:
    return tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)


def _incomplete_reason(response: NeMoGymResponse) -> str | None:
    details = getattr(response, "incomplete_details", None)
    return getattr(details, "reason", None)


class WebAgent(SimpleResponsesAPIAgent):
    config: WebAgentConfig

    async def _post_json(
        self,
        server_name: str,
        url_path: str,
        payload: Any,
        cookies: Any,
        timeout_seconds: float | None,
    ) -> tuple[Any, Any]:
        async def send() -> tuple[Any, Any]:
            raw_response = await self.server_client.post(
                server_name=server_name,
                url_path=url_path,
                json=payload,
                cookies=cookies,
            )
            await raise_for_status(raw_response)
            data = await get_response_json(raw_response)
            return data, _cookies_after(cookies, raw_response)

        if timeout_seconds is None:
            return await send()
        async with asyncio.timeout(timeout_seconds):
            return await send()

    def _incoming_input(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
    ) -> list[Any]:
        if self.config.include_initial_request_input:
            incoming = (
                [NeMoGymEasyInputMessage(role="user", content=body.input)]
                if isinstance(body.input, str)
                else list(body.input)
            )
        else:
            incoming = []
        if self.config.system_prompt:
            first_role = _item_dict(incoming[0]).get("role") if incoming else None
            if first_role != "system":
                incoming.insert(0, NeMoGymEasyInputMessage(role="system", content=self.config.system_prompt))
        return incoming

    @staticmethod
    def _compaction_seed_split(
        incoming: list[Any],
        seed_obs: list[Any],
        *,
        authority: bool,
    ) -> tuple[list[Any], list[Any]]:
        """Split the rollout opening into (initial_context, seed_observations) for CC.

        In shadow mode both sequences must reproduce the legacy history exactly
        (the identity check compares them byte-for-byte), so no restructuring
        happens. In authority mode, WebArena's step-1 observation packs the
        screenshot and any task-input images into one message; task images are
        moved into the initial context (permanently protected under
        ``protect_initial_context``) while the screenshot stays a seed
        observation that ages out under recency. The internal task-image marker
        is stripped before anything reaches the session, otherwise the media
        arena would canonicalize it back into model requests.
        """
        if not authority:
            return list(incoming), list(seed_obs)

        def strip_marker(item: Any) -> Any:
            data = _item_dict(item)
            content = data.get("content")
            if not isinstance(content, list) or not any(
                isinstance(part, dict) and TASK_INPUT_IMAGE_MARKER in part for part in content
            ):
                return item
            cleaned = [
                {key: value for key, value in part.items() if key != TASK_INPUT_IMAGE_MARKER}
                if isinstance(part, dict)
                else part
                for part in content
            ]
            return {**data, "content": cleaned}

        protected: list[dict[str, Any]] = []
        seeds: list[Any] = []
        for item in seed_obs:
            data = _item_dict(item)
            content = data.get("content")
            if not isinstance(content, list):
                seeds.append(data)
                continue
            task_parts: list[dict[str, Any]] = []
            rest_parts: list[Any] = []
            for part in content:
                if isinstance(part, dict) and part.get(TASK_INPUT_IMAGE_MARKER):
                    clean = dict(part)
                    clean.pop(TASK_INPUT_IMAGE_MARKER, None)
                    task_parts.append(clean)
                else:
                    rest_parts.append(part)
            base = {key: value for key, value in data.items() if key != "content"}
            if task_parts:
                protected.append({**base, "content": task_parts})
            if rest_parts:
                seeds.append({**base, "content": rest_parts})
        return [*[strip_marker(item) for item in incoming], *protected], seeds

    @staticmethod
    def _task_payload(body: WebArenaAgentRunRequest) -> dict[str, Any]:
        return body.task_payload.model_dump(mode="json")

    async def _call_model(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        cookies: Any,
    ) -> tuple[NeMoGymResponse, Any]:
        data, cookies = await self._post_json(
            self.config.model_server.name,
            "/v1/responses",
            body,
            cookies,
            None,
        )
        return NeMoGymResponse.model_validate(data), cookies

    @staticmethod
    def _function_calls(response: NeMoGymResponse) -> list[NeMoGymResponseFunctionToolCall]:
        return [item for item in response.output if item.type == "function_call"]

    def _protocol_violation(
        self,
        response: NeMoGymResponse,
        offered_tools: frozenset[str],
    ) -> tuple[TerminationReason, str] | None:
        """Return why this response cannot be executed, or None for exactly one valid call."""
        if _incomplete_reason(response) == "max_output_tokens":
            return TerminationReason.TRUNCATED, "response hit max_output_tokens"
        function_calls = self._function_calls(response)
        if not function_calls:
            return TerminationReason.NO_TOOL_CALL, "response contained no function call"
        if len(function_calls) != 1:
            return TerminationReason.MULTIPLE_TOOL_CALLS, f"response contained {len(function_calls)} function calls"
        call = function_calls[0]
        if offered_tools and call.name not in offered_tools:
            return TerminationReason.INVALID_TOOL_CALL, f"tool {call.name!r} was not offered"
        try:
            validate_tool_arguments(call.name, decode_tool_arguments(call.arguments))
        except (TypeError, ValueError) as exc:
            return TerminationReason.INVALID_TOOL_CALL, f"{call.name}: {exc}"
        return None

    async def _close(self, env_id: str, cookies: Any, request_id: str) -> str | None:
        try:
            await self._post_json(
                self.config.resources_server.name,
                "/close",
                {"env_id": env_id, "close_request_id": request_id},
                cookies,
                _CLOSE_TIMEOUT_SECONDS,
            )
        except BaseException as exc:
            logger.warning("Best-effort WebArena close failed for env_id=%s", env_id, exc_info=True)
            return repr(exc)[:200]
        return None

    async def _finalize(
        self,
        env_id: str,
        cookies: Any,
        finalize_request_id: str,
    ) -> WebArenaFinalizeResponse:
        payload = {
            "env_id": env_id,
            "finalize_request_id": finalize_request_id,
        }
        deadline = asyncio.get_running_loop().time() + (
            self.config.finalization_reserve_seconds - _CLOSE_TIMEOUT_SECONDS
        )
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(f"WebArena finalization timed out for env_id={env_id}")
            try:
                # Leave room for at least one same-ID retry if this HTTP wait is
                # interrupted while the server-owned finalization keeps running.
                attempt_timeout = min(
                    self.config.resource_request_timeout_seconds,
                    remaining,
                    max(1.0, remaining / 2),
                )
                data, _ = await self._post_json(
                    self.config.resources_server.name,
                    "/finalize",
                    payload,
                    cookies,
                    attempt_timeout,
                )
                response = WebArenaFinalizeResponse.model_validate(data)
                if response.env_id != env_id or response.finalize_request_id != finalize_request_id:
                    raise RuntimeError("WebArena finalize response identity did not match the request")
                return response
            except (TimeoutError, aiohttp.ClientConnectionError):
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise
                await asyncio.sleep(min(1.0, remaining))

    def _require_logical_rollout_id(self, logical_rollout_id: str | None) -> str:
        """Context management needs framework token capture and the framework-owned logical owner ID."""
        if not self._token_id_capture_enabled() or not logical_rollout_id or "/" in logical_rollout_id:
            raise ValueError(
                "context_history.enabled requires training token capture and a framework "
                "logical rollout ID (rollouts/<id>/... path); disable context_history for "
                "capture-less evaluation"
            )
        history = self.config.context_history
        rewrites = history.policy.type != "identity"
        capture = self.server_client.global_config_dict.get(TOKEN_CAPTURE_BLOCK) or {}
        if (rewrites or history.guards.max_total_tokens is not None) and capture.get(
            "parent_resolution"
        ) != "source_history":
            raise ValueError(
                "A rewriting context_history policy or a token guard requires "
                "token_id_capture.parent_resolution=source_history"
            )
        return logical_rollout_id

    async def _rollout(
        self,
        body: WebArenaAgentRunRequest,
        initial_cookies: Any,
        *,
        close_on_success: bool = True,
        logical_rollout_id: str | None = None,
    ) -> tuple[BaseModel, Any]:
        body = body.model_copy(deep=True)
        # CC v2 identity is validated before any environment mutation.
        owner_id = (
            self._require_logical_rollout_id(logical_rollout_id) if self.config.context_history.enabled else None
        )
        request_id = str(uuid4())
        seed_payload = {"request_id": request_id, "task": self._task_payload(body)}
        try:
            # Browser checkout can legitimately wait for pool capacity. Let the
            # resources server enforce its pool and reset-operation budgets so
            # queue time is not charged to the model/browser interaction phase.
            seed_data, resource_cookies = await self._post_json(
                self.config.resources_server.name,
                "/seed_session",
                seed_payload,
                initial_cookies,
                None,
            )
        except Exception as exc:
            raise WebArenaRolloutError(
                TerminationReason.ENVIRONMENT_ERROR,
                f"WebArena seed failed: {exc!r}",
            ) from exc
        seed = WebArenaSeedSessionResponse.model_validate(seed_data)
        if not seed.obs:
            raise WebArenaRolloutError(TerminationReason.ENVIRONMENT_ERROR, "Seed response contained no observation")

        interaction_deadline = asyncio.get_running_loop().time() + (
            self.config.rollout_timeout_seconds - self.config.finalization_reserve_seconds
        )
        env_id = seed.env_id
        seed_obs = list(seed.obs)
        incoming = self._incoming_input(body.responses_create_params)
        complete_history: list[Any] = [*incoming, *seed_obs]
        rollout_output: list[Any] = []
        model_cookies = initial_cookies
        last_model_response: NeMoGymResponse | None = None
        usage: NeMoGymResponseUsage | None = None
        reason = TerminationReason.UNKNOWN
        detail: str | None = None
        environment_steps = 0
        model_calls = 0
        primary_error: WebArenaRolloutError | None = None
        close_error: str | None = None
        rollout_phase_completed = False

        # Context management: the shared ContextManagedResponsesClient owns the
        # model-facing view of the history and the guard measurement; the trainer
        # rebuilds physical rows from token capture. The legacy compact_model_input
        # pruning is bypassed entirely (see WebAgentConfig validator).
        context_client: ContextManagedResponsesClient | None = None
        context_client_failed = False
        contract_body = body.responses_create_params.model_copy(
            update={"tools": list(seed.tools), "parallel_tool_calls": False}
        )
        offered_tools = frozenset(name for name in map(_tool_name, seed.tools) if name)
        if self.config.context_history.enabled:
            initial_context_items, session_seed_items = self._compaction_seed_split(
                incoming,
                seed_obs,
                authority=True,
            )
            context_client = ContextManagedResponsesClient(
                server_client=self.server_client,
                model_server=self.config.model_server,
                logical_rollout_id=owner_id,
                config=self.config.context_history,
                initial_request=contract_body.model_copy(
                    update={"input": [_item_dict(item) for item in initial_context_items]}
                ),
                seed_observations=[_item_dict(item) for item in session_seed_items],
                cookies=model_cookies,
            )

        try:
            try:
                async with asyncio.timeout_at(interaction_deadline):
                    # One model call per step. Anything other than a single valid tool call,
                    # and any tool error, ends the rollout: no feedback is injected and
                    # nothing is resampled.
                    while environment_steps < self.config.max_steps:
                        model_calls += 1
                        if context_client is not None:
                            try:
                                model_response = await context_client.create()
                            except ContextGuardRejected:
                                context_client_failed = True
                                reason = TerminationReason.CONTEXT_LIMIT
                                break
                            except Exception as exc:
                                # The client closes itself on any failed call: a lost
                                # acknowledgement may already have captured an action,
                                # so this logical rollout must not continue.
                                context_client_failed = True
                                if _is_context_limit_error(exc):
                                    reason = TerminationReason.CONTEXT_LIMIT
                                else:
                                    reason = TerminationReason.MODEL_ERROR
                                    primary_error = WebArenaRolloutError(reason, f"Model call failed: {exc!r}")
                                break
                            model_cookies = dict(context_client.cookies)
                        else:
                            request_input = compact_model_input(
                                complete_history,
                                self.config.max_recent_screenshots,
                            )
                            model_body = _validated_request_body(contract_body, request_input=request_input)
                            try:
                                model_response, model_cookies = await self._call_model(
                                    model_body,
                                    model_cookies,
                                )
                            except Exception as exc:
                                if _is_context_limit_error(exc):
                                    reason = TerminationReason.CONTEXT_LIMIT
                                else:
                                    reason = TerminationReason.MODEL_ERROR
                                    primary_error = WebArenaRolloutError(reason, f"Model call failed: {exc!r}")
                                break

                        last_model_response = model_response
                        usage = _aggregate_usage(usage, model_response.usage)
                        model_output = list(model_response.output)
                        complete_history.extend(model_output)
                        rollout_output.extend(model_output)

                        violation = self._protocol_violation(model_response, offered_tools)
                        if violation is not None:
                            reason, detail = violation
                            break

                        step_payload = {
                            "env_id": env_id,
                            "action_request_id": str(uuid4()),
                            "action": [self._function_calls(model_response)[0].model_dump(mode="json")],
                        }
                        try:
                            step_data, resource_cookies = await self._post_json(
                                self.config.resources_server.name,
                                "/step",
                                step_payload,
                                resource_cookies,
                                self.config.resource_request_timeout_seconds,
                            )
                            step_response = WebArenaStepResponse.model_validate(step_data)
                        except Exception as exc:
                            reason = TerminationReason.ENVIRONMENT_ERROR
                            primary_error = WebArenaRolloutError(reason, f"Environment step failed: {exc!r}")
                            break
                        if step_response.tool_error is not None:
                            reason = TerminationReason.TOOL_EXECUTION_ERROR
                            detail = step_response.tool_error
                            break

                        observations = list(step_response.obs)
                        complete_history.extend(observations)
                        rollout_output.extend(observations)
                        if context_client is not None:
                            context_client.append_observation([_item_dict(item) for item in observations])
                        environment_steps += 1
                        if step_response.done:
                            reason = TerminationReason.ENV_DONE
                            break
                    else:
                        reason = TerminationReason.MAX_STEPS
            except TimeoutError:
                reason = TerminationReason.ROLLOUT_TIMEOUT
                primary_error = WebArenaRolloutError(reason, "WebArena rollout deadline exceeded")
            except asyncio.CancelledError:
                reason = TerminationReason.CANCELLED
                raise
            rollout_phase_completed = True
        finally:
            if close_on_success or not rollout_phase_completed:
                close_error = await asyncio.shield(self._close(env_id, resource_cookies, str(uuid4())))

        if last_model_response is None:
            if not close_on_success:
                close_error = await asyncio.shield(self._close(env_id, resource_cookies, str(uuid4())))
            if primary_error is not None:
                raise primary_error
            raise WebArenaRolloutError(reason, f"WebArena rollout ended before a valid model response: {reason.value}")

        # Any failed or cancelled create() closes the client because a lost
        # acknowledgement may have captured an action; finish() is then unavailable.
        # The accepted actions are still returned, and the failure reason masks the
        # rollout.
        if context_client is not None and not context_client_failed and primary_error is None:
            try:
                context_client.finish(last_model_response)
            except Exception as exc:
                if not close_on_success:
                    await asyncio.shield(self._close(env_id, resource_cookies, str(uuid4())))
                raise (
                    primary_error or WebArenaRolloutError(reason, f"Context client finish failed: {exc!r}")
                ) from exc

        termination = WebArenaTerminationMetadata(
            reason=reason,
            detail=detail,
            environment_steps=environment_steps,
            model_calls=model_calls,
            close_error=close_error,
        )
        metadata = dict(last_model_response.metadata or {})
        metadata["webarena_termination"] = termination.model_dump_json()
        web_extras = {
            "env_id": env_id,
            "group_id": str(getattr(body, "task_idx", None) or body.task_payload.task_id),
            "contains_transitions": self.config.return_transitions,
            "usage": usage.model_dump(mode="json") if usage is not None else None,
            "metadata": metadata,
        }
        try:
            response_data = last_model_response.model_dump(mode="json")
            response_data.update(web_extras)
            # With context management, the output is the ordinary semantic conversation
            # (every observation occurrence) for the verifier and the trainer's
            # returned-history call selection, not the compacted model views.
            response_data.update(
                {
                    "seed_obs": [_item_dict(item) for item in seed_obs],
                    "output": (
                        context_client.output_items
                        if context_client is not None
                        else [_item_dict(item) for item in rollout_output]
                    ),
                }
            )
            result = WebArenaNeMoGymResponse.model_validate(response_data)
        except BaseException:
            if not close_on_success:
                await asyncio.shield(self._close(env_id, resource_cookies, str(uuid4())))
            raise
        return result, resource_cookies

    async def responses(
        self,
        request: Request,
        response: Response,
        body: WebArenaAgentRunRequest = Body(),
    ) -> WebArenaNeMoGymResponse:
        result, cookies = await self._rollout(body, request.cookies, logical_rollout_id=_path_rollout_id(request))
        for key, value in (cookies or {}).items():
            response.set_cookie(key, getattr(value, "value", value))
        return result

    @staticmethod
    def _apply_termination_outcome(merged: dict[str, Any], termination: WebArenaTerminationMetadata) -> None:
        """Zero the reward for policy mistakes and mask rollouts that failed for other reasons.

        A truncated turn (max output tokens) scores ``-terminate_properly_reward``
        when the resources server reports a positive value (CUA-Gym), else 0.
        """
        merged["webarena_termination_reason"] = termination.reason.value
        if termination.reason != TerminationReason.ENV_DONE:
            merged["failure_reason"] = (
                f"{termination.reason.value}: {termination.detail}" if termination.detail else termination.reason.value
            )
        if termination.reason in _ZERO_REWARD_REASONS:
            merged["reward"] = 0.0
        if termination.reason == TerminationReason.TRUNCATED:
            penalty = (merged.get("evaluator_details") or {}).get("terminate_properly_reward")
            if isinstance(penalty, (int, float)) and not isinstance(penalty, bool) and penalty > 0:
                merged["reward"] = -float(penalty)
        if termination.reason in _MASKED_REASONS:
            merged["instance_config"] = {"mask_sample": True}

    async def run(
        self,
        request: Request,
        body: WebArenaAgentRunRequest = Body(),
    ) -> WebArenaAgentVerifyResponse:
        rollout_response: BaseModel | None = None
        resource_cookies = request.cookies
        finalized = False
        finalize_request_id = str(uuid4())
        try:
            try:
                rollout_response, resource_cookies = await self._rollout(
                    body,
                    request.cookies,
                    close_on_success=False,
                    logical_rollout_id=_path_rollout_id(request) or self.rollout_id_from_run(body),
                )
            except WebArenaRolloutError as error:
                # A terminal rollout failure (model call lost, environment step failed,
                # deadline) is a property of this attempt, not of the server: report it
                # as 422 so trainers classify it as a per-prompt data failure. Letting it
                # escape would become a generic 500, which NeMo-RL's SingleController
                # treats as a transport failure and aborts the whole run on.
                raise HTTPException(status_code=422, detail=repr(error)) from error
            finalize_result = await self._finalize(
                rollout_response.env_id,
                resource_cookies,
                finalize_request_id,
            )
            finalized = True
            trainer_response: dict[str, Any] = rollout_response.model_dump(mode="json")
            if self.config.context_history.enabled:
                # Bound the trainer-facing payload: the full conversation repeats every
                # screenshot, and training tokens and media are framework-captured.
                trainer_response["output"] = _without_raw_images(rollout_response.output)
                trainer_response["seed_obs"] = _without_raw_images(rollout_response.seed_obs)
            merged = body.model_dump(mode="json")
            merged.update(
                {
                    "verify_request_id": finalize_request_id,
                    "response": trainer_response,
                    "reward": finalize_result.reward,
                    "evaluator_details": finalize_result.evaluator_details,
                    "termination_status": finalize_result.termination_status,
                    "answer": finalize_result.answer,
                }
            )
            termination_json = (rollout_response.metadata or {}).get("webarena_termination")
            if termination_json is not None:
                self._apply_termination_outcome(
                    merged, WebArenaTerminationMetadata.model_validate_json(termination_json)
                )
            return WebArenaAgentVerifyResponse.model_validate(merged)
        finally:
            if rollout_response is not None and not finalized:
                await asyncio.shield(
                    self._close(
                        rollout_response.env_id,
                        resource_cookies,
                        str(uuid4()),
                    )
                )


if __name__ == "__main__":
    WebAgent.run_webserver()
