# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One rollout loop shared by WebArena, VisualWebArena, and WebVoyager."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from urllib.parse import urlparse

from aiohttp import ClientResponseError
from fastapi import Body, Request, Response
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    accumulate_response_usage,
)
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY, NG_TERMINAL_KEY
from nemo_gym.server_utils import get_response_json, raise_for_status
from nemo_gym.web.actions import ActionParseError, parse_model_action, parse_native_tool_calls
from nemo_gym.web.api_models import (
    WebCloseResponse,
    WebEvaluateResponse,
    WebSeedSessionResponse,
    WebStepResponse,
)
from nemo_gym.web.judge_evidence import compact_webvoyager_judge_evidence
from nemo_gym.web.models import (
    BROWSER_TARGET_CLOSED_STATUS,
    CAPTCHA_BUDGET_EXHAUSTED_STATUS,
    WebActionProfile,
    WebArtifactRef,
    WebBenchmark,
    WebObservation,
    WebTask,
    WebVerifierResult,
)
from responses_api_agents.web_agent.render import parse_error_message, render_observation


LOG = logging.getLogger("nemo_gym.responses_api_agents.web_agent")

_MODEL_LOG_CONTEXT_HEADERS = {
    "adapter": "x-nemo-gym-log-adapter",
    "task_id": "x-nemo-gym-log-task-id",
    "domain": "x-nemo-gym-log-domain",
    "step": "x-nemo-gym-log-step",
    "parse_attempt": "x-nemo-gym-log-parse-attempt",
}


def _model_log_context_headers(task: WebTask, step: int, parse_attempt: int) -> dict[str, str]:
    """Attach non-secret rollout identity to transport logs without changing the model body."""

    values = {
        "adapter": "web_agent",
        "task_id": task.task_id,
        "domain": task.benchmark.value,
        "step": step,
        "parse_attempt": parse_attempt,
    }
    return {
        _MODEL_LOG_CONTEXT_HEADERS[field]: str(value).replace("\r", "").replace("\n", "")[:1024]
        for field, value in values.items()
    }


class WebAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    # Browser execution and verification are the same resource for
    # WebArena/VisualWebArena. WebVoyager supplies a dedicated browser here and
    # keeps ``resources_server`` as the canonical verifier used by Gym.
    environment_server: Optional[ResourcesServerRef] = None
    model_server: ModelServerRef
    max_steps: int = Field(default=15, ge=1, le=200)
    max_parse_retries: int = Field(default=2, ge=0, le=10)
    native_action_recovery: Literal["strict", "decode_string", "repair_single_closing_bracket"] = "strict"
    native_tool_alias_recovery: Literal["strict", "webvoyager_v3"] = "strict"
    native_max_computer_actions: int = Field(default=20, ge=1, le=100)
    native_parse_retry_feedback: bool = False
    native_parse_retry_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    native_parse_retry_delay_secs: float = Field(default=0.0, ge=0.0, le=60.0)
    repeated_action_warning_threshold: int = Field(default=0, ge=0, le=20)
    repeated_action_window: int = Field(default=5, ge=1, le=50)
    max_consecutive_execution_failures: int = Field(default=3, ge=1, le=20)
    # The reference native runner retries one policy call up to 20 times before
    # giving up, so the ceiling has to admit that budget.
    model_turn_max_retries: int = Field(default=0, ge=0, le=32)
    model_retry_delay_secs: float = Field(default=1.0, ge=0.0, le=60.0)
    max_image_history: int = Field(default=3, ge=1, le=20)
    judge_max_screenshots: int = Field(default=3, ge=1, le=200)
    # VisualWebArena JSONL stores relative reference-image paths. The agent
    # resolves them only below this explicitly mounted, read-only directory.
    task_image_root: str | None = None
    max_task_image_bytes: int = Field(default=25 * 1024 * 1024, ge=1, le=100 * 1024 * 1024)
    visual_observation_text: Literal["full_axtree", "som_only", "none"] = "full_axtree"
    action_prompt_profile: Literal["standard", "code_block"] = "standard"
    redact_old_visual_observations: bool = False
    resources_request_timeout_secs: float = Field(default=180.0, gt=0.0)
    seed_request_timeout_secs: float = Field(default=1800.0, gt=0.0)
    seed_retry_initial_delay_secs: float = Field(default=2.0, ge=0.0)
    seed_retry_max_delay_secs: float = Field(default=30.0, ge=0.0)
    model_request_timeout_secs: float = Field(default=600.0, gt=0.0)
    judge_request_timeout_secs: float = Field(default=300.0, gt=0.0)
    close_request_timeout_secs: float = Field(default=30.0, gt=0.0)
    run_timeout_secs: float = Field(default=1800.0, gt=0.0)


class WebAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    web_task: WebTask | None = None
    verifier_metadata: dict[str, Any] | None = None


class WebAgentRunResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    benchmark: str
    task_id: str
    raw_score: float = 0.0
    task_success: bool = False
    mask_sample: bool = False
    failure_kind: str | None = None
    terminated: bool = False
    truncated: bool = False
    environment_steps: int = 0
    model_turns: int = 0
    execution_failures: int = 0
    verifier_result: WebVerifierResult | None = None
    artifact_session_id: str | None = None
    recording_artifacts: list[WebArtifactRef] = Field(default_factory=list)


@dataclass
class _RunArtifacts:
    """Mutable artifact context retained when a bounded rollout fails."""

    session_id: str | None = None
    recordings: list[WebArtifactRef] = field(default_factory=list)


def _extract_output_text(response: NeMoGymResponse) -> str:
    parts: list[str] = []
    for item in response.output:
        if getattr(item, "type", None) != "message":
            continue
        content = getattr(item, "content", None)
        if isinstance(content, str):
            parts.append(content)
            continue
        for block in content or []:
            if getattr(block, "type", None) == "output_text":
                parts.append(str(getattr(block, "text", "")))
    return "\n".join(part for part in parts if part).strip()


def _parse_response_action(
    response: NeMoGymResponse,
    profile: WebActionProfile,
    *,
    native_action_recovery: Literal["strict", "decode_string", "repair_single_closing_bracket"] = "strict",
    native_tool_alias_recovery: Literal["strict", "webvoyager_v3"] = "strict",
    native_max_computer_actions: int = 20,
):
    if profile == WebActionProfile.NATIVE_TOOLCALL:
        return parse_native_tool_calls(
            response.output,
            recovery=native_action_recovery,
            alias_recovery=native_tool_alias_recovery,
            max_computer_actions=native_max_computer_actions,
        )
    return parse_model_action(_extract_output_text(response), profile)


def _incomplete_model_reason(response: NeMoGymResponse) -> str | None:
    """Return the normalized Responses API incomplete reason, if present."""

    if getattr(response, "status", None) != "incomplete":
        return None
    details = getattr(response, "incomplete_details", None)
    if isinstance(details, dict):
        reason = details.get("reason")
    else:
        reason = getattr(details, "reason", None)
    return str(reason) if reason else "unknown"


def _native_parse_retry_messages(response: NeMoGymResponse, error: ActionParseError) -> list[Any]:
    """Return parser feedback without copying the current screenshot."""

    invalid_items: list[dict[str, Any]] = []
    for item in response.output:
        item_type = getattr(item, "type", None)
        if item_type == "function_call":
            invalid_items.append(
                {
                    "type": "function_call",
                    "name": getattr(item, "name", ""),
                    "arguments": getattr(item, "arguments", ""),
                }
            )
        elif item_type == "message":
            invalid_items.append({"type": "message", "content": _extract_output_text(response)})
    invalid_text = json.dumps(invalid_items, ensure_ascii=False, separators=(",", ":"))[-8000:]
    return [
        NeMoGymEasyInputMessage(role="assistant", content=invalid_text or "<empty model response>"),
        NeMoGymEasyInputMessage(
            role="user",
            content=(
                f"The previous browser tool response was invalid: {str(error)[-2000:]}. "
                "Return a corrected call using only the declared tools. For browser clicks, typing, "
                "keys, scrolling, dragging, or waiting, call `computer` and make "
                "`arguments.actions` a JSON array of action objects, not a quoted string."
                " Use `left_click`, never `click`, as a computer action name."
            ),
        ),
    ]


def _http_error_payload(exc: Exception) -> dict[str, Any]:
    response_content = getattr(exc, "response_content", None)
    if isinstance(response_content, bytes):
        response_content = response_content.decode("utf-8", errors="replace")
    if not isinstance(response_content, str) or not response_content.strip():
        return {}
    try:
        payload = json.loads(response_content)
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_model_context_overflow(exc: Exception) -> bool:
    """Recognize deterministic vLLM context-limit failures, including wrapped 5xx responses."""

    if not isinstance(exc, ClientResponseError):
        return False
    response_content = getattr(exc, "response_content", None)
    if isinstance(response_content, bytes):
        response_content = response_content.decode("utf-8", errors="replace")
    haystack = " ".join((str(exc), str(response_content or ""))).lower()
    return ("decoder prompt" in haystack and "longer than the maximum model length" in haystack) or (
        "maximum context length" in haystack and ("tokens" in haystack or "token" in haystack)
    )


def _failure_route(exc: Exception) -> tuple[str, bool, str, dict[str, Any]]:
    """Map a bounded exception to sidecar retry and terminal semantics."""

    metadata: dict[str, Any] = {}
    failure_class = "retryable_infrastructure"
    terminal = False
    failure_kind = f"infrastructure_error:{type(exc).__name__}"
    if not isinstance(exc, ClientResponseError):
        return failure_class, terminal, failure_kind, metadata

    metadata["http_status"] = exc.status
    payload = _http_error_payload(exc)
    error_kind = payload.get("error_kind")
    retryable = payload.get("retryable")
    if _is_model_context_overflow(exc):
        metadata["error_kind"] = "model_context_overflow"
        failure_kind = "model_context_overflow"
    if isinstance(error_kind, str):
        metadata["error_kind"] = error_kind
    else:
        error_kind = None

    if retryable is False:
        terminal = True
        if error_kind in {"benchmark_precondition", "invalid_task"}:
            failure_class = "benchmark_precondition"
        else:
            failure_class = "configuration_error"
        failure_kind = error_kind or f"{failure_class}:http_{exc.status}"
    elif retryable is not True and exc.status in {400, 401, 403, 422}:
        # Backward-compatible fallback for an older resource server that does
        # not yet emit the structured retryability envelope.
        terminal = True
        failure_class = "configuration_error"
        failure_kind = f"configuration_error:http_{exc.status}"

    return failure_class, terminal, failure_kind, metadata


def _resolve_task(body: WebAgentRunRequest) -> WebTask:
    if body.web_task is not None:
        return body.web_task
    metadata = body.verifier_metadata or {}
    candidate = metadata.get("web_task") or metadata.get("task")
    if candidate is None and "benchmark" in metadata and "task_id" in metadata:
        candidate = metadata
    if candidate is None:
        extra = body.model_extra or {}
        candidate = extra.get("web_task") or extra.get("task")
    if candidate is None:
        raise ValueError("a normalized web_task is required in the row or verifier_metadata")
    return WebTask.model_validate(candidate)


def _merge_usage(total, response: NeMoGymResponse):
    return accumulate_response_usage(total, response.usage)


def _url_origin(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.hostname:
        return "unknown"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme or 'unknown'}://{parsed.hostname}{port}"


def _input_image_count(items: list[Any]) -> int:
    count = 0
    for item in items:
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            continue
        count += sum(
            1
            for block in content
            if (isinstance(block, dict) and block.get("type") == "input_image")
            or getattr(block, "type", None) == "input_image"
        )
    return count


def _is_input_image_block(block: Any) -> bool:
    return (isinstance(block, dict) and block.get("type") == "input_image") or (
        getattr(block, "type", None) == "input_image"
    )


def _block_text(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("text", ""))
    return str(getattr(block, "text", ""))


def _is_task_input_image_block(content: list[Any], index: int) -> bool:
    if index < 1 or not _is_input_image_block(content[index]):
        return False
    label = _block_text(content[index - 1])
    return label.startswith("Task image ") and " of " in label and label.endswith(":")


def _is_browser_image_block(content: list[Any], index: int) -> bool:
    return _is_input_image_block(content[index]) and not _is_task_input_image_block(content, index)


def _action_call_names(action: Any) -> str:
    calls = getattr(action, "arguments", {}).get("calls", [])
    names = [str(call.get("name", "unknown")) for call in calls if isinstance(call, dict)]
    return ",".join(names) or getattr(action, "name", "unknown")


def _native_recovery_modes(action: Any) -> str:
    metadata = getattr(action, "metadata", {})
    records = metadata.get("native_parse", {}).get("calls", []) if isinstance(metadata, dict) else []
    modes: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        modes.append(str(record.get("recovery_mode", "strict")))
        modes.extend(str(mode) for mode in record.get("alias_recovery_modes", []))
    return ",".join(modes) or "strict"


def _repeatable_action_signature(action: Any) -> str | None:
    if getattr(action, "terminal", False):
        return None
    calls = getattr(action, "arguments", {}).get("calls", [])
    if not calls:
        return None
    if all(
        isinstance(call, dict)
        and call.get("name") == "computer"
        and all(
            isinstance(item, dict) and item.get("action") == "wait"
            for item in (call.get("arguments") or {}).get("actions", [])
        )
        for call in calls
    ):
        return None
    return json.dumps(calls, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _redact_old_images(
    items: list[Any],
    max_image_history: int,
    *,
    redact_observation_text: bool = False,
    append_redaction_notice: bool = True,
) -> list[Any]:
    """Keep only the newest N image-bearing messages in the next model call."""

    copied = [item.model_copy(deep=True) if hasattr(item, "model_copy") else item for item in items]
    image_message_indices: list[int] = []
    for index, item in enumerate(copied):
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            continue
        if any(_is_browser_image_block(content, block_index) for block_index in range(len(content))):
            image_message_indices.append(index)
    for index in image_message_indices[:-max_image_history]:
        item = copied[index]
        content = getattr(item, "content", None)
        has_task_images = any(_is_task_input_image_block(content, block_index) for block_index in range(len(content)))
        if redact_observation_text and not has_task_images:
            item.content = [
                {
                    "type": "input_text",
                    "text": "[Earlier screenshot and page text omitted from context.]",
                }
            ]
            continue
        retained = [
            block for block_index, block in enumerate(content) if not _is_browser_image_block(content, block_index)
        ]
        if append_redaction_notice:
            retained.append({"type": "input_text", "text": "[Earlier screenshot omitted from context.]"})
        item.content = retained
    return copied


class WebAgent(SimpleResponsesAPIAgent):
    config: WebAgentConfig

    @property
    def environment_server_name(self) -> str:
        """Return the browser resource, defaulting to the verifier resource."""

        if self.config.environment_server is not None:
            return self.config.environment_server.name
        return self.config.resources_server.name

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        started = time.monotonic()
        LOG.info(
            "event=web_agent_responses_start model_server=%s input_items=%d tools=%d",
            self.config.model_server.name,
            len(body.input) if isinstance(body.input, list) else 1,
            len(body.tools or []),
        )
        try:
            model_response, model_payload = await self._post_json(
                server_name=self.config.model_server.name,
                url_path=self.url_path_for_request("/v1/responses", request),
                json=body,
                cookies=request.cookies,
                timeout_secs=self.config.model_request_timeout_secs,
            )
        except Exception:
            LOG.exception(
                "event=web_agent_responses_failed model_server=%s elapsed_seconds=%.3f",
                self.config.model_server.name,
                time.monotonic() - started,
            )
            raise
        result = NeMoGymResponse.model_validate(model_payload)
        for key, value in model_response.cookies.items():
            response.set_cookie(key, value)
        LOG.info(
            "event=web_agent_responses_complete model_server=%s output_items=%d elapsed_seconds=%.3f",
            self.config.model_server.name,
            len(result.output),
            time.monotonic() - started,
        )
        return result

    async def run(self, request: Request, body: WebAgentRunRequest) -> WebAgentRunResponse:
        task = _resolve_task(body)
        artifacts = _RunArtifacts()
        started = time.monotonic()
        LOG.info(
            "event=web_rollout_start benchmark=%s task=%s runtime=%s action_profile=%s "
            "max_steps=%d max_image_history=%d",
            task.benchmark.value,
            task.task_id,
            task.runtime_profile.value,
            task.action_profile.value,
            self.config.max_steps,
            self.config.max_image_history,
        )
        try:
            result = await asyncio.wait_for(
                self._run_once(request, body, task, artifacts),
                timeout=self.config.run_timeout_secs,
            )
            LOG.info(
                "event=web_rollout_complete benchmark=%s task=%s success=%s valid_sample=%s reward=%s "
                "steps=%d model_turns=%d execution_failures=%d terminated=%s truncated=%s "
                "failure_kind=%s elapsed_seconds=%.3f",
                task.benchmark.value,
                task.task_id,
                result.task_success,
                not result.mask_sample,
                result.reward,
                result.environment_steps,
                result.model_turns,
                result.execution_failures,
                result.terminated,
                result.truncated,
                result.failure_kind or "none",
                time.monotonic() - started,
            )
            return result
        except Exception as exc:  # noqa: BLE001 - one stalled browser must not abort the shard.
            LOG.exception(
                "event=web_rollout_failed benchmark=%s task=%s elapsed_seconds=%.3f",
                task.benchmark.value,
                task.task_id,
                time.monotonic() - started,
            )
            return self._failure_response(body, task, exc, artifacts)

    async def _run_once(
        self,
        request: Request,
        body: WebAgentRunRequest,
        task: WebTask,
        artifacts: _RunArtifacts,
    ) -> WebAgentRunResponse:
        env_cookies = request.cookies
        model_cookies = None
        seeded = False
        last_model_response: NeMoGymResponse | None = None
        usage = None
        trajectory: list[Any] = []
        screenshot_history: deque[str] = deque(maxlen=self.config.judge_max_screenshots)
        url_history: deque[str] = deque(maxlen=self.config.judge_max_screenshots)
        final_answer: str | None = None
        terminated = False
        truncated = False
        environment_steps = 0
        model_turns = 0
        execution_failures = 0
        consecutive_execution_failures = 0
        verifier_result: WebVerifierResult | None = None
        environment_failure_kind: str | None = None
        recent_action_signatures: deque[str] = deque(maxlen=self.config.repeated_action_window)

        base_body = body.responses_create_params.model_copy(deep=True)
        if isinstance(base_body.input, str):
            base_body.input = [NeMoGymEasyInputMessage(role="user", content=base_body.input)]

        try:
            seed_started = time.monotonic()
            LOG.info("event=web_seed_start benchmark=%s task=%s", task.benchmark.value, task.task_id)
            seed_response, seed_payload = await self._seed_session(
                task=task,
                cookies=env_cookies,
            )
            seed_data = WebSeedSessionResponse.model_validate(seed_payload)
            artifacts.session_id = seed_data.session_id
            env_cookies = seed_response.cookies
            seeded = True
            observation = seed_data.observation
            LOG.info(
                "event=web_seed_complete benchmark=%s task=%s session=%s origin=%s screenshot=%s elapsed_seconds=%.3f",
                task.benchmark.value,
                task.task_id,
                seed_data.session_id,
                _url_origin(observation.url),
                bool(observation.screenshot),
                time.monotonic() - seed_started,
            )
            self._remember_evidence(observation, screenshot_history, url_history)
            base_body.input = list(base_body.input) + [
                render_observation(
                    observation,
                    task,
                    step_index=0,
                    visual_observation_text=self.config.visual_observation_text,
                    action_prompt_profile=self.config.action_prompt_profile,
                    task_image_root=self.config.task_image_root,
                    max_task_image_bytes=self.config.max_task_image_bytes,
                )
            ]

            rollout_finished = False
            for step_index in range(self.config.max_steps):
                action = None
                parse_feedback: list[Any] = []
                for parse_attempt in range(self.config.max_parse_retries + 1):
                    model_input = _redact_old_images(
                        list(base_body.input) + trajectory + parse_feedback,
                        self.config.max_image_history,
                        redact_observation_text=self.config.redact_old_visual_observations,
                        append_redaction_notice=task.action_profile != WebActionProfile.NATIVE_TOOLCALL,
                    )
                    model_updates: dict[str, Any] = {"input": model_input}
                    if parse_attempt > 0 and self.config.native_parse_retry_temperature is not None:
                        model_updates["temperature"] = self.config.native_parse_retry_temperature
                    model_body = base_body.model_copy(update=model_updates)
                    LOG.info(
                        "event=web_model_turn_start benchmark=%s task=%s step=%d parse_attempt=%d "
                        "input_items=%d input_images=%d retry_feedback=%s temperature=%s",
                        task.benchmark.value,
                        task.task_id,
                        step_index,
                        parse_attempt,
                        len(model_input),
                        _input_image_count(model_input),
                        bool(parse_feedback),
                        getattr(model_body, "temperature", None),
                    )
                    model_error: Exception | None = None
                    for model_attempt in range(self.config.model_turn_max_retries + 1):
                        model_started = time.monotonic()
                        try:
                            raw_model_response, model_payload = await self._post_json(
                                server_name=self.config.model_server.name,
                                url_path=self.url_path_for_run("/v1/responses", body),
                                json=model_body,
                                cookies=model_cookies,
                                headers=_model_log_context_headers(task, step_index, parse_attempt),
                                timeout_secs=self.config.model_request_timeout_secs,
                            )
                            model_error = None
                            LOG.info(
                                "event=web_model_request_complete benchmark=%s task=%s step=%d "
                                "parse_attempt=%d model_attempt=%d elapsed_seconds=%.3f",
                                task.benchmark.value,
                                task.task_id,
                                step_index,
                                parse_attempt,
                                model_attempt,
                                time.monotonic() - model_started,
                            )
                            break
                        except Exception as exc:  # Bounded native API parity retry.
                            model_error = exc
                            retry_model_request = (
                                model_attempt < self.config.model_turn_max_retries
                                and not _is_model_context_overflow(exc)
                                and not (isinstance(exc, ClientResponseError) and _failure_route(exc)[1])
                            )
                            LOG.warning(
                                "event=web_model_request_failed benchmark=%s task=%s step=%d parse_attempt=%d "
                                "model_attempt=%d error_type=%s elapsed_seconds=%.3f retry=%s "
                                "failure_kind=%s",
                                task.benchmark.value,
                                task.task_id,
                                step_index,
                                parse_attempt,
                                model_attempt,
                                type(exc).__name__,
                                time.monotonic() - model_started,
                                retry_model_request,
                                _failure_route(exc)[2],
                            )
                            if not retry_model_request:
                                raise
                            await asyncio.sleep(self.config.model_retry_delay_secs)
                    if model_error is not None:
                        raise model_error
                    model_response = NeMoGymResponse.model_validate(model_payload)
                    model_cookies = raw_model_response.cookies
                    last_model_response = model_response
                    model_turns += 1
                    usage = _merge_usage(usage, model_response)
                    LOG.info(
                        "event=web_model_turn_complete benchmark=%s task=%s step=%d parse_attempt=%d "
                        "output_items=%d input_tokens=%s output_tokens=%s",
                        task.benchmark.value,
                        task.task_id,
                        step_index,
                        parse_attempt,
                        len(model_response.output),
                        getattr(model_response.usage, "input_tokens", "unknown"),
                        getattr(model_response.usage, "output_tokens", "unknown"),
                    )
                    incomplete_reason = _incomplete_model_reason(model_response)
                    if incomplete_reason == "max_output_tokens":
                        # VLLMModel converts both a generated length stop and a
                        # context-budget 400 into a Responses API incomplete
                        # result. The native reference runner treats either as
                        # a valid truncated policy outcome and does not retry
                        # the action parser against the same empty response.
                        truncated = True
                        LOG.warning(
                            "event=web_model_output_truncated benchmark=%s task=%s step=%d parse_attempt=%d reason=%s",
                            task.benchmark.value,
                            task.task_id,
                            step_index,
                            parse_attempt,
                            incomplete_reason,
                        )
                        break
                    if task.action_profile != WebActionProfile.NATIVE_TOOLCALL:
                        trajectory.extend(model_response.output)
                    try:
                        action = _parse_response_action(
                            model_response,
                            task.action_profile,
                            native_action_recovery=self.config.native_action_recovery,
                            native_tool_alias_recovery=self.config.native_tool_alias_recovery,
                            native_max_computer_actions=self.config.native_max_computer_actions,
                        )
                        if task.action_profile == WebActionProfile.NATIVE_TOOLCALL:
                            # The reference runner adds only a successfully
                            # parsed assistant tool-call turn to history.
                            trajectory.extend(model_response.output)
                        LOG.info(
                            "event=web_action_parsed benchmark=%s task=%s step=%d parse_attempt=%d "
                            "action=%s calls=%s terminal=%s recovery_modes=%s",
                            task.benchmark.value,
                            task.task_id,
                            step_index,
                            parse_attempt,
                            action.name,
                            _action_call_names(action),
                            action.terminal,
                            _native_recovery_modes(action),
                        )
                        break
                    except ActionParseError as exc:
                        LOG.warning(
                            "event=web_action_parse_failed benchmark=%s task=%s step=%d parse_attempt=%d "
                            "error_type=%s error=%r terminal_attempt=%s",
                            task.benchmark.value,
                            task.task_id,
                            step_index,
                            parse_attempt,
                            type(exc).__name__,
                            str(exc)[:500],
                            parse_attempt >= self.config.max_parse_retries,
                        )
                        if parse_attempt >= self.config.max_parse_retries:
                            if task.action_profile != WebActionProfile.NATIVE_TOOLCALL:
                                trajectory.append(
                                    NeMoGymEasyInputMessage(
                                        role="user",
                                        content=f"Action parsing failed permanently: {exc}",
                                    )
                                )
                            break
                        if task.action_profile == WebActionProfile.NATIVE_TOOLCALL:
                            if self.config.native_parse_retry_feedback:
                                parse_feedback = _native_parse_retry_messages(model_response, exc)
                            await asyncio.sleep(self.config.native_parse_retry_delay_secs)
                        else:
                            trajectory.append(
                                parse_error_message(
                                    exc,
                                    action_prompt_profile=self.config.action_prompt_profile,
                                    action_profile=task.action_profile,
                                )
                            )

                if truncated:
                    break

                if action is None:
                    LOG.warning(
                        "event=web_rollout_no_action benchmark=%s task=%s step=%d model_turns=%d",
                        task.benchmark.value,
                        task.task_id,
                        step_index,
                        model_turns,
                    )
                    break

                environment_started = time.monotonic()
                LOG.info(
                    "event=web_environment_step_start benchmark=%s task=%s step=%d action=%s calls=%s",
                    task.benchmark.value,
                    task.task_id,
                    step_index,
                    action.name,
                    _action_call_names(action),
                )
                step_response, step_payload = await self._post_json(
                    server_name=self.environment_server_name,
                    url_path="/step",
                    json={
                        "operation_id": f"step-{step_index}",
                        "action": action.model_dump(mode="json"),
                    },
                    cookies=env_cookies,
                    timeout_secs=self.config.resources_request_timeout_secs,
                )
                step_data = WebStepResponse.model_validate(step_payload)
                env_cookies = step_response.cookies
                environment_steps += 1
                if not step_data.execution_ok:
                    execution_failures += 1
                    consecutive_execution_failures += 1
                else:
                    consecutive_execution_failures = 0
                observation = step_data.observation
                terminated = step_data.terminated
                truncated = step_data.truncated
                LOG.info(
                    "event=web_environment_step_complete benchmark=%s task=%s step=%d execution_ok=%s "
                    "terminated=%s truncated=%s origin=%s elapsed_seconds=%.3f",
                    task.benchmark.value,
                    task.task_id,
                    step_index,
                    step_data.execution_ok,
                    terminated,
                    truncated,
                    _url_origin(observation.url),
                    time.monotonic() - environment_started,
                )
                native_status = step_data.info.get("native_status")
                # Native terminate returns the previous observation unchanged;
                # unlike a non-terminal action, it does not capture a new
                # screenshot. Keep valid observations from actions that cause
                # the environment to terminate, but do not duplicate evidence
                # for an explicit terminate or append a closed-target result.
                explicit_native_terminate = task.action_profile == WebActionProfile.NATIVE_TOOLCALL and action.terminal
                if native_status != BROWSER_TARGET_CLOSED_STATUS and not explicit_native_terminate:
                    self._remember_evidence(observation, screenshot_history, url_history)
                elif explicit_native_terminate:
                    LOG.info(
                        "event=web_terminal_evidence_reused benchmark=%s task=%s step=%d screenshots=%d",
                        task.benchmark.value,
                        task.task_id,
                        step_index,
                        len(screenshot_history),
                    )
                if native_status == CAPTCHA_BUDGET_EXHAUSTED_STATUS:
                    # The browser could not reach the site, so nothing the policy
                    # did is measurable. Mask instead of scoring a forced stop.
                    environment_failure_kind = native_status
                    LOG.warning(
                        "event=web_environment_access_failed benchmark=%s task=%s step=%d failure_kind=%s",
                        task.benchmark.value,
                        task.task_id,
                        step_index,
                        environment_failure_kind,
                    )
                    rollout_finished = True
                    break
                if native_status == BROWSER_TARGET_CLOSED_STATUS:
                    # A native coordinate action can close the active browser
                    # target (for example, by clicking browser chrome).  The
                    # reference runner records that as an action/task failure,
                    # not as retryable infrastructure.  Keep the last valid
                    # screenshot as judge evidence and let `terminated` finish
                    # the rollout normally.
                    LOG.warning(
                        "event=web_environment_target_closed_after_action benchmark=%s task=%s step=%d",
                        task.benchmark.value,
                        task.task_id,
                        step_index,
                    )
                if action.terminal:
                    final_answer = action.answer
                if action.terminal or terminated or truncated:
                    rollout_finished = True
                    break
                if consecutive_execution_failures >= self.config.max_consecutive_execution_failures:
                    LOG.warning(
                        "event=web_execution_failure_limit benchmark=%s task=%s step=%d consecutive=%d",
                        task.benchmark.value,
                        task.task_id,
                        step_index,
                        consecutive_execution_failures,
                    )
                    truncated = True
                    rollout_finished = True
                    break
                trajectory.append(
                    render_observation(
                        observation,
                        task,
                        step_index=step_index + 1,
                        visual_observation_text=self.config.visual_observation_text,
                        action_prompt_profile=self.config.action_prompt_profile,
                        task_image_root=self.config.task_image_root,
                        max_task_image_bytes=self.config.max_task_image_bytes,
                    )
                )
                signature = _repeatable_action_signature(action)
                if signature is not None:
                    recent_action_signatures.append(signature)
                    occurrences = sum(item == signature for item in recent_action_signatures)
                    if (
                        self.config.repeated_action_warning_threshold > 0
                        and occurrences >= self.config.repeated_action_warning_threshold
                    ):
                        LOG.warning(
                            "event=web_repeated_action_warning benchmark=%s task=%s step=%d occurrences=%d window=%d",
                            task.benchmark.value,
                            task.task_id,
                            step_index,
                            occurrences,
                            len(recent_action_signatures),
                        )
                        trajectory.append(
                            NeMoGymEasyInputMessage(
                                role="user",
                                content=(
                                    "Recovery check: the same non-trivial browser action has repeated "
                                    f"{occurrences} times in the last {len(recent_action_signatures)} actions. "
                                    "Re-read the latest screenshot and choose a different verifiable action "
                                    "unless the page is visibly progressing."
                                ),
                            )
                        )

            if not rollout_finished:
                truncated = True

            evaluate_started = time.monotonic()
            LOG.info(
                "event=web_environment_evaluate_start benchmark=%s task=%s final_answer_present=%s screenshots=%d",
                task.benchmark.value,
                task.task_id,
                bool(final_answer),
                len(screenshot_history),
            )
            evaluate_response, evaluate_payload = await self._post_json(
                server_name=self.environment_server_name,
                url_path="/evaluate",
                json={"final_answer": final_answer},
                cookies=env_cookies,
                timeout_secs=self.config.resources_request_timeout_secs,
            )
            evaluation = WebEvaluateResponse.model_validate(evaluate_payload)
            env_cookies = evaluate_response.cookies
            verifier_result = evaluation.result
            LOG.info(
                "event=web_environment_evaluate_complete benchmark=%s task=%s valid_sample=%s "
                "failure_kind=%s elapsed_seconds=%.3f",
                task.benchmark.value,
                task.task_id,
                verifier_result.valid_sample,
                verifier_result.failure_kind or "none",
                time.monotonic() - evaluate_started,
            )

        finally:
            if seeded:
                close_started = time.monotonic()
                try:
                    _close_response, close_payload = await self._post_json(
                        server_name=self.environment_server_name,
                        url_path="/close",
                        json={},
                        cookies=env_cookies,
                        timeout_secs=self.config.close_request_timeout_secs,
                    )
                    close_data = WebCloseResponse.model_validate(close_payload)
                    if close_data.session_id is not None:
                        artifacts.session_id = close_data.session_id
                    artifacts.recordings = close_data.recording_artifacts
                    LOG.info(
                        "event=web_session_close_complete benchmark=%s task=%s session=%s recordings=%d "
                        "elapsed_seconds=%.3f",
                        task.benchmark.value,
                        task.task_id,
                        artifacts.session_id or "unknown",
                        len(artifacts.recordings),
                        time.monotonic() - close_started,
                    )
                except Exception as exc:  # noqa: BLE001 - cleanup must not replace a completed result.
                    LOG.warning(
                        "event=web_session_close_failed benchmark=%s task=%s session=%s error_type=%s "
                        "elapsed_seconds=%.3f",
                        task.benchmark.value,
                        task.task_id,
                        artifacts.session_id or "unknown",
                        type(exc).__name__,
                        time.monotonic() - close_started,
                        exc_info=True,
                    )

        if last_model_response is None:
            raise RuntimeError("web rollout ended before the policy returned a response")
        last_model_response.output = trajectory
        last_model_response.usage = usage

        judge_failure_metadata: dict[str, Any] = {}
        # The WebVoyager judge only consumes retained evidence. Release the
        # browser before the potentially slow VLM call so one judged episode
        # does not occupy scarce browser/site capacity. WebArena-family native
        # evaluators still run above while their live page is available.
        if task.benchmark == WebBenchmark.WEBVOYAGER and environment_failure_kind is None:
            judge_started = time.monotonic()
            LOG.info(
                "event=web_judge_start benchmark=%s task=%s screenshots=%d urls=%d final_answer_present=%s",
                task.benchmark.value,
                task.task_id,
                len(screenshot_history),
                len(url_history),
                bool(final_answer),
            )
            verifier_result, judge_failure_metadata = await self._verify_webvoyager(
                task=task,
                final_answer=final_answer or "",
                screenshots=list(screenshot_history),
                urls=list(url_history),
                body=body,
                response=last_model_response,
            )
            LOG.info(
                "event=web_judge_complete benchmark=%s task=%s valid_sample=%s success=%s reward=%s "
                "failure_kind=%s elapsed_seconds=%.3f",
                task.benchmark.value,
                task.task_id,
                verifier_result.valid_sample,
                verifier_result.task_success,
                verifier_result.reward,
                verifier_result.failure_kind or "none",
                time.monotonic() - judge_started,
            )

        if environment_failure_kind is not None:
            verifier_result = WebVerifierResult(
                valid_sample=False,
                failure_kind=environment_failure_kind,
            )
        if verifier_result is None:
            verifier_result = WebVerifierResult(
                valid_sample=False,
                failure_kind="missing_verifier_result",
            )

        return WebAgentRunResponse(
            responses_create_params=base_body,
            response=last_model_response,
            reward=verifier_result.reward if verifier_result.valid_sample else 0.0,
            benchmark=task.benchmark.value,
            task_id=task.task_id,
            raw_score=verifier_result.raw_score,
            task_success=verifier_result.task_success,
            mask_sample=not verifier_result.valid_sample,
            failure_kind=verifier_result.failure_kind,
            terminated=terminated,
            truncated=truncated,
            environment_steps=environment_steps,
            model_turns=model_turns,
            execution_failures=execution_failures,
            verifier_result=verifier_result,
            artifact_session_id=artifacts.session_id,
            recording_artifacts=artifacts.recordings,
            **judge_failure_metadata,
        )

    async def _verify_webvoyager(
        self,
        *,
        task: WebTask,
        final_answer: str,
        screenshots: list[str],
        urls: list[str],
        body: WebAgentRunRequest,
        response: NeMoGymResponse,
    ) -> tuple[WebVerifierResult, dict[str, Any]]:
        evidence = compact_webvoyager_judge_evidence(
            response=response,
            final_answer=final_answer,
            screenshots=screenshots,
            page_urls=urls,
        )
        # Store one compact immutable representation in the persisted response.
        # Most screenshots already occur in the trajectory and are referenced by
        # index; only boundary screenshots absent from the trajectory stay inline.
        setattr(response, "webvoyager_judge_evidence", evidence)
        # Initial verification does not need to resend the full policy trajectory:
        # the top-level fields below carry the evidence once. Generic reverify later
        # receives the original persisted response and expands its compact sequence.
        verification_response = NeMoGymResponse.model_validate(
            response.model_dump(mode="json", exclude={"webvoyager_judge_evidence"})
        )
        verification_response.output = []
        request_body = body.model_dump(mode="json") | {
            "web_task": task.model_dump(mode="json"),
            "response": verification_response.model_dump(mode="json"),
            "final_answer": final_answer,
            "screenshots": screenshots,
            "page_urls": urls,
        }
        _judge_response, payload = await self._post_json(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=request_body,
            timeout_secs=self.config.judge_request_timeout_secs,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("WebVoyager verifier response was not an object")
        failure_metadata = {
            key: payload[key]
            for key in (NG_FAILURE_CLASS_KEY, NG_TERMINAL_KEY, "_ng_failure_judge_error")
            if key in payload
        }
        valid_sample = not bool(payload.get("mask_sample", False)) and NG_FAILURE_CLASS_KEY not in payload
        result = WebVerifierResult(
            reward=float(payload.get("reward", 0.0)),
            raw_score=float(payload.get("raw_score", payload.get("reward", 0.0))),
            task_success=bool(payload.get("task_success", False)),
            valid_sample=valid_sample,
            failure_kind=(
                str(payload["failure_kind"])
                if payload.get("failure_kind")
                else ("judge_failed" if NG_FAILURE_CLASS_KEY in payload else None)
            ),
            verifier_version=str(payload.get("verifier_version", "webvoyager-llm-judge-v1")),
            metadata=dict(payload.get("verifier_metadata") or {}),
        )
        return result, failure_metadata

    async def _post_json(
        self,
        *,
        server_name: str,
        url_path: str,
        timeout_secs: float,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """Bound headers, status handling, and response-body parsing as one hop."""

        async def invoke() -> tuple[Any, Any]:
            server_response = await self.server_client.post(
                server_name=server_name,
                url_path=url_path,
                **kwargs,
            )
            await raise_for_status(server_response)
            return server_response, await get_response_json(server_response)

        return await asyncio.wait_for(invoke(), timeout=timeout_secs)

    async def _seed_session(
        self,
        *,
        task: WebTask,
        cookies: Any,
    ) -> tuple[Any, Any]:
        """Wait through transient resource-server failures without spending a rollout attempt."""

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.config.seed_request_timeout_secs
        delay = self.config.seed_retry_initial_delay_secs
        attempt = 0
        while True:
            attempt += 1
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"seed_session exceeded {self.config.seed_request_timeout_secs:.1f}s retry budget")
            try:
                return await self._post_json(
                    server_name=self.environment_server_name,
                    url_path="/seed_session",
                    json={"task": task.model_dump(mode="json")},
                    cookies=cookies,
                    timeout_secs=remaining,
                )
            except ClientResponseError as exc:
                if not 500 <= exc.status < 600:
                    raise
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise
                sleep_for = min(delay, remaining)
                LOG.warning(
                    "event=web_seed_retry benchmark=%s task=%s attempt=%d http_status=%d sleep_seconds=%.1f",
                    task.benchmark.value,
                    task.task_id,
                    attempt,
                    exc.status,
                    sleep_for,
                )
                await asyncio.sleep(sleep_for)
                if delay > 0:
                    delay = min(
                        max(delay * 2, self.config.seed_retry_initial_delay_secs),
                        self.config.seed_retry_max_delay_secs,
                    )

    @staticmethod
    def _failure_response(
        body: WebAgentRunRequest,
        task: WebTask,
        exc: Exception,
        artifacts: _RunArtifacts | None = None,
    ) -> WebAgentRunResponse:
        """Return a classified sidecar row for a bounded runtime failure.

        A real scheduler/process kill cannot return a response and therefore
        naturally leaves only a checkpoint gap.  Once this method runs, the
        agent has converted the failure into a bounded, attributable outcome;
        persisting it in the failure sidecar is what lets resume enforce the
        per-task retry budget instead of redispatching the row forever. A
        structured non-retryable HTTP precondition is terminal and remains
        masked rather than being scored as a model failure.
        """

        detail = f"{type(exc).__name__}: {exc}"
        response_content = getattr(exc, "response_content", None)
        if isinstance(response_content, bytes):
            response_content = response_content.decode("utf-8", errors="replace")
        if response_content:
            detail = f"{detail}; response_body={str(response_content).strip()}"
        detail = detail[:500]
        failure_class, terminal, failure_kind, failure_metadata = _failure_route(exc)
        LOG.error(
            "event=web_rollout_classified_failure benchmark=%s task=%s failure_class=%s "
            "failure_kind=%s terminal=%s error_type=%s",
            task.benchmark.value,
            task.task_id,
            failure_class,
            failure_kind,
            terminal,
            type(exc).__name__,
        )
        verifier_result = WebVerifierResult(
            valid_sample=False,
            failure_kind=failure_kind,
            metadata={"error": detail, **failure_metadata},
        )
        empty_response = NeMoGymResponse(
            id=f"web-agent-failure-{task.benchmark.value}-{task.task_id}",
            created_at=0.0,
            model=body.responses_create_params.model or "web-policy",
            object="response",
            output=[],
            tools=[],
            parallel_tool_calls=False,
            tool_choice="auto",
        )
        routing = {
            NG_FAILURE_CLASS_KEY: failure_class,
            "error": detail,
        }
        if terminal:
            routing[NG_TERMINAL_KEY] = True
        artifacts = artifacts or _RunArtifacts()
        return WebAgentRunResponse(
            responses_create_params=body.responses_create_params,
            response=empty_response,
            reward=0.0,
            benchmark=task.benchmark.value,
            task_id=task.task_id,
            mask_sample=True,
            failure_kind=failure_kind,
            verifier_result=verifier_result,
            artifact_session_id=artifacts.session_id,
            recording_artifacts=artifacts.recordings,
            **routing,
        )

    @staticmethod
    def _remember_evidence(
        observation: WebObservation,
        screenshots: deque[str],
        urls: deque[str],
    ) -> None:
        if observation.screenshot is not None and observation.screenshot.data_url:
            screenshots.append(observation.screenshot.data_url)
        if observation.url:
            urls.append(observation.url)


if __name__ == "__main__":
    WebAgent.run_webserver()
