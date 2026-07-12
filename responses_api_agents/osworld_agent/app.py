# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI agent that wraps OSWorld's desktop-env benchmark.

OSWorld owns a complete agent harness: a VM provider, a multi-step rollout
loop, and a per-task evaluator. The cleanest way to plug it into NeMo Gym is
to wrap the harness at the *agent* layer (same pattern as ``mini_swe_agent``
and ``tau2``): ``/run`` is the single entrypoint that takes a Gym JSONL row,
runs the full OSWorld rollout against the Gym policy model, and returns a
``BaseVerifyResponse`` with the final reward.

There is no paired ``resources_servers/osworld/`` because OSWorld's evaluator
runs inline in ``env.evaluate()`` — splitting it out would add an extra hop
that nothing else uses.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import sys
import time
from asyncio import Semaphore
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional

import ray
from fastapi import Body
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import (
    ServerClient,
    get_first_server_config_dict,
)
from responses_api_agents.osworld_agent.runner_registry import DEFAULT_RUNNER_NAME


LOG = logging.getLogger("nemo_gym.osworld_agent")

_OSWORLD_LOG_CONTEXT_FIELDS = (
    "run_id",
    "adapter",
    "task_id",
    "domain",
    "task_attempt",
    "step",
    "parse_attempt",
)
_OSWORLD_LOG_CONTEXT_HEADERS = {field: f"x-osworld-{field.replace('_', '-')}" for field in _OSWORLD_LOG_CONTEXT_FIELDS}


def _normalize_log_context(context: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Keep the small, non-secret identity fields allowed in evidence logs."""

    if not isinstance(context, Mapping):
        return {}
    normalized: Dict[str, Any] = {}
    for field in _OSWORLD_LOG_CONTEXT_FIELDS:
        value = context.get(field)
        if value is None or value == "":
            continue
        if field in {"task_attempt", "step", "parse_attempt"}:
            try:
                normalized[field] = int(value)
            except (TypeError, ValueError):
                continue
        else:
            normalized[field] = str(value)
    return normalized


def _log_context_headers(context: Mapping[str, Any] | None) -> Dict[str, str]:
    """Encode OSWorld identity as headers without changing the model body."""

    headers: Dict[str, str] = {}
    for field, value in _normalize_log_context(context).items():
        header_value = str(value).replace("\r", "").replace("\n", "")
        headers[_OSWORLD_LOG_CONTEXT_HEADERS[field]] = header_value[:1024]
    return headers


def _jsonable(value: Any) -> Any:
    """Return a JSON-compatible representation for model-I/O logs."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _model_io_images(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Index embedded images without removing them from the full request log."""

    images: List[Dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if not isinstance(url, str):
                continue
            encoded = url.split(",", 1)[1] if url.startswith("data:") and "," in url else ""
            try:
                decoded = base64.b64decode(encoded, validate=False) if encoded else b""
            except Exception:  # noqa: BLE001 - logging must not break a rollout.
                decoded = b""
            images.append(
                {
                    "message_index": message_index,
                    "part_index": part_index,
                    "data_url_chars": len(url),
                    "encoded_sha256": hashlib.sha256(encoded.encode("ascii", errors="ignore")).hexdigest(),
                    "decoded_bytes": len(decoded),
                    "decoded_sha256": hashlib.sha256(decoded).hexdigest(),
                }
            )
    return images


def _append_model_io(event: Dict[str, Any]) -> None:
    """Append a complete model-I/O event when opt-in logging is enabled."""

    path = os.environ.get("OSWORLD_MODEL_IO_LOG", "").strip()
    if not path:
        return
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        line = json.dumps(_jsonable(event), ensure_ascii=False, sort_keys=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        LOG.exception("Failed to append OSWorld model-I/O log to %s", path)


def _resolve_policy_model_name(global_config: Dict[str, Any], runner_name: str) -> str:
    """Resolve the model that the rollout actually sends to the policy endpoint.

    Deployment snapshots may retain a stale ``policy_model_name`` in env.yaml.
    Local Omni runs already use ``OMNI_MINI_VLLM_MODEL`` to configure the
    outbound vLLM adapter, so prefer that runtime source of truth and surface a
    warning when it disagrees with the global config instead of mislabelling
    every rollout (for example, as Claude Opus).
    """

    configured_name = str(global_config.get("policy_model_name") or "").strip()
    runtime_name = os.environ.get("OSWORLD_POLICY_MODEL_NAME", "").strip()
    if not runtime_name and runner_name == "nemotron_v3_nano_omni_agent":
        runtime_name = os.environ.get("OMNI_MINI_VLLM_MODEL", "").strip()
    if runtime_name:
        if configured_name and configured_name != runtime_name:
            LOG.warning(
                "Using runtime policy model %s instead of stale global policy_model_name %s",
                runtime_name,
                configured_name,
            )
        return runtime_name
    return configured_name


class OSWorldAgentConfig(BaseResponsesAPIAgentConfig):
    """OSWorld agent config.

    Fields named after upstream OSWorld so behaviour stays comparable to the
    `run_multienv.py` harness.
    """

    model_server: ModelServerRef
    concurrency: int = 4
    provider_name: str = "docker"
    container_image: str = "docker://happysixd/osworld-docker:latest"  # OSWorld upstream's recommended VM image
    headless: bool = True
    screen_width: int = 1920
    screen_height: int = 1080
    require_a11y_tree: bool = False
    client_password: str = "password"
    max_steps: int = 15
    max_trajectory_length: int = 3
    sleep_after_execution: float = 0.5
    cache_dir: str = "cache"
    max_tokens: int = 1500
    temperature: float = 1.0
    top_p: Optional[float] = 0.9  # set to null in yaml when running a reasoning model that rejects top_p
    mem_limit_mb: int = 0  # the upstream Docker provider owns QEMU/container limits
    step_timeout: int = 60  # per-action subprocess timeout (forwarded to provider; advisory in client.py)
    task_timeout: int = 1800  # whole-rollout wall-clock cap; trips mask_sample=True
    evaluator_disable_gpu: bool = True
    reward_mode: Literal["binary", "raw"] = "binary"
    runner_name: str = DEFAULT_RUNNER_NAME
    action_space: Optional[str] = None
    observation_type: Optional[str] = None
    env_class_path: Optional[str] = None
    agent_class_path: Optional[str] = None
    agent_kwargs: Dict[str, Any] = Field(default_factory=dict)


class OSWorldRunRequest(BaseRunRequest):
    """Per-task request. ``verifier_metadata`` holds the OSWorld task spec."""

    model_config = ConfigDict(extra="allow")


class OSWorldVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    # NeMo-RL trainer drops the gradient when reward is unreliable. Set true on
    # timeout / max_steps exhaustion (no DONE/FAIL) / evaluator throw.
    mask_sample: bool = False


# Imported lazily by ``_run_osworld_task_remote`` so this module imports
# cleanly without OSWorld installed.
def _build_model_fn(
    *,
    base_url: str,
    model_name: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    top_p: Optional[float],
) -> Callable[[str, str, List[Dict[str, Any]]], str]:
    """Return a sync ``model_fn`` that hits a Gym vLLM/OpenAI-compatible model.

    OSWorld's loop is sync and runs inside Ray; we use the ``openai`` SDK in
    sync mode here. The actual NeMo Gym model server speaks the chat
    completions / responses API, so an OpenAI-compatible client over its
    ``host:port/v1`` URL is the right shape.
    """
    from openai import OpenAI  # noqa: PLC0415  (lazy — heavy import)

    client = OpenAI(base_url=base_url, api_key=api_key or "dummy")

    def _call(system_prompt: str, instruction: str, observation_history: List[Dict[str, Any]]) -> str:
        # Build chat-style messages: system → (prev screenshots) → current screenshot+task.
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if not observation_history:
            return ""
        for prev in observation_history[:-1]:
            messages.append({"role": "user", "content": _format_observation(prev, instruction, is_current=False)})
        messages.append(
            {
                "role": "user",
                "content": _format_observation(observation_history[-1], instruction, is_current=True),
            }
        )
        # Prompt-size instrumentation: log per-call bytes / approx tokens so we
        # can spot context bloat. With a11y_tree on + max_trajectory_length=3,
        # an LibreOffice task can accumulate >1M prompt tokens by step 3-4 and
        # blow the 1M-context model ceiling; vision-only stays around ~10K tok.
        # Counts:
        #  - text_chars: every "text" part + system_prompt
        #  - images: each "image_url" entry; Anthropic charges ~1568 tok per
        #    1.15 MP image, so 1920×1080 ≈ 3000 tok/image
        #  - approx_tok ≈ text_chars/4 + images*3000  (rough; final word from API)
        text_chars = 0
        img_count = 0
        for _m in messages:
            _content = _m.get("content")
            if isinstance(_content, str):
                text_chars += len(_content)
            elif isinstance(_content, list):
                for _part in _content:
                    if isinstance(_part, dict):
                        if _part.get("type") == "text":
                            text_chars += len(_part.get("text", "") or "")
                        elif _part.get("type") == "image_url":
                            img_count += 1
        approx_tok = text_chars // 4 + img_count * 3000
        # print() not LOG.info because the gym Ray-worker config filters
        # below-WARN from `nemo_gym.osworld_agent`; print to stdout is always
        # captured by Ray + flushed to ng_run.log via the worker tag.
        print(
            f"[prompt-size] messages={len(messages)} text_chars={text_chars} "
            f"images={img_count} ~approx_tok={approx_tok}",
            flush=True,
        )
        create_kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # Some reasoning models (e.g. openai/openai/gpt-5.5 via inference-api)
        # reject top_p outright with HTTP 400. Skip the kwarg when None so
        # the request goes through cleanly; set top_p=null in osworld_agent.yaml
        # to opt into this behaviour.
        if top_p is not None:
            create_kwargs["top_p"] = top_p
        resp = client.chat.completions.create(**create_kwargs)
        return resp.choices[0].message.content or ""

    return _call


def _build_messages_model_fn(
    *,
    base_url: str,
    model_name: str,
    api_key: str,
    log_context: Optional[Mapping[str, Any]] = None,
):
    """Return a sync model caller for native OSWorld agents.

    Native mm_agents such as PromptAgent construct their own OpenAI-style
    messages. Gym still owns the actual policy endpoint, so this thin adapter
    forwards those messages to the configured model server.
    """
    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI(base_url=base_url, api_key=api_key or "dummy")
    call_index = 0
    base_log_context = _normalize_log_context(log_context)

    def _call(messages: List[Dict[str, Any]], payload: Dict[str, Any]) -> Any:
        nonlocal call_index
        call_log_context = dict(base_log_context)
        call_log_context.update(_normalize_log_context(payload.get("_osworld_log_context")))
        create_kwargs: Dict[str, Any] = {
            "model": payload.get("model") or model_name,
            "messages": messages,
            "max_tokens": payload.get("max_tokens"),
            "temperature": payload.get("temperature"),
        }
        if payload.get("top_p") is not None:
            create_kwargs["top_p"] = payload["top_p"]
        model_io_enabled = bool(os.environ.get("OSWORLD_MODEL_IO_LOG", "").strip())
        current_call = 0
        started_ns = 0
        if model_io_enabled:
            call_index += 1
            current_call = call_index
            request_value = _jsonable(create_kwargs)
            agent_payload = _jsonable(payload)
            request_json = json.dumps(request_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            payload_json = json.dumps(agent_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            started_ns = time.time_ns()
            _append_model_io(
                {
                    **call_log_context,
                    "schema_version": 2,
                    "event": "model_request",
                    "call_index": current_call,
                    "timestamp_unix_ns": started_ns,
                    "pid": os.getpid(),
                    "base_url": base_url,
                    "agent_payload": agent_payload,
                    "agent_payload_sha256": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
                    "openai_request": request_value,
                    "openai_request_sha256": hashlib.sha256(request_json.encode("utf-8")).hexdigest(),
                    "embedded_images": _model_io_images(messages),
                }
            )
        try:
            request_kwargs = dict(create_kwargs)
            context_headers = _log_context_headers(call_log_context)
            if context_headers:
                request_kwargs["extra_headers"] = context_headers
            resp = client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            if model_io_enabled:
                finished_ns = time.time_ns()
                _append_model_io(
                    {
                        **call_log_context,
                        "schema_version": 2,
                        "event": "model_error",
                        "call_index": current_call,
                        "timestamp_unix_ns": finished_ns,
                        "elapsed_ns": finished_ns - started_ns,
                        "pid": os.getpid(),
                        "error_type": type(exc).__name__,
                        "error": repr(exc),
                    }
                )
            raise
        choice = resp.choices[0]
        if payload.get("_nemo_gym_require_stop") and choice.finish_reason not in {"stop", "tool_calls"}:
            raise ValueError(f"Model response did not finish cleanly: finish_reason={choice.finish_reason!r}")
        if not model_io_enabled:
            return _normalize_chat_message(
                choice.message,
                structured=bool(payload.get("_nemo_gym_return_message")),
            )

        normalization_error = None
        normalization_exc: Exception | None = None
        normalized = None
        try:
            normalized = _normalize_chat_message(
                choice.message,
                structured=bool(payload.get("_nemo_gym_return_message")),
            )
        except Exception as exc:  # noqa: BLE001 - log raw output before preserving the original error.
            normalization_exc = exc
            normalization_error = {"type": type(exc).__name__, "message": repr(exc)}
        finished_ns = time.time_ns()
        _append_model_io(
            {
                **call_log_context,
                "schema_version": 2,
                "event": "model_response",
                "call_index": current_call,
                "timestamp_unix_ns": finished_ns,
                "elapsed_ns": finished_ns - started_ns,
                "pid": os.getpid(),
                "finish_reason": choice.finish_reason,
                "raw_response": _jsonable(resp),
                "raw_choice_message": _jsonable(choice.message),
                "normalized_response": _jsonable(normalized),
                "normalization_error": normalization_error,
            }
        )
        if normalization_exc is not None:
            raise normalization_exc
        return normalized

    return _call


def _recover_first_fenced_action(content: str) -> str | None:
    """Recover the first code block from a malformed serialized text-part list."""

    stripped = content.strip()
    if not stripped.startswith("[") or "text" not in stripped[:256].lower():
        return None
    fence_start = stripped.find("```")
    if fence_start < 0:
        return None
    fence_end = stripped.find("```", fence_start + 3)
    if fence_end < 0:
        return None
    fence = stripped[fence_start : fence_end + 3]
    return "## Action:\nExecute the first generated action.\n## Code:\n" + fence


def _normalize_chat_content(content: Any, *, _depth: int = 0) -> str:
    """Recover one executable turn without serializing structured content.

    The external-vLLM path can expose Chat Completions content as a list of
    text parts.  ``str(list)`` preserves literal ``\\n`` escapes inside code
    fences, producing invalid Python actions.  Some model responses also put
    several complete actions in separate text parts.  OSWorld executes one
    action per observation, so retain text only through the first complete
    fenced block instead of accidentally selecting the final ``terminate``.
    """

    if _depth > 4:
        raise ValueError("Chat content nesting exceeds four levels")
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("["):
            decoded: Any = None
            try:
                decoded = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                try:
                    decoded = json.loads(stripped)
                except json.JSONDecodeError:
                    pass
            if isinstance(decoded, list):
                LOG.warning("Recovering serialized chat content containing %d parts", len(decoded))
                return _normalize_chat_content(decoded, _depth=_depth + 1)
            recovered = _recover_first_fenced_action(stripped)
            if recovered:
                LOG.warning("Recovering first action from malformed serialized chat content")
                return recovered
        return content
    if not isinstance(content, list):
        raise ValueError(f"Unsupported chat content type: {type(content).__name__}")

    text_parts: List[str] = []
    for part in content:
        part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
        text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        if part_type not in {"text", "output_text"} or not isinstance(text, str):
            raise ValueError(f"Unsupported chat content part: {part!r}")
        text_parts.append(_normalize_chat_content(text, _depth=_depth + 1))
    if not text_parts:
        raise ValueError("Chat content contains no text parts")
    if len(text_parts) == 1:
        return text_parts[0]

    candidate = ""
    for text in text_parts:
        candidate += ("\n" if candidate else "") + text
        fence = re.search(r"```(?:code|python|json)?\s*.*?```", candidate, re.DOTALL | re.IGNORECASE)
        if fence:
            candidate = candidate[: fence.end()].strip()
            break
    else:
        raise ValueError(f"No complete code block in {len(text_parts)} chat text parts")

    if not re.search(r"^\s*##\s*Action\s*:?", candidate, re.MULTILINE | re.IGNORECASE):
        candidate = "## Action:\n" + candidate
    LOG.warning(
        "Model returned %d chat text parts; executing only the first complete action",
        len(text_parts),
    )
    return candidate


def _normalize_chat_message(message: Any, *, structured: bool = False) -> Any:
    """Normalize OpenAI native tool calls for text-protocol OSWorld agents."""

    content = _normalize_chat_content(message.content or "")

    # Tool-aware vLLM deployments can return native OpenAI tool_calls even
    # when the OSWorld agent scaffold expects textual <tool_call> blocks.
    # Normalize at the Gym transport boundary instead of patching OSWorld.
    textual_tool_calls: List[str] = []
    for tool_call in getattr(message, "tool_calls", None) or []:
        function = getattr(tool_call, "function", None)
        name = getattr(function, "name", None)
        raw_arguments = getattr(function, "arguments", None)
        if not name:
            continue
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        except json.JSONDecodeError:
            continue
        if not isinstance(arguments, dict):
            continue
        textual_tool_calls.append(
            "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False) + "\n</tool_call>"
        )
    if textual_tool_calls and "<tool_call>" not in content:
        content = "\n".join(part for part in [content, *textual_tool_calls] if part)

    if structured:
        model_extra = getattr(message, "model_extra", None) or {}
        reasoning = getattr(message, "reasoning_content", None) or model_extra.get("reasoning_content") or ""
        # Gym's external-vLLM proxy must return a schema-valid OpenAI message,
        # so it wraps vLLM's separate reasoning field in <think> tags. Recover
        # that field here for NemotronV3NanoOmniAgent, matching a direct vLLM call.
        if not reasoning:
            think_match = re.match(
                r"^\s*<think(?:ing)?>\s*(.*?)\s*</think(?:ing)?>\s*",
                content,
                re.DOTALL | re.IGNORECASE,
            )
            if think_match:
                reasoning = think_match.group(1).strip()
                content = content[think_match.end() :]
        content = _normalize_chat_content(content)
        return {"content": content, "reasoning_content": reasoning}
    return content


def _format_observation(obs: Dict[str, Any], instruction: str, *, is_current: bool) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    if is_current:
        parts.append(
            {
                "type": "text",
                "text": (
                    f"Task: {instruction}\n"
                    "Given the screenshot below, what's the next step you will take "
                    "to help complete the task?"
                ),
            }
        )
    else:
        parts.append({"type": "text", "text": "Previous screenshot:"})
    screenshot = obs.get("screenshot_b64") or ""
    if screenshot:
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot}", "detail": "high"},
            }
        )
    a11y = obs.get("accessibility_tree")
    if a11y:
        parts.append({"type": "text", "text": f"Accessibility tree:\n{a11y}"})
    return parts


@ray.remote(num_cpus=1)
def _run_osworld_task_remote(task_config: Dict[str, Any], runner_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Ray entrypoint: runs a single OSWorld task and returns a dict.

    Each remote task gets its own DesktopEnv — VMs are not shareable.
    """
    from responses_api_agents.osworld_agent.client import run_osworld_task  # noqa: PLC0415

    base_url = runner_kwargs.pop("base_url")
    policy_base_url = runner_kwargs.pop("policy_base_url", "")
    model_name = runner_kwargs.pop("model_name")
    api_key = runner_kwargs.pop("api_key")
    max_tokens = runner_kwargs.pop("max_tokens")
    temperature = runner_kwargs.pop("temperature")
    top_p = runner_kwargs.pop("top_p")
    log_context = _normalize_log_context(runner_kwargs.pop("log_context", None))
    model_fn = _build_model_fn(
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    messages_model_fn = _build_messages_model_fn(
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        log_context=log_context,
    )
    result = run_osworld_task(
        task_config,
        model_fn=model_fn,
        messages_model_fn=messages_model_fn,
        policy_base_url=policy_base_url,
        policy_api_key=api_key,
        policy_model_name=model_name,
        policy_max_tokens=max_tokens,
        policy_temperature=temperature,
        policy_top_p=top_p,
        log_context=log_context,
        **runner_kwargs,
    )
    return {
        "reward": result.reward,
        "score": result.score,
        "finished": result.finished,
        "error": result.error,
        "mask_sample": result.mask_sample,
        "artifact_dir": result.artifact_dir,
        "termination_reason": result.termination_reason,
        "steps": [
            {
                "step": s.step,
                "model_text": s.model_text,
                "actions": s.actions,
                "reward": s.reward,
                "done": s.done,
                "info": s.info,
            }
            for s in result.steps
        ],
    }


class OSWorldAgent(SimpleResponsesAPIAgent):
    config: OSWorldAgentConfig
    sem: Optional[Semaphore] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Report binary completion and raw OSWorld evaluator reward together."""

        rollouts = [rollout for task in tasks for rollout in task]
        raw_scores: List[float] = []
        masked_count = 0
        for rollout in rollouts:
            metadata = rollout.get("verifier_metadata")
            if not isinstance(metadata, Mapping):
                metadata = {}
            score = metadata.get("osworld_score", rollout.get("reward", 0.0))
            try:
                raw_scores.append(float(score or 0.0))
            except (TypeError, ValueError):
                raw_scores.append(0.0)
            masked_count += int(bool(rollout.get("mask_sample", False)))

        count = len(raw_scores)
        binary_successes = sum(score >= 1.0 for score in raw_scores)
        raw_reward = sum(raw_scores)
        return {
            "osworld/scored_rollout_count": count,
            "osworld/masked_rollout_count": masked_count,
            "osworld/binary_success_count": binary_successes,
            "osworld/binary_success_rate": 100.0 * binary_successes / count if count else 0.0,
            "osworld/raw_reward_sum": raw_reward,
            "osworld/raw_reward_rate": 100.0 * raw_reward / count if count else 0.0,
        }

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        metrics = super().get_key_metrics(agent_metrics)
        for key in ("osworld/binary_success_rate", "osworld/raw_reward_rate"):
            if key in agent_metrics:
                metrics[key] = agent_metrics[key]
        return metrics

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        # OSWorld's loop runs sync inside Ray; we do not expose a stand-alone
        # /v1/responses endpoint for this agent.
        raise NotImplementedError("OSWorldAgent runs full rollouts via /run only.")

    async def run(self, body: OSWorldRunRequest = Body()) -> OSWorldVerifyResponse:
        async with self.sem:
            # The OSWorld task spec lives in verifier_metadata. Allow falling
            # back to model_extra so simple JSONL files can put it at the top
            # level — useful when hand-authoring examples.
            metadata = body.verifier_metadata or {}
            task_config = metadata.get("osworld_task") or (body.model_extra or {}).get("osworld_task")
            if not task_config:
                return _empty_response(body, error="No 'osworld_task' provided in verifier_metadata.")

            model_server_name = self.config.model_server.name
            global_config_dict = ServerClient.load_from_global_config().global_config_dict
            model_server_config = get_first_server_config_dict(global_config_dict, model_server_name)
            policy_model_name = _resolve_policy_model_name(global_config_dict, self.config.runner_name)
            policy_api_key = global_config_dict.get("policy_api_key", "")
            policy_base_url = global_config_dict.get("policy_base_url", "")
            base_url = f"http://{model_server_config['host']}:{model_server_config['port']}/v1"

            temperature = body.responses_create_params.temperature or self.config.temperature
            top_p = body.responses_create_params.top_p or self.config.top_p
            extra = body.model_extra or {}
            try:
                task_attempt = int(extra.get("_ng_rollout_index", 0)) + 1
            except (TypeError, ValueError):
                task_attempt = 1
            log_context = _normalize_log_context(
                {
                    "run_id": os.environ.get("OSWORLD_RUN_ID") or os.environ.get("RUN_TAG"),
                    "adapter": "gym",
                    "task_id": metadata.get("task_id") or task_config.get("id") or task_config.get("task_id"),
                    "domain": metadata.get("domain") or task_config.get("domain") or task_config.get("snapshot"),
                    "task_attempt": task_attempt,
                }
            )

            runner_kwargs: Dict[str, Any] = {
                "provider_name": self.config.provider_name,
                "container_image": self.config.container_image,
                "headless": self.config.headless,
                "screen_size": (self.config.screen_width, self.config.screen_height),
                "require_a11y_tree": self.config.require_a11y_tree,
                "client_password": self.config.client_password,
                "max_steps": self.config.max_steps,
                "max_trajectory_length": self.config.max_trajectory_length,
                "sleep_after_execution": self.config.sleep_after_execution,
                "cache_dir": self.config.cache_dir,
                "mem_limit_mb": self.config.mem_limit_mb,
                "step_timeout": self.config.step_timeout,
                "task_timeout": self.config.task_timeout,
                "evaluator_disable_gpu": self.config.evaluator_disable_gpu,
                "reward_mode": self.config.reward_mode,
                "base_url": base_url,
                "policy_base_url": policy_base_url,
                "model_name": policy_model_name,
                "api_key": policy_api_key,
                "max_tokens": self.config.max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "runner_name": self.config.runner_name,
                "action_space": self.config.action_space,
                "observation_type": self.config.observation_type,
                "env_class_path": self.config.env_class_path,
                "agent_class_path": self.config.agent_class_path,
                "agent_kwargs": self.config.agent_kwargs,
                "log_context": log_context,
            }

            try:
                future = _run_osworld_task_remote.options(
                    runtime_env={"py_executable": sys.executable},
                ).remote(task_config, runner_kwargs)
                result_dict: Dict[str, Any] = await asyncio.to_thread(ray.get, future)
            except Exception as exc:  # noqa: BLE001
                LOG.exception("OSWorld rollout failed")
                return _empty_response(body, error=f"{type(exc).__name__}: {exc}")

            return _build_response(body, result_dict, policy_model_name, temperature, top_p)


def _build_response(
    body: OSWorldRunRequest,
    result: Dict[str, Any],
    policy_model_name: str,
    temperature: float,
    top_p: Optional[float],
) -> OSWorldVerifyResponse:
    """Pack the OSWorld rollout into the shape the verify pipeline expects."""

    response_dict: Dict[str, Any] = {
        "id": f"osworld-{(body.verifier_metadata or {}).get('task_id', 'unknown')}",
        "created_at": 0.0,
        "model": policy_model_name,
        "object": "response",
        "output": [
            {
                "id": f"msg-step-{step['step']}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "text": step["model_text"],
                    }
                ],
            }
            for step in result.get("steps", [])
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "temperature": temperature,
        "top_p": top_p,
    }
    metadata = dict(body.verifier_metadata or {})
    metadata["osworld_score"] = result.get("score", 0.0)
    metadata["osworld_finished"] = result.get("finished", False)
    metadata["osworld_error"] = result.get("error")
    metadata["osworld_steps"] = result.get("steps", [])
    metadata["osworld_artifact_dir"] = result.get("artifact_dir")
    metadata["osworld_model_name"] = policy_model_name
    metadata["osworld_termination_reason"] = result.get("termination_reason")

    return OSWorldVerifyResponse(
        responses_create_params=body.responses_create_params,
        reward=float(result.get("reward", 0.0)),
        response=response_dict,
        verifier_metadata=metadata,
        mask_sample=bool(result.get("mask_sample", False)),
    )


def _empty_response(body: OSWorldRunRequest, *, error: str) -> OSWorldVerifyResponse:
    LOG.warning("Returning empty OSWorld response: %s", error)
    metadata = dict(body.verifier_metadata or {})
    metadata["osworld_error"] = error
    return OSWorldVerifyResponse(
        responses_create_params=body.responses_create_params,
        reward=0.0,
        response={
            "id": "osworld-error",
            "created_at": 0.0,
            "model": "",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        },
        verifier_metadata=metadata,
        mask_sample=True,  # drop gradient when we couldn't even start the rollout
    )


if __name__ == "__main__":
    OSWorldAgent.run_webserver()
