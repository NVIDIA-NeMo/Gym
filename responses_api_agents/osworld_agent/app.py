# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI agent that wraps OSWorld's desktop-env benchmark.

OSWorld owns a complete agent harness: a VM provider, a multi-step rollout
loop, and a per-task evaluator. The cleanest way to plug it into NeMo Gym is
to wrap the harness at the *agent* layer (same pattern as ``mini_swe_agent``
and ``tau2``): ``/run`` is the single entrypoint that takes a Gym JSONL row,
runs the full OSWorld rollout against the Gym policy model, and returns a
``BaseVerifyResponse`` with the final reward.

For a decoupled deployment, the optional Gym-native
``resources_servers/osworld/`` owns the live DesktopEnv and its inline
evaluator. The agent keeps the same rollout loop and talks to that server via
the DesktopEnv-compatible HTTP client.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import copy
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from asyncio import Semaphore
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional

import ray
from fastapi import Body
from pydantic import ConfigDict, Field, model_validator

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import EXECUTION_ID_SANDBOX_METADATA_KEY
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.rollout_correlation import maybe_explicit_execution_id_from_run_body
from nemo_gym.sandbox import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import (
    ServerClient,
    get_first_server_config_dict,
)
from responses_api_agents.osworld_agent.exact_trace import build_exact_trace_envelope
from responses_api_agents.osworld_agent.proxy import (
    inspect_proxy_config_file,
    parse_env_bool,
    task_requires_proxy,
)
from responses_api_agents.osworld_agent.runner_registry import DEFAULT_RUNNER_NAME, load_attr, resolve_runner_spec
from responses_api_agents.osworld_agent.trajectory import (
    build_trajectory_envelope,
    resolve_trajectory_identity,
)


LOG = logging.getLogger("nemo_gym.osworld_agent")

POINTER_PARALLEL_DISABLED_SENTINEL = "__nemo_gym_parallel_tools_disabled__"
POINTER_ANTHROPIC_VALIDATION_SENTINEL = "__nemo_gym_anthropic_key_deferred__"

_OSWORLD_LOG_CONTEXT_FIELDS = (
    "run_id",
    "adapter",
    "rollout_purpose",
    "sampling_event_id",
    "source_group_id",
    "execution_id",
    "rollout_id",
    "group_id",
    "rollout_index",
    "attempt_index",
    "task_id",
    "domain",
    "task_attempt",
    "step",
    "parse_attempt",
)
_MODEL_LOG_CONTEXT_HEADERS = {
    field: f"x-nemo-gym-log-{field.replace('_', '-')}" for field in _OSWORLD_LOG_CONTEXT_FIELDS
}
_ROLLOUT_DIAGNOSTIC_ENV_VARS = (
    "NEMO_GYM_RESPONSE_LOGGING",
    "OSWORLD_MODEL_IO_LOG",
    "OSWORLD_RUN_ID",
    "OSWORLD_TASK_ARTIFACT_ROOT",
    "OSWORLD_VM_EXEC_LOG",
    "RUN_TAG",
)


def _merge_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge a provider override without mutating shared config."""

    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mapping(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _apply_sandbox_provider_overrides(
    provider_config: Mapping[str, Any],
    overrides_by_provider: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply only the override matching the selected single provider."""

    if len(provider_config) != 1:
        raise ValueError("Resolved sandbox provider config must contain exactly one provider")
    provider_name, provider_options = next(iter(provider_config.items()))
    override = overrides_by_provider.get(provider_name)
    if override is None:
        return copy.deepcopy(dict(provider_config))
    if not isinstance(provider_options, Mapping) or not isinstance(override, Mapping):
        raise TypeError(f"Sandbox provider override for {provider_name!r} must merge two mappings")
    return {provider_name: _merge_mapping(provider_options, override)}


def _normalize_log_context(context: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Keep the small, non-secret identity fields allowed in evidence logs."""

    if not isinstance(context, Mapping):
        return {}
    normalized: Dict[str, Any] = {}
    for field in _OSWORLD_LOG_CONTEXT_FIELDS:
        value = context.get(field)
        if value is None or value == "":
            continue
        if field in {
            "rollout_index",
            "attempt_index",
            "task_attempt",
            "step",
            "parse_attempt",
        }:
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
        headers[_MODEL_LOG_CONTEXT_HEADERS[field]] = header_value[:1024]
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
    Local Nano Omni runs already use ``NANO_OMNI_VLLM_MODEL`` to configure the
    outbound vLLM adapter, so prefer that runtime source of truth and surface a
    warning when it disagrees with the global config instead of mislabelling
    every rollout (for example, as Claude Opus).
    """

    configured_name = str(global_config.get("policy_model_name") or "").strip()
    runtime_name = os.environ.get("OSWORLD_POLICY_MODEL_NAME", "").strip()
    if not runtime_name and runner_name == "nemotron_v3_nano_omni_agent":
        runtime_name = os.environ.get("NANO_OMNI_VLLM_MODEL", "").strip()
    if runtime_name:
        if configured_name and configured_name != runtime_name:
            LOG.warning(
                "Using runtime policy model %s instead of stale global policy_model_name %s",
                runtime_name,
                configured_name,
            )
        return runtime_name
    return configured_name


def _validate_runner_runtime(config: "OSWorldAgentConfig") -> Optional[str]:
    """Import the effective runner class inside the Gym-created agent venv."""

    runner_spec = resolve_runner_spec(
        config.runner_name,
        action_space=config.action_space,
        observation_type=config.observation_type,
        env_class_path=config.env_class_path,
        agent_class_path=config.agent_class_path,
        agent_kwargs=config.agent_kwargs,
    )
    is_pointer = runner_spec.kind == "pointer_agent"
    if is_pointer and not os.environ.get("PARALLEL_API_KEY"):
        # Pointer constructs its optional Parallel client while importing the
        # module. Match the rollout runtime's no-web-tools mode when no real
        # credential is configured.
        os.environ["PARALLEL_API_KEY"] = POINTER_PARALLEL_DISABLED_SENTINEL
    defer_anthropic_key = (
        is_pointer
        and bool(runner_spec.agent_kwargs.get("use_policy_endpoint", True))
        and not os.environ.get("ANTHROPIC_API_KEY")
    )
    if defer_anthropic_key:
        # Pointer validates this variable while importing, before Gym resolves
        # the per-rollout policy credential.
        os.environ["ANTHROPIC_API_KEY"] = POINTER_ANTHROPIC_VALIDATION_SENTINEL
    try:
        if runner_spec.agent_class_path:
            load_attr(runner_spec.agent_class_path)
    finally:
        if defer_anthropic_key and os.environ.get("ANTHROPIC_API_KEY") == POINTER_ANTHROPIC_VALIDATION_SENTINEL:
            os.environ.pop("ANTHROPIC_API_KEY", None)
    return runner_spec.agent_class_path


class OSWorldAgentConfig(BaseResponsesAPIAgentConfig):
    """OSWorld agent config.

    Fields named after upstream OSWorld so behaviour stays comparable to the
    `run_multienv.py` harness.
    """

    model_server: ModelServerRef
    resources_server: Optional[ResourcesServerRef] = None
    concurrency: int = 4
    provider_name: str = "docker"
    container_image: str = "docker://happysixd/osworld-docker:latest"  # OSWorld upstream's recommended VM image
    sandbox_provider: Optional[str | Dict[str, Any]] = None
    # Workload-scoped deltas applied after resolving a named provider. This lets
    # OSWorld bound OpenSandbox VM admission without changing the shared
    # provider profile used by long-startup workloads.
    sandbox_provider_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    sandbox_spec: Dict[str, Any] = Field(default_factory=dict)
    vm_path: Optional[str] = None
    sandbox_vm_path: Optional[str] = None
    sandbox_require_kvm: bool = True
    sandbox_ready_timeout_s: float = Field(default=600.0, gt=0)
    sandbox_ready_poll_s: float = Field(default=2.0, gt=0)
    headless: bool = True
    screen_width: int = 1920
    screen_height: int = 1080
    require_a11y_tree: bool = False
    client_password: str = "password"
    enable_proxy: bool = False
    # Preserve the upstream benchmark behavior by default. Training/deployment
    # profiles can set this to false to mask proxy-tagged tasks when no proxy
    # is configured.
    allow_direct_proxy_tasks: bool = True
    proxy_config_file: Optional[str] = None
    resources_server_token_env: str = "OSWORLD_RESOURCES_TOKEN"
    resources_request_timeout: float = Field(default=900.0, gt=0)
    resources_connect_timeout: float = Field(default=10.0, gt=0)
    resources_request_retries: int = Field(default=3, ge=1)
    max_steps: int = 15
    max_trajectory_length: int = 3
    sleep_after_execution: float = 0.5
    cache_dir: str = "cache"
    setup_cache_dir: Optional[str] = None
    asset_input_jsonl: Optional[str] = None
    max_tokens: int = 1500
    temperature: float = 1.0
    top_p: Optional[float] = 0.9  # set to null in yaml when running a reasoning model that rejects top_p
    mem_limit_mb: int = 0  # the upstream Docker provider owns QEMU/container limits
    step_timeout: int = 60  # per-action subprocess timeout (forwarded to provider; advisory in client.py)
    # End-to-end wall-clock deadline from Ray dispatch through VM creation,
    # desktop setup, agent steps, and evaluation.  The child also
    # receives this value as a cooperative deadline so model/step boundaries
    # can stop cleanly before the parent has to cancel the Ray task.
    task_timeout: int = 1800
    task_cancel_grace_s: float = Field(default=30.0, gt=0)
    docker_port_lock_timeout: float = Field(default=300.0, gt=0)  # concurrent Docker VM port allocation
    evaluator_disable_gpu: bool = True
    reward_mode: Literal["binary", "raw"] = "binary"
    runner_name: str = DEFAULT_RUNNER_NAME
    action_space: Optional[str] = None
    observation_type: Optional[str] = None
    env_class_path: Optional[str] = None
    agent_class_path: Optional[str] = None
    agent_kwargs: Dict[str, Any] = Field(default_factory=dict)
    # A NeMo-RL scheduler may stamp the request purpose. Keep the default
    # agent kwargs as the standalone benchmark behavior and apply only the
    # explicitly configured purpose-specific delta here. Gym never infers a
    # purpose from trajectory/token evidence because all rollout modes emit
    # the same semantic contract.
    agent_kwargs_by_rollout_purpose: Dict[Literal["training", "evaluation"], Dict[str, Any]] = Field(
        default_factory=dict
    )

    @model_validator(mode="before")
    @classmethod
    def reject_removed_training_switches(cls, value: Any) -> Any:
        """Fail loudly instead of silently accepting obsolete export modes."""

        if isinstance(value, Mapping):
            removed = sorted(field for field in ("training_mode", "training_turn_strategy") if field in value)
            if removed:
                raise ValueError("OSWorld trajectory evidence is now automatic; remove: " + ", ".join(removed))
        return value


class OSWorldRunRequest(BaseRunRequest):
    """Per-task request. ``verifier_metadata`` holds the OSWorld task spec."""

    model_config = ConfigDict(extra="allow")
    # Keep this scheduler contract at the OSWorld HTTP boundary. Adding it to
    # BaseRunRequest changes every Gym server's serialized request shape, while
    # the metadata carrier below already survives generic /run schemas.
    rollout_purpose: Optional[Literal["training", "evaluation"]] = None


_ROLLOUT_PURPOSE_METADATA_KEY = "nemo_rl_rollout_purpose"
_LEGACY_TRAJECTORY_IDENTITY_KEYS = (
    "context_compaction_contract_version",
    "context_compaction_rollout_id",
    "context_compaction_group_id",
    "context_compaction_task_id",
    "context_compaction_rollout_index",
    "context_compaction_attempt_index",
)


def _resolve_run_rollout_purpose(
    body: OSWorldRunRequest,
) -> Optional[Literal["training", "evaluation"]]:
    """Resolve and cross-check the scheduler purpose at the agent boundary.

    NeMo-RL sends the purpose both as the top-level control field and inside
    responses metadata. The latter survives generic /run schemas which know
    only standard Responses API fields. Standalone benchmark callers may omit
    both carriers.
    """

    top_level = body.rollout_purpose
    metadata = body.responses_create_params.metadata or {}
    metadata_purpose = metadata.get(_ROLLOUT_PURPOSE_METADATA_KEY)
    if metadata_purpose is not None and metadata_purpose not in {
        "training",
        "evaluation",
    }:
        raise ValueError(f"invalid {_ROLLOUT_PURPOSE_METADATA_KEY}: {metadata_purpose!r}")
    if top_level is not None and metadata_purpose is not None and top_level != metadata_purpose:
        raise ValueError(f"rollout purpose carriers disagree: top_level={top_level!r}, metadata={metadata_purpose!r}")
    return top_level or metadata_purpose


class OSWorldAgentResponse(NeMoGymResponse):
    """OSWorld response plus universal trajectory and optional exact evidence."""

    model_config = ConfigDict(extra="allow")

    media_assets: Optional[Dict[str, Dict[str, Any]]] = None
    completion_evidence: Optional[List[Dict[str, Any]]] = None
    final_policy_decision: Optional[Dict[str, Any]] = None
    lineage_deltas: Optional[List[Dict[str, Any]]] = None
    chunk_records: Optional[List[Dict[str, Any]]] = None
    boundary_events: Optional[List[Dict[str, Any]]] = None
    guard_records: Optional[List[Dict[str, Any]]] = None
    trajectory_contract: Optional[Dict[str, Any]] = None
    trajectory_transitions: Optional[List[Dict[str, Any]]] = None
    trajectory_model_calls: Optional[List[Dict[str, Any]]] = None
    model_call_summaries: Optional[List[Dict[str, Any]]] = None
    context_compaction_contract: Optional[Dict[str, Any]] = None
    execution_context: Optional[Dict[str, Any]] = None


class OSWorldVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    response: OSWorldAgentResponse
    # NeMo-RL trainer drops the gradient when reward is unreliable. Set true on
    # timeout / max_steps exhaustion (no DONE/FAIL) / evaluator throw.
    mask_sample: bool = False


def _explicit_trajectory_identity(
    body: OSWorldRunRequest,
) -> Optional[Dict[str, Any]]:
    """Resolve caller-owned semantic identity without inventing one on errors."""

    extra = body.model_extra or {}
    if "trajectory_identity" not in extra and not any(key in extra for key in _LEGACY_TRAJECTORY_IDENTITY_KEYS):
        return None
    return resolve_trajectory_identity(
        request_extra=extra,
        verifier_metadata=body.verifier_metadata or {},
        model_name="",
    )


def _build_execution_context(
    execution_id: str,
    trajectory_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    """Bind one physical execution to its logical sampling identity."""

    return {
        "schema_version": 1,
        "execution_id": execution_id,
        "sampling_event_id": trajectory_identity.get("sampling_event_id"),
        "source_group_id": trajectory_identity.get("source_group_id"),
        "rollout_id": trajectory_identity["rollout_id"],
        "group_id": trajectory_identity["group_id"],
        "task_id": trajectory_identity["task_id"],
    }


def _build_policy_openai_client(*, base_url: str, api_key: str):
    """Build a client for the Gym-managed, internal policy endpoint.

    The process may need an environment proxy for unrelated services such as
    W&B, but agent-to-policy traffic must remain inside the cluster. Disabling
    ``trust_env`` also prevents httpx from eagerly constructing an unused
    SOCKS transport when the target is covered by ``NO_PROXY``.
    """
    from openai import DefaultHttpxClient, OpenAI  # noqa: PLC0415

    return OpenAI(
        base_url=base_url,
        api_key=api_key or "dummy",
        http_client=DefaultHttpxClient(trust_env=False),
    )


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
    log_context: Optional[Mapping[str, Any]] = None,
) -> Callable[[str, str, List[Dict[str, Any]]], str]:
    """Return a sync ``model_fn`` that hits a Gym vLLM/OpenAI-compatible model.

    OSWorld's loop is sync and runs inside Ray; we use the ``openai`` SDK in
    sync mode here. The actual NeMo Gym model server speaks the chat
    completions / responses API, so an OpenAI-compatible client over its
    ``host:port/v1`` URL is the right shape.
    """
    client = _build_policy_openai_client(base_url=base_url, api_key=api_key)
    base_log_context = _normalize_log_context(log_context)

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
        context_headers = _log_context_headers(base_log_context)
        if context_headers:
            create_kwargs["extra_headers"] = context_headers
        resp = client.chat.completions.create(**create_kwargs)
        return resp.choices[0].message.content or ""

    return _call


def _build_messages_model_fn(
    *,
    base_url: str,
    model_name: str,
    api_key: str,
    log_context: Optional[Mapping[str, Any]] = None,
    rollout_purpose: Optional[Literal["training", "evaluation"]] = None,
):
    """Return a sync model caller for native OSWorld agents.

    Native mm_agents such as PromptAgent construct their own OpenAI-style
    messages. Gym still owns the actual policy endpoint, so this thin adapter
    forwards those messages to the configured model server.
    """
    client = _build_policy_openai_client(base_url=base_url, api_key=api_key)
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
        if rollout_purpose is not None:
            # The Gym vLLM proxy reads metadata.extra_body and forwards the
            # decoded fields to NeMo-RL's internal vLLM endpoint. Standalone
            # benchmark calls omit this side channel entirely.
            create_kwargs["metadata"] = {
                "extra_body": json.dumps(
                    {"nemo_rl_rollout_purpose": rollout_purpose},
                    separators=(",", ":"),
                )
            }
        print(
            "OSWORLD_MODEL_PURPOSE|"
            f"purpose={rollout_purpose}|"
            f"temperature={create_kwargs.get('temperature')}|"
            f"top_p={create_kwargs.get('top_p')}|"
            f"carrier={'metadata' if rollout_purpose is not None else 'none'}",
            flush=True,
        )
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


def _structured_action_code(part: Any) -> str | None:
    """Translate one Nano Omni structured GUI action into adapter code.

    Nano Omni occasionally serializes its native ``click`` part into the
    chat ``content`` string alongside a textual Action description.  The
    generation is still exact and trainable; only its semantic transport
    shape differs from the Markdown scaffold expected by OSWorld.
    """

    part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
    if part_type == "action":
        action = part.get("action") if isinstance(part, dict) else getattr(part, "action", None)
        if action != "click":
            return None
        action_input = part.get("input") if isinstance(part, dict) else getattr(part, "input", None)
        if not isinstance(action_input, Mapping):
            raise ValueError(f"Structured click action has invalid input: {part!r}")
        x = action_input.get("x")
        y = action_input.get("y")
    elif part_type == "click":
        x = part.get("x") if isinstance(part, dict) else getattr(part, "x", None)
        y = part.get("y") if isinstance(part, dict) else getattr(part, "y", None)
    else:
        return None
    if (
        isinstance(x, bool)
        or not isinstance(x, (int, float))
        or isinstance(y, bool)
        or not isinstance(y, (int, float))
        or not math.isfinite(float(x))
        or not math.isfinite(float(y))
    ):
        raise ValueError(f"Structured click has invalid coordinates: {part!r}")
    return f"pyautogui.click({float(x):.12g}, {float(y):.12g})"


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
    action_codes: List[str] = []
    for part in content:
        part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
        text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        if part_type in {"text", "output_text"} and isinstance(text, str):
            text_parts.append(_normalize_chat_content(text, _depth=_depth + 1))
            continue
        action_code = _structured_action_code(part)
        if action_code is None:
            raise ValueError(f"Unsupported chat content part: {part!r}")
        action_codes.append(action_code)
    if action_codes:
        if len(action_codes) != 1:
            raise ValueError(f"Expected one structured GUI action, received {len(action_codes)}")
        action_text = "\n".join(part.strip() for part in text_parts if part.strip())
        if not re.search(r"^\s*##\s*Action\s*:?", action_text, re.MULTILINE | re.IGNORECASE):
            action_text = "## Action:\n" + (action_text or "Execute the generated click action.")
        return action_text.rstrip() + "\n## Code:\n```python\n" + action_codes[0] + "\n```"
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

    raw_content = message.content or ""
    normalization_error = None
    try:
        content = _normalize_chat_content(raw_content)
    except Exception as exc:  # noqa: BLE001 - exact generation evidence must survive parser failures.
        if not structured:
            raise
        # A semantic adapter failure must not erase a completed model call's
        # prompt IDs, sampled IDs, logprobs, or routed experts.  Return the raw
        # content to the agent parser, which will reject the action normally,
        # while preserving exact evidence for trajectory reconstruction.
        normalization_error = {
            "type": type(exc).__name__,
            "message": repr(exc),
        }
        content = raw_content if isinstance(raw_content, str) else repr(_jsonable(raw_content))

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
        if normalization_error is None:
            try:
                # A vLLM proxy can wrap a serialized structured action after
                # <think>.  The first normalization pass sees only a string;
                # this second pass is where the structured payload is decoded.
                # Keep it inside the exact-evidence preservation boundary too.
                content = _normalize_chat_content(content)
            except Exception as exc:  # noqa: BLE001 - preserve exact sampled evidence.
                normalization_error = {
                    "type": type(exc).__name__,
                    "message": repr(exc),
                }
                content = content if isinstance(content, str) else repr(_jsonable(content))
        normalized = {
            "content": content,
            "reasoning_content": reasoning,
            "raw_content": raw_content,
        }
        if normalization_error is not None:
            normalized["normalization_error"] = normalization_error
        for field in (
            "prompt_token_ids",
            "generation_token_ids",
            "generation_log_probs",
            "routed_experts",
        ):
            value = getattr(message, field, None)
            if value is None:
                value = model_extra.get(field)
            if value is not None:
                normalized[field] = value
        return normalized
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
    rollout_purpose = runner_kwargs.pop("rollout_purpose", None)
    execution_id = runner_kwargs.pop("execution_id", None)
    log_context = _normalize_log_context(runner_kwargs.pop("log_context", None))
    print(f"OSWORLD_CHILD_PURPOSE|purpose={rollout_purpose}|temperature={temperature}|top_p={top_p}", flush=True)
    model_fn = _build_model_fn(
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        log_context=log_context,
    )
    messages_model_fn = _build_messages_model_fn(
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        log_context=log_context,
        rollout_purpose=rollout_purpose,
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
        "execution_id": execution_id,
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
                "state": s.state,
                "next_state": s.next_state,
            }
            for s in result.steps
        ],
    }


class OSWorldAgent(SimpleResponsesAPIAgent):
    config: OSWorldAgentConfig
    sem: Optional[Semaphore] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        removed_agent_kwargs = sorted(
            field for field in ("training_mode", "training_turn_strategy") if field in self.config.agent_kwargs
        )
        if removed_agent_kwargs:
            raise ValueError(
                "OSWorld training-specific export switches were removed; "
                "trajectory evidence is now automatic. Remove: "
                + ", ".join(f"agent_kwargs.{field}" for field in removed_agent_kwargs)
            )
        if self.config.resources_server is not None and self.config.sandbox_provider is not None:
            raise ValueError("OSWorld resources_server and sandbox_provider cannot be enabled together")
        _validate_runner_runtime(self.config)
        self.sem = Semaphore(self.config.concurrency)

    def setup_webserver(self):
        """Idempotently fill a configured asset cache before accepting work."""

        base_run_request_module = sys.modules[BaseRunRequest.__module__]
        osworld_request_has_purpose = "rollout_purpose" in OSWorldRunRequest.model_fields
        if not osworld_request_has_purpose:
            raise RuntimeError("OSWorldRunRequest is missing its scheduler-owned rollout_purpose field")
        runtime_identity = (
            "OSWORLD_GYM_RUNTIME_IDENTITY|"
            f"app={Path(__file__).resolve()}|"
            f"base_run_request={Path(base_run_request_module.__file__).resolve()}|"
            f"base_rollout_purpose_field={'rollout_purpose' in BaseRunRequest.model_fields}|"
            f"osworld_rollout_purpose_field={osworld_request_has_purpose}"
        )
        # Ray's server logging profile filters INFO records in production.
        # stdout is captured reliably and contains paths/schema only, no task
        # payload or credentials.
        print(runtime_identity, flush=True)
        LOG.info(
            "OSWORLD_GYM_RUNTIME_IDENTITY|app=%s|base_run_request=%s|base_rollout_purpose_field=%s|osworld_rollout_purpose_field=%s",
            Path(__file__).resolve(),
            Path(base_run_request_module.__file__).resolve(),
            "rollout_purpose" in BaseRunRequest.model_fields,
            osworld_request_has_purpose,
        )

        if self.config.asset_input_jsonl and self.config.setup_cache_dir:
            from benchmarks.osworld.assets import ensure_osworld_assets

            summary = ensure_osworld_assets(
                self.config.asset_input_jsonl,
                self.config.setup_cache_dir,
                token=os.environ.get("HF_TOKEN"),
                proxy_url=os.environ.get("OSWORLD_ASSET_PROXY_URL"),
            )
            LOG.info(
                "OSWorld assets ready: tasks=%d assets=%d new_entries=%d cache=%s",
                summary.task_count,
                summary.asset_count,
                summary.materialized_count,
                summary.cache_dir,
            )
        return super().setup_webserver()

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
            execution_id = maybe_explicit_execution_id_from_run_body(body)
            top_level_rollout_purpose = body.rollout_purpose
            resolved_rollout_purpose = _resolve_run_rollout_purpose(body)
            # Normalize once so every downstream consumer and the response use
            # the same checked value even if the generic HTTP boundary retained
            # only the metadata carrier.
            body.rollout_purpose = resolved_rollout_purpose
            print(
                "OSWORLD_RUN_PURPOSE|"
                f"top_level={top_level_rollout_purpose or 'none'}|"
                f"metadata={(body.responses_create_params.metadata or {}).get(_ROLLOUT_PURPOSE_METADATA_KEY, 'none')}|"
                f"resolved={resolved_rollout_purpose or 'none'}",
                flush=True,
            )
            print(
                f"OSWORLD_RUN_EXECUTION|execution_id={execution_id or 'none'}",
                flush=True,
            )
            # The OSWorld task spec lives in verifier_metadata. Allow falling
            # back to model_extra so simple JSONL files can put it at the top
            # level — useful when hand-authoring examples.
            metadata = body.verifier_metadata or {}
            task_config = metadata.get("osworld_task") or (body.model_extra or {}).get("osworld_task")
            if not task_config:
                return _empty_response(body, error="No 'osworld_task' provided in verifier_metadata.")

            try:
                requires_proxy = task_requires_proxy(task_config)
                enable_proxy = parse_env_bool("OSWORLD_ENABLE_PROXY", self.config.enable_proxy)
                allow_direct_proxy_tasks = parse_env_bool(
                    "OSWORLD_ALLOW_DIRECT_PROXY_TASKS",
                    self.config.allow_direct_proxy_tasks,
                )
            except ValueError as exc:
                return _empty_response(
                    body,
                    error=f"ProxyConfigurationError: {exc}",
                    termination_reason="proxy_configuration_error",
                )

            remote_resources = self.config.resources_server is not None
            proxy_config_file = os.environ.get("PROXY_CONFIG_FILE") or self.config.proxy_config_file
            if not requires_proxy or not enable_proxy or remote_resources:
                # A remote Resources Server owns its proxy configuration; do
                # not leak a control-plane path into the environment plane.
                proxy_config_file = None
            if requires_proxy and not enable_proxy and not allow_direct_proxy_tasks:
                return _empty_response(
                    body,
                    error=("ProxyRequiredButDisabled: task requires a proxy, but OSWORLD_ENABLE_PROXY is disabled"),
                    termination_reason="proxy_required_but_disabled",
                    proxy_required=True,
                    proxy_enabled=False,
                    proxy_configured=bool(proxy_config_file),
                )
            if requires_proxy and not enable_proxy and allow_direct_proxy_tasks:
                if remote_resources:
                    return _empty_response(
                        body,
                        error=(
                            "ProxyConfigurationError: direct proxy task mode is not supported "
                            "by the remote Resources Server"
                        ),
                        termination_reason="proxy_configuration_error",
                        proxy_required=True,
                        proxy_enabled=False,
                        proxy_configured=False,
                    )
                proxy_config_file = None
            if requires_proxy and enable_proxy and not remote_resources:
                try:
                    proxy_config_file = inspect_proxy_config_file(proxy_config_file).path
                except ValueError as exc:
                    return _empty_response(
                        body,
                        error=f"ProxyConfigurationError: {exc}",
                        termination_reason="proxy_configuration_error",
                        proxy_required=True,
                        proxy_enabled=True,
                        proxy_configured=bool(proxy_config_file),
                    )

            model_server_name = self.config.model_server.name
            global_config_dict = ServerClient.load_from_global_config().global_config_dict
            model_server_config = get_first_server_config_dict(global_config_dict, model_server_name)
            policy_model_name = _resolve_policy_model_name(global_config_dict, self.config.runner_name)
            policy_api_key = global_config_dict.get("policy_api_key", "")
            policy_base_url = global_config_dict.get("policy_base_url", "")
            sandbox_provider_config: Optional[Dict[str, Any]] = None
            sandbox_spec = dict(self.config.sandbox_spec)
            if self.config.sandbox_provider is not None:
                sandbox_provider_config = resolve_provider_config(
                    self.config.sandbox_provider,
                    global_config_dict,
                )
                sandbox_provider_config = _apply_sandbox_provider_overrides(
                    sandbox_provider_config,
                    self.config.sandbox_provider_overrides,
                )
                default_metadata = resolve_provider_metadata(
                    self.config.sandbox_provider,
                    global_config_dict,
                )
                sandbox_spec["metadata"] = {
                    **default_metadata,
                    **dict(sandbox_spec.get("metadata") or {}),
                }
            model_server_root = f"http://{model_server_config['host']}:{model_server_config['port']}"
            base_url = f"{self.base_url_for_run(model_server_root, body)}/v1"
            resources_server_url = ""
            if self.config.resources_server is not None:
                resources_server_config = get_first_server_config_dict(
                    global_config_dict,
                    self.config.resources_server.name,
                )
                resources_server_url = f"http://{resources_server_config['host']}:{resources_server_config['port']}"

            temperature = (
                body.responses_create_params.temperature
                if body.responses_create_params.temperature is not None
                else self.config.temperature
            )
            top_p = (
                body.responses_create_params.top_p
                if body.responses_create_params.top_p is not None
                else self.config.top_p
            )
            max_tokens = (
                body.responses_create_params.max_output_tokens
                if body.responses_create_params.max_output_tokens is not None
                else self.config.max_tokens
            )
            extra = body.model_extra or {}
            trajectory_identity = resolve_trajectory_identity(
                request_extra=extra,
                verifier_metadata=metadata,
                model_name=policy_model_name,
            )
            if execution_id is not None:
                sandbox_spec["metadata"] = {
                    **dict(sandbox_spec.get("metadata") or {}),
                    EXECUTION_ID_SANDBOX_METADATA_KEY: execution_id,
                }
            print(
                "OSWORLD_RUN_IDENTITY|"
                f"rollout_id={trajectory_identity['rollout_id']}|"
                f"group_id={trajectory_identity['group_id']}|"
                f"task_id={trajectory_identity['task_id']}|"
                f"rollout_index={trajectory_identity['rollout_index']}|"
                f"attempt_index={trajectory_identity['attempt_index']}|"
                f"sampling_event_id={trajectory_identity.get('sampling_event_id', 'none')}|"
                f"source_group_id={trajectory_identity.get('source_group_id', 'none')}|"
                f"execution_id={execution_id or 'none'}|"
                f"source={trajectory_identity['identity_source']}",
                flush=True,
            )
            try:
                task_attempt = int(extra.get("_ng_rollout_index", 0)) + 1
            except (TypeError, ValueError):
                task_attempt = 1
            log_context = _normalize_log_context(
                {
                    "run_id": os.environ.get("OSWORLD_RUN_ID") or os.environ.get("RUN_TAG"),
                    "adapter": "gym",
                    "rollout_purpose": body.rollout_purpose,
                    "sampling_event_id": trajectory_identity.get("sampling_event_id"),
                    "source_group_id": trajectory_identity.get("source_group_id"),
                    "execution_id": execution_id,
                    "rollout_id": trajectory_identity["rollout_id"],
                    "group_id": trajectory_identity["group_id"],
                    "rollout_index": trajectory_identity["rollout_index"],
                    "attempt_index": trajectory_identity["attempt_index"],
                    "task_id": trajectory_identity["task_id"],
                    "domain": metadata.get("domain") or task_config.get("domain") or task_config.get("snapshot"),
                    "task_attempt": task_attempt,
                }
            )

            effective_agent_kwargs = dict(self.config.agent_kwargs)
            if body.rollout_purpose is not None:
                effective_agent_kwargs.update(
                    self.config.agent_kwargs_by_rollout_purpose.get(body.rollout_purpose, {})
                )

            runner_kwargs: Dict[str, Any] = {
                "provider_name": self.config.provider_name,
                "container_image": self.config.container_image,
                "headless": self.config.headless,
                "screen_size": (self.config.screen_width, self.config.screen_height),
                "require_a11y_tree": self.config.require_a11y_tree,
                "client_password": self.config.client_password,
                "enable_proxy": enable_proxy,
                "allow_direct_proxy_tasks": allow_direct_proxy_tasks,
                "proxy_config_file": proxy_config_file,
                "resources_server_url": resources_server_url,
                "resources_server_auth_token": os.environ.get(
                    self.config.resources_server_token_env,
                    "",
                ),
                "resources_request_timeout": self.config.resources_request_timeout,
                "resources_connect_timeout": self.config.resources_connect_timeout,
                "resources_request_retries": self.config.resources_request_retries,
                "sandbox_provider_config": sandbox_provider_config,
                "sandbox_spec": sandbox_spec,
                "vm_path": self.config.vm_path,
                "sandbox_vm_path": self.config.sandbox_vm_path,
                "sandbox_require_kvm": self.config.sandbox_require_kvm,
                "sandbox_ready_timeout_s": self.config.sandbox_ready_timeout_s,
                "sandbox_ready_poll_s": self.config.sandbox_ready_poll_s,
                "max_steps": self.config.max_steps,
                "max_trajectory_length": self.config.max_trajectory_length,
                "sleep_after_execution": self.config.sleep_after_execution,
                "cache_dir": self.config.cache_dir,
                "setup_cache_dir": self.config.setup_cache_dir,
                "mem_limit_mb": self.config.mem_limit_mb,
                "step_timeout": self.config.step_timeout,
                "task_timeout": self.config.task_timeout,
                "docker_port_lock_timeout": self.config.docker_port_lock_timeout,
                "evaluator_disable_gpu": self.config.evaluator_disable_gpu,
                "reward_mode": self.config.reward_mode,
                "base_url": base_url,
                "policy_base_url": policy_base_url,
                "model_name": policy_model_name,
                "api_key": policy_api_key,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "rollout_purpose": body.rollout_purpose,
                "execution_id": execution_id,
                "runner_name": self.config.runner_name,
                "action_space": self.config.action_space,
                "observation_type": self.config.observation_type,
                "env_class_path": self.config.env_class_path,
                "agent_class_path": self.config.agent_class_path,
                "agent_kwargs": effective_agent_kwargs,
                "log_context": log_context,
            }

            future = None
            try:
                # Child Ray tasks do not reliably inherit the NemoGym actor's
                # runtime environment. Forward diagnostic paths explicitly so
                # parse failures retain their model I/O and task trajectory.
                rollout_env = {name: os.environ[name] for name in _ROLLOUT_DIAGNOSTIC_ENV_VARS if os.environ.get(name)}
                runtime_env: Dict[str, Any] = {"py_executable": sys.executable}
                if rollout_env:
                    runtime_env["env_vars"] = rollout_env
                future = _run_osworld_task_remote.options(
                    runtime_env=runtime_env,
                ).remote(task_config, runner_kwargs)
                # ``run_osworld_task`` checks task_timeout only after DesktopEnv
                # has been constructed.  A wedged sandbox create therefore used
                # to leave ``ray.get`` waiting indefinitely.  Keep the child-side
                # cooperative checks, but also bound the complete attempt here so
                # VM setup failures cannot stall a benchmark or an RL batch.
                result_dict: Dict[str, Any] = await asyncio.to_thread(
                    ray.get,
                    future,
                    timeout=float(self.config.task_timeout),
                )
                if result_dict.get("execution_id") != execution_id:
                    raise ValueError(
                        "OSWorld child returned the wrong execution identity: "
                        f"expected={execution_id!r}, "
                        f"observed={result_dict.get('execution_id')!r}"
                    )
            except ray.exceptions.GetTimeoutError:
                if future is not None:
                    try:
                        # A cooperative cancellation raises KeyboardInterrupt in
                        # the child, which still executes run_osworld_task's
                        # ``finally`` and closes an allocated sandbox.  Bound
                        # that cleanup separately, then force-cancel only if the
                        # worker remains stuck in a transport/native call.
                        ray.cancel(future, force=False)
                        try:
                            await asyncio.to_thread(
                                ray.get,
                                future,
                                timeout=float(self.config.task_cancel_grace_s),
                            )
                        except ray.exceptions.GetTimeoutError:
                            ray.cancel(future, force=True)
                        except Exception:  # noqa: BLE001
                            # Cancellation normally surfaces as RayTaskError or
                            # TaskCancelledError after child cleanup completes.
                            pass
                    except Exception:  # noqa: BLE001
                        LOG.exception("Failed to cancel timed-out OSWorld Ray task")
                error = f"task_timeout exceeded ({self.config.task_timeout}s) during end-to-end rollout"
                LOG.error("OSWorld rollout timed out: %s", error)
                return _empty_response(
                    body,
                    error=error,
                    termination_reason="task_timeout",
                    proxy_required=requires_proxy,
                    proxy_enabled=enable_proxy,
                    proxy_configured=bool(proxy_config_file),
                )
            except Exception as exc:  # noqa: BLE001
                LOG.exception("OSWorld rollout failed")
                return _empty_response(body, error=f"{type(exc).__name__}: {exc}")

            # These values are owned by the current request, not the Ray
            # result payload. Assign them explicitly so a reused/malformed
            # payload cannot carry stale proxy provenance across tasks.
            result_dict["proxy_required"] = requires_proxy
            result_dict["proxy_enabled"] = enable_proxy
            result_dict["proxy_configured"] = bool(proxy_config_file)

            return _build_response(
                body,
                result_dict,
                policy_model_name,
                temperature,
                top_p,
                max_trajectory_length=self.config.max_trajectory_length,
                max_output_tokens=max_tokens,
            )


def _build_response(
    body: OSWorldRunRequest,
    result: Dict[str, Any],
    policy_model_name: str,
    temperature: float,
    top_p: Optional[float],
    *,
    max_trajectory_length: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
) -> OSWorldVerifyResponse:
    """Pack one run without changing its prompt policy for training consumers."""

    execution_id = maybe_explicit_execution_id_from_run_body(body)
    if result.get("execution_id") != execution_id:
        raise ValueError(
            "OSWorld result execution identity disagrees with its request: "
            f"expected={execution_id!r}, "
            f"observed={result.get('execution_id')!r}"
        )
    steps = result.get("steps", [])
    if not isinstance(steps, list):
        raise TypeError("OSWorld rollout steps must be a list")
    verifier_metadata = body.verifier_metadata or {}
    trajectory_fields, model_calls = build_trajectory_envelope(
        steps=steps,
        request_extra=body.model_extra or {},
        verifier_metadata=verifier_metadata,
        model_name=policy_model_name,
        sample_eligible=not bool(result.get("mask_sample", False)),
    )
    output: List[Dict[str, Any]] = [
        {
            "id": f"msg-step-{step['step']}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "annotations": [],
                    "text": str(step.get("model_text") or ""),
                }
            ],
        }
        for step in steps
    ]

    exact_fields: Dict[str, Any] = {}
    capabilities = trajectory_fields["trajectory_contract"]["capabilities"]
    if capabilities["exact_model_call_evidence"]:
        exact_fields = build_exact_trace_envelope(
            model_calls=model_calls,
            trajectory_contract=trajectory_fields["trajectory_contract"],
            model_name=policy_model_name,
            sampling_config={
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_output_tokens,
            },
            policy_config={
                "adapter": "osworld_agent",
                "prompt_materialization_contract": ("nemotron_v3_nano_omni_bounded_history_v1"),
                "max_trajectory_length": max_trajectory_length,
            },
        )
        if exact_fields.get("media_assets") != trajectory_fields.get("media_assets"):
            raise ValueError("Semantic trajectory and exact evidence disagree about media assets")
        exact_fields.pop("media_assets")
        # Exact model calls, including parser retries, are the trainable units.
        # Semantic step messages remain available through trajectory_transitions.
        output = exact_fields.pop("model_call_output")

    response_dict: Dict[str, Any] = {
        "id": f"osworld-{(body.verifier_metadata or {}).get('task_id', 'unknown')}",
        "created_at": 0.0,
        "model": policy_model_name,
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "temperature": temperature,
        "top_p": top_p,
        **trajectory_fields,
        **exact_fields,
    }
    if execution_id is not None:
        # Physical execution correlation is intentionally outside both semantic
        # trajectory contracts, so VM retries cannot alter their digests.
        response_dict["execution_context"] = _build_execution_context(
            execution_id,
            trajectory_fields["trajectory_contract"],
        )
    metadata = dict(body.verifier_metadata or {})
    metadata_steps: List[Dict[str, Any]] = []
    for step in steps:
        projected_step = dict(step)
        info = step.get("info")
        if isinstance(info, Mapping):
            projected_info = dict(info)
            agent_info = info.get("agent")
            if isinstance(agent_info, Mapping):
                projected_agent_info = dict(agent_info)
                raw_calls = projected_agent_info.pop("model_calls", None)
                if isinstance(raw_calls, list):
                    projected_agent_info["model_call_count"] = len(raw_calls)
                projected_info["agent"] = projected_agent_info
            projected_step["info"] = projected_info
        metadata_steps.append(projected_step)
    metadata["osworld_score"] = result.get("score", 0.0)
    metadata["osworld_finished"] = result.get("finished", False)
    metadata["osworld_error"] = result.get("error")
    metadata["osworld_steps"] = metadata_steps
    metadata["osworld_artifact_dir"] = result.get("artifact_dir")
    metadata["osworld_model_name"] = policy_model_name
    metadata["osworld_termination_reason"] = result.get("termination_reason")
    metadata["osworld_proxy_required"] = bool(result.get("proxy_required", False))
    metadata["osworld_proxy_enabled"] = bool(result.get("proxy_enabled", False))
    metadata["osworld_proxy_configured"] = bool(result.get("proxy_configured", False))
    if execution_id is not None:
        metadata["osworld_execution_id"] = execution_id

    response_fields: Dict[str, Any] = {
        "responses_create_params": body.responses_create_params,
        "rollout_purpose": body.rollout_purpose,
        "reward": float(result.get("reward", 0.0)),
        "response": response_dict,
        "verifier_metadata": metadata,
        "mask_sample": bool(result.get("mask_sample", False)),
    }
    return OSWorldVerifyResponse(**response_fields)


def _empty_response(
    body: OSWorldRunRequest,
    *,
    error: str,
    termination_reason: Optional[str] = None,
    proxy_required: bool = False,
    proxy_enabled: bool = False,
    proxy_configured: Optional[bool] = None,
) -> OSWorldVerifyResponse:
    LOG.warning("Returning empty OSWorld response: %s", error)
    execution_id = maybe_explicit_execution_id_from_run_body(body)
    metadata = dict(body.verifier_metadata or {})
    metadata["osworld_error"] = error
    if termination_reason:
        metadata["osworld_termination_reason"] = termination_reason
    metadata["osworld_proxy_required"] = proxy_required
    metadata["osworld_proxy_enabled"] = proxy_enabled
    metadata["osworld_proxy_configured"] = (
        bool(os.environ.get("PROXY_CONFIG_FILE")) if proxy_configured is None else proxy_configured
    )
    if execution_id is not None:
        metadata["osworld_execution_id"] = execution_id
    response_dict: Dict[str, Any] = {
        "id": "osworld-error",
        "created_at": 0.0,
        "model": "",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    if execution_id is not None:
        trajectory_identity = _explicit_trajectory_identity(body)
        if trajectory_identity is not None:
            response_dict["execution_context"] = _build_execution_context(
                execution_id,
                trajectory_identity,
            )
        else:
            # Legacy benchmark callers may have no semantic rollout identity.
            response_dict["execution_context"] = {
                "schema_version": 1,
                "execution_id": execution_id,
            }
    response_fields: Dict[str, Any] = {
        "responses_create_params": body.responses_create_params,
        "rollout_purpose": body.rollout_purpose,
        "reward": 0.0,
        "response": response_dict,
        "verifier_metadata": metadata,
        "mask_sample": True,
    }
    return OSWorldVerifyResponse(**response_fields)


if __name__ == "__main__":
    from responses_api_agents.osworld_agent.runtime_dependencies import require_optional_runtime_dependencies

    require_optional_runtime_dependencies()
    OSWorldAgent.run_webserver()
