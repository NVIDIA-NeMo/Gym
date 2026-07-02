# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synchronous rollout loop around OSWorld's ``DesktopEnv``.

This module owns the *non-Gym* side of the integration: spin up an OSWorld VM
via the Docker provider, drive the multi-step model→action→VM loop, run the
evaluator, and return a structured result.

It is intentionally **synchronous** because it is meant to run inside a
``ray.remote`` task. The outer ``OSWorldAgent`` (``app.py``) is the async
FastAPI side and dispatches one Ray task per rollout.

OSWorld is an optional runtime dependency — it is heavy
(torch + paddleocr + …) and only needed when actually executing rollouts.
The imports are gated so unit tests and module-level introspection don't
trigger them.
"""

from __future__ import annotations

import base64
import datetime
import inspect
import json
import logging
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from responses_api_agents.osworld_agent.action_parser import parse_actions, strip_thinking
from responses_api_agents.osworld_agent.runner_registry import load_attr, resolve_runner_spec


LOG = logging.getLogger("nemo_gym.osworld_agent.client")

# Sentinel actions OSWorld recognises in step().
_TERMINAL_ACTIONS = {"DONE", "FAIL"}


@dataclass
class StepRecord:
    step: int
    model_text: str
    actions: List[Any]
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutResult:
    reward: float
    score: float
    steps: List[StepRecord]
    error: Optional[str] = None
    finished: bool = False  # True iff the loop ended on DONE/FAIL or env.done
    # NeMo-RL drops the gradient for this sample when reward is unreliable. True iff:
    #  • error is set (model/evaluator/timeout), or
    #  • loop exhausted max_steps without DONE/FAIL (finished=False), or
    #  • task_timeout tripped.
    mask_sample: bool = False


# `ModelFn` takes (system_prompt, instruction, observation_history) and
# returns the raw model output text. The caller is responsible for invoking
# the model server — we keep this layer agnostic of HTTP / asyncio so it can
# be driven from a Ray actor.
ObservationHistory = List[Dict[str, Any]]
ModelFn = Callable[[str, str, ObservationHistory], str]
MessagesModelFn = Callable[[List[Dict[str, Any]], Dict[str, Any]], str]


def _b64(screenshot_bytes: Optional[bytes]) -> str:
    if not screenshot_bytes:
        return ""
    return base64.b64encode(screenshot_bytes).decode("ascii")


def _is_terminal_action(action: Any) -> bool:
    if isinstance(action, str) and action in _TERMINAL_ACTIONS:
        return True
    return isinstance(action, dict) and action.get("action_type") in _TERMINAL_ACTIONS


def _flatten_actions(actions: Any) -> List[Any]:
    if actions is None:
        return []
    if not isinstance(actions, list):
        return [actions]
    flattened: List[Any] = []
    for action in actions:
        if isinstance(action, list):
            flattened.extend(_flatten_actions(action))
        else:
            flattened.append(action)
    return flattened


def _normalize_prompt_agent_computer_13_action(action: Any) -> Any:
    """Normalize native PromptAgent computer_13 actions for DesktopEnv.

    OSWorld's PromptAgent prompt and parser are permissive: they return JSON
    actions directly, while DesktopEnv's PythonController accepts a narrower
    schema. Keep the compatibility shim in the Gym adapter so upstream OSWorld
    can continue to evolve independently.
    """
    if not isinstance(action, dict):
        return action

    normalized = dict(action)
    action_type = str(normalized.get("action_type", "")).upper()
    if action_type in _TERMINAL_ACTIONS or action_type == "WAIT":
        return action
    parameters = normalized.get("parameters")
    if isinstance(parameters, dict):
        params = dict(parameters)
    else:
        params = {key: value for key, value in normalized.items() if key != "action_type"}

    aliases = {
        "LEFT_CLICK": "CLICK",
        "MOUSE_MOVE": "MOVE_TO",
        "TYPE": "TYPING",
        "KEY": "PRESS",
    }
    action_type = aliases.get(action_type, action_type)

    click_type = params.pop("click_type", None)
    if isinstance(click_type, str):
        click_type = click_type.upper()
        if click_type == "RIGHT" and action_type == "CLICK":
            params.setdefault("button", "right")
        elif click_type == "MIDDLE" and action_type == "CLICK":
            params.setdefault("button", "middle")
        elif click_type == "LEFT" and action_type in {"CLICK", "MOUSE_DOWN", "MOUSE_UP"}:
            params.setdefault("button", "left")
        elif click_type == "WHEEL_UP":
            action_type = "SCROLL"
            params.setdefault("dy", 1)
        elif click_type == "WHEEL_DOWN":
            action_type = "SCROLL"
            params.setdefault("dy", -1)

    if action_type == "CLICK":
        params.setdefault("button", "left")
    elif action_type == "TRIPLE_CLICK":
        action_type = "CLICK"
        params.setdefault("button", "left")
        params.setdefault("num_clicks", 3)

    if "varies" in params and "y" not in params:
        params["y"] = params.pop("varies")

    return {"action_type": action_type, "parameters": params}


def _escape_prompt_agent_format_template(template: str) -> str:
    """Escape prompt braces while preserving OSWorld's password placeholder."""
    marker = "__NEMO_GYM_CLIENT_PASSWORD_PLACEHOLDER__"
    return (
        template.replace("{CLIENT_PASSWORD}", marker)
        .replace("{", "{{")
        .replace("}", "}}")
        .replace(marker, "{CLIENT_PASSWORD}")
    )


def _patch_native_prompt_agent_templates(agent_cls: Any) -> None:
    """Make upstream PromptAgent prompt templates safe for str.format().

    OSWorld's computer_13 prompts contain JSON examples with literal braces,
    then PromptAgent.__init__ calls `.format(CLIENT_PASSWORD=...)` on the
    whole prompt. Patch the module constants once so JSON braces are not
    interpreted as format fields.
    """
    if agent_cls.__module__ != "mm_agents.agent" or agent_cls.__name__ != "PromptAgent":
        return
    module = sys.modules.get(agent_cls.__module__)
    if module is None or getattr(module, "_NEMO_GYM_PROMPT_FORMAT_SAFE", False):
        return
    for name in (
        "SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION",
        "SYS_PROMPT_IN_A11Y_OUT_ACTION",
        "SYS_PROMPT_IN_BOTH_OUT_ACTION",
        "SYS_PROMPT_IN_SCREENSHOT_OUT_CODE",
        "SYS_PROMPT_IN_A11Y_OUT_CODE",
        "SYS_PROMPT_IN_BOTH_OUT_CODE",
        "SYS_PROMPT_IN_SOM_OUT_TAG",
    ):
        prompt = getattr(module, name, None)
        if isinstance(prompt, str):
            setattr(module, name, _escape_prompt_agent_format_template(prompt))
    setattr(module, "_NEMO_GYM_PROMPT_FORMAT_SAFE", True)


def _patch_m3_native_tool_use(agent: Any) -> None:
    """Translate InferenceHub tool blocks into M3Agent's text protocol.

    Upstream M3Agent asks MiniMax for textual ``<tool_call>`` blocks and only
    collects Anthropic ``text``/``thinking`` response blocks. InferenceHub may
    instead return the same computer action as a native ``tool_use`` block.
    Preserve the upstream prompt/parser and adapt only that transport shape.
    """
    original_call = getattr(agent, "_call_llm", None)
    if not callable(original_call) or getattr(agent, "_NEMO_GYM_NATIVE_TOOL_USE_PATCHED", False):
        return

    def _call_llm(messages: List[Dict[str, Any]]):
        response_text, raw_response = original_call(messages)
        if "<tool_call>" in (response_text or ""):
            return response_text, raw_response

        wrapped_calls: List[str] = []
        for block in getattr(raw_response, "content", []) or []:
            block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            if block_type != "tool_use":
                continue
            name = block.get("name") if isinstance(block, dict) else getattr(block, "name", None)
            tool_input = block.get("input") if isinstance(block, dict) else getattr(block, "input", None)
            if not name or not isinstance(tool_input, dict):
                continue
            tool_call = {"name": name, "arguments": tool_input}
            wrapped_calls.append("<tool_call>\n" + json.dumps(tool_call, ensure_ascii=False) + "\n</tool_call>")

        if wrapped_calls:
            response_text = "\n".join(part for part in [response_text, *wrapped_calls] if part)
        return response_text, raw_response

    agent._call_llm = _call_llm
    agent._NEMO_GYM_NATIVE_TOOL_USE_PATCHED = True


def _inferencehub_anthropic_model(short_name: str) -> str:
    return short_name if short_name.startswith("azure/anthropic/") else f"azure/anthropic/{short_name}"


def _normalize_anthropic_base_url(base_url: str) -> str:
    """Return a root URL suitable for Anthropic SDK /v1 path construction."""

    if not base_url:
        return ""
    normalized = base_url.rstrip("/")
    for suffix in ("/v1/chat/completions", "/v1/messages", "/v1/responses", "/v1"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _configure_pointer_runtime(
    *,
    base_url: str,
    api_key: str,
    policy_model_name: str,
    use_policy_endpoint: bool = False,
    disable_parallel_tools: bool = False,
) -> str:
    """Configure OSWorld's PointerAgent for direct Anthropic-compatible calls.

    Pointer reads model names from environment variables when
    ``mm_agents.pointer.config`` is imported. Set them before loading the
    agent so proxied Gym deployments use the same API endpoint/model namespace
    as the Gym policy config.
    """

    if api_key and use_policy_endpoint:
        os.environ["ANTHROPIC_API_KEY"] = api_key
    anthropic_base_url = _normalize_anthropic_base_url(base_url)
    if anthropic_base_url and use_policy_endpoint:
        os.environ["ANTHROPIC_BASE_URL"] = anthropic_base_url

    if disable_parallel_tools and not os.environ.get("PARALLEL_API_KEY"):
        os.environ["PARALLEL_API_KEY"] = "__nemo_gym_parallel_tools_disabled__"  # pragma: allowlist secret
        LOG.warning(
            "PARALLEL_API_KEY is not set; disabling PointerAgent optional "
            "web_search/web_fetch tools. Provide PARALLEL_API_KEY if those "
            "tools are required for leaderboard-aligned runs."
        )

    if anthropic_base_url and "inference-api.nvidia.com" in anthropic_base_url:
        os.environ.setdefault("POINTER_GATE_AGENT_MODEL", _inferencehub_anthropic_model("claude-sonnet-4-6"))
        os.environ.setdefault("POINTER_PLANNER_AGENT_MODEL", _inferencehub_anthropic_model("claude-sonnet-4-6"))
        os.environ.setdefault("POINTER_VERIFIER_AGENT_MODEL", _inferencehub_anthropic_model("claude-sonnet-4-6"))
        os.environ.setdefault("POINTER_SUMMARIZATION_MODEL", _inferencehub_anthropic_model("claude-haiku-4-5"))
        if policy_model_name:
            os.environ["POINTER_EXECUTOR_AGENT_MODEL"] = policy_model_name
    return anthropic_base_url


def _sync_pointer_config(policy_model_name: str) -> None:
    """Update Pointer's already-imported config singleton, if present."""

    try:
        from mm_agents.pointer.config import config as pointer_config  # type: ignore
        from mm_agents.pointer.utils import APIProvider as PointerAPIProvider  # type: ignore
    except Exception:  # noqa: BLE001 - pointer is optional outside runtime.
        return

    if os.environ.get("ANTHROPIC_BASE_URL"):
        pointer_config.provider = PointerAPIProvider.ANTHROPIC
    if policy_model_name:
        pointer_config.executor_model = policy_model_name
    if "inference-api.nvidia.com" in os.environ.get("ANTHROPIC_BASE_URL", ""):
        pointer_config.gate_model = os.environ.get("POINTER_GATE_AGENT_MODEL", pointer_config.gate_model)
        pointer_config.planner_model = os.environ.get("POINTER_PLANNER_AGENT_MODEL", pointer_config.planner_model)
        pointer_config.verifier_model = os.environ.get("POINTER_VERIFIER_AGENT_MODEL", pointer_config.verifier_model)
        pointer_config.summarization_model = os.environ.get(
            "POINTER_SUMMARIZATION_MODEL",
            pointer_config.summarization_model,
        )


def _patch_pointer_optional_parallel_tools(disable_parallel_tools: bool) -> None:
    """Disable PointerAgent web tools when Parallel credentials are unavailable."""

    if not disable_parallel_tools:
        return
    try:
        from mm_agents.pointer import agent_feasibility_gate as gate_module  # type: ignore
        from mm_agents.pointer import agent_planner as planner_module  # type: ignore
    except Exception:  # noqa: BLE001 - pointer is an optional runtime dependency.
        return

    disabled_names = {"web_search", "web_fetch"}

    def _tool_name(tool: Any) -> str:
        schema = getattr(tool, "schema", {})
        return schema.get("name", "") if isinstance(schema, dict) else ""

    if hasattr(gate_module, "GATE_TOOLS"):
        gate_module.GATE_TOOLS[:] = [tool for tool in gate_module.GATE_TOOLS if _tool_name(tool) not in disabled_names]
    if hasattr(planner_module, "PLANNER_TOOLS"):
        planner_module.PLANNER_TOOLS[:] = [
            tool for tool in planner_module.PLANNER_TOOLS if _tool_name(tool) not in disabled_names
        ]
    for module in (gate_module, planner_module):
        dispatch = getattr(module, "_TOOL_DISPATCH", None)
        if isinstance(dispatch, dict):
            for name in disabled_names:
                dispatch.pop(name, None)


def _pointer_anthropic_client_options(base_url: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Anthropic SDK options for PointerAgent's InferenceHub path."""

    return {
        "api_key": api_key or os.environ.get("ANTHROPIC_API_KEY"),
        "base_url": base_url.rstrip("/"),
        "max_retries": int(os.environ.get("POINTER_ANTHROPIC_MAX_RETRIES", "4")),
        "timeout": float(os.environ.get("POINTER_ANTHROPIC_TIMEOUT_SECONDS", "120")),
    }


def _patch_pointer_anthropic_client(base_url: str) -> None:
    """Make Pointer's Anthropic SDK client honor the configured base URL."""

    if not base_url:
        return
    try:
        from mm_agents.pointer import llm_client as pointer_llm_client  # type: ignore
        from mm_agents.pointer import llm_context_manager as pointer_context_manager  # type: ignore
        from mm_agents.pointer import utils as pointer_utils  # type: ignore
    except Exception:  # noqa: BLE001 - pointer is an optional runtime dependency.
        return

    original = getattr(
        pointer_llm_client.LLMClient,
        "_nemo_gym_original_create_client",
        pointer_llm_client.LLMClient._create_client,
    )
    if not hasattr(pointer_llm_client.LLMClient, "_nemo_gym_original_create_client"):
        setattr(pointer_llm_client.LLMClient, "_nemo_gym_original_create_client", original)

    def _create_client(self: Any, provider: Any) -> Any:
        if provider == pointer_utils.APIProvider.ANTHROPIC:
            from anthropic import Anthropic  # noqa: PLC0415

            return Anthropic(**_pointer_anthropic_client_options(base_url, self.api_key))
        return original(self, provider)

    pointer_llm_client.LLMClient._create_client = _create_client

    original_counting = getattr(
        pointer_context_manager.LLMContextManager,
        "_nemo_gym_original_get_counting_client",
        pointer_context_manager.LLMContextManager._get_counting_client,
    )
    if not hasattr(pointer_context_manager.LLMContextManager, "_nemo_gym_original_get_counting_client"):
        setattr(
            pointer_context_manager.LLMContextManager,
            "_nemo_gym_original_get_counting_client",
            original_counting,
        )

    def _get_counting_client(self: Any) -> Any:
        from anthropic import Anthropic  # noqa: PLC0415

        if not hasattr(self, "_counting_client"):
            self._counting_client = Anthropic(**_pointer_anthropic_client_options(base_url))
        return self._counting_client

    pointer_context_manager.LLMContextManager._get_counting_client = _get_counting_client


def _safe_task_id(task_config: Dict[str, Any]) -> str:
    raw = str(task_config.get("id") or task_config.get("task_id") or "unknown")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[:120] or "unknown"


def _record_video_for_task(task_config: Dict[str, Any]) -> bool:
    """Return whether this task is selected for opt-in VM video recording."""

    task_ids_path = os.environ.get("OSWORLD_RECORD_VIDEO_TASK_IDS_FILE", "").strip()
    if not task_ids_path:
        return True

    task_id = str(task_config.get("id") or task_config.get("task_id") or "unknown")
    try:
        with open(task_ids_path, encoding="utf-8") as fh:
            selected_task_ids = {line.strip() for line in fh if line.strip() and not line.lstrip().startswith("#")}
    except OSError:
        LOG.exception(
            "Failed to read OSWORLD_RECORD_VIDEO_TASK_IDS_FILE=%s; recording disabled for this task", task_ids_path
        )
        return False

    return task_id in selected_task_ids


def _setup_pointer_task_logger(
    task_config: Dict[str, Any], task_results_dir: str
) -> tuple[logging.Logger, logging.Handler]:
    os.makedirs(task_results_dir, exist_ok=True)
    logger_name = f"nemo_gym.osworld_agent.pointer.{_safe_task_id(task_config)}.{os.getpid()}"
    task_logger = logging.getLogger(logger_name)
    task_logger.setLevel(logging.INFO)
    task_logger.propagate = False
    handler = logging.FileHandler(os.path.join(task_results_dir, "pointer.log"), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(agent)s] %(message)s"))

    class _PointerAgentFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if not hasattr(record, "agent"):
                record.agent = "pointer"
            return True

    handler.addFilter(_PointerAgentFilter())
    task_logger.addHandler(handler)
    setattr(handler, "_nemo_gym_logger", task_logger)
    return task_logger, handler


def _evaluate_osworld_env(env: Any, eval_logger: logging.Logger) -> float:
    """Call the OSWorld evaluator across DesktopEnv variants."""

    evaluate = env.evaluate
    try:
        params = inspect.signature(evaluate).parameters
    except (TypeError, ValueError):
        params = {}
    if not params:
        return float(evaluate())
    return float(evaluate(eval_logger))


def run_osworld_task(
    task_config: Dict[str, Any],
    model_fn: ModelFn,
    *,
    provider_name: str = "docker",
    container_image: str = "docker://happysixd/osworld-docker:latest",
    headless: bool = True,
    screen_size: tuple = (1920, 1080),
    require_a11y_tree: bool = False,
    client_password: str = "password",
    max_steps: int = 15,
    max_trajectory_length: int = 3,
    sleep_after_execution: float = 0.5,
    system_prompt: Optional[str] = None,
    cache_dir: str = "cache",
    mem_limit_mb: int = 16384,
    step_timeout: int = 60,  # advisory; per-action subprocess timeout (provider-dependent)
    task_timeout: int = 1800,  # wall-clock cap on the whole rollout
    runner_name: str = "gym_pyautogui",
    action_space: Optional[str] = None,
    observation_type: Optional[str] = None,
    env_class_path: Optional[str] = None,
    agent_class_path: Optional[str] = None,
    agent_kwargs: Optional[Dict[str, Any]] = None,
    messages_model_fn: Optional[MessagesModelFn] = None,
    policy_base_url: str = "",
    policy_api_key: str = "",
    policy_model_name: str = "",
    policy_max_tokens: int = 1500,
    policy_temperature: float = 1.0,
    policy_top_p: Optional[float] = 0.9,
) -> RolloutResult:
    """Run a single OSWorld task and return a structured result.

    Heavy imports (``desktop_env``) happen inside the function so importing
    this module on a machine without OSWorld installed still works — only
    actually *running* a rollout requires it.
    """
    # Apptainer provider reads these env vars at start_emulator time. Must
    # be set before DesktopEnv() is constructed so the provider's lazy import
    # picks them up. (No-op for docker provider — it ignores these.)
    os.environ["OSWORLD_DOCKER_IMAGE"] = container_image
    os.environ["OSWORLD_APPTAINER_SIF_CACHE"] = os.path.join(cache_dir, "sif")
    os.environ["OSWORLD_APPTAINER_VMS_DIR"] = os.path.join(cache_dir, "vms")
    # mem_limit_mb=0 means "no cgroup memory cap" → skip setting the env var, so
    # provider.py doesn't add `--memory` to apptainer argv. The `--memory` flag
    # can trigger apptainer cgroup setup, which fails under enroot+fakeroot+
    # no-systemd stacks with `rootless cgroups is not usable in fakeroot mode`.
    # Set mem_limit_mb=0 when running under such setups; the docker provider
    # accepts --memory normally.
    if mem_limit_mb > 0:
        os.environ["OSWORLD_APPTAINER_MEM_LIMIT"] = f"{mem_limit_mb}M"

    from responses_api_agents.osworld_agent.prompts import get_system_prompt

    if system_prompt is None:
        system_prompt = get_system_prompt(client_password)

    runner_spec = resolve_runner_spec(
        runner_name,
        action_space=action_space,
        observation_type=observation_type,
        env_class_path=env_class_path,
        agent_class_path=agent_class_path,
        agent_kwargs=agent_kwargs,
    )
    env_cls = load_attr(runner_spec.env_class_path)
    instruction = task_config.get("instruction", "")

    env: Optional[Any] = None
    steps: List[StepRecord] = []
    obs_history: ObservationHistory = []
    error: Optional[str] = None
    finished = False
    final_score = 0.0
    timed_out = False
    task_start = time.monotonic()

    # Opt-in mp4 recording of the entire rollout, captured server-side inside
    # the VM and pulled back on env.close(). Saved at
    # ${OSWORLD_RECORD_VIDEO_DIR}/{task_id}.mp4.
    _record_dir = os.environ.get("OSWORLD_RECORD_VIDEO_DIR", "")
    _recording_started = False

    try:
        env = env_cls(
            provider_name=provider_name,
            action_space=runner_spec.action_space,
            screen_size=screen_size,
            headless=headless,
            require_a11y_tree=require_a11y_tree
            or runner_spec.observation_type in {"a11y_tree", "screenshot_a11y_tree", "som"},
            os_type="Ubuntu",
            client_password=client_password,
            cache_dir=cache_dir,
        )
        env.reset(task_config=task_config)
        native_agent = None
        pointer_agent = None
        pointer_log_handler: Optional[logging.Handler] = None
        if runner_spec.kind == "prompt_agent":
            if not runner_spec.agent_class_path:
                raise ValueError(f"runner {runner_spec.name!r} requires agent_class_path")
            if messages_model_fn is None:
                raise ValueError(f"runner {runner_spec.name!r} requires messages_model_fn")
            agent_cls = load_attr(runner_spec.agent_class_path)
            _patch_native_prompt_agent_templates(agent_cls)
            native_agent = agent_cls(
                platform="ubuntu",
                model=policy_model_name or "policy_model",
                max_tokens=policy_max_tokens,
                top_p=policy_top_p,
                temperature=policy_temperature,
                action_space=runner_spec.action_space,
                observation_type=runner_spec.observation_type,
                max_trajectory_length=max_trajectory_length,
                client_password=client_password,
                **runner_spec.agent_kwargs,
            )

            def _call_llm(payload: Dict[str, Any]) -> str:
                return strip_thinking(messages_model_fn(payload["messages"], payload) or "")

            native_agent.call_llm = _call_llm
            try:
                native_agent.reset(LOG, vm_ip=getattr(env, "vm_ip", None))
            except TypeError:
                native_agent.reset(LOG)
        elif runner_spec.kind == "m3_agent":
            if not runner_spec.agent_class_path:
                raise ValueError(f"runner {runner_spec.name!r} requires agent_class_path")
            agent_cls = load_attr(runner_spec.agent_class_path)
            m3_kwargs: Dict[str, Any] = {
                "platform": "ubuntu",
                "model": policy_model_name or "policy_model",
                "max_tokens": policy_max_tokens,
                "top_p": policy_top_p,
                "temperature": policy_temperature,
                "action_space": runner_spec.action_space,
                "observation_type": runner_spec.observation_type,
                "max_trajectory_length": max_trajectory_length,
                "client_password": client_password,
                "base_url": _normalize_anthropic_base_url(policy_base_url),
                "api_key": policy_api_key,
            }
            m3_kwargs.update(runner_spec.agent_kwargs)
            native_agent = agent_cls(**m3_kwargs)
            _patch_m3_native_tool_use(native_agent)
            try:
                native_agent.reset(LOG, vm_ip=getattr(env, "vm_ip", None))
            except TypeError:
                native_agent.reset(LOG)
            if hasattr(native_agent, "set_api_log_dir"):
                m3_results_base = os.environ.get(
                    "OSWORLD_M3_RESULTS_DIR",
                    os.path.join(cache_dir, "m3_runs"),
                )
                m3_task_dir = os.path.join(
                    m3_results_base,
                    f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{_safe_task_id(task_config)}",
                )
                native_agent.set_api_log_dir(os.path.join(m3_task_dir, "api_logs"))
        elif runner_spec.kind == "pointer_agent":
            if not runner_spec.agent_class_path:
                raise ValueError(f"runner {runner_spec.name!r} requires agent_class_path")
            pointer_kwargs = dict(runner_spec.agent_kwargs)
            disable_parallel_tools = bool(
                pointer_kwargs.pop("disable_parallel_tools", not os.environ.get("PARALLEL_API_KEY"))
            )
            use_policy_endpoint = bool(pointer_kwargs.pop("use_policy_endpoint", True))
            anthropic_base_url = _configure_pointer_runtime(
                base_url=policy_base_url,
                api_key=policy_api_key,
                policy_model_name=policy_model_name,
                use_policy_endpoint=use_policy_endpoint,
                disable_parallel_tools=disable_parallel_tools,
            )
            agent_cls = load_attr(runner_spec.agent_class_path)
            _sync_pointer_config(policy_model_name)
            _patch_pointer_optional_parallel_tools(disable_parallel_tools)
            _patch_pointer_anthropic_client(anthropic_base_url)
            pointer_agent = agent_cls(
                env=env,
                screen_size=screen_size,
                **pointer_kwargs,
            )
            pointer_results_base = os.environ.get(
                "OSWORLD_POINTER_RESULTS_DIR",
                os.path.join(cache_dir, "pointer_runs"),
            )
            pointer_task_dir = os.path.join(
                pointer_results_base,
                f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{_safe_task_id(task_config)}",
            )
            pointer_logger, pointer_log_handler = _setup_pointer_task_logger(task_config, pointer_task_dir)
            pointer_agent.reset(instruction, pointer_logger, pointer_task_dir)
        if _record_dir and _record_video_for_task(task_config):
            try:
                env.controller.start_recording()
                _recording_started = True
                LOG.info("Started VM recording → %s/<task_id>.mp4", _record_dir)
            except Exception:  # noqa: BLE001 — best-effort, recording is opt-in.
                LOG.exception("start_recording() failed; continuing without recording")
        elif _record_dir:
            LOG.info(
                "Skipping VM recording for task %s; not selected by OSWORLD_RECORD_VIDEO_TASK_IDS_FILE",
                task_config.get("id") or task_config.get("task_id") or "unknown",
            )
        # Opt-in: log every controller.execute_python_command request +
        # response from the VM's /execute endpoint as JSONL. The /execute
        # endpoint returns {status, output, error, returncode}; OSWorld's
        # env.step throws this away, but for debugging we want to see if
        # pyautogui clicks land an error / non-zero returncode silently in
        # the VM (hypothesis observed in earlier experiments: clicks hit
        # the right pixel but the X event never reaches the target window).
        _vm_exec_log_path = os.environ.get("OSWORLD_VM_EXEC_LOG")
        if _vm_exec_log_path:
            os.makedirs(os.path.dirname(_vm_exec_log_path), exist_ok=True)
            _orig_exec_py = env.controller.execute_python_command
            _exec_call_idx = [0]

            def _exec_logged(command: str):
                idx = _exec_call_idx[0]
                _exec_call_idx[0] += 1
                result = _orig_exec_py(command)
                try:
                    with open(_vm_exec_log_path, "a") as fh:
                        fh.write(
                            json.dumps(
                                {
                                    "call_idx": idx,
                                    "command_head": (command or "")[:300],
                                    "response": result if isinstance(result, dict) else {"_repr": repr(result)[:200]},
                                },
                                default=str,
                            )
                            + "\n"
                        )
                except Exception:
                    LOG.exception("Failed to write VM exec log entry %d", idx)
                return result

            env.controller.execute_python_command = _exec_logged

        # Opt-in one-shot diagnostic: probe the VM's actual display +
        # pyautogui dimensions. Output lands in OSWORLD_VM_EXEC_LOG (via the
        # monkey-patch above, if that env var is also set). Useful for
        # verifying: is the 1920x1080 screenshot resolution actually matched
        # by what pyautogui sees inside the VM, or is there a coord-scaling
        # mismatch?
        if os.environ.get("OSWORLD_VM_DIAG"):
            try:
                env.controller.execute_python_command(
                    "import subprocess\n"
                    "print('PYAUTOGUI_SIZE:', pyautogui.size())\n"
                    "print('PYAUTOGUI_POSITION:', pyautogui.position())\n"
                    "for cmd in (['xrandr','--current'],['xdpyinfo'],['xwininfo','-root']):\n"
                    "    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)\n"
                    "    print(f'== {cmd[0]} stdout (rc={r.returncode}) ==')\n"
                    "    print(r.stdout[:800])\n"
                    "    if r.stderr: print(f'   stderr: {r.stderr[:200]}')\n"
                    "print('DISPLAY env:', subprocess.run(['printenv','DISPLAY'], capture_output=True, text=True).stdout.strip())\n"
                )
            except Exception:
                LOG.exception("VM diag probe failed (non-fatal)")

        # NOTE: OSWORLD_CONTROLLED_CLICK setup is wired AFTER the cold-boot
        # poll below — empirically, controlled-click is meaningless until
        # Chrome has rendered a real frame (cold-boot ordering rule).

        # Opt-in: replace controller.pkgs_prefix with a version that wraps
        # pyautogui.click in moveTo → sleep → click(at-current-pos). Some
        # CSS-styled UI elements (e.g. Chrome's per-row `⋮` menu on
        # chrome://settings/searchEngines) are hover-gated (CSS
        # `display:none` until `:hover`). A bare pyautogui.click(x, y)
        # does moveTo + immediate mouseDown/mouseUp — too fast for the
        # row's :hover state to render the icon, so the click lands on
        # empty space. With a 0.5s pause between move and click, the
        # hover state has time to register and the click lands correctly.
        if os.environ.get("OSWORLD_HOVER_BEFORE_CLICK"):
            env.controller.pkgs_prefix = (
                "import pyautogui as _pa\n"
                "import time as _t\n"
                "_pa.FAILSAFE = False\n"
                "_orig_click = _pa.click\n"
                "_orig_dbl = _pa.doubleClick\n"
                "_orig_right = _pa.rightClick\n"
                "def _hover(x=None, y=None, *a, _orig=None, **kw):\n"
                "    if x is not None and y is not None and isinstance(x, (int, float)):\n"
                "        _pa.moveTo(x, y, duration=0.2)\n"
                "        _t.sleep(0.5)\n"
                "        return _orig(*a, **kw)\n"
                "    return _orig(x, y, *a, **kw)\n"
                "_pa.click = lambda *a, **kw: _hover(*a, _orig=_orig_click, **kw)\n"
                "_pa.doubleClick = lambda *a, **kw: _hover(*a, _orig=_orig_dbl, **kw)\n"
                "_pa.rightClick = lambda *a, **kw: _hover(*a, _orig=_orig_right, **kw)\n"
                "pyautogui = _pa\n"
                "time = _t\n"
                "{command}"
            )

        # Poll the VM screenshot until it shows real desktop content (not a
        # solid black frame), or until OSWORLD_COLD_BOOT_TIMEOUT_S elapses.
        # Why polling beats a fixed sleep:
        #   - A black-screen PNG of a 1920x1080 frame compresses to ~6.4 KB;
        #     a real desktop with any content is 15-50 KB. The size gap is
        #     clean enough to use as a "ready" signal.
        #   - On KVM-enabled hosts (/dev/kvm exposed) the desktop is up in
        #     ~10-20s; on TCG-only hosts (software emulation, no /dev/kvm)
        #     it takes 60-90s. A fixed sleep either wastes time on the fast
        #     path or under-waits on the slow path; polling adapts.
        #   - The threshold is intentionally conservative so it errs toward
        #     waiting more, not less. Override via env if a task starts on a
        #     genuinely small / mostly-blank screen (e.g. a fullscreen black
        #     terminal).
        cold_boot_timeout = int(os.environ.get("OSWORLD_COLD_BOOT_TIMEOUT_S", "180"))
        # Empirical step-0 PNG sizes observed during chrome cold-boot:
        #   ~6 KB = solid black (qemu loading), ~17 KB = Chrome window loading
        #   with blank New Tab, 19-42 KB = loaded New Tab content, ~58 KB =
        #   New Tab with extra popups. 10K passes the loading state cleanly;
        #   25K requires real content. Default set to 25K to err on caution.
        cold_boot_min_png_bytes = int(os.environ.get("OSWORLD_COLD_BOOT_MIN_PNG_BYTES", "25000"))
        cold_boot_poll_s = float(os.environ.get("OSWORLD_COLD_BOOT_POLL_S", "5"))
        boot_start = time.monotonic()
        obs: Dict[str, Any] = {}
        while time.monotonic() - boot_start < cold_boot_timeout:
            try:
                obs = env._get_obs()  # noqa: SLF001 — OSWorld's official entrypoint.
                png = obs.get("screenshot") or b""
            except Exception as exc:  # noqa: BLE001 — VM may not be ready yet, retry.
                LOG.debug("VM cold-boot poll: env._get_obs() raised %s; retrying", exc)
                png = b""
            if len(png) > cold_boot_min_png_bytes:
                LOG.info(
                    "VM ready after %.1fs (screenshot %d bytes >= threshold %d)",
                    time.monotonic() - boot_start,
                    len(png),
                    cold_boot_min_png_bytes,
                )
                break
            time.sleep(cold_boot_poll_s)
        else:
            LOG.warning(
                "VM cold-boot timed out at %ds; proceeding with last obs (screenshot %d bytes)",
                cold_boot_timeout,
                len(obs.get("screenshot") or b""),
            )

        # Capture the focus state of the X server right before the
        # controlled-click fires. Useful when investigating: is Chrome
        # actually the X-focused window after cold-boot? If not, synthesized
        # clicks may land on a different window's client and the target app
        # never sees them. Output lands in OSWORLD_VM_EXEC_LOG via the
        # monkey-patch above.
        if os.environ.get("OSWORLD_CONTROLLED_CLICK"):
            try:
                env.controller.execute_python_command(
                    "import subprocess\n"
                    "for cmd in (['xdotool','getactivewindow'],\n"
                    "            ['xdotool','getactivewindow','getwindowname'],\n"
                    "            ['xdotool','getactivewindow','getwindowgeometry'],\n"
                    "            ['wmctrl','-l'],\n"
                    "            ['wmctrl','-lG']):\n"
                    "    r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)\n"
                    "    print(f'== {\" \".join(cmd)} rc={r.returncode} ==')\n"
                    "    print(r.stdout[:600])\n"
                    "    if r.stderr: print(f'   stderr: {r.stderr[:200]}')\n"
                )
            except Exception:
                LOG.exception("focus diag probe failed (non-fatal)")

        # Opt-in: drive a sequence of controlled moveTo + sleep + click
        # actions with before/after screenshot capture. Verifies whether a
        # known coord actually opens the expected UI, bypassing the model.
        # Useful for debugging click-delivery (does the X event reach the
        # target window?) without coupling to the model's spatial reasoning.
        # Pick a target on whatever page is rendered after cold-boot.
        # Format: OSWORLD_CONTROLLED_CLICK="x,y" or "x,y;x,y;..." for sequence.
        controlled_click = os.environ.get("OSWORLD_CONTROLLED_CLICK")
        if controlled_click:
            cc_dir = os.environ.get("OSWORLD_CONTROLLED_CLICK_DIR", "/tmp/controlled_click")
            try:
                os.makedirs(cc_dir, exist_ok=True)
                LOG.info("OSWORLD_CONTROLLED_CLICK=%r dir=%r", controlled_click, cc_dir)
                obs_before = env._get_obs()  # noqa: SLF001
                png_before = obs_before.get("screenshot") if isinstance(obs_before, dict) else None
                if png_before:
                    with open(os.path.join(cc_dir, "before.png"), "wb") as fh:
                        fh.write(png_before)
                for i, coord_str in enumerate(controlled_click.split(";")):
                    if not coord_str.strip():
                        continue
                    try:
                        x_str, y_str = coord_str.strip().split(",")
                        x, y = int(x_str), int(y_str)
                        # Long mouseDown hold (1.0s). Earlier experiments showed
                        # 200ms did not rescue some clicks (target app did not
                        # open even when cursor was visibly on the icon).
                        # TCG is roughly 50x slower than KVM, so a 200ms hold
                        # inside TCG may amount to a microscopic interval
                        # against the emulated input pipeline. 1.0s is
                        # comfortably in the TCG order-of-magnitude.
                        env.controller.execute_python_command(
                            "import time as _t\n"
                            f"pyautogui.moveTo({x}, {y}, duration=0.3)\n"
                            "_t.sleep(1.0)\n"
                            "pyautogui.mouseDown()\n"
                            "_t.sleep(1.0)\n"
                            "pyautogui.mouseUp()\n"
                            "_t.sleep(2.0)\n"
                        )
                        obs_after = env._get_obs()  # noqa: SLF001
                        png_after = obs_after.get("screenshot") if isinstance(obs_after, dict) else None
                        if png_after:
                            fname = f"after_click_{i:02d}_x{x}_y{y}.png"
                            with open(os.path.join(cc_dir, fname), "wb") as fh:
                                fh.write(png_after)
                    except Exception:
                        LOG.exception("Controlled click %d (%r) failed", i, coord_str)
            except Exception:
                LOG.exception("OSWORLD_CONTROLLED_CLICK setup failed")

        # Refresh obs so step 0 of the agent loop sees the post-controlled-
        # click state (whatever the click triggered should be visible to
        # the agent now).
        if os.environ.get("OSWORLD_CONTROLLED_CLICK"):
            try:
                obs = env._get_obs()  # noqa: SLF001
            except Exception:
                LOG.exception("Failed to refresh obs after controlled-click")

        for step_idx in range(max_steps):
            if time.monotonic() - task_start > task_timeout:
                error = f"task_timeout exceeded ({task_timeout}s) at step {step_idx}"
                timed_out = True
                LOG.warning(error)
                break
            obs_entry = {
                "screenshot_b64": _b64(obs.get("screenshot")),
                "accessibility_tree": obs.get("accessibility_tree"),
            }
            if os.environ.get("OSWORLD_OMIT_SCREENSHOT_IN_OBS"):
                obs_entry["screenshot_b64"] = ""
            # Opt-in screenshot dump for debug (set OSWORLD_SAVE_SCREENSHOTS_DIR
            # to a path on shared storage). Writes <task_id>-step<NN>.png so
            # we can eyeball what the agent actually saw at each step.
            screenshots_dir = os.environ.get("OSWORLD_SAVE_SCREENSHOTS_DIR")
            if screenshots_dir and obs.get("screenshot"):
                try:
                    os.makedirs(screenshots_dir, exist_ok=True)
                    fname = os.path.join(screenshots_dir, f"{task_config.get('id', 'unknown')}-step{step_idx:02d}.png")
                    with open(fname, "wb") as fh:
                        fh.write(obs["screenshot"])
                except Exception:
                    LOG.exception("Failed to save screenshot for step %d", step_idx)
            # Opt-in obs diag: per-step log of screenshot size + a11y_tree
            # size + head. Verifies whether require_a11y_tree=true actually
            # produces non-empty a11y data — useful when investigating "did
            # the model see DOM info, or did it fall back to vision-only?"
            # (same click coords with vs without a11y suggest the latter).
            obs_diag_log = os.environ.get("OSWORLD_OBS_DIAG_LOG")
            if obs_diag_log:
                try:
                    os.makedirs(os.path.dirname(obs_diag_log), exist_ok=True)
                    a11y = obs.get("accessibility_tree") or ""
                    with open(obs_diag_log, "a") as fh:
                        fh.write(
                            json.dumps(
                                {
                                    "step": step_idx,
                                    "png_bytes": len(obs.get("screenshot") or b""),
                                    "a11y_type": type(a11y).__name__,
                                    "a11y_bytes": len(a11y) if isinstance(a11y, (str, bytes)) else -1,
                                    "a11y_head": (a11y[:300] if isinstance(a11y, str) else repr(a11y)[:300])
                                    if a11y
                                    else "",
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    LOG.exception("Failed to write obs diag for step %d", step_idx)
            # Opt-in: dump the FULL a11y XML per step to a sidecar dir, so
            # we can grep for specific UI elements, check whether per-row
            # menu elements have positions reported, etc. a11y XML is
            # typically 60-150 KB per step for Chrome / file manager.
            a11y_dump_dir = os.environ.get("OSWORLD_A11Y_DUMP_DIR")
            if a11y_dump_dir and obs.get("accessibility_tree"):
                try:
                    os.makedirs(a11y_dump_dir, exist_ok=True)
                    a11y_fpath = os.path.join(
                        a11y_dump_dir,
                        f"{task_config.get('id', 'unknown')}-step{step_idx:02d}.xml",
                    )
                    a11y_data = obs["accessibility_tree"]
                    if isinstance(a11y_data, bytes):
                        a11y_data = a11y_data.decode("utf-8", errors="replace")
                    with open(a11y_fpath, "w") as fh:
                        fh.write(a11y_data)
                except Exception:
                    LOG.exception("Failed to dump a11y XML for step %d", step_idx)
            history_window = obs_history[-max_trajectory_length:] if max_trajectory_length else []

            try:
                if pointer_agent is not None:
                    model_text, actions = pointer_agent.predict(obs)
                    model_text = strip_thinking(model_text or "")
                    actions = _flatten_actions(actions)
                elif native_agent is not None:
                    model_text, actions = native_agent.predict(instruction, obs)
                    model_text = strip_thinking(model_text or "")
                    actions = _flatten_actions(actions)
                    if runner_spec.action_space == "computer_13":
                        actions = [_normalize_prompt_agent_computer_13_action(action) for action in actions]
                else:
                    model_text = model_fn(system_prompt, instruction, history_window + [obs_entry])
                    model_text = strip_thinking(model_text or "")
                    actions = parse_actions(model_text)
            except Exception as exc:  # noqa: BLE001 — record + abort, don't crash the VM.
                error = f"agent/model call failed at step {step_idx}: {exc}"
                LOG.exception("Agent/model call failed at step %d", step_idx)
                steps.append(StepRecord(step=step_idx, model_text="", actions=[], reward=0.0, done=False))
                break

            if not actions:
                # No parseable action — log the step and continue. The model
                # gets another chance next iteration with a fresh screenshot.
                steps.append(StepRecord(step=step_idx, model_text=model_text, actions=[], reward=0.0, done=False))
                obs_history.append(obs_entry)
                continue

            step_done = False
            step_reward = 0.0
            step_info: Dict[str, Any] = {}
            for action in actions:
                try:
                    obs, reward, done, info = env.step(action, sleep_after_execution)
                except Exception as exc:  # noqa: BLE001 - record bad model/controller actions.
                    error = f"env.step() failed at step {step_idx}: {exc}"
                    LOG.exception("Environment step failed at step %d for action %r", step_idx, action)
                    break
                step_reward += float(reward or 0.0)
                step_info = info if isinstance(info, dict) else {"info": info}
                if done:
                    step_done = True
                    break
                if _is_terminal_action(action):
                    step_done = True
                    break

            steps.append(
                StepRecord(
                    step=step_idx,
                    model_text=model_text,
                    actions=actions,
                    reward=step_reward,
                    done=step_done,
                    info=step_info,
                )
            )
            obs_history.append(obs_entry)

            if step_done:
                finished = True
                break
            if error:
                break

        # Let the VM settle before scoring, mirroring lib_run_single.py.
        time.sleep(2)
        try:
            eval_logger = pointer_logger if pointer_agent is not None else LOG
            final_score = _evaluate_osworld_env(env, eval_logger)
        except Exception as exc:  # noqa: BLE001
            error = f"env.evaluate() failed: {exc}"
            LOG.exception("Evaluator failed")
            final_score = 0.0
        if pointer_agent is not None and hasattr(pointer_agent, "log_usage"):
            try:
                pointer_agent.log_usage()
            except Exception:  # noqa: BLE001 - usage logging should not fail the rollout.
                LOG.exception("PointerAgent.log_usage() failed")

    except Exception as exc:  # noqa: BLE001 — top-level guard so caller sees error not crash.
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        LOG.exception("OSWorld rollout failed before evaluation")
    finally:
        if "pointer_log_handler" in locals() and pointer_log_handler is not None:
            try:
                pointer_log_handler.flush()
                pointer_log_handler.close()
                pointer_logger = getattr(pointer_log_handler, "_nemo_gym_logger", None)
                if pointer_logger is not None:
                    pointer_logger.removeHandler(pointer_log_handler)
            except Exception:
                LOG.exception("Pointer log handler close raised")
        if _recording_started and env is not None:
            try:
                os.makedirs(_record_dir, exist_ok=True)
                _task_id = task_config.get("id", "unknown")
                _mp4_path = os.path.join(_record_dir, f"{_task_id}.mp4")
                env.controller.end_recording(_mp4_path)
                LOG.info("Saved VM recording: %s", _mp4_path)
            except Exception:  # noqa: BLE001 — best-effort.
                LOG.exception("end_recording() failed; mp4 may be missing or partial")
        if env is not None:
            try:
                env.close()
            except Exception:
                LOG.exception("env.close() raised")

    reward = 1.0 if final_score >= 1.0 else 0.0
    # mask_sample: reward is unreliable if (a) anything errored, (b) timeout,
    # or (c) loop exhausted max_steps without the model emitting DONE/FAIL.
    mask_sample = bool(error) or timed_out or not finished

    return RolloutResult(
        reward=reward,
        score=final_score,
        steps=steps,
        error=error,
        finished=finished,
        mask_sample=mask_sample,
    )
