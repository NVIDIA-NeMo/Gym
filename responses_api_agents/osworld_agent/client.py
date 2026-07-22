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
import contextvars
import datetime
import hashlib
import inspect
import json
import logging
import os
import re
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Dict, List, Mapping, Optional

from responses_api_agents.osworld_agent.action_parser import parse_actions, strip_thinking
from responses_api_agents.osworld_agent.proxy import inspect_proxy_config_file, task_requires_proxy
from responses_api_agents.osworld_agent.runner_registry import load_attr, resolve_runner_spec


LOG = logging.getLogger("nemo_gym.osworld_agent.client")

SANDBOX_DESKTOP_ENV_CLASS = "responses_api_agents.osworld_agent.sandbox_desktop_env.SandboxDesktopEnv"
SANDBOX_POINTER_DESKTOP_ENV_CLASS = "responses_api_agents.osworld_agent.sandbox_desktop_env.SandboxPointerDesktopEnv"

# Sentinel actions OSWorld recognises in step().
_TERMINAL_ACTIONS = {"DONE", "FAIL"}


class _EvaluatorScoreZero(BaseException):
    """Control signal for declarative evaluator setup that proves a zero score."""


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
    # Absolute path to the per-task log and artifact directory when
    # OSWORLD_TASK_ARTIFACT_ROOT is configured.
    artifact_dir: Optional[str] = None
    termination_reason: Optional[str] = None


@dataclass
class _TaskArtifacts:
    """Per-rollout file handlers and artifact paths.

    Ray reuses worker processes, so every handler installed for a task must be
    removed and closed when that task finishes. Keeping the lifecycle in one
    object prevents cross-task log leakage.
    """

    task_id: str
    directory: str
    trajectory_path: str
    task_logger: logging.Logger
    worker_handler: logging.FileHandler
    runtime_handler: logging.FileHandler
    attached_loggers: List[tuple[logging.Logger, int]]
    started_at: str
    identity: Dict[str, Any]


# `ModelFn` takes (system_prompt, instruction, observation_history) and
# returns the raw model output text. The caller is responsible for invoking
# the model server — we keep this layer agnostic of HTTP / asyncio so it can
# be driven from a Ray actor.
ObservationHistory = List[Dict[str, Any]]
ModelFn = Callable[[str, str, ObservationHistory], str]
MessagesModelFn = Callable[[List[Dict[str, Any]], Dict[str, Any]], Any]


def _b64(screenshot_bytes: Optional[bytes]) -> str:
    if not screenshot_bytes:
        return ""
    return base64.b64encode(screenshot_bytes).decode("ascii")


def _is_terminal_action(action: Any) -> bool:
    if isinstance(action, str) and action in _TERMINAL_ACTIONS:
        return True
    return isinstance(action, dict) and action.get("action_type") in _TERMINAL_ACTIONS


def _terminal_action_name(action: Any) -> Optional[str]:
    if isinstance(action, str) and action in _TERMINAL_ACTIONS:
        return action
    if isinstance(action, dict) and action.get("action_type") in _TERMINAL_ACTIONS:
        return str(action["action_type"])
    return None


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


def _merge_consecutive_pyautogui_actions(actions: List[Any]) -> List[Any]:
    """Execute adjacent Qwen pyautogui calls as one OSWorld step.

    This prevents a compound tool call from adding an observation and delay
    between every individual key or click.
    """

    merged: List[Any] = []
    pending: List[str] = []
    for action in actions:
        if isinstance(action, str) and action.startswith("pyautogui."):
            pending.append(action)
            continue
        if pending:
            merged.append("\n".join(pending))
            pending = []
        merged.append(action)
    if pending:
        merged.append("\n".join(pending))
    return merged


def _model_response_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        return str(response.get("content") or "")
    return str(getattr(response, "content", "") or "")


def _link_if_present(source: str, destination: str) -> bool:
    if not os.path.exists(source) or os.path.lexists(destination):
        return False
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    try:
        os.symlink(os.path.abspath(source), destination, target_is_directory=os.path.isdir(source))
        return True
    except FileExistsError:
        # Multiple rollout workers can stage the same shared cache at once.
        return False


def _walk_cloud_file_configs(value: Any):
    if isinstance(value, Mapping):
        if value.get("type") == "cloud_file":
            yield value
        for child in value.values():
            yield from _walk_cloud_file_configs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_cloud_file_configs(child)


def _readonly_task_cache_names(task_config: Mapping[str, Any]) -> List[str]:
    names: List[str] = []
    action_groups = [task_config.get("config", [])]
    evaluator = task_config.get("evaluator")
    if isinstance(evaluator, Mapping):
        action_groups.append(evaluator.get("postconfig", []))
    for actions in action_groups:
        if not isinstance(actions, list):
            continue
        for setup_item in actions:
            if not isinstance(setup_item, Mapping) or setup_item.get("type") != "download":
                continue
            parameters = setup_item.get("parameters", {})
            files = parameters.get("files", []) if isinstance(parameters, Mapping) else []
            for file_config in files if isinstance(files, list) else []:
                if not isinstance(file_config, Mapping):
                    continue
                url = file_config.get("url")
                destination_path = file_config.get("path")
                if url and destination_path:
                    names.append(f"{uuid.uuid5(uuid.NAMESPACE_URL, url)}_{os.path.basename(destination_path)}")

    if isinstance(evaluator, Mapping):
        for cloud_file in _walk_cloud_file_configs(evaluator):
            destinations = cloud_file.get("dest")
            if not cloud_file.get("multi", False):
                destinations = [destinations]
            if isinstance(destinations, list):
                names.extend(
                    destination for destination in destinations if isinstance(destination, str) and destination
                )
    return list(dict.fromkeys(names))


def _stage_setup_cache(task_config: Dict[str, Any], cache_dir: str, setup_cache_dir: Optional[str] = None) -> int:
    """Expose pre-staged read-only artifacts through OSWorld's task cache.

    Staging before ``env.reset`` preserves OSWorld's existing
    ``SetupController`` flow without modifying the OSWorld checkout.  Only
    cache entries named by setup ``download`` actions are linked.  A shared
    task cache can also contain evaluator outputs such as ``*_pred`` and
    ``*_gold``; linking those mutable directories into a new rollout makes
    archive evaluators fail when they try to recreate their output folders.
    """

    task_id = str(task_config.get("id") or task_config.get("task_id") or "")
    if not task_id:
        return 0
    task_cache_dir = os.path.join(cache_dir, task_id)
    linked = 0

    flat_cache_env: Optional[str] = None
    if "spreadsheetbench" in task_id:
        flat_cache_env = "SPREADSHEETBENCH_SETUP_CACHE_DIR"
    elif "pptc" in task_id:
        flat_cache_env = "PPTC_SETUP_CACHE_DIR"

    setup_cache_names = _readonly_task_cache_names(task_config)

    if setup_cache_dir:
        source_dir = os.path.join(os.path.expanduser(setup_cache_dir), task_id)
    elif flat_cache_env:
        source_dir = os.environ.get(flat_cache_env, "")
    else:
        cache_env = "OW_SETUP_CACHE_DIR" if task_id.startswith("ow-") else "OSWORLD_SETUP_CACHE_DIR"
        source_root = os.environ.get(cache_env, "")
        if not source_root and cache_env == "OSWORLD_SETUP_CACHE_DIR":
            # Also consume prepare.py's standard cache when an older, preserved
            # env.yaml predates the explicit setup_cache_dir config field.
            try:
                from benchmarks.osworld.assets import DEFAULT_SETUP_CACHE

                source_root = str(DEFAULT_SETUP_CACHE)
            except ImportError:
                pass
        source_dir = os.path.join(source_root, task_id) if source_root else ""

    if not os.path.isdir(source_dir):
        return 0
    for name in setup_cache_names:
        linked += int(_link_if_present(os.path.join(source_dir, name), os.path.join(task_cache_dir, name)))
    return linked


def _patch_setup_execute_contract() -> None:
    """Support optional return-code policies used by newer OSWorld tasks.

    The pinned upstream method remains untouched for existing tasks. The
    compatibility path runs only when a task explicitly supplies
    ``expected_returncodes`` or ``on_nonzero``.
    """

    try:
        from desktop_env.controllers import setup as setup_module  # type: ignore
    except Exception:  # noqa: BLE001 - OSWorld is optional outside the runtime.
        return

    controller_class = setup_module.SetupController
    current = controller_class._execute_setup
    if getattr(current, "_nemo_gym_returncode_contract", False):
        return
    try:
        parameters = inspect.signature(current).parameters
    except (TypeError, ValueError):
        parameters = {}
    if {"expected_returncodes", "on_nonzero"}.issubset(parameters):
        return

    requests = setup_module.requests

    def execute_setup(
        self: Any,
        command: List[str] | str,
        stdout: str = "",
        stderr: str = "",
        shell: bool = False,
        until: Optional[Dict[str, Any]] = None,
        expected_returncodes: List[int] | int | None = None,
        on_nonzero: str | None = None,
    ) -> Any:
        if expected_returncodes is None and on_nonzero is None:
            return current(self, command, stdout=stdout, stderr=stderr, shell=shell, until=until)
        if not command:
            raise RuntimeError("Empty setup command")
        if expected_returncodes is None:
            expected_returncodes = [int(until["returncode"])] if until and "returncode" in until else [0]
        elif isinstance(expected_returncodes, int):
            expected_returncodes = [expected_returncodes]
        allowed = {int(code) for code in expected_returncodes}
        if not allowed:
            raise ValueError("expected_returncodes must not be empty")
        if on_nonzero not in {None, "score_zero"}:
            raise ValueError(f"unsupported on_nonzero policy: {on_nonzero!r}")

        replacements = {
            "{CLIENT_PASSWORD}": self.client_password,
            "{SCREEN_WIDTH_HALF}": str(self.screen_width // 2),
            "{SCREEN_HEIGHT_HALF}": str(self.screen_height // 2),
            "{SCREEN_WIDTH}": str(self.screen_width),
            "{SCREEN_HEIGHT}": str(self.screen_height),
        }
        rendered = [command] if isinstance(command, str) else list(command)
        for index, item in enumerate(rendered):
            for old, new in replacements.items():
                item = item.replace(old, new)
            rendered[index] = item
        rendered_command: List[str] | str = rendered[0] if isinstance(command, str) else rendered
        payload = json.dumps({"command": rendered_command, "shell": shell})
        headers = {"Content-Type": "application/json"}
        until = until or {}
        failures = 0

        while True:
            result = None
            try:
                response = requests.post(
                    self.http_server + "/setup/execute",
                    headers=headers,
                    data=payload,
                    timeout=130,
                )
                if response.status_code == 200:
                    result = response.json()
                    if "returncode" not in result:
                        raise RuntimeError("setup response omitted returncode")
                    if stdout:
                        with open(os.path.join(self.cache_dir, stdout), "w", encoding="utf-8") as handle:
                            handle.write(result.get("output", ""))
                    if stderr:
                        with open(os.path.join(self.cache_dir, stderr), "w", encoding="utf-8") as handle:
                            handle.write(result.get("error", ""))
                else:
                    failures += 1
            except requests.exceptions.RequestException:
                failures += 1
            if failures >= 5:
                raise RuntimeError(f"setup command failed after five request attempts: {rendered_command!r}")
            if result is None:
                continue

            returncode = int(result["returncode"])
            command_text = " ".join(rendered_command) if isinstance(rendered_command, list) else rendered_command
            if returncode not in allowed:
                if on_nonzero == "score_zero" and returncode not in {126, 127}:
                    raise _EvaluatorScoreZero(
                        f"evaluator command established score zero with return code {returncode}: {command_text}"
                    )
                raise RuntimeError(
                    f"setup command returned {returncode}; expected {sorted(allowed)}: {command_text}; "
                    f"stdout={result.get('output', '')!r}; stderr={result.get('error', '')!r}"
                )
            if (
                not until
                or ("returncode" in until and returncode == int(until["returncode"]))
                or ("stdout" in until and str(until["stdout"]) in result.get("output", ""))
                or ("stderr" in until and str(until["stderr"]) in result.get("error", ""))
            ):
                return result
            time.sleep(0.3)

    execute_setup._nemo_gym_returncode_contract = True  # type: ignore[attr-defined]
    controller_class._execute_setup = execute_setup


def _configure_docker_port_lock_timeout(timeout: float) -> None:
    """Set the pinned upstream Docker provider's global allocation lock wait."""

    if timeout <= 0:
        raise ValueError("docker_port_lock_timeout must be positive")
    try:
        from desktop_env.providers.docker import provider as docker_provider  # type: ignore
    except Exception:  # noqa: BLE001 - OSWorld is optional outside the runtime.
        return
    if hasattr(docker_provider, "LOCK_TIMEOUT"):
        docker_provider.LOCK_TIMEOUT = float(timeout)


def _patch_extension_name_aliases() -> None:
    """Normalize a renamed Chrome extension for stable task evaluation."""

    try:
        from desktop_env.evaluators import metrics as metrics_package  # type: ignore
        from desktop_env.evaluators.metrics import chrome as chrome_metrics  # type: ignore
    except Exception:  # noqa: BLE001 - OSWorld is optional outside the runtime.
        return

    current = chrome_metrics.is_expected_installed_extensions
    if getattr(current, "_nemo_gym_alias_patch", False):
        return

    aliases = {
        "Speechify — Text to Speech": "Speechify — Voice AI Assistant",
    }

    def canonicalize(name: Any) -> Any:
        return aliases.get(name, name)

    def wrapped(installed_extensions: Any, expected: Any) -> float:
        installed = (
            [canonicalize(name) for name in installed_extensions] if installed_extensions else installed_extensions
        )
        normalized_expected = expected
        if isinstance(expected, dict) and isinstance(expected.get("expected"), list):
            normalized_expected = dict(expected)
            normalized_expected["expected"] = [canonicalize(name) for name in expected["expected"]]
        return current(installed, normalized_expected)

    wrapped._nemo_gym_alias_patch = True  # type: ignore[attr-defined]
    chrome_metrics.is_expected_installed_extensions = wrapped
    metrics_package.is_expected_installed_extensions = wrapped


def _patch_pdf_image_evaluator_cleanup() -> None:
    """Make corrupt PDFs score zero instead of failing during double cleanup.

    Some OSWorld releases remove ``temp_pdf_comparison`` in both the exception
    handler and ``finally`` block. The second removal raises FileNotFoundError,
    masking an otherwise valid evaluator score of zero. Keep this adapter-side
    compatibility shim narrow so unrelated missing-file failures still surface.
    """

    try:
        from desktop_env.evaluators import metrics as metrics_package  # type: ignore
        from desktop_env.evaluators.metrics import chrome as chrome_metrics  # type: ignore
    except Exception:  # noqa: BLE001 - OSWorld is optional outside the runtime.
        return

    current = getattr(chrome_metrics, "compare_pdf_images", None)
    if current is None or getattr(current, "_nemo_gym_pdf_cleanup_patch", False):
        return

    def wrapped(*args: Any, **kwargs: Any) -> float:
        try:
            return float(current(*args, **kwargs))
        except FileNotFoundError as exc:
            missing_path = str(getattr(exc, "filename", "") or "")
            if os.path.basename(missing_path) != "temp_pdf_comparison":
                raise
            LOG.warning(
                "OSWorld compare_pdf_images repeated cleanup for %s; preserving its zero score",
                missing_path,
            )
            return 0.0

    wrapped._nemo_gym_pdf_cleanup_patch = True  # type: ignore[attr-defined]
    chrome_metrics.compare_pdf_images = wrapped
    metrics_package.compare_pdf_images = wrapped


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


@dataclass
class _PointerModelIOContext:
    """Task-scoped destination and identity for Pointer's direct model calls."""

    log_path: str
    identity: Dict[str, Any]
    step_ref: List[int]


_POINTER_MODEL_IO_CONTEXT: contextvars.ContextVar[Optional[_PointerModelIOContext]] = contextvars.ContextVar(
    "pointer_model_io_context",
    default=None,
)
_POINTER_SECRET_KEYS = {
    "api_key",
    "authorization",
    "cookie",
    "password",
    "proxy_authorization",
    "set_cookie",
    "token",
    "x_api_key",
}


def _pointer_io_jsonable(value: Any) -> Any:
    """Convert Anthropic SDK values into JSON-compatible data."""

    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()
        except Exception:  # noqa: BLE001 - observability must not affect a rollout.
            return repr(value)
    if isinstance(value, Mapping):
        return {str(key): _pointer_io_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pointer_io_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _redact_pointer_io_secrets(value: Any) -> Any:
    """Remove transport credentials while retaining the complete model body."""

    if isinstance(value, dict):
        redacted: Dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).lower().replace("-", "_")
            redacted[str(key)] = (
                "<redacted>" if normalized_key in _POINTER_SECRET_KEYS else _redact_pointer_io_secrets(item)
            )
        return redacted
    if isinstance(value, list):
        return [_redact_pointer_io_secrets(item) for item in value]
    return value


def _pointer_io_sha256(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _pointer_io_images(request: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Index Anthropic base64 image parts without removing their source data."""

    kwargs = request.get("kwargs")
    messages = kwargs.get("messages") if isinstance(kwargs, Mapping) else None
    if not isinstance(messages, list):
        return []
    images: List[Dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if not isinstance(part, Mapping) or part.get("type") != "image":
                continue
            source = part.get("source")
            encoded = source.get("data") if isinstance(source, Mapping) else None
            if not isinstance(encoded, str):
                continue
            try:
                decoded = base64.b64decode(encoded, validate=False)
            except Exception:  # noqa: BLE001 - malformed diagnostic data must not affect inference.
                decoded = b""
            images.append(
                {
                    "message_index": message_index,
                    "part_index": part_index,
                    "encoded_chars": len(encoded),
                    "encoded_sha256": hashlib.sha256(encoded.encode("ascii", errors="ignore")).hexdigest(),
                    "decoded_bytes": len(decoded),
                    "decoded_sha256": hashlib.sha256(decoded).hexdigest(),
                }
            )
    return images


def _append_pointer_io_event(context: _PointerModelIOContext, event: Dict[str, Any]) -> None:
    """Append one durable schema-v2 event without changing rollout behavior."""

    record = {
        **context.identity,
        "schema_version": 2,
        "event_id": uuid.uuid4().hex,
        "timestamp_unix_ns": time.time_ns(),
        "pid": os.getpid(),
        **event,
    }
    try:
        parent = os.path.dirname(context.log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(context.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_pointer_io_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:  # noqa: BLE001 - model-I/O logging is strictly best-effort.
        LOG.exception("Failed to append Pointer model-I/O log to %s", context.log_path)


class _PointerMessagesProxy:
    """Log exact Anthropic Messages calls while preserving SDK behavior."""

    def __init__(
        self,
        target: Any,
        *,
        context: _PointerModelIOContext,
        agent_role: str,
        api_surface: str,
        call_index_ref: List[int],
    ) -> None:
        self._target = target
        self._context = context
        self._agent_role = agent_role
        self._api_surface = api_surface
        self._call_index_ref = call_index_ref

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self._call_index_ref[0] += 1
        call_index = self._call_index_ref[0]
        call_id = uuid.uuid4().hex
        identity = {
            "call_id": call_id,
            "call_index": call_index,
            "agent_role": self._agent_role,
            "api_surface": self._api_surface,
            "step": self._context.step_ref[0],
        }
        started_ns = time.time_ns()
        try:
            request = _redact_pointer_io_secrets(
                _pointer_io_jsonable(
                    {
                        "args": args,
                        "kwargs": kwargs,
                    }
                )
            )
            _append_pointer_io_event(
                self._context,
                {
                    **identity,
                    "event": "model_request",
                    "timestamp_unix_ns": started_ns,
                    "anthropic_request": request,
                    "anthropic_request_sha256": _pointer_io_sha256(request),
                    "embedded_images": _pointer_io_images(request),
                },
            )
        except Exception:  # noqa: BLE001 - logging must not change the SDK call.
            LOG.exception("Failed to serialize Pointer model request for call %s", call_id)
        try:
            response = self._target.create(*args, **kwargs)
        except Exception as exc:
            finished_ns = time.time_ns()
            try:
                _append_pointer_io_event(
                    self._context,
                    {
                        **identity,
                        "event": "model_error",
                        "timestamp_unix_ns": finished_ns,
                        "elapsed_ns": finished_ns - started_ns,
                        "error_type": type(exc).__name__,
                        "error": repr(exc),
                    },
                )
            except Exception:  # noqa: BLE001 - preserve the provider exception.
                LOG.exception("Failed to serialize Pointer model error for call %s", call_id)
            raise
        finished_ns = time.time_ns()
        try:
            response_value = _redact_pointer_io_secrets(_pointer_io_jsonable(response))
            _append_pointer_io_event(
                self._context,
                {
                    **identity,
                    "event": "model_response",
                    "timestamp_unix_ns": finished_ns,
                    "elapsed_ns": finished_ns - started_ns,
                    "anthropic_response": response_value,
                    "anthropic_response_sha256": _pointer_io_sha256(response_value),
                },
            )
        except Exception:  # noqa: BLE001 - return the SDK response unchanged.
            LOG.exception("Failed to serialize Pointer model response for call %s", call_id)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _PointerBetaProxy:
    def __init__(
        self,
        target: Any,
        *,
        context: _PointerModelIOContext,
        agent_role: str,
        call_index_ref: List[int],
    ) -> None:
        self._target = target
        self.messages = _PointerMessagesProxy(
            target.messages,
            context=context,
            agent_role=agent_role,
            api_surface="beta.messages",
            call_index_ref=call_index_ref,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _PointerAnthropicClientProxy:
    def __init__(self, target: Any, *, context: _PointerModelIOContext, agent_role: str) -> None:
        self._target = target
        call_index_ref = [0]
        self.messages = _PointerMessagesProxy(
            target.messages,
            context=context,
            agent_role=agent_role,
            api_surface="messages",
            call_index_ref=call_index_ref,
        )
        if hasattr(target, "beta"):
            self.beta = _PointerBetaProxy(
                target.beta,
                context=context,
                agent_role=agent_role,
                call_index_ref=call_index_ref,
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def _pointer_model_io_context(
    event_context: Mapping[str, Any],
    *,
    endpoint: str,
    served_model: str,
    step_ref: List[int],
) -> Optional[_PointerModelIOContext]:
    """Build Pointer logging state only when full model-I/O is enabled."""

    configured_path = os.environ.get("OSWORLD_MODEL_IO_LOG", "").strip()
    if not configured_path:
        return None
    identity = {
        **dict(event_context),
        "adapter": "gym",
        "endpoint": endpoint,
        "served_model": served_model,
        "source_commit": os.environ.get("NEMO_GYM_SOURCE_COMMIT", ""),
    }
    return _PointerModelIOContext(
        log_path=os.path.abspath(os.path.expanduser(configured_path)),
        identity={key: value for key, value in identity.items() if value is not None and value != ""},
        step_ref=step_ref,
    )


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

            client = Anthropic(**_pointer_anthropic_client_options(base_url, self.api_key))
            context = _POINTER_MODEL_IO_CONTEXT.get()
            if context is None:
                return client
            return _PointerAnthropicClientProxy(
                client,
                context=context,
                agent_role=str(getattr(self, "name", type(self).__name__)),
            )
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


def _safe_artifact_component(value: Any, *, fallback: str = "unknown") -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or fallback)).strip("._")
    return component[:120] or fallback


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        fh.write("\n")


def _setup_task_artifacts(
    task_config: Dict[str, Any],
    *,
    run_metadata: Dict[str, Any],
    event_context: Optional[Mapping[str, Any]] = None,
) -> Optional[_TaskArtifacts]:
    """Create a log and artifact directory for one rollout.

    The Python boundary is opt-in; the multienv launch script enables it by
    default. A collision-safe directory supports ``num_repeats > 1`` and
    reruns that intentionally reuse the same result root.
    """

    artifact_root = os.environ.get("OSWORLD_TASK_ARTIFACT_ROOT", "").strip()
    if not artifact_root:
        return None

    task_id = _safe_task_id(task_config)
    domain = _safe_artifact_component(
        task_config.get("domain")
        or task_config.get("snapshot")
        or next(iter(task_config.get("related_apps") or []), None),
    )
    parent_dir = os.path.abspath(os.path.expanduser(os.path.join(artifact_root, domain)))
    base_dir = os.path.join(parent_dir, task_id)
    worker_handler: Optional[logging.FileHandler] = None
    runtime_handler: Optional[logging.FileHandler] = None
    attached_loggers: List[tuple[logging.Logger, int]] = []

    try:
        os.makedirs(parent_dir, exist_ok=True)
        artifact_dir = base_dir
        try:
            os.mkdir(artifact_dir)
        except FileExistsError:
            suffix = (
                f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
                f"-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            )
            artifact_dir = f"{base_dir}--{suffix}"
            os.mkdir(artifact_dir)

        log_format = "[%(asctime)s %(levelname)s %(name)s/%(lineno)d pid=%(process)d] %(message)s"
        formatter = logging.Formatter(log_format)
        worker_handler = logging.FileHandler(os.path.join(artifact_dir, "worker.log"), encoding="utf-8")
        worker_handler.setLevel(logging.DEBUG)
        worker_handler.setFormatter(formatter)
        runtime_handler = logging.FileHandler(os.path.join(artifact_dir, "runtime.log"), encoding="utf-8")
        runtime_handler.setLevel(logging.DEBUG)
        runtime_handler.setFormatter(formatter)

        # The adapter and native agent use nemo_gym.osworld_agent; upstream
        # OSWorld uses both desktopenv and desktop_env across versions.
        for logger_name in ("nemo_gym.osworld_agent", "desktopenv", "desktop_env", "mm_agents"):
            artifact_logger = logging.getLogger(logger_name)
            attached_loggers.append((artifact_logger, artifact_logger.level))
            artifact_logger.setLevel(logging.DEBUG)
            artifact_logger.addHandler(worker_handler)
            if logger_name == "nemo_gym.osworld_agent":
                artifact_logger.addHandler(runtime_handler)

        logger_name = f"nemo_gym.osworld_agent.task.{task_id}.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        task_logger = logging.getLogger(logger_name)
        task_logger.setLevel(logging.DEBUG)
        task_logger.propagate = True

        started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        identity = {
            "adapter": "gym",
            "task_id": task_id,
            "domain": domain,
            **dict(event_context or {}),
        }
        identity = {key: value for key, value in identity.items() if value is not None and value != ""}
        trajectory_path = os.path.join(artifact_dir, "traj.jsonl")
        with open(trajectory_path, "w", encoding="utf-8"):
            pass
        _write_json(os.path.join(artifact_dir, "task.json"), task_config)
        _write_json(
            os.path.join(artifact_dir, "run.json"),
            {
                **identity,
                "started_at": started_at,
                "pid": os.getpid(),
                **run_metadata,
            },
        )
        context = _TaskArtifacts(
            task_id=task_id,
            directory=artifact_dir,
            trajectory_path=trajectory_path,
            task_logger=task_logger,
            worker_handler=worker_handler,
            runtime_handler=runtime_handler,
            attached_loggers=attached_loggers,
            started_at=started_at,
            identity=identity,
        )
        task_logger.info("Created per-task artifact directory: %s", artifact_dir)
        return context
    except Exception:  # noqa: BLE001 - observability must not fail a rollout.
        for artifact_logger, previous_level in reversed(attached_loggers):
            if worker_handler is not None:
                artifact_logger.removeHandler(worker_handler)
            if runtime_handler is not None:
                artifact_logger.removeHandler(runtime_handler)
            artifact_logger.setLevel(previous_level)
        for handler in (runtime_handler, worker_handler):
            if handler is not None:
                handler.close()
        LOG.exception("Failed to initialize per-task OSWorld artifacts")
        return None


def _append_task_trajectory(
    artifacts: Optional[_TaskArtifacts],
    payload: Dict[str, Any],
) -> None:
    if artifacts is None:
        return
    record = {
        "schema_version": 2,
        **artifacts.identity,
        "event_id": uuid.uuid4().hex,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **payload,
    }
    try:
        with open(artifacts.trajectory_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001 - artifact I/O must not fail a rollout.
        artifacts.task_logger.exception("Failed to append task trajectory")


def _save_task_screenshot(
    artifacts: Optional[_TaskArtifacts],
    step_num: int,
    obs: Mapping[str, Any],
) -> Optional[str]:
    if artifacts is None:
        return None
    screenshot = obs.get("screenshot")
    if not isinstance(screenshot, (bytes, bytearray)) or not screenshot:
        return None
    filename = f"step_{step_num:03d}.png"
    try:
        with open(os.path.join(artifacts.directory, filename), "wb") as fh:
            fh.write(screenshot)
        return filename
    except Exception:  # noqa: BLE001
        artifacts.task_logger.exception("Failed to save screenshot for artifact step %d", step_num)
        return None


def _observation_identity(obs: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a compact join key for the screenshot without duplicating bytes."""

    screenshot = obs.get("screenshot")
    if not isinstance(screenshot, (bytes, bytearray)) or not screenshot:
        return {"screenshot_bytes": 0, "screenshot_sha256": None}
    screenshot_bytes = bytes(screenshot)
    return {
        "screenshot_bytes": len(screenshot_bytes),
        "screenshot_sha256": hashlib.sha256(screenshot_bytes).hexdigest(),
    }


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluator_result_destinations(value: Any) -> List[str]:
    """Collect evaluator result cache destinations from nested task specs."""

    if isinstance(value, list):
        destinations: List[str] = []
        for item in value:
            destinations.extend(_evaluator_result_destinations(item))
        return destinations
    if not isinstance(value, dict):
        return []
    destination = value.get("dest")
    if isinstance(destination, str):
        return [destination]
    if isinstance(destination, list):
        return [item for item in destination if isinstance(item, str)]
    return []


def _evaluator_result_artifacts(task_config: Mapping[str, Any], cache_dir: str) -> List[Dict[str, Any]]:
    """Record compact result-file evidence after evaluation.

    The evaluator cache remains on the execution host. Only destination,
    existence, size, and a bounded-file hash enter the textual trajectory.
    """

    evaluator = task_config.get("evaluator")
    if not isinstance(evaluator, dict):
        return []
    destinations = _evaluator_result_destinations(evaluator.get("result"))
    if not destinations:
        return []
    hash_limit = int(os.environ.get("OSWORLD_EVALUATOR_HASH_MAX_BYTES", str(16 * 1024 * 1024)))
    task_cache_dir = os.path.abspath(os.path.expanduser(os.path.join(cache_dir, _safe_task_id(dict(task_config)))))
    records: List[Dict[str, Any]] = []
    for destination in dict.fromkeys(destinations):
        candidate = os.path.abspath(os.path.join(task_cache_dir, destination))
        within_task_cache = os.path.commonpath([task_cache_dir, candidate]) == task_cache_dir
        record: Dict[str, Any] = {
            "destination": destination,
            "cache_path": candidate,
            "within_task_cache": within_task_cache,
            "exists": within_task_cache and os.path.exists(candidate),
            "is_file": within_task_cache and os.path.isfile(candidate),
        }
        if record["is_file"]:
            size = os.path.getsize(candidate)
            record["bytes"] = size
            if size <= hash_limit:
                record["sha256"] = _sha256_file(candidate)
        records.append(record)
    return records


def _finalize_task_artifacts(
    artifacts: Optional[_TaskArtifacts],
    *,
    result: RolloutResult,
    duration_seconds: float,
    recording_path: Optional[str],
) -> None:
    if artifacts is None:
        return

    try:
        artifacts.task_logger.info(
            "OSWorld rollout finished: score=%s reward=%s finished=%s mask_sample=%s error=%r",
            result.score,
            result.reward,
            result.finished,
            result.mask_sample,
            result.error,
        )
        _write_json(
            os.path.join(artifacts.directory, "result.json"),
            {
                "task_id": artifacts.task_id,
                "started_at": artifacts.started_at,
                "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "duration_seconds": duration_seconds,
                "reward": result.reward,
                "score": result.score,
                "finished": result.finished,
                "mask_sample": result.mask_sample,
                "error": result.error,
                "termination_reason": result.termination_reason,
                "step_count": len(result.steps),
                "recording_path": recording_path,
            },
        )
        if recording_path and os.path.isfile(recording_path):
            recording_link = os.path.join(artifacts.directory, "recording.mp4")
            if not os.path.lexists(recording_link):
                os.symlink(os.path.relpath(recording_path, artifacts.directory), recording_link)

        artifacts.worker_handler.flush()
        artifacts.runtime_handler.flush()
        files = []
        for name in sorted(os.listdir(artifacts.directory)):
            path = os.path.join(artifacts.directory, name)
            if os.path.isfile(path) and not os.path.islink(path):
                files.append({"path": name, "bytes": os.path.getsize(path), "sha256": _sha256_file(path)})
            elif os.path.islink(path):
                files.append({"path": name, "symlink": os.readlink(path)})
        _write_json(
            os.path.join(artifacts.directory, "manifest.json"),
            {
                "task_id": artifacts.task_id,
                "artifact_dir": artifacts.directory,
                "recording_path": recording_path,
                "files": files,
            },
        )
    except Exception:  # noqa: BLE001
        artifacts.task_logger.exception("Failed to finalize per-task OSWorld artifacts")
    finally:
        for artifact_logger, previous_level in reversed(artifacts.attached_loggers):
            artifact_logger.removeHandler(artifacts.worker_handler)
            artifact_logger.removeHandler(artifacts.runtime_handler)
            artifact_logger.setLevel(previous_level)
        artifacts.runtime_handler.flush()
        artifacts.worker_handler.flush()
        artifacts.runtime_handler.close()
        artifacts.worker_handler.close()


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


def _evaluate_osworld_env(
    env: Any,
    eval_logger: logging.Logger,
    *,
    disable_gpu: bool = True,
) -> float:
    """Call the OSWorld evaluator across DesktopEnv variants.

    When requested, temporarily hide CUDA and force EasyOCR onto CPU so the
    inline evaluator does not reserve GPU memory needed by rollout workers.
    """

    evaluate = env.evaluate
    original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    easyocr_module: Optional[Any] = None
    original_easyocr_reader: Optional[Any] = None
    if disable_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            easyocr_module = import_module("easyocr")
            original_easyocr_reader = easyocr_module.Reader

            def cpu_reader(*args: Any, **kwargs: Any) -> Any:
                kwargs["gpu"] = False
                return original_easyocr_reader(*args, **kwargs)

            easyocr_module.Reader = cpu_reader
        except Exception:  # noqa: BLE001 - not every OSWorld task imports EasyOCR.
            easyocr_module = None
            original_easyocr_reader = None
    try:
        try:
            params = inspect.signature(evaluate).parameters
        except (TypeError, ValueError):
            params = {}
        try:
            if not params:
                return float(evaluate())
            return float(evaluate(eval_logger))
        except _EvaluatorScoreZero as exc:
            eval_logger.info("OSWorld evaluator setup established score zero: %s", exc)
            return 0.0
    finally:
        if disable_gpu:
            if easyocr_module is not None and original_easyocr_reader is not None:
                easyocr_module.Reader = original_easyocr_reader
            if original_cuda_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible


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
    enable_proxy: bool = False,
    proxy_config_file: Optional[str] = None,
    resources_server_url: str = "",
    resources_server_auth_token: str = "",
    resources_request_timeout: float = 900.0,
    resources_connect_timeout: float = 10.0,
    resources_request_retries: int = 3,
    sandbox_provider_config: Optional[Dict[str, Any]] = None,
    sandbox_spec: Optional[Dict[str, Any]] = None,
    vm_path: Optional[str] = None,
    sandbox_vm_path: Optional[str] = None,
    sandbox_require_kvm: bool = True,
    sandbox_ready_timeout_s: float = 600.0,
    sandbox_ready_poll_s: float = 2.0,
    max_steps: int = 15,
    max_trajectory_length: int = 3,
    sleep_after_execution: float = 0.5,
    system_prompt: Optional[str] = None,
    cache_dir: str = "cache",
    setup_cache_dir: Optional[str] = None,
    mem_limit_mb: int = 0,
    step_timeout: int = 60,  # advisory; per-action subprocess timeout (provider-dependent)
    task_timeout: int = 1800,  # wall-clock cap on the whole rollout
    docker_port_lock_timeout: float = 300.0,
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
    evaluator_disable_gpu: bool = True,
    reward_mode: str = "binary",
    log_context: Optional[Mapping[str, Any]] = None,
) -> RolloutResult:
    """Run a single OSWorld task and return a structured result.

    Heavy imports (``desktop_env``) happen inside the function so importing
    this module on a machine without OSWorld installed still works — only
    actually *running* a rollout requires it.
    """
    if reward_mode not in {"raw", "binary"}:
        raise ValueError(f"Unsupported reward_mode: {reward_mode!r}")

    def proxy_precondition_failure(reason: str, message: str) -> RolloutResult:
        LOG.error("OSWorld proxy precondition failed for task %s: %s", _safe_task_id(task_config), message)
        return RolloutResult(
            reward=0.0,
            score=0.0,
            steps=[],
            error=message,
            finished=False,
            mask_sample=True,
            termination_reason=reason,
        )

    try:
        requires_proxy = task_requires_proxy(task_config)
    except ValueError as exc:
        return proxy_precondition_failure("proxy_configuration_error", f"ProxyConfigurationError: {exc}")
    use_remote_resources = bool(resources_server_url.strip())
    use_gym_sandbox = sandbox_provider_config is not None
    if use_remote_resources and use_gym_sandbox:
        raise ValueError("resources_server and sandbox_provider are mutually exclusive OSWorld backends")
    if vm_path and sandbox_vm_path and os.path.realpath(vm_path) != os.path.realpath(sandbox_vm_path):
        raise ValueError("vm_path and deprecated sandbox_vm_path refer to different qcow2 files")
    effective_vm_path = vm_path or sandbox_vm_path
    if use_remote_resources and effective_vm_path:
        raise ValueError("vm_path is only valid for local OSWorld providers")
    proxy_info = None
    if requires_proxy and enable_proxy and not use_remote_resources:
        try:
            proxy_info = inspect_proxy_config_file(proxy_config_file)
        except ValueError as exc:
            return proxy_precondition_failure("proxy_configuration_error", f"ProxyConfigurationError: {exc}")
        # OSWorld reads this lazily during DesktopEnv.reset(). Set it before
        # loading the environment class so both current and older runtimes see
        # the same explicit configuration.
        os.environ["PROXY_CONFIG_FILE"] = proxy_info.path

    # Keep the requested Docker image visible to future/custom providers. The
    # clean upstream main provider currently starts happysixd/osworld-docker
    # directly and therefore ignores this environment variable.
    os.environ["OSWORLD_DOCKER_IMAGE"] = container_image
    if mem_limit_mb > 0:
        LOG.warning(
            "mem_limit_mb=%d is not enforced by the clean upstream Docker provider; "
            "configure Docker/QEMU resources on the rollout host instead",
            mem_limit_mb,
        )

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
    if use_remote_resources:
        from responses_api_agents.osworld_agent.remote_environment import RemoteDesktopEnv

        env_cls = RemoteDesktopEnv
    else:
        if use_gym_sandbox:
            sandbox_env_class = {
                "desktop_env.desktop_env.DesktopEnv": SANDBOX_DESKTOP_ENV_CLASS,
                "desktop_env.desktop_env_pointer.DesktopEnv": SANDBOX_POINTER_DESKTOP_ENV_CLASS,
            }.get(runner_spec.env_class_path)
            if sandbox_env_class is None and env_class_path is None:
                raise ValueError(
                    "This runner uses a specialized DesktopEnv; set an explicit Gym Sandbox-compatible "
                    "env_class_path before enabling sandbox_provider"
                )
            env_cls = load_attr(sandbox_env_class or runner_spec.env_class_path)
        else:
            env_cls = load_attr(runner_spec.env_class_path)
        _patch_setup_execute_contract()
        if provider_name == "docker" and not use_gym_sandbox:
            _configure_docker_port_lock_timeout(docker_port_lock_timeout)
    instruction = task_config.get("instruction", "")
    event_context = dict(log_context or {})
    event_context.update(
        {
            "adapter": "gym",
            "task_id": _safe_task_id(task_config),
            "domain": task_config.get("domain")
            or task_config.get("snapshot")
            or next(iter(task_config.get("related_apps") or []), None),
        }
    )
    event_context = {key: value for key, value in event_context.items() if value is not None and value != ""}
    if not use_remote_resources:
        _patch_extension_name_aliases()
        _patch_pdf_image_evaluator_cleanup()

    env: Optional[Any] = None
    steps: List[StepRecord] = []
    obs_history: ObservationHistory = []
    error: Optional[str] = None
    finished = False
    final_score = 0.0
    timed_out = False
    setup_score_zero = False
    agent_terminal_action: Optional[str] = None
    evaluation_error: Optional[str] = None
    proxy_setup_error = False
    rollout_phase = "before_environment"
    task_start = time.monotonic()
    recording_path: Optional[str] = None
    task_artifacts = _setup_task_artifacts(
        task_config,
        event_context=event_context,
        run_metadata={
            "runner_name": runner_spec.name,
            "runner_kind": runner_spec.kind,
            "action_space": runner_spec.action_space,
            "observation_type": runner_spec.observation_type,
            "provider_name": provider_name,
            "container_image": container_image,
            "execution_backend": "gym_sandbox"
            if use_gym_sandbox
            else ("resources_server" if use_remote_resources else "osworld_provider"),
            "sandbox_provider": (next(iter(sandbox_provider_config)) if sandbox_provider_config else None),
            "sandbox_image": (sandbox_spec or {}).get("image") if use_gym_sandbox else None,
            "headless": headless,
            "screen_size": list(screen_size),
            "max_steps": max_steps,
            "max_trajectory_length": max_trajectory_length,
            "sleep_after_execution": sleep_after_execution,
            "task_timeout": task_timeout,
            "model_name": policy_model_name,
            "max_tokens": policy_max_tokens,
            "temperature": policy_temperature,
            "top_p": policy_top_p,
            "reward_mode": reward_mode,
            "proxy_required": requires_proxy,
            "proxy_enabled": enable_proxy,
            "proxy_config_file": proxy_info.path if proxy_info is not None else None,
            "proxy_config_sha256": proxy_info.sha256 if proxy_info is not None else None,
            "proxy_config_entry_count": proxy_info.entry_count if proxy_info is not None else 0,
            "resources_server_url": resources_server_url or None,
        },
    )
    task_logger = task_artifacts.task_logger if task_artifacts is not None else LOG
    task_logger.info(
        "Starting OSWorld rollout task_id=%s runner=%s model=%s",
        _safe_task_id(task_config),
        runner_spec.name,
        policy_model_name or "<unset>",
    )

    # Opt-in mp4 recording of the entire rollout, captured server-side inside
    # the VM and pulled back on env.close(). Saved at
    # ${OSWORLD_RECORD_VIDEO_DIR}/{task_id}.mp4.
    _record_dir = os.environ.get("OSWORLD_RECORD_VIDEO_DIR", "")
    _recording_started = False
    _current_step = [0]
    pointer_io_context_token: Optional[contextvars.Token[_PointerModelIOContext | None]] = None

    try:
        rollout_phase = "environment_start"
        env_kwargs: Dict[str, Any] = {
            "provider_name": provider_name,
            "action_space": runner_spec.action_space,
            "screen_size": screen_size,
            "headless": headless,
            "require_a11y_tree": require_a11y_tree
            or runner_spec.observation_type in {"a11y_tree", "screenshot_a11y_tree", "som"},
            "os_type": "Ubuntu",
            "client_password": client_password,
            "cache_dir": cache_dir,
            "enable_proxy": enable_proxy,
        }
        if use_remote_resources:
            env_kwargs.update(
                {
                    "resources_server_url": resources_server_url,
                    "auth_token": resources_server_auth_token,
                    "request_timeout": resources_request_timeout,
                    "connect_timeout": resources_connect_timeout,
                    "request_retries": resources_request_retries,
                }
            )
        elif use_gym_sandbox:
            effective_sandbox_spec = dict(sandbox_spec or {})
            effective_sandbox_spec.setdefault("image", container_image)
            env_kwargs.update(
                {
                    "sandbox_provider": dict(sandbox_provider_config or {}),
                    "sandbox_spec": effective_sandbox_spec,
                    "sandbox_require_kvm": sandbox_require_kvm,
                    "sandbox_ready_timeout_s": sandbox_ready_timeout_s,
                    "sandbox_ready_poll_s": sandbox_ready_poll_s,
                }
            )
        if not use_remote_resources and effective_vm_path:
            env_kwargs["path_to_vm"] = effective_vm_path
        env = env_cls(
            **env_kwargs,
        )
        linked_cache_files = 0 if use_remote_resources else _stage_setup_cache(task_config, cache_dir, setup_cache_dir)
        if linked_cache_files:
            LOG.info(
                "Linked %d pre-staged setup cache entries for task %s", linked_cache_files, _safe_task_id(task_config)
            )
        rollout_phase = "environment_reset"
        env.reset(task_config=task_config)
        rollout_phase = "rollout"
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
                native_agent.reset(task_logger, vm_ip=getattr(env, "vm_ip", None))
            except TypeError:
                native_agent.reset(task_logger)
        elif runner_spec.kind == "nemotron_v3_nano_omni_agent":
            if not runner_spec.agent_class_path:
                raise ValueError(f"runner {runner_spec.name!r} requires agent_class_path")
            if messages_model_fn is None:
                raise ValueError(f"runner {runner_spec.name!r} requires messages_model_fn")
            agent_cls = load_attr(runner_spec.agent_class_path)
            nemotron_kwargs: Dict[str, Any] = {
                "platform": "ubuntu",
                "model": policy_model_name or "policy_model",
                "max_steps": max_steps,
                "max_tokens": policy_max_tokens,
                "top_p": policy_top_p,
                "temperature": policy_temperature,
                "action_space": runner_spec.action_space,
                "observation_type": runner_spec.observation_type,
                "screen_size": screen_size,
                "client_password": client_password,
                "max_image_history_length": max_trajectory_length,
            }
            nemotron_kwargs.update(runner_spec.agent_kwargs)
            nemotron_kwargs["log_context"] = event_context
            native_agent = agent_cls(**nemotron_kwargs)

            def _call_nemotron_llm(payload: Dict[str, Any], _model: Optional[str] = None) -> Any:
                return messages_model_fn(payload["messages"], payload)

            native_agent.call_llm = _call_nemotron_llm
            native_agent.reset(task_logger)
        elif runner_spec.kind == "qwen3_omni_agent":
            if not runner_spec.agent_class_path:
                raise ValueError(f"runner {runner_spec.name!r} requires agent_class_path")
            if messages_model_fn is None:
                raise ValueError(f"runner {runner_spec.name!r} requires messages_model_fn")
            agent_cls = load_attr(runner_spec.agent_class_path)
            qwen_kwargs = dict(runner_spec.agent_kwargs)
            model_call_retries = int(qwen_kwargs.pop("model_call_retries", 3))
            require_tool_call = bool(qwen_kwargs.pop("require_tool_call", True))
            qwen_defaults: Dict[str, Any] = {
                "platform": "ubuntu",
                "model": policy_model_name or "policy_model",
                "max_tokens": policy_max_tokens,
                "top_p": policy_top_p,
                "temperature": policy_temperature,
                "action_space": runner_spec.action_space,
                "observation_type": runner_spec.observation_type,
                "history_n": max_trajectory_length,
            }
            qwen_defaults.update(qwen_kwargs)
            native_agent = agent_cls(**qwen_defaults)

            def _call_qwen_llm(payload: Dict[str, Any], _model: Optional[str] = None) -> str:
                last_response = ""
                last_error: Optional[Exception] = None
                for attempt in range(max(1, model_call_retries)):
                    payload["_nemo_gym_require_stop"] = True
                    try:
                        last_response = _model_response_content(messages_model_fn(payload["messages"], payload))
                        if last_response and (not require_tool_call or "<tool_call>" in last_response):
                            return last_response
                        LOG.warning(
                            "Qwen3-Omni response attempt %d/%d did not contain a tool call",
                            attempt + 1,
                            max(1, model_call_retries),
                        )
                    except Exception as exc:  # noqa: BLE001 - transport/finish failures are retryable.
                        last_error = exc
                        LOG.warning(
                            "Qwen3-Omni response attempt %d/%d failed: %s",
                            attempt + 1,
                            max(1, model_call_retries),
                            exc,
                        )
                raise ValueError("Qwen3-Omni model returned no parseable <tool_call> response") from last_error

            native_agent.call_llm = _call_qwen_llm
            native_agent.reset(task_logger)
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
                native_agent.reset(task_logger, vm_ip=getattr(env, "vm_ip", None))
            except TypeError:
                native_agent.reset(task_logger)
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
            pointer_results_base = os.environ.get(
                "OSWORLD_POINTER_RESULTS_DIR",
                os.path.join(cache_dir, "pointer_runs"),
            )
            pointer_task_dir = os.path.join(
                pointer_results_base,
                f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{_safe_task_id(task_config)}",
            )
            pointer_io_context = _pointer_model_io_context(
                event_context,
                endpoint=anthropic_base_url,
                served_model=policy_model_name,
                step_ref=_current_step,
            )
            if pointer_io_context is not None:
                pointer_io_context_token = _POINTER_MODEL_IO_CONTEXT.set(pointer_io_context)
            agent_cls = load_attr(runner_spec.agent_class_path)
            _sync_pointer_config(policy_model_name)
            _patch_pointer_optional_parallel_tools(disable_parallel_tools)
            _patch_pointer_anthropic_client(anthropic_base_url)
            pointer_agent = agent_cls(
                env=env,
                screen_size=screen_size,
                **pointer_kwargs,
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
        # Optionally log every controller command and VM response as JSONL.
        # OSWorld's env.step does not otherwise retain the /execute response.
        _vm_exec_log_paths: List[str] = []
        configured_vm_exec_log = os.environ.get("OSWORLD_VM_EXEC_LOG", "").strip()
        if configured_vm_exec_log:
            _vm_exec_log_paths.append(os.path.abspath(os.path.expanduser(configured_vm_exec_log)))
        if task_artifacts is not None:
            _vm_exec_log_paths.append(os.path.join(task_artifacts.directory, "vm-exec.jsonl"))
        _vm_exec_log_paths = list(dict.fromkeys(_vm_exec_log_paths))
        if _vm_exec_log_paths and hasattr(env.controller, "execute_python_command"):
            for vm_exec_log_path in _vm_exec_log_paths:
                os.makedirs(os.path.dirname(vm_exec_log_path), exist_ok=True)
            _orig_exec_py = env.controller.execute_python_command
            _exec_call_idx = [0]

            def _exec_logged(command: str):
                idx = _exec_call_idx[0]
                _exec_call_idx[0] += 1
                result = _orig_exec_py(command)
                try:
                    record = json.dumps(
                        {
                            **event_context,
                            "schema_version": 2,
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "task_id": _safe_task_id(task_config),
                            "step": _current_step[0],
                            "call_idx": idx,
                            "command": command or "",
                            "response": result if isinstance(result, dict) else {"_repr": repr(result)},
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    for vm_exec_log_path in _vm_exec_log_paths:
                        with open(vm_exec_log_path, "a", encoding="utf-8") as fh:
                            fh.write(record + "\n")
                except Exception:
                    task_logger.exception("Failed to write VM exec log entry %d", idx)
                return result

            env.controller.execute_python_command = _exec_logged
        elif _vm_exec_log_paths:
            task_logger.debug("Controller has no execute_python_command; VM exec trace is unavailable")

        # Wait for the VM to render a non-empty desktop. Polling adapts to
        # both KVM and slower software-emulated startup.
        cold_boot_timeout = int(os.environ.get("OSWORLD_COLD_BOOT_TIMEOUT_S", "180"))
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

        initial_screenshot = _save_task_screenshot(task_artifacts, 0, obs)
        _append_task_trajectory(
            task_artifacts,
            {
                "event": "initial_state",
                "step_num": 0,
                "instruction": instruction,
                "screenshot_file": initial_screenshot,
                **_observation_identity(obs),
            },
        )

        for step_idx in range(max_steps):
            if time.monotonic() - task_start > task_timeout:
                error = f"task_timeout exceeded ({task_timeout}s) at step {step_idx}"
                timed_out = True
                task_logger.warning(error)
                _append_task_trajectory(
                    task_artifacts,
                    {"event": "timeout", "step_num": step_idx + 1, "error": error},
                )
                break
            _current_step[0] = step_idx + 1
            obs_entry = {
                "screenshot_b64": _b64(obs.get("screenshot")),
                "accessibility_tree": obs.get("accessibility_tree"),
            }
            history_window = obs_history[-max_trajectory_length:] if max_trajectory_length else []

            agent_step_info: Dict[str, Any] = {}
            try:
                if pointer_agent is not None:
                    model_text, actions = pointer_agent.predict(obs)
                    model_text = strip_thinking(model_text or "")
                    actions = _flatten_actions(actions)
                elif native_agent is not None:
                    prediction = native_agent.predict(instruction, obs)
                    if not isinstance(prediction, (list, tuple)) or len(prediction) not in {2, 3}:
                        raise TypeError(
                            f"Native OSWorld agent returned unsupported prediction shape: {type(prediction).__name__}"
                        )
                    model_text, actions = prediction[:2]
                    if len(prediction) == 3 and isinstance(prediction[2], dict):
                        agent_step_info = prediction[2]
                    model_text = strip_thinking(_model_response_content(model_text))
                    actions = _flatten_actions(actions)
                    if runner_spec.kind == "qwen3_omni_agent":
                        actions = _merge_consecutive_pyautogui_actions(actions)
                    if runner_spec.action_space == "computer_13":
                        actions = [_normalize_prompt_agent_computer_13_action(action) for action in actions]
                else:
                    model_text = model_fn(system_prompt, instruction, history_window + [obs_entry])
                    model_text = strip_thinking(model_text or "")
                    actions = parse_actions(model_text)
            except Exception as exc:  # noqa: BLE001 — record + abort, don't crash the VM.
                error = f"agent/model call failed at step {step_idx}: {exc}"
                task_logger.exception("Agent/model call failed at step %d", step_idx)
                steps.append(StepRecord(step=step_idx, model_text="", actions=[], reward=0.0, done=False))
                screenshot_file = _save_task_screenshot(task_artifacts, step_idx + 1, obs)
                _append_task_trajectory(
                    task_artifacts,
                    {
                        "event": "step",
                        "step_num": step_idx + 1,
                        "response": "",
                        "actions": [],
                        "reward": 0.0,
                        "done": False,
                        "error": error,
                        "screenshot_file": screenshot_file,
                        **_observation_identity(obs),
                    },
                )
                break

            task_logger.info("Step %d model response:\n%s", step_idx + 1, model_text)
            task_logger.info("Step %d parsed actions: %r", step_idx + 1, actions)
            if not actions:
                # No parseable action — log the step and continue. The model
                # gets another chance next iteration with a fresh screenshot.
                steps.append(
                    StepRecord(
                        step=step_idx,
                        model_text=model_text,
                        actions=[],
                        reward=0.0,
                        done=False,
                        info={"agent": agent_step_info} if agent_step_info else {},
                    )
                )
                obs_history.append(obs_entry)
                screenshot_file = _save_task_screenshot(task_artifacts, step_idx + 1, obs)
                _append_task_trajectory(
                    task_artifacts,
                    {
                        "event": "step",
                        "step_num": step_idx + 1,
                        "response": model_text,
                        "actions": [],
                        "reward": 0.0,
                        "done": False,
                        "info": {"agent": agent_step_info} if agent_step_info else {},
                        "screenshot_file": screenshot_file,
                        **_observation_identity(obs),
                    },
                )
                continue

            step_done = False
            step_reward = 0.0
            step_info: Dict[str, Any] = {}
            for action in actions:
                terminal_action = _terminal_action_name(action)
                if terminal_action is not None:
                    agent_terminal_action = terminal_action
                task_logger.info("Step %d executing action: %r", step_idx + 1, action)
                try:
                    obs, reward, done, info = env.step(action, sleep_after_execution)
                except Exception as exc:  # noqa: BLE001 - record bad model/controller actions.
                    error = f"env.step() failed at step {step_idx}: {exc}"
                    task_logger.exception("Environment step failed at step %d for action %r", step_idx, action)
                    break
                step_reward += float(reward or 0.0)
                step_info = info if isinstance(info, dict) else {"info": info}
                if agent_step_info:
                    step_info = dict(step_info)
                    step_info.setdefault("agent", agent_step_info)
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
            screenshot_file = _save_task_screenshot(task_artifacts, step_idx + 1, obs)
            _append_task_trajectory(
                task_artifacts,
                {
                    "event": "step",
                    "step_num": step_idx + 1,
                    "response": model_text,
                    "actions": actions,
                    "reward": step_reward,
                    "done": step_done,
                    "info": step_info,
                    "error": error,
                    "screenshot_file": screenshot_file,
                    **_observation_identity(obs),
                },
            )
            task_logger.info(
                "Step %d completed: reward=%s done=%s error=%r",
                step_idx + 1,
                step_reward,
                step_done,
                error,
            )

            if step_done:
                finished = True
                break
            if error:
                break

        # Let the VM settle before scoring, mirroring lib_run_single.py.
        time.sleep(2)
        rollout_error_before_evaluation = error
        try:
            eval_logger = pointer_logger if pointer_agent is not None else task_logger
            final_score = _evaluate_osworld_env(env, eval_logger, disable_gpu=evaluator_disable_gpu)
        except Exception as exc:  # noqa: BLE001
            evaluation_error = f"env.evaluate() failed: {exc}"
            error = evaluation_error
            task_logger.exception("Evaluator failed")
            final_score = 0.0
        try:
            result_artifacts = _evaluator_result_artifacts(task_config, cache_dir)
        except Exception:  # noqa: BLE001 - logging must not change evaluator behavior.
            task_logger.exception("Failed to inventory evaluator result artifacts")
            result_artifacts = []
        evaluator = task_config.get("evaluator")
        evaluator_func = evaluator.get("func") if isinstance(evaluator, dict) else None
        _append_task_trajectory(
            task_artifacts,
            {
                "event": "evaluation",
                "score": final_score,
                "status": "error" if evaluation_error else "completed",
                "evaluator_func": evaluator_func,
                "agent_terminal_action": agent_terminal_action,
                "agent_declared_success": agent_terminal_action == "DONE",
                "rollout_error_before_evaluation": rollout_error_before_evaluation,
                "evaluation_error": evaluation_error,
                "result_artifacts": result_artifacts,
                "error": error,
            },
        )
        if pointer_agent is not None and hasattr(pointer_agent, "log_usage"):
            try:
                pointer_agent.log_usage()
            except Exception:  # noqa: BLE001 - usage logging should not fail the rollout.
                LOG.exception("PointerAgent.log_usage() failed")

    except _EvaluatorScoreZero as exc:
        setup_score_zero = True
        finished = True
        final_score = 0.0
        error = None
        task_logger.info("OSWorld setup established score zero before evaluation: %s", exc)
        _append_task_trajectory(
            task_artifacts,
            {"event": "evaluation", "score": 0.0, "status": "completed", "reason": "setup_score_zero"},
        )
    except Exception as exc:  # noqa: BLE001 — top-level guard so caller sees error not crash.
        proxy_setup_error = bool(requires_proxy and enable_proxy and rollout_phase == "environment_reset")
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        LOG.exception("OSWorld rollout failed before evaluation")
    finally:
        if pointer_io_context_token is not None:
            _POINTER_MODEL_IO_CONTEXT.reset(pointer_io_context_token)
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
                recording_path = _mp4_path
                task_logger.info("Saved VM recording: %s", _mp4_path)
            except Exception:  # noqa: BLE001 — best-effort.
                LOG.exception("end_recording() failed; mp4 may be missing or partial")
        if env is not None:
            try:
                env.close()
            except Exception:
                LOG.exception("env.close() raised")

    if reward_mode == "raw":
        reward = float(final_score)
    else:
        reward = 1.0 if final_score >= 1.0 else 0.0
    # mask_sample: reward is unreliable if (a) anything errored, (b) timeout,
    # or (c) loop exhausted max_steps without the model emitting DONE/FAIL.
    mask_sample = bool(error) or timed_out or not finished
    if timed_out:
        termination_reason = "timeout"
    elif setup_score_zero:
        termination_reason = "setup_score_zero"
    elif evaluation_error:
        termination_reason = "evaluator_error"
    elif proxy_setup_error:
        termination_reason = "proxy_setup_error"
    elif error:
        termination_reason = "rollout_error"
    elif agent_terminal_action is not None:
        termination_reason = f"agent_{agent_terminal_action.lower()}"
    elif finished:
        termination_reason = "environment_done"
    else:
        termination_reason = "max_steps"

    result = RolloutResult(
        reward=reward,
        score=final_score,
        steps=steps,
        error=error,
        finished=finished,
        mask_sample=mask_sample,
        artifact_dir=task_artifacts.directory if task_artifacts is not None else None,
        termination_reason=termination_reason,
    )
    _finalize_task_artifacts(
        task_artifacts,
        result=result,
        duration_seconds=time.monotonic() - task_start,
        recording_path=recording_path,
    )
    return result
