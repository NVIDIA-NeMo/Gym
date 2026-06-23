# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Agent server that runs mini-swe-agent against SWE-bench tasks inside a sandbox.

Each task is executed in an isolated sandbox via a Ray remote worker: the agent produces a code
patch, which is then graded either in-process with the swebench harness or by POSTing it to a
shared swe_env verifier. The server converts mini-swe-agent trajectories into Gym Responses API
rows and aggregates pass/resolution metrics across rollouts.
"""

import asyncio
import hashlib
import json
import os
import sys
import time
import traceback
from asyncio import Semaphore
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Literal, Optional, cast
from uuid import uuid4

import ray
import yaml
from fastapi import Body, FastAPI
from minisweagent.config import builtin_config_dir, get_config_path
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.server_utils import (
    ServerClient,
    get_first_server_config_dict,
    get_response_json,
    raise_for_status,
)


OPENSANDBOX_PROVIDER_NAME = "opensandbox"
OPENSANDBOX_API_KEY_ENV = "OPENSANDBOX_API_KEY"  # pragma: allowlist secret


class MiniSWEAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    env: Literal["sandbox"]
    concurrency: int
    sandbox_provider: Optional[dict[str, Any]] = None
    sandbox_spec: Optional[dict[str, Any]] = None
    sandbox_environment_kwargs: Optional[dict[str, Any]] = None
    run_golden: bool = False
    step_timeout: int = 600
    eval_timeout: int = 1800
    skip_if_exists: bool = False
    step_limit: int = 250
    tool_choice: Optional[str | dict[str, Any]] = None
    sandbox_resource_profiles: Optional[list[dict[str, str]]] = None

    # Opt-in: score the patch via the shared swe_env verifier instead of grading in-process.
    eval_via_verifier: bool = Field(
        default=False,
        description=(
            "If True, score the agent's patch by POSTing it to the shared swe_env verifier "
            "(verifier_server_name) instead of grading it in-process. Default False grades "
            "in-process; setting it True reuses the same swe_env verifier other agents use."
        ),
    )
    verifier_server_name: Optional[str] = Field(
        default=None,
        description=(
            "Name of the resources_servers/swe_env verifier to POST /verify to when "
            "eval_via_verifier=True. Required when eval_via_verifier is True."
        ),
    )


class MiniSWEAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class MiniSWEAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class MiniSWEAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
    },
)
def runner_ray_remote(runner: Callable, params: dict[str, Any]) -> Any:
    """Invoke a runner callable inside a Ray remote task.

    Args:
        runner: The callable to execute on the worker.
        params: Keyword arguments to pass to ``runner``.

    Returns:
        The value returned by ``runner``.
    """
    return runner(**params)


def _json_dict_from_metadata(value: Any, *, field_name: str) -> dict[str, Any]:
    """Coerce a metadata value into a dict, parsing JSON strings as needed.

    Args:
        value: The metadata value to coerce; may be None, a dict, or a JSON-encoded string.
        field_name: The metadata field name, used only for the error message.

    Returns:
        The value as a dict; an empty dict when ``value`` is None.

    Raises:
        ValueError: If the value is not None, a dict, or a JSON string decoding to an object.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"responses_create_params.metadata.{field_name} must be a JSON object")


def _responses_create_params_to_model_kwargs(
    params: dict[str, Any],
    *,
    default_tool_choice: Any = None,
) -> dict[str, Any]:
    """Convert Gym Responses API rollout params into mini-swe-agent chat-completions kwargs.

    Args:
        params: The Responses API create params (temperature, top_p, max_output_tokens, metadata,
            tool_choice, etc.).
        default_tool_choice: A tool choice that overrides ``params["tool_choice"]`` when provided.

    Returns:
        A dict of chat-completions kwargs (e.g. temperature, max_tokens, extra_body, tool_choice)
        containing only the fields that were set.
    """
    model_kwargs: dict[str, Any] = {}
    for key in ("temperature", "top_p", "top_logprobs", "parallel_tool_calls"):
        value = params.get(key)
        if value is not None:
            model_kwargs[key] = value

    max_output_tokens = params.get("max_output_tokens")
    if max_output_tokens is not None:
        model_kwargs["max_tokens"] = max_output_tokens

    metadata = params.get("metadata") or {}
    extra_body = _json_dict_from_metadata(metadata.get("extra_body"), field_name="extra_body")
    chat_template_kwargs = _json_dict_from_metadata(
        metadata.get("chat_template_kwargs"),
        field_name="chat_template_kwargs",
    )
    if chat_template_kwargs:
        extra_body["chat_template_kwargs"] = chat_template_kwargs
    if extra_body:
        model_kwargs["extra_body"] = extra_body

    tool_choice = default_tool_choice if default_tool_choice is not None else params.get("tool_choice")
    if tool_choice == "bash":
        model_kwargs["tool_choice"] = _bash_tool_choice()
    elif tool_choice is not None:
        model_kwargs["tool_choice"] = tool_choice

    return model_kwargs


def _opensandbox_connection(provider: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract the opensandbox connection dict from a sandbox provider config.

    Args:
        provider: The sandbox provider config mapping, or None.

    Returns:
        The nested opensandbox connection dict, or None if it is absent or malformed.
    """
    if provider is None:
        return None
    provider_config = provider.get(OPENSANDBOX_PROVIDER_NAME)
    if not isinstance(provider_config, dict):
        return None
    connection = provider_config.get("connection")
    if not isinstance(connection, dict):
        return None
    return connection


def _sandbox_provider_for_config_dump(provider: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the provider config safe to write to disk, with the API key removed.

    Args:
        provider: The sandbox provider config mapping.

    Returns:
        A deep copy of ``provider`` with any opensandbox connection ``api_key`` stripped.
    """
    provider_for_disk = deepcopy(provider)
    connection = _opensandbox_connection(provider_for_disk)
    if connection is not None:
        connection.pop("api_key", None)
    return provider_for_disk


def _sandbox_runtime_env(provider: dict[str, Any] | None) -> dict[str, Any]:
    """Build the Ray runtime env for the sandbox worker, injecting the API key when available.

    Args:
        provider: The sandbox provider config mapping, or None.

    Returns:
        A runtime env dict with ``py_executable`` set, and ``env_vars`` carrying the opensandbox
        API key when the provider supplies one.
    """
    runtime_env: dict[str, Any] = {"py_executable": sys.executable}
    connection = _opensandbox_connection(provider)
    if connection is None:
        return runtime_env
    api_key = connection.get("api_key")
    if api_key:
        runtime_env["env_vars"] = {OPENSANDBOX_API_KEY_ENV: str(api_key)}
    return runtime_env


def _restore_sandbox_provider_secrets(config: dict[str, Any]) -> None:
    """Repopulate the opensandbox API key in a loaded config from the environment.

    Mutates ``config`` in place so the provider connection has an ``api_key`` again after the key
    was stripped for on-disk storage, reading it from the environment variable.

    Args:
        config: The loaded agent config dict to mutate.
    """
    provider = config.get("environment", {}).get("provider")
    connection = _opensandbox_connection(provider if isinstance(provider, dict) else None)
    if connection is None or connection.get("api_key"):
        return
    api_key = os.getenv(OPENSANDBOX_API_KEY_ENV)
    if api_key:
        connection["api_key"] = api_key


def _bash_tool_choice() -> dict[str, Any]:
    """Build the tool-choice payload that forces the model to call the ``bash`` function.

    Returns:
        A tool-choice dict selecting the ``bash`` function.
    """
    return {"type": "function", "function": {"name": "bash"}}


def _sandbox_spec_for_instance(
    spec: dict[str, Any] | None,
    *,
    resource_profiles: list[dict[str, Any]] | None,
    instance_id: str,
) -> dict[str, Any]:
    """Build the per-instance sandbox spec, applying a deterministically chosen resource profile.

    Args:
        spec: The base sandbox spec mapping, or None.
        resource_profiles: Candidate resource profile mappings to choose from, or None to skip.
        instance_id: The instance identifier, hashed to pick a profile deterministically.

    Returns:
        A copy of the spec with the selected resource profile merged into its ``resources``, or the
        plain spec copy when no profiles are provided.
    """
    instance_spec = dict(spec or {})
    if not resource_profiles:
        return instance_spec

    resources = dict(instance_spec.get("resources") or {})
    digest = hashlib.sha256(instance_id.encode("utf-8")).digest()
    profile = resource_profiles[int.from_bytes(digest[:4], "big") % len(resource_profiles)]
    resources.update(profile)
    instance_spec["resources"] = resources
    return instance_spec


def _swebench_config_path() -> Path:
    """Locate the bundled mini-swe-agent SWE-bench config file.

    Returns:
        The path to the first existing candidate config, falling back to the canonical location.
    """
    for candidate in (
        builtin_config_dir / "extra" / "swebench.yaml",
        builtin_config_dir / "benchmarks" / "swebench.yaml",
    ):
        if candidate.exists():
            return candidate
    return builtin_config_dir / "extra" / "swebench.yaml"


def _swebench_image_name(instance: dict[str, Any], subset: str) -> str:
    """Resolve the Docker image name for a SWE-bench instance.

    Uses the instance's explicit ``image_name`` when present; otherwise derives the conventional
    image reference from the instance id and subset.

    Args:
        instance: The task instance dict; may carry ``image_name`` and must carry ``instance_id``.
        subset: The dataset subset (e.g. "verified"), which selects the image naming scheme.

    Returns:
        The fully qualified Docker image name, lowercased.
    """
    image_name = instance.get("image_name")
    if image_name:
        return str(image_name)

    instance_id = instance["instance_id"]
    if subset == "verified":
        docker_compatible_id = instance_id.replace("__", "_1776_")
        return f"docker.io/swebench/sweb.eval.x86_64.{docker_compatible_id}:latest".lower()

    docker_compatible_id = instance_id.replace("__", "_s_")
    return f"docker.io/xingyaoww/sweb.eval.x86_64.{docker_compatible_id}:latest".lower()


def _message_content_to_text(content: Any) -> str:
    """Flatten a message ``content`` field into a single text string.

    Args:
        content: A message content value: a string, a list of content parts, None, or other.

    Returns:
        The concatenated text; an empty string when ``content`` is None.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def _strip_extra(item: Any) -> dict[str, Any]:
    """Normalize a trajectory item to a dict and drop its ``extra`` key.

    Args:
        item: A pydantic model, a dict, or any other value to normalize.

    Returns:
        A dict without the ``extra`` key; non-dict, non-model values are wrapped as a user message.
    """
    if hasattr(item, "model_dump"):
        item = item.model_dump()
    if not isinstance(item, dict):
        return {"type": "message", "role": "user", "content": str(item)}
    return {key: value for key, value in item.items() if key != "extra"}


def _split_trajectory_for_responses(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a mini-swe-agent message trajectory into Responses API input, output, and responses.

    The leading system/user messages become input messages; subsequent assistant turns, tool calls,
    tool outputs, and raw response objects become output items.

    Args:
        messages: The ordered list of trajectory messages from the agent run.

    Returns:
        A tuple ``(input_messages, output_items, raw_responses)`` in Responses API form.
    """
    input_messages: list[dict[str, Any]] = []
    output_items: list[dict[str, Any]] = []
    raw_responses: list[dict[str, Any]] = []
    in_initial_prompt = True

    for message in messages:
        role = message.get("role")
        if in_initial_prompt and role in {"system", "user"}:
            input_messages.append(
                {"type": "message", "role": role, "content": _message_content_to_text(message.get("content"))}
            )
            continue

        in_initial_prompt = False
        if message.get("object") == "response":
            response = _strip_extra(message)
            raw_responses.append(response)
            output_items.extend(_strip_extra(item) for item in response.get("output", []))
        elif role == "assistant":
            content = _message_content_to_text(message.get("content"))
            if content:
                output_items.append(
                    {
                        "id": message.get("id") or f"msg_{uuid4()}",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": content, "annotations": []}],
                    }
                )
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                output_items.append(
                    {
                        "id": tool_call.get("id") or f"fc_{uuid4()}",
                        "type": "function_call",
                        "name": function.get("name") or tool_call.get("name") or "",
                        "call_id": tool_call.get("id") or tool_call.get("call_id") or "",
                        "arguments": function.get("arguments") or tool_call.get("arguments") or "{}",
                    }
                )
        elif role == "tool":
            output_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id") or message.get("call_id") or "",
                    "output": _message_content_to_text(message.get("content")),
                }
            )
        elif message.get("type") == "function_call_output":
            output_items.append(_strip_extra(message))

    return input_messages, output_items, raw_responses


def _default_response_object() -> dict[str, Any]:
    """Build a fully populated default Responses API response object.

    Returns:
        A dict with a fresh id, current timestamp, and default values for every Responses API field.
    """
    return {
        "id": f"resp_{str(uuid4())}",
        "created_at": int(time.time()),
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "object": "response",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "background": False,
        "max_output_tokens": None,
        "max_tool_calls": None,
        "previous_response_id": None,
        "prompt": None,
        "reasoning": {
            "effort": None,
            "generate_summary": None,
            "summary": None,
        },
        "service_tier": "default",
        "status": "completed",
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "top_logprobs": 0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        },
        "user": None,
        "prompt_cache_key": None,
        "safety_identifier": None,
        "store": True,
    }


def _is_resolved(instance_id: str, eval_report: dict[str, Any]) -> bool:
    """Determine whether an instance was resolved according to its eval report.

    An instance counts as resolved only when the harness marked it resolved and reported at least
    one test outcome.

    Args:
        instance_id: The instance identifier to look up in the report.
        eval_report: The eval report dict produced by the swebench harness.

    Returns:
        True if the instance is resolved with reported test results; False otherwise or on any error.
    """
    try:
        if not eval_report:
            return False
        report = eval_report["eval_report"][instance_id]
        resolved = bool(report["resolved"])
        if not report.get("tests_status"):
            return False

        tests_status = report["tests_status"]
        f2f = tests_status.get("FAIL_TO_PASS", {})
        p2p = tests_status.get("PASS_TO_PASS", {})
        total_reported = (
            len(f2f.get("success", []))
            + len(f2f.get("failure", []))
            + len(p2p.get("success", []))
            + len(p2p.get("failure", []))
        )
        return resolved and total_reported > 0
    except Exception as exc:
        print(f"Error in _is_resolved: {exc}", flush=True)
        return False


def _metadata_dict(verify_response: dict[str, Any]) -> dict[str, Any]:
    """Extract the ``metadata`` mapping from a verify/rollout response.

    Args:
        verify_response: The verify or rollout response dict.

    Returns:
        The metadata dict, or an empty dict when absent or not a dict.
    """
    metadata = verify_response.get("metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def _eval_report_map(verify_response: dict[str, Any]) -> dict[str, Any]:
    """Extract the per-instance eval report map from a verify/rollout response.

    Args:
        verify_response: The verify or rollout response dict.

    Returns:
        The ``eval_report`` mapping keyed by instance id, or an empty dict when absent.
    """
    report = _metadata_dict(verify_response).get("eval_report") or {}
    return report if isinstance(report, dict) else {}


def _eval_instance_report(verify_response: dict[str, Any]) -> dict[str, Any]:
    """Find the single instance's eval report within a verify/rollout response.

    Looks up the report by instance id, falling back to the first report entry that carries a
    ``resolved`` flag.

    Args:
        verify_response: The verify or rollout response dict.

    Returns:
        The matching per-instance report dict, or an empty dict when none is found.
    """
    report_map = _eval_report_map(verify_response)
    instance_id = verify_response.get("instance_id") or _metadata_dict(verify_response).get("instance_id")
    if instance_id is not None:
        report = report_map.get(str(instance_id))
        if isinstance(report, dict):
            return report

    for report in report_map.values():
        if isinstance(report, dict) and "resolved" in report:
            return report
    return {}


def _test_status_counts(verify_response: dict[str, Any]) -> dict[str, int]:
    """Count per-suite test successes and failures from a verify/rollout response.

    Args:
        verify_response: The verify or rollout response dict.

    Returns:
        A dict mapping ``<suite>_success`` and ``<suite>_failure`` keys to their counts; empty when
        no test status is present.
    """
    report = _eval_instance_report(verify_response)
    tests_status = report.get("tests_status") if isinstance(report, dict) else None
    if not isinstance(tests_status, dict):
        return {}

    counts: dict[str, int] = {}
    for suite_name, suite_report in tests_status.items():
        if not isinstance(suite_report, dict):
            continue
        prefix = str(suite_name).lower()
        counts[f"{prefix}_success"] = len(suite_report.get("success") or [])
        counts[f"{prefix}_failure"] = len(suite_report.get("failure") or [])
    return counts


def _run_eval_v2(
    *,
    instance: dict[str, Any],
    env: Any,
    model_patch: str,
    instance_dir: Path,
    run_id: str,
    is_golden: bool,
) -> dict[str, Any]:
    """Grade a patch in-process with the swebench harness inside the task's sandbox.

    Writes the patch, runs the harness eval script in the sandbox, captures test output, and builds
    the eval report. When grading a golden patch, applies the patch to the repo first.

    Args:
        instance: The SWE-bench instance dict.
        env: The sandbox environment used to execute the eval script.
        model_patch: The patch diff to grade.
        instance_dir: The directory where logs, patch, output, and report files are written.
        run_id: A unique identifier for this eval run, used in output filenames.
        is_golden: Whether ``model_patch`` is the golden patch and should be applied before eval.

    Returns:
        A dict with ``instance_id``, ``model_patch``, and the harness ``eval_report``.
    """
    from swebench.harness.constants import SWEbenchInstance
    from swebench.harness.docker_build import setup_logger
    from swebench.harness.grading import get_eval_report
    from swebench.harness.test_spec.test_spec import make_test_spec

    swebench_instance = cast(SWEbenchInstance, instance)
    test_spec = make_test_spec(swebench_instance)
    pred = {"instance_id": test_spec.instance_id, "model_patch": model_patch}

    instance_dir.mkdir(parents=True, exist_ok=True)
    log_file = instance_dir / f"run_instance_{run_id}.log"
    report_path = instance_dir / f"report_{run_id}.json"
    patch_file = instance_dir / f"patch_{run_id}.diff"
    patch_file.write_text(model_patch)

    logger = setup_logger(test_spec.instance_id, log_file)
    logger.info(f"DEBUG test_spec {test_spec}")
    logger.info(f"DEBUG eval_script {test_spec.eval_script}")

    if is_golden:
        env.execute(f"cat > patch.diff <<'EOF'\n{model_patch}\n\nEOF")
        env.execute("git status --porcelain")
        env.execute("git apply --check patch.diff")
        env.execute("git apply patch.diff")

    eval_script = test_spec.eval_script.replace("#!/bin/bash", "")
    result = env.execute(eval_script, is_eval=True)
    test_output = result["output"]
    returncode = result["returncode"]
    print(f"[EVAL]{test_spec.instance_id} returncode: {returncode}", flush=True)

    test_output_path = instance_dir / f"test_output_{run_id}.txt"
    test_output_path.write_text(test_output)
    print(f"[EVAL]{test_spec.instance_id} Test output written to {test_output_path}", flush=True)

    report = get_eval_report(
        test_spec=test_spec,
        prediction=pred,
        test_log_path=str(test_output_path),
        include_tests_status=True,
    )
    print(f"[EVAL]{test_spec.instance_id} Result: resolved: {report[test_spec.instance_id]['resolved']}", flush=True)

    report_path.write_text(json.dumps(report, indent=4))
    return {
        "instance_id": test_spec.instance_id,
        "model_patch": model_patch,
        "eval_report": report,
    }


def _run_mini_swe_v2(**params: Any) -> dict[str, Any]:
    """Run a single mini-swe-agent task in a sandbox and return its trajectory and eval report.

    Builds the agent config, starts the sandbox environment, runs the agent (or applies the golden
    patch), optionally grades the patch in-process, and converts the trajectory to Responses API form.

    Args:
        **params: Run parameters including ``instance_dict``, ``instance_id``, ``output``, ``config``,
            ``model``, ``api_key``, ``base_url``, ``subset``, ``step_timeout``, ``eval_timeout``,
            ``step_limit``, ``run_golden``, and ``eval_via_verifier``.

    Returns:
        A dict keyed by instance id whose value holds ``input_messages``, ``response_output``,
        ``responses``, ``eval_report``, and ``exit_status``.
    """
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments import get_environment
    from minisweagent.models import get_model

    instance = params.get("instance_dict")
    if isinstance(instance, str):
        instance = json.loads(instance)
    if not isinstance(instance, dict):
        raise ValueError("mini-swe-agent v2 path requires instance_dict")

    instance = dict(instance)
    instance_id = str(params.get("instance_id") or instance["instance_id"]).lower()
    instance["instance_id"] = instance_id

    output_dir = Path(params["output"])
    instance_dir = output_dir / instance_id
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(get_config_path(params["config"]).read_text())
    _restore_sandbox_provider_secrets(config)
    model_config = config.setdefault("model", {})
    model_config["model_class"] = "litellm"
    model_config["model_name"] = params["model"]
    model_config.setdefault("cost_tracking", "ignore_errors")
    model_kwargs = model_config.setdefault("model_kwargs", {})
    model_kwargs["api_key"] = params["api_key"]
    model_kwargs["base_url"] = params["base_url"]
    model_kwargs.pop("api_base", None)
    max_output_tokens = model_kwargs.pop("max_output_tokens", None)
    if max_output_tokens is not None and "max_tokens" not in model_kwargs:
        model_kwargs["max_tokens"] = max_output_tokens

    environment_config = config.setdefault("environment", {})
    environment_config["image"] = _swebench_image_name(instance, params["subset"])
    environment_config["step_timeout"] = params["step_timeout"]
    environment_config["eval_timeout"] = params["eval_timeout"]
    environment_config["instance_id"] = instance_id
    environment_config["environment_class"] = (
        "responses_api_agents.mini_swe_agent_2.sandbox_environment.MiniSWESandboxEnvironment"
    )

    agent_config = config.get("agent", {})
    agent_config["step_limit"] = params["step_limit"]
    agent_config.pop("collapse_limit", None)

    run_id = f"{int(time.time())}_{uuid4()}"
    trajectory_path = instance_dir / f"{instance_id}_{run_id}.traj.json"
    agent_config["output_path"] = trajectory_path
    env = None
    agent = None
    try:
        print(f"[EVAL]{instance_id} Creating environment...", flush=True)
        env = get_environment(environment_config)
        print(f"[EVAL]{instance_id} Environment created", flush=True)

        model = get_model(config=model_config)
        agent = DefaultAgent(model, env, **agent_config)

        if params["run_golden"]:
            exit_status = "Gold Patch Applied"
            model_patch = instance.get("patch", "")
            data = agent.save(None, {"messages": []})
        else:
            print(f"[EVAL]{instance_id} Running mini-swe-agent v2...", flush=True)
            info = agent.run(instance["problem_statement"])
            exit_status = info.get("exit_status", "")
            model_patch = info.get("submission", "")
            data = agent.save(
                trajectory_path,
                {"instance_id": instance_id},
            )

        if params.get("eval_via_verifier"):
            # Skip the in-worker swebench harness; the patch is graded out-of-band by the shared
            # swe_env verifier. Carry the patch forward so the caller can POST it.
            print(f"[EVAL]{instance_id} Skipping in-worker eval (eval_via_verifier)", flush=True)
            eval_report = {"instance_id": instance_id, "model_patch": model_patch}
        else:
            print(f"[EVAL]{instance_id} Running eval", flush=True)
            eval_report = _run_eval_v2(
                instance=instance,
                env=env,
                model_patch=model_patch,
                instance_dir=instance_dir,
                run_id=run_id,
                is_golden=params["run_golden"],
            )
            print(f"[EVAL]{instance_id} Eval completed", flush=True)

        input_messages, response_output, responses = _split_trajectory_for_responses(data.get("messages", []))

        return {
            instance_id: {
                "input_messages": input_messages,
                "response_output": response_output,
                "responses": responses,
                "eval_report": eval_report,
                "exit_status": exit_status,
            }
        }
    finally:
        if env and hasattr(env, "cleanup"):
            env.cleanup()


def run_mini_swe_with_sandbox(**params: Any) -> Any:
    """Run a mini-swe-agent task in a sandbox.

    Args:
        **params: The run parameters forwarded to the sandbox runner.

    Returns:
        The per-instance result dict produced by the sandbox runner.
    """
    return _run_mini_swe_v2(**params)


class MiniSWEAgent(SimpleResponsesAPIAgent):
    """Responses API agent that runs mini-swe-agent on SWE-bench tasks and scores the patches.

    Bounds concurrency with a semaphore, dispatches each task to a Ray sandbox worker, grades the
    resulting patch (in-process or via a shared verifier), and exposes rollout-level metrics.
    """

    config: MiniSWEAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """Initialize the concurrency semaphore from the configured concurrency limit.

        Args:
            __context: The pydantic post-init context (unused).
        """
        self.sem = Semaphore(self.config.concurrency)

    def setup_webserver(self) -> FastAPI:
        """Build the FastAPI app and register the agent's HTTP routes.

        Returns:
            The configured FastAPI application.
        """
        app = FastAPI()
        self.setup_session_middleware(app)
        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        return app

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Aggregate pass/resolution, eval-error, and test-status metrics across all rollouts.

        Args:
            tasks: A list of tasks, each a list of rollout dicts for that task.

        Returns:
            A dict of aggregate metrics, including pass@k, resolved counts and rates, eval error and
            report rates, patch-applied rates, per-task metrics, and per-suite test status totals.
        """
        metrics, _, _, max_k = compute_pass_majority_metrics(tasks)
        metrics.pop("per_sample_aggregate", None)

        all_rollouts = [rollout for task in tasks for rollout in task]
        rollout_count = len(all_rollouts)
        resolved_task_count = sum(1 for task in tasks if any(float(r.get("reward", 0.0) or 0.0) >= 1.0 for r in task))
        eval_error_count = sum(1 for rollout in all_rollouts if _metadata_dict(rollout).get("error"))
        eval_report_count = sum(1 for rollout in all_rollouts if _eval_report_map(rollout))
        tests_status_count = sum(1 for rollout in all_rollouts if _eval_instance_report(rollout).get("tests_status"))
        patch_applied_count = sum(
            1 for rollout in all_rollouts if _eval_instance_report(rollout).get("patch_successfully_applied")
        )

        metrics.update(
            {
                "task_count": len(tasks),
                "rollout_count": rollout_count,
                "max_rollouts_per_task": max_k,
                "resolved_task_count": resolved_task_count,
                "resolved_task_rate": 100.0 * resolved_task_count / len(tasks) if tasks else 0.0,
                "eval_error_rollout_count": eval_error_count,
                "eval_error_rate": 100.0 * eval_error_count / rollout_count if rollout_count else 0.0,
                "eval_report_rollout_count": eval_report_count,
                "eval_report_rate": 100.0 * eval_report_count / rollout_count if rollout_count else 0.0,
                "tests_status_rollout_count": tests_status_count,
                "tests_status_rate": 100.0 * tests_status_count / rollout_count if rollout_count else 0.0,
                "patch_applied_rollout_count": patch_applied_count,
                "patch_applied_rate": 100.0 * patch_applied_count / rollout_count if rollout_count else 0.0,
                "per_task_metrics": self._compute_per_task_eval_metrics(tasks),
            }
        )

        test_status_totals: dict[str, int] = {}
        for rollout in all_rollouts:
            for key, value in _test_status_counts(rollout).items():
                test_status_totals[key] = test_status_totals.get(key, 0) + value
        metrics.update({f"tests_status/{key}": value for key, value in sorted(test_status_totals.items())})

        return metrics

    def _compute_per_task_eval_metrics(self, tasks: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Compute per-task eval metrics from each task's rollouts.

        Args:
            tasks: A list of tasks, each a list of rollout dicts for that task.

        Returns:
            A list of per-task metric dicts (task index, instance id, rollout/resolved/error counts,
            and per-suite test status totals), skipping tasks with no rollouts.
        """
        per_task_metrics: list[dict[str, Any]] = []
        for fallback_idx, rollouts in enumerate(tasks):
            if not rollouts:
                continue

            first = rollouts[0]
            task_index = first.get(TASK_INDEX_KEY_NAME, fallback_idx)
            instance_id = first.get("instance_id") or _metadata_dict(first).get("instance_id")
            resolved_count = sum(1 for rollout in rollouts if float(rollout.get("reward", 0.0) or 0.0) >= 1.0)
            error_count = sum(1 for rollout in rollouts if _metadata_dict(rollout).get("error"))
            eval_report_count = sum(1 for rollout in rollouts if _eval_report_map(rollout))
            tests_status_count = sum(1 for rollout in rollouts if _eval_instance_report(rollout).get("tests_status"))
            patch_applied_count = sum(
                1 for rollout in rollouts if _eval_instance_report(rollout).get("patch_successfully_applied")
            )

            task_metrics: dict[str, Any] = {
                TASK_INDEX_KEY_NAME: task_index,
                "instance_id": instance_id,
                "rollout_count": len(rollouts),
                "resolved": resolved_count > 0,
                "resolved_rollout_count": resolved_count,
                "eval_error_rollout_count": error_count,
                "eval_report_rollout_count": eval_report_count,
                "tests_status_rollout_count": tests_status_count,
                "patch_applied_rollout_count": patch_applied_count,
            }

            test_status_totals: dict[str, int] = {}
            for rollout in rollouts:
                for key, value in _test_status_counts(rollout).items():
                    test_status_totals[key] = test_status_totals.get(key, 0) + value
            task_metrics.update({f"tests_status/{key}": value for key, value in sorted(test_status_totals.items())})
            per_task_metrics.append(task_metrics)

        return per_task_metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        """Select the headline metrics to surface from the full aggregate metrics.

        Args:
            agent_metrics: The full metrics dict produced by ``compute_metrics``.

        Returns:
            A dict containing the highest-k pass metrics plus selected reward, resolution, and error
            rate metrics that are present in ``agent_metrics``.
        """
        key_metrics: dict[str, Any] = {}
        key_metrics.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=["accuracy"]))
        key_metrics.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy"]))
        for key in (
            "mean/reward",
            "resolved_task_count",
            "task_count",
            "resolved_task_rate",
            "eval_error_rate",
            "tests_status_rate",
        ):
            if key in agent_metrics:
                key_metrics[key] = agent_metrics[key]
        return key_metrics

    async def _verify_patch_via_server(
        self,
        *,
        instance: dict[str, Any],
        patch: str,
        instance_id: str,
        subset: str,
        split: str,
        responses_create_params: NeMoGymResponseCreateParamsNonStreaming,
    ) -> dict[str, Any]:
        """POST the agent's patch to the shared swe_env verifier and return its eval subset.

        Builds a verify request carrying the per-task metadata the verifier reads (instance_id, image,
        base_commit, repo_workdir, test_command, test_framework, test_patch, fail_to_pass, pass_to_pass,
        benchmark, split) plus the patch in ``response.metadata.model_patch``, and POSTs it to the
        configured verifier server via the server client. On any transport failure it returns a masked
        subset (``resolved=False``, ``error_kind='sandbox'``) rather than raising, so the agent always
        emits a present (masked) row instead of dropping the rollout.

        Args:
            instance: The task instance dict holding fields such as FAIL_TO_PASS, PASS_TO_PASS,
                base_commit, test_command, test_framework, and test_patch.
            patch: The model-produced diff to grade.
            instance_id: The SWE-bench instance identifier.
            subset: The dataset subset (e.g. "verified"), used to resolve the Docker image name.
            split: The dataset split passed through to the verifier.
            responses_create_params: The Responses API params used to build the verify request.

        Returns:
            The verifier's eval subset as a dict; on failure, a masked dict with ``resolved=False``,
            ``error_kind='sandbox'``, and ``patch_exists`` reflecting whether a patch was present.
        """

        def _as_list(value: Any) -> list[str]:
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return [value]
            return value or []

        f2p = _as_list(instance.get("FAIL_TO_PASS"))
        p2p = _as_list(instance.get("PASS_TO_PASS"))
        # Forward the instance's own per-framework eval command and framework when present. Fall back
        # to the conda+pytest default only when the row ships no test_command. This supports both
        # SWE-bench-Verified (no per-row command) and multi-framework rows (cargo/go/npm/...) that
        # carry their own command and framework.
        test_framework = instance.get("test_framework", "") or ""
        test_command = instance.get("test_command", "") or ""
        if not test_command:
            nodeids = " ".join("'" + n + "'" for n in f2p + p2p)
            test_command = (
                "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && "
                f"python -m pytest -rA {nodeids}"
            )
        task_metadata = {
            "instance_id": instance_id,
            "image": _swebench_image_name(instance, subset),
            "base_commit": instance.get("base_commit", "") or "",
            "repo_workdir": "/testbed",
            "test_command": test_command,
            "test_framework": test_framework,
            "test_patch": instance.get("test_patch", "") or "",
            "fail_to_pass": f2p,
            "pass_to_pass": p2p,
            # "swe-bench-ext" is the flat host-graded harness the conda/pytest test_command above
            # targets; it is the registered swe_env harness key, not the dataset subset.
            "benchmark": "swe-bench-ext",
            "split": split,
        }
        verify_request = {
            "responses_create_params": responses_create_params.model_dump(exclude_none=True)
            | {"metadata": task_metadata},
            "response": {
                "id": f"mini-swe-{instance_id}",
                "created_at": int(time.time()),
                "model": responses_create_params.model,
                "object": "response",
                "output": [],
                "metadata": {"model_patch": patch},
            },
        }

        async def _do_verify() -> dict[str, Any]:
            verify_response = await self.server_client.post(
                server_name=self.config.verifier_server_name,
                url_path="/verify",
                json=verify_request,
            )
            await raise_for_status(verify_response)
            return await get_response_json(verify_response)

        # Bound the whole call (including ServerClient's unbounded disconnect-retry loop) so a
        # hung or retried verify cannot pin a rollout slot forever.
        verify_timeout_s = float(getattr(self.config, "eval_timeout", None) or 900) + 900
        try:
            return await asyncio.wait_for(_do_verify(), timeout=verify_timeout_s)
        except Exception as e:  # noqa: BLE001  (incl. asyncio.TimeoutError -> masked, never pins a slot)
            print(f"Verifier POST failed for {instance_id}: {e}", flush=True)
            return {"resolved": False, "error_kind": "sandbox", "patch_exists": bool(patch)}

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        """Responses endpoint placeholder; this agent drives generation through its own run loop.

        Args:
            body: The Responses API create params.

        Raises:
            NotImplementedError: Always, since this agent does not serve standalone responses.
        """
        raise NotImplementedError

    async def run(self, body: MiniSWEAgentRunRequest) -> MiniSWEAgentVerifyResponse:
        """Run a single SWE-bench task end to end and return a scored verify response.

        Resolves the model server, builds the mini-swe-agent config, dispatches the task to a Ray
        sandbox worker, scores the resulting patch (via the shared verifier or the in-worker harness),
        assembles the Responses API response, persists the result, and returns it.

        Args:
            body: The run request carrying the instance, dataset subset/split, and Responses API
                create params.

        Returns:
            A verify response with the reward, the assembled response object, the instance id, and the
            eval report metadata.
        """
        async with self.sem:
            model_server_name = self.config.model_server.name
            global_config_dict = ServerClient.load_from_global_config().global_config_dict

            model_server_config = get_first_server_config_dict(
                global_config_dict,
                model_server_name,
            )

            policy_model_name = global_config_dict["policy_model_name"]

            ##### MINI-SWE-AGENT CONFIG #####
            subset = body.subset
            split = body.split
            workers = 1
            run_golden = self.config.run_golden
            base_url = f"http://{model_server_config['host']}:{model_server_config['port']}/v1"
            dummy_key = "dummy_key"
            model_name = f"hosted_vllm/{policy_model_name}"
            step_timeout = self.config.step_timeout
            eval_timeout = self.config.eval_timeout
            step_limit = self.config.step_limit

            instance_id = body.instance_id

            mini_swe_config_path = _swebench_config_path()
            config = yaml.safe_load(get_config_path(mini_swe_config_path).read_text())
            responses_create_params_dict = body.responses_create_params.model_dump(exclude_none=True)

            default_model_kwargs = config["model"]["model_kwargs"]
            temperature = (
                body.responses_create_params.temperature
                if body.responses_create_params.temperature is not None
                else default_model_kwargs["temperature"]
            )
            top_p = (
                body.responses_create_params.top_p
                if body.responses_create_params.top_p is not None
                else default_model_kwargs["top_p"]
            )
            model_kwargs = _responses_create_params_to_model_kwargs(
                responses_create_params_dict,
                default_tool_choice=self.config.tool_choice,
            )
            if model_kwargs:
                config.setdefault("model", {}).setdefault("model_kwargs", {}).update(model_kwargs)

            output_file_dir = f"{Path.cwd()}/results/{subset}/{policy_model_name}"
            config_path = mini_swe_config_path
            should_write_config = bool(model_kwargs)
            if self.config.sandbox_provider is None:
                raise ValueError("mini_swe_agent_2 requires sandbox_provider")
            config.setdefault("environment", {}).update(self.config.sandbox_environment_kwargs or {})
            config["environment"]["provider"] = _sandbox_provider_for_config_dump(self.config.sandbox_provider)
            config["environment"]["spec"] = _sandbox_spec_for_instance(
                self.config.sandbox_spec,
                resource_profiles=self.config.sandbox_resource_profiles,
                instance_id=instance_id,
            )
            should_write_config = True

            if should_write_config:
                config_output_dir = Path(output_file_dir) / "_configs"
                config_output_dir.mkdir(parents=True, exist_ok=True)
                config_path = config_output_dir / f"{instance_id}.sandbox.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))

            if self.config.skip_if_exists:
                if Path(f"{output_file_dir}/{instance_id}/{instance_id}.json").exists():
                    with open(f"{output_file_dir}/{instance_id}/{instance_id}.json", "r") as f:
                        print(f"Skipping {instance_id} because it already exists")
                        verify_response = MiniSWEAgentVerifyResponse.model_validate_json(f.read())
                    return verify_response

            #### RUN MINI-SWE-AGENT #####
            try:
                params = dict(
                    subset=subset,
                    split=split,
                    workers=workers,
                    output=output_file_dir,
                    model=model_name,
                    api_key=dummy_key,
                    base_url=base_url,
                    env="sandbox",
                    run_golden=run_golden,
                    instance_id=instance_id,
                    config=config_path,
                    # TODO: add this later
                    instance_dict=body.model_dump(),
                    responses_create_params=json.dumps(responses_create_params_dict),
                    step_timeout=step_timeout,
                    eval_timeout=eval_timeout,
                    step_limit=step_limit,
                    eval_via_verifier=self.config.eval_via_verifier,
                )
                runner = runner_ray_remote
                runtime_env = _sandbox_runtime_env(self.config.sandbox_provider)
                if runtime_env.get("env_vars"):
                    runner = runner.options(runtime_env=runtime_env)
                future = runner.remote(run_mini_swe_with_sandbox, params)
                result = await asyncio.to_thread(ray.get, future)
                result = result[instance_id]
                input_messages = result["input_messages"]
                response_output = result["response_output"]
                responses = result["responses"]

                if self.config.eval_via_verifier:
                    # Score the patch by POSTing to the shared swe_env verifier instead of grading via
                    # the in-worker swebench harness. The Ray worker still produced the trajectory and
                    # patch; the verifier is the authoritative grader here.
                    in_worker_report = result.get("eval_report") or {}
                    patch = in_worker_report.get("model_patch", "") or ""
                    eval_subset = await self._verify_patch_via_server(
                        instance=body.model_dump(),
                        patch=patch,
                        instance_id=instance_id,
                        subset=subset,
                        split=split,
                        responses_create_params=body.responses_create_params,
                    )
                    reward = 1.0 if eval_subset.get("resolved") else 0.0
                    result["eval_report"] = eval_subset
                else:
                    reward = 1.0 if _is_resolved(instance_id, result["eval_report"]) else 0.0

            except Exception as e:
                error_info = {"error": str(e), "traceback": traceback.format_exc()}
                print(f"Error running mini-swe-agent: {e}\n{error_info['traceback']}", flush=True)
                result = {"eval_report": error_info}
                input_messages = []
                response_output = []
                responses = []
                reward = 0.0

            body.responses_create_params.input = input_messages
            response = _default_response_object()
            if responses:
                response.update(dict(responses[-1]))
            response.pop("extra", None)
            response["model"] = policy_model_name
            response["temperature"] = temperature
            response["top_p"] = top_p
            response["output"] = response_output

            verify_response = MiniSWEAgentVerifyResponse(
                responses_create_params=body.responses_create_params,
                reward=reward,
                response=response,
                instance_id=instance_id,
                metadata=result.get("eval_report", {}) if result else {},
            )

            output_path = Path(f"{output_file_dir}/{instance_id}")
            output_path.mkdir(parents=True, exist_ok=True)

            with open(f"{output_file_dir}/{instance_id}/{instance_id}.json", "w") as f:
                json.dump(verify_response.model_dump(), f)

            return verify_response


if __name__ == "__main__":
    MiniSWEAgent.run_webserver()
