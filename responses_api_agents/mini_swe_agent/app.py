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
import asyncio
import hashlib
import json
import sys
import time
import traceback
from asyncio import Semaphore
from os import environ, getenv, makedirs
from pathlib import Path
from typing import Any, Callable, Literal, Optional, cast
from uuid import uuid4

import ray
import yaml
from fastapi import Body, FastAPI
from minisweagent.config import builtin_config_dir, get_config_path
from pydantic import ConfigDict

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
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.sandbox.observability import event_context, record_event
from nemo_gym.server_utils import (
    ServerClient,
    get_first_server_config_dict,
    get_response_json,
    raise_for_status,
)
from nemo_gym.server_utils import (
    request as server_request,
)
from responses_api_agents.mini_swe_agent.utils import MiniSWEAgentUtils


try:
    from minisweagent.run.extra.swegym_runner import _main as run_swegym_v1
except ModuleNotFoundError:  # mini-swe-agent v2 moved the benchmark runner.
    run_swegym_v1 = None


class MiniSWEAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    env: Literal["docker", "singularity", "sandbox"]
    concurrency: int
    cache_dir_template: Optional[str] = None
    sandbox_provider: Optional[dict[str, Any]] = None
    sandbox_spec: Optional[dict[str, Any]] = None
    sandbox_environment_kwargs: Optional[dict[str, Any]] = None
    run_golden: bool = False
    step_timeout: int = 600
    eval_timeout: int = 1800
    skip_if_exists: bool = False
    step_limit: int = 250
    collapse_limit: int = 3
    runner_num_cpus: float = 1.0
    agentic_router_program_id: bool = False
    agentic_router_program_id_prefix: str = "mini_swe"
    agentic_router_release_program: bool = True
    tool_choice: Optional[str | dict[str, Any]] = None
    auto_tool_retry: bool = False
    sandbox_resource_profiles: Optional[list[dict[str, str]]] = None
    sandbox_ready_barrier_count: Optional[int] = None
    sandbox_ready_barrier_id: Optional[str] = None
    sandbox_ready_barrier_timeout_s: int = 1800
    sandbox_ready_barrier_poll_s: float = 2.0


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
    return runner(**params)


def _uses_sandbox_env(env: str) -> bool:
    return env == "sandbox"


def _json_dict_from_metadata(value: Any, *, field_name: str) -> dict[str, Any]:
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
    """Convert Responses API rollout params into mini-swe-agent LiteLLM kwargs."""
    model_kwargs: dict[str, Any] = {}
    for key in ("temperature", "top_p", "top_logprobs", "store", "parallel_tool_calls"):
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


def _bash_tool_choice() -> dict[str, Any]:
    return {"type": "function", "function": {"name": "bash"}}


def _is_missing_tool_call_error(error: Exception) -> bool:
    if type(error).__name__ != "FormatError":
        return False

    for message in getattr(error, "messages", ()):
        if not isinstance(message, dict):
            continue
        if message.get("extra", {}).get("interrupt_type") != "FormatError":
            continue
        if "No tool calls found" in str(message.get("content", "")):
            return True
    return False


def _single_registered_tool_choice(model: Any) -> Optional[dict[str, Any]]:
    """Return a named tool choice only when the underlying model has a known single tool."""
    model_class = type(model)
    if model_class.__module__ == "minisweagent.models.litellm_model" and model_class.__name__ == "LitellmModel":
        return _bash_tool_choice()
    return None


class _AutoToolRetryModel:
    """Retry one mini-SWE auto-mode no-tool response with the registered single tool.

    vLLM returns 500 for `tool_choice=required` on the current Qwen3.5 stack. Keeping `auto` as the public/default
    choice preserves multi-tool routing, while this wrapper handles the one-tool mini-SWE v2 compatibility case.
    """

    _missing = object()

    def __init__(self, model: Any) -> None:
        self._model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        try:
            return self._model.query(messages, **kwargs)
        except Exception as error:
            config = getattr(self._model, "config", None)
            model_kwargs = getattr(config, "model_kwargs", None)
            if not isinstance(model_kwargs, dict) or model_kwargs.get("tool_choice") != "auto":
                raise
            single_tool_choice = _single_registered_tool_choice(self._model)
            if single_tool_choice is None or not _is_missing_tool_call_error(error):
                raise

            old_tool_choice = model_kwargs.get("tool_choice", self._missing)
            model_kwargs["tool_choice"] = single_tool_choice
            try:
                return self._model.query(messages, **kwargs)
            finally:
                if old_tool_choice is self._missing:
                    model_kwargs.pop("tool_choice", None)
                else:
                    model_kwargs["tool_choice"] = old_tool_choice


def _agentic_router_program_id(prefix: str, instance_id: str) -> str:
    if not prefix or instance_id.startswith(f"{prefix}:"):
        return instance_id
    return f"{prefix}:{instance_id}"


def _barrier_file_name(instance_id: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in instance_id)[:180] or "unknown"


def _sandbox_spec_for_instance(
    spec: dict[str, Any] | None,
    *,
    resource_profiles: list[dict[str, str]] | None,
    instance_id: str,
) -> dict[str, Any]:
    instance_spec = dict(spec or {})
    if not resource_profiles:
        return instance_spec

    resources = dict(instance_spec.get("resources") or {})
    digest = hashlib.sha256(instance_id.encode("utf-8")).digest()
    profile = resource_profiles[int.from_bytes(digest[:4], "big") % len(resource_profiles)]
    resources.update(profile)
    instance_spec["resources"] = resources
    return instance_spec


def _wait_for_sandbox_ready_barrier(
    *,
    output_dir: Path,
    barrier_id: str,
    instance_id: str,
    count: int,
    timeout_s: float,
    poll_s: float,
) -> None:
    if count <= 1:
        return

    barrier_dir = output_dir / "_sandbox_ready_barriers" / _barrier_file_name(barrier_id)
    barrier_dir.mkdir(parents=True, exist_ok=True)
    ready_path = barrier_dir / f"{_barrier_file_name(instance_id)}.ready"
    ready_path.write_text(json.dumps({"instance_id": instance_id, "ready_at_s": time.time()}))

    deadline = time.monotonic() + timeout_s
    last_reported = -1
    while True:
        ready_count = sum(1 for _ in barrier_dir.glob("*.ready"))
        if ready_count >= count:
            print(
                f"[EVAL]{instance_id} Sandbox-ready barrier satisfied: {ready_count}/{count}",
                flush=True,
            )
            return

        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(
                f"Timed out waiting for sandbox-ready barrier {barrier_id}: "
                f"{ready_count}/{count} ready after {timeout_s:.1f}s"
            )

        if ready_count != last_reported and (ready_count == 1 or ready_count % 25 == 0):
            print(
                f"[EVAL]{instance_id} Waiting for sandbox-ready barrier: {ready_count}/{count}",
                flush=True,
            )
            last_reported = ready_count
        time.sleep(max(poll_s, 0.1))


def _swebench_config_path() -> Path:
    for candidate in (
        builtin_config_dir / "extra" / "swebench.yaml",
        builtin_config_dir / "benchmarks" / "swebench.yaml",
    ):
        if candidate.exists():
            return candidate
    return builtin_config_dir / "extra" / "swebench.yaml"


def _swebench_image_name(instance: dict[str, Any], subset: str) -> str:
    image_name = instance.get("image_name")
    if image_name:
        return str(image_name)

    instance_id = instance["instance_id"]
    if subset == "verified":
        docker_compatible_id = instance_id.replace("__", "_1776_")
        return f"swebench/sweb.eval.x86_64.{docker_compatible_id}:latest".lower()

    docker_compatible_id = instance_id.replace("__", "_s_")
    return f"xingyaoww/sweb.eval.x86_64.{docker_compatible_id}:latest".lower()


def _message_content_to_text(content: Any) -> str:
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


def _run_eval_v2(
    *,
    instance: dict[str, Any],
    env: Any,
    model_patch: str,
    instance_dir: Path,
    run_id: str,
    is_golden: bool,
) -> dict[str, Any]:
    from swegym.harness.constants import SWEbenchInstance
    from swegym.harness.docker_build import setup_logger
    from swegym.harness.grading import get_eval_report
    from swegym.harness.test_spec import make_test_spec

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
        log_path=test_output_path,
        include_tests_status=True,
    )
    print(f"[EVAL]{test_spec.instance_id} Result: resolved: {report[test_spec.instance_id]['resolved']}", flush=True)

    report_path.write_text(json.dumps(report, indent=4))
    return {
        "instance_id": test_spec.instance_id,
        "model_patch": model_patch,
        "eval_report": report,
    }


def _run_swegym_v2(**params: Any) -> dict[str, Any]:
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
    model_config = config.setdefault("model", {})
    model_config["model_name"] = params["model"]
    model_config.setdefault("cost_tracking", "ignore_errors")
    model_kwargs = model_config.setdefault("model_kwargs", {})
    model_kwargs["api_key"] = params["api_key"]
    model_kwargs["base_url"] = params["base_url"]
    max_output_tokens = model_kwargs.pop("max_output_tokens", None)
    if max_output_tokens is not None and "max_tokens" not in model_kwargs:
        model_kwargs["max_tokens"] = max_output_tokens

    environment_config = config.setdefault("environment", {})
    environment_config["image"] = _swebench_image_name(instance, params["subset"])
    environment_config["step_timeout"] = params["step_timeout"]
    environment_config["eval_timeout"] = params["eval_timeout"]
    environment_config["instance_id"] = instance_id
    if _uses_sandbox_env(params["env"]):
        environment_config["environment_class"] = (
            "responses_api_agents.mini_swe_agent.sandbox_environment.MiniSWESandboxEnvironment"
        )
    else:
        environment_config["environment_class"] = params["env"]

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
        barrier_id = params.get("sandbox_ready_barrier_id")
        barrier_count = params.get("sandbox_ready_barrier_count")
        if barrier_id and barrier_count:
            _wait_for_sandbox_ready_barrier(
                output_dir=output_dir,
                barrier_id=str(barrier_id),
                instance_id=instance_id,
                count=int(barrier_count),
                timeout_s=float(params.get("sandbox_ready_barrier_timeout_s", 1800)),
                poll_s=float(params.get("sandbox_ready_barrier_poll_s", 2.0)),
            )

        model = get_model(config=model_config)
        if params.get("auto_tool_retry", False):
            model = _AutoToolRetryModel(model)
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

        messages = []
        responses = []
        for message in data.get("messages", []):
            role = message.get("role")
            if role == "assistant":
                response = message.get("extra", {}).get("response")
                if response:
                    responses.append(response)
            if role in {"system", "user", "assistant"}:
                messages.append({"role": role, "content": _message_content_to_text(message.get("content"))})

        return {
            instance_id: {
                "messages": messages,
                "responses": responses,
                "eval_report": eval_report,
                "exit_status": exit_status,
            }
        }
    finally:
        if env and hasattr(env, "cleanup"):
            env.cleanup()


def run_swegym_with_optional_sandbox(**params: Any) -> Any:
    if _uses_sandbox_env(params.get("env", "")):
        try:
            from minisweagent.environments import ENV_MAP

            from responses_api_agents.mini_swe_agent.sandbox_environment import MiniSWESandboxEnvironment

            ENV_MAP["sandbox"] = MiniSWESandboxEnvironment
        except ImportError:
            pass

    instance_id = str(params.get("instance_id") or "unknown")
    start_s = time.monotonic()
    with event_context(
        trajectory_id=instance_id,
        instance_id=instance_id,
        harness="mini_swe_agent",
        environment_type=str(params.get("env") or "unknown"),
    ):
        try:
            if run_swegym_v1 is not None:
                result = run_swegym_v1(**params)
            else:
                result = _run_swegym_v2(**params)
        except Exception:
            record_event(
                "trajectory",
                "trajectory.complete",
                attributes={
                    "reward": 0.0,
                    "stop_reason": "error",
                    "duration_s": time.monotonic() - start_s,
                    "loss_multiplier": 1.0,
                },
            )
            raise

        reward = 0.0
        stop_reason = "complete"
        try:
            instance_result = result.get(instance_id, {}) if isinstance(result, dict) else {}
            if not isinstance(instance_result, dict):
                stop_reason = "missing_result"
            else:
                eval_report = instance_result.get("eval_report", {})
                reward = 1.0 if MiniSWEAgentUtils.is_resolved(instance_id, eval_report) else 0.0
        except Exception:
            reward = 0.0
            stop_reason = "reward_parse_error"

        record_event(
            "trajectory",
            "trajectory.complete",
            attributes={
                "reward": reward,
                "stop_reason": stop_reason,
                "duration_s": time.monotonic() - start_s,
                "loss_multiplier": 1.0,
            },
        )
        return result


class MiniSWEAgent(SimpleResponsesAPIAgent):
    config: MiniSWEAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        return app

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError

    async def _release_agentic_router_program(
        self,
        model_server_config: dict[str, Any],
        program_id: str,
    ) -> dict[str, Any]:
        url = f"http://{model_server_config['host']}:{model_server_config['port']}/agentic_router/release"
        response = await server_request("POST", url, json={"program_id": program_id})
        await raise_for_status(response)
        return await get_response_json(response)

    async def run(self, body: MiniSWEAgentRunRequest) -> MiniSWEAgentVerifyResponse:
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
            cache_dir_template = self.config.cache_dir_template
            run_golden = self.config.run_golden
            base_url = f"http://{model_server_config['host']}:{model_server_config['port']}/v1"
            dummy_key = "dummy_key"
            model_name = f"hosted_vllm/{policy_model_name}"
            step_timeout = self.config.step_timeout
            eval_timeout = self.config.eval_timeout
            env = self.config.env
            step_limit = self.config.step_limit
            collapse_limit = self.config.collapse_limit

            instance_id = body.instance_id
            agentic_program_id = None

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
            if self.config.agentic_router_program_id:
                agentic_program_id = _agentic_router_program_id(
                    self.config.agentic_router_program_id_prefix,
                    instance_id,
                )
                extra_body = model_kwargs.setdefault("extra_body", {})
                extra_body.setdefault("program_id", agentic_program_id)
            if model_kwargs:
                config.setdefault("model", {}).setdefault("model_kwargs", {}).update(model_kwargs)

            output_file_dir = f"{Path.cwd()}/results/{subset}/{policy_model_name}"
            config_path = mini_swe_config_path
            should_write_config = bool(model_kwargs)
            if _uses_sandbox_env(env):
                if self.config.sandbox_provider is None:
                    raise ValueError("env=sandbox requires sandbox_provider")
                config.setdefault("environment", {}).update(self.config.sandbox_environment_kwargs or {})
                config["environment"]["provider"] = self.config.sandbox_provider
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

            env_vars = environ.copy()
            if env == "singularity":
                slurm_job_id = getenv("SLURM_JOB_ID", str(uuid4()))
                env_vars.update(
                    {
                        "SINGULARITY_CACHEDIR": f"/tmp/singularity_cache_${slurm_job_id}_$$",
                        "APPTAINER_CACHEDIR": f"/tmp/apptainer_cache_${slurm_job_id}_$$",
                        "SINGULARITY_TMPDIR": f"/tmp/singularity_tmp_${slurm_job_id}_$$",
                        "APPTAINER_TMPDIR": f"/tmp/apptainer_tmp_${slurm_job_id}_$$",
                    }
                )
                for var in [
                    "SINGULARITY_CACHEDIR",
                    "APPTAINER_CACHEDIR",
                    "SINGULARITY_TMPDIR",
                    "APPTAINER_TMPDIR",
                ]:
                    makedirs(env_vars[var], exist_ok=True)

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
                    cache_dir_template=cache_dir_template,
                    env=env,
                    run_golden=run_golden,
                    instance_id=instance_id,
                    config=config_path,
                    # TODO: add this later
                    instance_dict=body.model_dump(),
                    responses_create_params=json.dumps(responses_create_params_dict),
                    step_timeout=step_timeout,
                    eval_timeout=eval_timeout,
                    step_limit=step_limit,
                    collapse_limit=collapse_limit,
                    auto_tool_retry=self.config.auto_tool_retry,
                    sandbox_ready_barrier_count=self.config.sandbox_ready_barrier_count,
                    sandbox_ready_barrier_id=self.config.sandbox_ready_barrier_id,
                    sandbox_ready_barrier_timeout_s=self.config.sandbox_ready_barrier_timeout_s,
                    sandbox_ready_barrier_poll_s=self.config.sandbox_ready_barrier_poll_s,
                )
                future = runner_ray_remote.options(num_cpus=self.config.runner_num_cpus).remote(
                    run_swegym_with_optional_sandbox,
                    params,
                )
                result = await asyncio.to_thread(ray.get, future)
                result = result[instance_id]
                messages = result["messages"]
                responses = result["responses"]
                reward = 1.0 if MiniSWEAgentUtils.is_resolved(instance_id, result["eval_report"]) else 0.0

            except Exception as e:
                error_info = {"error": str(e), "traceback": traceback.format_exc()}
                print(f"Error running swegym: {e}\n{error_info['traceback']}", flush=True)
                result = {"eval_report": error_info}
                messages = []
                responses = []
                reward = 0.0

            agentic_router_release = None
            if agentic_program_id and self.config.agentic_router_release_program:
                try:
                    agentic_router_release = await self._release_agentic_router_program(
                        model_server_config=model_server_config,
                        program_id=agentic_program_id,
                    )
                except Exception as e:
                    agentic_router_release = {"released": False, "error": f"{type(e).__name__}: {e}"}
                    print(
                        f"[agentic_router_release_failed program_id={agentic_program_id} error={agentic_router_release['error']}]",
                        flush=True,
                    )

            # The first two messages are the system and user message generated by the harness
            # TODO(sugam): what if the user only provides the system/user message
            body.responses_create_params.input = messages[:2]

            response = MiniSWEAgentUtils.get_default_response_object()
            response["model"] = policy_model_name
            response["temperature"] = temperature
            response["top_p"] = top_p

            # Wrap output messages in responses format
            response["output"] = MiniSWEAgentUtils.chat_cmp_to_responses(messages[2:], responses)

            verify_response = MiniSWEAgentVerifyResponse(
                responses_create_params=body.responses_create_params,
                reward=reward,
                response=response,
                instance_id=instance_id,
                metadata=(
                    (result.get("eval_report", {}) if result else {})
                    | ({"agentic_router_release": agentic_router_release} if agentic_router_release else {})
                ),
            )

            output_path = Path(f"{output_file_dir}/{instance_id}")
            output_path.mkdir(parents=True, exist_ok=True)

            with open(f"{output_file_dir}/{instance_id}/{instance_id}.json", "w") as f:
                json.dump(verify_response.model_dump(), f)

            return verify_response


if __name__ == "__main__":
    MiniSWEAgent.run_webserver()
