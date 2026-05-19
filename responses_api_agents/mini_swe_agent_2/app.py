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
from nemo_gym.sandbox.observability import event_context, observability_sync_span
from nemo_gym.server_utils import (
    ServerClient,
    get_first_server_config_dict,
)


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
    """Convert Gym Responses API rollout params into mini-swe-agent Responses API kwargs."""
    model_kwargs: dict[str, Any] = {}
    for key in ("temperature", "top_p", "top_logprobs", "store", "parallel_tool_calls"):
        value = params.get(key)
        if value is not None:
            model_kwargs[key] = value

    max_output_tokens = params.get("max_output_tokens")
    if max_output_tokens is not None:
        model_kwargs["max_output_tokens"] = max_output_tokens

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
    return {"type": "function", "name": "bash"}


class _ObservedModel:
    """Add an OTel span around each mini-SWE model query."""

    def __init__(self, model: Any, *, model_name: str) -> None:
        self._model = model
        self._model_name = model_name

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        model_kwargs = getattr(getattr(self._model, "config", None), "model_kwargs", None)
        model_kwargs = model_kwargs if isinstance(model_kwargs, dict) else {}
        attributes = {
            "model": self._model_name,
            "message_count": len(messages),
            "temperature": kwargs.get("temperature", model_kwargs.get("temperature")),
            "top_p": kwargs.get("top_p", model_kwargs.get("top_p")),
            "max_tokens": kwargs.get(
                "max_output_tokens",
                kwargs.get("max_tokens", model_kwargs.get("max_output_tokens", model_kwargs.get("max_tokens"))),
            ),
            "tool_choice": kwargs.get("tool_choice", model_kwargs.get("tool_choice")),
            "_record_exception_stacktrace": False,
        }
        with observability_sync_span("llm.request", phase="llm", attributes=attributes):
            return self._model.query(messages, **kwargs)


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


def _strip_extra(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        item = item.model_dump()
    if not isinstance(item, dict):
        return {"type": "message", "role": "user", "content": str(item)}
    return {key: value for key, value in item.items() if key != "extra"}


def _split_trajectory_for_responses(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
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
        elif message.get("type") == "function_call_output":
            output_items.append(_strip_extra(message))

    return input_messages, output_items, raw_responses


def _default_response_object() -> dict[str, Any]:
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


def _response_from_rollout(
    *,
    model_name: str,
    output_items: list[dict[str, Any]],
    raw_responses: list[dict[str, Any]],
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    response = _default_response_object()
    if raw_responses:
        response.update({key: value for key, value in raw_responses[-1].items() if key != "extra"})
    response["model"] = model_name
    response["temperature"] = temperature
    response["top_p"] = top_p
    response["output"] = output_items
    return response


def _is_resolved(instance_id: str, eval_report: dict[str, Any]) -> bool:
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
    model_config["model_class"] = "litellm_response"
    model_config["model_name"] = params["model"]
    model_config.setdefault("cost_tracking", "ignore_errors")
    model_kwargs = model_config.setdefault("model_kwargs", {})
    model_kwargs["api_key"] = params["api_key"]
    model_kwargs["base_url"] = params["base_url"]
    max_tokens = model_kwargs.pop("max_tokens", None)
    if max_tokens is not None and "max_output_tokens" not in model_kwargs:
        model_kwargs["max_output_tokens"] = max_tokens

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
        model = _ObservedModel(model, model_name=params["model"])
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


def run_swegym_with_optional_sandbox(**params: Any) -> Any:
    instance_id = str(params.get("instance_id") or "unknown")
    with event_context(
        trajectory_id=instance_id,
        instance_id=instance_id,
        harness="mini_swe_agent_2",
        environment_type="sandbox",
    ):
        return _run_swegym_v2(**params)


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
        return app

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError

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
                    sandbox_ready_barrier_count=self.config.sandbox_ready_barrier_count,
                    sandbox_ready_barrier_id=self.config.sandbox_ready_barrier_id,
                    sandbox_ready_barrier_timeout_s=self.config.sandbox_ready_barrier_timeout_s,
                    sandbox_ready_barrier_poll_s=self.config.sandbox_ready_barrier_poll_s,
                )
                future = runner_ray_remote.remote(run_swegym_with_optional_sandbox, params)
                result = await asyncio.to_thread(ray.get, future)
                result = result[instance_id]
                input_messages = result["input_messages"]
                response_output = result["response_output"]
                responses = result["responses"]
                reward = 1.0 if _is_resolved(instance_id, result["eval_report"]) else 0.0

            except Exception as e:
                error_info = {"error": str(e), "traceback": traceback.format_exc()}
                print(f"Error running swegym: {e}\n{error_info['traceback']}", flush=True)
                result = {"eval_report": error_info}
                input_messages = []
                response_output = []
                responses = []
                reward = 0.0

            body.responses_create_params.input = input_messages
            response = _response_from_rollout(
                model_name=policy_model_name,
                output_items=response_output,
                raw_responses=responses,
                temperature=temperature,
                top_p=top_p,
            )

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
