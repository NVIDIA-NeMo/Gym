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
from nemo_gym.sandbox.observability import (
    build_recorder_from_config,
    event_context,
    observability_sync_span,
    use_recorder,
)
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
    observability: Optional[dict[str, Any]] = None


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
    """Convert Gym Responses API rollout params into mini-swe-agent chat-completions kwargs."""
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


def _bash_tool_choice() -> dict[str, Any]:
    return {"type": "function", "function": {"name": "bash"}}


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


def _format_template(value: Any, context: dict[str, Any]) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return value.format(**context)
    except (KeyError, ValueError, IndexError):
        return value


def _observability_config_for_instance(
    config: dict[str, Any] | None,
    *,
    instance_id: str,
    task_index: Any = None,
    rollout_index: Any = None,
) -> dict[str, Any] | None:
    if not isinstance(config, dict):
        return None

    trajectory_id = str(config.get("trajectory_id") or instance_id)
    if rollout_index is not None:
        try:
            rollout_suffix = f"{int(rollout_index) + 1:02d}"
        except (TypeError, ValueError):
            rollout_suffix = str(rollout_index)
        trajectory_id = f"{trajectory_id}__rollout{rollout_suffix}"

    context = {
        "instance_id": instance_id,
        "task_index": "" if task_index is None else task_index,
        "rollout_index": "" if rollout_index is None else rollout_index,
        "trajectory_id": trajectory_id,
    }
    formatted = dict(config)
    formatted["trajectory_id"] = trajectory_id
    for key in ("output_dir", "run_id", "run_span_name", "job_name"):
        if key in formatted:
            formatted[key] = _format_template(formatted[key], context)

    otel = dict(formatted.get("otel") or {})
    for key in ("service_name", "run_span_name", "job_name"):
        if key in otel:
            otel[key] = _format_template(otel[key], context)
    if "resource_attributes" in otel and isinstance(otel["resource_attributes"], dict):
        otel["resource_attributes"] = {
            resource_key: _format_template(resource_value, context)
            for resource_key, resource_value in otel["resource_attributes"].items()
        }
    if otel:
        formatted["otel"] = otel
    return formatted


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
        model = _ObservedModel(model, model_name=model_config["model_name"])
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
    observability_config = params.pop("observability", None)
    recorder = build_recorder_from_config(
        observability_config,
        run_id=observability_config.get("run_id") if isinstance(observability_config, dict) else None,
    )
    try:
        with use_recorder(recorder):
            with event_context(
                trajectory_id=observability_config.get("trajectory_id", instance_id)
                if isinstance(observability_config, dict)
                else instance_id,
                instance_id=instance_id,
                harness="mini_swe_agent_2",
                environment_type="sandbox",
            ):
                return _run_swegym_v2(**params)
    finally:
        if recorder is not None:
            recorder.finalize()


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
            extra_fields = getattr(body, "__pydantic_extra__", {}) or {}

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
                    observability=_observability_config_for_instance(
                        self.config.observability,
                        instance_id=instance_id,
                        task_index=extra_fields.get("_ng_task_index"),
                        rollout_index=extra_fields.get("_ng_rollout_index"),
                    ),
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
