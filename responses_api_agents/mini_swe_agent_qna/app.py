# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Mini-SWE-Agent harness for SWE-Atlas Codebase QnA.

Forked from ``mini_swe_agent_2`` but specialized for open-ended codebase QnA
rather than SWE-Bench patch/test tasks:

- Drives the mini-swe-agent bash loop in a Gym sandbox (e.g. Apptainer ``.sif``
  images on a cluster) using a QnA-specific mini-swe config template.
- The task ends when the agent writes its answer to ``answer_path``
  (``/logs/agent/answer.txt`` by default, wrapped in ``<<FINAL_ANSWER>>`` tags)
  and submits with ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``. This harness reads
  that answer file out of the sandbox — there is no git patch.
- Reward is delegated to the ``swe_atlas_qna`` rubric-judge resources server's
  ``/verify`` (the answer is graded against the task's rubrics), keeping the
  verification logic in one tested place.
"""

import asyncio
import json
import sys
import time
import traceback
from asyncio import Semaphore
from pathlib import Path
from typing import Any, Callable, Literal, Optional
from uuid import uuid4

import ray
import yaml
from fastapi import Body, FastAPI
from minisweagent.config import get_config_path
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
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.sandbox import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import (
    ServerClient,
    get_first_server_config_dict,
    get_response_json,
    raise_for_status,
)


# Default filesystem path (inside the sandbox) the agent writes its final answer to.
DEFAULT_ANSWER_PATH = "/logs/agent/answer.txt"


class MiniSWEAgentQnaConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    # The swe_atlas_qna rubric-judge resources server that scores the answer.
    resources_server: ResourcesServerRef
    env: Literal["sandbox"] = "sandbox"
    concurrency: int = 16
    # A sandbox name resolved from a separate provider config (e.g. "sandbox"),
    # or an inline single-key provider mapping ({provider_name: {...}}).
    sandbox_provider: Optional[str | dict[str, Any]] = None
    sandbox_spec: Optional[dict[str, Any]] = None
    sandbox_environment_kwargs: Optional[dict[str, Any]] = None
    # QnA-specific mini-swe-agent config template (system/instance/observation).
    mini_swe_config_path: str = "responses_api_agents/mini_swe_agent_qna/configs/mswea_qa_config.yaml"
    # Python .format template for the sandbox image, filled from the row's
    # verifier_metadata (e.g. "/cluster/sifs/{sif_basename}" or "{docker_image}").
    # When None, the row's ``docker_image`` is used directly.
    image_template: Optional[str] = None
    answer_path: str = DEFAULT_ANSWER_PATH
    step_timeout: int = 900
    step_limit: int = 250
    skip_if_exists: bool = False


class MiniSWEAgentQnaRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: Optional[dict[str, Any]] = None
    instance_id: Optional[str] = None


class MiniSWEAgentQnaVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class MiniSWEAgentQnaVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={"py_executable": sys.executable},
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


def _responses_create_params_to_model_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    """Convert Gym Responses API rollout params into mini-swe-agent chat kwargs."""
    model_kwargs: dict[str, Any] = {}
    for key in ("temperature", "top_p", "top_logprobs"):
        value = params.get(key)
        if value is not None:
            model_kwargs[key] = value
    max_output_tokens = params.get("max_output_tokens")
    if max_output_tokens is not None:
        model_kwargs["max_tokens"] = max_output_tokens

    metadata = params.get("metadata") or {}
    extra_body = _json_dict_from_metadata(metadata.get("extra_body"), field_name="extra_body")
    chat_template_kwargs = _json_dict_from_metadata(
        metadata.get("chat_template_kwargs"), field_name="chat_template_kwargs"
    )
    if chat_template_kwargs:
        extra_body["chat_template_kwargs"] = chat_template_kwargs
    if extra_body:
        model_kwargs["extra_body"] = extra_body
    return model_kwargs


def _resolve_image(image_template: Optional[str], metadata: dict[str, Any]) -> str:
    """Resolve the sandbox image from the template + row metadata."""
    if image_template:
        return image_template.format(**metadata)
    image = metadata.get("docker_image")
    if not image:
        raise ValueError(
            "No sandbox image: set image_template or provide docker_image in the row's verifier_metadata."
        )
    return str(image)


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
        "id": f"resp_{uuid4()}",
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
        "reasoning": {"effort": None, "generate_summary": None, "summary": None},
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


def _answer_message_item(answer: str) -> dict[str, Any]:
    """A single assistant message carrying the final answer (what the judge grades)."""
    return {
        "id": f"msg_{uuid4()}",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": answer, "annotations": []}],
    }


def _run_mini_swe_qna(**params: Any) -> dict[str, Any]:
    """Blocking mini-swe-agent QnA run inside a sandbox (executed in a Ray task)."""
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments import get_environment
    from minisweagent.models import get_model

    instance = params["instance_dict"]
    if isinstance(instance, str):
        instance = json.loads(instance)
    instance_id = str(params["instance_id"])
    problem_statement = params["problem_statement"]
    answer_path = params["answer_path"]

    output_dir = Path(params["output"])
    instance_dir = output_dir / instance_id
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
    model_kwargs.update(params.get("model_kwargs") or {})

    environment_config = config.setdefault("environment", {})
    environment_config.update(params.get("sandbox_environment_kwargs") or {})
    environment_config["image"] = params["image"]
    environment_config["step_timeout"] = params["step_timeout"]
    environment_config["instance_id"] = instance_id
    environment_config["provider"] = params["provider"]
    environment_config["spec"] = params["spec"]
    environment_config["environment_class"] = (
        "responses_api_agents.mini_swe_agent_qna.sandbox_environment.MiniSWESandboxEnvironment"
    )

    agent_config = config.get("agent", {})
    agent_config["step_limit"] = params["step_limit"]
    agent_config.pop("collapse_limit", None)

    run_id = f"{int(time.time())}_{uuid4()}"
    trajectory_path = instance_dir / f"{instance_id}_{run_id}.traj.json"
    agent_config["output_path"] = trajectory_path

    env = None
    try:
        env = get_environment(environment_config)
        model = get_model(config=model_config)
        agent = DefaultAgent(model, env, **agent_config)

        info = agent.run(problem_statement)
        exit_status = info.get("exit_status", "")

        # The answer is the artifact the agent wrote, not the submit payload.
        answer_result = env.execute(f"cat {answer_path}", is_eval=False)
        answer = answer_result["output"] if answer_result.get("returncode") == 0 else ""

        data = agent.save(trajectory_path, {"instance_id": instance_id})
        input_messages, response_output, responses = _split_trajectory_for_responses(data.get("messages", []))

        return {
            instance_id: {
                "answer": answer,
                "input_messages": input_messages,
                "response_output": response_output,
                "responses": responses,
                "exit_status": exit_status,
            }
        }
    finally:
        if env and hasattr(env, "cleanup"):
            env.cleanup()


def run_mini_swe_qna_with_sandbox(**params: Any) -> Any:
    return _run_mini_swe_qna(**params)


class MiniSWEAgentQna(SimpleResponsesAPIAgent):
    config: MiniSWEAgentQnaConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_session_middleware(app)
        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        return app

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError("mini_swe_agent_qna drives the model via /run")

    async def run(self, body: MiniSWEAgentQnaRunRequest) -> MiniSWEAgentQnaVerifyResponse:
        async with self.sem:
            global_config_dict = ServerClient.load_from_global_config().global_config_dict
            model_server_config = get_first_server_config_dict(global_config_dict, self.config.model_server.name)
            policy_model_name = global_config_dict["policy_model_name"]

            metadata = body.verifier_metadata or {}
            instance_id = body.instance_id or metadata.get("instance_id") or str(uuid4())
            problem_statement = metadata.get("problem_statement") or ""

            base_url = f"http://{model_server_config['host']}:{model_server_config['port']}/v1"
            output_file_dir = f"{Path.cwd()}/results/swe_atlas_qna/{policy_model_name}"

            resolved_provider = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
            provider_default_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)
            spec = dict(self.config.sandbox_spec or {})
            if provider_default_metadata:
                spec["metadata"] = {**provider_default_metadata, **(spec.get("metadata") or {})}

            responses_params = body.responses_create_params.model_dump(exclude_none=True)

            answer = ""
            input_messages: list[dict[str, Any]] = []
            response_output: list[dict[str, Any]] = []
            run_error: Optional[str] = None
            try:
                params = dict(
                    instance_dict=body.model_dump(),
                    instance_id=instance_id,
                    problem_statement=problem_statement,
                    output=output_file_dir,
                    model=f"hosted_vllm/{policy_model_name}",
                    api_key="dummy_key",
                    base_url=base_url,
                    config=str(Path(self.config.mini_swe_config_path).resolve()),
                    image=_resolve_image(self.config.image_template, metadata),
                    provider=resolved_provider,
                    spec=spec,
                    sandbox_environment_kwargs=self.config.sandbox_environment_kwargs,
                    model_kwargs=_responses_create_params_to_model_kwargs(responses_params),
                    step_timeout=self.config.step_timeout,
                    step_limit=self.config.step_limit,
                    answer_path=self.config.answer_path,
                )
                future = runner_ray_remote.remote(run_mini_swe_qna_with_sandbox, params)
                result = (await asyncio.to_thread(ray.get, future))[instance_id]
                answer = result["answer"]
                input_messages = result["input_messages"]
                response_output = result["response_output"]
            except Exception as e:
                run_error = f"{e}\n{traceback.format_exc()}"
                print(f"Error running mini-swe-agent QnA: {run_error}", flush=True)

            # Grade the extracted answer via the rubric-judge resources server.
            reward, judge_extra = await self._verify_answer(body, answer)

            # Return the full trajectory for inspection, with the delegated reward.
            body.responses_create_params.input = input_messages
            full_response = _default_response_object()
            full_response["model"] = policy_model_name
            full_response["output"] = response_output or [_answer_message_item(answer)]

            return MiniSWEAgentQnaVerifyResponse(
                responses_create_params=body.responses_create_params,
                response=full_response,
                reward=reward,
                instance_id=instance_id,
                answer=answer,
                run_error=run_error,
                **judge_extra,
            )

    async def _verify_answer(self, body: MiniSWEAgentQnaRunRequest, answer: str) -> tuple[float, dict[str, Any]]:
        """POST the extracted answer to the resources server /verify; return (reward, extra)."""
        answer_response = _default_response_object()
        answer_response["output"] = [_answer_message_item(answer)]

        verify_request = body.model_dump() | {"response": answer_response}
        try:
            verify_obj = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request,
            )
            await raise_for_status(verify_obj)
            verify_result = await get_response_json(verify_obj)
        except Exception:
            print("Verify call failed; scoring 0.", flush=True)
            return 0.0, {}

        reward = float(verify_result.get("reward", 0.0) or 0.0)
        extra = {
            key: verify_result[key]
            for key in ("agg_score", "passed", "rubric_scores", "num_rubrics", "num_scored")
            if key in verify_result
        }
        return reward, extra

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _score_fn(r: dict[str, Any]) -> dict[str, float]:
        return {
            "pass": 1.0 if float(r.get("reward", 0.0) or 0.0) >= 1.0 else 0.0,
            "agg_score": float(r.get("agg_score", 0.0) or 0.0),
        }

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        metrics = compute_pass_majority_metrics(tasks, score_fn=self._score_fn)[0]
        all_rollouts = [rollout for task in tasks for rollout in task]
        metrics["task_count"] = len(tasks)
        metrics["rollout_count"] = len(all_rollouts)
        metrics["run_error_count"] = sum(1 for r in all_rollouts if r.get("run_error"))
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        key: dict[str, Any] = {}
        for name in ("mean/reward", "task_count", "run_error_count"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        return key


if __name__ == "__main__":
    MiniSWEAgentQna.run_webserver()
