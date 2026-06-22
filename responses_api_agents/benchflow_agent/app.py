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
"""
NeMo Gym environment for BenchFlow: https://github.com/benchflow-ai/benchflow.

Each `/run` call runs exactly one task using BenchFlow's Python API,
extracts the scalar reward, and returns a NeMo Gym response.
"""

import re
import shutil
import tempfile
from asyncio import Semaphore
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import Body
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import (
    get_first_server_config_dict,
    get_global_config_dict,
)
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from responses_api_agents.benchflow_agent.utils import BenchFlowAgentUtils


class BenchFlowAgentConfig(BaseResponsesAPIAgentConfig):
    # NeMo Gym model server reference.
    model_server: ModelServerRef

    # Local directory of BenchFlow task definitions (e.g. a SkillsBench `tasks/` dir).
    tasks_dir: str

    # Output directory for BenchFlow results/artifacts/logs.
    jobs_dir: str = "jobs"

    # BenchFlow agent harness.
    agent: str = "openhands"

    # Extra environment variables forwarded to the agent harness.
    # The LLM server base URL and API key are passed automatically.
    agent_env: Optional[dict[str, str]] = None

    # BenchFlow sandbox/environment type.
    environment: str = "singularity"

    # Sandbox user. "none" or null runs as root.
    sandbox_user: Optional[str] = None

    # Abort a rollout if the agent makes no tool call for this many seconds.
    agent_idle_timeout: Optional[int] = 1200

    # Skill mode: "with-skill" | "no-skill" | "self-gen".
    skill_mode: str = "with-skill"

    # Skills directory ("auto" = use task-bundled skills, None = do not use skills).
    skills_dir: Optional[str] = "auto"

    # How many times BenchFlow should retry a task when an error occurs.
    max_retries: int = 2

    # Overrides to apply to every task's task.md.
    task_config_overrides: Optional[dict[str, Any]] = None

    # Directory of prebuilt per-task Singularity .sif images. Files inside must be named "<task_name>.sif".
    singularity_images_dir: Optional[str] = None

    # Max number of concurrent `/run` requests handled by the NeMo Gym agent server process.
    concurrency: int = 256


class BenchFlowRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    instance_id: str


class BenchFlowVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class BenchFlowAgent(SimpleResponsesAPIAgent):
    config: BenchFlowAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError

    async def run(self, body: BenchFlowRunRequest) -> BenchFlowVerifyResponse:
        async with self.sem:
            task_name = self._parse_task_name(body.instance_id)
            params = body.responses_create_params.model_dump(exclude_unset=True, exclude_none=True)

            result = None
            error_message = None
            try:
                result = await self._run_benchflow_task(task_name)
            except Exception as e:
                error_message = f"{type(e).__name__}: {e}"
                print(f"Error running BenchFlow evaluation for task {task_name!r}: {error_message}")

            if result is not None:
                reward = BenchFlowAgentUtils.extract_reward(getattr(result, "rewards", None))
                output_items = BenchFlowAgentUtils.trajectory_to_output(getattr(result, "trajectory", None))
                usage = BenchFlowAgentUtils.extract_usage(result)
            else:
                reward = 0.0
                output_items = []
                usage = None

            response = BenchFlowAgentUtils.get_default_response_object()
            response["model"] = get_global_config_dict()["policy_model_name"]
            response["temperature"] = params.get("temperature")
            response["top_p"] = params.get("top_p")
            response["output"] = output_items
            if usage:
                response["usage"] = usage

            return BenchFlowVerifyResponse(
                responses_create_params=body.responses_create_params,
                reward=reward,
                response=response,
                instance_id=body.instance_id,
                metadata={
                    "task_name": task_name,
                    "jobs_dir": self.config.jobs_dir,
                    "rewards": getattr(result, "rewards", None) if result is not None else None,
                    "error": getattr(result, "error", None) if result is not None else error_message,
                    "error_category": getattr(result, "error_category", None) if result is not None else None,
                    "verifier_error": getattr(result, "verifier_error", None) if result is not None else None,
                    "n_tool_calls": getattr(result, "n_tool_calls", None) if result is not None else None,
                },
            )

    async def _run_benchflow_task(self, task_name: str) -> Any:
        """
        Runs a single BenchFlow task. Returns its RolloutResult or None if unavailable.
        If any config overrides are passed, they are applied on a temporary copy of the task folder.
        """
        overrides = self._build_task_config_overrides(task_name)
        if not overrides:
            return await self._run_evaluation(task_name, self.config.tasks_dir)

        with tempfile.TemporaryDirectory(prefix="benchflow-task-") as tmp_dir:
            src = Path(self.config.tasks_dir) / task_name
            dst = Path(tmp_dir) / task_name
            shutil.copytree(src, dst)
            BenchFlowAgentUtils.apply_task_config_overrides(dst, overrides)
            return await self._run_evaluation(task_name, tmp_dir)

    async def _run_evaluation(self, task_name: str, tasks_dir: str) -> Any:
        """Builds and runs a single-task BenchFlow Evaluation against `tasks_dir`."""
        from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig

        eval_config = EvaluationConfig(
            agent=self.config.agent,
            model=f"hosted_vllm/{get_global_config_dict()['policy_model_name']}",
            environment=self.config.environment,
            concurrency=1,
            agent_env=self._build_agent_env(),
            sandbox_user=self.config.sandbox_user,
            agent_idle_timeout=self.config.agent_idle_timeout,
            skill_mode=self.config.skill_mode,
            skills_dir=self.config.skills_dir,
            include_tasks={task_name},
            retry=RetryConfig(max_retries=self.config.max_retries),
        )

        # Each /run runs exactly one task; capture its single RolloutResult via on_result.
        job_name = f"{self._sanitize_path_component(task_name)}_{uuid4().hex[:8]}"
        captured: dict[str, Any] = {}

        def _on_result(name: str, result: Any) -> None:
            captured["result"] = result

        evaluation = Evaluation(
            tasks_dir=str(tasks_dir),
            jobs_dir=self.config.jobs_dir,
            config=eval_config,
            job_name=job_name,
            on_result=_on_result,
        )
        await evaluation.run()
        return captured.get("result")

    def _build_agent_env(self) -> dict[str, str]:
        """Builds the environment variables to forward to the agent harness."""
        agent_env = dict(self.config.agent_env or {})
        global_config_dict = get_global_config_dict()
        agent_env["BENCHFLOW_PROVIDER_BASE_URL"] = self._resolve_model_base_url(global_config_dict)
        agent_env["BENCHFLOW_PROVIDER_API_KEY"] = global_config_dict.get("policy_api_key", "EMPTY")
        return agent_env

    def _build_task_config_overrides(self, task_name: str) -> dict[str, Any]:
        """
        Builds the task config overrides dict,
        adding the task-specific .sif image path if `singularity_images_dir` is set.
        """
        overrides = deepcopy(self.config.task_config_overrides or {})
        if self.config.singularity_images_dir:
            sif_path = f"{self.config.singularity_images_dir.rstrip('/')}/{task_name}.sif"
            environment = overrides.get("environment")
            if not isinstance(environment, dict):
                environment = {}
                overrides["environment"] = environment
            environment["docker_image"] = sif_path
        return overrides

    @staticmethod
    def _parse_task_name(instance_id: str) -> str:
        """Extracts the task name from an instance id of form '<alias>::<task>' or '<task>'."""
        head, sep, tail = instance_id.partition("::")
        task_name = (tail if sep else head).strip()
        if not task_name:
            raise ValueError(f"instance_id must contain a task name (got: {instance_id!r})")
        return task_name

    @staticmethod
    def _sanitize_path_component(value: str) -> str:
        """Sanitizes a string for safe use as a single path component (job name)."""
        sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", value).strip("._")
        return sanitized or "task"

    def _resolve_model_base_url(self, global_config_dict: Any) -> str:
        """Resolves the model server base URL from the required model_server reference."""
        server_name = self.config.model_server.name
        model_server_config = get_first_server_config_dict(
            global_config_dict,
            server_name,
        )
        return f"http://{model_server_config['host']}:{model_server_config['port']}/v1"


if __name__ == "__main__":
    BenchFlowAgent.run_webserver()
