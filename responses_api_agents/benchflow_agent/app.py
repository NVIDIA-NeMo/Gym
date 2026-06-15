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
"""NeMo Gym agent that runs a single BenchFlow task per `/run` request.

BenchFlow does not implement its own agent — it installs an existing harness
(e.g. OpenHands) into a sandbox and lets it drive the model. This server wraps
BenchFlow's Python ``Evaluation`` API: each NeMo Gym ``/run`` call runs exactly
one task (``include_tasks={task}``, ``concurrency=1``) in a Singularity sandbox,
extracts the scalar reward, and returns a NeMo Gym response. Eval-only.

``benchflow`` is imported lazily inside ``_run_benchflow_task()`` so this module
stays importable (e.g. for unit tests that mock it) without the package installed.
"""

import re
from asyncio import Semaphore
from copy import deepcopy
from typing import Any, Optional
from uuid import uuid4

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
    # Max concurrent /run requests handled by this agent server process.
    concurrency: int

    # NeMo Gym model server reference used to resolve the OpenAI-compatible base URL
    # that the in-container agent harness (e.g. OpenHands) calls back to.
    model_server: ModelServerRef

    # Local directory of BenchFlow task definitions (e.g. a SkillsBench `tasks/` dir).
    tasks_dir: str

    # Base directory where BenchFlow writes per-run job artifacts/logs.
    jobs_dir: str = "jobs"

    # BenchFlow agent harness to install and run in the sandbox.
    agent: str = "openhands"

    # Extra environment variables forwarded to the agent harness. The provider base
    # URL and API key are injected automatically at request time (see run()).
    agent_env: dict[str, str] = Field(default_factory=dict)

    # Sandbox user. None (or "none") runs as root; matches `--sandbox-user none`.
    sandbox_user: Optional[str] = None

    # Abort a rollout if the agent makes no tool call for this many seconds.
    agent_idle_timeout: Optional[int] = 1200

    # Skills directory passed to BenchFlow ("auto" = use task-bundled skills).
    skills_dir: Optional[str] = "auto"

    # BENCHFLOW_SKILL_NUDGE value (None disables it). "name" matches the current eval.
    skill_nudge: Optional[str] = "name"

    # BenchFlow usage/cost telemetry mode: "auto" | "required" | "off".
    usage_tracking: str = "off"

    # API key forwarded to the agent for the model endpoint. A dummy value is fine for
    # a local vLLM server that ignores auth; OpenHands requires it to be non-empty.
    provider_api_key: str = "EMPTY"

    # BenchFlow per-task failure retries (RetryConfig.max_retries). Distinct from NeMo
    # Gym `num_repeats` (which samples for mean@k). Default matches BenchFlow's default.
    max_retries: int = 2

    # Overrides deep-merged into each task's parsed task.toml, mirroring task.toml
    # sections, e.g. {"environment": {"memory_mb": 131072}, "agent": {"timeout_sec": ...}}.
    # Requires a BenchFlow build with task_config_overrides support.
    task_config_overrides: dict[str, Any] = Field(default_factory=dict)

    # Directory of prebuilt Singularity .sif images. When set, each task's docker_image
    # is overridden to "<images_dir>/<task>.sif" (per-task — each /run handles one task).
    # Takes precedence over any docker_image in task_config_overrides.
    images_dir: Optional[str] = None


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
            global_config_dict = get_global_config_dict()
            policy_model_name = global_config_dict["policy_model_name"]
            base_url = self._resolve_model_base_url(global_config_dict)

            task_name = self._parse_task_name(body.instance_id)
            params = body.responses_create_params.model_dump(exclude_unset=True, exclude_none=True)

            result = None
            error_message = None
            try:
                result = await self._run_benchflow_task(task_name, policy_model_name, base_url)
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
            response["model"] = policy_model_name
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

    async def _run_benchflow_task(self, task_name: str, policy_model_name: str, base_url: str) -> Any:
        """Run a single BenchFlow task in a Singularity sandbox; return its RolloutResult.

        All ``benchflow`` imports and usage are confined to this method so the module
        imports cleanly without benchflow installed and unit tests can patch it.
        Returns the task's RolloutResult, or None if BenchFlow reported no result.
        """
        from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig

        eval_config = EvaluationConfig(
            agent=self.config.agent,
            model=f"hosted_vllm/{policy_model_name}",
            environment="singularity",
            concurrency=1,
            agent_env=self._build_agent_env(base_url),
            sandbox_user=self.config.sandbox_user,
            agent_idle_timeout=self.config.agent_idle_timeout,
            skills_dir=self.config.skills_dir,
            include_tasks={task_name},
            usage_tracking=self.config.usage_tracking,
            retry=RetryConfig(max_retries=self.config.max_retries),
            task_config_overrides=self._build_task_config_overrides(task_name) or None,
        )

        # Each /run runs exactly one task; capture its single RolloutResult via on_result.
        job_name = f"{self._sanitize_path_component(task_name)}_{uuid4().hex[:8]}"
        captured: dict[str, Any] = {}

        def _on_result(name: str, result: Any) -> None:
            captured["result"] = result

        evaluation = Evaluation(
            tasks_dir=self.config.tasks_dir,
            jobs_dir=self.config.jobs_dir,
            config=eval_config,
            job_name=job_name,
            on_result=_on_result,
        )
        await evaluation.run()
        return captured.get("result")

    def _build_agent_env(self, base_url: str) -> dict[str, str]:
        """Forward configured agent env plus the resolved provider endpoint + key."""
        agent_env = dict(self.config.agent_env or {})
        agent_env["BENCHFLOW_PROVIDER_BASE_URL"] = base_url
        agent_env["BENCHFLOW_PROVIDER_API_KEY"] = self.config.provider_api_key
        if self.config.skill_nudge:
            agent_env["BENCHFLOW_SKILL_NUDGE"] = self.config.skill_nudge
        return agent_env

    def _build_task_config_overrides(self, task_name: str) -> dict[str, Any]:
        """Merge the configured task.toml overrides with the per-task .sif image path.

        Each /run handles a single task, so we can pin a concrete per-task image.
        ``images_dir`` takes precedence over any docker_image in task_config_overrides.
        """
        overrides = deepcopy(self.config.task_config_overrides or {})
        if self.config.images_dir:
            sif_path = f"{self.config.images_dir.rstrip('/')}/{task_name}.sif"
            environment = overrides.get("environment")
            if not isinstance(environment, dict):
                environment = {}
                overrides["environment"] = environment
            environment["docker_image"] = sif_path
        return overrides

    @staticmethod
    def _parse_task_name(instance_id: str) -> str:
        """Extract the task name from an instance id of form '<alias>::<task>' or '<task>'."""
        head, sep, tail = instance_id.partition("::")
        task_name = (tail if sep else head).strip()
        if not task_name:
            raise ValueError(f"instance_id must contain a task name (got: {instance_id!r})")
        return task_name

    @staticmethod
    def _sanitize_path_component(value: str) -> str:
        """Sanitize a string for safe use as a single path component (job name)."""
        sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", value).strip("._")
        return sanitized or "task"

    def _resolve_model_base_url(self, global_config_dict: Any) -> str:
        """Resolve the model server base URL from the required model_server reference."""
        server_name = self.config.model_server.name
        model_server_config = get_first_server_config_dict(
            global_config_dict,
            server_name,
        )
        return f"http://{model_server_config['host']}:{model_server_config['port']}/v1"


if __name__ == "__main__":
    BenchFlowAgent.run_webserver()
