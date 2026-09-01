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
import json
import logging
import sys
import time
from pathlib import Path
from uuid import uuid4

import ray
from harbor.job import Job
from harbor.models.job.config import DatasetConfig, JobConfig, RetryConfig
from harbor.models.task.paths import TaskPaths
from harbor.models.trajectories import ContentPart, Step, Trajectory
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, VerifierConfig
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import TrialResult
from pydantic import ConfigDict, Field, PrivateAttr, ValidationError, field_validator

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME, get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY


logger = logging.getLogger(__name__)

NUM_SAMPLES_IN_PARALLEL_KEY_NAME = "num_samples_in_parallel"

_RAY_WORKER_EVENT_LOOP: asyncio.AbstractEventLoop | None = None


@ray.remote(scheduling_strategy="SPREAD", runtime_env={"py_executable": sys.executable})
def harbor_job_worker(job_config_dict: dict, task_name: str) -> str:
    global _RAY_WORKER_EVENT_LOOP
    logging.disable(logging.DEBUG)
    if _RAY_WORKER_EVENT_LOOP is None or _RAY_WORKER_EVENT_LOOP.is_closed():
        _RAY_WORKER_EVENT_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_RAY_WORKER_EVENT_LOOP)
    return _RAY_WORKER_EVENT_LOOP.run_until_complete(HarborAgent.run_job(job_config_dict, task_name))


class HarborAgentConfig(BaseResponsesAPIAgentConfig):
    harbor_ray_task_num_cpus: float = Field(default=0.25, ge=0)
    harbor_jobs_dir: Path
    harbor_debug: bool = Field(default=False)
    harbor_max_retries: int = Field(default=0)
    harbor_dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    harbor_environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    harbor_agent: AgentConfig = Field(default_factory=AgentConfig)
    harbor_verifier: VerifierConfig = Field(default_factory=VerifierConfig)

    @field_validator("harbor_jobs_dir", mode="after")
    @classmethod
    def normalize_jobs_dir(cls, harbor_jobs_dir: Path) -> Path:
        harbor_jobs_dir = harbor_jobs_dir.resolve()
        if harbor_jobs_dir.suffix.lower() == ".jsonl":
            return harbor_jobs_dir.parent / "harbor"
        return harbor_jobs_dir

    def build_job_config(self, task_name: str, job_name: str) -> JobConfig:
        return JobConfig(
            job_name=job_name,
            jobs_dir=self.harbor_jobs_dir,
            n_attempts=1,
            n_concurrent_trials=1,
            debug=self.harbor_debug,
            quiet=True,
            retry=RetryConfig(max_retries=self.harbor_max_retries),
            datasets=[
                self.harbor_dataset.model_copy(update={"task_names": [task_name]}),
            ],
            environment=self.harbor_environment.model_copy(update={"delete": True}),
            agents=[self.harbor_agent],
            verifier=self.harbor_verifier,
        )


class HarborRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_name: str
    task_index: int = Field(alias=TASK_INDEX_KEY_NAME)
    rollout_index: int = Field(alias=ROLLOUT_INDEX_KEY_NAME)


class HarborVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class HarborAgent(SimpleResponsesAPIAgent):
    config: HarborAgentConfig

    _sem: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context) -> None:
        self._sem = asyncio.Semaphore(get_global_config_dict().get(NUM_SAMPLES_IN_PARALLEL_KEY_NAME) or 1)

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        ## Harbor owns the full run() lifecycle.
        raise NotImplementedError

    async def run(self, body: HarborRunRequest) -> HarborVerifyResponse:
        async with self._sem:
            try:
                job_config = self.config.build_job_config(
                    task_name=body.task_name,
                    ## Use a stable job name to allow resume.
                    job_name=f"t{body.task_index}-r{body.rollout_index}",
                )

                job_ref = harbor_job_worker.options(num_cpus=self.config.harbor_ray_task_num_cpus).remote(
                    job_config.model_dump(mode="json"), body.task_name
                )
                try:
                    trial_dir = Path(await job_ref)
                except asyncio.CancelledError:
                    ray.cancel(job_ref, force=True)
                    raise

                return self.success_response(body, trial_dir)
            except Exception as err:
                logger.exception(
                    "Harbor rollout failed: task_index=%s rollout_index=%s",
                    body.task_index,
                    body.rollout_index,
                )
                return self.failure_response(body, err)

    @staticmethod
    def convert_atif_to_gym_responses(
        trajectory: Trajectory, conversion_warnings: list[str] | None = None
    ) -> list[dict]:
        output_items = []
        warnings = conversion_warnings if conversion_warnings is not None else []

        def warn(message: str) -> None:
            warnings.append(message)
            logger.warning("ATIF conversion: %s", message)

        def convert_input_content(parts: list[ContentPart], context: str) -> str | list[dict]:
            serialized_content = [part.model_dump(mode="json", exclude_none=True) for part in parts]
            local_image_paths = [
                part.source.path
                for part in parts
                if part.type == "image"
                and part.source is not None
                and not part.source.path.startswith(("http://", "https://", "data:"))
            ]
            if local_image_paths:
                warn(
                    f"{context}: content serialized as JSON because local image paths are not portable Gym image "
                    f"URLs: {local_image_paths}"
                )
                return json.dumps(serialized_content)

            return [
                {"type": "input_text", "text": part.text}
                if part.type == "text"
                else {"type": "input_image", "image_url": part.source.path, "detail": "auto"}
                for part in parts
                if part.type == "text" or part.source is not None
            ]

        def append_observations(step: Step) -> None:
            observation = step.observation
            for result_index, result in enumerate(observation.results if observation is not None else []):
                context = f"step {step.step_id} observation {result_index}"
                if isinstance(result.content, str):
                    tool_output = result.content
                elif result.content is None:
                    tool_output = ""
                    warn(f"{context}: absent content represented as empty text")
                else:
                    tool_output = convert_input_content(result.content, context)

                call_id = result.source_call_id
                if call_id is None:
                    call_id = f"atif-step-{step.step_id}-observation-{result_index}"
                    warn(f"{context}: missing source_call_id represented with synthetic call_id {call_id}")
                if result.subagent_trajectory_ref:
                    warn(f"{context}: subagent trajectory references are preserved only in the source ATIF trajectory")
                if result.extra:
                    warn(f"{context}: extra metadata is preserved only in the source ATIF trajectory")
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        call_id=call_id,
                        output=tool_output,
                        type="function_call_output",
                        id=f"fco_{uuid4().hex[:8]}",
                        status="completed",
                    ).model_dump()
                )

        if trajectory.continued_trajectory_ref is not None:
            warn("continued_trajectory_ref is preserved only in the source ATIF trajectory")
        if trajectory.subagent_trajectories:
            warn("embedded subagent trajectories are preserved only in the source ATIF trajectory")
        if trajectory.notes is not None or trajectory.extra:
            warn("trajectory notes or extra metadata are preserved only in the source ATIF trajectory")
        if trajectory.final_metrics is not None:
            warn("ATIF final_metrics are preserved only in the source trajectory; Gym usage comes from Harbor")
        if trajectory.agent.tool_definitions:
            warn("ATIF tool definitions are preserved only in the source trajectory; Gym response tools remain empty")
        if trajectory.agent.extra:
            warn("ATIF agent extra metadata is preserved only in the source trajectory")

        for step in trajectory.steps:
            if step.source != "agent":
                message_content = (
                    step.message
                    if isinstance(step.message, str)
                    else convert_input_content(step.message, f"step {step.step_id} {step.source} message")
                )
                output_items.append(
                    NeMoGymEasyInputMessage(
                        role=step.source,
                        content=message_content,
                        type="message",
                    ).model_dump()
                )
                append_observations(step)
                continue
            if (
                step.timestamp is not None
                or step.model_name is not None
                or step.reasoning_effort is not None
                or step.extra
            ):
                warn(
                    f"step {step.step_id}: timestamp, model, reasoning effort, or extra metadata is preserved only "
                    "in the source ATIF trajectory"
                )

            if step.reasoning_content:
                warn(f"step {step.step_id}: reasoning_content represented as a Gym reasoning summary")
                output_items.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rs_{uuid4().hex[:12]}",
                        summary=[NeMoGymSummary(text=step.reasoning_content, type="summary_text")],
                        status="completed",
                    ).model_dump()
                )

            if isinstance(step.message, str):
                message_text = step.message
            else:
                message_text = json.dumps([part.model_dump(mode="json", exclude_none=True) for part in step.message])
                warn(
                    f"step {step.step_id}: multimodal message serialized as JSON because Gym assistant output "
                    "messages support only text or refusal content"
                )

            content = [
                NeMoGymResponseOutputText(
                    annotations=[],
                    text=message_text,
                    type="output_text",
                    logprobs=None,
                )
            ]
            metrics = step.metrics
            metrics_extra = metrics.extra if metrics is not None and metrics.extra is not None else {}
            routed_experts = metrics_extra.get("routed_experts")
            if metrics is not None and (
                metrics.prompt_tokens is not None
                or metrics.completion_tokens is not None
                or metrics.cached_tokens is not None
                or metrics.cost_usd is not None
                or set(metrics_extra) - {"routed_experts"}
            ):
                warn(
                    f"step {step.step_id}: scalar metrics, cost, or unrecognized metric extras are preserved only "
                    "in the source ATIF trajectory"
                )
            prompt_token_ids = metrics.prompt_token_ids if metrics is not None else None
            completion_token_ids = metrics.completion_token_ids if metrics is not None else None
            logprobs = metrics.logprobs if metrics is not None else None
            token_metadata_present = any(
                value is not None for value in (prompt_token_ids, completion_token_ids, logprobs, routed_experts)
            )
            token_metadata_issues = []
            if not completion_token_ids:
                token_metadata_issues.append("completion_token_ids are missing or empty")
            if prompt_token_ids is None:
                token_metadata_issues.append("prompt_token_ids are missing")
            if logprobs is None:
                token_metadata_issues.append("logprobs are missing")
            elif completion_token_ids is None or len(logprobs) != len(completion_token_ids):
                token_metadata_issues.append("completion_token_ids and logprobs have different lengths")
            if step.is_copied_context:
                token_metadata_issues.append("step is copied context")
            if step.llm_call_count not in (None, 1):
                token_metadata_issues.append(f"llm_call_count is {step.llm_call_count}")
            if step.reasoning_content or step.tool_calls:
                token_metadata_issues.append("tokens cannot be attributed across reasoning or tool-call items")

            message = None
            if token_metadata_present and not token_metadata_issues:
                try:
                    message = NeMoGymResponseOutputMessageForTraining(
                        id=f"msg_{uuid4().hex[:12]}",
                        content=content,
                        role="assistant",
                        status="completed",
                        prompt_token_ids=prompt_token_ids,
                        generation_token_ids=completion_token_ids,
                        generation_log_probs=logprobs,
                        routed_experts=routed_experts,
                    )
                except ValidationError as err:
                    token_metadata_issues.append(f"Gym rejected token metadata: {err.errors(include_url=False)}")

            if message is None:
                message = NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex[:12]}",
                    content=content,
                    role="assistant",
                    status="completed",
                )
                if token_metadata_present:
                    warn(f"step {step.step_id}: training metadata omitted: {'; '.join(token_metadata_issues)}")
            output_items.append(message.model_dump())

            tool_calls = step.tool_calls or []
            if len(tool_calls) > 1:
                warn(
                    f"step {step.step_id}: ATIF does not record whether multiple tool calls were parallel; Gym "
                    "parallel_tool_calls remains false"
                )
            for tool_call in tool_calls:
                if tool_call.extra:
                    warn(
                        f"step {step.step_id} tool call {tool_call.tool_call_id}: extra metadata is preserved only "
                        "in the source ATIF trajectory"
                    )
                output_items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=json.dumps(tool_call.arguments),
                        call_id=tool_call.tool_call_id,
                        name=tool_call.function_name,
                        type="function_call",
                        id=f"fc_{uuid4().hex[:8]}",
                        status="completed",
                    ).model_dump()
                )

            append_observations(step)

        return output_items

    def success_response(self, body: HarborRunRequest, trial_dir: Path) -> HarborVerifyResponse:
        trial_paths = TrialPaths(trial_dir)
        trial = TrialResult.model_validate_json(trial_paths.result_path.read_text())
        task_paths = TaskPaths(trial.config.task.get_local_path())
        if trial.step_results:
            path_entries = [
                (
                    task_paths.step_instruction_path(step.step_name),
                    trial_paths.step_agent_dir(step.step_name) / TaskPaths.TRAJECTORY_FILENAME,
                )
                for step in trial.step_results
            ]
            step_rewards = [
                step.verifier_result.rewards if step.verifier_result is not None else None
                for step in trial.step_results
            ]
        else:
            path_entries = [(task_paths.instruction_path, trial_paths.agent_dir / TaskPaths.TRAJECTORY_FILENAME)]
            step_rewards = [trial.verifier_result.rewards if trial.verifier_result is not None else None]

        trajectories = [
            Trajectory.model_validate_json(trajectory_path.read_text()) for _, trajectory_path in path_entries
        ]
        conversion_warnings: list[str] = []
        output = [self.convert_atif_to_gym_responses(trajectory, conversion_warnings) for trajectory in trajectories]

        if len(output) > 1:
            logger.warning(
                "Multiple Harbor task steps found (%s); using the first step for Gym Input/Output and last step for reward",
                len(path_entries),
            )
        output = output[0]
        step_rewards = step_rewards[-1]
        instruction_path, _ = path_entries[0]

        n_input_tokens, n_cache_tokens, n_output_tokens, _ = trial.compute_token_cost_totals()

        if step_rewards is not None and "reward" not in step_rewards:
            logger.warning(
                "Harbor verifier result for trial %s has no 'reward' key; using the first available reward or 0.0",
                trial.id,
            )

        return HarborVerifyResponse.model_validate(
            body.model_dump(by_alias=True)
            | {
                "responses_create_params": body.responses_create_params.model_copy(
                    update={
                        "input": [
                            NeMoGymEasyInputMessage(
                                role="user",
                                content=instruction_path.read_text(),
                                type="message",
                            )
                        ]
                    }
                ),
                "response": NeMoGymResponse(
                    id=f"harbor-{trial.id}",
                    created_at=(trial.finished_at.timestamp() if trial.finished_at is not None else time.time()),
                    model=self.config.harbor_agent.model_name,
                    object="response",
                    output=output,
                    parallel_tool_calls=False,
                    temperature=body.responses_create_params.temperature,
                    tool_choice="auto",
                    tools=[],
                    top_p=body.responses_create_params.top_p,
                    status="completed",
                    usage={
                        "input_tokens": n_input_tokens or 0,
                        "input_tokens_details": {
                            "cached_tokens": n_cache_tokens or 0,
                            "cache_write_tokens": 0,
                        },
                        "output_tokens": n_output_tokens or 0,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": (n_input_tokens or 0) + (n_output_tokens or 0),
                    },
                ),
                "reward": float(
                    step_rewards.get("reward", next(iter(step_rewards.values()), 0.0)) if step_rewards else 0.0
                ),
                "atif_conversion": {
                    "lossless": not conversion_warnings,
                    "warnings": conversion_warnings,
                    "source_trajectory_paths": [str(trajectory_path) for _, trajectory_path in path_entries],
                    "trajectories": [
                        {
                            "schema_version": trajectory.schema_version,
                            "session_id": trajectory.session_id,
                            "trajectory_id": trajectory.trajectory_id,
                            "agent_name": trajectory.agent.name,
                            "agent_version": trajectory.agent.version,
                            "agent_model_name": trajectory.agent.model_name,
                        }
                        for trajectory in trajectories
                    ],
                },
            }
        )

    def failure_response(self, body: HarborRunRequest, err: Exception) -> HarborVerifyResponse:
        response = NeMoGymResponse(
            id=f"harbor-error-t{body.task_index}-r{body.rollout_index}",
            created_at=time.time(),
            model=self.config.harbor_agent.model_name,
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            status="failed",
        )
        return HarborVerifyResponse.model_validate(
            body.model_dump(by_alias=True)
            | {
                "response": response.model_dump(mode="json"),
                "reward": 0.0,
                NG_FAILURE_CLASS_KEY: "harbor_failed",
                "error": f"{type(err).__name__}: {err}",
            }
        )

    @staticmethod
    async def run_job(job_config_dict: dict, task_name: str) -> str:
        job_config = JobConfig.model_validate(job_config_dict)
        job_err: Exception | None = None

        try:
            job = await Job.create(job_config)
            await job.run()
        except Exception as err:  # noqa: BLE001 - recover Harbor's partial trial artifacts
            job_err = err

        job_dir = job_config.jobs_dir / job_config.job_name
        if job_dir.exists():
            for trial_dir in job_dir.iterdir():
                if not trial_dir.is_dir():
                    continue

                result_path = TrialPaths(trial_dir).result_path
                if not result_path.is_file():
                    continue

                trial_result = TrialResult.model_validate_json(result_path.read_text())
                if trial_result.task_name != task_name and Path(trial_result.task_name).name != task_name:
                    continue

                if trial_result.exception_info is not None:
                    exception_info = trial_result.exception_info
                    ## Deleting the trial result forces Harbor to delete the old trial and re-run when Gym retries.
                    result_path.unlink()
                    raise RuntimeError(
                        f"Harbor trial failed with {exception_info.exception_type}: {exception_info.exception_message}"
                    )

                return str(trial_dir.resolve())

        if job_err is not None:
            raise job_err

        raise FileNotFoundError(f"No Harbor trial result found in {job_dir}")


if __name__ == "__main__":
    HarborAgent.run_webserver()
