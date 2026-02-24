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
import re
import sys
from asyncio import Semaphore
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

import ray
from fastapi import Body, FastAPI
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
from responses_api_agents.harbor_agent.utils import HarborAgentUtils


class HarborAgentConfig(BaseResponsesAPIAgentConfig):
    concurrency: int

    # --- Harbor agent settings ---
    # Name of a built-in Harbor agent (e.g. "terminus-2", "claude-code", "aider").
    harbor_agent_name: Optional[str] = "terminus-2"
    # Python import path for a custom agent class (e.g. "my_pkg.my_mod:MyAgent").
    # Overrides harbor_agent_name when set.
    harbor_agent_import_path: Optional[str] = None
    # Extra kwargs forwarded to the Harbor AgentConfig (e.g. collect_rollout_details,
    # model_info). See harbor_agent.yaml for examples.
    harbor_agent_kwargs: Optional[dict[str, Any]] = None

    # --- Dataset ---
    # Registry dataset identifier (e.g. "terminal-bench@2.0"). Mutually exclusive
    # with harbor_local_dataset_path.
    harbor_dataset_name: Optional[str] = None
    harbor_dataset_version: Optional[str] = None
    # Absolute path to a local task directory. Mutually exclusive with harbor_dataset_name.
    harbor_local_dataset_path: Optional[str] = None

    # --- Environment ---
    # Harbor environment type: "singularity", "docker", "daytona", "modal", etc.
    harbor_environment_type: Optional[str] = "singularity"
    # Python import path for a custom environment class (e.g. "my_pkg.my_mod:MyEnv").
    # Overrides harbor_environment_type when set.
    harbor_environment_import_path: Optional[str] = None
    # Extra kwargs forwarded to the Harbor EnvironmentConfig (e.g.
    # singularity_image_cache_dir, singularity_force_pull).
    harbor_environment_kwargs: Optional[dict[str, Any]] = None

    # --- Timeouts ---
    # Per-agent timeout in seconds. None = use Harbor's default.
    harbor_agent_timeout: Optional[int] = None
    # Per-verifier timeout in seconds. None = use Harbor's default.
    harbor_verifier_timeout: Optional[int] = None
    # Multiplier applied to all Harbor timeouts. None = use Harbor's default (1.0).
    harbor_timeout_multiplier: Optional[float] = None

    # --- Job output ---
    # Directory where Harbor writes job results and trial artifacts.
    harbor_jobs_dir: str = "jobs"

    # --- Model routing ---
    # NeMo Gym model server reference used to resolve Harbor model base URL.
    model_server: ModelServerRef
class HarborRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    instance_id: str


class HarborVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


async def run_harbor_job(job_config_dict: dict) -> str:
    """Runs a single Harbor Job and returns the trial directory path.

    The trial directory contains:
    - result.json: Summary result with reward, agent_result, verifier_result, etc.
    - agent/trajectory.json: Full ATIF trajectory with per-step messages, tool
      calls, observations, and per-token logprobs.

    Harbor writes result.json and trajectory.json to disk even when the trial
    fails (e.g. verifier timeout, reward file not found, OOM).  We recover the
    trial directory after an exception so the caller can still use the partial
    trajectory for training.
    """
    from harbor.job import Job
    from harbor.models.job.config import JobConfig

    config = JobConfig(**job_config_dict)
    job = Job(config)

    job_error = None
    try:
        await job.run()
    except Exception as e:
        job_error = e

    # Find the trial directory from the job output directory.  Harbor writes
    # result.json before propagating most exceptions, so we can usually
    # recover the trial even when job.run() raised.
    job_dir = config.jobs_dir / config.job_name
    if job_dir.exists():
        for trial_dir in job_dir.iterdir():
            if not trial_dir.is_dir():
                continue
            result_path = trial_dir / "result.json"
            if result_path.exists():
                return str(trial_dir)

    # No trial directory found — re-raise the original error if there was one,
    # otherwise raise FileNotFoundError.
    if job_error is not None:
        raise job_error
    raise FileNotFoundError(f"No trial result found in {job_dir}")


_RAY_WORKER_EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _run_harbor_job_sync(job_config_dict: dict) -> str:
    """Synchronous wrapper for run_harbor_job for use in Ray remote.

    Ray workers are long-lived processes. Reusing a single event loop per worker
    avoids cross-loop issues with global async state (e.g., LiteLLM logging worker
    queues) when multiple jobs execute sequentially in the same process.
    """
    global _RAY_WORKER_EVENT_LOOP
    if _RAY_WORKER_EVENT_LOOP is None or _RAY_WORKER_EVENT_LOOP.is_closed():
        _RAY_WORKER_EVENT_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_RAY_WORKER_EVENT_LOOP)
    return _RAY_WORKER_EVENT_LOOP.run_until_complete(run_harbor_job(job_config_dict))


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
    },
)
def runner_ray_remote(runner: Callable, params: dict[str, Any]) -> Any:
    return runner(**params)


class HarborAgent(SimpleResponsesAPIAgent):
    config: HarborAgentConfig
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

    async def run(self, body: HarborRunRequest) -> HarborVerifyResponse:
        async with self.sem:
            global_config_dict = get_global_config_dict()

            policy_model_name = global_config_dict["policy_model_name"]
            base_url = self._resolve_model_base_url(global_config_dict)
            run_timestamp = datetime.now(timezone.utc)
            run_id = self._build_run_id(run_timestamp)

            instance_id = body.instance_id

            output_file_dir = self._get_results_output_dir(policy_model_name, run_timestamp)
            jobs_dir = self._get_jobs_output_dir(policy_model_name, run_timestamp)
            job_name = self._build_job_name(run_id)

            responses_create_params = body.responses_create_params.model_dump(
                exclude_unset=True,
                exclude_none=True,
            )

            job_config_dict = self._build_job_config(
                instance_id,
                policy_model_name,
                base_url,
                job_name=job_name,
                jobs_dir=jobs_dir,
                responses_create_params=responses_create_params,
            )

            try:
                params = dict(
                    job_config_dict=job_config_dict,
                )
                future = runner_ray_remote.remote(_run_harbor_job_sync, params)
                trial_dir_path = await asyncio.to_thread(ray.get, future)
                trial_dir = Path(trial_dir_path)

                # Read the trial result (summary: reward, agent_result, verifier_result)
                with open(trial_dir / "result.json", "r") as f:
                    trial_result = json.load(f)

                # Read the ATIF trajectory (full conversation with per-token logprobs)
                trajectory = None
                trajectory_path = trial_dir / "agent" / "trajectory.json"
                if trajectory_path.exists():
                    with open(trajectory_path, "r") as f:
                        trajectory = json.load(f)

                # Extract reward from verifier result
                verifier_result = trial_result.get("verifier_result")
                reward = HarborAgentUtils.extract_reward(verifier_result)

                # Convert Harbor outputs to NeMo Gym response items:
                # keep rich trajectory details, then overlay rollout token details when present.
                output_items = HarborAgentUtils.trial_result_to_responses(trial_result, trajectory)

                # Extract the initial instruction from the trajectory as input messages
                input_messages = HarborAgentUtils.extract_input_from_trajectory(trajectory)

                # Populate usage from trajectory final_metrics or agent_result
                usage = HarborAgentUtils.extract_usage(trial_result, trajectory)

            except Exception as e:
                print(f"Error running Harbor job: {e}")
                trial_result = None
                trajectory = None
                output_items = []
                input_messages = []
                usage = None
                reward = 0.0

            response = HarborAgentUtils.get_default_response_object()
            response["model"] = policy_model_name
            response["temperature"] = responses_create_params.get("temperature")
            response["top_p"] = responses_create_params.get("top_p")
            response["output"] = output_items
            if usage:
                response["usage"] = usage

            # Update responses_create_params with the actual input sent to the agent
            updated_params = body.responses_create_params
            if input_messages:
                updated_params = body.responses_create_params.model_copy(
                    update={"input": input_messages}
                )

            verify_response = HarborVerifyResponse(
                responses_create_params=updated_params,
                reward=reward,
                response=response,
                instance_id=instance_id,
                metadata=trial_result if trial_result else {},
            )

            # Save result to disk (folder = run_id/date, file = task name)
            output_path = output_file_dir / run_id
            output_path.mkdir(parents=True, exist_ok=True)

            with open(output_path / f"{instance_id}.json", "w") as f:
                json.dump(verify_response.model_dump(), f, indent=2)

            return verify_response

    def _get_results_output_dir(self, policy_model_name: str, run_timestamp: datetime) -> Path:
        """Build immutable run output directory grouped by dataset/model."""
        dataset_key = self._sanitize_path_component(self._get_dataset_key())
        model_key = self._sanitize_path_component(self._extract_model_name(policy_model_name))
        return (
            Path.cwd()
            / "results"
            / "runs"
            / dataset_key
            / model_key
        )

    def _get_jobs_output_dir(self, policy_model_name: str, run_timestamp: datetime) -> Path:
        """Build Harbor jobs directory grouped by dataset/model."""
        dataset_key = self._sanitize_path_component(self._get_dataset_key())
        model_key = self._sanitize_path_component(self._extract_model_name(policy_model_name))
        return (
            Path(self.config.harbor_jobs_dir)
            / dataset_key
            / model_key
        )

    def _get_dataset_key(self) -> str:
        """Derive a stable dataset key for folder naming."""
        if self.config.harbor_dataset_name:
            version = self.config.harbor_dataset_version or "latest"
            return f"{self.config.harbor_dataset_name}@{version}"
        if self.config.harbor_local_dataset_path:
            return Path(self.config.harbor_local_dataset_path).name
        return "unknown_dataset"

    def _build_run_id(self, run_timestamp: datetime) -> str:
        """Build a compact, sortable run id for immutable file naming."""
        time_key = run_timestamp.strftime("%Y%m%d_%H%M%S")
        return f"{time_key}_{uuid4().hex[:8]}"

    def _build_job_name(self, run_id: str) -> str:
        """Build a Harbor job name from run id only."""
        return run_id

    @staticmethod
    def _extract_model_name(policy_model_name: str) -> str:
        """Extract the final model name from a full path or HF-style identifier.

        '/lustre/.../nano-v3-sft-...-hf'  -> 'nano-v3-sft-...-hf'
        'Qwen/Qwen3-8B'                   -> 'Qwen3-8B'
        'my-model'                         -> 'my-model'
        """
        return Path(policy_model_name).name or policy_model_name

    def _sanitize_path_component(self, value: str) -> str:
        """Sanitize path components to avoid accidental nested directories."""
        sanitized = value.replace("/", "__").replace("\\", "__")
        sanitized = re.sub(r"\s+", "_", sanitized)
        sanitized = sanitized.strip("._")
        return sanitized or "unknown"

    def _resolve_model_base_url(self, global_config_dict: Any) -> str:
        """Resolve model base URL from required model_server reference."""
        server_name = self.config.model_server.name
        model_server_config = get_first_server_config_dict(
            global_config_dict,
            server_name,
        )
        return f"http://{model_server_config['host']}:{model_server_config['port']}/v1"

    def _build_job_config(
        self,
        instance_id: str,
        model_name: str,
        api_base: str,
        job_name: str,
        jobs_dir: Path,
        responses_create_params: Optional[dict[str, Any]] = None,
    ) -> dict:
        """Build a Harbor JobConfig dict for a single task."""
        from harbor.models.job.config import (
            JobConfig,
            LocalDatasetConfig,
            OrchestratorConfig,
            RegistryDatasetConfig,
        )
        from harbor.models.registry import RemoteRegistryInfo
        from harbor.models.trial.config import (
            AgentConfig,
            EnvironmentConfig,
            VerifierConfig,
        )

        agent_kwargs: dict[str, Any] = {"api_base": api_base}
        if responses_create_params:
            agent_kwargs["responses_create_params"] = responses_create_params
            # Terminus-2 accepts temperature as a top-level kwarg for trajectory metadata.
            if "temperature" in responses_create_params:
                agent_kwargs["temperature"] = responses_create_params["temperature"]
        if self.config.harbor_agent_kwargs:
            agent_kwargs.update(self.config.harbor_agent_kwargs)

        agent_config = AgentConfig(
            name=self.config.harbor_agent_name if not self.config.harbor_agent_import_path else None,
            import_path=self.config.harbor_agent_import_path,
            model_name=model_name,
            override_timeout_sec=(
                float(self.config.harbor_agent_timeout)
                if self.config.harbor_agent_timeout is not None
                else None
            ),
            kwargs=agent_kwargs,
        )

        environment_kwargs = {}
        if self.config.harbor_environment_kwargs:
            environment_kwargs.update(self.config.harbor_environment_kwargs)

        environment_config = EnvironmentConfig(
            type=self.config.harbor_environment_type if not self.config.harbor_environment_import_path else None,
            import_path=self.config.harbor_environment_import_path,
            kwargs=environment_kwargs,
        )

        verifier_config = VerifierConfig(
            override_timeout_sec=(
                float(self.config.harbor_verifier_timeout)
                if self.config.harbor_verifier_timeout is not None
                else None
            ),
        )

        orchestrator_config = OrchestratorConfig(
            n_concurrent_trials=1,
            quiet=True,
        )

        # Build dataset config — exactly one source must be configured
        if self.config.harbor_dataset_name:
            dataset_config = RegistryDatasetConfig(
                registry=RemoteRegistryInfo(),
                name=self.config.harbor_dataset_name,
                version=self.config.harbor_dataset_version,
                task_names=[instance_id],
            )
        elif self.config.harbor_local_dataset_path:
            dataset_config = LocalDatasetConfig(
                path=Path(self.config.harbor_local_dataset_path),
                task_names=[instance_id],
            )
        else:
            raise ValueError(
                "Harbor agent requires a dataset. Set either harbor_dataset_name or harbor_local_dataset_path."
            )

        job_config = JobConfig(
            job_name=job_name,
            jobs_dir=jobs_dir,
            timeout_multiplier=(
                self.config.harbor_timeout_multiplier
                if self.config.harbor_timeout_multiplier is not None
                else 1.0
            ),
            orchestrator=orchestrator_config,
            environment=environment_config,
            verifier=verifier_config,
            agents=[agent_config],
            datasets=[dataset_config],
        )

        return job_config.model_dump(mode="json")


if __name__ == "__main__":
    HarborAgent.run_webserver()
