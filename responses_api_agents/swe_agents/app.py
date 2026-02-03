# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import os
import shlex
import sys
import time
import uuid
from asyncio import Semaphore
from pathlib import Path
from typing import Any, Callable, Optional

import ray
from openai.types.responses.function_tool import FunctionTool
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import OmegaConf, get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.profiling import Profiler
from responses_api_agents.swe_agents.utils import (
    convert_trajectory_to_output_items,
    extract_problem_info,
    get_model_endpoint,
    run_swebench_evaluation,
    setup_openhands_environment,
    setup_r2e_gym_environment,
    setup_swebench_environment,
)


@ray.remote
class ConcurrentContainerCounter:
    def __init__(self):
        self.concurrent_containers = 0

    def increment(self):
        self.concurrent_containers += 1
        return self.concurrent_containers

    def decrement(self):
        self.concurrent_containers -= 1
        return self.concurrent_containers


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
    },
    num_cpus=1,
)
def runner_ray_remote(
    concurrent_container_counter: ConcurrentContainerCounter, runner: Callable, params: dict[str, Any]
) -> Any:
    ray_submit_time = time.time()
    params["ray_submit_time"] = ray_submit_time

    # This is the first instance so we don't need to load anything
    with params["metrics_fpath"].open("w") as f:
        json.dump({"ray_queue_time": ray_submit_time - params["ray_queue_time"]}, f)

    if params["debug"]:
        concurrent_containers = ray.get(concurrent_container_counter.increment.remote())
        print(f"Concurrent container #{concurrent_containers}", file=sys.stderr)

        instance_id = params["problem_info"].get("instance_id", "unknown")
        profiler = Profiler(name=instance_id, base_profile_dir=params["persistent_dir"] / "profiling")
        profiler.start()

    result = asyncio.run(runner(**params))

    if params["debug"]:
        profiler.stop()

        ray.get(concurrent_container_counter.decrement.remote())

    return result


class SWEBenchWrapperConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef

    # Agent framework configuration
    agent_framework: str = Field(
        default="swe_agent",
        description="Agent framework to use: swe_agent or openhands",
    )
    agent_config: Optional[str] = Field(default=None, description="Path to agent configuration file")
    agent_tools_file: Optional[str] = Field(
        default=None, description="Path to JSON file containing tool definitions in OpenAI format (for SWE-agent)"
    )
    agent_max_turns: int = Field(default=100, description="Maximum iterations for the agent")
    agent_framework_repo: Optional[str] = Field(
        default=None,
        description="URL of the SWE-agent/OpenHands repo to pass to git clone. If None, will use the official repo",
    )

    agent_framework_commit: str = Field(
        default="HEAD", description="Which commit to use when cloning the SWE-agent/OpenHands repo"
    )
    # Container configuration
    container_formatter: str | list[str] = Field(
        default="docker://swebench/sweb.eval.x86_64.{instance_id}", description="Container path template"
    )
    swebench_tests_timeout: int = Field(default=30 * 60, description="Timeout for running tests (seconds)")

    swebench_agent_timeout: int = Field(default=45 * 60, description="Timeout for running the agent (seconds)")

    apptainer_memory_limit_mb: int = Field(
        default=32 * 1024, description="Memory limit for the apptainer container (MB)"
    )

    command_exec_timeout: int = Field(default=5 * 60, description="Timeout for executing the command (seconds)")

    # Concurrency control
    concurrency: int = Field(default=256, description="Maximum number of concurrent SWE-bench runs")

    # Pre-built OpenHands directory path (set during initialization)
    openhands_setup_dir: Optional[Path] = Field(
        default=None,
        description="Path to pre-built OpenHands directory (automatically set during initialization)",
        exclude=True,
    )

    # Pre-built SWE-bench directory path (set during initialization)
    swebench_setup_dir: Optional[Path] = Field(
        default=None,
        description="Path to pre-built SWE-bench directory (automatically set during initialization)",
        exclude=True,
    )
    # Pre-built R2E-gym directory path (set during initialization)
    r2e_gym_setup_dir: Optional[Path] = Field(
        default=None,
        description="Path to pre-built R2E-gym directory (automatically set during initialization)",
        exclude=True,
    )
    dataset_path: Optional[str] = Field(
        default=None,
        description="Path to the dataset for SWE-bench evaluation",
    )

    run_session_id: str = Field(
        default=None,
        description="Session ID for the run",
    )

    openhands_should_log: bool = False
    debug: bool = False


class SWEBenchMetrics(BaseModel):
    agent_framework: str
    instance_id: str
    instance_dir: str

    resolved: bool
    patch_exists: bool
    patch_successfully_applied: bool

    # Profiling time metrics to report
    ray_queue_time: float
    # generation_apptainer_spinup_time: float
    # create_runtime_time: float
    # container_initialization_time: float
    # connect_to_runtime_time: float
    # runtime_initialization_fn_time: float
    # total_command_exec_time: float
    # total_model_call_time: float
    # final_eval_apptainer_spinup_time: float
    final_eval_time: float

    # Exit condition metrics to report
    # TODO add more exit conditions
    # hit_sample_timeout: bool
    # hit_trajectory_command_exec_timeout: bool
    # hit_eval_timeout: bool
    hit_empty_trajectory: bool
    hit_success: bool
    hit_responses_exception: bool


class SWEBenchVerifyResponse(SWEBenchMetrics, BaseVerifyResponse):
    pass


class SWEBenchWrapper(SimpleResponsesAPIAgent):
    """Wrapper for NeMo-Skills SWE-bench evaluation in NeMo-Gym."""

    config: SWEBenchWrapperConfig
    sem: Semaphore = None
    _container_counter: ConcurrentContainerCounter = None
    _global_config_dict_str: str = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        self._container_counter = ConcurrentContainerCounter.remote()

        # Pre-build OpenHands environment if using openhands framework
        if self.config.agent_framework == "openhands":
            self.config.openhands_setup_dir = setup_openhands_environment(
                agent_framework_repo=self.config.agent_framework_repo,
                agent_framework_commit=self.config.agent_framework_commit,
                debug=self.config.debug,
            )
        self.config.swebench_setup_dir = setup_swebench_environment()
        self.config.r2e_gym_setup_dir = setup_r2e_gym_environment()

        print("Dependencies repositories set up complete", flush=True)

        self.config.run_session_id = f"{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
        print(f"Run session ID: {self.config.run_session_id}", flush=True)

        self._global_config_dict_str = shlex.quote(OmegaConf.to_yaml(get_global_config_dict()))

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        # Extract problem information from request
        problem_info = extract_problem_info(
            body,
            self.config.container_formatter,
        )

        # Get model endpoint
        model_endpoint = get_model_endpoint(self.config.model_server.name)

        # Create persistent directory for I/O and logs in local workspace
        instance_dir = (
            f"{problem_info.get('instance_id', 'unknown')}_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
        )
        workspace_root = Path(os.path.dirname(os.path.abspath(__file__)))
        persistent_dir = workspace_root / f"swebench_results_{self.config.run_session_id}" / instance_dir
        persistent_dir.mkdir(parents=True, exist_ok=True)
        metrics_fpath = persistent_dir / "nemo_gym_metrics.json"

        ray_queue_time = time.time()
        params = {
            "problem_info": problem_info,
            "model_endpoint": model_endpoint,
            "body": body,
            "agent_framework": self.config.agent_framework,
            "agent_config": self.config.agent_config,
            "agent_tools_file": self.config.agent_tools_file,
            "agent_max_turns": self.config.agent_max_turns,
            "swebench_tests_timeout": self.config.swebench_tests_timeout,
            "swebench_agent_timeout": self.config.swebench_agent_timeout,
            "persistent_dir": persistent_dir,
            "metrics_fpath": metrics_fpath,
            "agent_framework_repo": self.config.agent_framework_repo,
            "agent_framework_commit": self.config.agent_framework_commit,
            "openhands_setup_dir": self.config.openhands_setup_dir,
            "swebench_setup_dir": self.config.swebench_setup_dir,
            "r2e_gym_setup_dir": self.config.r2e_gym_setup_dir,
            "dataset_path": self.config.dataset_path,
            "ray_queue_time": ray_queue_time,
            "openhands_should_log": self.config.openhands_should_log,
            "debug": self.config.debug,
            "model_server_name": self.config.model_server.name,
            "ng_global_config_dict_str": self._global_config_dict_str,
            "apptainer_memory_limit_mb": self.config.apptainer_memory_limit_mb,
            "command_exec_timeout": self.config.command_exec_timeout,
        }

        # Run SWE-bench evaluation
        future = runner_ray_remote.remote(self._container_counter, run_swebench_evaluation, params)
        result = await future

        # Extract trajectory and convert to proper NeMoGym format
        output_items = []
        trajectory = result.get("trajectory", [])

        # Convert tools from ChatCompletion format to Response FunctionTool format
        raw_tools = result.get("tools", [])
        tools = [FunctionTool.model_validate(tool["function"] | {"type": "function"}) for tool in raw_tools]

        # Convert trajectory to NeMoGym output items
        if trajectory:
            output_items = convert_trajectory_to_output_items(
                trajectory,
                self.config.agent_framework,
            )

        # Note: metadata values must be strings for NeMoGymResponse
        metadata = {
            "input": None,
            "metrics": None,
        }

        return NeMoGymResponse(
            id=f"swebench-{problem_info.get('instance_id', 'unknown')}",
            created_at=int(time.time()),
            model=body.model,
            object="response",
            output=output_items,
            parallel_tool_calls=body.parallel_tool_calls,
            tool_choice=body.tool_choice,
            tools=tools,
            metadata=metadata,
        )

    async def run(self, body: BaseRunRequest) -> SWEBenchVerifyResponse:
        async with self.sem:
            if self.config.debug:
                print(
                    f"Semaphore: {self.config.concurrency - self.sem._value} / {self.config.concurrency}", flush=True
                )
            body.responses_create_params.metadata["container_concurrency"] = str(
                self.config.concurrency - self.sem._value
            )

            return await self._run(body)

    async def _run(self, body: BaseRunRequest) -> SWEBenchVerifyResponse:
        # SWE-agent processes tool calls sequentially, OpenHands can do parallel
        body.responses_create_params.parallel_tool_calls = (
            False if self.config.agent_framework == "swe_agent" else True
        )
        if body.responses_create_params.tool_choice is None:
            body.responses_create_params.tool_choice = "auto"

        response = await self.responses(body.responses_create_params)

        metadata, response.metadata = response.metadata, None
        params_with_input = body.responses_create_params.model_copy(
            update={
                "input": json.loads(metadata["input"]),
                "tools": [t.model_dump() for t in response.tools] if response.tools else [],
            }
        )
        metrics = SWEBenchMetrics.model_validate_json(metadata["metrics"])

        return SWEBenchVerifyResponse(
            responses_create_params=params_with_input,
            response=response,
            reward=1.0 if metrics.resolved else 0.0,
            **metrics.model_dump(),
        )


if __name__ == "__main__":
    SWEBenchWrapper.run_webserver()
