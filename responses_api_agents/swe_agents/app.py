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
from typing import Any, Callable, Dict, Optional

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
    run_swebench_evaluation,
    setup_openhands_environment,
    setup_r2e_gym_environment,
    setup_swebench_environment,
)
from responses_api_models.vllm_model.app import VLLMConverter, split_responses_input_output_items


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
    },
    num_cpus=1,
)
def runner_ray_remote(runner: Callable, params: dict[str, Any]) -> Any:
    ray_submit_time = time.time()
    params["ray_submit_time"] = ray_submit_time

    # This is the first instance so we don't need to load anything
    with params["metrics_fpath"].open("w") as f:
        json.dump({"ray_queue_time": ray_submit_time - params["ray_queue_time"]}, f)

    if params["debug"]:
        instance_id = params["problem_info"].get("instance_id", "unknown")
        profiler = Profiler(name=instance_id, base_profile_dir=params["persistent_dir"] / "profiling")
        profiler.start()

    result = asyncio.run(runner(**params))

    if params["debug"]:
        profiler.stop()

    return result


class SWEBenchWrapperConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef

    # Agent framework configuration
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

    dataset_path: Optional[str] = Field(
        default=None,
        description="Path to the dataset for SWE-bench evaluation",
    )

    openhands_should_log: bool = False
    debug: bool = False


class SWEBenchWrapperServerConfig(BaseModel):
    ng_global_config_dict_str: str
    model_server_name: str
    openhands_setup_dir: Path
    swebench_setup_dir: Path
    r2e_gym_setup_dir: Path
    run_session_id: str
    base_results_dir: Path


class SWEBenchWrapperInstanceConfig(SWEBenchWrapperServerConfig, SWEBenchWrapperConfig):
    output_dir: str
    metrics_fpath: Path
    problem_info: Dict[str, Any]
    body: NeMoGymResponseCreateParamsNonStreaming
    persistent_dir: Path
    metrics_fpath: Path
    ray_queue_time: float


class SWEBenchMetrics(BaseModel):
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: SWEBenchWrapperConfig

    _sem: Semaphore
    _vllm_converter: VLLMConverter
    _swe_bench_wrapper_server_config: SWEBenchWrapperServerConfig

    def model_post_init(self, __context: Any) -> None:
        openhands_setup_dir = setup_openhands_environment(
            agent_framework_repo=self.config.agent_framework_repo,
            agent_framework_commit=self.config.agent_framework_commit,
            debug=self.config.debug,
        )
        swebench_setup_dir = setup_swebench_environment()
        r2e_gym_setup_dir = setup_r2e_gym_environment()
        print("Dependencies repositories set up complete")

        run_session_id = f"{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
        workspace_root = Path(os.path.dirname(os.path.abspath(__file__)))
        self._swe_bench_wrapper_server_config = SWEBenchWrapperServerConfig(
            run_session_id=run_session_id,
            base_results_dir=workspace_root / f"swebench_results_{run_session_id}",
            ng_global_config_dict_str=shlex.quote(OmegaConf.to_yaml(get_global_config_dict())),
            model_server_name=self.config.model_server.name,
            openhands_setup_dir=openhands_setup_dir,
            swebench_setup_dir=swebench_setup_dir,
            r2e_gym_setup_dir=r2e_gym_setup_dir,
        )
        self._swe_bench_wrapper_server_config.base_results_dir.mkdir(parents=True, exist_ok=True)

        self._sem = Semaphore(self.config.concurrency)
        self._vllm_converter = VLLMConverter(return_token_id_information=True)

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        problem_info = body.metadata | {"container_formatter": self.config.container_formatter}

        # Create persistent directory for I/O and logs in local workspace
        instance_dir = (
            f"{problem_info.get('instance_id', 'unknown')}_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
        )
        persistent_dir = self._swe_bench_wrapper_server_config.base_results_dir / instance_dir
        persistent_dir.mkdir(parents=True, exist_ok=True)

        params = SWEBenchWrapperInstanceConfig.model_validate(
            **self.config.model_dump(),
            **self._swe_bench_wrapper_server_config,
            problem_info=problem_info,
            body=body,
            persistent_dir=persistent_dir,
            metrics_fpath=persistent_dir / "nemo_gym_metrics.json",
            ray_queue_time=time.time(),
        )

        result = await runner_ray_remote.remote(self._container_counter, run_swebench_evaluation, params)

        tools = [FunctionTool.model_validate(tool["function"] | {"type": "function"}) for tool in result["tools"]]
        responses_items = self._vllm_converter.chat_completions_messages_to_responses_items(result["trajectory"])
        input_items, output_items = split_responses_input_output_items(responses_items)

        return NeMoGymResponse(
            id=f"swebench-{problem_info.get('instance_id', 'unknown')}",
            created_at=int(time.time()),
            model=body.model,
            object="response",
            output=output_items,
            parallel_tool_calls=body.parallel_tool_calls,
            tool_choice=body.tool_choice,
            tools=tools,
            metadata={
                "input": json.dumps([i.model_dump() for i in input_items]),
                "metrics": None,
            },
        )

    async def run(self, body: BaseRunRequest) -> SWEBenchVerifyResponse:
        async with self._sem:
            body.responses_create_params.parallel_tool_calls = True
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
