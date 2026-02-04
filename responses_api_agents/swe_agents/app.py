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
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path
from shutil import rmtree
from subprocess import DEVNULL, Popen
from subprocess import run as subprocess_run
from typing import Any, Dict, Optional

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
from responses_api_agents.swe_agents.run_openhands import (
    RunOpenHandsAgent,
)
from responses_api_models.vllm_model.app import VLLMConverter, split_responses_input_output_items


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
    },
    num_cpus=1,
)
def runner_ray_remote(params_dict: dict[str, Any]) -> None:
    params = SWEBenchWrapperInstanceConfig.model_validate(params_dict)
    instance_id = params.problem_info["instance_id"]

    if params.debug:
        profiler = Profiler(name=instance_id, base_profile_dir=params.persistent_dir / "profiling")
        profiler.start()

    run_oh = RunOpenHandsAgent(config=params)

    asyncio.run(run_oh.process_single_datapoint())

    if params.debug:
        profiler.stop()


########################################
# START Configuration
########################################


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
    inference_params: Dict[str, Any]
    agent_run_id: str
    instance_dataset_path: Path


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


########################################
# START Dataset and harness handling
########################################


class BaseDatasetHarnessProcessor(BaseModel):
    config: SWEBenchWrapperConfig

    ########################################
    # START Setup logic
    ########################################

    @property
    def parent_dir(self) -> Path:
        return Path(__file__).parent

    def _run_setup_command(self, command: str) -> None:
        std_params = dict()
        if not self.config.debug:
            std_params = dict(
                stdout=DEVNULL,
                stderr=DEVNULL,
            )

        process = Popen(command, shell=True, **std_params)
        return_code = process.wait()
        assert return_code == 0, f"Command failed: {command}"

    @contextmanager
    def _setup_directory_lock(self, setup_dir: Path, label: str):
        """File-based lock to ensure only one process performs the setup."""
        lock_dir = setup_dir.parent
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f".{setup_dir.name}.lock"

        with open(lock_path, "w") as lock_file:
            print(f"Acquiring {label} setup lock at {lock_path}", flush=True)
            flock(lock_file, LOCK_EX)
            try:
                yield
            finally:
                flock(lock_file, LOCK_UN)

    def setup(self) -> Path:
        pass


class SweBenchDatasetProcessor(BaseDatasetHarnessProcessor):
    def setup(self) -> Path:
        swebench_repo = "https://github.com/HeyyyyyyG/SWE-bench.git"
        swebench_commit = "HEAD"

        setup_dir = self.parent_dir / "swe_swebench_setup"
        setup_dir.mkdir(parents=True, exist_ok=True)

        with self._setup_directory_lock(setup_dir, "SWE-bench"):
            swebench_dir = setup_dir / "SWE-bench"
            uv_dir = setup_dir / "uv"
            python_dir = setup_dir / "python"

            if swebench_dir.exists():
                print(f"SWE-bench already set up at {setup_dir}")
                return

            print(f"Setting up SWE-bench environment at {setup_dir}...", flush=True)
            script_fpath = self.parent_dir / "setup_scripts/swebench.sh"
            command = f"""SETUP_DIR={setup_dir} \\
UV_DIR={uv_dir} \\
PYTHON_DIR={python_dir} \\
SWEBENCH_DIR={swebench_dir} \\
SWEBENCH_REPO={swebench_repo} \\
SWEBENCH_COMMIT={swebench_commit} \\
    ./{script_fpath}"""
            self._run_setup_command(command)

            return setup_dir


class R2EGymDatasetProcessor(BaseDatasetHarnessProcessor):
    def setup(self) -> Path:
        eval_harness_repo = "https://github.com/ludwig-n/R2E-Gym.git"
        eval_harness_commit = "local-eval"

        setup_dir = self.parent_dir / "swe_r2e_gym_setup"

        with self._setup_directory_lock(setup_dir, "R2E-Gym"):
            r2e_gym_dir = setup_dir / "R2E-Gym"
            uv_dir = setup_dir / "uv"
            python_dir = setup_dir / "python"

            # Check if setup is complete by verifying venv and installed module
            venv_dir = r2e_gym_dir / "venv"
            python_bin = venv_dir / "bin" / "python"
            if r2e_gym_dir.exists() and venv_dir.exists() and python_bin.exists():
                result = subprocess_run([str(python_bin), "-c", "import r2egym"])
                if result.returncode == 0:
                    print(f"R2E-Gym already set up at {setup_dir}", flush=True)
                    return

                print("R2E-Gym directory exists but module not properly installed, rebuilding...", flush=True)

            print(f"Setting up R2E-Gym environment at {setup_dir}...", flush=True)
            setup_dir.mkdir(parents=True, exist_ok=True)

            script_fpath = self.parent_dir / "setup_scripts/r2e_gym.sh"
            command = f"""SETUP_DIR={setup_dir} \\
UV_DIR={uv_dir} \\
PYTHON_DIR={python_dir} \\
R2E_GYM_DIR={r2e_gym_dir} \\
EVAL_HARNESS_REPO={eval_harness_repo} \\
EVAL_HARNESS_COMMIT={eval_harness_commit} \\
    ./{script_fpath}"""
            self._run_setup_command(command)

            return setup_dir


class OpenHandsHarnessProcessor(BaseDatasetHarnessProcessor):
    def setup(self) -> Path:
        setup_dir = self.parent_dir / "swe_openhands_setup"

        with self._setup_directory_lock(setup_dir, "OpenHands"):
            openhands_dir = setup_dir / "OpenHands"
            miniforge_dir = setup_dir / "miniforge3"

            if openhands_dir.exists() and Path(openhands_dir / ".venv" / "bin" / "python").exists():
                print(f"OpenHands already set up at {setup_dir}", flush=True)
                return

            print(f"Setting up OpenHands environment at {setup_dir}...", flush=True)
            rmtree(setup_dir, ignore_errors=True)
            setup_dir.mkdir(parents=True, exist_ok=True)

            script_fpath = self.parent_dir / "setup_scripts/openhands.sh"
            command = f"""SETUP_DIR={setup_dir} \\
MINIFORGE_DIR={miniforge_dir} \\
OPENHANDS_DIR={openhands_dir} \\
AGENT_FRAMEWORK_REPO={self.config.agent_framework_repo} \\
AGENT_FRAMEWORK_COMMIT={self.config.agent_framework_commit} \\
    ./{script_fpath}"""
            self._run_setup_command(command)

            return setup_dir


########################################
# START Server logic
########################################


class SWEBenchWrapper(SimpleResponsesAPIAgent):
    """Wrapper for NeMo-Skills SWE-bench evaluation in NeMo-Gym."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: SWEBenchWrapperConfig

    _sem: Semaphore
    _vllm_converter: VLLMConverter
    _swe_bench_wrapper_server_config: SWEBenchWrapperServerConfig

    ########################################
    # START Init
    ########################################

    def model_post_init(self, __context: Any) -> None:
        run_session_id = f"{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
        workspace_root = Path(os.path.dirname(os.path.abspath(__file__)))
        self._swe_bench_wrapper_server_config = SWEBenchWrapperServerConfig(
            run_session_id=run_session_id,
            base_results_dir=workspace_root / f"swebench_results_{run_session_id}",
            ng_global_config_dict_str=shlex.quote(OmegaConf.to_yaml(get_global_config_dict())),
            model_server_name=self.config.model_server.name,
            openhands_setup_dir=OpenHandsHarnessProcessor(self.config).setup(),
            swebench_setup_dir=SWEBenchWrapper(self.config).setup(),
            r2e_gym_setup_dir=R2EGymDatasetProcessor(self.config).setup(),
        )
        self._swe_bench_wrapper_server_config.base_results_dir.mkdir(parents=True, exist_ok=True)

        self._sem = Semaphore(self.config.concurrency)
        self._vllm_converter = VLLMConverter(return_token_id_information=True)

    ########################################
    # START Results processing logic
    ########################################

    def get_openhands_trajectory_from_completions(self, trajectories_dir: Path, instance_id: str) -> tuple:
        """
        This reads the trajectories directly dumped by OpenHands.
        """
        messages, tools = [], []

        completions_dir = trajectories_dir / instance_id / "llm_completions" / instance_id
        if not completions_dir.exists():
            print(f"No llm_completions directory found: {completions_dir}", flush=True)
            return messages, tools

        completion_files = sorted(completions_dir.glob("*.json"))
        if not completion_files:
            print(f"No completion files found in: {completions_dir}", flush=True)
            return messages, tools

        last_file = completion_files[-1]

        with open(last_file, "r") as f:
            data = json.load(f)

        messages = data["messages"]
        provider_specific_fields = data.get("provider_specific_fields", {})
        final_assistant_message = data["response"]["choices"][0]["message"]

        for key in ["prompt_token_ids", "generation_token_ids", "generation_log_probs"]:
            if key in provider_specific_fields:
                final_assistant_message[key] = provider_specific_fields[key]

        if final_assistant_message.get("content") or final_assistant_message.get("tool_calls"):
            messages.append(final_assistant_message)

        tools = data.get("kwargs", {}).get("tools", [])

        return messages, tools

    ########################################
    # START Main methods
    ########################################

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        problem_info = body.metadata | {"container_formatter": self.config.container_formatter}
        instance_id = problem_info.get("instance_id", "unknown")

        # Create persistent directory for I/O and logs in local workspace
        instance_dir = f"{instance_id}_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
        persistent_dir = self._swe_bench_wrapper_server_config.base_results_dir / instance_dir
        persistent_dir.mkdir(parents=True, exist_ok=True)

        agent_run_id = f"{instance_id}_{int(time.time())}_{str(uuid.uuid4())[:8]}"

        # To avoid making HF dataset API calls, we write the instance dictionary to a file and mount it in the container.
        instance_dataset_dir = Path(persistent_dir) / "instance_datasets"
        instance_dataset_dir.mkdir(parents=True, exist_ok=True)
        instance_dataset_path = instance_dataset_dir / f"{agent_run_id}.jsonl"
        instance_dict = json.loads(problem_info["instance_dict"])
        if "repo" in instance_dict and "repo_name" not in instance_dict:
            instance_dict["repo_name"] = instance_dict["repo"]
        with open(instance_dataset_path, "w") as f:
            f.write(json.dumps(instance_dict) + "\n")

        # Map from Responses to OpenHands
        inference_params = {}
        for param, key in [
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("max_output_tokens", "tokens_to_generate"),
        ]:
            value = getattr(body, param, None)
            if value is not None:
                inference_params[key] = value

        params: SWEBenchWrapperInstanceConfig = SWEBenchWrapperInstanceConfig.model_validate(
            **self.config.model_dump(),
            **self._swe_bench_wrapper_server_config,
            problem_info=problem_info,
            body=body,
            persistent_dir=persistent_dir,
            metrics_fpath=persistent_dir / "nemo_gym_metrics.json",
            ray_queue_time=time.time(),
            inference_params=inference_params,
            agent_run_id=agent_run_id,
            instance_dataset_path=instance_dataset_path,
        )

        await runner_ray_remote.remote(params.model_dump())

        trajectories_dir = persistent_dir / "trajectories"
        chat_completions_trajectory, chat_completions_tools = self.get_openhands_trajectory_from_completions(
            trajectories_dir, instance_id
        )

        tools = [
            FunctionTool.model_validate(tool["function"] | {"type": "function"}) for tool in chat_completions_tools
        ]
        responses_items = self._vllm_converter.chat_completions_messages_to_responses_items(
            chat_completions_trajectory
        )
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
                "metrics": params.metrics_fpath.read_text(),
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
