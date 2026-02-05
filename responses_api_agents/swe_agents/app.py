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
import glob
import json
import os
import re
import shlex
import shutil
import sys
import time
import uuid
from asyncio import Semaphore
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path
from shutil import rmtree
from subprocess import Popen
from subprocess import run as subprocess_run
from typing import Any, Dict, Literal, Optional, Tuple, Union

import ray
import tomlkit
from gprof2dot import main as gprof2dot_main
from openai.types.responses.function_tool import FunctionTool
from pydantic import BaseModel, ConfigDict, Field
from pydot import graph_from_dot_file

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
from responses_api_models.vllm_model.app import VLLMConverter, split_responses_input_output_items


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


class ExecuteContainerCommandArgs(BaseModel):
    command: str
    expected_file_pattern: str
    mode: Union[Literal["agent"], Literal["eval"]]
    timeout: int


class SWEBenchWrapperInstanceConfig(SWEBenchWrapperServerConfig, SWEBenchWrapperConfig):
    metrics_fpath: Path
    problem_info: Dict[str, Any]
    body: NeMoGymResponseCreateParamsNonStreaming
    persistent_dir: Path
    ray_queue_timestamp: float
    inference_params: Dict[str, Any]
    agent_run_id: str
    instance_dataset_path: Path
    trajectories_root: Path
    prediction_path: Path
    prediction_mounted_path: Path
    model_patch_path: Path
    container: str
    eval_dir_in_openhands: str
    openhands_config_file_path: str
    agent_script_path: Path

    # Set later
    eval_command: Optional[ExecuteContainerCommandArgs] = None
    eval_apptainer_command_str: Optional[str] = None
    agent_command: Optional[ExecuteContainerCommandArgs] = None
    agent_apptainer_command_str: Optional[str] = None
    agent_script: Optional[str] = None

    @property
    def instance_id(self) -> str:
        return self.problem_info["instance_id"]


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
    instance_config: SWEBenchWrapperInstanceConfig


########################################
# START Dataset and harness handling
########################################


class BaseDatasetHarnessProcessor(BaseModel):
    config: SWEBenchWrapperConfig | SWEBenchWrapperInstanceConfig

    ########################################
    # START Setup logic
    ########################################

    @property
    def parent_dir(self) -> Path:
        return Path(__file__).parent

    def _run_setup_command(self, command: str) -> None:
        process = Popen(command, shell=True)
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

    # Setup method is sync for now since there's been no need to concurrently set up
    def setup(self) -> Path:
        pass

    def get_run_command(self) -> ExecuteContainerCommandArgs:
        pass

    def postprocess_after_eval_run(self, report_file: Path) -> None:
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
                return setup_dir

            print(f"Setting up SWE-bench environment at {setup_dir}...", flush=True)
            script_fpath = self.parent_dir / "setup_scripts/swebench.sh"
            command = f"""SETUP_DIR={setup_dir} \\
UV_DIR={uv_dir} \\
PYTHON_DIR={python_dir} \\
SWEBENCH_DIR={swebench_dir} \\
SWEBENCH_REPO={swebench_repo} \\
SWEBENCH_COMMIT={swebench_commit} \\
    {script_fpath}"""
            self._run_setup_command(command)

            return setup_dir

    def get_run_command(self) -> ExecuteContainerCommandArgs:
        swebench_cmd = (
            # Use pre-built SWE-bench
            "cd /swebench_setup/SWE-bench && "
            # Set UV environment variables to use the mounted portable directories
            f'export UV_INSTALL_DIR="{self.config.swebench_setup_dir}/uv" && '
            f'export UV_PYTHON_INSTALL_DIR="{self.config.swebench_setup_dir}/python" && '
            f'export PATH="{self.config.swebench_setup_dir}/uv/bin:$PATH" && '
            f"ls -lrt /root/dataset && "
            # Run with clean environment to avoid venv contamination
            # Use the pre-built venv directly with its absolute path
            f"env -u VIRTUAL_ENV {self.config.swebench_setup_dir}/SWE-bench/venv/bin/python -m swebench.harness.run_local_evaluation "
            f"    --predictions_path {self.config.prediction_mounted_path} "
            f"    --instance_ids {self.config.instance_id} "
            f"    --timeout {self.config.swebench_tests_timeout} "
            f"    --dataset_name /root/dataset/data.jsonl "
            f"    --split {self.config.problem_info['split']} "
            f"    --run_id {self.config.agent_run_id} && "
            f"cp -r logs/run_evaluation/{self.config.agent_run_id} /trajectories_mount/ && "
            f"rm -rf logs/run_evaluation/{self.config.agent_run_id} && rm -rf *{self.config.agent_run_id}*"
        )

        # Execute SWE-bench evaluation command
        search_path = os.path.join(
            self.config.persistent_dir,
            self.config.agent_run_id,
            "**",
            f"{self.config.instance_id}/report.json",
        )

        return ExecuteContainerCommandArgs(
            command=swebench_cmd,
            expected_file_pattern=search_path,
            mode="eval",
            timeout=self.config.swebench_tests_timeout + 120,
        )


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
                    return setup_dir

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
    {script_fpath}"""
            self._run_setup_command(command)

            return setup_dir

    def get_run_command(self) -> ExecuteContainerCommandArgs:
        r2e_gym_cmd = (
            # Use mounted directory path for cd
            "cd /r2egym_setup/R2E-Gym && "
            # Set UV environment variables to use the mounted portable directories
            f'export UV_INSTALL_DIR="{self.config.r2e_gym_setup_dir}/uv" && '
            f'export UV_PYTHON_INSTALL_DIR="{self.config.r2e_gym_setup_dir}/python" && '
            f'export PATH="{self.config.r2e_gym_setup_dir}/uv/bin:$PATH" && '
            # Run with clean environment to avoid venv contamination
            # Use the pre-built venv directly with its absolute path
            f"env -u VIRTUAL_ENV {self.config.r2e_gym_setup_dir}/R2E-Gym/venv/bin/python src/r2egym/agenthub/run/run_local_evaluation.py "
            f"    --predictions_path {self.config.prediction_mounted_path} "
            f"    --instance_id {self.config.instance_id} "
            f"    --timeout {self.config.swebench_tests_timeout} "
            f"    --dataset /root/dataset/data.jsonl "
            f"    --output_dir /trajectories_mount/eval-outputs/{self.config.agent_run_id}"
        )

        search_path = os.path.join(
            self.config.persistent_dir,
            "eval-outputs",
            self.config.agent_run_id,
            "report.json",
        )

        return ExecuteContainerCommandArgs(
            command=r2e_gym_cmd,
            expected_file_pattern=search_path,
            mode="eval",
            timeout=self.config.swebench_tests_timeout + 120,
        )


class NVInternalDatasetProcessor(BaseDatasetHarnessProcessor):
    def get_run_command(self) -> ExecuteContainerCommandArgs:
        instance_dict = json.loads(self.config.problem_info["instance_dict"])
        base_dockerfile = instance_dict.get("base_dockerfile", "")
        instance_dockerfile = instance_dict.get("instance_dockerfile", "")

        env_lines = []
        for line in (base_dockerfile + "\n" + instance_dockerfile).split("\n"):
            line = line.strip()
            if line.startswith("ENV "):
                # Convert ENV KEY=VALUE or ENV KEY VALUE to export KEY="VALUE"
                export_line = line.replace("ENV ", "export ", 1)
                # Handle both Docker ENV formats:
                # 1. ENV KEY=VALUE (with equals)
                # 2. ENV KEY VALUE (space-separated)
                if "=" in export_line:
                    # Format: export KEY=VALUE -> normalize spaces around =
                    export_line = re.sub(r"\s*=\s*", "=", export_line)
                else:
                    # Format: export KEY VALUE -> convert to export KEY="VALUE"
                    parts = export_line.split(None, 2)  # Split into at most 3 parts
                    if len(parts) >= 3:  # export KEY VALUE
                        key = parts[1]
                        value = parts[2]
                        export_line = f'export {key}="{value}"'

                env_lines.append(export_line)

        env_exports = "\n".join(env_lines)

        # Get repo setup command
        repo_cmd = instance_dict.get("before_repo_set_cmd", "").strip()
        if repo_cmd:
            repo_cmd = repo_cmd.split("\n")[-1]

        # Get test files
        test_files_str = instance_dict.get("selected_test_files_to_run", "[]")
        if isinstance(test_files_str, str):
            test_files = ",".join(eval(test_files_str))
        else:
            test_files = ",".join(test_files_str)

        run_script = instance_dict["run_script.sh"]
        parsing_script = instance_dict["parsing_script.py"]
        run_script_path = self.config.persistent_dir / "run_script.sh"
        parsing_script_path = self.config.persistent_dir / "parsing_script.py"
        with open(run_script_path, "w") as f:
            f.write(run_script)
        with open(parsing_script_path, "w") as f:
            f.write(parsing_script)

        cmd = f"""#!/bin/bash
set -e

{env_exports}

# Apply patch
cd /app
git reset --hard {instance_dict.get("base_commit", "")}
git checkout {instance_dict.get("base_commit", "")}

# Apply patch with rejection to handle conflicts
git apply --ignore-space-change --ignore-whitespace --reject -v /root/patch.diff || true

# Setup repository
{repo_cmd}

# Run tests
bash /root/run_script.sh {test_files} > /root/stdout.log 2> /root/stderr.log || true

# Parse results
python /root/parsing_script.py /root/stdout.log /root/stderr.log /root/output.json

# Move outputs to the mounted directory
mkdir -p /trajectories_mount/eval_results
cp /root/output.json /trajectories_mount/eval_results/output.json
"""

        search_path = os.path.join(
            self.config.persistent_dir,
            "eval_results",
            "output.json",
        )

        return ExecuteContainerCommandArgs(
            command=cmd,
            expected_file_pattern=search_path,
            mode="eval",
            timeout=self.config.swebench_tests_timeout + 120,
        )

    def postprocess_after_run(self, report_file: Path) -> None:
        instance_dict = json.loads(self.config.problem_info["instance_dict"])

        fail_to_pass_str = instance_dict.get("fail_to_pass_select", instance_dict.get("fail_to_pass", "[]"))
        pass_to_pass_str = instance_dict.get("pass_to_pass_select", instance_dict.get("pass_to_pass", "[]"))

        if isinstance(fail_to_pass_str, str):
            f2p = set(json.loads(fail_to_pass_str))
        else:
            f2p = set(fail_to_pass_str)

        if isinstance(pass_to_pass_str, str):
            p2p = set(json.loads(pass_to_pass_str))
        else:
            p2p = set(pass_to_pass_str)

        with open(report_file, "r+") as f:
            test_results = json.loads(f.read())
            is_resolved = self.check_tests_passed(
                test_results,
                f2p,
                p2p,
            )
            report_dict = dict(
                resolved=is_resolved,
                patch_exists=True,
                patch_successfully_applied=is_resolved,
                metadata={
                    "test_results": test_results,
                    "f2p": list(f2p),
                    "p2p": list(p2p),
                },
            )
            f.seek(0)
            f.write(json.dumps({self.config.instance_id: report_dict}, indent=4))

    def check_tests_passed(
        self,
        test_results: dict[str, Any],
        f2p: set[str],
        p2p: set[str],
    ) -> bool:
        if not test_results:
            return False

        passed_tests = {test["name"] for test in test_results.get("tests", []) if test.get("status") == "PASSED"}
        required_tests = f2p.union(p2p)

        # Check if all required tests passed
        if len(passed_tests) == 0 or len(required_tests) == 0:
            return False

        return required_tests <= passed_tests


class OpenHandsHarnessProcessor(BaseDatasetHarnessProcessor):
    def setup(self) -> Path:
        setup_dir = self.parent_dir / "swe_openhands_setup"

        with self._setup_directory_lock(setup_dir, "OpenHands"):
            openhands_dir = setup_dir / "OpenHands"
            miniforge_dir = setup_dir / "miniforge3"

            if openhands_dir.exists() and Path(openhands_dir / ".venv" / "bin" / "python").exists():
                print(f"OpenHands already set up at {setup_dir}", flush=True)
                return setup_dir

            print(f"Setting up OpenHands environment at {setup_dir}...", flush=True)
            rmtree(setup_dir, ignore_errors=True)
            setup_dir.mkdir(parents=True, exist_ok=True)

            script_fpath = self.parent_dir / "setup_scripts/openhands.sh"
            command = f"""SETUP_DIR={setup_dir} \\
MINIFORGE_DIR={miniforge_dir} \\
OPENHANDS_DIR={openhands_dir} \\
AGENT_FRAMEWORK_REPO={self.config.agent_framework_repo} \\
AGENT_FRAMEWORK_COMMIT={self.config.agent_framework_commit} \\
    {script_fpath}"""
            self._run_setup_command(command)

            return setup_dir

    def get_run_command(self) -> ExecuteContainerCommandArgs:
        data_point = self.config.problem_info
        agent_run_id = self.config.agent_run_id

        agent_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/oh_config.toml")

        # Add parameters to config.toml
        # TODO(sugam): is there a better way to do this?
        with open(agent_config, "r") as f:
            config = tomlkit.parse(f.read())

        config["llm"]["model"] |= {
            "model": self.config.body.model,
            "base_url": "",  # May need to populate this
            "temperature": self.config.inference_params["temperature"],
            "top_p": self.config.inference_params["top_p"],
        }

        config_str = tomlkit.dumps(config)

        eval_dir_in_openhands = self.config.eval_dir_in_openhands
        local_dataset_path = "/root/dataset/data.jsonl"
        config_file_path = self.config.openhands_config_file_path

        assert self.config.openhands_setup_dir is not None, "OpenHands setup directory is not set"

        if self.config.debug:
            profiling_cmd = "export NG_PROFILING_DIR=/trajectories_mount/profiling && "
        else:
            profiling_cmd = ""

        if self.config.openhands_should_log:
            log_cmd = "export LOG_LEVEL=DEBUG && export LOG_TO_FILE=true && export NG_OPENHANDS_SHOULD_LOG=true && "
        else:
            log_cmd = (
                "export LOG_LEVEL=CRITICAL && "
                "export DEBUG=False && "
                "export DEBUG_LLM=False && "
                "export LOG_TO_FILE=False && "
                "export LOG_ALL_EVENTS=False && "
                "export DEBUG_RUNTIME=False && "
            )

        agent_main_cmd = (
            "if [ -d /workspace ]; then "
            "    echo 'Exiting because /workspace is mounted.' && "
            "    echo 'Please make sure /workspace is not mounted inside of Apptainer before running OpenHands.' && "
            "    echo 'This is because OpenHands DELETES EVERYTHING in the /workspace folder if it exists.' && "
            "    exit 1; "
            "fi && "
            # Add miniforge bin to PATH (for tmux, node, poetry, etc.)
            "mkdir -p /tmp/ && "
            "export PATH=/openhands_setup/miniforge3/bin:$PATH && "
            # Setup tmux socket (OpenHands requirement)
            "uid=$(id -ru 2>/dev/null || id -u) && "
            "export TMUX_TMPDIR=/tmp && "
            "export TMUX=/tmp/tmux-$uid/default && "
            "mkdir -p /tmp/tmux-$uid && "
            "chown $uid:$uid /tmp/tmux-$uid || true && "
            "chmod 700 /tmp/tmux-$uid && "
            "tmux -S /tmp/tmux-$uid/default start-server || true && "
            # Use pre-built OpenHands
            "cd /openhands_setup/OpenHands && "
            "export RUNTIME=local && "
            f"{log_cmd}"
            f"{profiling_cmd}"
            f"export NEMO_GYM_METRICS_FPATH={self.config.metrics_fpath} && "
            f"export NEMO_GYM_CONFIG_DICT={self.config.ng_global_config_dict_str} && "
            f"export NEMO_GYM_MODEL_SERVER_NAME={self.config.model_server_name} &&"
            "export VIRTUAL_ENV=/openhands_setup/OpenHands/.venv && "
            "export PATH=$PATH:/openhands_setup/OpenHands/.venv/bin && "
            # CRITICAL: Configure poetry to only use the OpenHands venv (ignore external venvs)
            "export POETRY_VIRTUALENVS_IN_PROJECT=true && "
            "export POETRY_VIRTUALENVS_CREATE=false && "
            "export POETRY_VIRTUALENVS_PATH=/openhands_setup/OpenHands && "
            f"export TMUX_MEMORY_LIMIT={self.config.apptainer_memory_limit_mb} && "
            f"export COMMAND_EXEC_TIMEOUT={self.config.command_exec_timeout} && "
            # TODO (sugam): fix cryptography issue
            # "override_dir=$(mktemp -d /tmp/cryptography_override.XXXX) && "
            # # Reinstall cryptography inside the container (via poetry's venv) using a compatible wheel
            # # Clean any broken installs to avoid missing-file errors, then force a wheel-only reinstall
            # "site_packages_dir=/openhands_setup/OpenHands/.venv/lib/python3.12/site-packages && "
            # 'if [ -d "$site_packages_dir" ]; then '
            # '    find "$site_packages_dir" -maxdepth 1 -name "cryptography*" -exec rm -rf {} +; '
            # "fi && "
            # "poetry run python -m pip install --index-url https://pypi.org/simple "
            # "    --trusted-host pypi.org --trusted-host files.pythonhosted.org "
            # "    --only-binary cryptography --no-deps --force-reinstall 'cryptography==42.0.8' && "
            # disable logging to file in the oh repo
            # set up config files
            f"echo {shlex.quote(config_str)} >{config_file_path} && "
            # f" export EVAL_OUTPUT_DIR={eval_dir_in_openhands} && "
            f"./evaluation/benchmarks/swe_bench/scripts/run_infer.sh "
            f"    llm.model "  # name of llm config section in config.toml
            f"    {self.config.agent_framework_commit} "  # openhands commit
            f"    CodeActAgent "  # agent
            f"    0 "  # Note: this is eval limit which randomly chooses an instance from the dataset
            f"    {self.config.agent_max_turns} "  # max agent iterations
            f"    1 "  # number of workers
            f"    {data_point['dataset_name']} "  # dataset name
            f"    {data_point['split']} "  # dataset split
            f"    {eval_dir_in_openhands} "
            f"    {data_point['instance_id']} "
            f"    {local_dataset_path} "
            f"    {config_file_path}"
        )

        agent_script_name = f"agent_script_{agent_run_id}.sh"
        agent_script_path = self.config.persistent_dir / agent_script_name
        with open(agent_script_path, "w") as f:
            f.write("#!/bin/bash\nset -e\n")
            f.write(agent_main_cmd)
            f.flush()
            os.fsync(f.fileno())

        agent_timeout_seconds = self.config.swebench_agent_timeout
        openhands_cmd = (
            f"timeout --signal=TERM --kill-after=30 {agent_timeout_seconds} "
            f"bash /trajectories_mount/{agent_script_name}"
        )

        search_path = os.path.join(
            self.config.openhands_setup_dir / "OpenHands" / eval_dir_in_openhands,
            "**",
            "output.jsonl",
        )

        # Execute OpenHands command
        return ExecuteContainerCommandArgs(
            command=openhands_cmd,
            expected_file_pattern=search_path,
            mode="agent",
            timeout=self.config.swebench_agent_timeout + 60,
        )


########################################
# START Ray worker logic
########################################


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
    },
    num_cpus=1,
)
def runner_ray_remote(params_dict: dict[str, Any]) -> Path:
    # For some reason Ray may not pick up the proper model fields if we don't rebuild the model here. Very strange.
    SWEBenchWrapperInstanceConfig.model_rebuild(force=True)
    RunOpenHandsAgent.model_rebuild(force=True)

    params = SWEBenchWrapperInstanceConfig.model_validate(params_dict)
    instance_id = params.instance_id

    if params.debug:
        profiler = Profiler(name=instance_id, base_profile_dir=params.persistent_dir / "profiling")
        profiler.start()

    run_oh = RunOpenHandsAgent(config=params)

    report_file = asyncio.run(run_oh.process_single_datapoint())

    if params.debug:
        profiler.stop()

    return report_file


class RunOpenHandsAgent(BaseModel):
    config: SWEBenchWrapperInstanceConfig

    def _openhands_dir_copy_from_host(self, output_file_path: Optional[str]) -> Optional[str]:
        data_point = self.config.problem_info
        eval_dir_in_openhands = self.config.eval_dir_in_openhands
        config_file_path = self.config.openhands_config_file_path

        eval_dir_on_host = Path(self.config.openhands_setup_dir) / "OpenHands" / eval_dir_in_openhands
        trajectories_root = self.config.trajectories_root
        llm_completions_dir = trajectories_root / "llm_completions" / data_point["instance_id"]
        trajectories_root.mkdir(parents=True, exist_ok=True)
        llm_completions_dir.mkdir(parents=True, exist_ok=True)

        dest_output: Optional[str] = None
        if output_file_path:
            source_output = Path(output_file_path)
            if not source_output.is_absolute():
                source_output = eval_dir_on_host / source_output
            if not source_output.exists():
                output_candidates = sorted(eval_dir_on_host.glob("*/*/*/output.jsonl"), key=os.path.getmtime)
                if not output_candidates:
                    raise FileNotFoundError(
                        f"No output.jsonl found under {eval_dir_on_host} for {data_point['instance_id']}."
                    )
                source_output = output_candidates[-1]

            dest_output_path = self.config.prediction_path
            shutil.copy2(source_output, dest_output_path)
            dest_output = str(dest_output_path)

        completion_candidates = glob.glob(str(eval_dir_on_host / "*/*/*/llm_completions/*/*.json"))
        if completion_candidates:
            latest_completion = max(completion_candidates, key=os.path.getmtime)
            shutil.copy2(
                latest_completion,
                llm_completions_dir / Path(latest_completion).name,
            )

        shutil.rmtree(eval_dir_on_host, ignore_errors=True)
        try:
            Path(config_file_path).unlink()
        except OSError:
            pass

        return dest_output

    async def _execute_container_command(self, command: ExecuteContainerCommandArgs, apptainer_cmd: str):
        data_point = self.config.problem_info

        # Stream output to log file as it appears
        logs_dir = self.config.persistent_dir / "apptainer_logs"
        logs_dir.mkdir(exist_ok=True)
        log_file_path = logs_dir / f"{self.config.instance_id}_{command.mode}.log"
        log_file = open(log_file_path, "w")

        process = await asyncio.create_subprocess_shell(apptainer_cmd, stdout=log_file, stderr=log_file)

        try:
            # Wait for completion with timeout
            await asyncio.wait_for(process.communicate(), timeout=command.timeout)
        except asyncio.TimeoutError:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise ValueError("Command timed out")
        finally:
            log_file.close()

        assert process.returncode == 0, f"Command failed with return code {process.returncode}"

        # Look for the expected file
        pred_files = glob.glob(command.expected_file_pattern, recursive=True)

        if len(pred_files) == 1:
            return pred_files[0]
        elif len(pred_files) > 1:
            latest_file = max(pred_files, key=os.path.getmtime)
            print(
                f"Multiple outputs found for {data_point['instance_id']} "
                f"({len(pred_files)}). Using latest: {latest_file}",
                flush=True,
            )
            return latest_file
        else:
            raise ValueError(
                f"Expected exactly one file matching {command.expected_file_pattern} for {data_point['instance_id']}, "
                f"found {len(pred_files)}."
            )

    async def process_single_datapoint(self) -> Path:
        out_file_in_eval = await self._execute_container_command(
            self.config.agent_command, self.config.agent_apptainer_command_str
        )
        out_file = self._openhands_dir_copy_from_host(output_file_path=out_file_in_eval)

        with open(out_file, "r") as f:
            out_dict = json.loads(f.read().strip())

        patch = out_dict["test_result"]["git_patch"]
        if not patch:
            patch = None

        # Create file in the SWE-bench evaluation format
        pred_file = out_file.replace("output.jsonl", "output_for_eval.jsonl")
        with open(pred_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "model_name_or_path": out_dict["metadata"]["llm_config"]["model"],
                        "instance_id": out_dict["instance_id"],
                        "model_patch": patch + "\n" if patch and not patch.endswith("\n") else patch,
                        "oh_time_metrics": out_dict["metrics"],
                    }
                )
            )

        # Dump out dot and png files from profiling on OpenHands level
        if self.config.debug:
            base_profile_dir = self.config.persistent_dir / "profiling"
            profiling_name = "openhands"
            callgrind_path = base_profile_dir / f"{profiling_name}.callgrind"
            callgrind_dotfile_path = base_profile_dir / f"{profiling_name}.dot"
            callgrind_graph_path = base_profile_dir / f"{profiling_name}.png"

            gprof2dot_main(
                argv=f"--format=callgrind --output={callgrind_dotfile_path} -e 5 -n 5 {callgrind_path}".split()
            )

            (graph,) = graph_from_dot_file(callgrind_dotfile_path)
            graph.write_png(callgrind_graph_path)

        if pred_file is None:
            return

        with open(pred_file, "r") as f:
            trajectory_dict = json.loads(f.read())

        # Check if the trajectory has an empty patch before running evaluation
        has_patch = trajectory_dict["model_patch"] is not None
        if not has_patch:
            return

        with open(self.config.model_patch_path, "w") as f:
            model_patch = trajectory_dict["model_patch"]

            # Add a newline to the end of the patch if it doesn't have one
            model_patch = model_patch + "\n" if not model_patch.endswith("\n") else model_patch
            f.write(model_patch)

        report_file = await self._execute_container_command(
            self.config.eval_command, self.config.eval_apptainer_command_str
        )
        return report_file


########################################
# START Server logic
########################################


class SWEBenchWrapper(SimpleResponsesAPIAgent):
    config: SWEBenchWrapperConfig

    _sem: Optional[Semaphore] = None
    _vllm_converter: Optional[VLLMConverter] = None
    _swe_bench_wrapper_server_config: Optional[SWEBenchWrapperServerConfig] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ########################################
    # START Init
    ########################################

    def model_post_init(self, context: Any) -> None:
        run_session_id = f"{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
        workspace_root = Path(__file__).parent
        self._swe_bench_wrapper_server_config = SWEBenchWrapperServerConfig(
            run_session_id=run_session_id,
            base_results_dir=workspace_root / f"swebench_results_{run_session_id}",
            ng_global_config_dict_str=shlex.quote(OmegaConf.to_yaml(get_global_config_dict())),
            model_server_name=self.config.model_server.name,
            openhands_setup_dir=OpenHandsHarnessProcessor(config=self.config).setup(),
            swebench_setup_dir=SweBenchDatasetProcessor(config=self.config).setup(),
            r2e_gym_setup_dir=R2EGymDatasetProcessor(config=self.config).setup(),
        )

        self._sem = Semaphore(self.config.concurrency)
        self._vllm_converter = VLLMConverter(return_token_id_information=True)

        return super().model_post_init(context)

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

    def _find_container(self, data_point: dict) -> str:
        """Find the container file using multiple strategies (Exact match > Fuzzy match).

        Strategies:
        1. Replace "__" with "_1776_" (Original case, then Lowercase)
        2. Replace "__" with "_s_" (Original case, then Lowercase)
        3. Fuzzy search directory for .sif files matching above patterns.

        Returns:
            str: Path to the container file.

        Raises:
            FileNotFoundError: If no matching container file is found.
        """
        instance_id = data_point["instance_id"]
        container_formatters = data_point["container_formatter"]

        if isinstance(container_formatters, str):
            container_formatters = [container_formatters]

        if "R2E-Gym" in data_point["dataset_name"]:
            instance_id_modified = re.sub(
                r"[^_]+__([^-]+)-", lambda m: m.group(1).lower() + "_final_", data_point["instance_id"]
            )
            for container_formatter in container_formatters:
                container_name = container_formatter.format(instance_id=instance_id_modified)
                if os.path.exists(container_name):
                    # print(f"container found: {container_name}", flush=True)
                    # print(f"container formatter: {container_formatter}", flush=True)
                    return container_name

        replacements = ["_1776_", "_s_"]

        # Generate all candidate IDs in order of priority
        candidate_ids = [instance_id]
        for replacement in replacements:
            replaced_id = instance_id.replace("__", replacement)
            candidate_ids.append(replaced_id)
            candidate_ids.append(replaced_id.lower())

        # Phase 1: Exact Matches - try all container formatters
        for container_formatter in container_formatters:
            for candidate_id in candidate_ids:
                path = container_formatter.format(instance_id=candidate_id)
                if os.path.exists(path):
                    return path

        # Phase 2: Fuzzy Search - try all container formatters
        search_terms = [instance_id, instance_id.lower()] + candidate_ids

        for container_formatter in container_formatters:
            # Define the default fallback path (Strategy 1, original case)
            fallback_path = container_formatter.format(instance_id=instance_id.replace("__", replacements[0]))
            container_dir = os.path.dirname(fallback_path)

            if os.path.exists(container_dir):
                for term in search_terms:
                    pattern = os.path.join(container_dir, f"*{term}*.sif")
                    matches = glob.glob(pattern)
                    if matches:
                        return matches[0]
            else:
                print(f"Container directory {container_dir} does not exist", flush=True)

        # Phase 3: Fallback
        tried_paths = []
        for container_formatter in container_formatters:
            for candidate_id in candidate_ids:
                tried_paths.append(container_formatter.format(instance_id=candidate_id))

        raise FileNotFoundError(
            f"No container file found for instance_id {instance_id}. "
            f"Tried the following candidate IDs: {candidate_ids}. "
            f"Searched in paths: {tried_paths}."
        )

    def _build_apptainer_command(
        self, params: SWEBenchWrapperInstanceConfig, command: ExecuteContainerCommandArgs
    ) -> str:
        dataset_path_to_mount = str(params.instance_dataset_path)
        data_point = params.problem_info

        # Fix localhost URLs not working sometimes
        container_commands = []
        container_commands.append("echo '127.0.0.1 localhost' >/etc/hosts")

        # Build mount arguments
        mount_args = [
            f"--mount type=bind,src={params.persistent_dir},dst=/trajectories_mount",
        ]

        # Add OpenHands setup directory mount if available (for OpenHands)
        # Mount the entire setup directory at both /openhands_setup and its original absolute path
        # This is needed because poetry and other tools have hardcoded absolute paths
        mount_args.append(f"--mount type=bind,src={params.openhands_setup_dir},dst=/openhands_setup,ro")
        mount_args.append(f"--mount type=bind,src={params.openhands_setup_dir},dst={params.openhands_setup_dir},ro")
        # Mount only the venv and miniforge as read-only to prevent mutation while keeping the rest writable
        venv_path = Path(params.openhands_setup_dir) / "OpenHands/.venv"
        mount_args.append(f"--mount type=bind,src={venv_path},dst=/openhands_setup/OpenHands/.venv,ro")
        mount_args.append(f"--mount type=bind,src={venv_path},dst={venv_path},ro")

        mount_args.extend(
            [
                # make everything in OpenHands read-only
                f"--mount type=bind,src={params.openhands_setup_dir}/OpenHands,dst=/openhands_setup/OpenHands,ro",
                f"--mount type=bind,src={params.openhands_setup_dir}/OpenHands/.eval_sessions,dst=/openhands_setup/OpenHands/.eval_sessions",
                f"--mount type=bind,src={params.openhands_setup_dir}/OpenHands/.eval_sessions,dst={params.openhands_setup_dir}/OpenHands/.eval_sessions",
                f"--mount type=bind,src={params.openhands_setup_dir}/OpenHands/logs,dst=/openhands_setup/OpenHands/logs",
                f"--mount type=bind,src={params.openhands_setup_dir}/OpenHands/logs,dst={params.openhands_setup_dir}/OpenHands/logs",
                f"--mount type=bind,src={params.openhands_setup_dir}/OpenHands/evaluation/oh,dst=/openhands_setup/OpenHands/evaluation/oh",
                f"--mount type=bind,src={params.openhands_setup_dir}/OpenHands/evaluation/oh,dst={params.openhands_setup_dir}/OpenHands/evaluation/oh",
                # Data
                f"--mount type=bind,src={dataset_path_to_mount},dst=/root/dataset/data.jsonl",
            ]
        )

        miniforge3_path = Path(params.openhands_setup_dir) / "miniforge3"
        mount_args.append(f"--mount type=bind,src={miniforge3_path},dst=/openhands_setup/miniforge3,ro")
        mount_args.append(f"--mount type=bind,src={miniforge3_path},dst={miniforge3_path},ro")

        # Add SWE-bench setup directory mount if available (for evaluation)
        if command.mode == "eval" and data_point["dataset_name"] != "nv-internal-1":
            # Mount the entire setup directory at both /swebench_setup and its original absolute path
            # This is needed because uv venv has hardcoded absolute paths
            # print(
            #     f"Mounting pre-built SWE-bench from: {self.swebench_setup_dir}",
            #     flush=True,
            # )
            mount_args.append(f"--mount type=bind,src={params.swebench_setup_dir},dst=/swebench_setup")
            mount_args.append(f"--mount type=bind,src={params.swebench_setup_dir},dst={params.swebench_setup_dir}")
            mount_args.append(f"--mount type=bind,src={dataset_path_to_mount},dst=/root/dataset/data.jsonl")

        if command.mode == "eval" and data_point["dataset_name"] == "nv-internal-1":
            run_script_path = params.persistent_dir / "run_script.sh"
            parsing_script_path = params.persistent_dir / "parsing_script.py"
            model_patch_path = params.persistent_dir / "patch.diff"

            mount_args.append(f"--mount type=bind,src={run_script_path},dst=/root/run_script.sh")
            mount_args.append(f"--mount type=bind,src={parsing_script_path},dst=/root/parsing_script.py")
            mount_args.append(f"--mount type=bind,src={model_patch_path},dst=/root/patch.diff")

        if command.mode == "eval" and "R2E-Gym" in data_point["dataset_name"]:
            # Mount the entire setup directory at both /r2egym_setup and its original absolute path
            # This is needed because uv venv has hardcoded absolute paths in its wrappers
            # print(f"Mounting R2E-Gym setup directory from: {self.r2e_gym_setup_dir}", flush=True)
            mount_args.append(f"--mount type=bind,src={params.r2e_gym_setup_dir},dst=/r2egym_setup")
            mount_args.append(f"--mount type=bind,src={params.r2e_gym_setup_dir},dst={params.r2e_gym_setup_dir}")
            mount_args.append(f"--mount type=bind,src={dataset_path_to_mount},dst=/root/dataset/data.jsonl")

        if command.mode == "agent" and "R2E-Gym" in data_point["dataset_name"]:
            # Remove R2E-Gym test-related files.
            for root_dir in ["", "/root", "/testbed"]:
                container_commands.append(
                    # /r2e_tests contains evaluation tests that the agent should not see.
                    f"rm -rf {root_dir}/r2e_tests && "
                    # run_tests.sh launches the tests in /r2e_tests, so the agent should not see this either.
                    # We check that it contains the substring "r2e_tests"
                    # to avoid accidentally deleting an unrelated file with that name.
                    f"if grep -qs r2e_tests {root_dir}/run_tests.sh; then rm -rf {root_dir}/run_tests.sh; fi"
                )
        container_commands.append(command.command)
        combined_command = " && ".join(container_commands)

        mount_str = " ".join(mount_args)

        # Launch Apptainer container and execute the command
        apptainer_cmd = (
            f"apptainer exec --writable-tmpfs --cleanenv --pid --no-mount home,tmp,bind-paths "
            f"{mount_str} "
            f" {params.container} bash -c {shlex.quote(combined_command)}"
        )
        memory_limit_mb = params.apptainer_memory_limit_mb
        if memory_limit_mb is not None and memory_limit_mb > 0:
            memory_limit_kb = int(memory_limit_mb) * 1024
            apptainer_cmd = f"ulimit -v {memory_limit_kb} && {apptainer_cmd}"

        return apptainer_cmd

    def _setup_params(
        self, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> Tuple[SWEBenchWrapperInstanceConfig, BaseDatasetHarnessProcessor]:
        problem_info = body.metadata | {"container_formatter": self.config.container_formatter}
        instance_id = problem_info.get("instance_id", "unknown")

        # Create persistent directory for I/O and logs in local workspace
        instance_dir = f"{instance_id}_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
        persistent_dir = self._swe_bench_wrapper_server_config.base_results_dir / instance_dir
        persistent_dir.mkdir(parents=True, exist_ok=True)

        agent_run_id = f"{instance_id}_{int(time.time())}_{str(uuid.uuid4())[:8]}"

        # To avoid making HF dataset API calls, we write the instance dictionary to a file and mount it in the container.
        instance_dataset_dir = persistent_dir / "instance_datasets"
        instance_dataset_dir.mkdir(parents=True, exist_ok=True)
        instance_dataset_path = instance_dataset_dir / f"{agent_run_id}.jsonl"
        instance_dict = json.loads(problem_info["instance_dict"])
        if "repo" in instance_dict and "repo_name" not in instance_dict:
            instance_dict["repo_name"] = instance_dict["repo"]
        with open(instance_dataset_path, "w") as f:
            f.write(json.dumps(instance_dict) + "\n")

        trajectories_root = persistent_dir / "trajectories" / instance_id
        prediction_mounted_path = Path("/trajectories_mount") / "trajectories" / instance_id
        prediction_path = trajectories_root / "output.jsonl"

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

        container = self._find_container(problem_info)

        eval_dir_in_openhands = f"evaluation/oh/{agent_run_id}"
        openhands_config_file_path = f"/tmp/config_{agent_run_id}.toml"

        agent_script_name = f"agent_script_{agent_run_id}.sh"
        agent_script_path = persistent_dir / agent_script_name

        params: SWEBenchWrapperInstanceConfig = SWEBenchWrapperInstanceConfig(
            **self.config.model_dump(),
            **self._swe_bench_wrapper_server_config.model_dump(),
            problem_info=problem_info,
            body=body,
            persistent_dir=persistent_dir,
            metrics_fpath=persistent_dir / "nemo_gym_metrics.json",
            ray_queue_timestamp=time.time(),
            inference_params=inference_params,
            agent_run_id=agent_run_id,
            instance_dataset_path=instance_dataset_path,
            trajectories_root=trajectories_root,
            prediction_path=prediction_path,
            prediction_mounted_path=prediction_mounted_path,
            model_patch_path=persistent_dir / "patch.diff",
            container=container,
            eval_dir_in_openhands=eval_dir_in_openhands,
            openhands_config_file_path=openhands_config_file_path,
            agent_script_path=agent_script_path,
        )

        if params.problem_info["dataset_name"] == "nv-internal-1":
            dataset_processor = NVInternalDatasetProcessor(config=params)
        elif "R2E-Gym" in params.problem_info["dataset_name"]:
            dataset_processor = R2EGymDatasetProcessor(config=params)
        else:
            dataset_processor = SweBenchDatasetProcessor(config=params)

        params.eval_command = dataset_processor.get_run_command()
        params.eval_apptainer_command_str = self._build_apptainer_command(params, params.eval_command)

        params.agent_command = OpenHandsHarnessProcessor(config=params).get_run_command()
        params.agent_apptainer_command_str = self._build_apptainer_command(params, params.agent_command)
        params.agent_script = params.agent_script_path.read_text()

        return params, dataset_processor

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        params, dataset_processor = self._setup_params(body)

        # TODO remove
        with open("temp.json", "w") as f:
            f.write(params.model_dump_json(indent=4))

        report_file = await runner_ray_remote.remote(params.model_dump())

        dataset_processor.postprocess_after_eval_run(report_file)

        trajectories_dir = params.persistent_dir / "trajectories"
        chat_completions_trajectory, chat_completions_tools = self.get_openhands_trajectory_from_completions(
            trajectories_dir, params.instance_id
        )

        tools = [
            FunctionTool.model_validate(tool["function"] | {"type": "function"}) for tool in chat_completions_tools
        ]
        responses_items = self._vllm_converter.chat_completions_messages_to_responses_items(
            chat_completions_trajectory
        )
        input_items, output_items = split_responses_input_output_items(responses_items)

        return NeMoGymResponse(
            id=f"swebench-{params.instance_id}",
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
                "instance_config": params.model_dump_json(),
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
                instance_config=SWEBenchWrapperInstanceConfig.model_validate_json(
                    metadata["instance_config"]
                ).model_dump(),
            )


if __name__ == "__main__":
    SWEBenchWrapper.run_webserver()
