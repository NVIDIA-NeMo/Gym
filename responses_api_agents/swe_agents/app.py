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
import importlib.util
import json
import logging
import os
import random
import re
import shlex
import shutil
import sys
import threading
import time
import uuid
from asyncio import Semaphore
from asyncio.subprocess import Process
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree
from subprocess import Popen
from subprocess import run as subprocess_run
from traceback import format_exc
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import ray
import tomlkit
from gprof2dot import main as gprof2dot_main
from openai.types.responses.function_tool import FunctionTool
from pydantic import BaseModel, ConfigDict, Field
from pydot import graph_from_dot_file

from nemo_gym import PARENT_DIR
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
from nemo_gym.vm_launcher import ensure_vm, run_in_vm
from responses_api_models.vllm_model.app import VLLMConverter, split_responses_input_output_items


logger = logging.getLogger(__name__)


########################################
# START Configuration
########################################


class AgentPromptOverride(BaseModel):
    user_prompt_template: Optional[str] = Field(
        default=None,
        description="Path to the user prompt template file",
    )
    system_prompt_template: Optional[str] = Field(
        default=None,
        description="Path to the system prompt template file",
    )
    agent_cls: Literal["CodeActAgent", "OpenCodeAgent", "CodexAgent", "Terminus2Agent"] = Field(
        default="CodeActAgent",
        description="Class to use for the agent",
    )
    diversify_tool_names: Optional[bool] = Field(
        default=False,
        description="If True, randomly select from tool names each run. If False, use the tool names in the order they are defined.",
    )
    camel_case_tool_names: Optional[bool] = Field(
        default=False,
        description="If True, convert tool names to camel case. If False, use the tool names as is.",
    )


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
    # [CUSTOM-CONTAINER-FORMATTER-DICT 2026-10-21] container_formatter is used in _find_container
    # to match a row to the right container. Various possibilities depending on how
    # container_formatter is passed in:
    # - str -> applied across rows, with a default fallback if no matching container is found.
    # - list[str] -> similar to str, but the first matching string is used. Same default fallback.
    # - dict[stt, str] -> uses the `dataset_name` attribute of each row to map to a specific
    #                     formatter string. In this case there is no default fallback: if a row
    #                     cannot be assigned to an existing container, an error is raised.
    container_formatter: str | list[str] | dict[str, str] = Field(
        default="docker://swebench/sweb.eval.x86_64.{instance_id}", description="Container path template"
    )

    # [CUSTOM-VM-SIDECAR 2026-10-27] SWE-Gym/R2E-Gym/SWE-smith/SWE-rebench-V1/SWE-rebench-V2
    # images are built with root-owned content. To circumvent this, run them from inside a VM,
    # where we are root. 
    # In detail,the  apptainer command is run inside a per-node QEMU/KVM VM sidecar (see
    # nemo_gym/vm_launcher.py). We only create one persistent VM per node, shared across every
    # rollout happening in that node -- see vm_launcher.py's module docstring for the full 
    # lifecycle description of the VM.
    vm_golden_image_dir: Optional[str] = Field(
        default=None,
        description=(
            "[CUSTOM-VM-SIDECAR] Directory holding golden.qcow2 and the unprivileged QEMU "
            "install. Required when a job's data includes any of "
            "SWE-Gym/R2E-Gym/SWE-smith/SWE-rebench-V1/SWE-rebench-V2 rows "
            "-- see _uses_vm_sidecar()."
        ),
    )
    vm_lustre_share_roots: list[str] = Field(
        default_factory=list,
        description=(
            "[CUSTOM-VM-SIDECAR] Host directories shared into the VM via virtio-9p at boot, at "
            "the same absolute path, so the existing _build_apptainer_command mount strings "
            "need no changes. Must cover every host path that command binds for the affected "
            "dataset types: the container_formatter image directory/directories, "
            "base_results_dir's root (persistent_dir lives under it), and the "
            "openhands/swebench/r2e_gym/swe_rebench/swesmith setup dirs' root(s)."
        ),
    )

    swebench_tests_timeout: int = Field(default=30 * 60, description="Timeout for running tests (seconds)")

    swebench_agent_timeout: int = Field(default=45 * 60, description="Timeout for running the agent (seconds)")

    apptainer_memory_limit_mb: int = Field(
        default=32 * 1024, description="Memory limit for the apptainer container (MB)"
    )

    apptainer_binary: str = Field(
        default="apptainer",
        description=(
            "Binary name/path used to launch the agent/eval containers (default 'apptainer', "
            "matches prior unconditional behavior). [CUSTOM] Set to 'singularity' on clusters "
            "that ship SingularityPRO/Singularity but have no 'apptainer' binary on PATH -- "
            "Apptainer and Singularity are CLI-compatible forks, so 'exec'/mount-flag syntax "
            "is unaffected by which binary name is used."
        ),
    )

    command_exec_timeout: int = Field(default=5 * 60, description="Timeout for executing the command (seconds)")

    # [CUSTOM-CONTEXT-OVERFLOW-PREFLIGHT 2026-10-11] generation context cap (ideally matches
    # policy.max_total_sequence_length / policy.generation.vllm_cfg.max_model_len
    # in the top-level NeMo-RL training YAML). NeMo-Gym's own config system does NOT carry
    # this value. Threading it through explicitly here lets nemo_gym_client.py do a pre-flight
    # token-budget check before ever sending a request to vLLM, instead of only discovering
    # overflow via vLLM's own 400 rejection. Optional/defaults to None (no pre-flight check).
    # don't set it yet.
    max_model_len: Optional[int] = Field(
        default=None,
        description=(
            "Real generation context cap (chars/tokens), matching policy.max_total_sequence_length. "
            "When set, enables a pre-flight token-budget check in OpenHands' NemoGymClient before "
            "sending each completion request, to fail a rollout cleanly. Dangerous since a failed"
            "rollout gets a 0 reward via a synthetic `finish` call and is hard to detect."
        ),
    )

    # [CUSTOM-CONTEXT-OVERFLOW-PREFLIGHT 2026-10-12] Fixed token headroom reserved below
    # max_model_len.
    preflight_headroom_tokens: int = Field(
        default=16384,
        description=(
            "Token headroom reserved below max_model_len for the context-overflow pre-flight "
            "check: a request is rejected before sending if current_tokens + this value would "
            "exceed max_model_len. Covers both tokenizer-approximation error and minimum "
            "generation room. Only used when max_model_len is set."
        ),
    )

    # Concurrency control
    concurrency: int = Field(default=256, description="Maximum number of concurrent SWE-bench runs")

    dataset_path: Optional[str] = Field(
        default=None,
        description="Path to the dataset for SWE-bench evaluation",
    )

    verify_golden_patch: bool = Field(
        default=False,
        description=(
            "If True, skip the agent run and use the sample's golden patch "
            "(instance_dict['patch']) as the model patch. The eval container "
            "still runs, so this verifies that the dataset sample actually "
            "resolves when its golden patch is applied. Currently supported "
            "for dataset_name == 'swe-bench-ext'."
        ),
    )

    agent_prompt_overrides: Optional[list[AgentPromptOverride]] = Field(
        default=None,
        description="List of (user_prompt_template, system_prompt_template, agent_cls) overrides. "
        "If multiple are provided, one is selected per instance_id (deterministic or random based on "
        "agent_prompt_override_random).",
    )
    agent_prompt_override_random: bool = Field(
        default=False,
        description="If True, randomly select from agent_prompt_overrides each run. "
        "If False (default), selection is deterministic per instance_id.",
    )

    openhands_should_log: bool = False
    debug: bool = False


class SWEBenchWrapperServerConfig(BaseModel):
    ng_global_config_dict_str: str
    model_server_name: str
    openhands_setup_dir: Path
    swebench_setup_dir: Path
    r2e_gym_setup_dir: Path
    swe_rebench_setup_dir: Path
    swebench_multilingual_setup_dir: Path
    swesmith_setup_dir: Path
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
    output_for_eval_mounted_path: Path
    output_for_eval_path: Path
    model_patch_path: Path
    container: str
    # The agent routinely creates untracked scaffold files during its own OpenHands session
    # (e.g. `install.sh`, `test_sleep.py`) that would otherwise persist into a shared mount.
    # When the model's own patch.diff later tries to recreate that same file as "new",
    # `git apply` refuses with "already exists in working directory" and the WHOLE patch
    # fails to apply, not just that one file's.
    # For this reason, use two independent fields, so agent and eval run against genuinely
    # separate mounts.
    copy_out_mount_args: List[str] = Field(default_factory=list)
    eval_copy_out_mount_args: List[str] = Field(default_factory=list)
    eval_dir_in_openhands: str
    openhands_config_file_path: str
    agent_script_path: Path
    final_eval_apptainer_spinup_timestamp_fpath: Path
    final_eval_apptainer_spinup_timestamp_mounted_fpath: Path
    generation_apptainer_spinup_timestamp_fpath: Path
    generation_apptainer_spinup_timestamp_mounted_fpath: Path
    base_mounted_dir: Path
    profiling_dir: Path
    profiling_mounted_dir: Path

    # Resolved prompt override fields (selected from agent_prompt_overrides based on instance_id)
    resolved_user_prompt_template: Optional[str] = None
    resolved_system_prompt_template: Optional[str] = None
    resolved_agent_cls: str = "CodeActAgent"
    resolved_diversify_tool_names: Optional[bool] = False
    resolved_camel_case_tool_names: Optional[bool] = False

    # Set later
    eval_command: Optional[ExecuteContainerCommandArgs] = None
    eval_apptainer_command_str: Optional[str] = None
    agent_command: Optional[ExecuteContainerCommandArgs] = None
    agent_apptainer_command_str: Optional[str] = None
    agent_script: Optional[str] = None

    # GRPO related fields
    mask_sample: bool = False

    @property
    def instance_id(self) -> str:
        return self.problem_info["instance_id"]


class SWEBenchMetrics(BaseModel):
    resolved: Optional[bool] = None
    patch_exists: Optional[bool] = None
    model_patch: Optional[str] = None

    # Failure-mode signals used to decide mask_sample downstream.
    # agent_error_kind is one of: "max_iteration", "context_window",
    # "stuck_in_loop", "other", or None if the agent finished cleanly.
    agent_error_kind: Optional[str] = None
    agent_timed_out: Optional[bool] = None
    eval_timed_out: Optional[bool] = None

    # Profiling time metrics to report
    ray_queue_time: Optional[float] = None
    openhands_run_time: Optional[float] = None
    generation_apptainer_spinup_time: Optional[float] = None
    create_runtime_time: Optional[float] = None
    connect_to_runtime_time: Optional[float] = None
    initialize_runtime_time: Optional[float] = None
    total_command_exec_time: Optional[float] = None
    total_model_call_time: Optional[float] = None
    final_eval_apptainer_spinup_time: Optional[float] = None
    final_eval_time: Optional[float] = None


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

    @staticmethod
    def _heartbeat_loop(heartbeat_path: Path, stop_event: threading.Event, interval: float) -> None:
        while not stop_event.wait(interval):
            try:
                heartbeat_path.touch()
            except OSError:
                pass

    @contextmanager
    def _setup_directory_lock(self, setup_dir: Path, label: str):
        """Cross-node lock using mkdir (atomic on Lustre/NFS, unlike fcntl.flock).

        [CUSTOM-SETUP-LOCK-HEARTBEAT 2026-11-03] Staleness used to be judged purely by the lock
        dir's own mtime -- but nothing ever touches that after the initial mkdir, so it could only
        ever measure "time since acquisition," never "is the holder still alive".
        Fix: the holder now actively proves it's alive via a periodically-touched heartbeat file
        inside the lock dir. A waiter judges the heartbeat's own freshness (dead in ~3 missed beats)
        instead of the lock dir's age, so a genuinely slow build -- however long it takes -- is never
        killed early, while a dead holder is caught in ~90s instead of up to an hour. `max_wait`
        stays as a hard backstop for the separate (rare, inherent to any heartbeat scheme) case of a
        holder that's alive but truly hung rather than dead.
        """
        lock_dir = setup_dir.parent
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f".{setup_dir.name}.lockdir"
        heartbeat_path = lock_path / "heartbeat"
        heartbeat_interval = 30
        heartbeat_stale_after = 3 * heartbeat_interval

        print(f"Acquiring {label} setup lock at {lock_path}", flush=True)
        max_wait = 3600
        poll_interval = 5
        waited = 0
        while True:
            try:
                lock_path.mkdir(exist_ok=False)
                break
            except FileExistsError:
                try:
                    heartbeat_age = time.time() - heartbeat_path.stat().st_mtime
                    stale = heartbeat_age > heartbeat_stale_after
                    age_desc = f"last heartbeat {heartbeat_age:.0f}s ago"
                except OSError:
                    # No heartbeat file yet -- either a brand-new holder (race right after its own
                    # mkdir, before its first touch) or a holder running older code that predates
                    # this mechanism entirely. Fall back to the lock dir's own age against the full
                    # max_wait, i.e. exactly today's pre-heartbeat behavior.
                    try:
                        lock_age = time.time() - lock_path.stat().st_mtime
                        stale = lock_age > max_wait
                        age_desc = f"no heartbeat yet, lock dir {lock_age:.0f}s old"
                    except OSError:
                        stale = False
                        age_desc = "no heartbeat yet"
                if stale:
                    print(f"  Lock appears stale ({age_desc}), breaking it", flush=True)
                    shutil.rmtree(lock_path, ignore_errors=True)
                    continue
                if waited >= max_wait:
                    raise TimeoutError(f"Timed out waiting for {label} setup lock after {max_wait}s")
                if waited % 30 == 0:
                    print(
                        f"  Waiting for {label} setup lock (held by another process, {age_desc}, "
                        f"{waited}s elapsed)...",
                        flush=True,
                    )
                time.sleep(poll_interval)
                waited += poll_interval
        heartbeat_path.touch()
        stop_heartbeat = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(heartbeat_path, stop_heartbeat, heartbeat_interval),
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            yield
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=heartbeat_interval)
            shutil.rmtree(lock_path, ignore_errors=True)

    # Setup method is sync for now since there's been no need to concurrently set up
    def setup(self) -> Path:
        pass

    def get_run_command(self) -> ExecuteContainerCommandArgs:
        pass

    def postprocess_after_run(self, report_file: Path) -> None:
        pass

    def _get_command_sleep_until_predictions_file(self) -> str:
        return f"until [ -f {self.config.output_for_eval_mounted_path} ]; do sleep 5; done"

    # [CUSTOM-VM-SIDECAR-PREFLIGHT 2026-11-02] VM-sidecar .img readiness check.
    #
    # Note on blast radius: this runs inside SWEBenchWrapper.model_post_init(), which executes
    # exactly once per training job, synchronously, during construction of the single
    # job-wide `NemoGym` Ray actor. A raised exception here is not isolated to one rollout.
    # It IS somewhat undermined by nemo_rl/environments/nemo_gym.py's
    # `@ray.remote(max_restarts=-1, max_task_retries=-1)` on NemoGym -- Ray will retry actor
    # construction indefinitely rather than ever surfacing this as fatal to the driver, so a
    # raised exception alone may manifest as a silent, repeating restart loop rather than a
    # clean job death. The loud, repeated log line below is the fallback for that case: every
    # restart attempt re-runs this check and re-prints the same actionable message, so the
    # failure is at least immediately visible to anyone tailing the job, even when it isn't
    # cleanly fatal at the SLURM level.
    def _require_image_ready(self, label: str, image_path: Path) -> None:
        if image_path.exists():
            return
        logger.error(
            "=" * 88 + "\n"
            f"FATAL: {label} VM-sidecar image is missing at {image_path}\n"
            "This is no longer built lazily at job runtime -- it must be built ahead of time.\n"
            "Fix: re-run presync with BUILD_SWE_ENVS=true (scripts/presync.sbatch) before "
            "launching this job.\n" + "=" * 88
        )
        raise RuntimeError(
            f"{label} VM-sidecar image missing at {image_path} -- "
            "run presync with BUILD_SWE_ENVS=true before launching this job."
        )


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
                self._require_image_ready("SWE-bench", setup_dir / "swebench.img")
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

            # The venv was just freshly rebuilt here -- its VM-sidecar image is now guaranteed
            # stale (see BaseDatasetHarnessProcessor._require_image_ready's own docstring for
            # why that rebuild is no longer done inline).
            self._require_image_ready("SWE-bench", setup_dir / "swebench.img")
            return setup_dir

    def get_run_command(self) -> ExecuteContainerCommandArgs:
        swebench_cmd = (
            f'date +"%s.%N" > {self.config.final_eval_apptainer_spinup_timestamp_mounted_fpath} && '
            f"{self._get_command_sleep_until_predictions_file()} && "
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
            f"    --predictions_path {self.config.output_for_eval_mounted_path} "
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


class SweBenchMultilingualDatasetProcessor(BaseDatasetHarnessProcessor):
    def setup(self) -> Path:
        swebench_repo = "https://github.com/Kipok/SWE-bench.git"
        swebench_commit = "HEAD"

        setup_dir = self.parent_dir / "swe_swebench_multilingual_setup"
        setup_dir.mkdir(parents=True, exist_ok=True)

        with self._setup_directory_lock(setup_dir, "SWE-bench_Multilingual"):
            swebench_multilingual_dir = setup_dir / "SWE-bench_Multilingual"
            uv_dir = setup_dir / "uv"
            python_dir = setup_dir / "python"

            if swebench_multilingual_dir.exists():
                print(f"SWE-bench_Multilingual already set up at {setup_dir}")
                return setup_dir

            print(f"Setting up SWE-bench_Multilingual environment at {setup_dir}...", flush=True)
            script_fpath = self.parent_dir / "setup_scripts/swebench_multilingual.sh"
            command = f"""SETUP_DIR={setup_dir} \\
UV_DIR={uv_dir} \\
PYTHON_DIR={python_dir} \\
SWEBENCH_DIR={swebench_multilingual_dir} \\
SWEBENCH_REPO={swebench_repo} \\
SWEBENCH_COMMIT={swebench_commit} \\
    {script_fpath}"""
            self._run_setup_command(command)

            return setup_dir

    def get_run_command(self) -> ExecuteContainerCommandArgs:
        swebench_cmd = (
            f'date +"%s.%N" > {self.config.final_eval_apptainer_spinup_timestamp_mounted_fpath} && '
            f"{self._get_command_sleep_until_predictions_file()} && "
            # Use pre-built SWE-bench
            "cd /swebench_multilingual_setup/SWE-bench_Multilingual && "
            # Set UV environment variables to use the mounted portable directories
            f'export UV_INSTALL_DIR="{self.config.swebench_multilingual_setup_dir}/uv" && '
            f'export UV_PYTHON_INSTALL_DIR="{self.config.swebench_multilingual_setup_dir}/python" && '
            f'export PATH="{self.config.swebench_multilingual_setup_dir}/uv/bin:$PATH" && '
            f"ls -lrt /root/dataset && "
            # Run with clean environment to avoid venv contamination
            # Use the pre-built venv directly with its absolute path
            f"env -u VIRTUAL_ENV {self.config.swebench_multilingual_setup_dir}/SWE-bench_Multilingual/venv/bin/python -m swebench.harness.run_local_evaluation "
            f"    --predictions_path {self.config.output_for_eval_mounted_path} "
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
        eval_harness_repo = "https://github.com/sdevare-nv/nv-R2E-Gym.git"
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
                    self._require_image_ready("R2E-Gym", setup_dir / "r2e_gym.img")
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

            # The venv was just freshly rebuilt here -- its VM-sidecar image is now guaranteed
            # stale (see BaseDatasetHarnessProcessor._require_image_ready's own docstring for
            # why that rebuild is no longer done inline).
            self._require_image_ready("R2E-Gym", setup_dir / "r2e_gym.img")
            return setup_dir

    def get_run_command(self) -> ExecuteContainerCommandArgs:
        instance_dict = json.loads(self.config.problem_info["instance_dict"])
        base_commit = instance_dict.get("base_commit", "")
        r2e_gym_cmd = (
            f'date +"%s.%N" > {self.config.final_eval_apptainer_spinup_timestamp_mounted_fpath} && '
            f"{self._get_command_sleep_until_predictions_file()} && "
            # [CUSTOM-R2EGYM-EVAL-BASECOMMIT 2026-10-11] the vendored
            # src/r2egym/agenthub/run/run_local_evaluation.py never resets /testbed 
            # itself -- it applies the model's patch via a plain `git apply` against
            # whatever commit the container already is on. Unlike agent mode, eval mode
            # never invokes that harness at all, so /testbed was left at the base image's raw
            # default checkout. To fix this, git checkout {base_commit}` under `cd /app`,
            # just scoped to /testbed and run before.
            # [CUSTOM-R2EGYM-EVAL-HOME 2026-10-11] Apptainer always injects $HOME as our real
            # host home dir (e.g. /leonardo/home/userexternal/<user>), same gotcha already
            # handled for agent mode (see OpenHandsHarnessProcessor.get_run_command's
            # `export HOME=$(mktemp -d /tmp/oh_home.XXXXXX)`, a few hundred lines down).
            # Same fix as agent mode, just scoped to this eval command instead.
            # [CUSTOM-TMPDIR-SCRATCH 2026-10-19] Same fix as OpenHandsHarnessProcessor's identical
            # pattern: `${TMPDIR:-/tmp}` picks up `_build_apptainer_command`'s real, Lustre-backed
            # scratch dir instead of silently overwriting its `-H`-provided $HOME with a fresh dir
            # back on the tiny 64MB overlay.
            f"export HOME=$(mktemp -d \"${{TMPDIR:-/tmp}}/r2egym_eval_home.XXXXXX\") && "
            f"cd /testbed && git config --global --add safe.directory '*' && "
            f"git reset --hard && git checkout --detach "
            f"$(git rev-parse --verify {base_commit}^{{commit}}) && "
            # Use mounted directory path for cd
            "cd /r2egym_setup/R2E-Gym && "
            # Set UV environment variables to use the mounted portable directories
            f'export UV_INSTALL_DIR="{self.config.r2e_gym_setup_dir}/uv" && '
            f'export UV_PYTHON_INSTALL_DIR="{self.config.r2e_gym_setup_dir}/python" && '
            f'export PATH="{self.config.r2e_gym_setup_dir}/uv/bin:$PATH" && '
            # Run with clean environment to avoid venv contamination
            # Use the pre-built venv directly with its absolute path
            f"env -u VIRTUAL_ENV {self.config.r2e_gym_setup_dir}/R2E-Gym/venv/bin/python src/r2egym/agenthub/run/run_local_evaluation.py "
            f"    --predictions_path {self.config.output_for_eval_mounted_path} "
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


class SWESmithDatasetProcessor(BaseDatasetHarnessProcessor):
    # [CUSTOM 2026-10-08] SWE-smith's own `swesmith.harness.eval`/`RepoProfile.get_container()`
    # launch a NEW Docker container internally. We replicate the same logic here without the
    # containerm since we already run inside Singularity.
    def setup(self) -> Path:
        eval_harness_repo = "https://github.com/SWE-bench/SWE-smith.git"
        eval_harness_commit = "HEAD"

        setup_dir = self.parent_dir / "swe_swesmith_setup"

        with self._setup_directory_lock(setup_dir, "SWE-smith"):
            swesmith_dir = setup_dir / "SWE-smith"
            uv_dir = setup_dir / "uv"
            python_dir = setup_dir / "python"

            venv_dir = swesmith_dir / "venv"
            python_bin = venv_dir / "bin" / "python"
            driver_dst = swesmith_dir / "swesmith_eval_driver.py"
            driver_src = self.parent_dir / "swesmith_eval_driver.py"
            if swesmith_dir.exists() and venv_dir.exists() and python_bin.exists():
                driver_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(driver_src, driver_dst)
                result = subprocess_run([str(python_bin), "-c", "from swesmith.profiles import registry"])
                if result.returncode == 0:
                    print(f"SWE-smith already set up at {setup_dir}", flush=True)
                    self._require_image_ready("SWE-smith", setup_dir / "swesmith.img")
                    return setup_dir

                print("SWE-smith directory exists but module not properly installed, rebuilding...", flush=True)

            print(f"Setting up SWE-smith environment at {setup_dir}...", flush=True)
            setup_dir.mkdir(parents=True, exist_ok=True)

            script_fpath = self.parent_dir / "setup_scripts/swesmith.sh"
            command = f"""SETUP_DIR={setup_dir} \\
UV_DIR={uv_dir} \\
PYTHON_DIR={python_dir} \\
SWESMITH_DIR={swesmith_dir} \\
EVAL_HARNESS_REPO={eval_harness_repo} \\
EVAL_HARNESS_COMMIT={eval_harness_commit} \\
    {script_fpath}"""
            self._run_setup_command(command)

            # Our own driver (not part of the vendored package) needs to live inside
            # swesmith_dir so it's covered by the SAME bind-mount as the venv -- see
            # _build_apptainer_command's SWE-smith branch, which mounts swesmith_setup_dir only
            # once (uv's venv wrappers use hardcoded absolute paths, same reasoning as
            # R2E-Gym/SWE-bench's own setup dirs).
            shutil.copy(driver_src, driver_dst)

            # The venv was just freshly rebuilt here -- its VM-sidecar image is now guaranteed
            # stale (see BaseDatasetHarnessProcessor._require_image_ready's own docstring for
            # why that rebuild is no longer done inline).
            self._require_image_ready("SWE-smith", setup_dir / "swesmith.img")
            return setup_dir

    def get_run_command(self) -> ExecuteContainerCommandArgs:
        swesmith_cmd = (
            f'date +"%s.%N" > {self.config.final_eval_apptainer_spinup_timestamp_mounted_fpath} && '
            f"{self._get_command_sleep_until_predictions_file()} && "
            f'export UV_INSTALL_DIR="{self.config.swesmith_setup_dir}/uv" && '
            f'export UV_PYTHON_INSTALL_DIR="{self.config.swesmith_setup_dir}/python" && '
            f'export PATH="{self.config.swesmith_setup_dir}/uv/bin:$PATH" && '
            # Run with clean environment to avoid venv contamination, use the pre-built venv
            # directly with its absolute path -- same pattern as every other processor here.
            f"env -u VIRTUAL_ENV {self.config.swesmith_setup_dir}/SWE-smith/venv/bin/python "
            f"{self.config.swesmith_setup_dir}/SWE-smith/swesmith_eval_driver.py "
            f"    --dataset_path /root/dataset/data.jsonl "
            f"    --predictions_path {self.config.output_for_eval_mounted_path} "
            f"    --instance_id {self.config.instance_id} "
            f"    --testbed_dir /testbed "
            f"    --timeout {self.config.swebench_tests_timeout} "
            f"    --report_path /trajectories_mount/eval-outputs/{self.config.agent_run_id}/report.json"
        )

        search_path = os.path.join(
            self.config.persistent_dir,
            "eval-outputs",
            self.config.agent_run_id,
            "report.json",
        )

        return ExecuteContainerCommandArgs(
            command=swesmith_cmd,
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

date +\"%s.%N\" > {self.config.final_eval_apptainer_spinup_timestamp_mounted_fpath}

{self._get_command_sleep_until_predictions_file()}

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
            timeout=self.config.swebench_tests_timeout,
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


def _load_rebench_log_parsers(rebench_repo_dir: Path):
    lp_path = rebench_repo_dir / "lib" / "agent" / "log_parsers.py"
    if not lp_path.exists():
        lp_path = rebench_repo_dir / "agent" / "log_parsers.py"

    extra_paths = [str(rebench_repo_dir), str(rebench_repo_dir / "lib")]
    added: list[str] = []
    for p in extra_paths:
        if p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)
    try:
        spec = importlib.util.spec_from_file_location("_rebench_log_parsers", str(lp_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for p in added:
            try:
                sys.path.remove(p)
            except ValueError:
                pass


class SWERebenchDatasetProcessor(BaseDatasetHarnessProcessor):
    def setup(self) -> Path:
        setup_dir = self.parent_dir / "swe_rebench_setup"

        with self._setup_directory_lock(setup_dir, "SWE-rebench"):
            rebench_dir = setup_dir / "SWE-rebench-V2"

            if rebench_dir.exists() and (rebench_dir / "agent" / "log_parsers.py").exists():
                print(f"SWE-rebench-V2 already set up at {setup_dir}", flush=True)
                return setup_dir

            print(f"Setting up SWE-rebench-V2 environment at {setup_dir}...", flush=True)
            setup_dir.mkdir(parents=True, exist_ok=True)

            script_fpath = self.parent_dir / "setup_scripts/swe_rebench.sh"
            command = f"""SETUP_DIR={setup_dir} \
REBENCH_DIR={rebench_dir} \
    {script_fpath}"""
            self._run_setup_command(command)

            return setup_dir

    @staticmethod
    def _normalize_test_name(name: str) -> str:
        _REBENCH_TIMING_NORMALIZE_RES = [
            re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
            re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
            re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
        ]
        for pattern in _REBENCH_TIMING_NORMALIZE_RES:
            name = pattern.sub("", name)
        return name.strip()

    def get_run_command(self) -> ExecuteContainerCommandArgs:
        instance_dict = json.loads(self.config.problem_info["instance_dict"])
        install_config = instance_dict.get("install_config", {})
        test_cmds = install_config.get("test_cmd", [])
        if isinstance(test_cmds, str):
            test_cmds = [test_cmds]
        install_cmds = install_config.get("install", [])
        if isinstance(install_cmds, str):
            install_cmds = [install_cmds]
        # log_parser_name = install_config.get("log_parser", "")

        repo = instance_dict.get("repo", "")
        repo_name = repo.split("/")[1] if "/" in repo else repo

        # [CUSTOM-SWEREBENCH-V1-EVAL-CD-FIX 2026-10-16] This eval command is shared by both V1
        # (plain "SWE-rebench") and V2 ("SWE-rebench-V2"), but `cd /{repo_name}` below is only
        # correct for V2 -- V2's per-instance images check the repo out at exactly that path.
        # V1's content lives at /testbed instead (and is separately `cp -a`'d to
        # /workspace/<repo>__<version> for both modes, see _build_apptainer_command's
        # CUSTOM-VM-SIDECAR SWE-rebench branch).
        eval_cd_target = (
            f"/{repo_name}" if self.config.problem_info["dataset_name"] == "SWE-rebench-V2" else "/testbed"
        )

        test_patch = instance_dict.get("test_patch", "")
        test_patch_path = self.config.persistent_dir / "test_patch.diff"
        test_patch_path.write_text(test_patch)

        fail_to_pass = instance_dict.get("FAIL_TO_PASS", [])
        pass_to_pass = instance_dict.get("PASS_TO_PASS", [])
        if isinstance(fail_to_pass, str):
            fail_to_pass = json.loads(fail_to_pass)
        if isinstance(pass_to_pass, str):
            pass_to_pass = json.loads(pass_to_pass)

        # Write test metadata to files to avoid exceeding OS argument length limits
        eval_meta_dir = self.config.persistent_dir / "eval_meta"
        eval_meta_dir.mkdir(parents=True, exist_ok=True)
        # Pre-normalize all expected test names so the in-container eval script
        # can compare directly without duplicating the normalization regexes.
        norm_fail_to_pass = sorted(self._normalize_test_name(n) for n in fail_to_pass)
        norm_pass_to_pass = sorted(self._normalize_test_name(n) for n in pass_to_pass)
        (eval_meta_dir / "expected_passed.json").write_text(
            json.dumps(sorted(set(norm_fail_to_pass + norm_pass_to_pass)))
        )
        (eval_meta_dir / "fail_to_pass.json").write_text(json.dumps(norm_fail_to_pass))
        (eval_meta_dir / "pass_to_pass.json").write_text(json.dumps(norm_pass_to_pass))

        install_block = "\n".join(install_cmds) if install_cmds else ""
        test_block = "\n".join(test_cmds)

        # [CUSTOM-SWEREBENCH-V2 2026-10-09] Was `git reset --hard HEAD`, which only discards
        # working-tree edits -- correct for the old per-instance-image design where HEAD is
        # already pinned at base_commit by construction, but silently wrong for the new
        # shared-mirror design, where /{repo_name} is a copy of a generic multi-commit clone
        # whose HEAD can be anywhere. Resetting to base_commit explicitly is a strict
        # generalization: a no-op where HEAD == base_commit already, and the actual fix
        # otherwise. Safe for both designs since the agent-side OpenHands harness already does
        # the equivalent deep-reset-to-base_commit in its own initialize_runtime().
        base_commit = instance_dict.get("base_commit", "")
        cmd = f"""#!/bin/bash
set -e

date +\"%s.%N\" > {self.config.final_eval_apptainer_spinup_timestamp_mounted_fpath}

{self._get_command_sleep_until_predictions_file()}

cd {eval_cd_target}
git reset --hard {base_commit}

# Apply model patch
git apply --reject --recount --ignore-space-change --whitespace=nowarn /root/patch.diff || true

# Apply test patch
git apply --reject --recount --ignore-space-change --whitespace=nowarn /root/test_patch.diff || true

# Run install commands (non-fatal, some may fail harmlessly)
set +e
{install_block}
set -e

# Run tests and write output to bind-mounted path (parsed on host, no python3 needed)
mkdir -p /trajectories_mount/eval_results
set +e
(
{test_block}
) > /trajectories_mount/eval_results/test_output.log 2>&1
TEST_EXIT=$?
set -e

printf '{{"_test_completed": true, "exit_code": %d}}\\n' $TEST_EXIT \
  > /trajectories_mount/eval_results/report.json
"""

        search_path = os.path.join(
            self.config.persistent_dir,
            "eval_results",
            "report.json",
        )

        return ExecuteContainerCommandArgs(
            command=cmd,
            expected_file_pattern=search_path,
            mode="eval",
            timeout=self.config.swebench_tests_timeout,
        )

    def postprocess_after_run(self, report_file: Path) -> None:
        """Parse test output on the host (avoids needing python3 inside the container)."""
        report_path = Path(report_file)
        test_output_path = report_path.parent / "test_output.log"

        instance_id = self.config.instance_id
        instance_dict = json.loads(self.config.problem_info["instance_dict"])
        install_config = instance_dict.get("install_config", {})
        log_parser_name = install_config.get("log_parser", "")

        if not test_output_path.exists():
            report = {
                instance_id: {
                    "resolved": False,
                    "patch_exists": True,
                    "patch_successfully_applied": False,
                    "error": "No test output produced inside container",
                }
            }
            report_path.write_text(json.dumps(report, indent=2))
            return

        setup_dir = self.parent_dir / "swe_rebench_setup"
        log_parsers = _load_rebench_log_parsers(setup_dir / "SWE-rebench-V2")

        parser = log_parsers.NAME_TO_PARSER.get(log_parser_name) or getattr(log_parsers, log_parser_name, None)
        if parser is None:
            report = {
                instance_id: {
                    "resolved": False,
                    "patch_exists": True,
                    "patch_successfully_applied": True,
                    "error": f"Unknown log parser: {log_parser_name}",
                }
            }
            report_path.write_text(json.dumps(report, indent=2))
            return

        test_output = test_output_path.read_text(errors="replace")
        results = parser(test_output)
        results = {self._normalize_test_name(k): v for k, v in results.items()}
        passed = sorted(k for k, v in results.items() if v == "PASSED")

        eval_meta_dir = self.config.persistent_dir / "eval_meta"
        expected_passed = json.loads((eval_meta_dir / "expected_passed.json").read_text())
        norm_f2p = json.loads((eval_meta_dir / "fail_to_pass.json").read_text())
        norm_p2p = json.loads((eval_meta_dir / "pass_to_pass.json").read_text())

        passed_set = set(passed)
        fail_to_pass_set = set(norm_f2p)
        pass_to_pass_set = set(norm_p2p)

        from_fail_to_pass = sorted(passed_set & fail_to_pass_set)
        failed_from_pass_to_pass = sorted(pass_to_pass_set - passed_set)
        resolved = (fail_to_pass_set <= passed_set) and (pass_to_pass_set <= passed_set)

        report = {
            instance_id: {
                "resolved": resolved,
                "patch_exists": True,
                "patch_successfully_applied": True,
                "from_fail_to_pass": from_fail_to_pass,
                "failed_from_pass_to_pass": failed_from_pass_to_pass,
                "passed_match": passed == expected_passed,
            }
        }
        report_path.write_text(json.dumps(report, indent=2))


class SweBenchExtDatasetProcessor(BaseDatasetHarnessProcessor):
    """Dataset processor for SWE-Bench-Ext format tasks."""

    def _get_instance_dict(self) -> dict:
        raw = self.config.problem_info.get("instance_dict", "{}")
        if isinstance(raw, str):
            return json.loads(raw)
        return raw

    def get_run_command(self) -> ExecuteContainerCommandArgs:
        from responses_api_agents.swe_agents.swe_bench_ext.frameworks import (
            get_framework_config,
            get_test_command_with_output,
        )

        inst = self._get_instance_dict()

        base_command = inst.get("test_command", "")
        base_commit = inst.get("base_commit", "")
        test_patch = inst.get("test_patch", "")
        test_framework = inst.get("test_framework", "")

        # Write test patch to persistent_dir (mounted into container)
        test_patch_path = self.config.persistent_dir / "test_patch.diff"
        test_patch_path.write_text(test_patch)

        # Write eval metadata for host-side postprocessing
        fail_to_pass = inst.get("FAIL_TO_PASS", inst.get("fail_to_pass", []))
        pass_to_pass = inst.get("PASS_TO_PASS", inst.get("pass_to_pass", []))
        if isinstance(fail_to_pass, str):
            fail_to_pass = json.loads(fail_to_pass)
        if isinstance(pass_to_pass, str):
            pass_to_pass = json.loads(pass_to_pass)

        eval_meta_dir = self.config.persistent_dir / "eval_meta"
        eval_meta_dir.mkdir(parents=True, exist_ok=True)
        (eval_meta_dir / "fail_to_pass.json").write_text(json.dumps(fail_to_pass))
        (eval_meta_dir / "pass_to_pass.json").write_text(json.dumps(pass_to_pass))
        (eval_meta_dir / "test_framework.txt").write_text(test_framework)

        reset_cmd = f"git reset --hard {base_commit}" if base_commit else ""

        # Use lighthouse to add structured output flags (--json, --junitxml, etc.)
        # This is the same transformation swe_bench_ext_agent/task.py applies.
        test_cmd = get_test_command_with_output(base_command, test_framework)
        config = get_framework_config(test_framework, base_command)
        result_file = config.get("result_file")

        # Build the result file dump block (mirrors task.py's generate_test_run_script)
        result_file_block = ""
        if result_file:
            if "*" in result_file:
                result_file_block = f"""
echo "<<<SWE_BENCH_EXT_RESULT_FILE_START>>>"
for f in {result_file}; do
    if [ -f "$f" ]; then
        echo "=== FILE: $f ==="
        cat "$f"
        echo ""
    fi
done 2>/dev/null || true
echo "<<<SWE_BENCH_EXT_RESULT_FILE_END>>>"
"""
            else:
                result_file_block = f"""
echo "<<<SWE_BENCH_EXT_RESULT_FILE_START>>>"
if [ -f "{result_file}" ]; then
    cat "{result_file}"
fi
echo "<<<SWE_BENCH_EXT_RESULT_FILE_END>>>"
"""

        cmd = f"""#!/bin/bash
set -o pipefail

date +\"%s.%N\" > {self.config.final_eval_apptainer_spinup_timestamp_mounted_fpath}

{self._get_command_sleep_until_predictions_file()}

# Try common repo locations in the container
cd /testbed 2>/dev/null || cd /workspace/repo 2>/dev/null || cd /app 2>/dev/null || true

# Reset to base commit if specified
{reset_cmd}

# Apply model patch (agent output or golden patch)
git apply --reject --recount --ignore-space-change --ignore-whitespace /root/patch.diff || true

# Apply test patch (adds/modifies test files)
git apply --reject --recount --ignore-space-change --ignore-whitespace /root/test_patch.diff || true

# Run tests with structured output and capture to log
mkdir -p /trajectories_mount/eval_results /workspace/test-results
set +e
(
echo "<<<SWE_BENCH_EXT_TEST_OUTPUT_START>>>"
{test_cmd}
test_exit_code=$?
{result_file_block}
echo "<<<SWE_BENCH_EXT_TEST_OUTPUT_END>>>"
exit $test_exit_code
) > /trajectories_mount/eval_results/test_output.log 2>&1
TEST_EXIT=$?
set -e

printf '{{"_test_completed": true, "exit_code": %d}}\\n' $TEST_EXIT \
  > /trajectories_mount/eval_results/report.json
"""

        search_path = os.path.join(
            self.config.persistent_dir,
            "eval_results",
            "report.json",
        )

        return ExecuteContainerCommandArgs(
            command=cmd,
            expected_file_pattern=search_path,
            mode="eval",
            timeout=self.config.swebench_tests_timeout,
        )

    def postprocess_after_run(self, report_file: Path) -> None:
        """Parse test output on the host using lighthouse's parsing library."""
        from responses_api_agents.swe_agents.swe_bench_ext.utils import parse_and_check_tests

        report_path = Path(report_file)
        test_output_path = report_path.parent / "test_output.log"
        instance_id = self.config.instance_id

        if not test_output_path.exists():
            report = {
                instance_id: {
                    "resolved": False,
                    "patch_exists": True,
                    "patch_successfully_applied": False,
                    "error": "No test output produced inside container",
                }
            }
            report_path.write_text(json.dumps(report, indent=2))
            return

        eval_meta_dir = self.config.persistent_dir / "eval_meta"
        fail_to_pass = json.loads((eval_meta_dir / "fail_to_pass.json").read_text())
        pass_to_pass = json.loads((eval_meta_dir / "pass_to_pass.json").read_text())
        test_framework = (eval_meta_dir / "test_framework.txt").read_text().strip()

        test_output = test_output_path.read_text(errors="replace")

        result = parse_and_check_tests(
            test_output=test_output,
            test_framework=test_framework,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            instance_id=instance_id,
        )

        report = {instance_id: result}
        report_path.write_text(json.dumps(report, indent=2))


class OpenHandsHarnessProcessor(BaseDatasetHarnessProcessor):
    def setup(self) -> Path:
        setup_dir = self.parent_dir / "swe_openhands_setup"

        with self._setup_directory_lock(setup_dir, "OpenHands"):
            openhands_dir = setup_dir / "OpenHands"
            miniforge_dir = setup_dir / "miniforge3"

            # [CUSTOM-OPENHANDS-VENV-CHECK 2026-10-14] Previously just checked that
            # .venv/bin/python exists. Howevr an interrupted `poetry install` left `python`
            # present but other dependencies not available, causing `ModuleNotFoundError` 
            # when importing inside OpenHands.
            # Mirrors the real-import-check pattern 
            # R2EGymDatasetProcessor/SWESmithDatasetProcessor.setup() already use above.
            # [CUSTOM-OPENHANDS-VENV-CHECK-PPTX 2026-10-15] Added `pptx` after another
            # `ModuleNotFoundError`. Fixed by adding a second `poetry install` pass in
            # setup_scripts/openhands.sh; this addition just makes this check catch the gap if
            # it recurs.
            python_bin = openhands_dir / ".venv" / "bin" / "python"
            if openhands_dir.exists() and python_bin.exists():
                result = subprocess_run(
                    [
                        str(python_bin),
                        "-c",
                        "from openhands.core.main import create_runtime\n"
                        "import PIL\n"
                        "import browsergym\n"
                        "import pandas\n"
                        "import reportlab\n"
                        "import gevent\n"
                        "import pptx\n",
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    print(f"OpenHands already set up at {setup_dir}", flush=True)
                    self._require_image_ready("OpenHands", setup_dir / "openhands.img")
                    return setup_dir

                print(
                    f"OpenHands directory exists at {setup_dir} but failed a real import check "
                    f"(venv is incomplete/stale, not just missing) -- rebuilding. Import error "
                    f"was:\n{result.stderr}",
                    flush=True,
                )

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

            # [CUSTOM-OPENHANDS-POST-REBUILD-VERIFY 2026-10-16] A final re-check here as
            # a hard backstop in case the complete poetry install failed.
            result = subprocess_run(
                [
                    str(python_bin),
                    "-c",
                    "from openhands.core.main import create_runtime\n"
                    "import PIL\n"
                    "import browsergym\n"
                    "import pandas\n"
                    "import reportlab\n"
                    "import gevent\n"
                    "import pptx\n",
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (
                f"OpenHands rebuild at {setup_dir} still failed the real import check afterward "
                f"-- refusing to proceed with a known-broken venv. Import error was:\n{result.stderr}"
            )

            # The venv was just freshly rebuilt here -- its VM-sidecar image is now guaranteed
            # stale (built, if at all, from the PREVIOUS venv content) -- see
            # BaseDatasetHarnessProcessor._require_image_ready's own docstring for why that
            # rebuild is no longer done inline.
            self._require_image_ready("OpenHands", setup_dir / "openhands.img")
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
            "model": self.config.body.model or get_global_config_dict()["policy_model_name"],
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
            profiling_cmd = f"export NG_PROFILING_DIR={self.config.profiling_mounted_dir} && "
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

        if data_point["dataset_name"] == "nv-internal-1" or data_point["dataset_name"] == "swe-bench-ext":
            crypto_fix_cmd = (
                # [CUSTOM-TMPDIR-SCRATCH 2026-10-19] ${TMPDIR:-/tmp}, not a literal /tmp -- same
                # overlay-size reasoning as OpenHandsHarnessProcessor's/R2E-Gym's HOME fixes above.
                "_crypto_fix_dir=$(mktemp -d \"${TMPDIR:-/tmp}/crypto_fix_XXXXXX\") && "
                "/openhands_setup/OpenHands/.venv/bin/python -m pip install "
                "    --target=$_crypto_fix_dir "
                "    --index-url https://pypi.org/simple "
                "    --trusted-host pypi.org --trusted-host files.pythonhosted.org "
                "    --only-binary :all: "
                "    --no-deps --no-cache-dir "
                "    --quiet "
                "    'cryptography<43' && "
                "export PYTHONPATH=$_crypto_fix_dir:${PYTHONPATH:-} &&"
            )
        else:
            crypto_fix_cmd = ""

        if self.config.resolved_diversify_tool_names:
            diversify_tool_names_cmd = "export DIVERSIFY_TOOL_NAMES=true &&"
        else:
            diversify_tool_names_cmd = ""

        if self.config.resolved_camel_case_tool_names:
            camel_case_tool_names_cmd = "export CAMEL_CASE_TOOL_NAMES=true &&"
        else:
            camel_case_tool_names_cmd = ""

        workspace_check_cmd = ""

        agent_main_cmd = (
            f"{workspace_check_cmd}"
            # Add miniforge bin to PATH (for tmux, node, poetry, etc.)
            "mkdir -p /tmp/ && "
            "export PATH=/openhands_setup/miniforge3/bin:$PATH && "
            # Our apptainer invocation uses --cleanenv, so $USER is unset inside the container.
            # OpenHands' LocalRuntime (openhands/runtime/utils/command.py) falls back to the
            # hardcoded username "root" whenever override_username/$USER/$RUNTIME_USERNAME are
            # all unset. Since we e have no root privileges here (no --fakeroot available in 
            # Leonardo), this breaks things. SU_TO_USER=false is OpenHands' flag to avoid using
            # root-privilege commands such as `su`.
            "export SU_TO_USER=false && "
            # The task repo (/testbed, baked into each instance's image at Docker build time,
            # typically root-owned) belongs to a different UID than us (no --fakeroot, we're our
            # real host UID). Git >=2.35.2's "dubious ownership" safety check then refuses any
            # operation on it. `--add safe.directory '*'` is git's flag to avoid this.
            "git config --global --add safe.directory '*' && "
            # Setup tmux socket (OpenHands requirement)
            "uid=$(id -ru 2>/dev/null || id -u) && "
            "export TMUX_TMPDIR=/tmp && "
            "export TMUX=/tmp/tmux-$uid/default && "
            "mkdir -p /tmp/tmux-$uid && "
            "chown $uid:$uid /tmp/tmux-$uid || true && "
            "chmod 700 /tmp/tmux-$uid && "
            "tmux -S /tmp/tmux-$uid/default start-server || true && "
            "cp /openhands_setup/miniforge3/bin/jq /usr/local/bin/jq 2>/dev/null || true && "
            # Use pre-built OpenHands
            "cd /openhands_setup/OpenHands && "
            "export RUNTIME=local && "
            f'date +"%s.%N" > {self.config.generation_apptainer_spinup_timestamp_mounted_fpath} && '
            f"{log_cmd}"
            f"{profiling_cmd}"
            f"export NEMO_GYM_METRICS_FPATH={self.config.base_mounted_dir}/nemo_gym_metrics.json && "
            f"export NEMO_GYM_CONFIG_DICT={self.config.ng_global_config_dict_str} && "
            f"export NEMO_GYM_MODEL_SERVER_NAME={self.config.model_server_name} &&"
            # [CUSTOM-CONTEXT-OVERFLOW-PREFLIGHT 2026-10-11] see the field docstring on
            # SWEBenchWrapperConfig.max_model_len above for the usage of 
            # NEMO_GYM_MAX_MODEL_LEN and NEMO_GYM_PREFLIGHT_HEADROOM_TOKENS.
            f"export NEMO_GYM_MAX_MODEL_LEN={self.config.max_model_len or ''} &&"
            f"export NEMO_GYM_PREFLIGHT_HEADROOM_TOKENS={self.config.preflight_headroom_tokens} &&"
            "export VIRTUAL_ENV=/openhands_setup/OpenHands/.venv && "
            "export PATH=$PATH:/openhands_setup/OpenHands/.venv/bin && "
            # CRITICAL: Configure poetry to only use the OpenHands venv (ignore external venvs)
            "export POETRY_VIRTUALENVS_IN_PROJECT=true && "
            "export POETRY_VIRTUALENVS_CREATE=false && "
            "export POETRY_VIRTUALENVS_PATH=/openhands_setup/OpenHands && "
            f"export TMUX_MEMORY_LIMIT={self.config.apptainer_memory_limit_mb} && "
            f"export COMMAND_EXEC_TIMEOUT={self.config.command_exec_timeout} && "
            f"{crypto_fix_cmd}"
            f"{diversify_tool_names_cmd}"
            f"{camel_case_tool_names_cmd}"
            f"echo {shlex.quote(config_str)} >{config_file_path} && "
            # f" export EVAL_OUTPUT_DIR={eval_dir_in_openhands} && "
            f"./evaluation/benchmarks/swe_bench/scripts/run_infer.sh "
            f"    llm.model "  # name of llm config section in config.toml
            f"    {self.config.agent_framework_commit} "  # openhands commit
            f"    {self.config.resolved_agent_cls} "  # agent
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

        if self.config.resolved_user_prompt_template is not None:
            agent_main_cmd += "    /openhands_setup/OpenHands/user_prompt.j2 "
        if self.config.resolved_user_prompt_template is not None:
            agent_main_cmd += "    /openhands_setup/OpenHands/system_prompt.j2 "
            agent_main_cmd += "    /openhands_setup/OpenHands/system_prompt_long_horizon.j2 "

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


def _classify_agent_error(err: Optional[str]) -> Optional[str]:
    if not err:
        return None
    s = str(err)
    if "maximum iteration" in s:
        return "max_iteration"
    if "ContextWindow" in s or "context window" in s.lower():
        return "context_window"
    if "stuck in a loop" in s.lower():
        return "stuck_in_loop"
    return "other"


# [CUSTOM-R2EGYM-EVAL-ISOLATION 2026-10-09] R2E-Gym's own per-instance images ship these two
# files baked in at image-BUILD time: they are R2E-Gym's own infra (how-to-install / 
# how-to-run-tests helpers).
#
# The bug this works around: OpenHands' own patch extraction
# (evaluation/benchmarks/swe_bench/run_infer.py) does `git add -A && git diff --cached
# <base_commit>`, which shows them as "new files", on every single rollout. At eval time,
# a FRESH container (same image) already has these same files present, so `git apply`'s
# fails with "already exists in working directory", masking whatever changes the agent
# also made.
_R2EGYM_INFRA_FILENAMES = frozenset({"install.sh", "run_tests.sh"})


def _strip_r2egym_infra_files_from_patch(patch: str) -> str:
    """Drop any unified-diff file block touching R2E-Gym's own baked-in infra files.

    See _R2EGYM_INFRA_FILENAMES' docstring above for why these must never end up in the
    model's graded patch.
    """
    if not patch:
        return patch
    lines = patch.splitlines(keepends=True)
    out: List[str] = []
    skipping = False
    for line in lines:
        if line.startswith("diff --git "):
            # Format: "diff --git a/<path> b/<path>\n" -- path may itself contain spaces, so
            # only rely on the well-known "a/" / "b/" prefixes, not naive whitespace splitting.
            parts = line.rstrip("\n").split(" b/", 1)
            a_path = parts[0][len("diff --git a/") :] if len(parts) == 2 else ""
            b_path = parts[1] if len(parts) == 2 else ""
            skipping = os.path.basename(a_path) in _R2EGYM_INFRA_FILENAMES or (
                os.path.basename(b_path) in _R2EGYM_INFRA_FILENAMES
            )
        if not skipping:
            out.append(line)
    return "".join(out)


_PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")


def _container_proxy_env() -> dict[str, str]:
    return {var: os.environ[var] for var in _PROXY_ENV_VARS if os.environ.get(var)}


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
    },
    num_cpus=0.1,
)
def runner_ray_remote(params_dict: dict[str, Any]) -> Optional[Path]:
    # For some reason Ray may not pick up the proper model fields if we don't rebuild the model here. Very strange.
    SWEBenchWrapperInstanceConfig.model_rebuild(force=True)
    RunOpenHandsAgent.model_rebuild(force=True)

    params = SWEBenchWrapperInstanceConfig.model_validate(params_dict)
    run_oh = RunOpenHandsAgent(config=params)
    report_file = asyncio.run(run_oh.process_single_datapoint())

    return report_file


def update_metrics(metrics_fpath: Path, update_dict: Dict[str, Any]) -> None:
    with metrics_fpath.open() as f:
        existing_dict = json.loads(f.read())

    existing_dict = {k: v for k, v in existing_dict.items() if v is not None}
    update_dict = {k: v for k, v in update_dict.items() if v is not None}

    with metrics_fpath.open("w") as f:
        json.dump(existing_dict | update_dict, f)


# _TOOL_PARAM_BOOL_FIELDS_DEFAULT_FALSE = ("defer_loading",)


# def _dump_tool_as_tool_param(tool: BaseModel) -> Dict[str, Any]:
#     """Dump a response Tool pydantic model to a ToolParam-compatible dict."""
#     data = tool.model_dump()
#     for key in _TOOL_PARAM_BOOL_FIELDS_DEFAULT_FALSE:
#         if data.get(key) is None:
#             data[key] = False
#     return data


class ActiveContainerCommand(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    process: Process
    log_file: Any
    log_file_path: Path


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

    async def _start_container_command(
        self, command: ExecuteContainerCommandArgs, apptainer_cmd: str
    ) -> ActiveContainerCommand:
        # Stream output to log file as it appears
        logs_dir = self.config.persistent_dir / "apptainer_logs"
        logs_dir.mkdir(exist_ok=True)
        log_file_path = logs_dir / f"{self.config.instance_id}_{command.mode}.log"
        log_file = open(log_file_path, "w")

        dataset_name = self.config.problem_info.get("dataset_name")
        if _uses_vm_sidecar(dataset_name):
            # [CUSTOM-VM-SIDECAR 2026-10-27] Route through this node's persistent VM sidecar
            # instead of running the apptainer command directly on the host -- see
            # nemo_gym/vm_launcher.py's.
            #
            # ensure_vm() is a blocking call (subprocess/file-lock based, like
            # sandbox_launcher.ensure_sandbox()) -- safe to call directly here despite this
            # being an async method because each rollout is its own Ray task running
            # asyncio.run() in its own process; there is no other concurrent coroutine sharing
            # this event loop for a blocking call to stall.
            #
            # Do a passthrough of HTTP_PROXY/HTTPS_PROXY (SSH does not punch through
            # arbitrary environment variables by default).
            proxy_env = {
                var: os.environ[var]
                for var in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")
                if os.environ.get(var)
            }
            # [CUSTOM-VM-SIDECAR-BLOCKDEV 2026-10-28] Every dataset harness's own venv is imaged
            # as a dedicated read-only block device: these can be though of as basically 
            # "pre-compiled disks" that are attached to the VM and contain all the files needed
            # for the harness to work.
            # Each image shadows ONLY its venv/.venv dir, NOT the whole setup dir.
            openhands_dir = f"{self.config.openhands_setup_dir}/OpenHands"
            openhands_image = f"{self.config.openhands_setup_dir}/openhands.img"
            swebench_dir = f"{self.config.swebench_setup_dir}/SWE-bench"
            swebench_image = f"{self.config.swebench_setup_dir}/swebench.img"
            r2e_gym_dir = f"{self.config.r2e_gym_setup_dir}/R2E-Gym"
            r2e_gym_image = f"{self.config.r2e_gym_setup_dir}/r2e_gym.img"
            swesmith_dir = f"{self.config.swesmith_setup_dir}/SWE-smith"
            swesmith_image = f"{self.config.swesmith_setup_dir}/swesmith.img"
            vm_info = ensure_vm(
                golden_image_dir=self.config.vm_golden_image_dir,
                lustre_share_roots=self.config.vm_lustre_share_roots,
                block_device_mounts=[
                    (openhands_image, f"{openhands_dir}/.venv"),
                    (swebench_image, f"{swebench_dir}/venv"),
                    (r2e_gym_image, f"{r2e_gym_dir}/venv"),
                    (swesmith_image, f"{swesmith_dir}/venv"),
                ],
            )
            process = await run_in_vm(vm_info, apptainer_cmd, env=proxy_env, stdout=log_file, stderr=log_file)
        else:
            process = await asyncio.create_subprocess_shell(apptainer_cmd, stdout=log_file, stderr=log_file)

        return ActiveContainerCommand(process=process, log_file=log_file, log_file_path=log_file_path)

    async def _finish_container_command(
        self, active_command: ActiveContainerCommand, command: ExecuteContainerCommandArgs
    ) -> str:
        data_point = self.config.problem_info

        try:
            # Wait for completion with timeout
            await asyncio.wait_for(active_command.process.communicate(), timeout=command.timeout)
        except asyncio.TimeoutError:
            await _kill_and_reap_process(
                active_command.process, context=f"{command.mode} container command for {self.config.instance_id}"
            )
            active_command.log_file.close()
            log_tail = active_command.log_file_path.read_text(errors="replace")[-4000:]
            raise ValueError(
                f"Command timed out after {command.timeout}s (mode={command.mode}, "
                f"instance={data_point.get('instance_id')}, expected_file_pattern="
                f"{command.expected_file_pattern!r}). Log tail:\n{log_tail}"
            )
        finally:
            if not active_command.log_file.closed:
                active_command.log_file.close()

        if active_command.process.returncode != 0:
            raise RuntimeError(
                f"Command failed with return code {active_command.process.returncode}. "
                f"Logs:\n{active_command.log_file_path.read_text(errors='replace')}"
            )

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
            log_tail = active_command.log_file_path.read_text(errors="replace")[-4000:]
            raise ValueError(
                f"Expected exactly one file matching {command.expected_file_pattern} for {data_point['instance_id']}, "
                f"found {len(pred_files)} (mode={command.mode}, process returncode="
                f"{active_command.process.returncode}, process exited cleanly but never produced "
                f"the expected output file). Log tail:\n{log_tail}"
            )

    async def _kill_active_command(self, active_command: ActiveContainerCommand) -> None:
        await _kill_and_reap_process(active_command.process, context=f"container command for {self.config.instance_id}")
        active_command.log_file.close()

    async def process_single_datapoint(self) -> Optional[Path]:
        if self.config.verify_golden_patch:
            return await self._run_golden_patch_verification()

        instance_id = self.config.instance_id
        if self.config.debug:
            profiler = Profiler(name=instance_id, base_profile_dir=self.config.profiling_mounted_dir)
            profiler.start()

        metrics = SWEBenchMetrics(ray_queue_time=time.time() - self.config.ray_queue_timestamp)

        metrics.openhands_run_time = -time.time()
        metrics.generation_apptainer_spinup_time = metrics.openhands_run_time
        metrics.final_eval_apptainer_spinup_time = metrics.openhands_run_time

        openhands_active_command = await self._start_container_command(
            self.config.agent_command, self.config.agent_apptainer_command_str
        )
        eval_active_command = await self._start_container_command(
            self.config.eval_command, self.config.eval_apptainer_command_str
        )

        try:
            out_file_in_eval = await self._finish_container_command(
                openhands_active_command, self.config.agent_command
            )
            out_file = self._openhands_dir_copy_from_host(output_file_path=out_file_in_eval)
        except Exception as e:
            print(f"Agent command failed for {instance_id}: {e}", flush=True)
            try:
                self._openhands_dir_copy_from_host(output_file_path=None)
            except Exception:
                pass
            await self._kill_active_command(eval_active_command)
            metrics.openhands_run_time += time.time()
            metrics.patch_exists = False
            metrics.final_eval_apptainer_spinup_time = None
            # Detect wall-clock agent timeout: openhands_run_time (elapsed since start)
            # reached or exceeded the configured swebench_agent_timeout.
            metrics.agent_timed_out = (
                metrics.openhands_run_time is not None
                and metrics.openhands_run_time >= self.config.swebench_agent_timeout
            )
            update_metrics(self.config.metrics_fpath, metrics.model_dump())
            if self.config.debug:
                profiler.stop()
            return None

        generation_apptainer_spinup_timestamp = float(
            self.config.generation_apptainer_spinup_timestamp_fpath.read_text()
        )
        metrics.generation_apptainer_spinup_time += generation_apptainer_spinup_timestamp
        metrics.openhands_run_time += time.time()

        with open(out_file, "r") as f:
            out_dict = json.loads(f.read().strip())

        metrics.agent_error_kind = _classify_agent_error(out_dict.get("error"))

        patch = out_dict["test_result"]["git_patch"] or None
        patch = patch + "\n" if patch and not patch.endswith("\n") else patch
        # [CUSTOM-R2EGYM-EVAL-ISOLATION 2026-10-09] see _strip_r2egym_infra_files_from_patch's
        # docstring: fixing the "already exists" git-apply failure at eval time.
        if patch and "R2E-Gym" in (self.config.problem_info.get("dataset_name", "") or ""):
            patch = _strip_r2egym_infra_files_from_patch(patch)
        metrics.model_patch = patch

        # [CUSTOM-PATCH-WRITE-ORDER] Write the real patch before the eval-ready signal file
        # below, not after: the eval container's `until -f output_for_eval_path` gate must
        # never be able to observe readiness before model_patch_path holds the real content.
        if patch:
            with open(self.config.model_patch_path, "w") as f:
                f.write(patch)

        # Create file in the SWE-bench evaluation format
        self.config.output_for_eval_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.output_for_eval_path.open("w") as f:
            f.write(
                json.dumps(
                    {
                        "model_name_or_path": out_dict["metadata"]["llm_config"]["model"],
                        "instance_id": out_dict["instance_id"],
                        "model_patch": patch,
                        "oh_time_metrics": out_dict["metrics"],
                    }
                )
            )

        # Dump out dot and png files from profiling on OpenHands level
        if self.config.debug:
            try:
                profiling_name = "openhands"
                callgrind_path = self.config.profiling_dir / f"{profiling_name}.callgrind"
                callgrind_dotfile_path = self.config.profiling_dir / f"{profiling_name}.dot"
                callgrind_graph_path = self.config.profiling_dir / f"{profiling_name}.png"

                gprof2dot_main(
                    argv=f"--format=callgrind --output={callgrind_dotfile_path} -e 5 -n 5 {callgrind_path}".split()
                )

                (graph,) = graph_from_dot_file(callgrind_dotfile_path)
                graph.write_png(callgrind_graph_path)
            except Exception as e:
                print(f"Error dumping profiling files: {e}", flush=True)

        if not patch:
            metrics.patch_exists = False
            metrics.final_eval_apptainer_spinup_time = None

            await self._kill_active_command(eval_active_command)

            update_metrics(self.config.metrics_fpath, metrics.model_dump())
            return

        metrics.final_eval_time = -time.time()
        try:
            report_file = await self._finish_container_command(eval_active_command, self.config.eval_command)
        except Exception as e:
            print(f"Eval command failed for {instance_id}: {e}", flush=True)
            metrics.final_eval_time += time.time()
            metrics.patch_exists = True
            # Detect wall-clock eval timeout: final_eval_time (elapsed since eval start)
            # reached or exceeded the configured swebench_tests_timeout.
            metrics.eval_timed_out = (
                metrics.final_eval_time is not None and metrics.final_eval_time >= self.config.swebench_tests_timeout
            )
            update_metrics(self.config.metrics_fpath, metrics.model_dump())
            if self.config.debug:
                profiler.stop()
            return None

        final_eval_apptainer_spinup_timestamp = float(
            self.config.final_eval_apptainer_spinup_timestamp_fpath.read_text()
        )
        metrics.final_eval_apptainer_spinup_time += final_eval_apptainer_spinup_timestamp
        metrics.final_eval_time += time.time()

        metrics.patch_exists = True
        update_metrics(self.config.metrics_fpath, metrics.model_dump())

        if self.config.debug:
            profiler.stop()

        return report_file

    async def _run_golden_patch_verification(self) -> Optional[Path]:
        instance_id = self.config.instance_id
        dataset_name = self.config.problem_info.get("dataset_name")
        # TODO(sugam): add support for other datasets
        if dataset_name != "swe-bench-ext":
            raise NotImplementedError(
                f"verify_golden_patch is only supported for dataset_name=='swe-bench-ext' (got {dataset_name!r})."
            )

        instance_dict = json.loads(self.config.problem_info["instance_dict"])
        golden_patch = instance_dict.get("patch") or ""
        if not golden_patch.strip():
            raise ValueError(f"No golden patch found in instance_dict['patch'] for {instance_id}.")
        if not golden_patch.endswith("\n"):
            golden_patch += "\n"

        metrics = SWEBenchMetrics(ray_queue_time=time.time() - self.config.ray_queue_timestamp)
        metrics.model_patch = golden_patch
        metrics.patch_exists = True

        # Write golden patch where the agent would have written the model patch.
        # [CUSTOM-PATCH-WRITE-ORDER] Before the eval-ready signal file below, not after --
        # see the matching comment in the agent-patch path above.
        with open(self.config.model_patch_path, "w") as f:
            f.write(golden_patch)

        self.config.output_for_eval_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.output_for_eval_path.open("w") as f:
            f.write(
                json.dumps(
                    {
                        "model_name_or_path": "golden_patch_verification",
                        "instance_id": instance_id,
                        "model_patch": golden_patch,
                    }
                )
            )

        metrics.final_eval_apptainer_spinup_time = -time.time()
        metrics.final_eval_time = -time.time()

        eval_active_command = await self._start_container_command(
            self.config.eval_command, self.config.eval_apptainer_command_str
        )
        try:
            report_file = await self._finish_container_command(eval_active_command, self.config.eval_command)
        except Exception as e:
            print(f"Golden-patch eval failed for {instance_id}: {e}", flush=True)
            metrics.final_eval_time += time.time()
            update_metrics(self.config.metrics_fpath, metrics.model_dump())
            return None

        final_eval_apptainer_spinup_timestamp = float(
            self.config.final_eval_apptainer_spinup_timestamp_fpath.read_text()
        )
        metrics.final_eval_apptainer_spinup_time += final_eval_apptainer_spinup_timestamp
        metrics.final_eval_time += time.time()

        update_metrics(self.config.metrics_fpath, metrics.model_dump())

        return report_file


########################################
# START Server logic
########################################


def _find_sif_normalized(container_dir: str, image_stem: str) -> list[str]:
    """Hyphen/underscore/case-insensitive fallback for a shared-image `.sif` lookup.

    Do a fuzzy search to match a dataset's own naming field (e.g. SWE-rebench-V2's
    "kotlin-jdk-11") against the `.sif` file name (e.g. "kotlin_jdk_11.sif"). Colons are
    stripped too (e.g. "php_8.3.16" vs "php:8.3.16").
    """
    if not os.path.isdir(container_dir):
        return []
    target = re.sub(r"[-_:]", "", image_stem).lower()
    if not target:
        return []
    return [
        os.path.join(container_dir, name)
        for name in sorted(os.listdir(container_dir))
        if name.endswith(".sif") and target in re.sub(r"[-_:]", "", name).lower()
    ]


# [CUSTOM-COPYOUT-DSTATE-REAP 2026-10-26] `process.kill()` sends SIGKILL, but a process blocked
# in kernel D-state (uninterruptible I/O wait) cannot be killed by ANY signal until the
# I/O it's blocked on completes or the filesystem driver gives up. `await process.wait()`
# immediately after `kill()` therefore hangs for exactly as long as `communicate()` itself would
# have -- the timeout just moves one line down instead of being eliminated.
# Fix: bound the post-kill reap with its own short grace period. If the process hasn't exited
# within it, it's genuinely D-state-stuck -- stop awaiting it (the orphan is leaked at the OS
# level, but that's strictly better than the coroutine, its semaphore slot, and its trainer slot
# being leaked right along with it) and let the caller treat this attempt as failed/retryable.
async def _kill_and_reap_process(process: asyncio.subprocess.Process, context: str, grace_seconds: float = 15.0) -> bool:
    """Kill `process` and wait up to `grace_seconds` for it to actually exit.

    Returns True if the process was reaped, False if it's still alive after the grace period
    (almost certainly D-state) and has been abandoned -- callers must not wait on it further.
    """
    if process.returncode is not None:
        return True
    process.kill()
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        return True
    except asyncio.TimeoutError:
        print(
            f"WARNING: {context} did not exit within {grace_seconds:.0f}s of being killed -- "
            f"likely stuck in kernel D-state on a degraded node. Abandoning it (leaked at the OS "
            f"level) rather than blocking this rollout on it indefinitely.",
            flush=True,
        )
        return False


# [CUSTOM-VM-SIDECAR 2026-10-27] Which dataset types need the per-node VM sidecar (real root,
# no --fakeroot) instead of running the apptainer command directly on the host -- see
# nemo_gym/vm_launcher.py. Exactly the five dataset types that used to call
# _copy_out_writable_paths for a writable /testbed or /{repo_name}; everything else (Harbor,
# plain SWE-bench, SWE-bench_Multilingual, nv-internal-1, swe-bench-ext) never hit the
# root-owned-path problem in the first place and keeps running on the bare host. Module-level
# (not a method) because it's needed from both SWEBenchWrapper (_build_apptainer_command) and
# RunOpenHandsAgent (_start_container_command), two different classes in this file.
def _uses_vm_sidecar(dataset_name: Optional[str]) -> bool:
    dataset_name = dataset_name or ""
    return (
        "SWE-Gym" in dataset_name
        or "R2E-Gym" in dataset_name
        or "SWE-smith" in dataset_name
        or "SWE-rebench" in dataset_name  # covers both V1 and V2
    )


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
            swebench_multilingual_setup_dir=SweBenchMultilingualDatasetProcessor(config=self.config).setup(),
            r2e_gym_setup_dir=R2EGymDatasetProcessor(config=self.config).setup(),
            swe_rebench_setup_dir=SWERebenchDatasetProcessor(config=self.config).setup(),
            swesmith_setup_dir=SWESmithDatasetProcessor(config=self.config).setup(),
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

        for key in [
            "prompt_token_ids",
            "generation_token_ids",
            "generation_log_probs",
            "routed_experts",
        ]:
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
        dataset_name = data_point["dataset_name"]

        if isinstance(container_formatters, dict):
            # [CUSTOM-CONTAINER-FORMATTER-DICT 2026-10-21] Assign each row to its category 
            # via it dataset_name. Raise if the category is missing from the 
            # container_formatters dict.
            if "SWE-smith" in dataset_name:
                category = "SWE-smith"
            elif dataset_name == "SWE-rebench-V2":
                category = "SWE-rebench-V2"
            elif "SWE-rebench" in dataset_name:
                category = "SWE-rebench"
            elif "R2E-Gym" in dataset_name:
                category = "R2E-Gym"
            else:
                category = "default"
            if category not in container_formatters:
                raise ValueError(
                    f"No container_formatter entry for dataset_name {dataset_name!r} (resolved "
                    f"category {category!r}, instance_id {instance_id}). container_formatter "
                    f"dict only has keys: {sorted(container_formatters)}. Add an entry for "
                    f"{category!r} to this swe_agents instance's container_formatter config."
                )
            container_formatters = [container_formatters[category]]
        elif isinstance(container_formatters, str):
            container_formatters = [container_formatters]

        # [CUSTOM 2026-10-08] SWE-smith ships ONE SHARED IMAGE PER REPO, not per task instance.
        # The row's `image_name` field uniquely identifies the shared image.
        if "SWE-smith" in data_point["dataset_name"]:
            instance_dict = json.loads(data_point["instance_dict"])
            image_stem = instance_dict["image_name"].split("/")[-1]
            for container_formatter in container_formatters:
                container_path = container_formatter.format(instance_id=image_stem)
                if os.path.exists(container_path):
                    return container_path

                container_dir = os.path.dirname(container_formatter.format(instance_id="dummy"))
                matches = glob.glob(os.path.join(container_dir, f"*{image_stem}*.sif"))
                if not matches:
                    # Hyphen/underscore-insensitive fallback (see the SWE-rebench-V2 branch
                    # below for why this is needed, e.g. "kotlin-jdk-11" vs "kotlin_jdk_11").
                    matches = _find_sif_normalized(container_dir, image_stem)
                if matches:
                    return matches[0]
            raise FileNotFoundError(
                f"No SIF found for SWE-smith repo image {image_stem!r} (instance_id {instance_id}). "
                f"Searched directories: {[os.path.dirname(cf.format(instance_id='dummy')) for cf in container_formatters]}"
            )

        # [CUSTOM-SWEREBENCH-V2 2026-10-14] Per-instance image lookup. The filename
        # convention is `swerebenchv2-<org>-<repo>-<issue>-<commit_short>.sif`, e.g.
        # `swerebenchv2-open-telemetry-opamp-go-201-146b792.sif`. We glob on repo+issue,
        # then verify the match's commit prefix against our own `base_commit` to avoid a
        # wrong-commit false positive if a repo is ever re-tagged.
        if data_point["dataset_name"] == "SWE-rebench-V2":
            instance_dict = json.loads(data_point["instance_dict"])
            repo = instance_dict.get("repo", "")
            base_commit = instance_dict.get("base_commit", "")
            org_repo_key = repo.replace("/", "-").lower()
            issue_match = re.search(r"-(\d+)$", instance_id)
            issue = issue_match.group(1) if issue_match else None
            for container_formatter in container_formatters:
                container_dir = os.path.dirname(container_formatter.format(instance_id="dummy"))
                pattern = os.path.join(container_dir, f"swerebenchv2-{org_repo_key}-{issue}-*.sif")
                for candidate in glob.glob(pattern):
                    commit_suffix = os.path.basename(candidate)[: -len(".sif")].rsplit("-", 1)[-1]
                    if base_commit.startswith(commit_suffix):
                        return candidate
            raise FileNotFoundError(
                f"No per-instance SIF found for SWE-rebench-V2 {repo!r} issue {issue!r} "
                f"(instance_id {instance_id}, base_commit {base_commit[:10]}). "
                f"Searched directories: {[os.path.dirname(cf.format(instance_id='dummy')) for cf in container_formatters]}"
            )

        if "SWE-rebench" in data_point["dataset_name"]:
            # [CUSTOM-SWEREBENCH-V1 2026-10-14] Some images use the same "_1776_"
            # SWE-bench-mirror naming convention as SWE-smith's images, e.g.
            # `swerebench-sweb.eval.x86_64.conan-io_1776_conan-12086.sif`.
            # Reuse the same "__" -> "_1776_" substitution the generic candidate_ids
            # fallback below already does for other datasets.
            instance_id_1776 = instance_id.replace("__", "_1776_")
            # [CUSTOM-SWEREBENCH-V1-CASE-INSENSITIVE 2026-10-16] some images are lowercased
            # on disk. Adding the lowercased instance_id/instance_id_1776 as additional
            # candidates.
            for container_formatter in container_formatters:
                # Exact match: {instance_id}.sif (e.g. badges__shields-4557.sif)
                container_path = container_formatter.format(instance_id=instance_id)
                if os.path.exists(container_path):
                    return container_path

                # Fuzzy match: glob for files containing the instance_id, raw, "_1776_"-ified,
                # or lowercased (disk filenames are always lowercase).
                container_dir = os.path.dirname(container_formatter.format(instance_id="dummy"))
                for pattern in [
                    f"{instance_id}*.sif",
                    f"*{instance_id}*.sif",
                    f"*{instance_id_1776}*.sif",
                    f"*{instance_id_1776.lower()}*.sif",
                ]:
                    matches = glob.glob(os.path.join(container_dir, pattern))
                    if matches:
                        return matches[0]
            raise FileNotFoundError(
                f"No SIF found for SWE-rebench instance {instance_id}. "
                f"Searched directories: {[os.path.dirname(cf.format(instance_id='dummy')) for cf in container_formatters]}"
            )

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

            # [CUSTOM-R2EGYM-SIFPULLKIT-NAMING 2026-10-14] Some other images are pulled
            # and named via their own sif_slug() (e.g. "namanjain12-pillow_final-5dfe5f0f....sif"), 
            # NOT the "<repo>_final_<hash>.sif" convention the transform above expects.
            # Here we use another logic: strip a dotted/colon-bearing registry host, replace
            # every non [a-zA-Z0-9._-] run with a single "-", then try and match.
            instance_dict = json.loads(data_point["instance_dict"])
            docker_image = instance_dict.get("docker_image", "")
            if docker_image:
                ref = docker_image
                parts = ref.split("/")
                if len(parts) >= 2 and ("." in parts[0] or ":" in parts[0]):
                    ref = "/".join(parts[1:])
                slug = re.sub(r"[^a-zA-Z0-9._-]", "-", ref)
                slug = re.sub(r"-+", "-", slug).strip("-") + ".sif"
                for container_formatter in container_formatters:
                    container_dir = os.path.dirname(container_formatter.format(instance_id="dummy"))
                    container_name = os.path.join(container_dir, slug)
                    if os.path.exists(container_name):
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
                if self.config.debug:
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

    # [CUSTOM-VM-SIDECAR 2026-10-27] SWE-Gym, R2E-Gym, SWE-smith, SWE-rebench-V1, and
    # SWE-rebench-V2 now run their apptainer command inside the per-node VM sidecar as
    # real root (see nemo_gym/vm_launcher.py and _uses_vm_sidecar below), so those paths
    # are directly writable. 
    # One small quirk: R2E-Gym's held-out tests (/r2e_tests) must stay invisible to the
    # AGENT, so do an explicit empty-dir bind mount in agent mode.
    def _vm_task_integrity_mount_args(
        self, problem_info: Dict[str, Any], persistent_dir: Path, mode: str
    ) -> List[str]:
        dataset_name = problem_info.get("dataset_name", "") or ""
        if "R2E-Gym" in dataset_name and mode == "agent":
            hide_paths = ["/r2e_tests"]
        else:
            return []

        hide_root = persistent_dir / "vm_hidden_paths" / mode
        mount_args: List[str] = []
        for image_path in hide_paths:
            dest = hide_root / image_path.lstrip("/")
            dest.mkdir(parents=True, exist_ok=True)
            mount_args.append(f"--mount type=bind,src={dest},dst={image_path}")
        return mount_args

    def _build_apptainer_command(
        self, params: SWEBenchWrapperInstanceConfig, command: ExecuteContainerCommandArgs
    ) -> str:
        dataset_path_to_mount = str(params.instance_dataset_path)
        data_point = params.problem_info

        # Fix localhost URLs not working sometimes.
        # [CUSTOM] Best-effort: without --fakeroot, /etc/hosts is the real host file
        # bind-mounted in, and we can't write it as a non-root UID. All container_commands
        # below are "&&"-joined into one script, so letting this fail would abort the entire
        # agent run on line 1 ("agent_script.sh: line 1: /etc/hosts: Permission denied").
        # Adding `|| true` keeps this best-effort without masking real failures in the rest
        # of the chain.
        container_commands = []
        container_commands.append("echo '127.0.0.1 localhost' >/etc/hosts || true")

        # Build mount arguments
        mount_args = [
            f"--mount type=bind,src={params.persistent_dir},dst=/trajectories_mount",
        ]

        openhands_dir = f"{params.openhands_setup_dir}/OpenHands"
        mount_args.extend(
            [
                # Read-only base mounts (parent first)
                f"--mount type=bind,src={openhands_dir},dst=/openhands_setup/OpenHands,ro",
                f"--mount type=bind,src={openhands_dir},dst={openhands_dir},ro",
                f"--mount type=bind,src={openhands_dir}/.eval_sessions,dst=/openhands_setup/OpenHands/.eval_sessions",
                f"--mount type=bind,src={openhands_dir}/.eval_sessions,dst={openhands_dir}/.eval_sessions",
                f"--mount type=bind,src={openhands_dir}/logs,dst=/openhands_setup/OpenHands/logs",
                f"--mount type=bind,src={openhands_dir}/logs,dst={openhands_dir}/logs",
                f"--mount type=bind,src={openhands_dir}/evaluation/oh,dst=/openhands_setup/OpenHands/evaluation/oh",
                f"--mount type=bind,src={openhands_dir}/evaluation/oh,dst={openhands_dir}/evaluation/oh",
            ]
        )
        # /root/dataset/data.jsonl is mounted near the end of this function (search for
        # dataset_path_to_mount below), not here with the other base mounts.

        if params.resolved_user_prompt_template:
            mount_args.append(
                f"--mount type=bind,src={params.resolved_user_prompt_template},dst=/openhands_setup/OpenHands/user_prompt.j2"
            )
        if params.resolved_system_prompt_template:
            mount_args.append(
                f"--mount type=bind,src={params.resolved_system_prompt_template},dst=/openhands_setup/OpenHands/system_prompt.j2"
            )
            mount_args.append(
                f"--mount type=bind,src={params.resolved_system_prompt_template},dst=/openhands_setup/OpenHands/system_prompt_long_horizon.j2"
            )

        miniforge3_path = Path(params.openhands_setup_dir) / "miniforge3"
        mount_args.append(f"--mount type=bind,src={miniforge3_path},dst=/openhands_setup/miniforge3,ro")
        mount_args.append(f"--mount type=bind,src={miniforge3_path},dst={miniforge3_path},ro")

        # Add SWE-bench setup directory mount if available (for evaluation)
        # swe-bench-ext and nv-internal-1 don't use the swebench harness
        if command.mode == "eval" and data_point["dataset_name"] not in ("nv-internal-1", "swe-bench-ext"):
            # Mount the entire setup directory at both /swebench_setup and its original absolute path
            # This is needed because uv venv has hardcoded absolute paths
            mount_args.append(f"--mount type=bind,src={params.swebench_setup_dir},dst=/swebench_setup")
            mount_args.append(f"--mount type=bind,src={params.swebench_setup_dir},dst={params.swebench_setup_dir}")
            # [CUSTOM-VENV-RO 2026-11-17] Shadow just the venv subtree read-only, same pattern
            # as the VM sidecar's block-device mounts ("Each image shadows ONLY its venv/.venv
            # dir, NOT the whole setup dir" -- see _spawn's block_device_mounts). Plain
            # SWE-bench never runs inside the VM sidecar (see _uses_vm_sidecar), so unlike
            # SWE-Gym/R2E-Gym/SWE-smith/SWE-rebench its shared venv has no QEMU-readonly-
            # block-device backstop -- without this, an in-container `pip install` can corrupt
            # it for every later rollout in the job. Can't make the whole setup dir read-only:
            # run_local_evaluation writes its own logs/run_evaluation/<run_id> under
            # {swebench_setup_dir}/SWE-bench before get_run_command's own cp-then-rm-rf copies
            # them out (see that command's cd into /swebench_setup/SWE-bench).
            swebench_venv = f"{params.swebench_setup_dir}/SWE-bench/venv"
            mount_args.append(f"--mount type=bind,src={swebench_venv},dst=/swebench_setup/SWE-bench/venv,ro")
            mount_args.append(f"--mount type=bind,src={swebench_venv},dst={swebench_venv},ro")

        if command.mode == "eval" and "SWE-bench_Multilingual" in data_point["dataset_name"]:
            mount_args.append(
                f"--mount type=bind,src={params.swebench_multilingual_setup_dir},dst=/swebench_multilingual_setup"
            )
            mount_args.append(
                f"--mount type=bind,src={params.swebench_multilingual_setup_dir},dst={params.swebench_multilingual_setup_dir}"
            )
            # [CUSTOM-VENV-RO 2026-11-17] See the matching comment on the plain-SWE-bench venv
            # mount above -- same reasoning, same shape.
            swebench_ml_venv = f"{params.swebench_multilingual_setup_dir}/SWE-bench_Multilingual/venv"
            mount_args.append(
                f"--mount type=bind,src={swebench_ml_venv},dst=/swebench_multilingual_setup/SWE-bench_Multilingual/venv,ro"
            )
            mount_args.append(f"--mount type=bind,src={swebench_ml_venv},dst={swebench_ml_venv},ro")

        if command.mode == "eval" and data_point["dataset_name"] == "nv-internal-1":
            run_script_path = params.persistent_dir / "run_script.sh"
            parsing_script_path = params.persistent_dir / "parsing_script.py"

            # Placeholder needed: eval container starts before agent writes the patch
            params.model_patch_path.write_text("")

            mount_args.append(f"--mount type=bind,src={run_script_path},dst=/root/run_script.sh")
            mount_args.append(f"--mount type=bind,src={parsing_script_path},dst=/root/parsing_script.py")
            mount_args.append(f"--mount type=bind,src={params.model_patch_path},dst=/root/patch.diff")

        if command.mode == "eval" and "R2E-Gym" in data_point["dataset_name"]:
            # Mount the entire setup directory at both /r2egym_setup and its original absolute path
            # This is needed because uv venv has hardcoded absolute paths in its wrappers
            # print(f"Mounting R2E-Gym setup directory from: {self.r2e_gym_setup_dir}", flush=True)
            mount_args.append(f"--mount type=bind,src={params.r2e_gym_setup_dir},dst=/r2egym_setup")
            mount_args.append(f"--mount type=bind,src={params.r2e_gym_setup_dir},dst={params.r2e_gym_setup_dir}")
            container_commands.append("mkdir -p /root/r2e_tests && cp -a /r2e_tests/. /root/r2e_tests/ || true")

        if command.mode == "eval" and "SWE-smith" in data_point["dataset_name"]:
            # Mount the entire setup directory (venv + our own driver script, see
            # SWESmithDatasetProcessor.setup()) at both /swesmith_setup and its original
            # absolute path -- same uv-hardcoded-absolute-paths reasoning as R2E-Gym/SWE-bench.
            mount_args.append(f"--mount type=bind,src={params.swesmith_setup_dir},dst=/swesmith_setup")
            mount_args.append(f"--mount type=bind,src={params.swesmith_setup_dir},dst={params.swesmith_setup_dir}")

        if command.mode == "eval" and "SWE-rebench" in data_point["dataset_name"]:
            rebench_setup_dir = params.swe_rebench_setup_dir
            mount_args.append(f"--mount type=bind,src={rebench_setup_dir},dst=/swe_rebench_setup,ro")

            test_patch_path = params.persistent_dir / "test_patch.diff"
            # model_patch_path placeholder needed: eval container starts before agent writes the patch
            if not params.model_patch_path.exists():
                params.model_patch_path.write_text("")
            mount_args.append(f"--mount type=bind,src={test_patch_path},dst=/root/test_patch.diff")
            mount_args.append(f"--mount type=bind,src={params.model_patch_path},dst=/root/patch.diff")

            # Mount eval metadata files explicitly (directory bind mounts may not expose subdirs on Lustre)
            eval_meta_dir = params.persistent_dir / "eval_meta"
            mount_args.append(
                f"--mount type=bind,src={eval_meta_dir / 'expected_passed.json'},dst=/eval_meta/expected_passed.json,ro"
            )
            mount_args.append(
                f"--mount type=bind,src={eval_meta_dir / 'fail_to_pass.json'},dst=/eval_meta/fail_to_pass.json,ro"
            )
            mount_args.append(
                f"--mount type=bind,src={eval_meta_dir / 'pass_to_pass.json'},dst=/eval_meta/pass_to_pass.json,ro"
            )

        if command.mode == "eval" and data_point.get("dataset_name") == "swe-bench-ext":
            test_patch_path = params.persistent_dir / "test_patch.diff"
            if not params.model_patch_path.exists():
                params.model_patch_path.write_text("")
            mount_args.append(f"--mount type=bind,src={test_patch_path},dst=/root/test_patch.diff")
            mount_args.append(f"--mount type=bind,src={params.model_patch_path},dst=/root/patch.diff")

        # [CUSTOM-VM-SIDECAR 2026-10-27] SWE-rebench V1 (not V2) expects its repo checked out at
        # /workspace/<repo>__<version>, not /testbed.
        if "SWE-rebench" in data_point["dataset_name"] and data_point["dataset_name"] != "SWE-rebench-V2":
            instance_dict = json.loads(data_point["instance_dict"])
            workspace_dir_name = f"{instance_dict.get('repo', '')}__{instance_dict.get('version', '')}".replace(
                "/", "__"
            )
            container_commands.append(
                f"mkdir -p /workspace && cp -a /testbed /workspace/{workspace_dir_name} || true"
            )

        if command.mode == "agent" and "R2E-Gym" in data_point["dataset_name"]:
            # Remove R2E-Gym test-related files.
            for root_dir in ["", "/root", "/testbed"]:
                container_commands.append(
                    # /r2e_tests holds evaluation tests the agent should not see. `|| true`:
                    # this rm fails either way -- "Device or resource busy" at the path
                    # _vm_task_integrity_mount_args' empty-dir bind mount is active (/r2e_tests
                    # itself), "No such file or directory" at the others -- but in all cases the
                    # agent never sees real r2e_tests content anyway, so let the step fail
                    # without aborting the rest of the && chain instead of blocking the real
                    # agent command from ever running.
                    f"rm -rf {root_dir}/r2e_tests || true && "
                    # run_tests.sh launches the tests in /r2e_tests, so the agent should not see this either.
                    # We check that it contains the substring "r2e_tests"
                    # to avoid accidentally deleting an unrelated file with that name.
                    f"if grep -qs r2e_tests {root_dir}/run_tests.sh; then rm -rf {root_dir}/run_tests.sh || true; fi"
                )
        container_commands.append(command.command)
        combined_command = " && ".join(container_commands)

        script_dir = params.persistent_dir / "container_scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / f"{command.mode}_script.sh"
        script_path.write_text(combined_command)
        container_script_path = f"/container_scripts/{command.mode}_script.sh"
        mount_args.append(f"--mount type=bind,src={script_path},dst={container_script_path},ro")

        # [CUSTOM-VM-SIDECAR 2026-10-27] see _vm_task_integrity_mount_args -- only R2E-Gym
        # agent-mode's /r2e_tests hide mount lands here. Mode-specific fields kept (agent
        # and eval get independently-computed mounts) even though only one dataset/mode
        # combination populates them today, to avoid agent/eval cross-contamination.
        mount_args.extend(
            params.eval_copy_out_mount_args if command.mode == "eval" else params.copy_out_mount_args
        )

        mount_args.append(f"--mount type=bind,src={dataset_path_to_mount},dst=/root/dataset/data.jsonl")

        mount_str = " ".join(mount_args)

        env_args = ""
        if "SWE-rebench" in data_point["dataset_name"]:
            env_args = "--env _JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false "

        # [CUSTOM-SWEREBENCH-V2-PROXY 2026-10-14, GENERALIZED 2026-10-15] SWE-rebench-V2's eval
        # runs install_config.install (`npm ci`, `pip install`, etc.) inside the container to set
        # up the test environment, which needs real internet access -- forward the proxy env vars.
        for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"):
            proxy_val = os.environ.get(proxy_var)
            if proxy_val:
                env_args += f"--env {proxy_var}={shlex.quote(proxy_val)} "

        # Launch Apptainer container and execute the script file
        #
        # [CUSTOM-CONTAIN-ISOLATION 2026-10-17/19] `--contain` restricts the container's
        # filesystem view to only explicit binds -- without it, Apptainer transparently passes
        # through any host path that exists identically inside and outside the container, so an
        # explicit `,ro` bind to that same path doesn't actually enforce read-only (writes
        # silently succeed and can corrupt shared state, e.g. a rollout's own `pip install`
        # wiping packages from the shared OpenHands venv for every later rollout in the job).
        # `--contain` also means `$HOME` and `/tmp` no longer pass through to the real host
        # filesystem, so both fall onto `--writable-tmpfs`'s tiny 64MB overlay instead --
        # toolchains that write there by default (Go's GOPATH, npm's cache, pip/R scratch files)
        # then hit "no space left on device" mid build/install. Fix: give every container a real
        # per-rollout home directory via `-H`, and a separate real scratch directory bound to
        # `$TMPDIR` for tools that write to `/tmp` instead of `$HOME`.
        #
        # VM-sidecar containers get a guest-local scratch/home dir under /var/tmp instead of the
        # host-created, 9p-shared persistent_dir used below for the host-direct path: writes made
        # as root inside the guest do not propagate real ownership back through the 9p share, so
        # tools that check tmpdir ownership (e.g. pytest) fail against a 9p-backed path. /var/tmp
        # is used rather than /tmp because the latter is RAM-backed (tmpfs) and shared by every
        # concurrent rollout on the node. Nothing reclaims this directory automatically, so it is
        # removed explicitly in the background after each rollout (see below).
        uses_vm_sidecar = _uses_vm_sidecar(data_point["dataset_name"])
        if uses_vm_sidecar:
            guest_scratch_dir = f"/var/tmp/swe_agents_scratch/{params.persistent_dir.name}_{command.mode}"
            guest_home_dir = f"{guest_scratch_dir}_home"
            # Capture apptainer's real exit code into `_rc` before backgrounding the cleanup
            # `rm -rf`: backgrounding as the script's last statement would otherwise reset `$?`
            # to the backgrounding operator's own (always-0) status.
            #
            # `--contain` only gives `-H {guest_home_dir}` an implicit bind; `--env TMPDIR=...`
            # does not get one, so guest_scratch_dir also needs its own explicit same-path bind
            # mount to be visible inside the container.
            apptainer_cmd = (
                f"mkdir -p {guest_scratch_dir} {guest_home_dir} && "
                f"{params.apptainer_binary} exec --contain --writable-tmpfs --cleanenv --pid "
                f"--no-mount tmp,bind-paths -H {guest_home_dir} --env TMPDIR={guest_scratch_dir} "
                f"--mount type=bind,src={guest_scratch_dir},dst={guest_scratch_dir} "
                f"{env_args}"
                f"{mount_str} "
                f" {params.container} bash {container_script_path}"
                f"; _rc=$?"
                f"; nohup timeout 30 rm -rf {guest_scratch_dir} {guest_home_dir} >/dev/null 2>&1 "
                f"& exit $_rc"
            )
        else:
            tmp_scratch_dir = params.persistent_dir / "tmp_scratch"
            tmp_scratch_dir.mkdir(parents=True, exist_ok=True)
            home_scratch_dir = params.persistent_dir / "home_scratch"
            home_scratch_dir.mkdir(parents=True, exist_ok=True)
            apptainer_cmd = (
                f"{params.apptainer_binary} exec --contain --writable-tmpfs --cleanenv --pid "
                f"--no-mount tmp,bind-paths -H {home_scratch_dir} "
                f"--mount type=bind,src={tmp_scratch_dir},dst=/tmp_scratch --env TMPDIR=/tmp_scratch "
                f"{env_args}"
                f"{mount_str} "
                f" {params.container} bash {container_script_path}"
            )
        memory_limit_mb = params.apptainer_memory_limit_mb
        if memory_limit_mb is not None and memory_limit_mb > 0:
            memory_limit_kb = int(memory_limit_mb) * 1024
            apptainer_cmd = f"ulimit -v {memory_limit_kb} && {apptainer_cmd}"

        return apptainer_cmd

    def _resolve_absolute_path(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(PARENT_DIR / p)

    async def _setup_params(
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
        output_for_eval_mounted_path = (
            Path("/trajectories_mount") / "trajectories" / instance_id / "output_for_eval.jsonl"
        )
        output_for_eval_path = trajectories_root / "output_for_eval.jsonl"
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
        # [CUSTOM-R2EGYM-EVAL-ISOLATION 2026-10-09] Two independent copies, not one shared
        # between agent and eval -- see SWEBenchWrapperInstanceConfig.copy_out_mount_args'
        # docstring for the bug this fixes (agent's own untracked scaffold files leaking into
        # eval's supposedly-clean checkout and breaking `git apply` on the model's patch).
        copy_out_mount_args = self._vm_task_integrity_mount_args(problem_info, persistent_dir, "agent")
        eval_copy_out_mount_args = self._vm_task_integrity_mount_args(problem_info, persistent_dir, "eval")

        eval_dir_in_openhands = f"evaluation/oh/{agent_run_id}"
        openhands_config_file_path = f"/tmp/config_{agent_run_id}.toml"

        agent_script_name = f"agent_script_{agent_run_id}.sh"
        agent_script_path = persistent_dir / agent_script_name

        # persistent_dir is mounted here in each container
        base_mounted_dir = Path("/trajectories_mount")

        params: SWEBenchWrapperInstanceConfig = SWEBenchWrapperInstanceConfig(
            **self.config.model_dump(),
            **self._swe_bench_wrapper_server_config.model_dump(),
            problem_info=problem_info,
            body=body,
            persistent_dir=persistent_dir,
            metrics_fpath=persistent_dir / "nemo_gym_metrics.json",
            base_mounted_dir=base_mounted_dir,
            profiling_dir=persistent_dir / "profiling",
            profiling_mounted_dir=base_mounted_dir / "profiling",
            ray_queue_timestamp=time.time(),
            inference_params=inference_params,
            agent_run_id=agent_run_id,
            instance_dataset_path=instance_dataset_path,
            trajectories_root=trajectories_root,
            output_for_eval_mounted_path=output_for_eval_mounted_path,
            output_for_eval_path=output_for_eval_path,
            prediction_path=prediction_path,
            model_patch_path=persistent_dir / "patch.diff",
            container=container,
            copy_out_mount_args=copy_out_mount_args,
            eval_copy_out_mount_args=eval_copy_out_mount_args,
            eval_dir_in_openhands=eval_dir_in_openhands,
            openhands_config_file_path=openhands_config_file_path,
            agent_script_path=agent_script_path,
            final_eval_apptainer_spinup_timestamp_fpath=persistent_dir / "final_eval_apptainer_spinup_timestamp",
            final_eval_apptainer_spinup_timestamp_mounted_fpath=base_mounted_dir
            / "final_eval_apptainer_spinup_timestamp",
            generation_apptainer_spinup_timestamp_fpath=persistent_dir / "generation_apptainer_spinup_timestamp",
            generation_apptainer_spinup_timestamp_mounted_fpath=base_mounted_dir
            / "generation_apptainer_spinup_timestamp",
        )

        params.metrics_fpath.write_text("{}")

        if params.agent_prompt_overrides:
            overrides = params.agent_prompt_overrides
            if params.agent_prompt_override_random:
                selected = random.choice(overrides)
            else:
                rng = random.Random(instance_id)
                selected = rng.choice(overrides)

            params.resolved_user_prompt_template = self._resolve_absolute_path(selected.user_prompt_template)
            params.resolved_system_prompt_template = self._resolve_absolute_path(selected.system_prompt_template)
            params.resolved_agent_cls = selected.agent_cls
            params.resolved_diversify_tool_names = selected.diversify_tool_names

        if params.problem_info["dataset_name"] == "nv-internal-1":
            dataset_processor = NVInternalDatasetProcessor(config=params)
        elif params.problem_info["dataset_name"] == "swe-bench-ext":
            dataset_processor = SweBenchExtDatasetProcessor(config=params)
        elif "SWE-rebench" in params.problem_info["dataset_name"]:
            dataset_processor = SWERebenchDatasetProcessor(config=params)
        elif "R2E-Gym" in params.problem_info["dataset_name"]:
            dataset_processor = R2EGymDatasetProcessor(config=params)
        elif "SWE-smith" in params.problem_info["dataset_name"]:
            dataset_processor = SWESmithDatasetProcessor(config=params)
        elif "SWE-bench_Multilingual" in params.problem_info["dataset_name"]:
            dataset_processor = SweBenchMultilingualDatasetProcessor(config=params)
        else:
            dataset_processor = SweBenchDatasetProcessor(config=params)

        params.eval_command = dataset_processor.get_run_command()
        params.eval_apptainer_command_str = self._build_apptainer_command(params, params.eval_command)

        params.agent_command = OpenHandsHarnessProcessor(config=params).get_run_command()
        params.agent_apptainer_command_str = self._build_apptainer_command(params, params.agent_command)
        params.agent_script = params.agent_script_path.read_text()

        return params, dataset_processor

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        params, dataset_processor = await self._setup_params(body)

        with (params.persistent_dir / "params.json").open("w") as f:
            f.write(params.model_dump_json(indent=4))

        try:
            return await self._inner_responses(params, dataset_processor)
        except Exception as e:
            traceback_file = params.persistent_dir / "traceback.err"
            with traceback_file.open("w") as f:
                f.write(format_exc())

            print(f"Hit an exception in {self.config.name}! See {traceback_file} for more details", file=sys.stderr)

            raise e

    async def _inner_responses(
        self, params: SWEBenchWrapperInstanceConfig, dataset_processor: BaseDatasetHarnessProcessor
    ) -> NeMoGymResponse:
        # options() replaces the decorator's runtime_env: keep py_executable
        maybe_report_file = await runner_ray_remote.options(
            runtime_env={"py_executable": sys.executable, "env_vars": _container_proxy_env()},
        ).remote(params.model_dump())
        metrics_to_update = dict()

        if maybe_report_file:
            dataset_processor.postprocess_after_run(maybe_report_file)

            report = json.loads(Path(maybe_report_file).read_text())
            assert params.instance_id in report, (
                f"Report is malformatted. Expected instance ID key: {params.instance_id}. Report: {report}"
            )
            resolved = report[params.instance_id]["resolved"]
            metrics_to_update["resolved"] = resolved
        else:
            metrics_to_update["resolved"] = False

        # Decide whether to mask this sample from the GRPO gradient.
        # 1) Patch passed eval but agent did not actually submit (hit max-turns
        #    or blew the context window) — the reward is accidental.
        # 2) Final eval step timed out — reward is unreliable.
        # 3) Agent itself timed out (wall-clock) — mask regardless of resolved.
        persisted_metrics = SWEBenchMetrics.model_validate_json(params.metrics_fpath.read_text())
        resolved_now = metrics_to_update.get("resolved", False)
        agent_error_kind = persisted_metrics.agent_error_kind
        eval_timed_out = bool(persisted_metrics.eval_timed_out)
        agent_timed_out = bool(persisted_metrics.agent_timed_out)
        if (
            (resolved_now and agent_error_kind in ("max_iteration", "context_window"))
            or eval_timed_out
            or agent_timed_out
        ):
            params.mask_sample = True

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

        update_metrics(params.metrics_fpath, metrics_to_update)

        return NeMoGymResponse(
            id=f"swebench-{params.instance_id}",
            created_at=int(time.time()),
            model=params.body.model or get_global_config_dict()["policy_model_name"],
            object="response",
            output=output_items,
            parallel_tool_calls=params.body.parallel_tool_calls,
            tool_choice=params.body.tool_choice,
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
            responses_create_params = body.responses_create_params.model_dump() | {
                "input": json.loads(metadata["input"]),
                "tools": [t.model_dump() for t in response.tools] if response.tools else [],
            }
            metrics = SWEBenchMetrics.model_validate_json(metadata["metrics"])

            return SWEBenchVerifyResponse(
                responses_create_params=responses_create_params,
                response=response,
                reward=1.0 if metrics.resolved else 0.0,
                **metrics.model_dump(),
                instance_config=SWEBenchWrapperInstanceConfig.model_validate_json(
                    metadata["instance_config"]
                ).model_dump(),
            )


if __name__ == "__main__":
    SWEBenchWrapper.run_webserver()
