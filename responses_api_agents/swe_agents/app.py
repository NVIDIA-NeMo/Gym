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
import base64
import glob
import hashlib
import importlib.util
import json
import os
import random
import re
import shlex
import shutil
import stat
import sys
import time
import uuid
from asyncio import Semaphore
from asyncio.subprocess import Process
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from shutil import rmtree
from subprocess import Popen
from subprocess import run as subprocess_run
from traceback import format_exc
from typing import Any, Dict, Literal, Optional, Tuple, Union

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
from nemo_gym.server_utils import get_first_server_config_dict
from responses_api_models.vllm_model.app import VLLMConverter, split_responses_input_output_items


########################################
# START Configuration
########################################


# ---------------------------------------------------------------------------
# Model-in-netns bridge (terminal-bench offline service tasks)
# ---------------------------------------------------------------------------
# Offline service tasks (allow_internet=false) run under `apptainer --net
# --network=none` so daemons can bind privileged ports (nginx :80) and parallel
# workers don't collide on the shared host netns. That isolated netns also severs
# the in-SIF opencode agent from the host policy-model endpoint. We carry the model
# in over a bind-mounted unix-domain socket (a filesystem object that crosses the
# netns boundary) bridged with socat on both ends:
#   in-SIF agent -> 127.0.0.1:PORT (inner socat) -> unix sock -> outer socat ->
#   real model server (host:dynport, in the rl-container netns).
# REQUIRED for allow_internet=false: every offline Terminal-Bench task uses the private
# netns and a usable portable socat/ip bundle. allow_internet=true / absent keeps the
# shared host netns. The bundle path is the env override TB_NET_BRIDGE_PKG, else the staged
# default below. Missing or invalid private-network setup is fatal; it must never silently
# widen an offline task back onto the shared network. The model endpoint the agent dials is
# overridden to loopback in OpenCodeHarnessProcessor.
# See logbook P260611 H3 run 2026-06-14-netns-model-bridge (plumbing 5/5, rollout confirmed).
# NOTE: the staged default bundle is arm64 (HSG); other architectures must set
# TB_NET_BRIDGE_PKG to their validated bundle.
_TB_NET_BRIDGE_DEFAULT_PKG = (
    "/lustre/fsw/portfolios/llmservice/users/charlwang/cluster/tb/data/tb_net_bridge/socat_portable"
)


def _tb_private_network_requested(data_point: Dict[str, Any]) -> bool:
    if data_point.get("dataset_name") != "terminal-bench":
        return False
    try:
        instance_dict = json.loads(data_point.get("instance_dict", "{}"))
    except Exception as error:
        raise RuntimeError("Terminal-Bench instance_dict is not valid JSON") from error
    if not isinstance(instance_dict, dict):
        raise RuntimeError("Terminal-Bench instance_dict must be a JSON object")
    return instance_dict.get("allow_internet", True) is False


def _tb_net_bridge_cfg(data_point: Dict[str, Any], agent_run_id: str) -> Optional[Dict[str, Any]]:
    if not _tb_private_network_requested(data_point):
        return None
    pkg = os.environ.get("TB_NET_BRIDGE_PKG") or _TB_NET_BRIDGE_DEFAULT_PKG
    if not (
        (Path(pkg) / "socat").is_file()
        and (Path(pkg) / "ld-linux.so").is_file()
        and (Path(pkg) / "libs").is_dir()
    ):
        raise RuntimeError(
            f"Terminal-Bench private network bridge bundle is missing or invalid: pkg={pkg}"
        )
    # Full executable/content validation runs again on the Ray worker immediately before
    # the outer bridge starts. Keeping this first check structural lets command-only tests
    # use inert placeholders without creating an execution path that can fall back to the
    # shared network.
    try:
        port = int(os.environ.get("TB_NET_BRIDGE_PORT", "18900"))
    except ValueError as error:
        raise RuntimeError("TB_NET_BRIDGE_PORT must be an integer") from error
    if not 1 <= port <= 65535:
        raise RuntimeError(f"TB_NET_BRIDGE_PORT is out of range: {port}")
    tok = hashlib.sha1(agent_run_id.encode()).hexdigest()[:16]
    return {"pkg": pkg, "port": port, "sock": f"/tmp/tbnb/{tok}.sock", "sockdir": "/tmp/tbnb"}


def _tb_validate_net_bridge_bundle(pkg: str) -> None:
    required_executables = [Path(pkg) / name for name in ("socat", "ip", "ld-linux.so")]
    invalid = [
        str(path)
        for path in required_executables
        if not path.is_file() or path.stat().st_size == 0 or not os.access(path, os.X_OK)
    ]
    libs_dir = Path(pkg) / "libs"
    libraries = list(libs_dir.iterdir()) if libs_dir.is_dir() else []
    if invalid or not libraries or any(not path.is_file() or path.stat().st_size == 0 for path in libraries):
        raise RuntimeError(
            "Terminal-Bench private network bridge bundle is missing or invalid: "
            f"pkg={pkg} invalid_executables={invalid} libs={libs_dir}"
        )


def _tb_private_host_dir(persistent_dir: Path) -> Path:
    """Return a host-only sibling outside the agent-visible persistent mount."""
    return persistent_dir.parent / f".{persistent_dir.name}.tb_private"


def _tb_private_tests_host_dir(persistent_dir: Path) -> Path:
    return _tb_private_host_dir(persistent_dir) / "tests"


def _tb_private_verifier_host_dir(persistent_dir: Path) -> Path:
    """Return the eval-only host directory mounted at /logs/verifier."""
    return _tb_private_host_dir(persistent_dir) / "verifier"


def _tb_agent_safe_instance_dict(instance_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remove evaluator-only fields before mounting a TB row into the agent."""
    safe = dict(instance_dict)
    for hidden_key in (
        "test_files",
        "test_files_b64",
        "test_entrypoint",
        "tests_dir",
        "verifier_timeout_sec",
    ):
        safe.pop(hidden_key, None)
    return safe


_TB_TERMINAL_TOOL_OUTPUT = (
    "[terminal tool execution ended before the harness captured an output]"
)


def _append_terminal_tool_outputs(messages: list[dict]) -> list[str]:
    """Pair only dangling tool calls in the final assistant message.

    Some task commands use ``pkill -f`` with a pattern that also matches their
    own command shell. OpenCode then persists the assistant's final tool call
    but cannot persist a tool result before the shell disappears. Preserve the
    event explicitly instead of emitting a structurally orphaned trajectory.
    Earlier unmatched calls are not repaired because appending their results at
    the end would falsify message ordering.
    """
    assistant_indices = [
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]
    if not assistant_indices:
        return []
    final_assistant_index = assistant_indices[-1]
    tail = messages[final_assistant_index + 1 :]
    if any(message.get("role") != "tool" for message in tail):
        return []
    result_ids = {
        message.get("tool_call_id")
        for message in tail
        if message.get("role") == "tool" and message.get("tool_call_id")
    }
    missing = []
    for call in messages[final_assistant_index].get("tool_calls") or []:
        call_id = call.get("id")
        if call_id and call_id not in result_ids:
            missing.append(call_id)
    for call_id in missing:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": _TB_TERMINAL_TOOL_OUTPUT,
            }
        )
    return missing


def _tb_flatten_multipart_content(messages: list[dict]) -> int:
    """Flatten list-form chat content to plain text (terminal-bench only).

    Opencode formats some tool-result notices as a LIST of chat-completions
    parts — e.g. when a tool result carries an image and the policy model is
    text-only, opencode substitutes text parts reading "Attached image(s) from
    tool result:" / "ERROR: Cannot read image (this model does not support
    image input). Inform the user." — but ``NeMoGymEasyInputMessage`` accepts
    only plain-string content, so the Responses conversion raises and 500s the
    whole rollout. Flattening to the concatenated part text records exactly
    what a text-only model consumed, so the trajectory stays faithful.

    NOTE (vision LLMs): this is correct for TEXT-ONLY policies only. If
    NeMo-Gym ever serves a vision policy, opencode will attach real image
    parts and flattening would silently drop them from the recorded
    trajectory — the implementation path must be reconsidered then (proper
    multi-part image support through the Responses conversion, end to end).
    Image-fixture task inventory: P260705 H2 `2026-07-05_image-task-flagging`.
    """
    flattened = 0
    for message in messages:
        content = message.get("content")
        if message.get("role") == "assistant" or not isinstance(content, list):
            continue
        texts = []
        for part in content:
            if isinstance(part, dict):
                value = part.get("text", "")
                texts.append("" if value is None else str(value))
            elif part is not None:
                texts.append(str(part))
        message["content"] = "\n".join(text for text in texts if text)
        flattened += 1
    return flattened


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
    agent_framework: Literal["openhands", "opencode"] = Field(
        default="openhands",
        description="Which agent harness drives the SWE-bench rollout. 'openhands' uses the nv-OpenHands "
        "fork at swe_openhands_setup/. 'opencode' uses the opencode fork at swe_opencode_setup/ via "
        "its bench/ entry point.",
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

    tb_single_exec: bool = Field(
        default=False,
        description=(
            "Terminal-Bench legacy compatibility mode: run agent and eval in one Apptainer exec. "
            "This cannot provide strict hidden-test mount isolation; production collection should "
            "leave it False and use the two-exec shared-overlay flow."
        ),
    )
    tb_service_settle_sec: int = Field(
        default=20, description="Terminal-Bench single-exec: seconds to let agent services settle before eval."
    )
    tb_verifier_timeout_floor_sec: Optional[int] = Field(
        default=None,
        description=(
            "Optional Terminal-Bench-only lower bound for verifier wall time. "
            "The effective timeout is max(instance verifier_timeout_sec, this floor); "
            "useful on slower architectures where the task-declared limit assumes warm installs."
        ),
    )
    opencode_setup_receipt_enforce: bool = Field(
        default=False,
        description=(
            "Opt-in H8/H9 supply-chain preauthorization for the opencode setup: verify "
            "setup_receipt.json against the committed expected-receipt reference (hash-pinned "
            "repo/commit/bun/bundle identities) and rebuild on mismatch. Default False keeps the "
            "original existence-checked setup for every dataset — enforcement hard-pins the "
            "opencode (repo, commit) and would reject configs pinned to moving branches "
            "(e.g. sdd/dev). Only terminal-bench configs should enable this."
        ),
    )

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

    verify_golden_patch: bool = Field(
        default=False,
        description=(
            "If True, skip the agent run and use the sample's golden patch "
            "(instance_dict['patch']) as the model patch."
        ),
    )

    skip_eval: bool = Field(
        default=False,
        description=(
            "If True, run the agent normally but skip the eval container "
            "entirely. The reward is forced to 0 since the patch is never "
            "graded. Useful for collecting agent trajectories without paying "
            "the eval cost."
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

    opencode_subagents_enabled: bool = Field(
        default=False,
        description=(
            "If True (opencode harness only), enable opencode's `task` tool so "
            "the main agent can spawn subagent sessions. Each session's "
            "trajectory is captured to its own `llm_completions/<id>/*.json` "
            "files keyed by sessionID."
        ),
    )


class SWEBenchWrapperServerConfig(BaseModel):
    ng_global_config_dict_str: str
    model_server_name: str
    swebench_setup_dir: Path
    r2e_gym_setup_dir: Path
    swe_rebench_setup_dir: Path
    swebench_multilingual_setup_dir: Path
    run_session_id: str
    base_results_dir: Path
    openhands_setup_dir: Optional[Path] = None
    opencode_setup_dir: Optional[Path] = None


class ExecuteContainerCommandArgs(BaseModel):
    command: str
    expected_file_pattern: str
    mode: Union[Literal["agent"], Literal["eval"], Literal["agent_eval"]]
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
    agent_instance_dataset_path: Optional[Path] = None
    trajectories_root: Path
    prediction_path: Path
    output_for_eval_mounted_path: Path
    output_for_eval_path: Path
    model_patch_path: Path
    container: str
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
    # tb_single_exec (terminal-bench): merged agent+eval command, built in _setup_params
    # where _build_apptainer_command is available (it lives on the wrapper, not the runner).
    merged_command: Optional[ExecuteContainerCommandArgs] = None
    merged_apptainer_command_str: Optional[str] = None
    # Model-in-netns bridge: real model-server "host:port" + portable socat bundle path, set
    # in _setup_params when the bridge is active, so the Ray worker can start the outer socat
    # hop WITHOUT depending on the env var propagating into the worker process.
    tb_bridge_real_hostport: Optional[str] = None
    tb_bridge_pkg: Optional[str] = None

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
    synthetic_terminal_tool_outputs: int = 0

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

    @contextmanager
    def _setup_directory_lock(self, setup_dir: Path, label: str):
        """Cross-node lock using mkdir (atomic on Lustre/NFS, unlike fcntl.flock)."""
        lock_dir = setup_dir.parent
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f".{setup_dir.name}.lockdir"

        print(f"Acquiring {label} setup lock at {lock_path}", flush=True)
        max_wait = 3600
        poll_interval = 5
        waited = 0
        while True:
            try:
                lock_path.mkdir(exist_ok=False)
                break
            except FileExistsError:
                stale_threshold = 3600
                try:
                    lock_age = time.time() - lock_path.stat().st_mtime
                    if lock_age > stale_threshold:
                        print(f"  Lock appears stale ({lock_age:.0f}s old), breaking it", flush=True)
                        shutil.rmtree(lock_path, ignore_errors=True)
                        continue
                except OSError:
                    pass
                if waited >= max_wait:
                    raise TimeoutError(f"Timed out waiting for {label} setup lock after {max_wait}s")
                if waited % 30 == 0:
                    print(
                        f"  Waiting for {label} setup lock (held by another process, {waited}s elapsed)...", flush=True
                    )
                time.sleep(poll_interval)
                waited += poll_interval
        try:
            yield
        finally:
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
            f'date +"%s.%N" > {self.config.final_eval_apptainer_spinup_timestamp_mounted_fpath} && '
            f"{self._get_command_sleep_until_predictions_file()} && "
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

        cmd = f"""#!/bin/bash
set -e

date +\"%s.%N\" > {self.config.final_eval_apptainer_spinup_timestamp_mounted_fpath}

{self._get_command_sleep_until_predictions_file()}

cd /{repo_name}
git reset --hard HEAD

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


# Size of the per-instance ext3 overlay shared between the Terminal-Bench agent
# and eval containers. Holds the agent's writes to the task filesystem so the
# eval phase can grade live state. Tunable; tasks rarely write more than a few
# hundred MB, but uv/pip caches inside run-tests.sh can be sizeable.
TB_OVERLAY_SIZE_MB = 8192


def _validated_tb_tests_dir(raw_tests_dir: Any, workspace_path: str) -> str:
    tests_dir = str(raw_tests_dir or "/tests")
    path = PurePosixPath(tests_dir)
    workspace = PurePosixPath(workspace_path)
    allowed = {PurePosixPath("/tests")}
    normalized_workspace = workspace.is_absolute() and workspace.as_posix() == workspace_path
    if normalized_workspace:
        allowed.add(workspace / "tests")
    destroys_workspace = normalized_workspace and (
        path == workspace or path in workspace.parents
    )
    if (
        not path.is_absolute()
        or path.as_posix() != tests_dir
        or path not in allowed
        or destroys_workspace
    ):
        raise ValueError(
            "terminal-bench tests_dir must be the normalized absolute path /tests "
            f"or <workspace_path>/tests; observed={tests_dir!r} "
            f"workspace_path={workspace_path!r}"
        )
    return path.as_posix()


def _validated_tb_test_entrypoint(raw_entrypoint: Any) -> str:
    entrypoint = str(raw_entrypoint or "test.sh")
    path = PurePosixPath(entrypoint)
    if (
        path == PurePosixPath(".")
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != entrypoint
    ):
        raise ValueError(
            "terminal-bench test_entrypoint must be a normalized relative path; "
            f"observed={entrypoint!r}"
        )
    return path.as_posix()


def _tb_hidden_tests_install_script(
    tests_dir: str,
    test_entrypoint: str,
    *,
    bundle_dir: str = "/root/tb_tests/bundle",
) -> str:
    destination = shlex.quote(tests_dir)
    destination_copy = shlex.quote(tests_dir + "/")
    source_copy = shlex.quote(bundle_dir.rstrip("/") + "/.")
    entrypoint = shlex.quote(str(PurePosixPath(tests_dir) / test_entrypoint))
    return (
        f"rm -rf -- {destination} || exit 1\n"
        f"mkdir -p -- {destination} || exit 1\n"
        f"cp -a -- {source_copy} {destination_copy} || exit 1\n"
        f"chmod +x -- {entrypoint} || exit 1"
    )


class TerminalBenchDatasetProcessor(BaseDatasetHarnessProcessor):
    """Terminal-Bench (Harbor-style) tasks — Path 2 (shared-overlay, live-state).

    Unlike the SWE-bench processors there is NO test patch and NO git-diff
    reconstruction. The opencode agent runs under a persistent Apptainer overlay
    (see `_build_apptainer_command` / `_process_single_datapoint_terminal_bench`);
    this eval step runs *sequentially* in a second container that mounts the SAME
    overlay, so it grades the agent's live filesystem. It reproduces the Harbor
    verifier contract: the task's `tests/` bundle (staged in a host-only sibling
    outside `persistent_dir`, mounted read-only at `/root/tb_tests` ONLY in the
    eval container — never in the agent container) is copied to `tests_dir`
    (default `/tests`); the entrypoint (default `test.sh`) is run with CWD =
    `workspace_path` (default `/app`); reward is read from Harbor's
    `/logs/verifier/reward.txt` (falling back to the entrypoint exit code), giving
    a single all-or-nothing 0/1 `resolved`.

    Required `instance_dict` fields:
      - `test_files`: {relative-path-within-the-task-tests-dir: file-contents} —
        the full hidden test bundle, INCLUDING the entrypoint (e.g. `test.sh`).
    Optional:
      - `workspace_path`: in-SIF task dir / CWD for the entrypoint (default `/app`).
      - `tests_dir`: where the bundle is placed in-container (default `/tests`).
      - `test_entrypoint`: script to run, relative to `tests_dir` (default `test.sh`).
      - `verifier_timeout_sec`: eval timeout (default `swebench_tests_timeout`).
    """

    @property
    def _tb_tests_host_dir(self) -> Path:
        return _tb_private_tests_host_dir(self.config.persistent_dir)

    @property
    def _tb_verifier_host_dir(self) -> Path:
        return _tb_private_verifier_host_dir(self.config.persistent_dir)

    def setup(self) -> Path:
        pass

    def get_run_command(self) -> ExecuteContainerCommandArgs:
        instance_dict = json.loads(self.config.problem_info["instance_dict"])
        workspace_path = str(instance_dict.get("workspace_path") or "/app")
        tests_dir = _validated_tb_tests_dir(instance_dict.get("tests_dir"), workspace_path)
        test_entrypoint = _validated_tb_test_entrypoint(instance_dict.get("test_entrypoint"))

        test_files = instance_dict.get("test_files") or {}
        if not isinstance(test_files, dict) or not test_files:
            raise ValueError(
                f"terminal-bench instance {self.config.instance_id} is missing "
                "instance_dict['test_files'] (a dict of rel-path -> contents for the task's tests/ bundle)."
            )
        if test_entrypoint not in test_files:
            raise ValueError(
                f"terminal-bench instance {self.config.instance_id}: test_entrypoint "
                f"'{test_entrypoint}' not present in instance_dict['test_files'] {sorted(test_files)}."
            )

        # Stage the authoritative test bundle on the host. Mounted read-only into
        # the eval container at /root/tb_tests/bundle (see _build_apptainer_command).
        tests_host_dir = self._tb_tests_host_dir
        if tests_host_dir.exists():
            shutil.rmtree(tests_host_dir, ignore_errors=True)
        bundle_dir = tests_host_dir / "bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        for rel_path, contents in test_files.items():
            dest = bundle_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(contents if isinstance(contents, str) else json.dumps(contents))

        # Binary fixtures (reference images, archives, model weights, sqlite DBs,
        # .pt/.npy/...) cannot be inlined as text; they ride in
        # instance_dict['test_files_b64'] (rel-path -> base64) and are written
        # byte-identically here so the staged bundle matches the task's real tests/
        # dir. Gated on the key's presence -> zero impact on rows without it (e.g.
        # the existing afterquery rollouts). Common in terminal tasks across sets.
        test_files_b64 = instance_dict.get("test_files_b64") or {}
        if isinstance(test_files_b64, dict):
            for rel_path, b64 in test_files_b64.items():
                dest = bundle_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(base64.b64decode(b64))

        report_mounted = "/trajectories_mount/eval_results/report.json"
        report_host = self.config.persistent_dir / "eval_results" / "report.json"
        report_host.parent.mkdir(parents=True, exist_ok=True)

        # This directory is mounted only in the eval container. Recreate it
        # before each verification so an agent-created overlay path cannot
        # supply or retain a reward file.
        verifier_host_dir = self._tb_verifier_host_dir
        if verifier_host_dir.exists():
            shutil.rmtree(verifier_host_dir, ignore_errors=True)
        verifier_host_dir.mkdir(parents=True, exist_ok=True)

        ws = shlex.quote(workspace_path)
        td = shlex.quote(tests_dir)
        ep = shlex.quote(test_entrypoint)
        install_hidden_tests = _tb_hidden_tests_install_script(tests_dir, test_entrypoint)
        # Launched sequentially AFTER the agent finishes, so no sleep-until-
        # predictions busy-wait — the overlay already holds the agent's work.
        # No `set -e`: a failing test run is a valid (unresolved) outcome, not a
        # container error, so the script must still exit 0 for _finish_container_command.
        cmd = f"""#!/bin/bash
set -uo pipefail

date +"%s.%N" > {self.config.final_eval_apptainer_spinup_timestamp_mounted_fpath} 2>/dev/null || true

mkdir -p /trajectories_mount/eval_results /logs/verifier {ws}
find /logs/verifier -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} + 2>/dev/null || true
VERIFIER_STARTED_EPOCH=$(date +%s)

# Inject the authoritative hidden test bundle (Harbor mounts tests/ at /tests).
{install_hidden_tests}

# Reproduce the env the Harbor/terminal-bench verifier normally provides, and fix
# apt inside the UNPRIVILEGED Apptainer container. Vendor test.sh scripts assume:
#   - TEST_DIR / T_BENCH_TEST_DIR  -> the mounted tests dir (else `: ${{TEST_DIR:?}}` aborts)
#   - T_BENCH_CONTAINER_LOGS_PATH  -> /logs (LOG_DIR for solution.log etc.)
#   - HOME                         -> writable home for the `curl | sh` uv installer
#   - working apt-get              -> ~57% `apt-get install -y curl` before uv; apt's
#       _apt sandbox user cannot setgroups in an unprivileged container, giving
#       "Could not switch group" / "Method http has died" -> APT::Sandbox::User root.
mkdir -p /etc/apt/apt.conf.d 2>/dev/null && printf 'APT::Sandbox::User "root";\\nAcquire::Retries "3";\\n' > /etc/apt/apt.conf.d/99tb-no-sandbox 2>/dev/null || true
export TEST_DIR={td}
export T_BENCH_TEST_DIR={td}
export T_BENCH_CONTAINER_LOGS_PATH=/logs
export HOME="${{HOME:-/root}}"
export DEBIAN_FRONTEND=noninteractive
export PYTHONDONTWRITEBYTECODE=1

# Run the task's own verifier entrypoint (e.g. test.sh) with CWD = workspace.
cd {ws}
bash {td}/{ep} > /trajectories_mount/eval_results/pytest_stdout.log 2> /trajectories_mount/eval_results/pytest_stderr.log
TB_EXIT=$?

# Prefer a fresh regular Harbor reward.txt from the eval-only verifier mount;
# otherwise fall back to the entrypoint exit code. All-or-nothing: reward 1 /
# exit 0 == resolved.
REWARD=""
REWARD_SOURCE="exit_code"
if [ -f /logs/verifier/reward.txt ] && [ ! -L /logs/verifier/reward.txt ]; then
    REWARD_MTIME=$(stat -c %Y /logs/verifier/reward.txt 2>/dev/null || echo 0)
    REWARD_CANDIDATE=$(tr -dc '0-9' < /logs/verifier/reward.txt 2>/dev/null || echo "")
    if [ "$REWARD_MTIME" -ge "$VERIFIER_STARTED_EPOCH" ] && \
       {{ [ "$REWARD_CANDIDATE" = "0" ] || [ "$REWARD_CANDIDATE" = "1" ]; }}; then
        REWARD="$REWARD_CANDIDATE"
        REWARD_SOURCE="verifier_file"
        cp -f /logs/verifier/reward.txt /trajectories_mount/eval_results/reward.txt
    fi
fi
if [ "$REWARD" = "1" ]; then RESOLVED=true
elif [ "$REWARD" = "0" ]; then RESOLVED=false
elif [ "$TB_EXIT" -eq 0 ]; then RESOLVED=true
else RESOLVED=false; fi
cat > {report_mounted} <<EOF
{{"{self.config.instance_id}": {{"resolved": $RESOLVED, "exit_code": $TB_EXIT, "reward_txt": "$REWARD", "reward_source": "$REWARD_SOURCE", "patch_exists": true}}}}
EOF
"""

        declared_timeout = int(
            instance_dict.get("verifier_timeout_sec") or self.config.swebench_tests_timeout
        )
        timeout_floor = int(self.config.tb_verifier_timeout_floor_sec or 0)
        effective_timeout = max(declared_timeout, timeout_floor)
        return ExecuteContainerCommandArgs(
            command=cmd,
            expected_file_pattern=str(report_host),
            mode="eval",
            timeout=effective_timeout,
        )

    def postprocess_after_run(self, report_file: Path) -> None:
        # The eval container already wrote {instance_id: {resolved, ...}}; just
        # validate/normalize so _inner_responses can read ["resolved"] safely.
        with open(report_file, "r+") as f:
            report = json.loads(f.read())
            entry = report.get(self.config.instance_id, {}) or {}
            entry["resolved"] = bool(entry.get("resolved", False))
            report[self.config.instance_id] = entry
            f.seek(0)
            f.truncate()
            f.write(json.dumps(report, indent=2))


class OpenHandsHarnessProcessor(BaseDatasetHarnessProcessor):
    def _sync_openhands_to_config_commit(self, openhands_dir: Path) -> None:
        """Ensure OpenHands checkout matches config.agent_framework_commit.

        The config is treated as the golden truth. If the local HEAD differs
        from the target commit, this fetches from the remote, discards any
        local changes (tracked modifications and untracked files, while
        preserving gitignored paths like `.venv`), and checks out the target.
        """
        target = self.config.agent_framework_commit

        def _git(*args: str) -> str:
            result = subprocess_run(
                ["git", "-C", str(openhands_dir), *args],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()

        current_commit = _git("rev-parse", "HEAD")
        try:
            resolved_target = _git("rev-parse", "--verify", f"{target}^{{commit}}")
        except Exception:
            resolved_target = None

        if resolved_target and resolved_target == current_commit:
            print(
                f"OpenHands already at config commit {current_commit[:12]} (target={target})",
                flush=True,
            )
            return

        print(
            f"OpenHands commit mismatch: local={current_commit[:12]}, target={target}. "
            f"Syncing to config (discarding local changes)...",
            flush=True,
        )
        _git("fetch", "--all", "--tags", "--prune")
        _git("reset", "--hard", "HEAD")
        _git("clean", "-fd")
        _git("checkout", "--force", target)

        new_commit = _git("rev-parse", "HEAD")
        print(f"OpenHands now at commit {new_commit[:12]} (target={target})", flush=True)

    def setup(self) -> Path:
        setup_dir = self.parent_dir / "swe_openhands_setup"

        with self._setup_directory_lock(setup_dir, "OpenHands"):
            openhands_dir = setup_dir / "OpenHands"
            miniforge_dir = setup_dir / "miniforge3"

            if openhands_dir.exists() and Path(openhands_dir / ".venv" / "bin" / "python").exists():
                print(f"OpenHands already set up at {setup_dir}", flush=True)
                self._sync_openhands_to_config_commit(openhands_dir)
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

        # body.model may be None (replay JSONLs intentionally omit it — the
        # openai_model proxy force-overrides). tomlkit refuses to serialize None,
        # so coerce to an empty string here; the value is unused once the proxy
        # substitutes its configured backend.
        config["llm"]["model"] |= {
            "model": self.config.body.model or "",
            "base_url": "",  # May need to populate this
            "temperature": self.config.inference_params["temperature"],
            "top_p": self.config.inference_params["top_p"],
        }

        config_str = tomlkit.dumps(config)

        eval_dir_in_openhands = self.config.eval_dir_in_openhands
        local_dataset_path = "/root/dataset/data.jsonl"
        config_file_path = self.config.openhands_config_file_path

        assert self.config.openhands_setup_dir is not None, "OpenHands setup directory is not set"

        # REPLAY_MESSAGES_PATH support: if the body.metadata carries `replay_messages`
        # (a JSON-encoded list of OpenAI chat-completion-format messages — same
        # encoding pattern that `instance_dict` follows), dump it to a file under
        # persistent_dir (mounted as /trajectories_mount inside the apptainer) and
        # forward the path as positional arg #18 to run_infer.sh.
        replay_messages_mounted_path = ""
        replay_messages_raw = self.config.problem_info.get("replay_messages")
        if isinstance(replay_messages_raw, str):
            try:
                replay_messages_list = json.loads(replay_messages_raw)
            except json.JSONDecodeError:
                replay_messages_list = None
        else:
            replay_messages_list = replay_messages_raw
        if replay_messages_list:
            replay_messages_host_path = self.config.persistent_dir / "replay_messages.json"
            replay_messages_host_path.write_text(json.dumps(replay_messages_list))
            replay_messages_mounted_path = f"{self.config.base_mounted_dir}/replay_messages.json"

            # The replay file already encodes the original system prompt, but
            # OpenHands' replay_utils.messages_to_replay_events() drops system
            # messages entirely — the agent would otherwise render its OWN
            # system_prompt.j2 for the live continuation, causing drift between
            # the recorded conversation and the resumed one. The replay's system
            # message is the canonical source of truth, so we extract it and pin
            # it as the agent's system prompt UNCONDITIONALLY (overriding any
            # YAML-level agent_prompt_overrides). The standard mount logic in
            # _build_apptainer_command bind-mounts this file at the container's
            # system_prompt.j2 path.
            replay_system_content = None
            for m in replay_messages_list:
                if isinstance(m, dict) and m.get("role") == "system":
                    c = m.get("content")
                    if isinstance(c, str) and c.strip():
                        replay_system_content = c
                        break
            if replay_system_content:
                sp_host_path = self.config.persistent_dir / "replay_system_prompt.j2"
                sp_host_path.write_text(replay_system_content)
                self.config.resolved_system_prompt_template = str(sp_host_path)

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
                "_crypto_fix_dir=$(mktemp -d /tmp/crypto_fix_XXXXXX) && "
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

        # Positional args 13..18 of run_infer.sh. Empty = shell-side default.
        # Build the full slot list, then emit up to the LAST non-empty slot so
        # trailing placeholders are only inserted when a later arg (e.g.
        # REPLAY_MESSAGES_PATH at #18) needs them at the right shift index.
        oh_dir = "/openhands_setup/OpenHands"
        sp_set = self.config.resolved_system_prompt_template is not None
        up_set = self.config.resolved_user_prompt_template is not None
        positional_args = [
            f"{oh_dir}/user_prompt.j2" if up_set else "",  # 13 INSTRUCTION_TEMPLATE_PATH
            f"{oh_dir}/system_prompt.j2" if sp_set else "",  # 14 SYSTEM_PROMPT_PATH
            f"{oh_dir}/system_prompt_long_horizon.j2" if sp_set else "",  # 15 SYSTEM_PROMPT_LONG_HORIZON_PATH
            "",  # 16 N_RUNS (default 1)
            "",  # 17 MODE   (default "swe")
            replay_messages_mounted_path or "",  # 18 REPLAY_MESSAGES_PATH
        ]
        last_set = max((i for i, a in enumerate(positional_args) if a), default=-1)
        for a in positional_args[: last_set + 1]:
            agent_main_cmd += f"    {a} " if a else "    '' "

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


def _resolve_swebench_workspace_dir_name(instance: Dict[str, Any], dataset_name: str) -> str:
    """Workspace directory NAME (no leading slash). Mirrors openhands'
    `_get_swebench_workspace_dir_name` at
    `temp/nv-OpenHands/.../run_infer.py:210`.
    """
    if "SWE-bench-Live" in dataset_name:
        return str(instance.get("instance_id", ""))
    repo = str(instance.get("repo", instance.get("repo_name", "")))
    version = str(instance.get("version", ""))
    return f"{repo}__{version}".replace("/", "__")


def _resolve_opencode_workspace_path(problem_info: Dict[str, Any]) -> str:
    """Resolve the absolute repo path inside the SIF (dataset-aware)."""
    dataset_name = str(problem_info.get("dataset_name", ""))
    instance = _extract_instance_dict(problem_info)

    if dataset_name == "nv-internal-1":
        return "/app"
    if dataset_name == "swe-bench-ext":
        return "/workspace/repo"
    if "SWE-rebench-V2" in dataset_name:
        repo = str(instance.get("repo", problem_info.get("repo", "")))
        repo_name = repo.split("/", 1)[1] if "/" in repo else repo
        return f"/{repo_name}"
    if dataset_name == "terminal-bench":
        # Terminal-Bench task images bake the task files at the Dockerfile
        # WORKDIR (riship's tbench-*.sif use /workspace; Harbor v2 uses /app).
        # The data prep sets workspace_path explicitly; default to /app.
        return str(instance.get("workspace_path") or "/app")
    return "/testbed"


def _extract_instance_dict(problem_info: Dict[str, Any]) -> Dict[str, Any]:
    raw = problem_info.get("instance_dict")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if isinstance(raw, dict):
        return raw
    return {}


_DEFAULT_OPENCODE_USER_PROMPT_TEMPLATE = (
    Path(__file__).parent / "prompts" / "opencode_harness" / "user_prompt.txt"
).read_text()

H8_OPENCODE_SETUP_RECEIPT_SHA256 = (
    "e64f199895d8da7b1a1044f5088fe1de7758f01039b2b4d77f2fc94b40ac69b3"
)
_OPENCODE_EXPECTED_RECEIPT_PATH = (
    Path(__file__).parent / "opencode_setup_receipt_expected.json"
)
H9_OPENCODE_BUN_ARTIFACTS = {
    "aarch64": {
        "size": 101309000,
        "sha256": "67892bb6734ef352f90a964f0a2a680d5f6e6d15bdb5510e56b8065ac4d599ae",
    },
    "arm64": {
        "size": 101309000,
        "sha256": "67892bb6734ef352f90a964f0a2a680d5f6e6d15bdb5510e56b8065ac4d599ae",
    },
    "x86_64": {
        "size": 101814712,
        "sha256": "b29d78892abd5a9398e0700f0cb602f725089602ed1a5082d681c7257b2bf4d0",
    },
}
H9_OPENCODE_BUNDLE_ARTIFACTS = {
    "aarch64": {
        "size": 23260101,
        "sha256": "027862bbde53b41acc2814d8d2b7a48168c7aa74f8a34326d51ab76a6a1c2709",
    },
    "arm64": {
        "size": 23260101,
        "sha256": "027862bbde53b41acc2814d8d2b7a48168c7aa74f8a34326d51ab76a6a1c2709",
    },
    "x86_64": {
        "size": 23260101,
        "sha256": "86cbb97d27389c5098998749e153518b4ea0540716bdb92faa3c7115fda4f70a",
    },
}

OPENCODE_SETUP_RECEIPT_SCHEMA = "h8_opencode_setup_receipt_v2"
OPENCODE_SOURCE_TREE_EXCLUDED_DIRECTORIES = (".bench-build", "node_modules")


def _opencode_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _opencode_file_identity(path: Path, *, relative_path: Optional[str] = None) -> Dict[str, Any]:
    metadata = path.stat()
    identity: Dict[str, Any] = {
        "mode": stat.S_IMODE(metadata.st_mode),
        "size": metadata.st_size,
        "sha256": _opencode_sha256(path),
    }
    if relative_path is None:
        identity["path"] = str(path)
    else:
        identity["relative_path"] = relative_path
    return identity


def _opencode_tree_digest(members: list[dict]) -> str:
    canonical = json.dumps(
        members,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _opencode_tree_identity(
    root: Path,
    *,
    excluded_directories: tuple[str, ...] = (),
) -> Dict[str, Any]:
    if root.is_symlink():
        raise RuntimeError(f"OpenCode tree root must not be a symlink: {root}")
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"OpenCode tree root is missing or invalid: {root}")
    excluded = tuple(sorted(excluded_directories))
    members: list[dict] = []

    def _walk_error(error: OSError) -> None:
        raise error

    for directory, dirnames, filenames in os.walk(
        root, followlinks=False, onerror=_walk_error
    ):
        directory_path = Path(directory)
        dirnames.sort()
        filenames.sort()
        for dirname in list(dirnames):
            path = directory_path / dirname
            if dirname in excluded:
                dirnames.remove(dirname)
                continue
            if path.is_symlink():
                dirnames.remove(dirname)
                members.append(
                    {
                        "relative_path": path.relative_to(root).as_posix(),
                        "type": "symlink",
                        "mode": stat.S_IMODE(path.lstat().st_mode),
                        "target": os.readlink(path),
                    }
                )
        for filename in filenames:
            path = directory_path / filename
            relative_path = path.relative_to(root).as_posix()
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                members.append(
                    {
                        "relative_path": relative_path,
                        "type": "symlink",
                        "mode": stat.S_IMODE(metadata.st_mode),
                        "target": os.readlink(path),
                    }
                )
            elif stat.S_ISREG(metadata.st_mode):
                members.append(
                    {
                        "relative_path": relative_path,
                        "type": "file",
                        "mode": stat.S_IMODE(metadata.st_mode),
                        "size": metadata.st_size,
                        "sha256": _opencode_sha256(path),
                    }
                )
            else:
                raise RuntimeError(f"unsupported OpenCode tree member: {path}")
    members.sort(key=lambda record: record["relative_path"])
    return {
        "root": str(root),
        "excluded_directories": list(excluded),
        "member_count": len(members),
        "sha256": _opencode_tree_digest(members),
        "members": members,
    }


def _build_opencode_setup_receipt(
    opencode_dir: Path,
    bun_binary: Path,
    agent_framework_repo: str,
    agent_framework_commit: str,
) -> Dict[str, Any]:
    opencode_dir = opencode_dir.resolve()
    bun_binary = bun_binary.resolve()
    source_commit_path = opencode_dir / ".source_commit"
    if source_commit_path.read_text() != agent_framework_commit + "\n":
        raise RuntimeError("OpenCode .source_commit does not match the configured commit")
    build_dir = opencode_dir / ".bench-build"
    receipt = {
        "schema_version": OPENCODE_SETUP_RECEIPT_SCHEMA,
        "agent_framework_repo": agent_framework_repo,
        "agent_framework_commit": agent_framework_commit,
        "source_commit": _opencode_file_identity(
            source_commit_path, relative_path=".source_commit"
        ),
        "bun": _opencode_file_identity(bun_binary),
        "source_tree": _opencode_tree_identity(
            opencode_dir,
            excluded_directories=OPENCODE_SOURCE_TREE_EXCLUDED_DIRECTORIES,
        ),
        "build_tree": _opencode_tree_identity(build_dir),
    }
    build_paths = {
        record["relative_path"]
        for record in receipt["build_tree"]["members"]
        if record.get("type") == "file"
    }
    missing_bundles = {"bench-cli.js", "opencode.js"} - build_paths
    if missing_bundles:
        raise RuntimeError(f"OpenCode build tree is incomplete: {sorted(missing_bundles)}")
    return receipt


def _opencode_tree_record_is_valid(
    tree: Any,
    *,
    expected_excluded_directories: tuple[str, ...],
) -> bool:
    if not isinstance(tree, dict) or set(tree) != {
        "root",
        "excluded_directories",
        "member_count",
        "sha256",
        "members",
    }:
        return False
    if not isinstance(tree.get("root"), str) or not Path(tree["root"]).is_absolute():
        return False
    if tree.get("excluded_directories") != sorted(expected_excluded_directories):
        return False
    members = tree.get("members")
    if not isinstance(members, list) or not members or tree.get("member_count") != len(members):
        return False
    paths = []
    for record in members:
        if not isinstance(record, dict) or not isinstance(record.get("relative_path"), str):
            return False
        relative = PurePosixPath(record["relative_path"])
        if (
            relative.is_absolute()
            or relative == PurePosixPath(".")
            or ".." in relative.parts
            or relative.as_posix() != record["relative_path"]
        ):
            return False
        paths.append(record["relative_path"])
        if record.get("type") == "file":
            if set(record) != {"relative_path", "type", "mode", "size", "sha256"}:
                return False
            if not isinstance(record.get("mode"), int) or not isinstance(record.get("size"), int):
                return False
            if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
                return False
        elif record.get("type") == "symlink":
            if set(record) != {"relative_path", "type", "mode", "target"}:
                return False
            if not isinstance(record.get("mode"), int) or not isinstance(record.get("target"), str):
                return False
        else:
            return False
    return (
        paths == sorted(paths)
        and len(paths) == len(set(paths))
        and tree.get("sha256") == _opencode_tree_digest(members)
    )


def _opencode_receipt_structure_is_valid(receipt: Any) -> bool:
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema_version",
        "agent_framework_repo",
        "agent_framework_commit",
        "source_commit",
        "bun",
        "source_tree",
        "build_tree",
    }:
        return False
    if receipt.get("schema_version") != OPENCODE_SETUP_RECEIPT_SCHEMA:
        return False
    source_commit = receipt.get("source_commit")
    bun = receipt.get("bun")
    if not isinstance(source_commit, dict) or set(source_commit) != {
        "relative_path",
        "mode",
        "size",
        "sha256",
    }:
        return False
    if source_commit.get("relative_path") != ".source_commit":
        return False
    if not isinstance(bun, dict) or set(bun) != {"path", "mode", "size", "sha256"}:
        return False
    if not isinstance(bun.get("path"), str) or not Path(bun["path"]).is_absolute():
        return False
    for record in (source_commit, bun):
        if not isinstance(record.get("mode"), int) or not isinstance(record.get("size"), int):
            return False
        if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
            return False
    if not _opencode_tree_record_is_valid(
        receipt.get("source_tree"),
        expected_excluded_directories=OPENCODE_SOURCE_TREE_EXCLUDED_DIRECTORIES,
    ) or not _opencode_tree_record_is_valid(
        receipt.get("build_tree"), expected_excluded_directories=()
    ):
        return False
    source_by_path = {
        record["relative_path"]: record for record in receipt["source_tree"]["members"]
    }
    if source_by_path.get(".source_commit") != {
        "type": "file",
        **source_commit,
    }:
        return False
    build_paths = {
        record["relative_path"]
        for record in receipt["build_tree"]["members"]
        if record.get("type") == "file"
    }
    return {"bench-cli.js", "opencode.js"}.issubset(build_paths)


def _opencode_authorized_identity(receipt: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a receipt to the fields preauthorization compares.

    Authorization pins the build INPUTS — repo/commit, the patched source tree,
    and the bun binary — plus the SHAPE of the build tree (member paths/types).
    Bundle bytes (size/sha256) are deliberately excluded: `bun build` output is
    not byte-reproducible across environments (confirmed empirically: identical
    source tree + bun binary + lockfile produced an opencode.js differing by
    ~850 bytes), so byte-pinning outputs only creates false setup failures.
    Receipt↔disk consistency of the actual bundles is still enforced separately
    by `_opencode_receipt_matches_runtime`.
    """
    identity = json.loads(json.dumps(receipt))
    identity["bun"].pop("path")
    identity["source_tree"].pop("root")
    build_tree = identity["build_tree"]
    identity["build_tree"] = {
        "members": sorted(
            (
                {"relative_path": member.get("relative_path"), "type": member.get("type")}
                for member in build_tree.get("members", [])
            ),
            key=lambda member: str(member["relative_path"]),
        ),
        "member_count": build_tree.get("member_count"),
    }
    return identity


def _opencode_receipt_is_authorized(receipt: Any, expected_receipt: Any) -> bool:
    return (
        _opencode_receipt_structure_is_valid(receipt)
        and _opencode_receipt_structure_is_valid(expected_receipt)
        and _opencode_authorized_identity(receipt)
        == _opencode_authorized_identity(expected_receipt)
    )


def _opencode_receipt_matches_runtime(
    receipt: Any,
    opencode_dir: Path,
    bun_binary: Path,
    agent_framework_repo: str,
    agent_framework_commit: str,
) -> bool:
    if not _opencode_receipt_structure_is_valid(receipt):
        return False
    try:
        observed = _build_opencode_setup_receipt(
            opencode_dir,
            bun_binary,
            agent_framework_repo,
            agent_framework_commit,
        )
    except (OSError, RuntimeError):
        return False
    return receipt == observed


def _render_opencode_user_message(
    problem_info: Dict[str, Any],
    workspace_path: str,
    template_override_path: Optional[str] = None,
) -> str:
    """Render the user prompt for the opencode session."""
    if template_override_path:
        try:
            template = Path(template_override_path).read_text()
        except OSError as e:
            print(
                f"[opencode] failed to read user prompt override at {template_override_path}: {e}; "
                f"falling back to default template",
                flush=True,
            )
            template = _DEFAULT_OPENCODE_USER_PROMPT_TEMPLATE
    else:
        template = _DEFAULT_OPENCODE_USER_PROMPT_TEMPLATE

    problem_statement = str(problem_info.get("problem_statement", ""))
    try:
        return template.format(workspace_path=workspace_path, problem_statement=problem_statement)
    except (KeyError, IndexError) as e:
        print(
            f"[opencode] template format error ({e}); rendering with the default template instead",
            flush=True,
        )
        return _DEFAULT_OPENCODE_USER_PROMPT_TEMPLATE.format(
            workspace_path=workspace_path, problem_statement=problem_statement
        )


class OpenCodeHarnessProcessor(BaseDatasetHarnessProcessor):
    """Drives the opencode fork; mirrors OpenHandsHarnessProcessor."""

    def setup(self) -> Path:
        if not self.config.opencode_setup_receipt_enforce:
            return self._setup_without_receipt_verification(self.parent_dir / "swe_opencode_setup")

        # Enforced setups live in their own directory: receipt invalidation
        # rmtree-rebuilds the tree at the pinned commit, and doing that inside
        # the shared swe_opencode_setup cache would poison later default-path
        # runs, whose fork-point existence check would silently reuse the
        # pinned/patched tree instead of their configured commit.
        setup_dir = self.parent_dir / "swe_opencode_setup_preauthorized"

        try:
            expected_bytes = _OPENCODE_EXPECTED_RECEIPT_PATH.read_bytes()
            if hashlib.sha256(expected_bytes).hexdigest() != H8_OPENCODE_SETUP_RECEIPT_SHA256:
                raise RuntimeError("preauthorized OpenCode receipt reference hash mismatch")
            expected_receipt = json.loads(expected_bytes)
        except (OSError, json.JSONDecodeError, TypeError) as error:
            raise RuntimeError(
                f"cannot load preauthorized OpenCode receipt reference: {error}"
            ) from error
        if not _opencode_receipt_structure_is_valid(expected_receipt):
            raise RuntimeError(
                f"preauthorized OpenCode receipt must use {OPENCODE_SETUP_RECEIPT_SCHEMA}"
            )
        if (
            expected_receipt.get("agent_framework_repo")
            != self.config.agent_framework_repo
            or expected_receipt.get("agent_framework_commit")
            != self.config.agent_framework_commit
        ):
            raise RuntimeError(
                "OpenCode repository/commit differs from the preauthorized receipt reference"
            )
        machine = os.uname().machine
        authorized_bun = H9_OPENCODE_BUN_ARTIFACTS.get(machine)
        if authorized_bun is None:
            raise RuntimeError(
                f"no preauthorized OpenCode Bun artifact for architecture {machine}"
            )
        expected_receipt["bun"].update(authorized_bun)
        # Bundle bytes are intentionally NOT overlaid or compared — see
        # _opencode_authorized_identity: authorization pins inputs (source tree,
        # bun, repo/commit) and the build-tree shape; bundle output bytes are
        # not reproducible across environments.

        with self._setup_directory_lock(setup_dir, "opencode"):
            opencode_dir = setup_dir / "opencode"
            bun_dir = setup_dir / "bun"

            bun_binary = bun_dir / "bin" / "bun"
            receipt_path = setup_dir / "setup_receipt.json"

            def _receipt_is_valid(receipt: Any) -> bool:
                return (
                    (opencode_dir / "node_modules").is_dir()
                    and _opencode_receipt_matches_runtime(
                        receipt,
                        opencode_dir,
                        bun_binary,
                        self.config.agent_framework_repo,
                        self.config.agent_framework_commit,
                    )
                    and _opencode_receipt_is_authorized(receipt, expected_receipt)
                )

            try:
                receipt = json.loads(receipt_path.read_bytes())
                receipt_valid = _receipt_is_valid(receipt)
            except (OSError, json.JSONDecodeError, TypeError):
                receipt_valid = False
            if receipt_valid:
                print(f"verified cached opencode setup at {setup_dir}", flush=True)
                return setup_dir

            if setup_dir.exists():
                print(f"discarding unverified opencode setup at {setup_dir}", flush=True)
                shutil.rmtree(setup_dir)

            print(f"Setting up opencode environment at {setup_dir}...", flush=True)
            setup_dir.mkdir(parents=True, exist_ok=True)

            self._run_setup_command(self._opencode_setup_script_command(setup_dir, opencode_dir, bun_dir))
            try:
                rebuilt_bytes = receipt_path.read_bytes()
                rebuilt = json.loads(rebuilt_bytes)
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(f"OpenCode setup did not create a valid receipt: {error}") from error
            if not _receipt_is_valid(rebuilt):
                raise RuntimeError(
                    "OpenCode setup output differs from the preauthorized receipt reference"
                )
            return setup_dir

    def _setup_without_receipt_verification(self, setup_dir: Path) -> Path:
        """Original (fork-point) setup behavior: existence-checked cache, no receipt gate.

        This is the default for every dataset. The H8/H9 receipt preauthorization above is
        opt-in via ``opencode_setup_receipt_enforce`` because it hard-pins the opencode
        (repo, commit) and rmtree-rebuilds unverified caches — behavior only terminal-bench
        configs ask for.
        """
        with self._setup_directory_lock(setup_dir, "opencode"):
            opencode_dir = setup_dir / "opencode"
            bun_dir = setup_dir / "bun"

            opencode_bundle = opencode_dir / ".bench-build" / "opencode.js"
            if (
                (opencode_dir / "node_modules").exists()
                and (bun_dir / "bin" / "bun").exists()
                and opencode_bundle.exists()
            ):
                print(f"opencode already set up at {setup_dir}", flush=True)
                return setup_dir

            print(f"Setting up opencode environment at {setup_dir}...", flush=True)
            setup_dir.mkdir(parents=True, exist_ok=True)

            self._run_setup_command(self._opencode_setup_script_command(setup_dir, opencode_dir, bun_dir))
            return setup_dir

    def _opencode_setup_script_command(self, setup_dir: Path, opencode_dir: Path, bun_dir: Path) -> str:
        script_fpath = self.parent_dir / "setup_scripts/opencode.sh"
        return (
            f"SETUP_DIR={setup_dir} "
            f"OPENCODE_DIR={opencode_dir} "
            f"BUN_DIR={bun_dir} "
            f"AGENT_FRAMEWORK_REPO={self.config.agent_framework_repo} "
            f"AGENT_FRAMEWORK_COMMIT={self.config.agent_framework_commit} "
            f"bash {script_fpath}"
        )

    def get_run_command(self) -> ExecuteContainerCommandArgs:
        data_point = self.config.problem_info
        agent_run_id = self.config.agent_run_id

        eval_dir_in_opencode = self.config.eval_dir_in_openhands
        local_dataset_path = "/root/dataset/data.jsonl"
        config_file_path = self.config.openhands_config_file_path

        assert self.config.opencode_setup_dir is not None, (
            "opencode setup directory is not set; agent_framework='opencode' requires that "
            "OpenCodeHarnessProcessor.setup() ran in model_post_init."
        )

        # openai_model.yaml uses `openai_model`; vllm_model.yaml uses `model`.
        try:
            model_server_cfg = get_first_server_config_dict(get_global_config_dict(), self.config.model_server_name)
            model_server_base_url = f"http://{model_server_cfg.host}:{model_server_cfg.port}"
            default_model_name = (
                getattr(model_server_cfg, "openai_model", None) or getattr(model_server_cfg, "model", None) or ""
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not resolve model server '{self.config.model_server_name}' for opencode bench: {e}"
            )

        # Model-in-netns bridge: for offline service tasks the agent runs in a private
        # netns (--network=none) and reaches the model over a unix-socket socat bridge, so
        # override the endpoint it dials to loopback (the inner socat listens there). The
        # REAL server host:port is recorded onto params in _setup_params (for the outer hop).
        # The runner starts the outer socat hop around the agent in both the
        # two-exec production path and the legacy single-exec path.
        _tb_bridge = _tb_net_bridge_cfg(data_point, self.config.agent_run_id)
        if _tb_bridge is not None:
            model_server_base_url = f"http://127.0.0.1:{_tb_bridge['port']}"

        # Falls back to the policy model so opencode doesn't POST model=default.
        effective_model = self.config.body.model or default_model_name
        config_str = json.dumps({"llm": {"model": {"model": effective_model}}})

        workspace_path = _resolve_opencode_workspace_path(data_point)
        user_message = _render_opencode_user_message(
            data_point,
            workspace_path,
            template_override_path=self.config.resolved_user_prompt_template,
        )
        user_message_host_path = self.config.persistent_dir / f"user_message_{agent_run_id}.txt"
        user_message_host_path.write_text(user_message)
        user_message_in_sif = "/opencode_setup/opencode/user_message.txt"

        # Dataset-aware env activation before launching the agent. Activating
        # in the parent shell means the PATH / VIRTUAL_ENV / CONDA_DEFAULT_ENV
        # propagate down through run_infer.sh -> bun -> opencode's bash tool.
        # Wrapped in `|| true` so a missing env doesn't kill the rollout.
        dataset_name = str(data_point.get("dataset_name", ""))
        if "SWE-Gym" in dataset_name:
            # SWE-Gym: deactivate any active venv, then activate conda testbed.
            conda_activate_cmd = (
                "{ deactivate >/dev/null 2>&1 || true; unset VIRTUAL_ENV; "
                "if [ -d /opt/miniconda3 ]; then "
                ". /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed || true; "
                "fi; } && "
            )
        elif "R2E-Gym" in dataset_name:
            # R2E-Gym: deactivate any active venv, then source the bundled venv.
            conda_activate_cmd = (
                "{ deactivate >/dev/null 2>&1 || true; unset VIRTUAL_ENV; "
                "if [ -f /testbed/.venv/bin/activate ]; then "
                ". /testbed/.venv/bin/activate || true; "
                "fi; } && "
            )
        elif dataset_name in ("nv-internal-1", "swe-bench-ext") or "SWE-rebench-V2" in dataset_name:
            # These SIFs ship with the right interpreter already on PATH.
            conda_activate_cmd = ""
        else:
            # Default SWE-bench (Verified/Lite/Multilingual/Live): conda
            # testbed env at /opt/miniconda3 holds the per-instance Python.
            conda_activate_cmd = (
                "if [ -d /opt/miniconda3 ]; then "
                ". /opt/miniconda3/etc/profile.d/conda.sh "
                "&& conda activate testbed || true; "
                "fi && "
            )

        agent_main_cmd = (
            "mkdir -p /tmp/ && "
            "export PATH=/opencode_setup/bun/bin:$PATH && "
            "cd /opencode_setup/opencode && "
            f'date +"%s.%N" > {self.config.generation_apptainer_spinup_timestamp_mounted_fpath} && '
            f"export NEMO_GYM_METRICS_FPATH={self.config.base_mounted_dir}/nemo_gym_metrics.json && "
            f"export NEMO_GYM_CONFIG_DICT={self.config.ng_global_config_dict_str} && "
            f"export NEMO_GYM_MODEL_SERVER_NAME={self.config.model_server_name} && "
            f"export NEMO_GYM_MODEL_SERVER_BASE_URL={shlex.quote(model_server_base_url)} && "
            f"export COMMAND_EXEC_TIMEOUT={self.config.command_exec_timeout} && "
            f"export ENABLE_SUBAGENTS={'1' if self.config.opencode_subagents_enabled else '0'} && "
            "export OPENCODE_DISABLE_MODELS_FETCH=1 && "
            "mkdir -p /root/.cache/opencode && "
            "echo '{}' >/root/.cache/opencode/models.json && "
            f"echo {shlex.quote(config_str)} >{config_file_path} && "
            f"{conda_activate_cmd}"
            "./evaluation/benchmarks/swe_bench/scripts/run_infer.sh "
            f"    {self.config.agent_framework_commit} "  # $1: opencode commit
            f"    {self.config.resolved_agent_cls} "  # $2: agent class (informational)
            f"    {self.config.agent_max_turns} "  # $3: max turns
            f"    {data_point['dataset_name']} "  # $4: dataset name
            f"    {data_point['split']} "  # $5: split
            f"    {eval_dir_in_opencode} "  # $6: eval output dir (relative to opencode dir)
            f"    {data_point['instance_id']} "  # $7: selected id
            f"    {local_dataset_path} "  # $8: instance dict path
            f"    {config_file_path} "  # $9: config file
            f"    {shlex.quote(workspace_path)} "  # $10: resolved workspace path inside SIF
            f"    {user_message_in_sif} "  # $11: pre-rendered user message file
        )

        if self.config.resolved_system_prompt_template is not None:
            agent_main_cmd += "    /opencode_setup/opencode/system_prompt.txt "  # $12: system override

        agent_script_name = f"agent_script_{agent_run_id}.sh"
        agent_script_path = self.config.persistent_dir / agent_script_name
        # set +e inside the trap so a failing log-copy (e.g. when $HOME != /root
        # and /root/.local/share/opencode never gets created, as happens with
        # SWE-rebench-V2 Python images) cannot override the script's real exit
        # code under the outer `set -e`.
        opencode_log_trap = (
            "trap '_rc=$?; set +e; "
            "mkdir -p /trajectories_mount/opencode_logs 2>/dev/null; "
            "cp -r /root/.local/share/opencode /trajectories_mount/opencode_logs/xdg 2>/dev/null; "
            "for d in /tmp/bench-*; do "
            '[ -d "$d/data/log" ] && '
            'cp -r "$d/data/log" "/trajectories_mount/opencode_logs/bench_$(basename "$d")" 2>/dev/null; '
            "done; "
            "exit $_rc' EXIT\n"
        )
        with open(agent_script_path, "w") as f:
            f.write("#!/bin/bash\nset -e\n")
            f.write(opencode_log_trap)
            f.write(agent_main_cmd)
            f.flush()
            os.fsync(f.fileno())

        agent_timeout_seconds = self.config.swebench_agent_timeout
        opencode_cmd = (
            f"timeout --signal=TERM --kill-after=30 {agent_timeout_seconds} "
            f"bash /trajectories_mount/{agent_script_name}"
        )

        search_path = os.path.join(
            self.config.opencode_setup_dir / "opencode" / eval_dir_in_opencode,
            "**",
            "output.jsonl",
        )

        return ExecuteContainerCommandArgs(
            command=opencode_cmd,
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

        # Read from whichever harness wrote — they both use the same in-SIF
        # eval_dir_in_openhands path, but the host-side root differs.
        if self.config.agent_framework == "opencode":
            assert self.config.opencode_setup_dir is not None
            eval_dir_on_host = Path(self.config.opencode_setup_dir) / "opencode" / eval_dir_in_openhands
        else:
            assert self.config.openhands_setup_dir is not None
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

        # Recursive glob handles both openhands' 3-dirs-deep path
        # (<eval>/<inst>/<cfg>/<run>/llm_completions/<id>/*.json) and opencode's
        # 2-dirs-deep path (<eval>/<inst>/bench_run/llm_completions/<id>/*.json).
        completion_candidates = glob.glob(
            str(eval_dir_on_host / "**" / "llm_completions" / "*" / "*.json"),
            recursive=True,
        )
        # When subagents are enabled (opencode) we get multiple sessions, each
        # writing its own per-turn JSONs. Group by session_id (from the file
        # payload) and copy each session's most recent turn — that file's
        # `messages` field carries the full cumulative history for the session.
        if completion_candidates:
            latest_per_session: dict[str, str] = {}
            session_mtime: dict[str, float] = {}
            for path_str in completion_candidates:
                sess_id = "main"
                try:
                    with open(path_str, "r") as f:
                        payload = json.load(f)
                    if isinstance(payload, dict) and payload.get("session_id"):
                        sess_id = str(payload["session_id"])
                except (OSError, json.JSONDecodeError):
                    pass
                mtime = os.path.getmtime(path_str)
                if mtime > session_mtime.get(sess_id, -1):
                    session_mtime[sess_id] = mtime
                    latest_per_session[sess_id] = path_str
            for path_str in latest_per_session.values():
                shutil.copy2(path_str, llm_completions_dir / Path(path_str).name)

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
            if active_command.process.returncode is None:
                active_command.process.kill()
                await active_command.process.wait()
            raise ValueError("Command timed out")
        finally:
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
            raise ValueError(
                f"Expected exactly one file matching {command.expected_file_pattern} for {data_point['instance_id']}, "
                f"found {len(pred_files)}."
            )

    async def _kill_active_command(self, active_command: ActiveContainerCommand) -> None:
        if active_command.process.returncode is None:
            active_command.process.kill()
            await active_command.process.wait()
        active_command.log_file.close()

    async def _create_overlay_image(self, overlay_img: Path) -> None:
        """Create the persistent ext3 overlay image shared by the Terminal-Bench
        agent and eval containers. Prefers `apptainer overlay create`; falls back
        to a raw mkfs.ext3 image (apptainer creates the upper/work dirs on first
        use). Runs on the Ray worker, so apptainer is on PATH."""
        overlay_img.parent.mkdir(parents=True, exist_ok=True)
        if overlay_img.exists():
            overlay_img.unlink()

        # Size the overlay from the task's declared storage budget
        # (task.toml [environment] storage_mb, carried in instance_dict), + 2 GB fs
        # headroom. Falls back to TB_OVERLAY_SIZE_MB. A fixed small overlay caused
        # "No space left on device" on storage-heavy tasks (e.g. mteb model downloads).
        size_mb = TB_OVERLAY_SIZE_MB
        try:
            _idict = json.loads(self.config.problem_info.get("instance_dict") or "{}")
            _smb = int(_idict.get("storage_mb") or 0)
            if _smb > 0:
                size_mb = _smb + 2048
        except Exception:
            pass

        create_cmd = f"apptainer overlay create --size {size_mb} {shlex.quote(str(overlay_img))}"
        proc = await asyncio.create_subprocess_shell(
            create_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0 and overlay_img.exists():
            return

        print(
            f"[terminal-bench] 'apptainer overlay create' failed (rc={proc.returncode}): "
            f"{(stdout or b'').decode(errors='replace')[:500]}; falling back to mkfs.ext3",
            flush=True,
        )
        fallback_cmd = (
            f"dd if=/dev/zero of={shlex.quote(str(overlay_img))} bs=1M count={size_mb} status=none && "
            f"mkfs.ext3 -F -q {shlex.quote(str(overlay_img))}"
        )
        proc2 = await asyncio.create_subprocess_shell(
            fallback_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        stdout2, _ = await proc2.communicate()
        if proc2.returncode != 0 or not overlay_img.exists():
            raise RuntimeError(
                f"overlay image creation failed: {(stdout2 or b'').decode(errors='replace')[:500]}"
            )

    async def _tb_start_outer_bridge(self):
        """Model-in-netns bridge OUTER hop: start `socat UNIX-LISTEN:<sock> -> TCP:<real
        model server>` (in the rl-container netns) so the in-SIF inner hop can reach the
        host model over the bind-mounted unix socket. Returns the subprocess, or None if
        the bridge isn't active for this task. Killed via _tb_stop_outer_bridge."""
        # Worker-side: rely ONLY on params (set in _setup_params on the driver) so this does
        # not depend on TB_NET_BRIDGE_PKG propagating into the Ray worker process. The socket
        # name is recomputed deterministically from agent_run_id (matches the inner hop).
        if not _tb_private_network_requested(self.config.problem_info):
            return None
        pkg = self.config.tb_bridge_pkg
        real_hostport = self.config.tb_bridge_real_hostport
        if not pkg or not real_hostport:
            raise RuntimeError(
                "Terminal-Bench private network bridge lacks its bundle or model endpoint"
            )
        _tb_validate_net_bridge_bundle(pkg)
        sockdir = "/tmp/tbnb"
        sock = f"{sockdir}/{hashlib.sha1(self.config.agent_run_id.encode()).hexdigest()[:16]}.sock"
        try:
            os.makedirs(sockdir, exist_ok=True)
            if os.path.exists(sock):
                os.unlink(sock)
            outer_cmd = (
                f"{pkg}/ld-linux.so --library-path {pkg}/libs {pkg}/socat "
                f"UNIX-LISTEN:{shlex.quote(sock)},fork,reuseaddr "
                f"TCP:{real_hostport}"
            )
            proc = await asyncio.create_subprocess_shell(
                outer_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
            )
            await asyncio.sleep(0.5)
            # Fast-fail: if socat exited already (bad bundle/path/bind) don't report success.
            if proc.returncode is not None:
                raise RuntimeError(
                    f"outer socat exited early rc={proc.returncode} for {self.config.instance_id}"
                )
            # Wait for the listening socket to appear before the agent connects through it.
            for _ in range(25):
                if os.path.exists(sock):
                    break
                await asyncio.sleep(0.2)
            if not os.path.exists(sock):
                proc.terminate()
                await proc.wait()
                raise RuntimeError(
                    f"outer socat did not create its Unix socket for {self.config.instance_id}"
                )
            print(
                f"[terminal-bench bridge] outer socat up: {sock} -> {real_hostport}",
                flush=True,
            )
            return proc
        except Exception as e:
            raise RuntimeError(
                f"Terminal-Bench private network bridge start failed for {self.config.instance_id}: {e}"
            ) from e

    async def _tb_stop_outer_bridge(self, proc) -> None:
        # Async + await wait() so the terminated socat is reaped (no zombie accrual across
        # the many datapoints/repeats a single gym process serves).
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        except Exception:
            pass

    async def _process_single_datapoint_terminal_bench(self) -> Optional[Path]:
        """Path 2 orchestration for Terminal-Bench tasks.

        Sequential (NOT the SWE-bench concurrent-launch + busy-wait): run the
        opencode agent under a persistent overlay, then run the eval container
        against the SAME overlay so the task's tests grade the agent's live
        filesystem. The hidden tests are never present in the agent container.
        Unlike the SWE path, we do NOT bail on an empty git patch — terminal-bench
        success can be non-git state.
        """
        instance_id = self.config.instance_id
        metrics = SWEBenchMetrics(ray_queue_time=time.time() - self.config.ray_queue_timestamp)

        # 1) Create the persistent overlay both containers share.
        overlay_img = self.config.persistent_dir / "agent_overlay.img"
        try:
            await self._create_overlay_image(overlay_img)
        except Exception as e:
            print(f"[terminal-bench] overlay creation failed for {instance_id}: {e}", flush=True)
            metrics.patch_exists = False
            update_metrics(self.config.metrics_fpath, metrics.model_dump())
            return None

        # 1b) SINGLE-EXEC path (opt-in via tb_single_exec): run the agent AND the eval
        # in ONE apptainer exec so background services the agent starts (grpc/nginx/
        # pypi/...) stay alive for the test phase (shared PID namespace). The two-exec
        # default kills them between execs. Apptainer `instance` would be cleaner but
        # user namespaces are blocked on the nodes. This legacy path cannot provide
        # strict mount isolation because agent_eval needs the eval test mount for the
        # whole exec; H8 production collection therefore uses the two-exec path.
        if self.config.tb_single_exec and self.config.merged_command is not None:
            # The merged agent+eval command was built in _setup_params (the wrapper has
            # _build_apptainer_command; this runner does not). Just use it here.
            agent_cmd = self.config.agent_command
            merged_cmd = self.config.merged_command
            merged_apptainer = self.config.merged_apptainer_command_str
            metrics.openhands_run_time = -time.time()
            metrics.generation_apptainer_spinup_time = metrics.openhands_run_time
            # Model-in-netns bridge: start the OUTER socat hop before the exec; the in-SIF
            # inner hop connects to the bind-mounted socket. No-op unless the bridge is
            # active for this task. Always torn down in finally.
            _outer = await self._tb_start_outer_bridge()
            try:
                active = await self._start_container_command(merged_cmd, merged_apptainer)
                try:
                    report_file = await self._finish_container_command(active, merged_cmd)
                except Exception as e:
                    print(f"[terminal-bench single-exec] merged command failed for {instance_id}: {e}", flush=True)
                    metrics.openhands_run_time += time.time()
                    metrics.patch_exists = False
                    metrics.agent_timed_out = (
                        metrics.openhands_run_time is not None
                        and metrics.openhands_run_time >= self.config.swebench_agent_timeout
                    )
                    update_metrics(self.config.metrics_fpath, metrics.model_dump())
                    return None
                metrics.openhands_run_time += time.time()
                metrics.patch_exists = True
                # best-effort agent metrics (patch / error) from the agent's output file
                try:
                    agent_out = glob.glob(agent_cmd.expected_file_pattern, recursive=True)
                    if agent_out:
                        out_file = self._openhands_dir_copy_from_host(output_file_path=agent_out[0])
                        with open(out_file, "r") as f:
                            out_dict = json.loads(f.read().strip())
                        metrics.agent_error_kind = _classify_agent_error(out_dict.get("error"))
                        patch = (out_dict.get("test_result") or {}).get("git_patch") or None
                        metrics.model_patch = (patch + "\n") if patch and not patch.endswith("\n") else patch
                except Exception as e:
                    print(f"[terminal-bench single-exec] agent-metrics read failed for {instance_id}: {e}", flush=True)
                update_metrics(self.config.metrics_fpath, metrics.model_dump())
                return report_file
            finally:
                await self._tb_stop_outer_bridge(_outer)

        # 2) Run the opencode agent (sequential; writes land in the overlay).
        metrics.openhands_run_time = -time.time()
        metrics.generation_apptainer_spinup_time = metrics.openhands_run_time
        _outer = await self._tb_start_outer_bridge()
        try:
            try:
                agent_active = await self._start_container_command(
                    self.config.agent_command, self.config.agent_apptainer_command_str
                )
                out_file_in_eval = await self._finish_container_command(agent_active, self.config.agent_command)
                out_file = self._openhands_dir_copy_from_host(output_file_path=out_file_in_eval)
            except Exception as e:
                print(f"[terminal-bench] agent command failed for {instance_id}: {e}", flush=True)
                try:
                    self._openhands_dir_copy_from_host(output_file_path=None)
                except Exception:
                    pass
                metrics.openhands_run_time += time.time()
                metrics.patch_exists = False
                metrics.agent_timed_out = (
                    metrics.openhands_run_time is not None
                    and metrics.openhands_run_time >= self.config.swebench_agent_timeout
                )
                update_metrics(self.config.metrics_fpath, metrics.model_dump())
                return None
        finally:
            await self._tb_stop_outer_bridge(_outer)

        metrics.openhands_run_time += time.time()
        try:
            with open(out_file, "r") as f:
                out_dict = json.loads(f.read().strip())
            metrics.agent_error_kind = _classify_agent_error(out_dict.get("error"))
            patch = (out_dict.get("test_result") or {}).get("git_patch") or None
            metrics.model_patch = (patch + "\n") if patch and not patch.endswith("\n") else patch
        except Exception as e:
            print(f"[terminal-bench] could not read agent output for {instance_id}: {e}", flush=True)

        # 3) Run the eval container against the same overlay (sequential).
        metrics.final_eval_time = -time.time()
        eval_active = await self._start_container_command(
            self.config.eval_command, self.config.eval_apptainer_command_str
        )
        try:
            report_file = await self._finish_container_command(eval_active, self.config.eval_command)
        except Exception as e:
            print(f"[terminal-bench] eval command failed for {instance_id}: {e}", flush=True)
            metrics.final_eval_time += time.time()
            metrics.patch_exists = True
            metrics.eval_timed_out = (
                str(e) == "Command timed out"
                or (
                    metrics.final_eval_time is not None
                    and metrics.final_eval_time >= float(self.config.eval_command.timeout) + 0.05
                )
            )
            update_metrics(self.config.metrics_fpath, metrics.model_dump())
            return None

        metrics.final_eval_time += time.time()
        metrics.patch_exists = True
        update_metrics(self.config.metrics_fpath, metrics.model_dump())
        return report_file

    async def process_single_datapoint(self) -> Optional[Path]:
        if self.config.verify_golden_patch:
            return await self._run_golden_patch_verification()

        if self.config.problem_info.get("dataset_name") == "terminal-bench":
            return await self._process_single_datapoint_terminal_bench()

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
        eval_active_command = (
            None
            if self.config.skip_eval
            else await self._start_container_command(self.config.eval_command, self.config.eval_apptainer_command_str)
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
            if eval_active_command is not None:
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
        metrics.model_patch = patch

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

            if eval_active_command is not None:
                await self._kill_active_command(eval_active_command)

            update_metrics(self.config.metrics_fpath, metrics.model_dump())
            return

        with open(self.config.model_patch_path, "w") as f:
            f.write(patch)

        if self.config.skip_eval:
            # Eval is intentionally skipped — record that the patch exists,
            # leave eval timings unset, and return None so the caller treats
            # this sample as unresolved (reward = 0).
            metrics.final_eval_apptainer_spinup_time = None
            metrics.final_eval_time = None
            update_metrics(self.config.metrics_fpath, metrics.model_dump())
            if self.config.debug:
                profiler.stop()
            return None

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
        dataset_name = self.config.problem_info.get("dataset_name") or ""
        supported = dataset_name == "swe-bench-ext" or "SWE-bench" in dataset_name or "SWE-rebench" in dataset_name
        if not supported:
            raise NotImplementedError(
                "verify_golden_patch is only supported for dataset_name=='swe-bench-ext' "
                "or the SWE-bench / SWE-bench_Multilingual families "
                f"(got {dataset_name!r})."
            )

        raw_instance_dict = self.config.problem_info["instance_dict"]
        instance_dict = json.loads(raw_instance_dict) if isinstance(raw_instance_dict, str) else raw_instance_dict
        golden_patch = instance_dict.get("patch") or ""
        if not golden_patch.strip():
            raise ValueError(f"No golden patch found in instance_dict['patch'] for {instance_id}.")
        if not golden_patch.endswith("\n"):
            golden_patch += "\n"

        metrics = SWEBenchMetrics(ray_queue_time=time.time() - self.config.ray_queue_timestamp)
        metrics.model_patch = golden_patch
        metrics.patch_exists = True

        # Write golden patch where the agent would have written the model patch.
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
        with open(self.config.model_patch_path, "w") as f:
            f.write(golden_patch)

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
        # Only set up the agent harness that's actually selected. Both share the
        # same dataset/eval setup paths.
        openhands_setup_dir, opencode_setup_dir = None, None
        if self.config.agent_framework == "opencode":
            opencode_setup_dir = OpenCodeHarnessProcessor(config=self.config).setup()
        else:
            openhands_setup_dir = OpenHandsHarnessProcessor(config=self.config).setup()

        self._swe_bench_wrapper_server_config = SWEBenchWrapperServerConfig(
            run_session_id=run_session_id,
            base_results_dir=workspace_root / "results" / f"swebench_results_{run_session_id}",
            ng_global_config_dict_str=shlex.quote(OmegaConf.to_yaml(get_global_config_dict())),
            model_server_name=self.config.model_server.name,
            openhands_setup_dir=openhands_setup_dir,
            opencode_setup_dir=opencode_setup_dir,
            swebench_setup_dir=SweBenchDatasetProcessor(config=self.config).setup(),
            swebench_multilingual_setup_dir=SweBenchMultilingualDatasetProcessor(config=self.config).setup(),
            r2e_gym_setup_dir=R2EGymDatasetProcessor(config=self.config).setup(),
            swe_rebench_setup_dir=SWERebenchDatasetProcessor(config=self.config).setup(),
        )

        self._sem = Semaphore(self.config.concurrency)
        self._vllm_converter = VLLMConverter(return_token_id_information=True)

        return super().model_post_init(context)

    ########################################
    # START Results processing logic
    ########################################

    @staticmethod
    def _materialize_trajectory(data: dict) -> tuple[list, list]:
        """Inflate one completion-file payload into (messages, tools)."""
        messages = list(data.get("messages") or [])
        tools = data.get("kwargs", {}).get("tools", [])
        provider_specific_fields = data.get("provider_specific_fields", {})
        try:
            final_assistant_message = data["response"]["choices"][0]["message"]
        except (KeyError, IndexError):
            return messages, tools

        for key in ["prompt_token_ids", "generation_token_ids", "generation_log_probs"]:
            if key in provider_specific_fields:
                final_assistant_message[key] = provider_specific_fields[key]

        if final_assistant_message.get("content") or final_assistant_message.get("tool_calls"):
            messages.append(final_assistant_message)

        return messages, tools

    def get_openhands_trajectory_from_completions(self, trajectories_dir: Path, instance_id: str) -> tuple:
        """Extract the main session's trajectory for the API response.

        When opencode subagents are enabled there are multiple session_id
        files; we return the main session (no parent_session_id). All other
        per-session files stay on disk for offline training pickup.

        Returns (messages, tools, prefix_message_count). `prefix_message_count`
        is the number of chat messages the live model saw on its first call
        (= the replay prefix in chat-completion format, or just system+user for
        non-replay runs). Used by `_inner_responses` to split input/output at
        the live continuation boundary.
        """
        messages, tools = [], []

        completions_dir = trajectories_dir / instance_id / "llm_completions" / instance_id
        if not completions_dir.exists():
            print(f"No llm_completions directory found: {completions_dir}", flush=True)
            return messages, tools, 0

        completion_files = sorted(completions_dir.glob("*.json"))
        if not completion_files:
            print(f"No completion files found in: {completions_dir}", flush=True)
            return messages, tools, 0

        # The FIRST completion file (lex-sorted on the timestamp suffix in the
        # filename) is the agent's first live LLM call. Its `messages` count is
        # the chat-format prefix length — exactly the input/output boundary.
        first_prefix_count = 0
        try:
            with open(completion_files[0], "r") as f:
                first_data = json.load(f)
            first_prefix_count = len(first_data.get("messages") or [])
        except (OSError, json.JSONDecodeError):
            pass

        # Prefer the main session (no parent_session_id). Fall back to the
        # last file if the payload predates session tagging (openhands).
        main_data = None
        for fpath in completion_files:
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if "session_id" in data and data.get("parent_session_id") in (None, ""):
                main_data = data
        if main_data is None:
            with open(completion_files[-1], "r") as f:
                main_data = json.load(f)

        messages, tools = self._materialize_trajectory(main_data)
        return messages, tools, first_prefix_count

    def get_all_session_trajectories_from_completions(self, trajectories_dir: Path, instance_id: str) -> list[dict]:
        """All per-session trajectories on disk (opencode subagent capture).

        Returns one entry per session_id with its full message history, tools,
        and parent_session_id link. Empty list when no session-tagged dumps
        exist (e.g. openhands path).
        """
        out: list[dict] = []
        completions_dir = trajectories_dir / instance_id / "llm_completions" / instance_id
        if not completions_dir.exists():
            return out
        by_session: dict[str, dict] = {}
        for fpath in sorted(completions_dir.glob("*.json")):
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            sess_id = data.get("session_id")
            if not sess_id:
                continue
            by_session[sess_id] = data
        for sess_id, data in by_session.items():
            messages, tools = self._materialize_trajectory(data)
            out.append(
                {
                    "session_id": sess_id,
                    "parent_session_id": data.get("parent_session_id"),
                    "messages": messages,
                    "tools": tools,
                }
            )
        return out

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
        trusted_sif_root = os.environ.get("TB_TRUSTED_SIF_ROOT")
        if data_point.get("dataset_name") == "terminal-bench" and trusted_sif_root:
            root = Path(trusted_sif_root)
            if not root.is_absolute():
                raise FileNotFoundError(
                    f"TB_TRUSTED_SIF_ROOT must be absolute: {trusted_sif_root}"
                )
            if Path(instance_id).name != instance_id or instance_id in {".", ".."}:
                raise FileNotFoundError(
                    f"invalid Terminal-Bench instance_id for trusted SIF lookup: {instance_id}"
                )
            trusted_container = root / f"{instance_id}_amd64.sif"
            if trusted_container.is_file():
                return str(trusted_container)
            raise FileNotFoundError(
                "No exact trusted Terminal-Bench SIF found: "
                f"{trusted_container} (TB_TRUSTED_SIF_ROOT is set; fuzzy fallback disabled)"
            )
        container_formatters = data_point["container_formatter"]

        if isinstance(container_formatters, str):
            container_formatters = [container_formatters]

        if "SWE-rebench" in data_point["dataset_name"]:
            for container_formatter in container_formatters:
                # Exact match: {instance_id}.sif (e.g. badges__shields-4557.sif)
                container_path = container_formatter.format(instance_id=instance_id)
                if os.path.exists(container_path):
                    return container_path

                # Fuzzy match: glob for files containing the instance_id
                container_dir = os.path.dirname(container_formatter.format(instance_id="dummy"))
                for pattern in [
                    f"{instance_id}*.sif",
                    f"*{instance_id}*.sif",
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

    def _build_apptainer_command(
        self, params: SWEBenchWrapperInstanceConfig, command: ExecuteContainerCommandArgs
    ) -> str:
        data_point = params.problem_info
        dataset_path = params.instance_dataset_path
        if (
            data_point.get("dataset_name") == "terminal-bench"
            and command.mode in ("agent", "agent_eval")
            and params.agent_instance_dataset_path is not None
        ):
            dataset_path = params.agent_instance_dataset_path
        dataset_path_to_mount = str(dataset_path)

        # Private-netns config for offline service tasks. The model bridge is only
        # needed during the agent phase; eval boots the modified service state in
        # a fresh isolated container and has no model traffic.
        tb_net = _tb_net_bridge_cfg(data_point, params.agent_run_id)
        tb_bridge = tb_net if command.mode in ("agent", "agent_eval") else None

        # Fix localhost URLs not working sometimes. Under the bridge (--network=none) also map
        # $(hostname) to loopback so sudo/tools that resolve the node hostname don't fail
        # ("unable to resolve host") with no DNS in the private netns. Gated so the shared-netns
        # path (incl. existing rollouts) keeps the exact original /etc/hosts behaviour.
        container_commands = []
        if tb_net is not None:
            container_commands.append('echo "127.0.0.1 localhost $(hostname 2>/dev/null)" >/etc/hosts')
        else:
            container_commands.append("echo '127.0.0.1 localhost' >/etc/hosts")

        # Apptainer uid namespacing makes the eval-image's `chmod` against
        # /var/run/postgresql fail with "Value too large for defined data
        # type". Pre-create the directory and best-effort chown to a
        # postgres user if one exists, so the dataset's later chmod is a
        # no-op against an already-correct path. Each step is independently
        # tolerant of failure (using `;` and trailing `|| true`) — earlier
        # gating on `getent passwd postgres` skipped the mkdir on images
        # that have /var/run/postgresql but no nss postgres entry.
        container_commands.append(
            "(mkdir -p /var/run/postgresql; "
            "chown postgres:postgres /var/run/postgresql 2>/dev/null; "
            "chmod 2775 /var/run/postgresql 2>/dev/null) >/dev/null 2>&1 || true"
        )

        # Redirect Maven Central lookups to the Google mirror to avoid HTTP 429
        # throttling under heavy parallelism. Write configs inline rather than
        # via bind mount because Apptainer's file-bind into a non-existent
        # parent (e.g. /root/.gradle/init.d/) is unreliable, and silently
        # losing the Gradle init script costs hundreds of failed dep fetches.
        #
        # Gradle home is placed under /trajectories_mount (bind-mounted from
        # persistent_dir on Lustre) rather than /root/.gradle — the writable
        # tmpfs overlay has a size cap that the ~150MB Gradle distribution
        # download can blow through, leaving the wrapper unable to create
        # /root/.gradle/wrapper/dists/* (Gradle "could not create parent
        # directory for lock file" errors).
        maven_mirror_dir = Path(__file__).parent / "maven_mirror"
        mvn_settings_path = maven_mirror_dir / "settings.xml"
        gradle_init_path = maven_mirror_dir / "init.gradle"
        if mvn_settings_path.exists() and gradle_init_path.exists():
            mvn_b64 = base64.b64encode(mvn_settings_path.read_bytes()).decode()
            grad_b64 = base64.b64encode(gradle_init_path.read_bytes()).decode()
            # Belt-and-suspenders: write init.gradle to every common
            # $GRADLE_USER_HOME location on the in-container tmpfs. Some
            # images set GRADLE_USER_HOME via gradle.properties or a
            # wrapper script, ignoring our --env override.
            container_commands.append(
                "mkdir -p /root/.m2 /root/.gradle/init.d "
                "/home/gradle/.gradle/init.d "
                "/home/user/.gradle/init.d 2>/dev/null || true"
            )
            container_commands.append(f"echo {mvn_b64} | base64 -d > /root/.m2/settings.xml")
            container_commands.append(
                f"echo {grad_b64} | base64 -d > /root/.gradle/init.d/maven_central_mirror.gradle"
            )
            # Fan-out the init script. Use `cp` from the canonical write so
            # we don't repeat the long base64 string.
            container_commands.append(
                "for d in /home/gradle/.gradle/init.d /home/user/.gradle/init.d; do "
                '[ -d "$d" ] && cp /root/.gradle/init.d/maven_central_mirror.gradle '
                '"$d/maven_central_mirror.gradle" 2>/dev/null; done; true'
            )

        # Chrome wrapper for Karma-based JS tests. Karma's default
        # ChromeHeadless launcher reads CHROME_BIN; we point it at a wrapper
        # that exec's whatever real chrome lives in the image, with the
        # standard apptainer-container flags applied. Karma configs that
        # use `customLaunchers` with hardcoded flag lists override
        # CHROME_BIN and stay unfixable from outside.
        chrome_wrapper = (
            "#!/bin/sh\n"
            "for b in /opt/google/chrome/google-chrome /opt/google/chrome/chrome "
            "/usr/bin/google-chrome-stable /usr/bin/google-chrome "
            "/usr/lib/chromium/chrome /usr/bin/chromium-browser /usr/bin/chromium; do\n"
            '  if [ -x "$b" ] && [ "$(realpath "$b")" != "$(realpath "$0")" ]; then\n'
            '    exec "$b" --no-sandbox --disable-dev-shm-usage "$@"\n'
            "  fi\n"
            "done\n"
            "echo 'chrome-wrapper.sh: no real chrome binary found' >&2\n"
            "exit 1\n"
        )
        chrome_b64 = base64.b64encode(chrome_wrapper.encode()).decode()
        container_commands.append(
            f"echo {chrome_b64} | base64 -d > /tmp/chrome-wrapper.sh && chmod +x /tmp/chrome-wrapper.sh"
        )

        # Build mount arguments
        mount_args = [
            f"--mount type=bind,src={params.persistent_dir},dst=/trajectories_mount",
        ]

        if params.agent_framework == "opencode" and command.mode == "eval":
            mount_args.append(f"--mount type=bind,src={dataset_path_to_mount},dst=/root/dataset/data.jsonl")
        elif params.agent_framework == "opencode" and command.mode in ("agent", "agent_eval"):
            assert params.opencode_setup_dir is not None, "opencode_setup_dir not set"
            opencode_dir = f"{params.opencode_setup_dir}/opencode"
            bun_dir = f"{params.opencode_setup_dir}/bun"
            (Path(opencode_dir) / "evaluation" / "oh").mkdir(parents=True, exist_ok=True)
            # opencode reads SQLite migrations from `<bundle>/../../migration`
            # (packages/opencode/src/storage/db.ts) → /opencode_setup/migration.
            mount_args.extend(
                [
                    f"--mount type=bind,src={opencode_dir},dst=/opencode_setup/opencode,ro",
                    f"--mount type=bind,src={opencode_dir},dst={opencode_dir},ro",
                    f"--mount type=bind,src={opencode_dir}/evaluation/oh,dst=/opencode_setup/opencode/evaluation/oh",
                    f"--mount type=bind,src={opencode_dir}/evaluation/oh,dst={opencode_dir}/evaluation/oh",
                    f"--mount type=bind,src={bun_dir},dst=/opencode_setup/bun,ro",
                    f"--mount type=bind,src={bun_dir},dst={bun_dir},ro",
                    f"--mount type=bind,src={dataset_path_to_mount},dst=/root/dataset/data.jsonl",
                    f"--mount type=bind,src={opencode_dir}/packages/opencode/migration,dst=/opencode_setup/migration,ro",
                ]
            )
            user_message_host = params.persistent_dir / f"user_message_{params.agent_run_id}.txt"
            mount_args.append(
                f"--mount type=bind,src={user_message_host},dst=/opencode_setup/opencode/user_message.txt,ro"
            )
            if params.resolved_system_prompt_template:
                mount_args.append(
                    f"--mount type=bind,src={params.resolved_system_prompt_template},"
                    f"dst=/opencode_setup/opencode/system_prompt.txt,ro"
                )
        else:
            # OpenHands path (default).
            assert params.openhands_setup_dir is not None, "openhands_setup_dir not set"
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
                    # Data
                    f"--mount type=bind,src={dataset_path_to_mount},dst=/root/dataset/data.jsonl",
                ]
            )

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
        if command.mode == "eval" and data_point["dataset_name"] not in (
            "nv-internal-1",
            "swe-bench-ext",
            "terminal-bench",
        ):
            # Mount the entire setup directory at both /swebench_setup and its original absolute path
            # This is needed because uv venv has hardcoded absolute paths
            mount_args.append(f"--mount type=bind,src={params.swebench_setup_dir},dst=/swebench_setup")
            mount_args.append(f"--mount type=bind,src={params.swebench_setup_dir},dst={params.swebench_setup_dir}")

        if command.mode == "eval" and "SWE-bench_Multilingual" in data_point["dataset_name"]:
            mount_args.append(
                f"--mount type=bind,src={params.swebench_multilingual_setup_dir},dst=/swebench_multilingual_setup"
            )
            mount_args.append(
                f"--mount type=bind,src={params.swebench_multilingual_setup_dir},dst={params.swebench_multilingual_setup_dir}"
            )

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

        if command.mode in ("eval", "agent_eval") and data_point.get("dataset_name") == "terminal-bench":
            # Authoritative hidden tests, staged on the host by
            # TerminalBenchDatasetProcessor.get_run_command in a private sibling
            # directory outside the agent-visible /trajectories_mount bind.
            tb_tests_host_dir = _tb_private_tests_host_dir(params.persistent_dir)
            mount_args.append(f"--mount type=bind,src={tb_tests_host_dir},dst=/root/tb_tests,ro")
            tb_verifier_host_dir = _tb_private_verifier_host_dir(params.persistent_dir)
            tb_verifier_host_dir.mkdir(parents=True, exist_ok=True)
            mount_args.append(
                f"--mount type=bind,src={tb_verifier_host_dir},dst=/logs/verifier"
            )

        if command.mode in ("agent", "agent_eval") and data_point.get("dataset_name") == "terminal-bench":
            # Defense-in-depth: if the task SIF baked the hidden tests in, delete
            # them so the agent cannot read them. (Harbor builds the agent image
            # from environment/ only, so normally there are none.) The canonical
            # tests are injected solely in the eval container (mirrors the R2E-Gym
            # test-hiding below).
            tb_idict = json.loads(data_point["instance_dict"])
            tb_ws_q = shlex.quote(str(tb_idict.get("workspace_path") or "/app"))
            tb_td_q = shlex.quote(str(tb_idict.get("tests_dir") or "/tests"))
            container_commands.append(
                f"rm -rf {tb_td_q} {tb_ws_q}/tests {tb_ws_q}/test_outputs.py "
                f"{tb_ws_q}/solution.sh /logs/verifier 2>/dev/null || true"
            )
            # systemctl shim (no systemd in the unprivileged container): map to
            # `service` so tasks that do `systemctl restart nginx` (re)start the daemon.
            container_commands.append(
                "command -v systemctl >/dev/null 2>&1 || { "
                "printf '#!/bin/bash\\ncase \"$1\" in start|stop|restart|reload|try-restart|force-reload|status) "
                "service \"$2\" \"$1\" 2>/dev/null || /etc/init.d/\"$2\" \"$1\" 2>/dev/null || true;; *) true;; esac\\nexit 0\\n' "
                "> /usr/local/bin/systemctl 2>/dev/null && chmod +x /usr/local/bin/systemctl 2>/dev/null; }; true"
            )
            # Let the AGENT run apt-get inside the unprivileged Apptainer container
            # (else apt's _apt sandbox user cannot setgroups -> "Could not switch
            # group"). Many tasks expect the agent to `apt-get install` tooling, as
            # it could in the real (privileged) Terminal-Bench container.
            container_commands.append(
                "mkdir -p /etc/apt/apt.conf.d 2>/dev/null && "
                "printf 'APT::Sandbox::User \"root\";\\nAcquire::Retries \"3\";\\n' "
                "> /etc/apt/apt.conf.d/99tb-no-sandbox 2>/dev/null || true"
            )

        if data_point.get("dataset_name") == "terminal-bench" and tb_net is not None:
            # Give each offline service phase a private netns, bring loopback up,
            # and boot the image runscript from the shared modified overlay. Only
            # the agent phase needs the unix-socket hop to the policy model.
            _ld = f"{tb_net['pkg']}/ld-linux.so --library-path {tb_net['pkg']}/libs"
            container_commands.append(f"{_ld} {tb_net['pkg']}/ip link set lo up 2>/dev/null || true")
            if tb_bridge is not None:
                container_commands.append(
                    f"{_ld} {tb_bridge['pkg']}/socat "
                    f"TCP-LISTEN:{tb_bridge['port']},fork,reuseaddr,bind=127.0.0.1 "
                    f"UNIX-CONNECT:{tb_bridge['sock']} >/tmp/tb_inner_socat.log 2>&1 & sleep 1"
                )
            container_commands.append(
                "if [ -f /.singularity.d/runscript ]; then "
                "( bash /.singularity.d/runscript </dev/null >/tmp/tb_boot.log 2>&1 & ); sleep 5; fi; true"
            )
            mount_args.append(f"--mount type=bind,src={tb_net['pkg']},dst={tb_net['pkg']},ro")
            if tb_bridge is not None:
                mount_args.append(f"--mount type=bind,src={tb_bridge['sockdir']},dst={tb_bridge['sockdir']}")

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

        script_dir = params.persistent_dir / "container_scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / f"{command.mode}_script.sh"
        script_path.write_text(combined_command)
        container_script_path = f"/container_scripts/{command.mode}_script.sh"
        mount_args.append(f"--mount type=bind,src={script_path},dst={container_script_path},ro")

        mount_str = " ".join(mount_args)

        # _JAVA_OPTIONS bundles JVM-wide settings that benefit Maven, Gradle,
        # and Robolectric-using tests:
        #   - preferIPv6Addresses=false: dual-stack networks misroute
        #     Maven/Gradle requests over IPv6 in some apptainer setups.
        #   - robolectric.dependency.repo.url: Robolectric's MavenArtifactFetcher
        #     does its own Maven Central pull bypassing project repositories,
        #     so we redirect it to the Google mirror directly.
        java_options = (
            "-Djava.net.preferIPv6Addresses=false "
            "-Drobolectric.dependency.repo.url=https://maven-central.storage-download.googleapis.com/maven2/"
        )
        env_args = f"--env _JAVA_OPTIONS='{java_options}' "

        # Force Gradle to read init.d from /root/.gradle regardless of the
        # container image's default user. We use the tmpfs path (not the
        # bind-mounted /trajectories_mount) because Gradle's per-instance
        # caches can be tens of thousands of files and quickly exhaust the
        # operator's Lustre inode quota when run across many parallel
        # containers. The init.gradle script is fan-written to multiple
        # candidate home paths a few lines below.
        env_args += "--env GRADLE_USER_HOME=/root/.gradle "

        # Cap CoreCLR heap reservation to 8 GiB so .NET / PowerShell-Pester
        # tests survive the apptainer-imposed `ulimit -v` virtual-memory
        # limit. Without this, CoreCLR's default upfront heap reservation
        # exceeds the v-limit and fails with HRESULT 0x8007000E
        # ("GC heap initialization failed").
        env_args += "--env DOTNET_GCHeapHardLimit=0x200000000 "

        # Point Karma at our Chrome-flags wrapper. Real chrome is at well-known
        # paths inside the eval images; the wrapper exec's the first one it
        # finds with --no-sandbox --disable-dev-shm-usage. Karma reads
        # CHROME_BIN for Chrome/ChromeHeadless launchers and CHROMIUM_BIN
        # for Chromium/ChromiumHeadless launchers — we set both so a karma
        # config using either name routes through the wrapper. Configs that
        # use `customLaunchers` with hardcoded flag lists override these
        # vars and stay unfixable from outside.
        env_args += "--env CHROME_BIN=/tmp/chrome-wrapper.sh "
        env_args += "--env CHROMIUM_BIN=/tmp/chrome-wrapper.sh "

        # Launch Apptainer container and execute the script file.
        #
        # Terminal-Bench (Path 2): use a PERSISTENT per-instance overlay image
        # instead of an ephemeral --writable-tmpfs, so the agent's writes to the
        # task filesystem survive and the sequentially-launched eval container
        # (which mounts the SAME overlay) grades the live state. The overlay image
        # is created on the Ray worker before the agent launches (see
        # _process_single_datapoint_terminal_bench). All other datasets keep the
        # original ephemeral tmpfs overlay (each container fully isolated).
        net_flag = ""
        if data_point.get("dataset_name") == "terminal-bench":
            overlay_img = params.persistent_dir / "agent_overlay.img"
            writable_overlay_flag = f"--overlay {shlex.quote(str(overlay_img))}"
            # Private loopback netns for offline service tasks. During the agent
            # phase the model is carried over the unix-socket bridge; eval only
            # needs isolated service ports and the modified overlay.
            if tb_net is not None:
                net_flag = " --net --network=none"
        else:
            writable_overlay_flag = "--writable-tmpfs"
        apptainer_cmd = (
            f"apptainer exec {writable_overlay_flag}{net_flag} --cleanenv --pid --no-mount home,tmp,bind-paths "
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

    def _maybe_build_replay_messages(self, body: NeMoGymResponseCreateParamsNonStreaming) -> Optional[str]:
        """Convert Responses-format input into a chat-completion JSON string when
        the input carries a prior agent trajectory.

        Returns None for plain seed inputs (system + user only) — there is nothing
        to replay in that case. Returns a JSON-encoded `list[dict]` (chat-completion
        message format) when function_call / function_call_output items are present.
        """
        if self.config.agent_framework != "openhands":
            return None
        input_items = body.input if isinstance(body.input, list) else []

        def _item_type(item) -> Optional[str]:
            if isinstance(item, dict):
                return item.get("type")
            return getattr(item, "type", None)

        has_trajectory = any(_item_type(item) in ("function_call", "function_call_output") for item in input_items)
        if not has_trajectory:
            return None
        ccp = self._vllm_converter.responses_to_chat_completion_create_params(body)
        chat_messages: list[dict] = []
        for m in ccp.messages:
            if hasattr(m, "model_dump"):
                chat_messages.append(m.model_dump(exclude_none=True))
            elif isinstance(m, dict):
                chat_messages.append(m)
        return json.dumps(chat_messages)

    def _setup_params(
        self, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> Tuple[SWEBenchWrapperInstanceConfig, BaseDatasetHarnessProcessor]:
        problem_info = body.metadata | {"container_formatter": self.config.container_formatter}
        instance_id = problem_info.get("instance_id", "unknown")

        # REPLAY_MESSAGES_PATH support (OpenHands harness only): when the request's
        # input carries a prior agent trajectory (function_call / function_call_output
        # items beyond the initial system+user messages), convert the partial
        # Responses-format input to OpenAI chat-completion format here in the gym
        # layer and surface it via problem_info["replay_messages"] (JSON-encoded
        # because metadata is typed as Dict[str, str]). OpenHandsHarnessProcessor
        # then writes it to a file and forwards the path to run_infer.sh as
        # positional arg #18.
        replay_messages_json = self._maybe_build_replay_messages(body)
        if replay_messages_json is not None:
            problem_info = {**problem_info, "replay_messages": replay_messages_json}

        # Create persistent directory for I/O and logs in local workspace
        instance_dir = f"{instance_id}_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
        persistent_dir = self._swe_bench_wrapper_server_config.base_results_dir / instance_dir
        persistent_dir.mkdir(parents=True, exist_ok=True)

        agent_run_id = f"{instance_id}_{int(time.time())}_{str(uuid.uuid4())[:8]}"

        # To avoid HF dataset API calls, write the instance dictionary to a file.
        # Terminal-Bench's authoritative row contains hidden test payloads, so it
        # lives in a host-only sibling rather than the agent-visible persistent bind.
        is_terminal_bench = problem_info.get("dataset_name") == "terminal-bench"
        instance_dataset_dir = (
            _tb_private_host_dir(persistent_dir) / "instance_datasets"
            if is_terminal_bench
            else persistent_dir / "instance_datasets"
        )
        instance_dataset_dir.mkdir(parents=True, exist_ok=True)
        instance_dataset_path = instance_dataset_dir / f"{agent_run_id}.jsonl"
        instance_dict = json.loads(problem_info["instance_dict"])
        if "repo" in instance_dict and "repo_name" not in instance_dict:
            instance_dict["repo_name"] = instance_dict["repo"]
        with open(instance_dataset_path, "w") as f:
            f.write(json.dumps(instance_dict) + "\n")

        agent_instance_dataset_path = None
        if is_terminal_bench:
            agent_instance_dataset_dir = persistent_dir / "agent_instance_datasets"
            agent_instance_dataset_dir.mkdir(parents=True, exist_ok=True)
            agent_instance_dataset_path = agent_instance_dataset_dir / f"{agent_run_id}.jsonl"
            agent_instance_dict = _tb_agent_safe_instance_dict(instance_dict)
            agent_instance_dataset_path.write_text(json.dumps(agent_instance_dict) + "\n")

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
            agent_instance_dataset_path=agent_instance_dataset_path,
            trajectories_root=trajectories_root,
            output_for_eval_mounted_path=output_for_eval_mounted_path,
            output_for_eval_path=output_for_eval_path,
            prediction_path=prediction_path,
            model_patch_path=persistent_dir / "patch.diff",
            container=container,
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

        if params.problem_info["dataset_name"] == "terminal-bench":
            dataset_processor = TerminalBenchDatasetProcessor(config=params)
        elif params.problem_info["dataset_name"] == "nv-internal-1":
            dataset_processor = NVInternalDatasetProcessor(config=params)
        elif params.problem_info["dataset_name"] == "swe-bench-ext":
            dataset_processor = SweBenchExtDatasetProcessor(config=params)
        elif "SWE-rebench" in params.problem_info["dataset_name"]:
            dataset_processor = SWERebenchDatasetProcessor(config=params)
        elif "R2E-Gym" in params.problem_info["dataset_name"]:
            dataset_processor = R2EGymDatasetProcessor(config=params)
        elif "SWE-bench_Multilingual" in params.problem_info["dataset_name"]:
            dataset_processor = SweBenchMultilingualDatasetProcessor(config=params)
        else:
            dataset_processor = SweBenchDatasetProcessor(config=params)

        params.eval_command = dataset_processor.get_run_command()
        params.eval_apptainer_command_str = self._build_apptainer_command(params, params.eval_command)

        if self.config.agent_framework == "opencode":
            params.agent_command = OpenCodeHarnessProcessor(config=params).get_run_command()
        else:
            params.agent_command = OpenHandsHarnessProcessor(config=params).get_run_command()
        params.agent_apptainer_command_str = self._build_apptainer_command(params, params.agent_command)
        params.agent_script = params.agent_script_path.read_text()

        # Legacy tb_single_exec (terminal-bench): build the merged agent+eval command HERE,
        # where the wrapper's _build_apptainer_command is available (RunOpenHandsAgent,
        # which runs the datapoint, does NOT have that method). The merged exec runs the
        # agent then the eval in ONE apptainer exec so the agent's services persist.
        if params.tb_single_exec and params.problem_info.get("dataset_name") == "terminal-bench":
            settle = int(params.tb_service_settle_sec)
            merged_inner = (
                "set +e\n( " + params.agent_command.command + "\n) || true\n"
                f'echo "[tb_single_exec] agent phase done; settling services {settle}s"\n'
                f"sleep {settle}\n" + params.eval_command.command
            )
            params.merged_command = ExecuteContainerCommandArgs(
                command=merged_inner,
                expected_file_pattern=params.eval_command.expected_file_pattern,
                mode="agent_eval",
                timeout=params.swebench_agent_timeout
                + int(params.eval_command.timeout or params.swebench_tests_timeout)
                + 120,
            )
            params.merged_apptainer_command_str = self._build_apptainer_command(params, params.merged_command)

        # Model-in-netns bridge: record the REAL model-server host:port on params (survives
        # the Ray model_dump) so the runner can point the outer socat hop at the live
        # endpoint. The agent's own base_url was overridden to loopback in get_run_command.
        _b = _tb_net_bridge_cfg(params.problem_info, params.agent_run_id)
        if _b is not None:
            params.tb_bridge_pkg = _b["pkg"]
            try:
                _msc = get_first_server_config_dict(get_global_config_dict(), params.model_server_name)
                params.tb_bridge_real_hostport = f"{_msc.host}:{_msc.port}"
            except Exception as _e:
                raise RuntimeError(
                    "Terminal-Bench private network bridge could not resolve model server host:port"
                ) from _e

        return params, dataset_processor

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        params, dataset_processor = self._setup_params(body)

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
        maybe_report_file = await runner_ray_remote.remote(params.model_dump())
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
        chat_completions_trajectory, chat_completions_tools, prefix_msg_count = (
            self.get_openhands_trajectory_from_completions(trajectories_dir, params.instance_id)
        )
        if params.problem_info.get("dataset_name") == "terminal-bench":
            synthetic_terminal_tool_outputs = _append_terminal_tool_outputs(
                chat_completions_trajectory
            )
            metrics_to_update["synthetic_terminal_tool_outputs"] = len(
                synthetic_terminal_tool_outputs
            )
            _tb_flatten_multipart_content(chat_completions_trajectory)

        tools = [
            FunctionTool.model_validate(tool["function"] | {"type": "function"}) for tool in chat_completions_tools
        ]

        # For replay rollouts, the request's `body.input` already carries a partial
        # trajectory. Echo it back verbatim as `input` and isolate only the
        # genuinely new live-continuation messages as `output`. We find the boundary
        # by matching tool_call ids against body.input: every replayed action's
        # call_id appears in body.input, so the live continuation begins immediately
        # after the last chat message that references one of those ids.
        def _item_type(item) -> Optional[str]:
            if isinstance(item, dict):
                return item.get("type")
            return getattr(item, "type", None)

        def _item_field(item, name: str):
            if isinstance(item, dict):
                return item.get(name)
            return getattr(item, name, None)

        body_input = params.body.input if isinstance(params.body.input, list) else None
        has_replay_prefix = bool(
            body_input and any(_item_type(it) in ("function_call", "function_call_output") for it in body_input)
        )
        if has_replay_prefix:
            body_call_ids: set[str] = {
                _item_field(it, "call_id")
                for it in body_input
                if _item_type(it) in ("function_call", "function_call_output") and _item_field(it, "call_id")
            }
            # Walk the chat trajectory and find the last chat-message index whose
            # tool_call.id (assistant) or tool_call_id (tool) is in body_call_ids.
            last_replay_idx = -1
            for i, msg in enumerate(chat_completions_trajectory):
                role = msg.get("role")
                if role == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        if tc.get("id") in body_call_ids:
                            last_replay_idx = i
                            break
                elif role == "tool":
                    if msg.get("tool_call_id") in body_call_ids:
                        last_replay_idx = i
            split_at = last_replay_idx + 1 if last_replay_idx >= 0 else 0
            input_items = list(body_input)
            output_items = self._vllm_converter.chat_completions_messages_to_responses_items(
                chat_completions_trajectory[split_at:]
            )
        else:
            responses_items = self._vllm_converter.chat_completions_messages_to_responses_items(
                chat_completions_trajectory
            )
            input_items, output_items = split_responses_input_output_items(responses_items)

        update_metrics(params.metrics_fpath, metrics_to_update)

        # body.model can be None (replay JSONLs omit it; the openai_model proxy
        # picks the backend). NeMoGymResponse.model is a required non-None string,
        # so fall back to the agent's configured model server name.
        return NeMoGymResponse(
            id=f"swebench-{params.instance_id}",
            created_at=int(time.time()),
            model=params.body.model or self.config.model_server.name,
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
