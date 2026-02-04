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
import fcntl
import json
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from subprocess import Popen
from typing import Dict, Optional

from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
)
from responses_api_agents.swe_agents.run_openhands import (
    RunOpenHandsAgent,
    SweBenchGenerationConfig,
    SweBenchInferenceConfig,
)


def get_openhands_trajectory_from_completions(
    trajectories_dir: Path,
    instance_id: str,
) -> tuple:
    """Get trajectory from llm_completions directory for OpenHands.

    Args:
        trajectories_dir: Trajectories directory
        instance_id: Instance ID

    Returns:
        Tuple of (messages, tools)
    """
    messages = []
    tools = []
    completions_dir = trajectories_dir / instance_id / "llm_completions" / instance_id

    if not completions_dir.exists():
        print(f"No llm_completions directory found: {completions_dir}", flush=True)
        return messages, tools

    completion_files = sorted(completions_dir.glob("*.json"))

    if not completion_files:
        print(f"No completion files found in: {completions_dir}", flush=True)
        return messages, tools

    last_file = completion_files[-1]

    try:
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

        # print(
        #     f"Loaded {len(messages)} messages from last completion file: {last_file}",
        #     flush=True,
        # )

    except Exception as e:
        print(f"Failed to read completion file {last_file}: {e}", flush=True)
        return [], []

    for msg in messages:
        if "content" in msg:
            msg["content"] = msg["content"] or ""
            if isinstance(msg["content"], list):
                # Handle empty content lists (e.g., assistant messages with only tool calls)
                if len(msg["content"]) == 0:
                    msg["content"] = ""
                elif len(msg["content"]) == 1:
                    item = msg["content"][0]
                    if not isinstance(item, dict) or item.get("type") != "text" or "text" not in item:
                        raise ValueError(f"Expected content item to be {{type: 'text', text: '...'}}, got {item}")
                    msg["content"] = item["text"]
                else:
                    raise ValueError(f"Expected 0 or 1 content items, got {len(msg['content'])}")
        else:
            raise ValueError(f"Expected content in message, got {msg}")

    return messages, tools


### Run SWE Harness Utils ###


async def run_swebench_evaluation(
    problem_info: Dict,
    model_endpoint: str,
    body: NeMoGymResponseCreateParamsNonStreaming,
    agent_framework: str,
    agent_config: Optional[str],
    agent_tools_file: Optional[str],
    agent_max_turns: int,
    swebench_tests_timeout: int,
    swebench_agent_timeout: int,
    persistent_dir: Path,
    metrics_fpath: Path,
    ng_global_config_dict_str: str,
    model_server_name: str,
    agent_framework_repo: Optional[str] = None,
    agent_framework_commit: str = "HEAD",
    openhands_setup_dir: Optional[Path] = None,
    swebench_setup_dir: Optional[Path] = None,
    r2e_gym_setup_dir: Optional[Path] = None,
    dataset_path: Optional[str] = None,
    ray_queue_time: Optional[float] = None,
    ray_submit_time: Optional[float] = None,
    openhands_should_log: bool = False,
    debug: bool = False,
    apptainer_memory_limit_mb: Optional[int] = None,
    command_exec_timeout: Optional[int] = None,
) -> Dict:
    instance_id = problem_info.get("instance_id", "unknown")
    output_file = persistent_dir / "output.jsonl"

    inference_params = {}

    for param, key in [
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("max_output_tokens", "tokens_to_generate"),
    ]:
        value = getattr(body, param, None)
        if value is not None:
            inference_params[key] = value

    inference_config = SweBenchInferenceConfig(**inference_params)
    server = {
        "model": body.model,
        "base_url": model_endpoint,
    }

    cfg = SweBenchGenerationConfig(
        output_file=output_file,
        agent_framework_repo=agent_framework_repo,
        agent_framework_commit=agent_framework_commit,
        agent_config=agent_config,
        agent_max_turns=agent_max_turns,
        swebench_tests_timeout=swebench_tests_timeout,
        swebench_agent_timeout=swebench_agent_timeout,
        apptainer_memory_limit_mb=apptainer_memory_limit_mb,
        command_exec_timeout=command_exec_timeout,
        inference=inference_config,
        server=server,
    )

    run_oh = RunOpenHandsAgent(
        cfg=cfg,
        openhands_setup_dir=openhands_setup_dir,
        swebench_setup_dir=swebench_setup_dir,
        r2e_gym_setup_dir=r2e_gym_setup_dir,
        dataset_path=dataset_path,
        ng_global_config_dict_str=ng_global_config_dict_str,
        openhands_should_log=openhands_should_log,
        debug=debug,
        model_server_name=model_server_name,
        metrics_fpath=metrics_fpath,
    )

    result = await run_oh.process_single_datapoint(problem_info, persistent_dir)
    print(f"Process completed for {instance_id}", flush=True)

    result["oh_time_metrics"]["ray_time_in_queue"] = ray_submit_time - ray_queue_time

    try:
        with open(output_file, "w") as f:
            json.dump(result, f)
    except Exception as e:
        print(f"Failed to write result to {output_file}: {e}", flush=True)
        raise e

    # Read results
    if not output_file.exists():
        raise RuntimeError(f"No output file generated: {output_file}")

    # Try to find and include trajectory file
    trajectories_dir = persistent_dir / "trajectories"
    trajectory_data, tools = get_openhands_trajectory_from_completions(trajectories_dir, instance_id)

    result["tools"] = tools
    result["trajectory"] = trajectory_data

    return result


### Harness and Evaluation Setup Utils ###

PARENT_DIR = Path(__file__).parent


def _run_setup_command(command: str, debug: bool) -> None:
    std_params = dict()
    if not debug:
        std_params = dict(
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    process = Popen(command, shell=True, **std_params)
    return_code = process.wait()
    assert return_code == 0, f"Command failed: {command}"


@contextmanager
def _setup_directory_lock(setup_dir: Path, label: str):
    """File-based lock to ensure only one process performs the setup."""
    lock_dir = setup_dir.parent
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f".{setup_dir.name}.lock"

    with open(lock_path, "w") as lock_file:
        print(f"Acquiring {label} setup lock at {lock_path}", flush=True)
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def setup_swebench_environment(swebench_repo: str, swebench_commit: str, debug: bool) -> Path:
    setup_dir = PARENT_DIR / "swe_swebench_setup"

    with _setup_directory_lock(setup_dir, "SWE-bench"):
        swebench_dir = setup_dir / "SWE-bench"
        uv_dir = setup_dir / "uv"
        python_dir = setup_dir / "python"

        if swebench_dir.exists():
            print(f"SWE-bench already set up at {setup_dir}", flush=True)
        else:
            print(f"Setting up SWE-bench environment at {setup_dir}...", flush=True)
            setup_dir.mkdir(parents=True, exist_ok=True)

            script_fpath = PARENT_DIR / "setup_scripts/swebench.sh"
            command = f"""SETUP_DIR={setup_dir} \\
UV_DIR={uv_dir} \\
PYTHON_DIR={python_dir} \\
SWEBENCH_DIR={swebench_dir} \\
SWEBENCH_REPO={swebench_repo} \\
SWEBENCH_COMMIT={swebench_commit} \\
    ./{script_fpath}"""
            _run_setup_command(command, debug)

            print(f"Setup directory: {setup_dir}", flush=True)

        print(f"  - SWE-bench: {swebench_dir}", flush=True)
        print(f"  - venv: {swebench_dir / 'venv'}", flush=True)
        print(f"  - uv: {uv_dir}", flush=True)
        print(f"  - Python: {python_dir}", flush=True)

        return setup_dir


def setup_r2e_gym_environment(debug: bool) -> Path:
    eval_harness_repo = "https://github.com/ludwig-n/R2E-Gym.git"
    eval_harness_commit = "local-eval"

    setup_dir = PARENT_DIR / "swe_r2e_gym_setup"

    with _setup_directory_lock(setup_dir, "R2E-Gym"):
        r2e_gym_dir = setup_dir / "R2E-Gym"
        uv_dir = setup_dir / "uv"
        python_dir = setup_dir / "python"

        # Check if setup is complete by verifying venv and installed module
        venv_dir = r2e_gym_dir / "venv"
        python_bin = venv_dir / "bin" / "python"
        should_setup = True
        if r2e_gym_dir.exists() and venv_dir.exists() and python_bin.exists():
            result = subprocess.run([str(python_bin), "-c", "import r2egym"])
            if result.returncode == 0:
                print(f"R2E-Gym already set up at {setup_dir}", flush=True)
                should_setup = False
            else:
                print("R2E-Gym directory exists but module not properly installed, rebuilding...", flush=True)

        if should_setup:
            print(f"Setting up R2E-Gym environment at {setup_dir}...", flush=True)
            setup_dir.mkdir(parents=True, exist_ok=True)

            script_fpath = PARENT_DIR / "setup_scripts/r2e_gym.sh"
            command = f"""SETUP_DIR={setup_dir} \\
UV_DIR={uv_dir} \\
PYTHON_DIR={python_dir} \\
R2E_GYM_DIR={r2e_gym_dir} \\
EVAL_HARNESS_REPO={eval_harness_repo} \\
EVAL_HARNESS_COMMIT={eval_harness_commit} \\
    ./{script_fpath}"""
            _run_setup_command(command, debug)

            print(f"Setup directory: {setup_dir}", flush=True)

        print(f"  - R2E-Gym: {r2e_gym_dir}", flush=True)
        print(f"  - venv: {r2e_gym_dir / '.venv'}", flush=True)
        print(f"  - uv: {uv_dir}", flush=True)
        print(f"  - Python: {python_dir}", flush=True)

        return setup_dir


def setup_openhands_environment(
    agent_framework_repo: str,
    agent_framework_commit: str,
    setup_dir: Optional[Path] = None,
    debug: bool = False,
) -> Path:
    setup_dir = PARENT_DIR / "swe_openhands_setup"

    with _setup_directory_lock(setup_dir, "OpenHands"):
        openhands_dir = setup_dir / "OpenHands"
        miniforge_dir = setup_dir / "miniforge3"

        if openhands_dir.exists() and Path(openhands_dir / ".venv" / "bin" / "python").exists():
            print(f"OpenHands already set up at {setup_dir}", flush=True)
        else:
            print(f"Setting up OpenHands environment at {setup_dir}...", flush=True)
            shutil.rmtree(setup_dir, ignore_errors=True)
            setup_dir.mkdir(parents=True, exist_ok=True)

            script_fpath = PARENT_DIR / "setup_scripts/openhands.sh"
            command = f"""SETUP_DIR={setup_dir} \\
MINIFORGE_DIR={miniforge_dir} \\
OPENHANDS_DIR={openhands_dir} \\
AGENT_FRAMEWORK_REPO={agent_framework_repo} \\
AGENT_FRAMEWORK_COMMIT={agent_framework_commit} \\
    ./{script_fpath}"""
            _run_setup_command(command, debug)

            print(f"Setup directory: {setup_dir}", flush=True)

        print(f"  - Miniforge: {miniforge_dir}", flush=True)
        print(f"  - OpenHands: {openhands_dir}", flush=True)

        return setup_dir
