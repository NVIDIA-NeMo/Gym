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

"""
Core SWE-bench execution functions - decoupled from nemo-skills.
Processes a single problem at a time without batch processing infrastructure.
"""

import asyncio
import glob
import json
import logging
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional


LOG = logging.getLogger(__name__)

# Path to assets directory where repos will be cloned once
ASSETS_DIR = Path(__file__).parent.parent.parent / "assets"


# Mapping from inference parameter names to OpenAI parameter names
NS_TO_OPENAI_PARAM = {
    "tokens_to_generate": "max_tokens",
    "top_logprobs": "top_logprobs",
    "random_seed": "seed",
    "top_k": "top_k",
    "min_p": "min_p",
    "repetition_penalty": "repetition_penalty",
}


def ensure_repo_cloned(repo_url: str, repo_name: str, commit: str = "HEAD") -> Path:
    """Ensure a repository is cloned in the assets directory.

    Args:
        repo_url: Git repository URL
        repo_name: Name for the repo directory
        commit: Commit/branch to checkout (default: HEAD)

    Returns:
        Path: Path to the cloned repository
    """
    repo_path = ASSETS_DIR / repo_name

    # Create assets directory if it doesn't exist
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Clone or update the repository
    if not repo_path.exists():
        LOG.info(f"Cloning {repo_url} to {repo_path}")
        subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True, capture_output=True)
    else:
        LOG.info(f"Repository {repo_name} already exists at {repo_path}")

    # Checkout the specified commit/branch
    if commit != "HEAD":
        LOG.info(f"Checking out {commit} in {repo_path}")
        subprocess.run(["git", "-C", str(repo_path), "checkout", commit], check=True, capture_output=True)

    return repo_path


def get_config_path(config_name: str) -> str:
    """Simple config path resolver.

    If config_name starts with eval/, it's a path within the SWE-agent repo.
    Otherwise, resolves to absolute path.

    Args:
        config_name: Config file path

    Returns:
        str: Resolved config path (absolute for custom configs)
    """
    if config_name.startswith("eval/") or config_name.startswith("config/"):
        # It's a path within the agent framework repo, return as-is
        return config_name

    # If it's already an absolute path, return as-is
    if os.path.isabs(config_name):
        return config_name

    # Otherwise, resolve to absolute path relative to this file's directory
    config_dir = Path(__file__).parent / "configs"
    resolved_path = (config_dir / config_name).resolve()

    # Return absolute path (will be mounted in container)
    return str(resolved_path)


def find_container(instance_id: str, container_formatter: str) -> str:
    """Find the container file using multiple strategies.

    Tries in order:
    1. Exact match with "__" replaced by "_1776_"
    2. Exact match with "__" replaced by "_s_"
    3. Fuzzy search in container directory for files with either replacement

    Args:
        instance_id: Instance ID
        container_formatter: Container path template with {instance_id} placeholder

    Returns:
        str: Path to the container file (may not exist if all strategies fail)
    """
    # Strategy 1: Try _1776_ replacement (original case and lowercase)
    container_name = container_formatter.format(instance_id=instance_id.replace("__", "_1776_"))
    if os.path.exists(container_name):
        return container_name

    # Try lowercase version
    container_name_lower = container_formatter.format(instance_id=instance_id.replace("__", "_1776_").lower())
    if os.path.exists(container_name_lower):
        LOG.info(f"Using _1776_ replacement (lowercase): {container_name_lower}")
        return container_name_lower

    # Strategy 2: Try _s_ replacement (original case and lowercase)
    container_name_s = container_formatter.format(instance_id=instance_id.replace("__", "_s_"))
    if os.path.exists(container_name_s):
        LOG.info(f"Using _s_ replacement: {container_name_s}")
        return container_name_s

    # Try lowercase version
    container_name_s_lower = container_formatter.format(instance_id=instance_id.replace("__", "_s_").lower())
    if os.path.exists(container_name_s_lower):
        LOG.info(f"Using _s_ replacement (lowercase): {container_name_s_lower}")
        return container_name_s_lower

    # Strategy 3: Fuzzy search in container directory
    container_dir = os.path.dirname(container_name)
    if os.path.exists(container_dir):
        # Build search patterns for both replacements
        replaced_id_1776 = instance_id.replace("__", "_1776_")
        replaced_id_s = instance_id.replace("__", "_s_")

        # Search for .sif files with either replacement pattern (case-insensitive)
        patterns = [
            os.path.join(container_dir, f"*{replaced_id_1776}*.sif"),
            os.path.join(container_dir, f"*{replaced_id_s}*.sif"),
            os.path.join(container_dir, f"*{replaced_id_1776.lower()}*.sif"),
            os.path.join(container_dir, f"*{replaced_id_s.lower()}*.sif"),
        ]

        matching_files = []
        for pattern in patterns:
            matching_files.extend(glob.glob(pattern))

        if matching_files:
            container_path = matching_files[0]
            LOG.info(f"Using fuzzy match: {container_path}")
            return container_path
        else:
            LOG.warning(
                f"No container found with instance_id replacements "
                f"'{replaced_id_1776}' or '{replaced_id_s}' in {container_dir}"
            )
    else:
        LOG.warning(f"Container directory {container_dir} does not exist")

    # Return the original name as fallback (even though it doesn't exist)
    LOG.warning(f"Using non-existent container path: {container_name}")
    return container_name


async def execute_container_command(
    instance_id: str,
    container_formatter: str,
    command: str,
    expected_file_pattern: str,
    output_dir: Path,
    mode: str,
    max_retries: int = 3,
    timeout: int = 3600,  # 1 hour default timeout (was 100000)
    agent_config: Optional[str] = None,  # For mounting custom config directories
    additional_mounts: Optional[Dict[str, str]] = None,  # Additional mount paths {src: dst}
) -> str:
    """Execute a command in an Apptainer container with retry logic.

    Args:
        instance_id: Instance ID
        container_formatter: Container path template
        command: Shell command to execute inside container
        expected_file_pattern: Glob pattern for expected output file
        output_dir: Output directory (will be mounted in container)
        mode: Mode string for logging (e.g., "agent", "eval")
        max_retries: Maximum number of retry attempts
        timeout: Command timeout in seconds

    Returns:
        str: Path to the output file that was created

    Raises:
        ValueError: If command fails after all retries
    """
    # Find the container using multiple strategies
    container_name = find_container(instance_id, container_formatter)

    # Create logs directory if it doesn't exist
    logs_dir = output_dir / "apptainer_logs"
    logs_dir.mkdir(exist_ok=True)
    log_file_path = logs_dir / f"{instance_id}_{mode}.log"
    LOG.info("Starting execution of an apptainer command. Logs are available at %s", log_file_path)

    # Fix localhost URLs not working sometimes
    command = f"echo '127.0.0.1 localhost' >/etc/hosts && {command}"

    # Build environment variable flags for Apptainer
    env_flags = ""
    if os.getenv("HF_TOKEN"):
        env_flags += f" --env HF_TOKEN={shlex.quote(os.getenv('HF_TOKEN'))}"
        LOG.info("Passing HF_TOKEN to Apptainer container")
    if os.getenv("HF_HOME"):
        env_flags += f" --env HF_HOME={shlex.quote(os.getenv('HF_HOME'))}"
        LOG.info(f"Passing HF_HOME={os.getenv('HF_HOME')} to Apptainer container")
    if os.getenv("HF_DATASETS_OFFLINE"):
        env_flags += f" --env HF_DATASETS_OFFLINE={shlex.quote(os.getenv('HF_DATASETS_OFFLINE'))}"
        LOG.info("Passing HF_DATASETS_OFFLINE to Apptainer container")
    if os.getenv("TRANSFORMERS_OFFLINE"):
        env_flags += f" --env TRANSFORMERS_OFFLINE={shlex.quote(os.getenv('TRANSFORMERS_OFFLINE'))}"
        LOG.info("Passing TRANSFORMERS_OFFLINE to Apptainer container")

    # Build mount flags
    mount_flags = f"--mount type=bind,src={output_dir},dst=/trajectories_mount"

    # Mount config directory if agent_config is a file path (not a built-in config)
    if agent_config and not agent_config.startswith("eval/"):
        config_path = get_config_path(agent_config)
        if os.path.isfile(config_path):
            # Mount the parent directory so the config is accessible in container
            config_dir = os.path.dirname(os.path.abspath(config_path))
            mount_flags = f"--mount type=bind,src={config_dir},dst={config_dir} {mount_flags}"
            LOG.info(f"Mounting config directory: {config_dir}")

    # Add additional mounts if provided
    if additional_mounts:
        for src, dst in additional_mounts.items():
            mount_flags = f"--mount type=bind,src={src},dst={dst} {mount_flags}"
            LOG.info(f"Mounting {src} to {dst}")

    # Launch Apptainer container and execute the command
    apptainer_cmd = (
        f"apptainer exec --writable-tmpfs --no-mount home,tmp,bind-paths "
        f"{mount_flags} "
        f"{env_flags} {container_name} bash -c {shlex.quote(command)}"
    )

    # Retry apptainer command up to max_retries times
    for attempt in range(max_retries):
        try:
            # Stream output to log file as it appears
            with open(log_file_path, "w") as log_file:
                try:
                    # Create async subprocess
                    process = await asyncio.create_subprocess_shell(apptainer_cmd, stdout=log_file, stderr=log_file)
                    # Wait for completion with timeout
                    await asyncio.wait_for(process.communicate(), timeout=timeout)

                    if process.returncode != 0:
                        raise ValueError(f"Command failed with return code {process.returncode}")

                except asyncio.TimeoutError:
                    # Kill the process if it's still running
                    if process.returncode is None:
                        process.kill()
                        await process.wait()
                    attempt = max_retries  # Force exit the loop on timeout
                    raise ValueError("Command timed out")

            # Look for the expected file
            pred_files = glob.glob(expected_file_pattern, recursive=True)

            if len(pred_files) == 1:
                # Success, break out of retry loop
                return pred_files[0]
            else:
                raise ValueError(
                    f"Expected exactly one file matching {expected_file_pattern} for {instance_id}, "
                    f"found {len(pred_files)}."
                )
        except Exception:
            if attempt < max_retries - 1:
                LOG.warning(
                    "Attempt %d failed for instance %s. Retrying...",
                    attempt + 1,
                    instance_id,
                )
                continue
            else:
                LOG.error("All %d attempts failed for instance %s", max_retries, instance_id)
                LOG.error("Apptainer command failed. Check logs at: %s", log_file_path)
                raise ValueError(
                    f"Job failed for {instance_id}. Check logs at: {log_file_path}. "
                    f"Expected exactly one file matching {expected_file_pattern}, "
                    f"found {len(pred_files) if 'pred_files' in locals() else 'unknown'}."
                )


async def run_swe_agent_and_evaluate(
    problem_info: Dict,
    model_name: str,
    api_base: str,
    output_dir: Path,
    agent_config: Optional[str],
    agent_max_turns: int,
    inference_params: Dict,
    swebench_tests_timeout: int,
    agent_framework_repo: Optional[str] = None,
    agent_framework_commit: str = "HEAD",
    agent_timeout: int = 3600,  # Agent execution timeout in seconds (default 1 hour)
) -> Dict:
    """Run SWE-agent and evaluation in a single container execution.

    This combines agent execution and evaluation into one container call for efficiency.

    Args:
        problem_info: Problem information dict with keys: instance_id, problem_statement,
                     base_commit, dataset_name, split, container_formatter
        model_name: Model name to use
        api_base: API base URL
        output_dir: Output directory for results
        agent_config: Path to agent configuration file
        agent_max_turns: Maximum agent iterations
        inference_params: Dict with temperature, top_p, tokens_to_generate, etc.
        swebench_tests_timeout: Timeout for tests in seconds
        agent_framework_repo: URL of SWE-agent repo (optional)
        agent_framework_commit: Commit/branch to use (default: HEAD)
        agent_timeout: Timeout for agent execution in seconds

    Returns:
        dict: Evaluation results with keys:
            - swe-bench-metrics: Evaluation metrics
            - swe-bench-outputs: Trajectory and patch
            - generation: Empty string (for compatibility)

    Raises:
        ValueError: If execution fails
    """
    if agent_config is None:
        agent_config = "eval/swe-bench/swe-agent/default"
    if agent_framework_repo is None:
        agent_framework_repo = "https://github.com/SWE-agent/SWE-agent.git"

    # Ensure both repos are cloned to assets directory
    swe_agent_path = ensure_repo_cloned(agent_framework_repo, "SWE-agent", agent_framework_commit)
    swe_bench_path = ensure_repo_cloned("https://github.com/HeyyyyyyG/SWE-bench.git", "SWE-bench", "HEAD")

    # Build completion_kwargs for OpenAI API parameters
    completion_kwargs = {
        openai_param: inference_params.get(ns_param)
        for ns_param, openai_param in NS_TO_OPENAI_PARAM.items()
        if ns_param in inference_params and inference_params.get(ns_param) is not None
    }
    if "top_logprobs" in completion_kwargs:
        completion_kwargs["logprobs"] = True

    # Mount both repos in container
    combined_mounts = {str(swe_agent_path): "/root/SWE-agent", str(swe_bench_path): "/root/SWE-bench"}

    # Combined command: run agent, then evaluate
    # We'll use a bash variable to track the pred file path
    combined_cmd = (
        # Install uv once for both
        "curl -LsSf https://astral.sh/uv/install.sh | sh && "
        "source /root/.local/bin/env && "
        # Setup and run SWE-agent
        "cd /root/SWE-agent && "
        "uv venv --python 3.12 venv && "
        "uv pip install -p /root/SWE-agent/venv/bin/python -e . && "
        f"/root/SWE-agent/venv/bin/python -m sweagent run "
        f"    --config {get_config_path(agent_config)} "
        f"    --agent.model.name hosted_vllm/{model_name} "
        f"    --agent.model.api_base {api_base} "
        f"    --agent.model.temperature {inference_params.get('temperature', 0.0)} "
        f"    --agent.model.top_p {inference_params.get('top_p', 0.95)} "
        f"    --agent.model.completion_kwargs {shlex.quote(json.dumps(completion_kwargs))} "
        f"    --agent.model.per_instance_call_limit {agent_max_turns} "
        f"    --env.deployment.type local "
        f"    --env.repo.type preexisting "
        f"    --env.repo.repo_name testbed "
        f"    --env.repo.base_commit {problem_info['base_commit']} "
        f"    --problem_statement.text {shlex.quote(problem_info['problem_statement'])} "
        f"    --problem_statement.id {problem_info['instance_id']} && "
        # Convert .pred to .jsonl and move to mounted directory
        f"PRED_FILE=$(find trajectories -name '{problem_info['instance_id']}.pred' | head -1) && "
        f"PRED_JSONL=$(echo $PRED_FILE | sed 's/.pred$/.jsonl/') && "
        f"PRED_JSONL_NAME=$(basename $PRED_JSONL) && "
        f"TRAJ_DIR=$(dirname $PRED_FILE) && "
        f"cp $PRED_FILE $PRED_JSONL && "
        # Copy only the specific trajectory directory for this instance, not all trajectories
        f"mkdir -p /trajectories_mount/trajectories && "
        f"cp -r $TRAJ_DIR /trajectories_mount/trajectories/ && "
        # Also copy the .jsonl file to the root for easy access by evaluation
        f"cp $PRED_JSONL /trajectories_mount/$PRED_JSONL_NAME && "
        # Setup and run SWE-bench evaluation (only if patch exists)
        "cd /root/SWE-bench && "
        "uv venv --python 3.12 venv-eval && "
        "uv pip install -p /root/SWE-bench/venv-eval/bin/python -e . && "
        # Check if patch exists before running evaluation (check for both None and empty string)
        "if python3 -c \"import json; data=json.load(open('/trajectories_mount/'+'${PRED_JSONL_NAME}', 'r')); patch=data.get('model_patch'); exit(0 if patch and str(patch).strip() else 1)\"; then "
        f"  env -u VIRTUAL_ENV /root/SWE-bench/venv-eval/bin/python -m swebench.harness.run_local_evaluation "
        "    --predictions_path /trajectories_mount/${PRED_JSONL_NAME} "
        f"    --instance_ids {problem_info['instance_id']} "
        f"    --run_id eval-outputs "
        f"    --timeout {swebench_tests_timeout} "
        f"    --dataset_name {problem_info['dataset_name']} "
        f"    --split {problem_info['split']} && "
        # Copy only the specific instance's evaluation results, not all eval-outputs
        f"  EVAL_DIR=$(find logs/run_evaluation/eval-outputs -type d -name '{problem_info['instance_id']}' | head -1) && "
        f'  if [ -n "$EVAL_DIR" ]; then '
        f"    EVAL_PARENT=$(dirname $EVAL_DIR) && "
        f"    mkdir -p /trajectories_mount/eval-outputs && "
        f"    cp -r $EVAL_PARENT /trajectories_mount/eval-outputs/; "
        f"  fi; "
        f"else "
        f"  echo 'No patch found, skipping evaluation'; "
        f"fi"
    )

    # Execute combined command
    # We expect both the trajectory and potentially the report.json
    search_path = os.path.join(output_dir / "trajectories", "**", f"{problem_info['instance_id']}.jsonl")

    # Calculate total timeout (agent + evaluation + buffer)
    total_timeout = agent_timeout + swebench_tests_timeout + 120

    try:
        pred_file = await execute_container_command(
            problem_info["instance_id"],
            problem_info["container_formatter"],
            combined_cmd,
            search_path,
            output_dir,
            mode="combined",
            timeout=total_timeout,
            agent_config=agent_config,
            additional_mounts=combined_mounts,
        )
    except ValueError as e:
        LOG.error(f"Combined execution failed for {problem_info['instance_id']}: {e}")
        raise

    # Read the trajectory/patch file
    with open(pred_file, "r") as f:
        trajectory_dict = json.loads(f.read().strip())

    # Determine patch status
    # patch_exists: Did the model attempt to generate a patch? (field exists in trajectory)
    # has_valid_patch: Is the patch non-empty and potentially valid?
    model_patch = trajectory_dict.get("model_patch")
    patch_field_exists = "model_patch" in trajectory_dict
    has_valid_patch = model_patch is not None and model_patch != ""

    # Try to read evaluation report if valid patch existed
    report_json = None
    if has_valid_patch:
        report_path = os.path.join(output_dir, "eval-outputs", "**", f"{problem_info['instance_id']}/report.json")
        report_files = glob.glob(report_path, recursive=True)

        if report_files:
            try:
                with open(report_files[0], "r") as f:
                    report_json = json.loads(f.read().strip())
                LOG.info(f"Successfully read evaluation report for {problem_info['instance_id']}")
            except Exception as e:
                LOG.warning(f"Failed to read evaluation report: {e}")

    # Build response based on whether we have evaluation results
    if report_json and problem_info["instance_id"] in report_json:
        metrics = report_json[problem_info["instance_id"]]
    elif has_valid_patch:
        # Valid patch exists but evaluation failed or report missing
        metrics = {
            "resolved": False,
            "patch_exists": True,
            "patch_successfully_applied": False,
        }
    elif patch_field_exists:
        # Model attempted to generate patch but it's empty/None
        # This happens when the model runs but fails to generate a valid patch
        metrics = {
            "resolved": False,
            "patch_exists": True,  # Model attempted, so patch exists (even if empty)
            "patch_successfully_applied": False,
        }
    else:
        # Model never attempted to generate a patch (field doesn't exist)
        # This can happen if the agent errors out before reaching submission
        metrics = {
            "resolved": False,
            "patch_exists": False,
            "patch_successfully_applied": False,
        }

    output_dict = {
        "swe-bench-metrics": metrics,
        "swe-bench-outputs": trajectory_dict,
        "generation": "",  # required for compatibility
    }
    print("output_dict", output_dict)
    return output_dict


async def run_single_swe_agent_problem(
    problem_info: Dict,
    model_name: str,
    model_endpoint: str,
    output_dir: Path,
    agent_config: Optional[str],
    agent_max_turns: int,
    swebench_tests_timeout: int,
    inference_params: Dict,
    agent_framework_repo: Optional[str] = None,
    agent_framework_commit: str = "HEAD",
    agent_timeout: int = 3600,  # Agent execution timeout in seconds (default 1 hour)
) -> Dict:
    """Run SWE-agent on a single problem and evaluate the results.

    This is the main entry point for processing a single SWE-bench problem.
    Now uses a single container execution for both agent and evaluation.

    Args:
        problem_info: Problem information dict with keys: instance_id, problem_statement,
                     base_commit, dataset_name, split, container_formatter
        model_name: Model name
        model_endpoint: Model API endpoint URL
        output_dir: Output directory for results
        agent_config: Path to agent configuration file
        agent_max_turns: Maximum agent iterations
        swebench_tests_timeout: Timeout for tests in seconds
        inference_params: Dict with keys like temperature, top_p, tokens_to_generate, etc.
        agent_framework_repo: URL of SWE-agent repo (optional)
        agent_framework_commit: Commit/branch to use (default: HEAD)
        agent_timeout: Timeout for agent execution in seconds

    Returns:
        dict: Results dictionary with keys:
            - swe-bench-metrics: Evaluation metrics
            - swe-bench-outputs: Trajectory and patch
            - generation: Empty string (for compatibility)
    """
    LOG.info(f"Running SWE-agent and evaluation on problem: {problem_info['instance_id']}")

    # Start timing
    start_time = time.time()

    # Run both agent and evaluation in a single container execution
    results = await run_swe_agent_and_evaluate(
        problem_info=problem_info,
        model_name=model_name,
        api_base=model_endpoint,
        output_dir=output_dir,
        agent_config=agent_config,
        agent_max_turns=agent_max_turns,
        inference_params=inference_params,
        swebench_tests_timeout=swebench_tests_timeout,
        agent_framework_repo=agent_framework_repo,
        agent_framework_commit=agent_framework_commit,
        agent_timeout=agent_timeout,
    )

    # Calculate generation time
    end_time = time.time()
    generation_time = end_time - start_time

    LOG.info(f"Combined execution completed for {problem_info['instance_id']} in {generation_time:.2f} seconds")
    LOG.info("results", results)

    # Log generation time to a file in the instance's directory
    timing_file = output_dir / "generation_time.json"
    timing_data = {
        "instance_id": problem_info["instance_id"],
        "generation_time_seconds": generation_time,
        "start_time": start_time,
        "end_time": end_time,
        "model_name": model_name,
        "resolved": results.get("swe-bench-metrics", {}).get("resolved", False),
    }

    try:
        with open(timing_file, "w") as f:
            json.dump(timing_data, f, indent=2)
        LOG.info(f"Generation time logged to {timing_file}")
    except Exception as e:
        LOG.warning(f"Failed to write timing file: {e}")

    return results
