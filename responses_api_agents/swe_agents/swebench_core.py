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
import uuid
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
    
    # Add LiteLLM retry configuration for better connection handling
    env_flags += " --env LITELLM_NUM_RETRIES=8"  # Increase retries from default 6
    env_flags += " --env LITELLM_RETRY_DELAY=2"  # Initial delay 2s (exponential backoff)
    env_flags += " --env LITELLM_MAX_RETRY_DELAY=120"  # Max delay 2 minutes
    env_flags += " --env LITELLM_TIMEOUT=180"  # Request timeout 3 minutes
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
    sweagent_setup_dir: Optional[Path] = None,  # Pre-built SWE-agent directory
    swebench_setup_dir: Optional[Path] = None,  # Pre-built SWE-bench directory
    run_id: str = None,  # Unique run ID for organizing evaluation outputs
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

    # Use pre-built directories if provided, otherwise clone on-the-fly
    if sweagent_setup_dir is not None:
        swe_agent_path = sweagent_setup_dir / "SWE-agent"
        if not swe_agent_path.exists():
            raise ValueError(f"SWE-agent directory not found at {swe_agent_path}")
        LOG.info(f"Using pre-built SWE-agent from: {swe_agent_path}")
        # Mount entire setup dir at BOTH locations:
        # 1. Original absolute path (for hardcoded paths in venv wrappers)
        # 2. Convenient path (for our scripts)
        # This is needed because uv venv has hardcoded absolute paths in its wrappers
        sweagent_mount_src = str(sweagent_setup_dir.resolve())
        sweagent_original_path = sweagent_mount_src
        sweagent_convenient_path = "/root/sweagent_setup"
    else:
        LOG.warning("No sweagent_setup_dir provided, cloning on-the-fly (this is slower!)")
        if agent_framework_repo is None:
            agent_framework_repo = "https://github.com/SWE-agent/SWE-agent.git"
        swe_agent_path = ensure_repo_cloned(agent_framework_repo, "SWE-agent", agent_framework_commit)
        # For on-the-fly, just mount the repo directly
        sweagent_mount_src = str(swe_agent_path)
        sweagent_original_path = None  # No dual mount needed
        sweagent_convenient_path = "/root/SWE-agent"

    if swebench_setup_dir is not None:
        swe_bench_path = swebench_setup_dir / "SWE-bench"
        if not swe_bench_path.exists():
            raise ValueError(f"SWE-bench directory not found at {swe_bench_path}")
        LOG.info(f"Using pre-built SWE-bench from: {swe_bench_path}")
        # Mount entire setup dir at BOTH locations:
        # 1. Original absolute path (for hardcoded paths in venv wrappers)
        # 2. Convenient path (for our scripts)
        # This is needed because uv venv has hardcoded absolute paths in its wrappers
        swebench_mount_src = str(swebench_setup_dir.resolve())
        swebench_original_path = swebench_mount_src
        swebench_convenient_path = "/root/swebench_setup"
    else:
        LOG.warning("No swebench_setup_dir provided, cloning on-the-fly (this is slower!)")
        swe_bench_path = ensure_repo_cloned("https://github.com/HeyyyyyyG/SWE-bench.git", "SWE-bench", "HEAD")
        # For on-the-fly, just mount the repo directly
        swebench_mount_src = str(swe_bench_path)
        swebench_original_path = None  # No dual mount needed
        swebench_convenient_path = "/root/SWE-bench"

    # Build completion_kwargs for OpenAI API parameters
    completion_kwargs = {
        openai_param: inference_params.get(ns_param)
        for ns_param, openai_param in NS_TO_OPENAI_PARAM.items()
        if ns_param in inference_params and inference_params.get(ns_param) is not None
    }
    if "top_logprobs" in completion_kwargs:
        completion_kwargs["logprobs"] = True

    # Mount repos/setup dirs in container
    # For pre-built setups: Mount at BOTH the original path (for hardcoded venv paths) AND convenient path
    # For on-the-fly: Just mount at convenient path
    combined_mounts = {}

    # Mount SWE-agent setup
    if sweagent_original_path:
        # Pre-built: Mount at original path (this is what venv scripts expect)
        combined_mounts[sweagent_mount_src] = sweagent_original_path
        LOG.info(f"Mounting SWE-agent at original path: {sweagent_mount_src} -> {sweagent_original_path}")
    else:
        # On-the-fly: Mount at convenient path only
        combined_mounts[sweagent_mount_src] = sweagent_convenient_path
        LOG.info(f"Mounting SWE-agent at: {sweagent_mount_src} -> {sweagent_convenient_path}")

    # Mount SWE-bench setup
    if swebench_original_path:
        # Pre-built: Mount at original path (this is what venv scripts expect)
        combined_mounts[swebench_mount_src] = swebench_original_path
        LOG.info(f"Mounting SWE-bench at original path: {swebench_mount_src} -> {swebench_original_path}")
    else:
        # On-the-fly: Mount at convenient path only
        combined_mounts[swebench_mount_src] = swebench_convenient_path
        LOG.info(f"Mounting SWE-bench at: {swebench_mount_src} -> {swebench_convenient_path}")

    # Combined command: run agent, then evaluate
    # Use pre-built venvs if they exist, otherwise setup from scratch
    # We'll use a bash variable to track the pred file path
    if sweagent_setup_dir is not None:
        # Pre-built setup: Use ORIGINAL path (where venv was built) so hardcoded paths work
        # But the SWE-agent repo is under that path
        swe_agent_dir = f"{sweagent_original_path}/SWE-agent"
        agent_setup_cmd = f"cd {swe_agent_dir} && "
    else:
        # On-the-fly setup: install uv and create venv, paths are in /root/SWE-agent
        swe_agent_dir = "/root/SWE-agent"
        agent_setup_cmd = (
            "curl -LsSf https://astral.sh/uv/install.sh | sh && "
            "source /root/.local/bin/env && "
            f"cd {swe_agent_dir} && "
            "uv venv --python 3.12 venv && "
            f"uv pip install -p {swe_agent_dir}/venv/bin/python -e . && "
        )

    combined_cmd = (
        agent_setup_cmd + f"{swe_agent_dir}/venv/bin/python -m sweagent run "
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
        # Copy only the specific trajectory directory for this instance
        f"mkdir -p /trajectories_mount/trajectories && "
        f"cp -r $TRAJ_DIR /trajectories_mount/trajectories/ && "
        # Also copy the .jsonl file to the root for easy access by evaluation
        f"cp $PRED_JSONL /trajectories_mount/$PRED_JSONL_NAME && "
        # Setup and run SWE-bench evaluation (only if patch exists)
        # Use pre-built venv if available, otherwise create it
    )

    # Determine SWE-bench directory path in container
    if swebench_setup_dir is not None:
        # Pre-built setup: Use ORIGINAL path (where venv was built) so hardcoded paths work
        swe_bench_dir = f"{swebench_original_path}/SWE-bench"
    else:
        swe_bench_dir = "/root/SWE-bench"

    combined_cmd += (
        f"cd {swe_bench_dir} && "
        f"if [ -d 'venv' ] && [ -f 'venv/bin/python' ]; then "
        f"  echo 'Using pre-built SWE-bench venv'; "
        f"else "
        f"  echo 'Creating SWE-bench venv'; "
        f"  if [ ! -f /root/.local/bin/uv ]; then "
        f"    curl -LsSf https://astral.sh/uv/install.sh | sh && source /root/.local/bin/env; "
        f"  fi; "
        f"  uv venv --python 3.12 venv && "
        f"  uv pip install -p {swe_bench_dir}/venv/bin/python -e .; "
        f"fi && "
        # Check if patch exists before running evaluation (check for both None and empty string)
        "if python3 -c \"import json; data=json.load(open('/trajectories_mount/'+'${PRED_JSONL_NAME}', 'r')); patch=data.get('model_patch'); exit(0 if patch and str(patch).strip() else 1)\"; then "
        f"  env -u VIRTUAL_ENV {swe_bench_dir}/venv/bin/python -m swebench.harness.run_local_evaluation "
        "    --predictions_path /trajectories_mount/${PRED_JSONL_NAME} "
        f"    --instance_ids {problem_info['instance_id']} "
        f"    --run_id {run_id} "
        f"    --timeout {swebench_tests_timeout} "
        f"    --dataset_name {problem_info['dataset_name']} "
        f"    --split {problem_info['split']} && "
        # Copy evaluation results preserving run_id directory structure
        f"  cp -r logs/run_evaluation/{run_id} /trajectories_mount/; "
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
        # Search for report within the run_id directory structure
        report_path = os.path.join(output_dir, run_id, "**", f"{problem_info['instance_id']}/report.json")
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
    sweagent_setup_dir: Optional[Path] = None,
    swebench_setup_dir: Optional[Path] = None,
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
        agent_framework_repo: URL of SWE-agent repo (optional, only used if sweagent_setup_dir not provided)
        agent_framework_commit: Commit/branch to use (default: HEAD, only used if sweagent_setup_dir not provided)
        agent_timeout: Timeout for agent execution in seconds
        sweagent_setup_dir: Pre-built SWE-agent directory (if None, will clone on-the-fly)
        swebench_setup_dir: Pre-built SWE-bench directory (if None, will clone on-the-fly)

    Returns:
        dict: Results dictionary with keys:
            - swe-bench-metrics: Evaluation metrics
            - swe-bench-outputs: Trajectory and patch
            - generation: Empty string (for compatibility)
    """
    LOG.info(f"Running SWE-agent and evaluation on problem: {problem_info['instance_id']}")
    # Generate unique run_id for organizing evaluation outputs (matching OpenHands pattern)
    run_id = f"{problem_info['instance_id']}_{int(time.time())}_{str(uuid.uuid4())[:8]}"

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
        sweagent_setup_dir=sweagent_setup_dir,
        swebench_setup_dir=swebench_setup_dir,
        run_id=run_id,
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
