import asyncio
import glob
import json
import os
import shlex
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import tomlkit


class SupportedAgentFrameworks(str, Enum):
    swe_agent = "swe_agent"
    openhands = "openhands"


@dataclass
class SweBenchInferenceConfig:
    temperature: float = 0.0  # Temperature of 0 means greedy decoding
    top_k: int | None = None
    top_p: float = 0.95
    min_p: float | None = None
    random_seed: int | None = None
    tokens_to_generate: int | None = None
    repetition_penalty: float | None = None
    top_logprobs: int | None = None


@dataclass
class SweBenchGenerationConfig:
    input_file: str  # Path to the input file with data
    output_file: str  # Where to save the generations

    agent_framework: SupportedAgentFrameworks  # Which agentic framework to use

    # URL of the SWE-agent/OpenHands repo to pass to git clone. If None, will use the official repo
    agent_framework_repo: str | None = None
    agent_framework_commit: str = "HEAD"  # Which commit to use when cloning the SWE-agent/OpenHands repo

    # SWE-agent/OpenHands configuration file path. Can be specified in the same way as ns prompt configs
    # If None, will use the default for the chosen framework
    agent_config: str | None = None
    agent_max_turns: int = 100  # Max iterations for the agent

    swebench_tests_timeout: int = 60 * 30  # Timeout for the tests after applying the patch, in seconds

    inference: SweBenchInferenceConfig = field(default_factory=SweBenchInferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    max_samples: int = -1  # If > 0, will stop after generating this many samples. Useful for debugging
    skip_filled: bool = False  # If True, will skip the generations that are already in the output file

    # maximum number of concurrent requests to the server for the async loop
    # if sync loop is used, this is the batch size
    max_concurrent_requests: int = 512
    # chunk the dataset into equal sized parts and index into them
    num_chunks: int | None = None  # if specified, will split the data into chunks and only generate for one chunk
    chunk_id: int | None = None  # if specified, will index the specified chunk only

    # if False, will not add num_generated_tokens and generation_time values.
    # Useful when running judge jobs to keep the original generation statistics
    add_generation_stats: bool = True
    generation_key: str = "generation"
    async_position_key: str = "_async_position"  # key to use for preserving position in async loop in data dict
    dry_run: bool = False

    # if True, will move full generation to _full_generation key and keep cfg.generation_key without thinking tokens
    remove_thinking: bool = False
    thinking_begin: str = "<think>"
    thinking_end: str = "</think>"


# Converts the parameter names above to the corresponding OpenAI parameter names.
NS_TO_OPENAI_PARAM = {
    # Officially part of the OpenAI Chat Completions API.
    "tokens_to_generate": "max_tokens",
    "top_logprobs": "top_logprobs",
    "random_seed": "seed",
    # Not in the official API, but still supported by some servers, e.g. vllm.
    "top_k": "top_k",
    "min_p": "min_p",
    "repetition_penalty": "repetition_penalty",
    # temperature and top_p are passed as separate SWE-agent parameters.
}


# Converts the parameter names above to the corresponding parameters in OpenHands's LLM config.
# https://github.com/All-Hands-AI/OpenHands/blob/main/openhands/core/config/llm_config.py#L12
NS_TO_OPENHANDS_PARAM = {
    # Supported on OpenHands's side. top_k is not OpenAI-compatible and so may break some servers.
    "tokens_to_generate": "max_output_tokens",
    "top_k": "top_k",
    "random_seed": "seed",
    # Not supported by OpenHands. Nemo-Skills will raise an error if they are passed.
    "min_p": None,
    "repetition_penalty": None,
    "top_logprobs": None,
    # temperature and top_p are passed separately.
}


@dataclass
class RunOpenHandsAgent:
    cfg: SweBenchGenerationConfig
    output_dir: str = None
    openhands_setup_dir: Path | None = None  # Pre-built OpenHands directory to mount

    async def _run_swe_agent(self, data_point, api_base):
        """
        Runs SWE-agent on one instance.
        Returns the absolute (not mounted) path to a .jsonl file in the SWE-bench evaluation format.
        """
        if self.cfg.agent_config is None:
            self.cfg.agent_config = "eval/swe-bench/swe-agent/default"
        if self.cfg.agent_framework_repo is None:
            self.cfg.agent_framework_repo = "https://github.com/SWE-agent/SWE-agent.git"

        completion_kwargs = {
            openai_param: getattr(self.cfg.inference, ns_param)
            for ns_param, openai_param in NS_TO_OPENAI_PARAM.items()
            if getattr(self.cfg.inference, ns_param) is not None
        }
        if "top_logprobs" in completion_kwargs:
            completion_kwargs["logprobs"] = True

        swe_agent_cmd = (
            # first installing swe-agent repo
            "curl -LsSf https://astral.sh/uv/install.sh | sh && "
            "source /root/.local/bin/env && "
            "cd /root && "
            "mkdir SWE-agent && "
            "cd SWE-agent && "
            f"git clone {self.cfg.agent_framework_repo} . && "
            f"git checkout {self.cfg.agent_framework_commit} && "
            "uv venv --python 3.12 venv && "
            # do not activate venv, use uv pip with -p flag instead
            # "source venv/bin/activate && "
            # "uv pip install -e . && "
            "uv pip install -p /root/SWE-agent/venv/bin/python -e . && "
            # then running the agent
            f"/root/SWE-agent/venv/bin/python -m sweagent run "
            f"    --config {self.cfg.agent_config} "
            f"    --agent.model.name hosted_vllm/{self.cfg.server.model} "
            f"    --agent.model.api_base {api_base} "
            f"    --agent.model.temperature {self.cfg.inference.temperature} "
            f"    --agent.model.top_p {self.cfg.inference.top_p} "
            f"    --agent.model.completion_kwargs {shlex.quote(json.dumps(completion_kwargs))} "
            f"    --agent.model.per_instance_call_limit {self.cfg.agent_max_turns} "
            f"    --env.deployment.type local "
            f"    --env.repo.type preexisting "
            f"    --env.repo.repo_name testbed "
            f"    --env.repo.base_commit {data_point['base_commit']} "
            f"    --problem_statement.text {shlex.quote(data_point['problem_statement'])} "
            f"    --problem_statement.id {data_point['instance_id']} && "
            # move trajectories to the mounted directory
            f"cp -r trajectories /trajectories_mount/"
        )

        # Execute SWE-agent command
        search_path = os.path.join(self.output_dir / "trajectories", "**", f"{data_point['instance_id']}.pred")
        pred_file = await self._execute_container_command(data_point, swe_agent_cmd, search_path, mode="agent")

        with open(pred_file, "r") as f:
            trajectory_dict = json.loads(f.read().strip())

        # need to rename .pred to .jsonl
        pred_jsonl_file = pred_file.replace(".pred", ".jsonl")
        with open(pred_jsonl_file, "w") as f:
            f.write(json.dumps(trajectory_dict))

        # TODO: get num_generated_tokens and other stats from .traj file
        # looks like data['info']['model_stats']
        # {'instance_cost': 0, 'tokens_sent': 40858, 'tokens_received': 1775, 'api_calls': 9}

        return pred_jsonl_file

    async def _run_openhands(self, data_point, api_base):
        """
        Runs OpenHands on one instance.
        Returns the absolute (not mounted) path to a .jsonl file in the SWE-bench evaluation format.
        """
        if self.cfg.agent_config is None:
            self.cfg.agent_config = "eval/swe-bench/openhands/default"
        if self.cfg.agent_framework_repo is None:
            self.cfg.agent_framework_repo = "https://github.com/All-Hands-AI/OpenHands.git"

        # Add parameters to config.toml

        with open(self.cfg.agent_config, "r") as f:
            config = tomlkit.parse(f.read())

        config["llm"]["model"] |= {
            "model": self.cfg.server["model"],
            "base_url": api_base,
            "temperature": self.cfg.inference.temperature,
            "top_p": self.cfg.inference.top_p,
        }

        for ns_param, oh_param in NS_TO_OPENHANDS_PARAM.items():
            if getattr(self.cfg.inference, ns_param) is not None:
                if oh_param is not None:
                    config["llm"]["model"][oh_param] = getattr(self.cfg.inference, ns_param)
                else:
                    supported_params = [key for key, value in NS_TO_OPENHANDS_PARAM.items() if value is not None]
                    raise ValueError(
                        f"Inference parameter {ns_param} is not supported by OpenHands. "
                        f"Supported inference parameters: temperature, top_p, {', '.join(supported_params)}."
                    )

        config_str = tomlkit.dumps(config)

        eval_dir_in_openhands = f"evaluation/oh/{data_point['instance_id']}_{int(time.time())}_{str(uuid.uuid4())[:8]}"

        # Check if we have a pre-built OpenHands setup to use
        if self.openhands_setup_dir is not None:
            # Use pre-built OpenHands mounted at /openhands_setup
            openhands_cmd = (
                # make sure /workspace isn't mounted as a safety precaution
                "if [ -d /workspace ]; then "
                "    echo 'Exiting because /workspace is mounted.' && "
                "    echo 'Please make sure /workspace is not mounted inside of Apptainer before running OpenHands.' && "
                "    echo 'This is because OpenHands DELETES EVERYTHING in the /workspace folder if it exists.' && "
                "    exit 1; "
                "fi && "
                # Add miniforge bin to PATH (for tmux, node, poetry, etc.)
                "export PATH=/openhands_setup/miniforge3/bin:$PATH && "
                # Setup tmux socket (OpenHands requirement)
                "uid=$(id -ru 2>/dev/null || id -u) && "
                "export TMUX_TMPDIR=/tmp && "
                "export TMUX=/tmp/tmux-$uid/default && "
                "mkdir -p /tmp/tmux-$uid && "
                "chown $uid:$uid /tmp/tmux-$uid || true && "
                "chmod 700 /tmp/tmux-$uid && "
                "tmux -S /tmp/tmux-$uid/default start-server || true && "
                # Add miniforge bin to PATH to get Python, poetry, tmux, etc. (no conda activation needed)
                "export PATH=/openhands_setup/miniforge3/bin:$PATH && "
                "echo 'Using miniforge tools from PATH' && "
                # Use pre-built OpenHands
                "cd /openhands_setup/OpenHands && "
                # CRITICAL: Configure poetry to only use the OpenHands venv (ignore external venvs)
                "export POETRY_VIRTUALENVS_IN_PROJECT=true && "
                "export POETRY_VIRTUALENVS_CREATE=false && "
                "export POETRY_VIRTUALENVS_PATH=/openhands_setup/OpenHands && "
                # Directly activate the existing venv (so 'poetry run' uses it)
                "export VIRTUAL_ENV=/openhands_setup/OpenHands/.venv && "
                "export PATH=/openhands_setup/OpenHands/.venv/bin:$PATH && "
                # set up config files
                f"echo {shlex.quote(config_str)} >config.toml && "
                # set local runtime & force verbose logs
                "export RUNTIME=local && "
                "export LOG_ALL_EVENTS=true && "
                "export LOG_LEVEL=DEBUG && "
                # run the agent
                # f" export EVAL_OUTPUT_DIR={eval_dir_in_openhands} && "
                f"./evaluation/benchmarks/swe_bench/scripts/run_infer.sh "
                f"    llm.model "  # name of llm config section in config.toml
                f"    {self.cfg.agent_framework_commit} "  # openhands commit
                f"    CodeActAgent "  # agent
                f"    0 "  # Note: this is eval limit which randomly chooses an instance from the dataset
                f"    {self.cfg.agent_max_turns} "  # max agent iterations
                f"    1 "  # number of workers
                f"    {data_point['dataset_name']} "  # dataset name
                f"    {data_point['split']} "  # dataset split
                f"    {eval_dir_in_openhands} "
                f"    {data_point['instance_id']} "
                f"    {shlex.quote(data_point['instance_dict'])} && "
                # move outputs to the mounted directory
                f"mkdir -p /trajectories_mount/trajectories && "
                f"cp -r {eval_dir_in_openhands}/*/*/* /trajectories_mount/trajectories/{data_point['instance_id']}/"
            )
        else:
            # Fall back to in-container setup (original behavior)
            openhands_cmd = (
                # make sure /workspace isn't mounted as a safety precaution
                # (mounting it in the nemo-skills cluster config is ok, just not inside of apptainer specifically)
                "if [ -d /workspace ]; then "
                "    echo 'Exiting because /workspace is mounted.' && "
                "    echo 'Please make sure /workspace is not mounted inside of Apptainer before running OpenHands.' && "
                "    echo 'This is because OpenHands DELETES EVERYTHING in the /workspace folder if it exists.' && "
                "    exit 1; "
                "fi && "
                # install openhands repo + dependencies
                "cd /root && "
                'curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && '
                "bash Miniforge3-$(uname)-$(uname -m).sh -b && "
                'eval "$(/root/miniforge3/bin/conda shell.bash hook)" && '
                "mamba install -y --override-channels conda-forge::python=3.12 conda-forge::nodejs conda-forge::poetry conda-forge::tmux && "
                # OpenHands LocalRuntime uses tmux to manage a bash session. In Apptainer, the real UID (id -ru)
                # can differ from the effective UID (root) due to `su root -`. tmux chooses its default socket
                # path based on the REAL UID, e.g., /tmp/tmux-<real-uid>/default. Below we:
                #  - derive the real UID (fallback to id -u)
                #  - force tmux to use that socket path via TMUX/TMUX_TMPDIR
                #  - ensure the directory exists, has proper ownership and permissions
                #  - start the tmux server idempotently on that exact socket
                # This avoids 'error connecting to /tmp/tmux-<uid>/default' during LocalRuntime startup.
                # Ensure tmux socket directory exists with proper permissions, using real UID when available
                "uid=$(id -ru 2>/dev/null || id -u) && "
                "export TMUX_TMPDIR=/tmp && "
                "export TMUX=/tmp/tmux-$uid/default && "
                "mkdir -p /tmp/tmux-$uid && "
                "chown $uid:$uid /tmp/tmux-$uid || true && "
                "chmod 700 /tmp/tmux-$uid && "
                # Start tmux server on the exact socket path (idempotent)
                "tmux -S /tmp/tmux-$uid/default start-server || true && "
                "mkdir OpenHands && "
                "cd OpenHands && "
                f"git clone {self.cfg.agent_framework_repo} . && "
                f"git checkout {self.cfg.agent_framework_commit} && "
                "export INSTALL_DOCKER=0 && "
                "make build && "
                "poetry run python -m pip install datasets && "
                # set up config files
                f"echo {shlex.quote(config_str)} >config.toml && "
                f"echo \"selected_ids = ['{data_point['instance_id']}']\" >evaluation/benchmarks/swe_bench/config.toml && "
                # set local runtime & force verbose logs
                "export RUNTIME=local && "
                "export LOG_ALL_EVENTS=true && "
                "export LOG_LEVEL=DEBUG && "
                # run the agent
                f"./evaluation/benchmarks/swe_bench/scripts/run_infer.sh "
                f"    llm.model "  # name of llm config section in config.toml
                f"    {self.cfg.agent_framework_commit} "  # openhands commit
                f"    CodeActAgent "  # agent
                f"    0 "  # number of instances
                f"    {self.cfg.agent_max_turns} "  # max agent iterations
                f"    1 "  # number of workers
                f"    {data_point['dataset_name']} "  # dataset name
                f"    {data_point['split']} && "  # dataset split
                # move outputs to the mounted directory
                f"mkdir -p /trajectories_mount/trajectories && "
                f"cp -r evaluation/evaluation_outputs/outputs/*/*/* /trajectories_mount/trajectories/{data_point['instance_id']}"
            )

        # Execute OpenHands command
        search_path = os.path.join(
            self.output_dir / "trajectories",
            "**",
            data_point["instance_id"],
            "output.jsonl",
        )
        out_file = await self._execute_container_command(data_point, openhands_cmd, search_path, mode="agent")

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
                        "model_patch": patch,
                    }
                )
            )
        return pred_file

    def _find_container(self, data_point):
        """Find the container file using multiple strategies.

        Tries in order:
        1. Exact match with "__" replaced by "_1776_"
        2. Exact match with "__" replaced by "_s_"
        3. Fuzzy search in container directory for files with either replacement

        Returns:
            str: Path to the container file (may not exist if all strategies fail)
        """
        instance_id = data_point["instance_id"]
        container_formatter = data_point["container_formatter"]

        # Strategy 1: Try _1776_ replacement (original case and lowercase)
        container_name = container_formatter.format(instance_id=instance_id.replace("__", "_1776_"))
        if os.path.exists(container_name):
            return container_name

        # Try lowercase version
        container_name_lower = container_formatter.format(instance_id=instance_id.replace("__", "_1776_").lower())
        if os.path.exists(container_name_lower):
            print(
                f"Using _1776_ replacement (lowercase): {container_name_lower}",
                flush=True,
            )
            return container_name_lower

        # Strategy 2: Try _s_ replacement (original case and lowercase)
        container_name_s = container_formatter.format(instance_id=instance_id.replace("__", "_s_"))
        if os.path.exists(container_name_s):
            print(f"Using _s_ replacement: {container_name_s}", flush=True)
            return container_name_s

        # Try lowercase version
        container_name_s_lower = container_formatter.format(instance_id=instance_id.replace("__", "_s_").lower())
        if os.path.exists(container_name_s_lower):
            print(
                f"Using _s_ replacement (lowercase): {container_name_s_lower}",
                flush=True,
            )
            return container_name_s_lower

        # Strategy 3: Fuzzy search in container directory
        container_dir = os.path.dirname(container_name)
        if os.path.exists(container_dir):
            # Build search patterns for both replacements
            replaced_id_1776 = instance_id.replace("__", "_1776_")
            replaced_id_s = instance_id.replace("__", "_s_")

            # Search for .sif files with either replacement pattern (case-insensitive)
            # Include both original case and lowercase versions
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
                # Use the first matching file found
                container_path = matching_files[0]
                print(f"Using fuzzy match: {container_path}", flush=True)
                return container_path
            else:
                print(
                    f"No container found with instance_id replacements "
                    f"'{replaced_id_1776}' or '{replaced_id_s}' in {container_dir}"
                )
        else:
            print(f"Container directory {container_dir} does not exist")

        # Return the original name as fallback (even though it doesn't exist)
        print(f"Using non-existent container path: {container_name}", flush=True)
        return container_name

    async def _execute_container_command(
        self,
        data_point,
        command,
        expected_file_pattern,
        mode,
        max_retries=1,
        timeout=100000,
    ):
        """Execute a command in an Apptainer container with retry logic."""
        # Find the container using multiple strategies
        container_name = self._find_container(data_point)

        # Create logs directory if it doesn't exist
        logs_dir = self.output_dir / "apptainer_logs"
        logs_dir.mkdir(exist_ok=True)
        log_file_path = logs_dir / f"{data_point['instance_id']}_{mode}.log"
        print(
            "Starting execution of an apptainer command. Logs are available at %s",
            log_file_path,
        )

        # Fix localhost URLs not working sometimes
        command = f"echo '127.0.0.1 localhost' >/etc/hosts && {command}"

        # Build mount arguments
        mount_args = [
            "--mount type=bind,src=/nemo_run/code,dst=/nemo_run/code",
            f"--mount type=bind,src={self.output_dir},dst=/trajectories_mount",
        ]

        # Add OpenHands setup directory mount if available (for OpenHands)
        if self.openhands_setup_dir is not None:
            # Mount the entire setup directory at both /openhands_setup and its original absolute path
            # This is needed because poetry and other tools have hardcoded absolute paths in their wrappers
            print(f"Mounting pre-built OpenHands from: {self.openhands_setup_dir}", flush=True)
            mount_args.append(f"--mount type=bind,src={self.openhands_setup_dir},dst=/openhands_setup")
            mount_args.append(f"--mount type=bind,src={self.openhands_setup_dir},dst={self.openhands_setup_dir}")

        mount_str = " ".join(mount_args)

        # Launch Apptainer container and execute the command
        apptainer_cmd = (
            f"apptainer exec --writable-tmpfs --no-mount home,tmp,bind-paths "
            f"{mount_str} "
            f"{container_name} bash -c {shlex.quote(command)}"
        )

        # Retry apptainer command up to max_retries times
        for attempt in range(max_retries):
            try:
                # Stream output to log file as it appears
                with open(log_file_path, "w") as log_file:
                    try:
                        # Create async subprocess
                        process = await asyncio.create_subprocess_shell(
                            apptainer_cmd, stdout=log_file, stderr=log_file
                        )
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
                        f"Expected exactly one file matching {expected_file_pattern} for {data_point['instance_id']}, "
                        f"found {len(pred_files)}."
                    )
            except Exception:
                if attempt < max_retries - 1:
                    print(
                        f"Attempt {attempt + 1} failed for instance {data_point['instance_id']}. Retrying...",
                        flush=True,
                    )
                    continue
                else:
                    print(
                        f"All {max_retries} attempts failed for instance {data_point['instance_id']}",
                        flush=True,
                    )
                    print(
                        f"Apptainer command failed. Check logs at: {log_file_path}",
                        flush=True,
                    )
                    raise ValueError(
                        f"Job failed for {data_point['instance_id']}. Check logs at: {log_file_path}. "
                        f"Expected exactly one file matching {expected_file_pattern}, "
                        f"found {len(pred_files) if 'pred_files' in locals() else 'unknown'}."
                    )

    async def process_single_datapoint(self, data_point):
        """Will do all necessary generations to get a single answer for the data point."""
        self.output_dir = Path(self.cfg.output_file).parent

        # TODO: what's the right way to support api models, so that our standard parameters for that can be used?
        # TODO: use self.cfg.server.base_url, etc. Can we pass in API key?

        if "base_url" in self.cfg.server:
            api_base = self.cfg.server["base_url"]

        if self.cfg.agent_framework == SupportedAgentFrameworks.swe_agent:
            pred_file = await self._run_swe_agent(data_point, api_base)
        elif self.cfg.agent_framework == SupportedAgentFrameworks.openhands:
            pred_file = await self._run_openhands(data_point, api_base)
        else:
            raise ValueError(
                f"Unsupported agent framework: {self.cfg.agent_framework}. "
                f"Supported frameworks: {', '.join(SupportedAgentFrameworks)}."
            )

        pred_mounted_path = pred_file.replace(str(self.output_dir), "/trajectories_mount")
        with open(pred_file, "r") as f:
            trajectory_dict = json.loads(f.read())

        # Check if the trajectory has an empty patch before running evaluation
        has_patch = trajectory_dict["model_patch"] is not None

        if not has_patch:
            report_json = {
                data_point["instance_id"]: {
                    "resolved": False,
                    "patch_exists": False,
                    "patch_successfully_applied": False,
                }
            }
        else:
            # Run full evaluation with streaming output
            swe_bench_cmd = (
                # first installing SWE-bench repo
                "curl -LsSf https://astral.sh/uv/install.sh | sh && "
                "source /root/.local/bin/env && "
                "cd /root && "
                "git clone https://github.com/HeyyyyyyG/SWE-bench.git && "
                "cd SWE-bench && "
                "uv venv --python 3.12 venv && "
                # DO NOT activate venv, use uv pip with -p flag instead
                "uv pip install -p /root/SWE-bench/venv/bin/python -e . && "
                # Run with clean environment to avoid venv contamination
                f"env -u VIRTUAL_ENV /root/SWE-bench/venv/bin/python -m swebench.harness.run_local_evaluation "
                f"    --predictions_path {pred_mounted_path} "
                f"    --instance_ids {data_point['instance_id']} "
                f"    --run_id eval-outputs "
                f"    --timeout {self.cfg.swebench_tests_timeout} "
                f"    --dataset_name {data_point['dataset_name']} "
                f"    --split {data_point['split']} && "
                f"cp -r logs/run_evaluation/eval-outputs /trajectories_mount/"
            )

            # Execute SWE-bench evaluation command
            search_path = os.path.join(
                self.output_dir,
                "eval-outputs",
                "**",
                f"{data_point['instance_id']}/report.json",
            )
            # TODO: should we fail on errors here? Seems that json isn't always generated
            try:
                report_file = await self._execute_container_command(
                    data_point,
                    swe_bench_cmd,
                    search_path,
                    mode="eval",
                    timeout=self.cfg.swebench_tests_timeout + 120,
                )
            except ValueError:
                print(
                    f"Failed to execute SWE-bench evaluation command for {data_point['instance_id']}",
                    flush=True,
                )
                report_json = {
                    data_point["instance_id"]: {
                        "resolved": False,
                        "patch_exists": True,
                        "patch_successfully_applied": False,
                    }
                }
                report_file = None

            if report_file is not None:
                with open(report_file, "r") as f:
                    report_json = json.loads(f.read().strip())

        output_dict = {
            "swe-bench-metrics": report_json[data_point["instance_id"]],
            "swe-bench-outputs": trajectory_dict,
            "generation": "",  # required TODO: we should fix this
        }

        return output_dict
