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
from typing import Any, Optional

import tomlkit


class SupportedAgentFrameworks(str, Enum):
    swe_agent = "swe_agent"
    openhands = "openhands"


@dataclass
class SweBenchInferenceConfig:
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float = 0.95
    min_p: float | None = None
    random_seed: int | None = None
    tokens_to_generate: int | None = None
    repetition_penalty: float | None = None
    top_logprobs: int | None = None


@dataclass
class SweBenchGenerationConfig:
    output_file: Path
    agent_framework: SupportedAgentFrameworks
    agent_framework_repo: str | None = None
    agent_framework_commit: str = "HEAD"
    agent_config: str | None = None
    agent_max_turns: int = 100
    swebench_tests_timeout: int = 60 * 30
    inference: SweBenchInferenceConfig = field(default_factory=SweBenchInferenceConfig)
    server: dict = field(default_factory=dict)


# Converts the parameter names above to the corresponding OpenAI parameter names.
NS_TO_OPENAI_PARAM = {
    "tokens_to_generate": "max_tokens",
    "top_logprobs": "top_logprobs",
    "random_seed": "seed",
    "top_k": "top_k",
    "min_p": "min_p",
    "repetition_penalty": "repetition_penalty",
}


# Converts the parameter names above to the corresponding parameters in OpenHands's LLM config.
# https://github.com/All-Hands-AI/OpenHands/blob/main/openhands/core/config/llm_config.py#L12
NS_TO_OPENHANDS_PARAM = {
    "tokens_to_generate": "max_output_tokens",
    "top_k": "top_k",
    "random_seed": "seed",
    "min_p": None,
    "repetition_penalty": None,
    "top_logprobs": None,
}


@dataclass
class RunOpenHandsAgent:
    cfg: SweBenchGenerationConfig
    output_dir: str = None
    openhands_setup_dir: Path | None = None
    swebench_setup_dir: Path | None = None
    dataset_path: str | None = None

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

    async def _run_openhands(
        self,
        data_point: dict[str, Any],
        api_base: str,
        agent_run_id: str,
        dataset_mount_path: Optional[str] = None,
    ):
        """
        Runs OpenHands on one instance.
        Returns the absolute (not mounted) path to a .jsonl file in the SWE-bench evaluation format.
        """
        agent_config = self.cfg.agent_config or "eval/swe-bench/openhands/default"

        # Add parameters to config.toml
        # TODO(sugam): is there a better way to do this?
        with open(agent_config, "r") as f:
            config = tomlkit.parse(f.read())

        config["llm"]["model"] |= {
            "model": self.cfg.server["model"],
            "base_url": api_base,
            "temperature": self.cfg.inference.temperature,
            "top_p": self.cfg.inference.top_p,
        }

        for ns_param, oh_param in NS_TO_OPENHANDS_PARAM.items():
            if not getattr(self.cfg.inference, ns_param):
                continue
            if oh_param:
                config["llm"]["model"][oh_param] = getattr(self.cfg.inference, ns_param)
            else:
                supported_params = [key for key, value in NS_TO_OPENHANDS_PARAM.items() if value is not None]
                raise ValueError(
                    f"Inference parameter {ns_param} is not supported by OpenHands. "
                    f"Supported inference parameters: temperature, top_p, {', '.join(supported_params)}."
                )

        config_str = tomlkit.dumps(config)

        eval_dir_in_openhands = f"evaluation/oh/{agent_run_id}"
        local_dataset_path = "/root/dataset/data.jsonl"

        assert self.openhands_setup_dir is not None, "OpenHands setup directory is not set"

        openhands_cmd = (
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
            # disable logging to file in the oh repo
            "export LOG_TO_FILE=false && "
            # set up config files
            f"echo {shlex.quote(config_str)} >config.toml && "
            # set local runtime & force verbose logs
            "export RUNTIME=local && "
            "export LOG_LEVEL=INFO && "
            "export LOG_TO_FILE=false && "
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
            f"    {local_dataset_path} && "
            # move outputs to the mounted directory
            f"mkdir -p /trajectories_mount/trajectories && "
            f"cp -r {eval_dir_in_openhands}/*/*/* /trajectories_mount/trajectories/{data_point['instance_id']}/ && "
            # remove the eval_dir_in_openhands directory after the evaluation is done
            f"rm -rf {eval_dir_in_openhands}"
        )

        search_path = os.path.join(
            self.output_dir / "trajectories",
            "**",
            data_point["instance_id"],
            "output.jsonl",
        )

        # Execute OpenHands command
        out_file = await self._execute_container_command(
            data_point=data_point,
            command=openhands_cmd,
            expected_file_pattern=search_path,
            mode="agent",
            dataset_mount_path=dataset_mount_path,
        )

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

    def _write_instance_dataset(self, data_point: dict[str, Any], agent_run_id: str) -> Path:
        """
        To avoid making HF dataset API calls, we write the instance dictionary to a file and mount it in the container.
        """
        instance_dataset_dir = Path(self.output_dir) / "instance_datasets"
        instance_dataset_dir.mkdir(parents=True, exist_ok=True)
        instance_dataset_path = instance_dataset_dir / f"{agent_run_id}.jsonl"
        with open(instance_dataset_path, "w") as f:
            f.write(data_point["instance_dict"] + "\n")
        return instance_dataset_path

    def _cleanup_instance_dataset(self, dataset_path):
        if dataset_path is None:
            return
        try:
            Path(dataset_path).unlink(missing_ok=True)
        except OSError:
            pass
        try:
            parent_dir = Path(dataset_path).parent
            if parent_dir.exists() and not any(parent_dir.iterdir()):
                parent_dir.rmdir()
        except OSError:
            pass

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
        container_formatter = data_point["container_formatter"]

        replacements = ["_1776_", "_s_"]

        # Generate all candidate IDs in order of priority
        candidate_ids = [instance_id]
        for replacement in replacements:
            replaced_id = instance_id.replace("__", replacement)
            candidate_ids.append(replaced_id)
            candidate_ids.append(replaced_id.lower())

        # Phase 1: Exact Matches
        for candidate_id in candidate_ids:
            path = container_formatter.format(instance_id=candidate_id)
            if os.path.exists(path):
                return path

        # Define the default fallback path (Strategy 1, original case)
        fallback_path = container_formatter.format(instance_id=instance_id.replace("__", replacements[0]))
        container_dir = os.path.dirname(fallback_path)

        # Phase 2: Fuzzy Search
        if os.path.exists(container_dir):
            # Create glob patterns for all candidates plus the original instance_id
            search_terms = [instance_id] + candidate_ids

            for term in search_terms:
                pattern = os.path.join(container_dir, f"*{term}*.sif")
                matches = glob.glob(pattern)
                if matches:
                    return matches[0]

            print(f"No container found with replacements {replacements} in {container_dir}", flush=True)
        else:
            print(f"Container directory {container_dir} does not exist", flush=True)

        # Phase 3: Fallback
        raise FileNotFoundError(
            f"No container file found for instance_id {instance_id}. "
            f"Tried the following candidate IDs: {candidate_ids}. "
            f"Also looked for .sif files in {container_dir}."
        )

    async def _execute_container_command(
        self,
        data_point: dict[str, Any],
        command: str,
        expected_file_pattern: str,
        mode: str,
        max_retries: int = 5,
        timeout: int = 100000,
        dataset_mount_path: Optional[str] = None,
    ):
        """Execute a command in an Apptainer container with retry logic."""
        # Find the container using multiple strategies
        container_name = self._find_container(data_point)

        dataset_path_to_mount = dataset_mount_path or self.dataset_path
        if dataset_path_to_mount is None:
            raise ValueError("Dataset path is not set")
        dataset_path_to_mount = str(dataset_path_to_mount)

        logs_dir = self.output_dir / "apptainer_logs"
        logs_dir.mkdir(exist_ok=True)
        log_file_path = logs_dir / f"{data_point['instance_id']}_{mode}.log"
        print(
            f"Starting execution of an apptainer command. Logs are available at {log_file_path}",
        )

        # Fix localhost URLs not working sometimes
        command = f"echo '127.0.0.1 localhost' >/etc/hosts && {command}"

        # Build mount arguments
        mount_args = [
            f"--mount type=bind,src={self.output_dir},dst=/trajectories_mount",
        ]

        # Add OpenHands setup directory mount if available (for OpenHands)
        if mode == "agent" and self.cfg.agent_framework == SupportedAgentFrameworks.openhands:
            # Mount the entire setup directory at both /openhands_setup and its original absolute path
            # This is needed because poetry and other tools have hardcoded absolute paths
            print(f"Mounting pre-built OpenHands from: {self.openhands_setup_dir}", flush=True)
            mount_args.append(f"--mount type=bind,src={self.openhands_setup_dir},dst=/openhands_setup")
            mount_args.append(f"--mount type=bind,src={self.openhands_setup_dir},dst={self.openhands_setup_dir}")
            mount_args.append(f"--mount type=bind,src={dataset_path_to_mount},dst=/root/dataset/data.jsonl")

        # Add SWE-bench setup directory mount if available (for evaluation)
        if mode == "eval":
            # Mount the entire setup directory at both /swebench_setup and its original absolute path
            # This is needed because uv venv has hardcoded absolute paths
            print(f"Mounting pre-built SWE-bench from: {self.swebench_setup_dir}", flush=True)
            mount_args.append(f"--mount type=bind,src={self.swebench_setup_dir},dst=/swebench_setup")
            mount_args.append(f"--mount type=bind,src={self.swebench_setup_dir},dst={self.swebench_setup_dir}")
            mount_args.append(f"--mount type=bind,src={dataset_path_to_mount},dst=/root/dataset/data.jsonl")

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

    async def process_single_datapoint(self, data_point: dict[str, Any]):
        self.output_dir = Path(self.cfg.output_file).parent

        agent_run_id = f"{data_point['instance_id']}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        instance_dataset_path = self._write_instance_dataset(data_point, agent_run_id)
        api_base = self.cfg.server["base_url"]

        start_time = asyncio.get_running_loop().time()
        generation_time = None
        evaluation_time = None
        try:
            if self.cfg.agent_framework == SupportedAgentFrameworks.swe_agent:
                pred_file = await self._run_swe_agent(
                    data_point,
                    api_base,
                    instance_dataset_path,
                )
            elif self.cfg.agent_framework == SupportedAgentFrameworks.openhands:
                pred_file = await self._run_openhands(
                    data_point,
                    api_base,
                    agent_run_id,
                    instance_dataset_path,
                )
            else:
                raise ValueError(
                    f"Unsupported agent framework: {self.cfg.agent_framework}. "
                    f"Supported frameworks: {', '.join(SupportedAgentFrameworks)}."
                )

            generation_time = asyncio.get_running_loop().time() - start_time

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
                        "generation_time": generation_time,
                        "evaluation_time": evaluation_time,
                    }
                }

            else:
                # Run full evaluation with streaming output
                start_time = asyncio.get_running_loop().time()
                assert self.swebench_setup_dir is not None, "SWE-bench setup directory is not set"
                assert self.dataset_path is not None, "Dataset path is not set"
                swebench_cmd = (
                    # Use pre-built SWE-bench
                    "cd /swebench_setup/SWE-bench && "
                    # Set UV environment variables to use the mounted portable directories
                    f'export UV_INSTALL_DIR="{self.swebench_setup_dir}/uv" && '
                    f'export UV_PYTHON_INSTALL_DIR="{self.swebench_setup_dir}/python" && '
                    f'export PATH="{self.swebench_setup_dir}/uv/bin:$PATH" && '
                    f"ls -lrt /root/dataset && "
                    # Run with clean environment to avoid venv contamination
                    # Use the pre-built venv directly with its absolute path
                    f"env -u VIRTUAL_ENV {self.swebench_setup_dir}/SWE-bench/venv/bin/python -m swebench.harness.run_local_evaluation "
                    f"    --predictions_path {pred_mounted_path} "
                    f"    --instance_ids {data_point['instance_id']} "
                    f"    --timeout {self.cfg.swebench_tests_timeout} "
                    f"    --dataset_name /root/dataset/data.jsonl "
                    f"    --split {data_point['split']} "
                    f"    --run_id {agent_run_id} && "
                    f"cp -r logs/run_evaluation/{agent_run_id} /trajectories_mount/ && "
                    f"rm -rf logs/run_evaluation/{agent_run_id}"
                )

                # Execute SWE-bench evaluation command
                search_path = os.path.join(
                    self.output_dir,
                    agent_run_id,
                    "**",
                    f"{data_point['instance_id']}/report.json",
                )
                # TODO: should we fail on errors here? Seems that json isn't always generated
                try:
                    report_file = await self._execute_container_command(
                        data_point,
                        swebench_cmd,
                        search_path,
                        mode="eval",
                        timeout=self.cfg.swebench_tests_timeout + 120,
                        dataset_mount_path=instance_dataset_path,
                    )
                    evaluation_time = asyncio.get_running_loop().time() - start_time
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
                            "generation_time": generation_time,
                            "evaluation_time": evaluation_time,
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
                "generation_time": generation_time,
                "evaluation_time": evaluation_time,
            }

            return output_dict
        finally:
            self._cleanup_instance_dataset(instance_dataset_path)
