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


SUPPORTED_DATASETS = [
    "SWE-Gym/SWE-Gym",
    "R2E-Gym/R2E-Gym-Subset",
    "princeton-nlp/SWE-bench_Verified",
    "nv-internal-1",
]


@dataclass
class SweBenchInferenceConfig:
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0
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
    swebench_tests_timeout: int = 30 * 60
    swebench_agent_timeout: int = 45 * 60
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
    r2e_gym_setup_dir: Path | None = None
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
        search_path = os.path.join(
            self.output_dir / "trajectories",
            "**",
            f"{data_point['instance_id']}.pred",
        )
        pred_file = await self._execute_container_command(
            data_point,
            swe_agent_cmd,
            search_path,
            mode="agent",
        )

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
        agent_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/oh_config.toml")

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
        config_file_path = f"/tmp/config_{agent_run_id}.toml"

        assert self.openhands_setup_dir is not None, "OpenHands setup directory is not set"

        agent_script_name = f"agent_script_{agent_run_id}.sh"
        cleanup_commands = (
            f"cd /openhands_setup/OpenHands && "
            f"mkdir -p /trajectories_mount/trajectories && "
            f"cp -r {eval_dir_in_openhands}/*/*/* /trajectories_mount/trajectories/{data_point['instance_id']}/ &&"
            f"rm -rf {eval_dir_in_openhands} && rm -rf {config_file_path}"
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
            # "export LOG_LEVEL=DEBUG && "
            # "export LOG_TO_FILE=true && "
            "export LOG_LEVEL=CRITICAL && "
            "export DEBUG=False && "
            "export DEBUG_LLM=False && "
            "export LOG_TO_FILE=False && "
            "export LOG_ALL_EVENTS=False && "
            "export DEBUG_RUNTIME=False && "
            "export VIRTUAL_ENV=/openhands_setup/OpenHands/.venv && "
            "export PATH=$PATH:/openhands_setup/OpenHands/.venv/bin && "
            # CRITICAL: Configure poetry to only use the OpenHands venv (ignore external venvs)
            "export POETRY_VIRTUALENVS_IN_PROJECT=true && "
            "export POETRY_VIRTUALENVS_CREATE=false && "
            "export POETRY_VIRTUALENVS_PATH=/openhands_setup/OpenHands && "
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
            f"    {self.cfg.agent_framework_commit} "  # openhands commit
            f"    CodeActAgent "  # agent
            f"    0 "  # Note: this is eval limit which randomly chooses an instance from the dataset
            f"    {self.cfg.agent_max_turns} "  # max agent iterations
            f"    1 "  # number of workers
            f"    {data_point['dataset_name']} "  # dataset name
            f"    {data_point['split']} "  # dataset split
            f"    {eval_dir_in_openhands} "
            f"    {data_point['instance_id']} "
            f"    {local_dataset_path} "
            f"    {config_file_path}"
        )

        agent_script_path = Path(self.output_dir) / agent_script_name
        with open(agent_script_path, "w") as f:
            f.write("#!/bin/bash\nset -e\n")
            f.write(agent_main_cmd)
            f.flush()
            os.fsync(f.fileno())

        for _ in range(10):
            if agent_script_path.exists():
                break
            time.sleep(0.5)

        if not agent_script_path.exists():
            raise FileNotFoundError(f"Failed to create agent script at {agent_script_path}")

        agent_timeout_seconds = self.cfg.swebench_agent_timeout
        openhands_cmd = (
            f"timeout --signal=TERM --kill-after=30 {agent_timeout_seconds} "
            f"bash /trajectories_mount/{agent_script_name}; "
            f"echo 'Cleaning up...'; "
            f"{cleanup_commands}"
        )

        search_path = os.path.join(
            self.output_dir / "trajectories",
            "**",
            data_point["instance_id"],
            "**",
            "output.jsonl",
        )

        try:
            # Execute OpenHands command
            out_file = await self._execute_container_command(
                data_point=data_point,
                command=openhands_cmd,
                expected_file_pattern=search_path,
                mode="agent",
                max_retries=1,
                timeout=self.cfg.swebench_agent_timeout + 60,
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
                            "model_patch": patch + "\n" if patch and not patch.endswith("\n") else patch,
                        }
                    )
                )
        except Exception as e:
            print(f"oh run_infer.sh output parsing failed: {e}", flush=True)
            return None
        return pred_file

    def _write_instance_dataset(self, data_point: dict[str, Any], agent_run_id: str) -> Path:
        """
        To avoid making HF dataset API calls, we write the instance dictionary to a file and mount it in the container.
        """
        instance_dataset_dir = Path(self.output_dir) / "instance_datasets"
        instance_dataset_dir.mkdir(parents=True, exist_ok=True)
        instance_dataset_path = instance_dataset_dir / f"{agent_run_id}.jsonl"

        # Parse instance_dict to ensure repo_name field exists
        instance_dict = json.loads(data_point["instance_dict"])
        if "repo" in instance_dict and "repo_name" not in instance_dict:
            instance_dict["repo_name"] = instance_dict["repo"]

        with open(instance_dataset_path, "w") as f:
            f.write(json.dumps(instance_dict) + "\n")
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

    async def _execute_container_command(
        self,
        data_point: dict[str, Any],
        command: str,
        expected_file_pattern: str,
        mode: str,
        max_retries: int = 2,
        timeout: int = 45 * 60,  # 45 minutes
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
        # print(
        #     f"Starting execution of an apptainer command. Logs are available at {log_file_path}",
        # )

        # Fix localhost URLs not working sometimes
        container_commands = []
        container_commands.append("echo '127.0.0.1 localhost' >/etc/hosts")

        # Build mount arguments
        mount_args = [
            f"--mount type=bind,src={self.output_dir},dst=/trajectories_mount",
        ]

        # Add OpenHands setup directory mount if available (for OpenHands)
        if mode == "agent" and self.cfg.agent_framework == SupportedAgentFrameworks.openhands:
            # Mount the entire setup directory at both /openhands_setup and its original absolute path
            # This is needed because poetry and other tools have hardcoded absolute paths
            # print(
            #     f"Mounting pre-built OpenHands from: {self.openhands_setup_dir}",
            #     flush=True,
            # )
            mount_args.append(f"--mount type=bind,src={self.openhands_setup_dir},dst=/openhands_setup,ro")
            mount_args.append(f"--mount type=bind,src={self.openhands_setup_dir},dst={self.openhands_setup_dir},ro")
            # Mount only the venv and miniforge as read-only to prevent mutation while keeping the rest writable
            venv_path = Path(self.openhands_setup_dir) / "OpenHands/.venv"
            mount_args.append(f"--mount type=bind,src={venv_path},dst=/openhands_setup/OpenHands/.venv,ro")
            mount_args.append(f"--mount type=bind,src={venv_path},dst={venv_path},ro")

            # make everything in OpenHands read-only
            mount_args.append(
                f"--mount type=bind,src={self.openhands_setup_dir}/OpenHands,dst=/openhands_setup/OpenHands,ro"
            )
            mount_args.append(
                f"--mount type=bind,src={self.openhands_setup_dir}/OpenHands/.eval_sessions,dst=/openhands_setup/OpenHands/.eval_sessions"
            )
            mount_args.append(
                f"--mount type=bind,src={self.openhands_setup_dir}/OpenHands/.eval_sessions,dst={self.openhands_setup_dir}/OpenHands/.eval_sessions"
            )
            mount_args.append(
                f"--mount type=bind,src={self.openhands_setup_dir}/OpenHands/logs,dst=/openhands_setup/OpenHands/logs"
            )
            mount_args.append(
                f"--mount type=bind,src={self.openhands_setup_dir}/OpenHands/logs,dst={self.openhands_setup_dir}/OpenHands/logs"
            )
            mount_args.append(
                f"--mount type=bind,src={self.openhands_setup_dir}/OpenHands/evaluation/oh,dst=/openhands_setup/OpenHands/evaluation/oh"
            )
            mount_args.append(
                f"--mount type=bind,src={self.openhands_setup_dir}/OpenHands/evaluation/oh,dst={self.openhands_setup_dir}/OpenHands/evaluation/oh"
            )

            mount_args.append(f"--mount type=bind,src={dataset_path_to_mount},dst=/root/dataset/data.jsonl")

            miniforge3_path = Path(self.openhands_setup_dir) / "miniforge3"
            mount_args.append(f"--mount type=bind,src={miniforge3_path},dst=/openhands_setup/miniforge3,ro")
            mount_args.append(f"--mount type=bind,src={miniforge3_path},dst={miniforge3_path},ro")

        # Add SWE-bench setup directory mount if available (for evaluation)
        if mode == "eval" and data_point["dataset_name"] != "nv-internal-1":
            # Mount the entire setup directory at both /swebench_setup and its original absolute path
            # This is needed because uv venv has hardcoded absolute paths
            # print(
            #     f"Mounting pre-built SWE-bench from: {self.swebench_setup_dir}",
            #     flush=True,
            # )
            mount_args.append(f"--mount type=bind,src={self.swebench_setup_dir},dst=/swebench_setup")
            mount_args.append(f"--mount type=bind,src={self.swebench_setup_dir},dst={self.swebench_setup_dir}")
            mount_args.append(f"--mount type=bind,src={dataset_path_to_mount},dst=/root/dataset/data.jsonl")

        if mode == "eval" and data_point["dataset_name"] == "nv-internal-1":
            run_script_path = self.output_dir / "run_script.sh"
            parsing_script_path = self.output_dir / "parsing_script.py"
            model_patch_path = self.output_dir / "patch.diff"

            mount_args.append(f"--mount type=bind,src={run_script_path},dst=/root/run_script.sh")
            mount_args.append(f"--mount type=bind,src={parsing_script_path},dst=/root/parsing_script.py")
            mount_args.append(f"--mount type=bind,src={model_patch_path},dst=/root/patch.diff")

        if mode == "eval" and "R2E-Gym" in data_point["dataset_name"]:
            # Mount the entire setup directory at both /r2egym_setup and its original absolute path
            # This is needed because uv venv has hardcoded absolute paths in its wrappers
            # print(f"Mounting R2E-Gym setup directory from: {self.r2e_gym_setup_dir}", flush=True)
            mount_args.append(f"--mount type=bind,src={self.r2e_gym_setup_dir},dst=/r2egym_setup")
            mount_args.append(f"--mount type=bind,src={self.r2e_gym_setup_dir},dst={self.r2e_gym_setup_dir}")
            mount_args.append(f"--mount type=bind,src={dataset_path_to_mount},dst=/root/dataset/data.jsonl")

        if mode == "agent" and "R2E-Gym" in data_point["dataset_name"]:
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
        container_commands.append(command)
        combined_command = " && ".join(container_commands)

        mount_str = " ".join(mount_args)

        # Launch Apptainer container and execute the command
        apptainer_cmd = (
            f"apptainer exec --writable-tmpfs --cleanenv --no-mount home,tmp,bind-paths "
            f"{mount_str} "
            f" {container_name} bash -c {shlex.quote(combined_command)}"
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
                        if process.returncode is None:
                            process.terminate()
                            try:
                                await asyncio.wait_for(process.wait(), timeout=10)
                            except asyncio.TimeoutError:
                                # Force kill if still running
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
            except Exception as e:
                if attempt < max_retries - 1:
                    print(
                        f"Attempt {attempt + 1} failed for instance {data_point['instance_id']}. Retrying... Error: {repr(e)}",
                        flush=True,
                    )
                    continue
                else:
                    print(
                        f"All {max_retries} attempts failed for instance {data_point['instance_id']}. Error: {repr(e)}",
                        flush=True,
                    )
                    print(
                        f"Apptainer command failed. Check logs at: {log_file_path}. Error: {repr(e)}",
                        flush=True,
                    )
                    raise ValueError(
                        f"Job failed for {data_point['instance_id']}. Check logs at: {log_file_path}. Error: {repr(e)}. "
                        f"Expected exactly one file matching {expected_file_pattern}, "
                        f"found {len(pred_files) if 'pred_files' in locals() else 'unknown'}."
                    )

    async def _run_r2e_gym_eval(
        self,
        pred_mounted_path: str,
        data_point: dict[str, Any],
        agent_run_id: str,
        instance_dataset_path: str,
    ):
        assert self.r2e_gym_setup_dir is not None, "R2E-Gym setup directory is not set"
        assert self.dataset_path is not None, "Dataset path is not set"

        r2e_gym_cmd = (
            # Use mounted directory path for cd
            "cd /r2egym_setup/R2E-Gym && "
            # Set UV environment variables to use the mounted portable directories
            f'export UV_INSTALL_DIR="{self.r2e_gym_setup_dir}/uv" && '
            f'export UV_PYTHON_INSTALL_DIR="{self.r2e_gym_setup_dir}/python" && '
            f'export PATH="{self.r2e_gym_setup_dir}/uv/bin:$PATH" && '
            # Run with clean environment to avoid venv contamination
            # Use the pre-built venv directly with its absolute path
            f"env -u VIRTUAL_ENV {self.r2e_gym_setup_dir}/R2E-Gym/venv/bin/python src/r2egym/agenthub/run/run_local_evaluation.py "
            f"    --predictions_path {pred_mounted_path} "
            f"    --instance_id {data_point['instance_id']} "
            f"    --timeout {self.cfg.swebench_tests_timeout} "
            f"    --dataset /root/dataset/data.jsonl "
            f"    --output_dir /trajectories_mount/eval-outputs/{agent_run_id}"
        )

        search_path = os.path.join(
            self.output_dir,
            "eval-outputs",
            agent_run_id,
            "report.json",
        )
        report_file = await self._execute_container_command(
            data_point,
            r2e_gym_cmd,
            search_path,
            mode="eval",
            timeout=self.cfg.swebench_tests_timeout + 120,
            dataset_mount_path=instance_dataset_path,
        )
        return report_file

    async def _run_swebench_eval(
        self,
        pred_mounted_path: str,
        data_point: dict[str, Any],
        agent_run_id: str,
        instance_dataset_path: str,
    ):
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
            f"rm -rf logs/run_evaluation/{agent_run_id} && rm -rf *{agent_run_id}*"
        )

        # Execute SWE-bench evaluation command
        search_path = os.path.join(
            self.output_dir,
            agent_run_id,
            "**",
            f"{data_point['instance_id']}/report.json",
        )

        report_file = await self._execute_container_command(
            data_point,
            swebench_cmd,
            search_path,
            mode="eval",
            timeout=self.cfg.swebench_tests_timeout + 120,
            dataset_mount_path=instance_dataset_path,
        )

        return report_file

    async def _run_nv_internal_eval(
        self, data_point: dict[str, Any], model_patch: str, instance_dataset_path: str
    ) -> str:
        nv_internal_eval_cmd = await self.prepare_nv_internal_eval(data_point, model_patch)
        instance_dict = json.loads(data_point["instance_dict"])

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

        search_path = os.path.join(
            self.output_dir,
            "eval_results",
            "output.json",
        )
        report_file = await self._execute_container_command(
            data_point,
            nv_internal_eval_cmd,
            search_path,
            mode="eval",
            timeout=self.cfg.swebench_tests_timeout + 120,
            dataset_mount_path=instance_dataset_path,
        )

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
            f.write(json.dumps({data_point["instance_id"]: report_dict}, indent=4))
            return report_file

    async def prepare_nv_internal_eval(self, data_point: dict[str, Any], model_patch: str):
        instance_dict = json.loads(data_point["instance_dict"])
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
        run_script_path = self.output_dir / "run_script.sh"
        parsing_script_path = self.output_dir / "parsing_script.py"
        model_patch_path = self.output_dir / "patch.diff"
        with open(model_patch_path, "w") as f:
            # Add a newline to the end of the patch if it doesn't have one
            model_patch = model_patch + "\n" if not model_patch.endswith("\n") else model_patch
            f.write(model_patch)
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

        return cmd

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

    async def process_single_datapoint(self, data_point: dict[str, Any]):
        self.output_dir = Path(self.cfg.output_file).parent

        agent_run_id = f"{data_point['instance_id']}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        instance_dataset_path = self._write_instance_dataset(data_point, agent_run_id)
        api_base = self.cfg.server["base_url"]

        start_time = asyncio.get_running_loop().time()
        generation_time = None
        evaluation_time = None
        trajectory_dict = None
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

            if pred_file is None:
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
                    # TODO: should we fail on errors here? Seems that json isn't always generated
                    try:
                        start_time = asyncio.get_running_loop().time()
                        if data_point["dataset_name"] == "nv-internal-1":
                            report_file = await self._run_nv_internal_eval(
                                data_point,
                                trajectory_dict["model_patch"],
                                instance_dataset_path,
                            )
                        elif "R2E-Gym" in data_point["dataset_name"]:
                            report_file = await self._run_r2e_gym_eval(
                                pred_mounted_path,
                                data_point,
                                agent_run_id,
                                instance_dataset_path,
                            )
                        else:
                            report_file = await self._run_swebench_eval(
                                pred_mounted_path,
                                data_point,
                                agent_run_id,
                                instance_dataset_path,
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
