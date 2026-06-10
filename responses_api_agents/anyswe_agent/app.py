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
"""Agent-agnostic SWE-bench runner."""

import asyncio
import glob
import hashlib
import json
import os
import shlex
import shutil
import sys
import time
import uuid
from asyncio import Semaphore
from asyncio.subprocess import Process
from contextlib import contextmanager
from pathlib import Path
from subprocess import Popen
from subprocess import run as subprocess_run
from traceback import format_exc
from typing import Any, Dict, Optional, Tuple

import ray
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming


# Eval primitives


class ExecuteContainerCommandArgs(BaseModel):
    command: str
    expected_file_pattern: str
    mode: str  # "agent" | "eval"
    timeout: int


class ActiveContainerCommand(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    process: Process
    log_file: Any
    log_file_path: Path


class SWEBenchMetrics(BaseModel):
    resolved: Optional[bool] = None
    patch_exists: Optional[bool] = None
    model_patch: Optional[str] = None

    agent_timed_out: Optional[bool] = None
    eval_timed_out: Optional[bool] = None

    ray_queue_time: Optional[float] = None
    openhands_run_time: Optional[float] = None  # kept name for metric-schema parity
    generation_apptainer_spinup_time: Optional[float] = None
    final_eval_apptainer_spinup_time: Optional[float] = None
    final_eval_time: Optional[float] = None


def update_metrics(metrics_fpath: Path, update_dict: Dict[str, Any]) -> None:
    existing = {k: v for k, v in json.loads(metrics_fpath.read_text()).items() if v is not None}
    update = {k: v for k, v in update_dict.items() if v is not None}
    metrics_fpath.write_text(json.dumps(existing | update))


_SECRET_HINTS = ("api_key", "token", "secret", "password")


def _scrub_secrets(obj: Any) -> Any:
    """Redact secret-looking values."""
    if isinstance(obj, dict):
        return {k: ("***" if any(h in k.lower() for h in _SECRET_HINTS) else _scrub_secrets(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_secrets(v) for v in obj]
    return obj


def _safe_config_json(params: "AnySweInstanceConfig", indent: Optional[int] = None) -> str:
    """Serialize config without secrets or rendered commands."""
    d = json.loads(params.model_dump_json())
    d.pop("agent_apptainer_command_str", None)
    d.pop("eval_apptainer_command_str", None)
    return json.dumps(_scrub_secrets(d), indent=indent)


# Dataset processors


class BaseDatasetHarnessProcessor(BaseModel):
    config: Any  # AnySweAgentConfig at setup time; AnySweInstanceConfig at run time

    @property
    def parent_dir(self) -> Path:
        return Path(__file__).parent

    def _run_setup_command(self, command: str) -> None:
        assert Popen(command, shell=True).wait() == 0, f"Command failed: {command}"

    @contextmanager
    def _setup_directory_lock(self, setup_dir: Path, label: str):
        """Cross-node lock using mkdir (atomic on Lustre/NFS, unlike fcntl.flock)."""
        lock_dir = setup_dir.parent
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f".{setup_dir.name}.lockdir"

        print(f"Acquiring {label} setup lock at {lock_path}", flush=True)
        max_wait, poll_interval, waited = 3600, 5, 0
        while True:
            try:
                lock_path.mkdir(exist_ok=False)
                break
            except FileExistsError:
                try:
                    if time.time() - lock_path.stat().st_mtime > 3600:
                        print("  Lock appears stale, breaking it", flush=True)
                        shutil.rmtree(lock_path, ignore_errors=True)
                        continue
                except OSError:
                    pass
                if waited >= max_wait:
                    raise TimeoutError(f"Timed out waiting for {label} setup lock after {max_wait}s")
                time.sleep(poll_interval)
                waited += poll_interval
        try:
            yield
        finally:
            shutil.rmtree(lock_path, ignore_errors=True)

    def setup(self) -> Path:
        return self.parent_dir

    def get_run_command(self) -> ExecuteContainerCommandArgs:
        raise NotImplementedError

    def postprocess_after_run(self, report_file: Path) -> None:
        # Harnesses write `resolved` directly.
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
            if swebench_dir.exists():
                print(f"SWE-bench already set up at {setup_dir}")
                return setup_dir
            print(f"Setting up SWE-bench environment at {setup_dir}...", flush=True)
            script_fpath = self.parent_dir / "setup_scripts/swebench.sh"
            self._run_setup_command(
                f"SETUP_DIR={setup_dir} UV_DIR={setup_dir / 'uv'} PYTHON_DIR={setup_dir / 'python'} "
                f"SWEBENCH_DIR={swebench_dir} SWEBENCH_REPO={swebench_repo} "
                f"SWEBENCH_COMMIT={swebench_commit} {script_fpath}"
            )
            return setup_dir

    def get_run_command(self) -> ExecuteContainerCommandArgs:
        c = self.config
        cmd = (
            f'date +"%s.%N" > {c.final_eval_apptainer_spinup_timestamp_mounted_fpath} && '
            f"{self._get_command_sleep_until_predictions_file()} && "
            "cd /swebench_setup/SWE-bench && "
            f'export UV_INSTALL_DIR="{c.swebench_setup_dir}/uv" && '
            f'export UV_PYTHON_INSTALL_DIR="{c.swebench_setup_dir}/python" && '
            f'export PATH="{c.swebench_setup_dir}/uv/bin:$PATH" && '
            f"env -u VIRTUAL_ENV {c.swebench_setup_dir}/SWE-bench/venv/bin/python -m swebench.harness.run_local_evaluation "
            f"    --predictions_path {c.output_for_eval_mounted_path} "
            f"    --instance_ids {c.instance_id} "
            f"    --timeout {c.swebench_tests_timeout} "
            f"    --dataset_name /root/dataset/data.jsonl "
            f"    --split {c.problem_info['split']} "
            f"    --run_id {c.agent_run_id} && "
            f"cp -r logs/run_evaluation/{c.agent_run_id} /trajectories_mount/ && "
            f"rm -rf logs/run_evaluation/{c.agent_run_id} && rm -rf *{c.agent_run_id}*"
        )
        search_path = os.path.join(c.persistent_dir, c.agent_run_id, "**", f"{c.instance_id}/report.json")
        return ExecuteContainerCommandArgs(
            command=cmd, expected_file_pattern=search_path, mode="eval", timeout=c.swebench_tests_timeout + 120
        )


class SweBenchMultilingualDatasetProcessor(BaseDatasetHarnessProcessor):
    def setup(self) -> Path:
        swebench_repo = "https://github.com/Kipok/SWE-bench.git"
        swebench_commit = "HEAD"
        setup_dir = self.parent_dir / "swe_swebench_multilingual_setup"
        setup_dir.mkdir(parents=True, exist_ok=True)
        with self._setup_directory_lock(setup_dir, "SWE-bench_Multilingual"):
            ml_dir = setup_dir / "SWE-bench_Multilingual"
            if ml_dir.exists():
                print(f"SWE-bench_Multilingual already set up at {setup_dir}")
                return setup_dir
            print(f"Setting up SWE-bench_Multilingual environment at {setup_dir}...", flush=True)
            script_fpath = self.parent_dir / "setup_scripts/swebench_multilingual.sh"
            self._run_setup_command(
                f"SETUP_DIR={setup_dir} UV_DIR={setup_dir / 'uv'} PYTHON_DIR={setup_dir / 'python'} "
                f"SWEBENCH_DIR={ml_dir} SWEBENCH_REPO={swebench_repo} "
                f"SWEBENCH_COMMIT={swebench_commit} {script_fpath}"
            )
            return setup_dir

    def get_run_command(self) -> ExecuteContainerCommandArgs:
        c = self.config
        cmd = (
            f'date +"%s.%N" > {c.final_eval_apptainer_spinup_timestamp_mounted_fpath} && '
            f"{self._get_command_sleep_until_predictions_file()} && "
            "cd /swebench_multilingual_setup/SWE-bench_Multilingual && "
            f'export UV_INSTALL_DIR="{c.swebench_multilingual_setup_dir}/uv" && '
            f'export UV_PYTHON_INSTALL_DIR="{c.swebench_multilingual_setup_dir}/python" && '
            f'export PATH="{c.swebench_multilingual_setup_dir}/uv/bin:$PATH" && '
            f"env -u VIRTUAL_ENV {c.swebench_multilingual_setup_dir}/SWE-bench_Multilingual/venv/bin/python "
            f"    -m swebench.harness.run_local_evaluation "
            f"    --predictions_path {c.output_for_eval_mounted_path} "
            f"    --instance_ids {c.instance_id} "
            f"    --timeout {c.swebench_tests_timeout} "
            f"    --dataset_name /root/dataset/data.jsonl "
            f"    --split {c.problem_info['split']} "
            f"    --run_id {c.agent_run_id} && "
            f"cp -r logs/run_evaluation/{c.agent_run_id} /trajectories_mount/ && "
            f"rm -rf logs/run_evaluation/{c.agent_run_id} && rm -rf *{c.agent_run_id}*"
        )
        search_path = os.path.join(c.persistent_dir, c.agent_run_id, "**", f"{c.instance_id}/report.json")
        return ExecuteContainerCommandArgs(
            command=cmd, expected_file_pattern=search_path, mode="eval", timeout=c.swebench_tests_timeout + 120
        )


class R2EGymDatasetProcessor(BaseDatasetHarnessProcessor):
    def setup(self) -> Path:
        eval_harness_repo = "https://github.com/sdevare-nv/nv-R2E-Gym.git"
        eval_harness_commit = "local-eval"
        setup_dir = self.parent_dir / "swe_r2e_gym_setup"
        with self._setup_directory_lock(setup_dir, "R2E-Gym"):
            r2e_dir = setup_dir / "R2E-Gym"
            python_bin = r2e_dir / "venv" / "bin" / "python"
            if r2e_dir.exists() and python_bin.exists():
                if subprocess_run([str(python_bin), "-c", "import r2egym"]).returncode == 0:
                    print(f"R2E-Gym already set up at {setup_dir}", flush=True)
                    return setup_dir
                print("R2E-Gym present but module missing, rebuilding...", flush=True)
            print(f"Setting up R2E-Gym environment at {setup_dir}...", flush=True)
            setup_dir.mkdir(parents=True, exist_ok=True)
            script_fpath = self.parent_dir / "setup_scripts/r2e_gym.sh"
            self._run_setup_command(
                f"SETUP_DIR={setup_dir} UV_DIR={setup_dir / 'uv'} PYTHON_DIR={setup_dir / 'python'} "
                f"R2E_GYM_DIR={r2e_dir} EVAL_HARNESS_REPO={eval_harness_repo} "
                f"EVAL_HARNESS_COMMIT={eval_harness_commit} {script_fpath}"
            )
            return setup_dir

    def get_run_command(self) -> ExecuteContainerCommandArgs:
        c = self.config
        cmd = (
            f'date +"%s.%N" > {c.final_eval_apptainer_spinup_timestamp_mounted_fpath} && '
            f"{self._get_command_sleep_until_predictions_file()} && "
            "cd /r2egym_setup/R2E-Gym && "
            f'export UV_INSTALL_DIR="{c.r2e_gym_setup_dir}/uv" && '
            f'export UV_PYTHON_INSTALL_DIR="{c.r2e_gym_setup_dir}/python" && '
            f'export PATH="{c.r2e_gym_setup_dir}/uv/bin:$PATH" && '
            f"env -u VIRTUAL_ENV {c.r2e_gym_setup_dir}/R2E-Gym/venv/bin/python "
            f"    src/r2egym/agenthub/run/run_local_evaluation.py "
            f"    --predictions_path {c.output_for_eval_mounted_path} "
            f"    --instance_id {c.instance_id} "
            f"    --timeout {c.swebench_tests_timeout} "
            f"    --dataset /root/dataset/data.jsonl "
            f"    --output_dir /trajectories_mount/eval-outputs/{c.agent_run_id}"
        )
        search_path = os.path.join(c.persistent_dir, "eval-outputs", c.agent_run_id, "report.json")
        return ExecuteContainerCommandArgs(
            command=cmd, expected_file_pattern=search_path, mode="eval", timeout=c.swebench_tests_timeout + 120
        )


# Runner template

_RUNNER_TEMPLATE = """\
#!/usr/bin/env python3
# Auto-generated by anyswe_agent; do not edit.
import asyncio, json, os, subprocess, sys
from pathlib import Path

sys.path.insert(0, "/nemo_gym_mount")
os.environ["PATH"] = "/agent_deps_mount/bin:" + os.environ.get("PATH", "")

MODEL_URL   = os.environ.get("NGSWE_MODEL_URL", "")
MODEL_NAME  = os.environ["NGSWE_MODEL_NAME"]
INSTRUCTION = Path("/trajectories_mount/instruction.txt").read_text()
AGENT_KWARGS = json.loads(os.environ.get("NGSWE_AGENT_KWARGS", "{{}}"))
SAMPLING = json.loads(os.environ.get("NGSWE_SAMPLING", "{{}}"))

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming, NeMoGymEasyInputMessage
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from {agent_module} import {agent_class}, {agent_cfg_class}


_mock_client = ServerClient.model_construct(global_config_dict={{}})
_mock_client._build_server_base_url = lambda cfg: MODEL_URL


_cfg_sampling = {{k: v for k, v in SAMPLING.items() if k in {agent_cfg_class}.model_fields}}
# Without a Gym model server, let the agent resolve its own endpoint/key.
_model_server = ModelServerRef(name="policy_model", type="responses_api_models") if MODEL_URL else None
config = {agent_cfg_class}(
    host="0.0.0.0",
    port=0,
    name="{agent_class_lower}",
    entrypoint="app.py",
    model_server=_model_server,
    resources_server=ResourcesServerRef(name="anyswe", type="resources_servers"),
    **{{**AGENT_KWARGS, **_cfg_sampling}},
)
agent = {agent_class}(config=config, server_client=_mock_client)

if MODEL_URL:
    if hasattr(agent, "_resolve_model_base_url"):
        _v1 = MODEL_URL if MODEL_URL.endswith("/v1") else MODEL_URL + "/v1"
        agent._resolve_model_base_url = lambda: _v1
    if hasattr(agent, "_resolve_base_url"):
        agent._resolve_base_url = lambda: MODEL_URL

body = NeMoGymResponseCreateParamsNonStreaming(
    input=[NeMoGymEasyInputMessage(role="user", content=INSTRUCTION)],
    model=MODEL_NAME,
    **SAMPLING,
)
response = asyncio.run(agent.responses(request=None, body=body))
Path("/trajectories_mount/response.json").write_text(response.model_dump_json())
print(f"agent finished: {{len(response.output)}} output items", flush=True)

# Patch extraction: git diff HEAD.
patch = ""
for candidate in ["/testbed", "/workspace/repo", "/app", "/root/repo"]:
    p = Path(candidate)
    if p.exists() and (p / ".git").exists():
        patch = subprocess.run(
            ["git", "diff", "HEAD"], capture_output=True, text=True, errors="replace", cwd=str(p)
        ).stdout
        print(f"patch: {{len(patch)}} chars from {{p}}", flush=True)
        break
Path("/trajectories_mount/patch.diff").write_text(patch)
"""

# Configuration


class AnySweAgentConfig(BaseResponsesAPIAgentConfig):
    # Unset for external-API agents.
    model_server: Optional[ModelServerRef] = None

    agent_server_module: str = Field(
        description="Import path to the agent module, e.g. responses_api_agents.hermes_agent.app"
    )
    agent_server_class: str = Field(description="Agent class name, e.g. HermesAgent")
    agent_config_class: str = Field(description="Agent config class name, e.g. HermesAgentConfig")
    agent_kwargs: Dict[str, Any] = Field(default_factory=dict)

    container_formatter: str | list[str] = "docker://swebench/sweb.eval.x86_64.{instance_id}"
    swebench_tests_timeout: int = 1800
    swebench_agent_timeout: int = 2700
    apptainer_memory_limit_mb: int = 32768
    concurrency: int = 256
    skip_eval: bool = False


class AnySweServerConfig(BaseModel):
    """Startup config."""

    run_session_id: str
    base_results_dir: Path
    model_server_url: str
    nemo_gym_root: Path
    agent_deps_dir: Path
    swebench_setup_dir: Path
    r2e_gym_setup_dir: Path
    swebench_multilingual_setup_dir: Path


class AnySweInstanceConfig(AnySweAgentConfig, AnySweServerConfig):
    """Per-instance config."""

    problem_info: Dict[str, Any]
    body: NeMoGymResponseCreateParamsNonStreaming
    persistent_dir: Path
    agent_run_id: str
    instance_dataset_path: Path
    model_patch_path: Path
    output_for_eval_path: Path
    output_for_eval_mounted_path: Path
    metrics_fpath: Path
    container: str
    ray_queue_timestamp: float
    final_eval_apptainer_spinup_timestamp_fpath: Path
    final_eval_apptainer_spinup_timestamp_mounted_fpath: Path
    generation_apptainer_spinup_timestamp_fpath: Path
    generation_apptainer_spinup_timestamp_mounted_fpath: Path

    eval_command: Optional[ExecuteContainerCommandArgs] = None
    eval_apptainer_command_str: Optional[str] = None
    agent_command: Optional[ExecuteContainerCommandArgs] = None
    agent_apptainer_command_str: Optional[str] = None
    mask_sample: bool = False

    @property
    def instance_id(self) -> str:
        return self.problem_info["instance_id"]


class AnySweVerifyResponse(SWEBenchMetrics, BaseVerifyResponse):
    instance_config: Dict[str, Any]


# Agent harness


class GymAgentHarnessProcessor(BaseModel):
    """Agent dependency and runner setup."""

    config: Any  # AnySweAgentConfig at setup time; AnySweInstanceConfig at run time

    @property
    def _parent(self) -> Path:
        return Path(__file__).parent

    @property
    def _agent_key(self) -> str:
        # responses_api_agents.hermes_agent.app -> hermes_agent
        return self.config.agent_server_module.split(".")[-2]

    def setup(self) -> Path:
        """Install agent deps into a portable prefix."""
        deps_dir = self._parent / f"anyswe_{self._agent_key}_deps"
        sentinel = deps_dir / ".installed"
        script = self._parent / "setup_scripts" / f"{self._agent_key}_deps.sh"
        # Reinstall when setup inputs change.
        shared = self._parent / "setup_scripts" / "_portable_python.sh"
        reqs = PARENT_DIR / "responses_api_agents" / self._agent_key / "requirements.txt"
        recipe_src = b"".join(p.read_bytes() for p in (script, shared, reqs) if p.exists()) or b"no-script"
        recipe = hashlib.sha256(recipe_src).hexdigest()
        if sentinel.exists() and sentinel.read_text().strip() == recipe:
            print(f"Agent deps already at {deps_dir}", flush=True)
            return deps_dir

        if not script.exists():
            print(f"No setup script for {self._agent_key}, skipping deps install", flush=True)
            deps_dir.mkdir(parents=True, exist_ok=True)
            sentinel.write_text(recipe)
            return deps_dir

        deps_dir.mkdir(parents=True, exist_ok=True)
        proc = Popen(f"DEPS_DIR={deps_dir} NEMO_GYM_ROOT={PARENT_DIR} bash {script}", shell=True)
        assert proc.wait() == 0, f"Agent deps setup failed ({script})"
        sentinel.write_text(recipe)
        return deps_dir

    def get_run_command(self) -> ExecuteContainerCommandArgs:
        """Write the runner and command."""
        cfg: AnySweInstanceConfig = self.config

        (cfg.persistent_dir / "instruction.txt").write_text(cfg.problem_info.get("problem_statement", ""))

        runner = _RUNNER_TEMPLATE.format(
            agent_module=cfg.agent_server_module,
            agent_class=cfg.agent_server_class,
            agent_cfg_class=cfg.agent_config_class,
            agent_class_lower=cfg.agent_server_class.lower(),
        )
        (cfg.persistent_dir / "agent_runner.py").write_text(runner)

        cmd = (
            f'date +"%s.%N" > {cfg.generation_apptainer_spinup_timestamp_mounted_fpath} && '
            f"/agent_deps_mount/bin/python /trajectories_mount/agent_runner.py"
        )
        return ExecuteContainerCommandArgs(
            command=cmd,
            expected_file_pattern=str(cfg.persistent_dir / "response.json"),
            mode="agent",
            timeout=cfg.swebench_agent_timeout + 60,
        )


# Container lifecycle


class RunGymAgent(BaseModel):
    """Runs agent and eval containers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: AnySweInstanceConfig

    async def _start(self, cmd: ExecuteContainerCommandArgs, apptainer_cmd: str) -> ActiveContainerCommand:
        logs_dir = self.config.persistent_dir / "apptainer_logs"
        logs_dir.mkdir(exist_ok=True)
        log_path = logs_dir / f"{self.config.instance_id}_{cmd.mode}.log"
        log_file = open(log_path, "w")
        proc = await asyncio.create_subprocess_shell(apptainer_cmd, stdout=log_file, stderr=log_file)
        return ActiveContainerCommand(process=proc, log_file=log_file, log_file_path=log_path)

    async def _wait(self, active: ActiveContainerCommand, cmd: ExecuteContainerCommandArgs) -> str:
        try:
            await asyncio.wait_for(active.process.communicate(), timeout=cmd.timeout)
        except asyncio.TimeoutError:
            if active.process.returncode is None:
                active.process.kill()
                await active.process.wait()
            raise ValueError("Command timed out")
        finally:
            active.log_file.close()

        if active.process.returncode != 0:
            raise RuntimeError(
                f"returncode={active.process.returncode}\n{active.log_file_path.read_text(errors='replace')[-3000:]}"
            )

        matches = glob.glob(cmd.expected_file_pattern, recursive=True)
        if not matches:
            raise ValueError(f"Expected file not found: {cmd.expected_file_pattern}")
        return max(matches, key=os.path.getmtime)

    async def _kill(self, active: ActiveContainerCommand) -> None:
        if active.process.returncode is None:
            active.process.kill()
            await active.process.wait()
        active.log_file.close()

    async def process_single_datapoint(self) -> Optional[Path]:
        cfg = self.config
        metrics = SWEBenchMetrics(ray_queue_time=time.time() - cfg.ray_queue_timestamp)
        t0 = -time.time()
        metrics.openhands_run_time = t0
        metrics.generation_apptainer_spinup_time = t0
        metrics.final_eval_apptainer_spinup_time = t0

        agent_ctr = await self._start(cfg.agent_command, cfg.agent_apptainer_command_str)
        eval_ctr = None if cfg.skip_eval else await self._start(cfg.eval_command, cfg.eval_apptainer_command_str)

        try:
            await self._wait(agent_ctr, cfg.agent_command)
        except Exception as e:
            print(f"[{cfg.instance_id}] agent failed: {e}", flush=True)
            if eval_ctr:
                await self._kill(eval_ctr)
            metrics.openhands_run_time += time.time()
            metrics.patch_exists = False
            metrics.agent_timed_out = (
                metrics.openhands_run_time is not None and metrics.openhands_run_time >= cfg.swebench_agent_timeout
            )
            update_metrics(cfg.metrics_fpath, metrics.model_dump())
            return None

        metrics.generation_apptainer_spinup_time += float(cfg.generation_apptainer_spinup_timestamp_fpath.read_text())
        metrics.openhands_run_time += time.time()

        patch_file = cfg.persistent_dir / "patch.diff"
        patch_raw = patch_file.read_text() if patch_file.exists() else ""
        patch = (patch_raw.strip() + "\n") if patch_raw.strip() else ""
        metrics.model_patch = patch or None

        if not patch:
            metrics.patch_exists = False
            if eval_ctr:
                await self._kill(eval_ctr)
            update_metrics(cfg.metrics_fpath, metrics.model_dump())
            return None

        cfg.model_patch_path.write_text(patch)

        cfg.output_for_eval_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.output_for_eval_path.write_text(
            json.dumps(
                {
                    "model_name_or_path": cfg.body.model,
                    "instance_id": cfg.instance_id,
                    "model_patch": patch,
                }
            )
        )
        metrics.patch_exists = True

        if cfg.skip_eval:
            metrics.final_eval_apptainer_spinup_time = None
            metrics.final_eval_time = None
            update_metrics(cfg.metrics_fpath, metrics.model_dump())
            return None

        metrics.final_eval_time = -time.time()
        try:
            report_file = await self._wait(eval_ctr, cfg.eval_command)
        except Exception as e:
            print(f"[{cfg.instance_id}] eval failed: {e}", flush=True)
            metrics.final_eval_time += time.time()
            metrics.eval_timed_out = (
                metrics.final_eval_time is not None and metrics.final_eval_time >= cfg.swebench_tests_timeout
            )
            update_metrics(cfg.metrics_fpath, metrics.model_dump())
            return None

        metrics.final_eval_apptainer_spinup_time += float(cfg.final_eval_apptainer_spinup_timestamp_fpath.read_text())
        metrics.final_eval_time += time.time()
        update_metrics(cfg.metrics_fpath, metrics.model_dump())
        return Path(report_file)


@ray.remote(scheduling_strategy="SPREAD", runtime_env={"py_executable": sys.executable}, num_cpus=0.1)
def _run_remote(params_dict: dict) -> Optional[Path]:
    AnySweInstanceConfig.model_rebuild(force=True)
    RunGymAgent.model_rebuild(force=True)
    params = AnySweInstanceConfig.model_validate(params_dict)
    return asyncio.run(RunGymAgent(config=params).process_single_datapoint())


# Agent server


class AnySweAgent(SimpleResponsesAPIAgent):
    """Runs a Gym agent in each task container."""

    config: AnySweAgentConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _sem: Optional[Semaphore] = None
    _server: Optional[AnySweServerConfig] = None

    def model_post_init(self, context: Any) -> None:
        self._sem = Semaphore(self.config.concurrency)

        model_url = ""
        if self.config.model_server is not None:
            model_cfg = get_first_server_config_dict(
                self.server_client.global_config_dict, self.config.model_server.name
            )
            model_url = self.server_client._build_server_base_url(model_cfg)

        agent_deps_dir = GymAgentHarnessProcessor(config=self.config).setup()

        swebench_setup = SweBenchDatasetProcessor(config=self.config).setup()
        multilingual_setup = SweBenchMultilingualDatasetProcessor(config=self.config).setup()
        r2e_setup = R2EGymDatasetProcessor(config=self.config).setup()

        session_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        workspace = Path(__file__).parent

        self._server = AnySweServerConfig(
            run_session_id=session_id,
            base_results_dir=workspace / f"anyswe_results_{session_id}",
            model_server_url=model_url,
            nemo_gym_root=PARENT_DIR,
            agent_deps_dir=agent_deps_dir,
            swebench_setup_dir=swebench_setup,
            r2e_gym_setup_dir=r2e_setup,
            swebench_multilingual_setup_dir=multilingual_setup,
        )
        super().model_post_init(context)

    # Container utilities

    @staticmethod
    def _container_path_variants(path: str) -> list[str]:
        """Return direct and package-relative SIF paths."""
        if "://" in path or os.path.isabs(path):
            return [path]
        return [path, str(Path(__file__).parent / path)]

    @classmethod
    def _find_container(cls, data_point: dict) -> str:
        """Locate the Apptainer SIF for this instance."""
        instance_id = data_point["instance_id"]
        formatters = data_point["container_formatter"]
        if isinstance(formatters, str):
            formatters = [formatters]

        replacements = ["_1776_", "_s_"]
        candidates = [instance_id]
        for r in replacements:
            replaced = instance_id.replace("__", r)
            candidates += [replaced, replaced.lower()]

        # Return absolute local SIFs; Apptainer may run from another CWD.
        for fmt in formatters:
            for cid in candidates:
                for path in cls._container_path_variants(fmt.format(instance_id=cid)):
                    if os.path.exists(path):
                        return path if "://" in path else os.path.abspath(path)

        for fmt in formatters:
            fallback = fmt.format(instance_id=instance_id.replace("__", replacements[0]))
            for d in (os.path.dirname(p) for p in cls._container_path_variants(fallback)):
                if os.path.exists(d):
                    for term in candidates:
                        matches = glob.glob(os.path.join(d, f"*{term}*.sif"))
                        if matches:
                            return os.path.abspath(matches[0])

        raise FileNotFoundError(f"No container found for {instance_id}")

    @staticmethod
    def _apptainer_exec(params: AnySweInstanceConfig, mounts: list[str], exec_cmd: str, env: str = "") -> str:
        cmd = (
            f"apptainer exec --writable-tmpfs --cleanenv --pid --no-mount home,tmp,bind-paths "
            f"{env}{' '.join(mounts)} {params.container} {exec_cmd}"
        )
        if params.apptainer_memory_limit_mb > 0:
            cmd = f"ulimit -v {params.apptainer_memory_limit_mb * 1024} && {cmd}"
        return cmd

    def _build_agent_apptainer_cmd(self, params: AnySweInstanceConfig) -> str:
        mounts = [
            f"--mount type=bind,src={params.persistent_dir},dst=/trajectories_mount",
            f"--mount type=bind,src={params.nemo_gym_root},dst=/nemo_gym_mount,ro",
            f"--mount type=bind,src={params.agent_deps_dir},dst=/agent_deps_mount,ro",
            f"--mount type=bind,src={params.instance_dataset_path},dst=/root/dataset/data.jsonl",
        ]
        sampling = {
            k: getattr(params.body, k)
            for k in ("temperature", "top_p", "max_output_tokens")
            if getattr(params.body, k, None) is not None
        }
        # External-API agents resolve their own endpoint/key.
        model_name = params.agent_kwargs.get("model") or params.body.model or "model"
        env = (
            (f"--env NGSWE_MODEL_URL={shlex.quote(params.model_server_url)} " if params.model_server_url else "")
            + f"--env NGSWE_MODEL_NAME={shlex.quote(model_name)} "
            + f"--env NGSWE_AGENT_KWARGS={shlex.quote(json.dumps(params.agent_kwargs))} "
            + f"--env NGSWE_SAMPLING={shlex.quote(json.dumps(sampling))} "
        )
        script_dir = params.persistent_dir / "container_scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / "agent_script.sh"
        script_path.write_text(params.agent_command.command)
        mounts.append(f"--mount type=bind,src={script_path},dst=/container_scripts/agent_script.sh,ro")
        return self._apptainer_exec(
            params, mounts, "bash /container_scripts/agent_script.sh", env=env
        )

    def _build_eval_apptainer_cmd(self, params: AnySweInstanceConfig) -> str:
        dataset_name = params.problem_info.get("dataset_name", "")

        mounts = [
            f"--mount type=bind,src={params.persistent_dir},dst=/trajectories_mount",
            # SWE-bench venvs contain absolute setup paths.
            f"--mount type=bind,src={params.swebench_setup_dir},dst=/swebench_setup",
            f"--mount type=bind,src={params.swebench_setup_dir},dst={params.swebench_setup_dir}",
        ]

        if "SWE-bench_Multilingual" in dataset_name:
            mounts += [
                f"--mount type=bind,src={params.swebench_multilingual_setup_dir},dst=/swebench_multilingual_setup",
                f"--mount type=bind,src={params.swebench_multilingual_setup_dir},dst={params.swebench_multilingual_setup_dir}",
            ]

        if "R2E-Gym" in dataset_name:
            mounts += [
                f"--mount type=bind,src={params.r2e_gym_setup_dir},dst=/r2egym_setup",
                f"--mount type=bind,src={params.r2e_gym_setup_dir},dst={params.r2e_gym_setup_dir}",
            ]

        mounts.append(f"--mount type=bind,src={params.instance_dataset_path},dst=/root/dataset/data.jsonl")

        script_dir = params.persistent_dir / "container_scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / "eval_script.sh"
        script_path.write_text(params.eval_command.command)
        mounts.append(f"--mount type=bind,src={script_path},dst=/container_scripts/eval_script.sh,ro")

        return self._apptainer_exec(params, mounts, "bash /container_scripts/eval_script.sh")

    # Per-instance setup

    def _setup_params(self, body: NeMoGymResponseCreateParamsNonStreaming) -> Tuple[AnySweInstanceConfig, Any]:
        problem_info = body.metadata | {"container_formatter": self.config.container_formatter}
        instance_id = problem_info.get("instance_id", "unknown")

        instance_dir = f"{instance_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        persistent_dir = self._server.base_results_dir / instance_dir
        persistent_dir.mkdir(parents=True, exist_ok=True)

        agent_run_id = f"{instance_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        # Avoid HF API calls inside the container.
        dataset_dir = persistent_dir / "instance_datasets"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        instance_dataset_path = dataset_dir / f"{agent_run_id}.jsonl"
        instance_dict = json.loads(problem_info["instance_dict"])
        instance_dict.setdefault("repo_name", instance_dict.get("repo", ""))
        instance_dataset_path.write_text(json.dumps(instance_dict) + "\n")

        base_mounted = Path("/trajectories_mount")
        traj_root = persistent_dir / "trajectories" / instance_id

        params = AnySweInstanceConfig(
            **self.config.model_dump(),
            **self._server.model_dump(),
            problem_info=problem_info,
            body=body,
            persistent_dir=persistent_dir,
            agent_run_id=agent_run_id,
            instance_dataset_path=instance_dataset_path,
            model_patch_path=persistent_dir / "patch.diff",
            output_for_eval_path=traj_root / "output_for_eval.jsonl",
            output_for_eval_mounted_path=base_mounted / "trajectories" / instance_id / "output_for_eval.jsonl",
            metrics_fpath=persistent_dir / "nemo_gym_metrics.json",
            container=self._find_container(problem_info),
            ray_queue_timestamp=time.time(),
            final_eval_apptainer_spinup_timestamp_fpath=persistent_dir / "final_eval_apptainer_spinup_timestamp",
            final_eval_apptainer_spinup_timestamp_mounted_fpath=base_mounted / "final_eval_apptainer_spinup_timestamp",
            generation_apptainer_spinup_timestamp_fpath=persistent_dir / "generation_apptainer_spinup_timestamp",
            generation_apptainer_spinup_timestamp_mounted_fpath=base_mounted / "generation_apptainer_spinup_timestamp",
        )
        params.metrics_fpath.write_text("{}")

        dataset_name = problem_info.get("dataset_name", "")
        if "R2E-Gym" in dataset_name:
            dp_cls = R2EGymDatasetProcessor
        elif "SWE-bench_Multilingual" in dataset_name:
            dp_cls = SweBenchMultilingualDatasetProcessor
        else:
            dp_cls = SweBenchDatasetProcessor
        dataset_processor = dp_cls(config=params)

        params.eval_command = dataset_processor.get_run_command()
        params.eval_apptainer_command_str = self._build_eval_apptainer_cmd(params)

        params.agent_command = GymAgentHarnessProcessor(config=params).get_run_command()
        params.agent_apptainer_command_str = self._build_agent_apptainer_cmd(params)

        return params, dataset_processor

    # Request handlers

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        params, dataset_processor = self._setup_params(body)
        (params.persistent_dir / "params.json").write_text(_safe_config_json(params, indent=2))
        try:
            return await self._inner_responses(params, dataset_processor)
        except Exception:
            tb_path = params.persistent_dir / "traceback.err"
            tb_path.write_text(format_exc())
            print(f"[{params.instance_id}] exception: see {tb_path}", file=sys.stderr)
            raise

    async def _inner_responses(self, params: AnySweInstanceConfig, dataset_processor: Any) -> NeMoGymResponse:
        maybe_report = await _run_remote.remote(params.model_dump())

        resolved = False
        if maybe_report:
            dataset_processor.postprocess_after_run(maybe_report)
            report = json.loads(Path(maybe_report).read_text())
            if params.instance_id in report:
                resolved = bool(report[params.instance_id].get("resolved", False))

        # Timeouts make reward unreliable.
        persisted = SWEBenchMetrics.model_validate_json(params.metrics_fpath.read_text())
        if persisted.eval_timed_out or persisted.agent_timed_out:
            params.mask_sample = True

        update_metrics(params.metrics_fpath, {"resolved": resolved})

        response_path = params.persistent_dir / "response.json"
        if response_path.exists():
            saved = NeMoGymResponse.model_validate_json(response_path.read_text())
            output_items = saved.output
            tools = saved.tools or []
        else:
            output_items, tools = [], []

        return NeMoGymResponse(
            id=f"anyswe-{params.instance_id}",
            created_at=int(time.time()),
            model=params.body.model,
            object="response",
            output=output_items,
            parallel_tool_calls=params.body.parallel_tool_calls,
            tool_choice=params.body.tool_choice,
            tools=tools,
            metadata={
                "input": json.dumps([]),
                "metrics": params.metrics_fpath.read_text(),
                "instance_config": _safe_config_json(params),
            },
        )

    async def run(self, body: BaseRunRequest) -> AnySweVerifyResponse:
        async with self._sem:
            body.responses_create_params.parallel_tool_calls = True
            body.responses_create_params.tool_choice = "auto"
            response = await self.responses(body.responses_create_params)

            meta, response.metadata = response.metadata, None
            metrics = SWEBenchMetrics.model_validate_json(meta["metrics"])

            return AnySweVerifyResponse(
                responses_create_params=body.responses_create_params.model_dump()
                | {"input": json.loads(meta["input"]), "tools": [t.model_dump() for t in (response.tools or [])]},
                response=response,
                reward=1.0 if metrics.resolved else 0.0,
                **metrics.model_dump(),
                instance_config=AnySweInstanceConfig.model_validate_json(meta["instance_config"]).model_dump(),
            )


if __name__ == "__main__":
    AnySweAgent.run_webserver()
