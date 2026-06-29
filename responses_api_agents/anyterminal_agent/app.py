# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import hashlib
import json
import os
import shlex
import shutil
import signal
import sys
import time
import uuid
from asyncio import Semaphore
from asyncio.subprocess import Process
from pathlib import Path
from subprocess import Popen
from traceback import format_exc
from typing import Any, Dict, Optional

import ray
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming


def _read_task_meta(task_dir: Path) -> dict:
    """Read workdir and timeouts from task.toml + Dockerfile at runtime (fallback when not in JSONL)."""
    result = {}
    toml_path = task_dir / "task.toml"
    if toml_path.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(toml_path, "rb") as f:
            cfg = tomllib.load(f)
        result["agent_timeout_sec"] = (cfg.get("agent") or {}).get("timeout_sec")
        result["verifier_timeout_sec"] = (cfg.get("verifier") or {}).get("timeout_sec")
    dockerfile = task_dir / "environment" / "Dockerfile"
    if dockerfile.exists():
        for line in dockerfile.read_text().splitlines():
            if line.strip().upper().startswith("WORKDIR"):
                parts = line.strip().split(None, 1)
                if len(parts) > 1:
                    result["workdir"] = parts[1]
    return result


def _instruction_from_input(body: NeMoGymResponseCreateParamsNonStreaming) -> str:
    """Extract the task prompt from the Responses-API input messages.

    Joins the text of all messages (handling str or content-part list, dict or model form).
    """
    items = body.input
    if isinstance(items, str):
        return items
    parts: list[str] = []
    for item in items or []:
        content = getattr(item, "content", None) if not isinstance(item, dict) else item.get("content")
        if isinstance(content, list):
            content = "".join((p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content)
        if content:
            parts.append(content)
    return "\n".join(parts)


### Container process handle


class ActiveContainerProcess(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    process: Process
    log_file: Any
    log_file_path: Path


### Metrics


class TerminalBenchMetrics(BaseModel):
    resolved: Optional[bool] = None
    agent_timed_out: bool = False
    container_timed_out: bool = False
    mask_sample: bool = False

    ray_queue_time: Optional[float] = None
    agent_run_time: Optional[float] = None
    eval_run_time: Optional[float] = None
    total_run_time: Optional[float] = None


def update_metrics(metrics_fpath: Path, update_dict: Dict[str, Any]) -> None:
    existing = {k: v for k, v in json.loads(metrics_fpath.read_text()).items() if v is not None}
    update = {k: v for k, v in update_dict.items() if v is not None}
    metrics_fpath.write_text(json.dumps(existing | update))


def _safe_config_json(params: "AnyTerminalInstanceConfig", indent: Optional[int] = None) -> str:
    """Serialize config without secrets — redact secret-looking agent_kwargs."""
    d = json.loads(params.model_dump_json())
    d.pop("agent_command_str", None)
    d["agent_kwargs"] = {
        k: ("***" if any(s in k.lower() for s in ("api_key", "secret", "password", "token")) else v)
        for k, v in (d.get("agent_kwargs") or {}).items()
    }
    return json.dumps(d, indent=indent)


# Recreates /etc/dpkg in the writable tmpfs overlay so dpkg's rename() calls
# don't cross filesystem boundaries (squashfs base → tmpfs overlay = EXDEV).
_DPKG_FIX = """\
if [ -d /etc/dpkg ]; then
    cp -a /etc/dpkg /tmp/_dpkg_backup
    rm -rf /etc/dpkg
    cp -a /tmp/_dpkg_backup /etc/dpkg
    mkdir -p /etc/dpkg/dpkg.cfg.d
    printf 'force-overwrite\\nforce-overwrite-dir\\nforce-unsafe-io\\n' \
        > /etc/dpkg/dpkg.cfg.d/singularity-compat
fi
"""

### Agent runner template
# Injected into the task container; imports any agent class and calls responses().

_RUNNER_TEMPLATE = """\
#!/usr/bin/env python3
import asyncio, json, os, sys
from pathlib import Path

sys.path.insert(0, "/nemo_gym_mount")
os.environ["PATH"] = "/agent_deps_mount/bin:" + os.environ.get("PATH", "")

MODEL_URL    = os.environ.get("NGTB_MODEL_URL", "")
MODEL_NAME   = os.environ["NGTB_MODEL_NAME"]
INSTRUCTION  = Path("/trajectories_mount/instruction.txt").read_text()
AGENT_KWARGS = json.loads(os.environ.get("NGTB_AGENT_KWARGS", "{{}}"))
SAMPLING     = json.loads(os.environ.get("NGTB_SAMPLING", "{{}}"))

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming, NeMoGymEasyInputMessage
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from {agent_module} import {agent_class}, {agent_cfg_class}

_mock_client = ServerClient.model_construct(global_config_dict={{}})
_mock_client._build_server_base_url = lambda cfg: MODEL_URL

_cfg_sampling = {{k: v for k, v in SAMPLING.items() if k in {agent_cfg_class}.model_fields}}

_model_server = ModelServerRef(name="policy_model", type="responses_api_models") if MODEL_URL else None
config = {agent_cfg_class}(
    host="0.0.0.0",
    port=0,
    name="{agent_class_lower}",
    entrypoint="app.py",
    model_server=_model_server,
    resources_server=ResourcesServerRef(name="anyterminal", type="resources_servers"),
    **{{**_cfg_sampling, **AGENT_KWARGS}},
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
"""


### Agent harness installer
# Mirrors GymAgentHarnessProcessor in anyswe_agent: installs portable python + agent
# deps into a persistent prefix and writes agent_runner.py + instruction.txt.


class GymAgentHarnessProcessor(BaseModel):
    config: Any  # AnyTerminalAgentConfig at setup time; AnyTerminalInstanceConfig at run time

    @property
    def _parent(self) -> Path:
        return Path(__file__).parent

    @property
    def _agent_key(self) -> str:
        # responses_api_agents.hermes_agent.app -> hermes_agent
        return self.config.agent_server_module.split(".")[-2]

    def setup(self) -> Path:
        """Install agent deps into a portable prefix (idempotent, hash-keyed)."""
        deps_dir = self._parent / "deps" / f"anyterminal_{self._agent_key}_deps"
        sentinel = deps_dir / ".installed"
        script = self._parent / "setup_scripts" / f"{self._agent_key}_deps.sh"
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

    def get_run_command(self) -> str:
        """Write instruction.txt and agent_runner.py; return the shell command to run the agent."""
        cfg: AnyTerminalInstanceConfig = self.config
        instruction = _instruction_from_input(cfg.body)
        (cfg.persistent_dir / "instruction.txt").write_text(instruction)
        runner = _RUNNER_TEMPLATE.format(
            agent_module=cfg.agent_server_module,
            agent_class=cfg.agent_server_class,
            agent_cfg_class=cfg.agent_config_class,
            agent_class_lower=cfg.agent_server_class.lower(),
        )
        (cfg.persistent_dir / "agent_runner.py").write_text(runner)
        return (
            f"timeout {cfg.tb_agent_timeout} /agent_deps_mount/bin/python /trajectories_mount/agent_runner.py || true"
        )


### Configuration


class AnyTerminalAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: Optional[ModelServerRef] = None

    agent_server_module: str = Field(description="Import path to the agent module")
    agent_server_class: str = Field(description="Agent class name")
    agent_config_class: str = Field(description="Agent config class name")
    agent_kwargs: Dict[str, Any] = Field(default_factory=dict)

    tb_sif_dir: Optional[str] = Field(
        default=None,
        description="Directory of pre-built Apptainer SIF files. Falls back to docker:// pull if absent.",
    )
    tb_agent_timeout: int = 1800
    tb_eval_timeout: int = 300
    apptainer_memory_limit_mb: int = 32768  # fallback container memory cap when a task omits memory_mb
    agent_overhead_mb: int = 2048  # extra container memory on top of the task's memory_mb for the
    # in-container agent harness
    concurrency: int = 256


class AnyTerminalServerConfig(BaseModel):
    run_session_id: str
    base_results_dir: Path
    model_server_url: str
    model_name: str = ""
    nemo_gym_root: Path
    agent_deps_dir: Path


class AnyTerminalInstanceConfig(AnyTerminalAgentConfig, AnyTerminalServerConfig):
    problem_info: Dict[str, Any]
    body: NeMoGymResponseCreateParamsNonStreaming
    persistent_dir: Path
    verifier_dir: Path
    agent_run_id: str
    metrics_fpath: Path
    container: str
    ray_queue_timestamp: float
    agent_command_str: Optional[str] = None

    @property
    def task_name(self) -> str:
        return self.problem_info.get("task_name", self.problem_info.get("instance_id", "unknown"))

    @property
    def instance_id(self) -> str:
        return self.problem_info.get("instance_id", self.task_name)


class AnyTerminalVerifyResponse(TerminalBenchMetrics, BaseVerifyResponse):
    instance_config: Dict[str, Any]


### Container lifecycle


class RunTerminalAgent(BaseModel):
    """Single container: agent runs, signals done, host stages tests, container runs test.sh."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: AnyTerminalInstanceConfig

    async def _start(self, apptainer_cmd: str) -> ActiveContainerProcess:
        logs_dir = self.config.persistent_dir / "apptainer_logs"
        logs_dir.mkdir(exist_ok=True)
        log_path = logs_dir / f"{self.config.task_name}.log"
        log_file = open(log_path, "w")
        proc = await asyncio.create_subprocess_shell(
            apptainer_cmd, stdout=log_file, stderr=log_file, start_new_session=True
        )
        return ActiveContainerProcess(process=proc, log_file=log_file, log_file_path=log_path)

    async def _stage_tests(self, cfg: AnyTerminalInstanceConfig) -> None:
        """Copy test files into the staging dir once the agent signals it is done."""
        agent_done = cfg.persistent_dir / "agent_done"
        tests_ready = cfg.persistent_dir / "tests_ready"
        staging_tests = cfg.persistent_dir / "staging" / "tests"

        # Poll until agent writes the sentinel or the process dies.
        while not agent_done.exists():
            await asyncio.sleep(1)

        src = Path(cfg.problem_info["task_dir"]) / "tests"
        if staging_tests.exists():
            shutil.rmtree(staging_tests)
        shutil.copytree(src, staging_tests)
        tests_ready.touch()

    async def process_single_datapoint(self) -> bool:
        cfg = self.config
        cfg.verifier_dir.mkdir(parents=True, exist_ok=True)
        (cfg.persistent_dir / "staging").mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        ctr = await self._start(cfg.agent_command_str)

        # Stage tests concurrently while the container is running.
        staging_task = asyncio.create_task(self._stage_tests(cfg))
        total_timeout = cfg.tb_agent_timeout + cfg.tb_eval_timeout + 120
        container_timed_out = False
        try:
            await asyncio.wait_for(ctr.process.communicate(), timeout=total_timeout)
        except asyncio.TimeoutError:
            container_timed_out = True
            if ctr.process.returncode is None:
                try:
                    os.killpg(os.getpgid(ctr.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await ctr.process.wait()
            print(f"[{cfg.task_name}] container timed out after {total_timeout}s", flush=True)
        except Exception as e:
            print(f"[{cfg.task_name}] container error: {e}", flush=True)
        finally:
            ctr.log_file.close()
            staging_task.cancel()
            # Clean up staging dir — tests are already run, no need to keep the copy.
            staging_dir = cfg.persistent_dir / "staging"
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

        if ctr.process.returncode not in (0, None):
            print(
                f"[{cfg.task_name}] container exit {ctr.process.returncode}: "
                f"{ctr.log_file_path.read_text(errors='replace')[-2000:]}",
                flush=True,
            )

        total_run_time = time.time() - t0

        # Reconstruct per-phase timing from timestamps written by the script.
        agent_run_time, eval_run_time, agent_timed_out = None, None, False
        spinup_path = cfg.persistent_dir / "agent_spinup_timestamp"
        eval_start_path = cfg.persistent_dir / "eval_start_timestamp"
        if spinup_path.exists():
            try:
                spinup_t = float(spinup_path.read_text().strip())
                if eval_start_path.exists():
                    eval_start_t = float(eval_start_path.read_text().strip())
                    agent_run_time = eval_start_t - spinup_t
                    eval_run_time = max(0.0, total_run_time - agent_run_time)
                    agent_timed_out = agent_run_time >= cfg.tb_agent_timeout - 5
            except (ValueError, OSError):
                pass

        # Read reward written by test.sh.
        reward_path = cfg.verifier_dir / "reward.txt"
        resolved = False
        if reward_path.exists():
            try:
                resolved = float(reward_path.read_text().strip()) > 0
            except (ValueError, OSError):
                pass

        metrics = TerminalBenchMetrics(
            ray_queue_time=time.time() - cfg.ray_queue_timestamp,
            resolved=resolved,
            agent_timed_out=agent_timed_out,
            container_timed_out=container_timed_out,
            mask_sample=bool(container_timed_out or agent_timed_out),
            agent_run_time=agent_run_time,
            eval_run_time=eval_run_time,
            total_run_time=total_run_time,
        )
        update_metrics(cfg.metrics_fpath, metrics.model_dump())
        return resolved


@ray.remote(scheduling_strategy="SPREAD", runtime_env={"py_executable": sys.executable}, num_cpus=0.1)
def _run_remote(params_dict: dict) -> bool:
    AnyTerminalInstanceConfig.model_rebuild(force=True)
    RunTerminalAgent.model_rebuild(force=True)
    params = AnyTerminalInstanceConfig.model_validate(params_dict)
    return asyncio.run(RunTerminalAgent(config=params).process_single_datapoint())


### Agent server


class AnyTerminalAgent(SimpleResponsesAPIAgent):
    """Runs any Gym agent harness inside a Terminal Bench task container."""

    config: AnyTerminalAgentConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _sem: Optional[Semaphore] = None
    _server: Optional[AnyTerminalServerConfig] = None

    def model_post_init(self, context: Any) -> None:
        self._sem = Semaphore(self.config.concurrency)

        model_url = ""
        if self.config.model_server is not None:
            model_cfg = get_first_server_config_dict(
                self.server_client.global_config_dict, self.config.model_server.name
            )
            model_url = self.server_client._build_server_base_url(model_cfg)

        # Real model identifier the policy server serves, set via +policy_model_name=... at run time.
        model_name = str(self.server_client.global_config_dict.get("policy_model_name") or "")

        agent_deps_dir = GymAgentHarnessProcessor(config=self.config).setup()
        session_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        workspace = Path(__file__).parent

        results_dir = workspace / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        self._server = AnyTerminalServerConfig(
            run_session_id=session_id,
            base_results_dir=results_dir / f"anyterminal_results_{session_id}",
            model_server_url=model_url,
            model_name=model_name,
            nemo_gym_root=PARENT_DIR,
            agent_deps_dir=agent_deps_dir,
        )
        super().model_post_init(context)

    # Container resolution

    def _find_container(self, task_name: str, docker_image: str) -> str:
        """Return a pre-built SIF path if available, otherwise a docker:// URI for Apptainer to pull."""
        if self.config.tb_sif_dir:
            sif_dir = Path(self.config.tb_sif_dir)
            for name in [task_name, task_name.replace("-", "_"), task_name.lower()]:
                sif_path = sif_dir / f"{name}.sif"
                if sif_path.exists():
                    return str(sif_path.resolve())
        if not docker_image.startswith("docker://"):
            return f"docker://{docker_image}"
        return docker_image

    @staticmethod
    def _apptainer_exec(
        params: AnyTerminalInstanceConfig,
        mounts: list[str],
        exec_cmd: str,
        env: str = "",
        workdir: Optional[str] = None,
    ) -> str:
        # NOTE: Apptainer cgroup flags (--cpus/--memory) are unusable here — rootless cgroups don't
        # work under --fakeroot ("cannot use cgroups - rootless cgroups is not usable in fakeroot
        # mode"). Instead we apply `ulimit -v` as a generous virtual-memory ceiling (default 32 GB)
        # to prevent runaway tasks from exhausting the host. The task's cpu/memory footprint is also
        # *reserved* in the Ray scheduler (_ray_resource_opts) to avoid host oversubscription.
        pwd_flag = f"--pwd {shlex.quote(workdir)} " if workdir else ""
        cmd = (
            f"apptainer exec --writable-tmpfs --fakeroot --cleanenv --pid --no-mount home,tmp,bind-paths "
            f"{pwd_flag}{env}{' '.join(mounts)} {params.container} {exec_cmd}"
        )
        if params.apptainer_memory_limit_mb > 0:
            cmd = f"ulimit -v {params.apptainer_memory_limit_mb * 1024} && {cmd}"
        return cmd

    def _build_agent_cmd(self, params: AnyTerminalInstanceConfig) -> str:
        # Single container script:
        # 1. Agent runs (tests not visible yet).
        # 2. Agent writes agent_done sentinel → host copies tests into /trajectories_mount/staging/tests/.
        # 3. Script waits for tests_ready sentinel, then runs test.sh from staging.
        script = (
            "#!/bin/bash\n"
            + _DPKG_FIX
            + 'date +"%s.%N" > /trajectories_mount/agent_spinup_timestamp\n'
            + f"{GymAgentHarnessProcessor(config=params).get_run_command()}\n"
            + 'date +"%s.%N" > /trajectories_mount/eval_start_timestamp\n'
            # Signal host that agent is done so it can stage the tests.
            + "touch /trajectories_mount/agent_done\n"
            # Wait for host to stage tests (poll with timeout matching eval budget).
            + f"deadline=$(( $(date +%s) + {params.tb_eval_timeout} ))\n"
            + "while [ ! -f /trajectories_mount/tests_ready ]; do\n"
            + "  [ $(date +%s) -ge $deadline ] && echo 'timed out waiting for tests' && exit 1\n"
            + "  sleep 1\n"
            + "done\n"
            # Symlink /tests into the writable tmpfs so test.sh's hardcoded /tests/... paths work.
            # Remove any pre-existing /tests first: if the agent (or image) already created a
            # /tests directory, `ln -s` would nest the link inside it (/tests/tests) instead of
            # replacing it, leaving test.sh unreachable at /tests/test.sh.
            + "rm -rf /tests\n"
            + "ln -s /trajectories_mount/staging/tests /tests\n"
            # Drop a minimal pytest.ini at / to prevent pytest from picking up Gym's pyproject.toml
            # (host filesystem is overlaid by Apptainer; Gym's config has --pyargs which breaks paths).
            + "printf '[pytest]\\naddopts =\\n' > /pytest.ini\n"
            # Run test.sh from the staged copy (no execute bit needed with bash).
            + "mkdir -p /logs/verifier\n"
            + f"timeout {params.tb_eval_timeout} bash /tests/test.sh"
            + " > /logs/verifier/test-stdout.txt 2>&1 || true\n"
        )
        script_dir = params.persistent_dir / "container_scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        (script_dir / "run_script.sh").write_text(script)

        mounts = [
            f"--mount type=bind,src={params.persistent_dir},dst=/trajectories_mount",
            f"--mount type=bind,src={params.nemo_gym_root},dst=/nemo_gym_mount,ro",
            f"--mount type=bind,src={params.agent_deps_dir},dst=/agent_deps_mount,ro",
            f"--mount type=bind,src={params.verifier_dir},dst=/logs/verifier",
            f"--mount type=bind,src={script_dir / 'run_script.sh'},dst=/container_scripts/run_script.sh,ro",
        ]
        sampling = {
            k: getattr(params.body, k)
            for k in ("temperature", "top_p", "max_output_tokens")
            if getattr(params.body, k, None) is not None
        }
        model_name = params.agent_kwargs.get("model") or params.body.model or "model"
        env = (
            (f"--env NGTB_MODEL_URL={shlex.quote(params.model_server_url)} " if params.model_server_url else "")
            + f"--env NGTB_MODEL_NAME={shlex.quote(model_name)} "
            + f"--env NGTB_AGENT_KWARGS={shlex.quote(json.dumps(params.agent_kwargs))} "
            + f"--env NGTB_SAMPLING={shlex.quote(json.dumps(sampling))} "
        )
        workdir = params.problem_info.get("workdir")
        return self._apptainer_exec(params, mounts, "bash /container_scripts/run_script.sh", env=env, workdir=workdir)

    # Per-instance setup

    def _setup_params(self, body: NeMoGymResponseCreateParamsNonStreaming) -> AnyTerminalInstanceConfig:
        problem_info = dict(body.metadata or {})
        task_name = problem_info.get("task_name", problem_info.get("instance_id", "unknown"))

        # Fill in workdir and timeouts from task.toml/Dockerfile if not in JSONL metadata.
        task_dir = Path(problem_info["task_dir"])
        if not all(k in problem_info for k in ("workdir", "agent_timeout_sec", "verifier_timeout_sec")):
            problem_info.update({k: v for k, v in _read_task_meta(task_dir).items() if k not in problem_info})

        instance_dir = f"{task_name}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        persistent_dir = self._server.base_results_dir / instance_dir
        persistent_dir.mkdir(parents=True, exist_ok=True)
        verifier_dir = persistent_dir / "verifier"
        verifier_dir.mkdir(parents=True, exist_ok=True)

        agent_run_id = f"{task_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        # Per-task timeouts override config defaults when available.
        config_overrides = {}
        if problem_info.get("agent_timeout_sec"):
            config_overrides["tb_agent_timeout"] = int(float(problem_info["agent_timeout_sec"]))
        if problem_info.get("verifier_timeout_sec"):
            config_overrides["tb_eval_timeout"] = int(float(problem_info["verifier_timeout_sec"]))

        params = AnyTerminalInstanceConfig(
            **{**self.config.model_dump(), **config_overrides},
            **self._server.model_dump(),
            problem_info=problem_info,
            body=body,
            persistent_dir=persistent_dir,
            verifier_dir=verifier_dir,
            agent_run_id=agent_run_id,
            metrics_fpath=persistent_dir / "nemo_gym_metrics.json",
            container=self._find_container(task_name, problem_info.get("docker_image", "ubuntu:22.04")),
            ray_queue_timestamp=time.time(),
        )
        params.metrics_fpath.write_text("{}")

        # Write instruction.txt + agent_runner.py, then build the apptainer command.
        params.agent_command_str = self._build_agent_cmd(params)

        return params

    # Request handlers

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        params = self._setup_params(body)
        (params.persistent_dir / "params.json").write_text(_safe_config_json(params, indent=2))
        try:
            return await self._inner_responses(params)
        except Exception:
            tb_path = params.persistent_dir / "traceback.err"
            tb_path.write_text(format_exc())
            print(f"[{params.task_name}] exception: see {tb_path}", file=sys.stderr)
            raise

    @staticmethod
    def _ray_resource_opts(params: AnyTerminalInstanceConfig) -> dict:
        """Reserve the container's cpu/memory footprint in the Ray scheduler so concurrent containers
        don't oversubscribe the host — the main cause of the compute-starved "productive-but-timed-out"
        runs. The launcher task mostly awaits the apptainer subprocess, so these reservations are a
        proxy for the container's real resource use. Since rootless cgroups can't hard-cap the
        container here, this scheduling reservation is our only resource-isolation lever. Memory
        reserves the task's memory_mb + agent_overhead_mb (the in-container Hermes harness)."""

        def _f(key):
            v = params.problem_info.get(key)
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        cpus = _f("cpus")
        mem_mb = _f("memory_mb")
        opts: dict = {"num_cpus": cpus if (cpus and cpus > 0) else 1}
        if mem_mb and mem_mb > 0:
            opts["memory"] = (int(mem_mb) + params.agent_overhead_mb) * 1024 * 1024
        gpus = _f("gpus") or 0
        if gpus > 0:
            opts["num_gpus"] = gpus
        return opts

    async def _inner_responses(self, params: AnyTerminalInstanceConfig) -> NeMoGymResponse:
        await _run_remote.options(**self._ray_resource_opts(params)).remote(params.model_dump())

        persisted = TerminalBenchMetrics.model_validate_json(params.metrics_fpath.read_text())
        mask_sample = bool(persisted.container_timed_out or persisted.agent_timed_out)
        update_metrics(params.metrics_fpath, {"mask_sample": mask_sample})

        response_path = params.persistent_dir / "response.json"
        if response_path.exists():
            data = json.loads(response_path.read_text())
            data["model"] = params.model_name
            saved = NeMoGymResponse.model_validate(data)
            output_items = saved.output
            tools = saved.tools or []
        else:
            output_items, tools = [], []

        return NeMoGymResponse(
            id=f"anyterminal-{params.instance_id}",
            created_at=int(time.time()),
            model=params.model_name,
            object="response",
            output=output_items,
            parallel_tool_calls=params.body.parallel_tool_calls,
            tool_choice=params.body.tool_choice,
            tools=tools,
            metadata={
                "input": json.dumps(params.body.model_dump(mode="json").get("input") or []),
                "metrics": params.metrics_fpath.read_text(),
                "instance_config": _safe_config_json(params),
            },
        )

    async def run(self, body: BaseRunRequest) -> AnyTerminalVerifyResponse:
        async with self._sem:
            body.responses_create_params.parallel_tool_calls = True
            body.responses_create_params.tool_choice = "auto"
            response = await self.responses(body.responses_create_params)

            meta, response.metadata = response.metadata, None
            metrics = TerminalBenchMetrics.model_validate_json(meta["metrics"])

            return AnyTerminalVerifyResponse(
                responses_create_params=body.responses_create_params.model_dump()
                | {
                    "input": json.loads(meta["input"]),
                    "tools": [t.model_dump() for t in (response.tools or [])],
                    "model": response.model,
                },
                response=response,
                reward=1.0 if metrics.resolved else 0.0,
                **metrics.model_dump(),
                instance_config=AnyTerminalInstanceConfig.model_validate_json(meta["instance_config"]).model_dump(),
            )


if __name__ == "__main__":
    AnyTerminalAgent.run_webserver()
