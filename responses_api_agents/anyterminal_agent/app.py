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
import shutil
import sys
import tarfile
import tempfile
import time
import uuid
from asyncio import Semaphore
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
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox.providers.apptainer import ApptainerProvider
from nemo_gym.sandbox.providers.docker import DockerCreateConfig, DockerProvider
from nemo_gym.server_utils import apply_rollout_prefix


def _format_container(container_formatter: str | list[str], task_name: str, docker_image: str) -> str:
    """Resolve the pullable/local image reference for a task from a formatter template."""

    fmt = container_formatter[0] if isinstance(container_formatter, list) else container_formatter
    fmt = fmt or "docker://{docker_image}"
    docker_image = docker_image[len("docker://") :] if docker_image.startswith("docker://") else docker_image
    if fmt.endswith(".sif") or fmt.startswith(("/", ".")):
        return fmt.format(task_name=task_name, docker_image=docker_image)
    if fmt.startswith("docker://"):
        fmt = fmt[len("docker://") :]
    return f"docker://{fmt.format(task_name=task_name, docker_image=docker_image)}"


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


### Metrics


class TerminalBenchMetrics(BaseModel):
    resolved: Optional[bool] = None
    agent_timed_out: bool = False
    container_timed_out: bool = False
    sandbox_failed: bool = False
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
    """Serialize config without secrets."""

    def redact(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: (
                    "***"
                    if any(secret in key.lower() for secret in ("api_key", "secret", "password", "token"))
                    else redact(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    d = json.loads(params.model_dump_json())
    d.pop("agent_command_str", None)
    return json.dumps(redact(d), indent=indent)


### Agent runner template
# Injected into the task container; imports any agent class and calls responses().

_RUNNER_TEMPLATE = """\
#!/usr/bin/env python3
import asyncio, json, os, sys
from pathlib import Path

sys.path.insert(0, "/nemo_gym_mount")
# Append (not prepend) agent-deps bin so the task's own python/pip win — else the agent's
# builds/installs land in a Python the verifier can't see. Harness CLIs stay findable as a fallback.
os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + "/agent_deps_mount/bin"

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
    _v1 = MODEL_URL if MODEL_URL.endswith("/v1") else MODEL_URL + "/v1"
    if hasattr(agent, "resolve_model_base_url"):
        object.__setattr__(agent, "resolve_model_base_url", lambda *args, **kwargs: _v1)
    if hasattr(agent, "_resolve_model_base_url"):
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
        agent_dir = PARENT_DIR / "responses_api_agents" / self._agent_key
        deps_dir = self._parent / "deps" / f"anyterminal_{self._agent_key}_deps"
        sentinel = deps_dir / ".installed"
        script = agent_dir / "scripts" / f"{self._agent_key}_deps.sh"
        shared = self._parent / "setup_scripts" / "_portable_python.sh"
        reqs = agent_dir / "requirements.txt"

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
        proc = Popen(
            f"PORTABLE_PYTHON_SH={shared} DEPS_DIR={deps_dir} NEMO_GYM_ROOT={PARENT_DIR} bash {script}", shell=True
        )
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
        return "/agent_deps_mount/bin/python /trajectories_mount/agent_runner.py"


### Configuration


class AnyTerminalAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: Optional[ModelServerRef] = None

    agent_server_module: str = Field(description="Import path to the agent module")
    agent_server_class: str = Field(description="Agent class name")
    agent_config_class: str = Field(description="Agent config class name")
    agent_kwargs: Dict[str, Any] = Field(default_factory=dict)

    container_formatter: str | list[str] = Field(
        default="docker://{docker_image}",
        description="Template for the task's image reference: use as a path if it ends with .sif or starts with / or ., else as a docker:// URI.",
    )
    sandbox_provider: Dict[str, Any] = Field(default_factory=lambda: {"docker": {}})
    # Docker network for the agent container. "host" lets the in-container agent reach a
    # model server on host loopback; None uses the docker default (e.g. for a remote server).
    docker_network: Optional[str] = "host"
    sandbox_model_base_url: Optional[str] = None
    tb_agent_timeout: int = 1800
    tb_eval_timeout: int = 300
    tb_sandbox_ttl: int = 7200
    agent_overhead_mb: int = 2048  # extra container memory on top of the task's memory_mb for the
    # in-container agent harness
    concurrency: int = 256
    results_dir: Optional[Path] = None


class AnyTerminalRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class AnyTerminalServerConfig(BaseModel):
    run_session_id: str
    base_results_dir: Path
    model_server_url: str
    model_name: str = ""
    nemo_gym_root: Path
    agent_deps_dir: Path
    agent_deps_archive: Optional[Path] = None


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


### Sandbox provider selection


def _apt_root_sandbox(cfg: AnyTerminalInstanceConfig) -> str:
    # apt drops root to _apt before fetching; fakeroot's single-ID userns can't setgid to it.
    if next(iter(cfg.sandbox_provider), "docker") != "apptainer":
        return ""
    return (
        "mkdir -p /etc/apt/apt.conf.d && printf 'APT::Sandbox::User \"root\";\\n' "
        "> /etc/apt/apt.conf.d/99nemo-gym-apt-root; "
    )


def _build_provider(params: AnyTerminalInstanceConfig):
    """Build a sandbox provider with the per-instance mounts the run needs.

    Docker and Apptainer bind the local runtime directories at the paths expected by the
    agent runner. Other providers use the sandbox file API in RunTerminalAgent.
    """
    name = next(iter(params.sandbox_provider), "docker")
    if name == "apptainer":
        appt = {
            k: v
            for k, v in (params.sandbox_provider.get("apptainer") or {}).items()
            if k in ("exec", "create", "probe")
        }
        exec_cfg = dict(appt.get("exec") or {})
        exec_cfg["default_binds"] = list(exec_cfg.get("default_binds") or []) + [
            f"{params.persistent_dir}:/trajectories_mount",
            f"{params.nemo_gym_root}:/nemo_gym_mount:ro",
            f"{params.agent_deps_dir}:/agent_deps_mount:ro",
            f"{params.verifier_dir}:/logs/verifier",
        ]
        exec_cfg["extra_exec_args"] = list(exec_cfg.get("extra_exec_args") or []) + [
            "--cleanenv",
            "--pid",
            "--no-mount",
            "tmp",
        ]
        appt["exec"] = exec_cfg

        create_cfg = dict(appt.get("create") or {})
        start_args = list(create_cfg.get("extra_start_args") or [])
        if "--writable-tmpfs" not in start_args:
            start_args.append("--writable-tmpfs")
        if "--no-mount" not in start_args:
            start_args += ["--no-mount", "home"]
        create_cfg["extra_start_args"] = start_args
        appt["create"] = create_cfg
        return ApptainerProvider(**appt)
    if name != "docker":
        return params.sandbox_provider
    return DockerProvider(
        create=DockerCreateConfig(
            network=params.docker_network,
            extra_run_args=[
                "-v",
                f"{params.persistent_dir}:/trajectories_mount",
                "-v",
                f"{params.nemo_gym_root}:/nemo_gym_mount:ro",
                "-v",
                f"{params.agent_deps_dir}:/agent_deps_mount:ro",
                "-v",
                f"{params.verifier_dir}:/logs/verifier",
            ],
        ),
    )


### Container lifecycle


class RunTerminalAgent(BaseModel):
    """Single sandbox: agent runs, host stages tests, sandbox runs test.sh."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: AnyTerminalInstanceConfig

    @staticmethod
    def _uses_bind_mounts(cfg: AnyTerminalInstanceConfig) -> bool:
        return next(iter(cfg.sandbox_provider), "docker") in {"apptainer", "docker"}

    @staticmethod
    def _archive(source: Path) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temporary:
            archive = Path(temporary.name)
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(source, arcname=".")
        return archive

    async def _stage_remote_runtime(self, sandbox: AsyncSandbox, cfg: AnyTerminalInstanceConfig) -> None:
        result = await sandbox.exec(
            "mkdir -p /agent_deps_mount /trajectories_mount /logs/verifier",
            timeout_s=30,
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(result.stderr or "failed to create sandbox runtime directories")
        await sandbox.upload(cfg.persistent_dir / "instruction.txt", "/trajectories_mount/instruction.txt")
        await sandbox.upload(cfg.persistent_dir / "agent_runner.py", "/trajectories_mount/agent_runner.py")
        if cfg.agent_deps_archive is None:
            raise RuntimeError("remote sandbox requires an agent runtime archive")
        await sandbox.upload(cfg.agent_deps_archive, "/tmp/anyterminal-agent-deps.tar.gz")
        result = await sandbox.exec(
            "tar -xzf /tmp/anyterminal-agent-deps.tar.gz -C /agent_deps_mount",
            timeout_s=900,
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(result.stderr or "failed to extract agent runtime")

    async def _stage_remote_tests(self, sandbox: AsyncSandbox, cfg: AnyTerminalInstanceConfig) -> None:
        archive = await asyncio.to_thread(self._archive, cfg.persistent_dir / "staging" / "tests")
        try:
            await sandbox.upload(archive, "/tmp/anyterminal-tests.tar.gz")
            result = await sandbox.exec(
                "mkdir -p /trajectories_mount/staging/tests && "
                "tar -xzf /tmp/anyterminal-tests.tar.gz -C /trajectories_mount/staging/tests",
                timeout_s=300,
                user="root",
            )
            if result.return_code != 0:
                raise RuntimeError(result.stderr or "failed to stage verifier tests")
        finally:
            archive.unlink(missing_ok=True)

    async def _collect_remote_outputs(self, sandbox: AsyncSandbox, cfg: AnyTerminalInstanceConfig) -> None:
        for remote, local in (
            ("/trajectories_mount/response.json", cfg.persistent_dir / "response.json"),
            ("/logs/verifier/reward.txt", cfg.verifier_dir / "reward.txt"),
            ("/logs/verifier/test-stdout.txt", cfg.verifier_dir / "test-stdout.txt"),
        ):
            exists = await sandbox.exec(f"test -f {remote}", timeout_s=30, user="root")
            if exists.return_code == 0:
                await sandbox.download(remote, local)

    def _agent_env(self, cfg: AnyTerminalInstanceConfig) -> Dict[str, str]:
        sampling = {
            k: getattr(cfg.body, k)
            for k in ("temperature", "top_p", "max_output_tokens")
            if getattr(cfg.body, k, None) is not None
        }
        model_name = cfg.agent_kwargs.get("model") or cfg.body.model or "model"
        env = {
            "NGTB_MODEL_NAME": model_name,
            "NGTB_AGENT_KWARGS": json.dumps(cfg.agent_kwargs),
            "NGTB_SAMPLING": json.dumps(sampling),
        }
        if cfg.model_server_url:
            env["NGTB_MODEL_URL"] = cfg.model_server_url
        return env

    async def _run_agent(self, sandbox: AsyncSandbox, cfg: AnyTerminalInstanceConfig) -> tuple[float, bool]:
        t0 = time.time()
        result = await sandbox.exec(
            _apt_root_sandbox(cfg) + (cfg.agent_command_str or ""),
            timeout_s=cfg.tb_agent_timeout,
            user="root",
            env=self._agent_env(cfg),
        )
        if result.return_code != 0:
            print(f"[{cfg.task_name}] agent exit {result.return_code}: {(result.stderr or '')[-2000:]}", flush=True)
        return time.time() - t0, result.error_type == "timeout"

    async def _stage_tests(self, cfg: AnyTerminalInstanceConfig) -> None:
        """Copy the task's test files into the staging dir, visible to the sandbox at /tests."""
        src = Path(cfg.problem_info["task_dir"]) / "tests"
        staging_tests = cfg.persistent_dir / "staging" / "tests"
        if staging_tests.exists():
            shutil.rmtree(staging_tests)
        shutil.copytree(src, staging_tests)

    async def _run_eval(self, sandbox: AsyncSandbox, cfg: AnyTerminalInstanceConfig) -> tuple[float, bool]:
        t0 = time.time()
        test_cmd = (
            "rm -rf /tests && ln -s /trajectories_mount/staging/tests /tests && "
            # A minimal pytest.ini at / stops pytest from picking up Gym's pyproject.toml
            # (whose --pyargs breaks paths inside the sandbox).
            "printf '[pytest]\\naddopts =\\n' > /pytest.ini && "
            "mkdir -p /logs/verifier && bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1"
        )
        result = await sandbox.exec(_apt_root_sandbox(cfg) + test_cmd, timeout_s=cfg.tb_eval_timeout, user="root")
        if result.return_code != 0:
            print(f"[{cfg.task_name}] eval exit {result.return_code}: {(result.stderr or '')[-2000:]}", flush=True)
        return time.time() - t0, result.error_type == "timeout"

    async def process_single_datapoint(self) -> bool:
        cfg = self.config
        cfg.verifier_dir.mkdir(parents=True, exist_ok=True)
        (cfg.persistent_dir / "staging").mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        sandbox = AsyncSandbox(
            _build_provider(cfg),
            SandboxSpec(
                image=cfg.container.removeprefix("docker://") if not self._uses_bind_mounts(cfg) else cfg.container,
                ttl_s=cfg.tb_sandbox_ttl,
                workdir=cfg.problem_info.get("workdir"),
            ),
        )
        agent_timed_out = container_timed_out = False
        sandbox_failed = False
        agent_run_time = eval_run_time = None
        try:
            await sandbox.start()
            if not self._uses_bind_mounts(cfg):
                await self._stage_remote_runtime(sandbox, cfg)
            agent_run_time, agent_timed_out = await self._run_agent(sandbox, cfg)
            await self._stage_tests(cfg)
            if not self._uses_bind_mounts(cfg):
                await self._stage_remote_tests(sandbox, cfg)
            eval_run_time, container_timed_out = await self._run_eval(sandbox, cfg)
            if not self._uses_bind_mounts(cfg):
                await self._collect_remote_outputs(sandbox, cfg)
        except Exception as e:
            sandbox_failed = True
            print(f"[{cfg.task_name}] sandbox run failed: {e}", flush=True)
        finally:
            try:
                await sandbox.stop()
            except Exception as e:
                sandbox_failed = True
                print(f"[{cfg.task_name}] sandbox cleanup failed: {e}", flush=True)
            shutil.rmtree(cfg.persistent_dir / "staging", ignore_errors=True)

        total_run_time = time.time() - t0

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
            sandbox_failed=sandbox_failed,
            mask_sample=bool(container_timed_out or agent_timed_out or sandbox_failed),
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
    """Runs any Gym agent harness inside a Terminal Bench task sandbox."""

    config: AnyTerminalAgentConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _sem: Optional[Semaphore] = None
    _server: Optional[AnyTerminalServerConfig] = None

    def model_post_init(self, context: Any) -> None:
        self._sem = Semaphore(self.config.concurrency)

        model_url = self.config.sandbox_model_base_url or ""
        if self.config.model_server is not None:
            model_cfg = get_first_server_config_dict(
                self.server_client.global_config_dict, self.config.model_server.name
            )
            if not model_url:
                model_url = self.server_client._build_server_base_url(model_cfg)

        # Real model identifier the policy server serves, set via +policy_model_name=... at run time.
        model_name = str(self.server_client.global_config_dict.get("policy_model_name") or "")

        workspace = Path(__file__).parent
        agent_deps_dir = GymAgentHarnessProcessor(config=self.config).setup()
        agent_deps_archive = None
        if next(iter(self.config.sandbox_provider), "docker") not in {"apptainer", "docker"}:
            agent_deps_archive = workspace / f".{agent_deps_dir.name}.tar.gz"
            sentinel = agent_deps_dir / ".installed"
            if not agent_deps_archive.exists() or agent_deps_archive.stat().st_mtime < sentinel.stat().st_mtime:
                temporary = agent_deps_archive.with_suffix(".tmp")
                with tarfile.open(temporary, "w:gz") as archive:
                    archive.add(agent_deps_dir, arcname=".")
                temporary.replace(agent_deps_archive)
        results_dir = workspace / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        base_results_dir = self.config.results_dir
        if base_results_dir is None:
            session_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            base_results_dir = results_dir / f"anyterminal_results_{session_id}"
        else:
            session_id = base_results_dir.name
        base_results_dir.mkdir(parents=True, exist_ok=True)

        self._server = AnyTerminalServerConfig(
            run_session_id=session_id,
            base_results_dir=base_results_dir,
            model_server_url=model_url,
            model_name=model_name,
            nemo_gym_root=PARENT_DIR,
            agent_deps_dir=agent_deps_dir,
            agent_deps_archive=agent_deps_archive,
        )
        super().model_post_init(context)

    @staticmethod
    def _ray_resource_opts(params: AnyTerminalInstanceConfig) -> dict:
        """Reserve the container's cpu/memory footprint in the Ray scheduler so concurrent containers
        don't oversubscribe the host — the main cause of the compute-starved "productive-but-timed-out"
        runs. The launcher task mostly awaits the sandbox exec calls, so these reservations are a
        proxy for the container's real resource use. Memory reserves the task's memory_mb +
        agent_overhead_mb (the in-container agent harness)."""

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

    # Per-instance setup

    def _setup_params(
        self, body: NeMoGymResponseCreateParamsNonStreaming, rollout_id: Optional[str] = None
    ) -> AnyTerminalInstanceConfig:
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

        server_config = self._server.model_dump()
        if rollout_id and server_config["model_server_url"]:
            server_config["model_server_url"] = apply_rollout_prefix(server_config["model_server_url"], rollout_id)

        params = AnyTerminalInstanceConfig(
            **{**self.config.model_dump(), **config_overrides},
            **server_config,
            problem_info=problem_info,
            body=body,
            persistent_dir=persistent_dir,
            verifier_dir=verifier_dir,
            agent_run_id=agent_run_id,
            metrics_fpath=persistent_dir / "nemo_gym_metrics.json",
            container=_format_container(
                self.config.container_formatter, task_name, problem_info.get("docker_image", "ubuntu:22.04")
            ),
            ray_queue_timestamp=time.time(),
        )
        params.metrics_fpath.write_text("{}")

        # Write instruction.txt + agent_runner.py, then resolve the in-sandbox run command.
        params.agent_command_str = GymAgentHarnessProcessor(config=params).get_run_command()

        return params

    # Request handlers

    async def _responses(
        self, body: NeMoGymResponseCreateParamsNonStreaming, rollout_id: Optional[str] = None
    ) -> NeMoGymResponse:
        params = self._setup_params(body, rollout_id)
        (params.persistent_dir / "params.json").write_text(_safe_config_json(params, indent=2))
        try:
            return await self._inner_responses(params)
        except Exception:
            tb_path = params.persistent_dir / "traceback.err"
            tb_path.write_text(format_exc())
            print(f"[{params.task_name}] exception: see {tb_path}", file=sys.stderr)
            raise

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        return await self._responses(body)

    async def _inner_responses(self, params: AnyTerminalInstanceConfig) -> NeMoGymResponse:
        await _run_remote.options(**self._ray_resource_opts(params)).remote(params.model_dump())

        persisted = TerminalBenchMetrics.model_validate_json(params.metrics_fpath.read_text())
        mask_sample = bool(
            persisted.mask_sample
            or persisted.container_timed_out
            or persisted.agent_timed_out
            or persisted.sandbox_failed
        )
        update_metrics(params.metrics_fpath, {"mask_sample": mask_sample})

        response_path = params.persistent_dir / "response.json"
        output_items, tools = [], []
        if response_path.exists():
            try:
                data = json.loads(response_path.read_text())
                data["model"] = params.model_name
                saved = NeMoGymResponse.model_validate(data)
                output_items = saved.output
                tools = saved.tools or []
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[{params.task_name}] response.json unreadable ({e}), treating as empty response", flush=True)

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

    async def run(self, body: AnyTerminalRunRequest) -> AnyTerminalVerifyResponse:
        async with self._sem:
            body.responses_create_params.parallel_tool_calls = True
            body.responses_create_params.tool_choice = "auto"
            response = await self._responses(body.responses_create_params, self.rollout_id_from_run(body))

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
