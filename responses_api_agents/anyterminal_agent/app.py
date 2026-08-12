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
from urllib.parse import urlsplit, urlunsplit

import ray
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY, NG_TERMINAL_KEY
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox.providers.apptainer import ApptainerProvider
from nemo_gym.sandbox.providers.docker import DockerCreateConfig, DockerProvider
from nemo_gym.sandbox.providers.enroot import EnrootProvider
from nemo_gym.server_utils import apply_rollout_prefix, get_response_json, raise_for_status


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


def _sandbox_model_url(url: str, sandbox_provider: Dict[str, Any]) -> str:
    """Return a model URL reachable from the selected local sandbox.

    Docker Desktop does not expose macOS host loopback as container loopback,
    even when ``--network host`` is requested. Its stable host alias reaches
    the same Gym proxy without changing paths (including rollout prefixes).
    """
    provider = next(iter(sandbox_provider), "docker")
    if not url or provider != "docker" or sys.platform != "darwin":
        return url
    parsed = urlsplit(url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "0.0.0.0"}:
        return url
    netloc = "host.docker.internal"
    if parsed.port is not None:
        netloc += f":{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _parse_allowed_domains(value: Any) -> list[str]:
    """Normalize a task metadata allowlist, including JSON-string metadata values."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return sorted({str(domain).strip().lower().rstrip(".") for domain in value if str(domain).strip()})


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
    agent_failed: bool = False
    container_timed_out: bool = False
    sandbox_failed: bool = False
    mask_sample: bool = False

    ray_queue_time: Optional[float] = None
    agent_run_time: Optional[float] = None
    eval_run_time: Optional[float] = None
    total_run_time: Optional[float] = None


def _failure_routing_from_metrics(metrics: TerminalBenchMetrics) -> Dict[str, Any]:
    """Keep masked harness failures out of the scored rollout file.

    Agent/container timeouts are terminal for this attempt configuration;
    sandbox and agent-process failures are retryable on a later resume.
    """

    if metrics.agent_timed_out or metrics.container_timed_out:
        return {NG_FAILURE_CLASS_KEY: "timeout_exceeded", NG_TERMINAL_KEY: True}
    if metrics.agent_failed or metrics.sandbox_failed or metrics.mask_sample:
        return {NG_FAILURE_CLASS_KEY: "anyterminal_runtime_error"}
    return {}


def update_metrics(metrics_fpath: Path, update_dict: Dict[str, Any]) -> None:
    existing = {k: v for k, v in json.loads(metrics_fpath.read_text()).items() if v is not None}
    update = {k: v for k, v in update_dict.items() if v is not None}
    metrics_fpath.write_text(json.dumps(existing | update))


def _safe_config_json(params: "AnyTerminalInstanceConfig", indent: Optional[int] = None) -> str:
    """Serialize config without secrets."""

    def is_secret_key(key: str) -> bool:
        lowered = key.lower()
        return (
            any(marker in lowered for marker in ("api_key", "secret", "password"))
            or lowered == "token"
            or lowered.endswith(("_token", "-token"))
        )

    def redact(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: ("***" if is_secret_key(key) else redact(item)) for key, item in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    d = json.loads(params.model_dump_json())
    d.pop("agent_command_str", None)
    return json.dumps(redact(d), indent=indent)


def _prepare_agent_source_tree(destination: Path, agent_module: str) -> Path:
    """Stage only runtime Python source needed by the in-sandbox agent.

    Mounting the repository root would also expose benchmark datasets,
    prepared rubrics, prior trajectories, and unrelated task files to the
    policy. Build a small, read-only import tree instead.
    """

    module_parts = agent_module.split(".")
    if not module_parts or not all(part.isidentifier() for part in module_parts):
        raise ValueError(f"agent_server_module must be a dotted Python module: {agent_module!r}")

    def ignore_runtime_noise(_directory: str, names: list[str]) -> set[str]:
        excluded = {"__pycache__", ".pytest_cache", ".venv", "deps", "results", "tests"}
        return {name for name in names if name in excluded or name.endswith((".pyc", ".pyo"))}

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    nemo_source = PARENT_DIR / "nemo_gym"
    if not nemo_source.is_dir():
        raise FileNotFoundError(f"NeMo Gym source package does not exist: {nemo_source}")
    shutil.copytree(nemo_source, destination / "nemo_gym", ignore=ignore_runtime_noise)

    # A conventional module path ends in a file such as ``app``. Copy its
    # containing package plus only the parent package initializers required
    # for imports (for example responses_api_agents/claude_code_agent).
    package_parts = module_parts[:-1]
    package_source = PARENT_DIR.joinpath(*package_parts)
    if package_parts and package_source.is_dir():
        for depth in range(1, len(package_parts)):
            parent_source = PARENT_DIR.joinpath(*package_parts[:depth])
            parent_target = destination.joinpath(*package_parts[:depth])
            parent_target.mkdir(parents=True, exist_ok=True)
            initializer = parent_source / "__init__.py"
            if initializer.is_file():
                shutil.copy2(initializer, parent_target / "__init__.py")
        shutil.copytree(
            package_source,
            destination.joinpath(*package_parts),
            ignore=ignore_runtime_noise,
            dirs_exist_ok=True,
        )
    else:
        module_source = PARENT_DIR.joinpath(*module_parts).with_suffix(".py")
        if not module_source.is_file():
            raise FileNotFoundError(f"agent source module {agent_module!r} is not under the NeMo Gym checkout")
        module_target = destination.joinpath(*module_parts).with_suffix(".py")
        module_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(module_source, module_target)

    return destination


### Agent runner template
# Injected into the task container; imports any agent class and calls responses().

_RUNNER_TEMPLATE = """\
#!/usr/bin/env python3
import asyncio, json, os, sys
from pathlib import Path
from types import SimpleNamespace

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
request = SimpleNamespace(path_params={{"rollout_id": os.environ.get("NGTB_ROLLOUT_ID", "anyterminal")}})
response = asyncio.run(agent.responses(request=request, body=body))
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

        lockfile = PARENT_DIR / "uv.lock"
        recipe_src = b"".join(p.read_bytes() for p in (script, shared, reqs, lockfile) if p.exists()) or b"no-script"
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
        setup_image = getattr(self.config, "agent_deps_setup_image", None)
        if sys.platform == "darwin" and not setup_image:
            raise RuntimeError(
                "AnyTerminal agent dependency setup on macOS requires agent_deps_setup_image; "
                "set it to a Linux image compatible with the task sandbox"
            )

        # Both native Linux setup and the macOS-in-Linux bootstrap must use the
        # repository lock. Installing the editable source directly lets pip
        # resolve newer transitive releases than Gym was tested with (for
        # example a wandb wheel that omitted its vendored wandb_gql module).
        requirements_path = deps_dir / ".bootstrap_requirements.txt"
        export_env = os.environ.copy()
        export_env["UV_PYTHON"] = sys.executable
        export = Popen(
            [
                "uv",
                "export",
                "--quiet",
                "--locked",
                "--no-dev",
                "--no-hashes",
                "--no-emit-project",
                "--no-header",
                "--no-annotate",
                "--output-file",
                str(requirements_path),
            ],
            cwd=PARENT_DIR,
            env=export_env,
        )
        assert export.wait() == 0, f"Failed to export NeMo Gym dependencies from {PARENT_DIR / 'uv.lock'}"

        if sys.platform == "darwin":
            shared_in_container = f"/nemo_gym/{shared.relative_to(PARENT_DIR)}"
            script_in_container = f"/nemo_gym/{script.relative_to(PARENT_DIR)}"
            command = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{deps_dir}:/agent_deps",
                "-v",
                f"{PARENT_DIR}:/nemo_gym:ro",
                "-e",
                f"PORTABLE_PYTHON_SH={shared_in_container}",
                "-e",
                "DEPS_DIR=/agent_deps",
                "-e",
                "NEMO_GYM_ROOT=/nemo_gym",
                "-e",
                "NEMO_GYM_REQUIREMENTS=/agent_deps/.bootstrap_requirements.txt",
                setup_image,
                "bash",
                script_in_container,
            ]
            proc = Popen(command)
        else:
            env = os.environ.copy()
            env.update(
                {
                    "PORTABLE_PYTHON_SH": str(shared),
                    "DEPS_DIR": str(deps_dir),
                    "NEMO_GYM_ROOT": str(PARENT_DIR),
                    "NEMO_GYM_REQUIREMENTS": str(requirements_path),
                }
            )
            proc = Popen(["bash", str(script)], env=env)
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
    resources_server: Optional[ResourcesServerRef] = Field(
        default=None,
        description=(
            "Optional native Gym resources server used to verify the completed sandbox response. "
            "When unset, verification continues to use the task's tests/test.sh."
        ),
    )

    agent_server_module: str = Field(description="Import path to the agent module")
    agent_server_class: str = Field(description="Agent class name")
    agent_config_class: str = Field(description="Agent config class name")
    agent_kwargs: Dict[str, Any] = Field(default_factory=dict)
    agent_deps_setup_image: Optional[str] = Field(
        default=None,
        description=(
            "Linux image used to build the portable in-sandbox agent runtime on macOS. "
            "It should match the task sandbox architecture and provide bash, curl, tar, and xz."
        ),
    )

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
    run_task_tests: bool = Field(
        default=True,
        description=(
            "Run the task-local tests/test.sh after the agent. Disable when a native Gym resources server "
            "performs hidden host-side verification."
        ),
    )
    task_data_mount_path: str = Field(
        default="/data",
        description=(
            "Read-only in-sandbox path for a host directory supplied as responses_create_params.metadata.data_dir. "
            "Host-directory mounts are supported by the Docker and Apptainer providers."
        ),
    )
    include_allowed_domains_in_system_prompt: bool = Field(
        default=False,
        description=(
            "Append metadata.allowed_domains to the in-sandbox agent system prompt. "
            "This makes per-task egress policy visible to generic agent harnesses."
        ),
    )
    agent_overhead_mb: int = 2048  # extra container memory on top of the task's memory_mb for the
    # in-container agent harness
    ray_memory_reservation_mb: Optional[int] = Field(
        default=None,
        description=(
            "Optional total per-task memory reservation used only by the Ray scheduler. "
            "The sandbox keeps its task-declared memory limit; use this when that limit is a "
            "conservative ceiling rather than the task's expected working set."
        ),
    )
    ray_cpu_reservation: Optional[float] = Field(
        default=None,
        description=(
            "Optional per-task CPU reservation used only by the Ray scheduler. "
            "The sandbox keeps its task-declared CPU limit."
        ),
    )
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
    agent_source_dir: Path
    agent_deps_dir: Path
    agent_deps_archive: Optional[Path] = None
    agent_source_archive: Optional[Path] = None


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
    model_config = ConfigDict(extra="allow")

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

    Docker, Apptainer, and Enroot bind the local runtime directories at the paths expected
    by the agent runner. Other providers use the sandbox file API in RunTerminalAgent.
    """
    name = next(iter(params.sandbox_provider), "docker")
    task_data_dir = None
    if params.problem_info.get("data_dir"):
        task_data_dir = Path(params.problem_info["data_dir"]).expanduser().resolve()
        if not task_data_dir.is_dir():
            raise FileNotFoundError(f"task data directory does not exist: {task_data_dir}")
    if name == "apptainer":
        appt = {
            k: v
            for k, v in (params.sandbox_provider.get("apptainer") or {}).items()
            if k in ("exec", "create", "probe")
        }
        exec_cfg = dict(appt.get("exec") or {})
        exec_cfg["default_binds"] = list(exec_cfg.get("default_binds") or []) + [
            f"{params.persistent_dir}:/trajectories_mount",
            f"{params.agent_source_dir}:/nemo_gym_mount:ro",
            f"{params.agent_deps_dir}:/agent_deps_mount:ro",
            f"{params.verifier_dir}:/logs/verifier",
        ]
        if task_data_dir is not None:
            exec_cfg["default_binds"].append(f"{task_data_dir}:{params.task_data_mount_path}:ro")
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
    if name == "enroot":
        enroot = {
            k: v for k, v in (params.sandbox_provider.get("enroot") or {}).items() if k in ("exec", "create", "probe")
        }
        exec_cfg = dict(enroot.get("exec") or {})
        exec_cfg["default_mounts"] = list(exec_cfg.get("default_mounts") or []) + [
            f"{params.persistent_dir}:/trajectories_mount:none:x-create=dir,bind,rw",
            f"{params.agent_source_dir}:/nemo_gym_mount:none:x-create=dir,bind,ro",
            f"{params.agent_deps_dir}:/agent_deps_mount:none:x-create=dir,bind,ro",
            f"{params.verifier_dir}:/logs/verifier:none:x-create=dir,bind,rw",
        ]
        if task_data_dir is not None:
            exec_cfg["default_mounts"].append(
                f"{task_data_dir}:{params.task_data_mount_path}:none:x-create=dir,bind,ro"
            )
        enroot["exec"] = exec_cfg

        # AnyTerminal creates runtime directories and executes the agent as root. Enroot's
        # rootless --root mapping provides that container identity without host privileges.
        create_cfg = dict(enroot.get("create") or {})
        create_cfg.setdefault("remap_root", True)
        enroot["create"] = create_cfg
        return EnrootProvider(**enroot)
    if name != "docker":
        if task_data_dir is not None:
            raise ValueError(
                f"metadata.data_dir read-only mounts are not supported by sandbox provider {name!r}; "
                "use Docker/Apptainer or pre-stage the task data in the provider image/volume"
            )
        return params.sandbox_provider
    extra_run_args = [
        "-v",
        f"{params.persistent_dir}:/trajectories_mount",
        "-v",
        f"{params.agent_source_dir}:/nemo_gym_mount:ro",
        "-v",
        f"{params.agent_deps_dir}:/agent_deps_mount:ro",
        "-v",
        f"{params.verifier_dir}:/logs/verifier",
    ]
    if task_data_dir is not None:
        extra_run_args += ["-v", f"{task_data_dir}:{params.task_data_mount_path}:ro"]
    return DockerProvider(
        create=DockerCreateConfig(
            network=params.docker_network,
            extra_run_args=extra_run_args,
        ),
    )


### Container lifecycle


class RunTerminalAgent(BaseModel):
    """Run an agent in one sandbox, with optional task-local test execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: AnyTerminalInstanceConfig

    @staticmethod
    def _uses_bind_mounts(cfg: AnyTerminalInstanceConfig) -> bool:
        return next(iter(cfg.sandbox_provider), "docker") in {"apptainer", "docker", "enroot"}

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
        if cfg.agent_source_archive is None:
            raise RuntimeError("remote sandbox requires an isolated agent source archive")
        await sandbox.upload(cfg.agent_source_archive, "/tmp/anyterminal-agent-source.tar.gz")
        result = await sandbox.exec(
            "mkdir -p /nemo_gym_mount && tar -xzf /tmp/anyterminal-agent-source.tar.gz -C /nemo_gym_mount",
            timeout_s=300,
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(result.stderr or "failed to extract isolated agent source")
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
        agent_kwargs = dict(cfg.agent_kwargs)
        allowed_domains = _parse_allowed_domains(cfg.problem_info.get("allowed_domains"))
        if cfg.include_allowed_domains_in_system_prompt and allowed_domains:
            policy = (
                "\n\nNetwork policy: only connect to these task-approved domains: "
                + ", ".join(allowed_domains)
                + ". Do not connect to any other domain. Add explicit connection and total timeouts "
                "to every network command (for example, curl --connect-timeout 10 --max-time 60)."
            )
            agent_kwargs["system_prompt"] = str(agent_kwargs.get("system_prompt") or "") + policy

        model_name = agent_kwargs.get("model") or cfg.body.model or "model"
        env = {
            "NGTB_MODEL_NAME": model_name,
            "NGTB_AGENT_KWARGS": json.dumps(agent_kwargs),
            "NGTB_SAMPLING": json.dumps(sampling),
            "NGTB_ROLLOUT_ID": cfg.agent_run_id,
        }
        if next(iter(cfg.sandbox_provider), "docker") == "enroot":
            env.update(
                {
                    "MAMBA_ROOT_PREFIX": "/opt/conda",
                    "PATH": (
                        "/opt/conda/bin:/agent_deps_mount/bin:/usr/local/sbin:/usr/local/bin:"
                        "/usr/sbin:/usr/bin:/sbin:/bin"
                    ),
                }
            )
        if cfg.model_server_url:
            env["NGTB_MODEL_URL"] = _sandbox_model_url(cfg.model_server_url, cfg.sandbox_provider)
        return env

    async def _run_agent(self, sandbox: AsyncSandbox, cfg: AnyTerminalInstanceConfig) -> tuple[float, bool, bool]:
        t0 = time.time()
        result = await sandbox.exec(
            _apt_root_sandbox(cfg) + (cfg.agent_command_str or ""),
            timeout_s=cfg.tb_agent_timeout,
            user="root",
            env=self._agent_env(cfg),
        )
        if result.return_code != 0:
            print(f"[{cfg.task_name}] agent exit {result.return_code}: {(result.stderr or '')[-2000:]}", flush=True)
        timed_out = result.error_type == "timeout"
        return time.time() - t0, timed_out, bool(result.return_code != 0 and not timed_out)

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

    async def process_single_datapoint(self) -> Optional[bool]:
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
        agent_timed_out = agent_failed = container_timed_out = False
        sandbox_failed = False
        agent_run_time = eval_run_time = None
        try:
            await sandbox.start()
            if not self._uses_bind_mounts(cfg):
                await self._stage_remote_runtime(sandbox, cfg)
            agent_run_time, agent_timed_out, agent_failed = await self._run_agent(sandbox, cfg)
            if cfg.run_task_tests:
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
        resolved = None
        if cfg.run_task_tests:
            resolved = False
            if reward_path.exists():
                try:
                    resolved = float(reward_path.read_text().strip()) > 0
                except (ValueError, OSError):
                    resolved = False

        metrics = TerminalBenchMetrics(
            ray_queue_time=time.time() - cfg.ray_queue_timestamp,
            resolved=resolved,
            agent_timed_out=agent_timed_out,
            agent_failed=agent_failed,
            container_timed_out=container_timed_out,
            sandbox_failed=sandbox_failed,
            mask_sample=bool(container_timed_out or agent_timed_out or agent_failed or sandbox_failed),
            agent_run_time=agent_run_time,
            eval_run_time=eval_run_time,
            total_run_time=total_run_time,
        )
        update_metrics(cfg.metrics_fpath, metrics.model_dump())
        return resolved


@ray.remote(scheduling_strategy="SPREAD", runtime_env={"py_executable": sys.executable}, num_cpus=0.1)
def _run_remote(params_dict: dict) -> Optional[bool]:
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
        if next(iter(self.config.sandbox_provider), "docker") not in {"apptainer", "docker", "enroot"}:
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

        agent_source_dir = _prepare_agent_source_tree(
            base_results_dir / "_agent_source",
            self.config.agent_server_module,
        )
        agent_source_archive = None
        if next(iter(self.config.sandbox_provider), "docker") not in {"apptainer", "docker", "enroot"}:
            agent_source_archive = base_results_dir / ".agent-source.tar.gz"
            with tarfile.open(agent_source_archive, "w:gz") as archive:
                archive.add(agent_source_dir, arcname=".")

        self._server = AnyTerminalServerConfig(
            run_session_id=session_id,
            base_results_dir=base_results_dir,
            model_server_url=model_url,
            model_name=model_name,
            nemo_gym_root=PARENT_DIR,
            agent_source_dir=agent_source_dir,
            agent_deps_dir=agent_deps_dir,
            agent_deps_archive=agent_deps_archive,
            agent_source_archive=agent_source_archive,
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

        cpus = params.ray_cpu_reservation
        if cpus is None:
            cpus = _f("cpus")
        mem_mb = params.ray_memory_reservation_mb
        if mem_mb is None:
            task_mem_mb = _f("memory_mb")
            mem_mb = task_mem_mb + params.agent_overhead_mb if task_mem_mb and task_mem_mb > 0 else None
        opts: dict = {"num_cpus": cpus if (cpus and cpus > 0) else 1}
        if mem_mb and mem_mb > 0:
            opts["memory"] = int(mem_mb) * 1024 * 1024
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
        task_dir_value = problem_info.get("task_dir")
        if task_dir_value and not all(
            k in problem_info for k in ("workdir", "agent_timeout_sec", "verifier_timeout_sec")
        ):
            task_dir = Path(task_dir_value)
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
            or persisted.agent_failed
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

            if self.config.resources_server is not None:
                verify_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/verify",
                    json=body.model_dump() | {"response": response.model_dump(mode="json")},
                )
                await raise_for_status(verify_response)
                result = await get_response_json(verify_response)
                result["resolved"] = bool(result.get("reward", 0.0))
                result["agent_metrics"] = metrics.model_dump(mode="json")
                result["mask_sample"] = bool(result.get("mask_sample", False) or metrics.mask_sample)
                result.update(_failure_routing_from_metrics(metrics))
                result["instance_config"] = AnyTerminalInstanceConfig.model_validate_json(
                    meta["instance_config"]
                ).model_dump()
                return AnyTerminalVerifyResponse.model_validate(result)

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
                **_failure_routing_from_metrics(metrics),
                instance_config=AnyTerminalInstanceConfig.model_validate_json(meta["instance_config"]).model_dump(),
            )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        if self.config.resources_server is None:
            return await super().aggregate_metrics(body)
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    AnyTerminalAgent.run_webserver()
