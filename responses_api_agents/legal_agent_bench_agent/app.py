# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a config-selected Gym agent inside a Legal Agent Bench task sandbox."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from fastapi import Body, Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.providers.docker import DockerProvider
from nemo_gym.server_utils import apply_rollout_prefix
from resources_servers.legal_agent_bench.prepare import (
    DEFAULT_RUNTIME_TASKS_DIR,
    DEFAULT_SKILLS_DIR,
    REQUIRED_SKILLS,
    resolve_repo_path,
    validate_harness_skills,
)


PACKAGE_DIR = Path(__file__).resolve().parent
PORTABLE_PYTHON_SH = PACKAGE_DIR / "setup_scripts" / "_portable_python.sh"
DATASET_ALIAS = "legal_agent_bench"
INITIAL_USER_PROMPT = "Please begin working on the task described in the system prompt."
AGENT_CLI_PINS = {
    "claude_code_agent": ("claude_code_version", "CLAUDE_SPEC", "@anthropic-ai/claude-code"),
    "codex_agent": ("codex_version", "CODEX_SPEC", "@openai/codex"),
}

GENERIC_HARNESS_PREAMBLE = """\
You are an AI agent running in an automated Legal Agent Bench evaluation.

## Workspace layout

- Source documents are under `/workspace/vdr` and `$VDR_DIR`. Treat them as read-only.
- Write every final deliverable under `/workspace/output` and `$OUTPUT_DIR`.
- Use `/workspace/workspace` for scratch files.
- Skill manuals and their supporting assets are under `/workspace/skills`.
- Do not search for task configuration, rubric, verifier, test, or judge files. They are intentionally
  unavailable while you work.

Use the terminal and filesystem tools provided by your agent harness. The image already contains the
document tooling required by the skills; do not install packages during the task. Read the skill manuals
included below before creating deliverables.
"""

_RUNNER_SOURCE = r"""#!/usr/bin/env python3
import asyncio
import inspect
import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

sys.path.insert(0, "/agent_source_mount")
os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + "/agent_deps_mount/bin"
os.environ["VDR_DIR"] = "/workspace/vdr"
os.environ["OUTPUT_DIR"] = "/workspace/output"
os.environ["WORKSPACE_DIR"] = "/workspace/workspace"
os.environ["SKILLS_DIR"] = "/workspace/skills"
os.environ["HOME"] = "/workspace/workspace"
os.chdir("/workspace/output")

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient

runner = json.loads(Path("/trajectories_mount/runner.json").read_text())
model_url = runner["model_url"].rstrip("/")
model_url_v1 = model_url if model_url.endswith("/v1") else model_url + "/v1"
status_path = Path("/trajectories_mount/runner_status.json")


def write_status(*, ok, phase, error=None):
    status_path.write_text(json.dumps({"ok": ok, "phase": phase, "error": error}, indent=2))


try:
    parsed_model_url = urlsplit(model_url)
    if not parsed_model_url.hostname:
        raise ValueError(f"Policy model URL has no hostname: {model_url!r}")
    model_port = parsed_model_url.port or (443 if parsed_model_url.scheme == "https" else 80)
    with socket.create_connection(
        (parsed_model_url.hostname, model_port),
        timeout=runner.get("model_connect_timeout_seconds", 10),
    ):
        pass
except Exception as exc:
    message = f"Policy model is unreachable from the LAB sandbox: {model_url} ({type(exc).__name__}: {exc})"
    write_status(ok=False, phase="model_connectivity", error=message)
    raise RuntimeError(message) from exc

try:
    module = __import__(
        runner["agent_server_module"],
        fromlist=[runner["agent_server_class"], runner["agent_config_class"]],
    )
    agent_class = getattr(module, runner["agent_server_class"])
    config_class = getattr(module, runner["agent_config_class"])

    if runner.get("disable_endpoint_metadata_probe"):
        # Hermes probes /models for optional pricing and context metadata.
        # Gym already supplies the selected model and its internal proxy does not
        # expose model discovery, so skip only these optional metadata lookups.
        from agent import model_metadata as hermes_model_metadata
        from agent import usage_pricing as hermes_usage_pricing

        def empty_endpoint_model_metadata(*args, **kwargs):
            return {}

        hermes_model_metadata.fetch_endpoint_model_metadata = empty_endpoint_model_metadata
        hermes_usage_pricing.fetch_endpoint_model_metadata = empty_endpoint_model_metadata

    client = ServerClient.model_construct(global_config_dict={})
    client._build_server_base_url = lambda _cfg: model_url

    base = {
        "host": "0.0.0.0",
        "port": 0,
        "name": "legal_agent_bench_inner_agent",
        "entrypoint": "app.py",
        "resources_server": ResourcesServerRef(name="legal_agent_bench", type="resources_servers"),
        "model_server": ModelServerRef(name="policy_model", type="responses_api_models"),
    }
    kwargs = {key: value for key, value in base.items() if key in config_class.model_fields}
    kwargs.update(runner.get("agent_kwargs") or {})
    config = config_class(**kwargs)
    agent = agent_class(config=config, server_client=client)

    if hasattr(agent, "resolve_model_base_url"):
        object.__setattr__(agent, "resolve_model_base_url", lambda *args, **kwargs: model_url_v1)
    if hasattr(agent, "_resolve_model_base_url"):
        object.__setattr__(agent, "_resolve_model_base_url", lambda *args, **kwargs: model_url_v1)
    if hasattr(agent, "_resolve_base_url"):
        object.__setattr__(agent, "_resolve_base_url", lambda *args, **kwargs: model_url)

    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(runner["responses_create_params"])
    response_kwargs = {"body": body}
    if "request" in inspect.signature(agent.responses).parameters:
        response_kwargs["request"] = SimpleNamespace(path_params={})
    response = asyncio.run(agent.responses(**response_kwargs))
    Path("/trajectories_mount/response.json").write_text(response.model_dump_json())
except Exception as exc:
    write_status(ok=False, phase="agent_execution", error=f"{type(exc).__name__}: {exc}")
    raise
else:
    write_status(ok=True, phase="complete")
"""


class LegalAgentBenchAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    agent_server_module: str
    agent_server_class: str
    agent_config_class: str
    agent_kwargs: dict[str, Any] = Field(default_factory=dict)

    runtime_tasks_dir: str = str(DEFAULT_RUNTIME_TASKS_DIR)
    skills_dir: str = str(DEFAULT_SKILLS_DIR)
    concurrency: int = Field(default=1, ge=1)
    agent_timeout_seconds: int = Field(default=10800, ge=1)
    model_connect_timeout_seconds: int = Field(default=10, ge=1)
    verifier_timeout_seconds: int = Field(default=3600, ge=1)
    runtime_build_timeout_seconds: int = Field(default=3600, ge=1)
    image_build_timeout_seconds: int = Field(default=3600, ge=1)
    sandbox_ttl_seconds: int = Field(default=14400, ge=1)
    docker_network: Optional[str] = "host"
    sandbox_model_base_url: Optional[str] = None
    image_repository: str = "nemo-gym-legal-agent-bench"
    results_dir: str = "results/legal_agent_bench"


class LegalAgentBenchRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    instance_id: str


class LegalAgentBenchAgentResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    instance_id: str
    criteria_pass_rate: float = 0.0
    judge_error_count: int = 0
    verifier_error: int = 0
    mask_sample: bool = False
    agent_failed: bool = False
    model_connection_failed: bool = False
    agent_timed_out: bool = False
    verifier_failed: bool = False
    verifier_timed_out: bool = False
    sandbox_failed: bool = False
    task_failed: bool = False
    configuration_failed: bool = False
    failure_reason: Optional[str] = None
    artifact_dir: Optional[str] = None
    run_summary_path: Optional[str] = None
    agent_trace_path: Optional[str] = None
    agent_stdout_path: Optional[str] = None
    agent_stderr_path: Optional[str] = None
    verifier_report_path: Optional[str] = None
    output_dir: Optional[str] = None


class LegalAgentBenchTaskError(ValueError):
    """The requested LAB task is unsafe, unknown, incomplete, or malformed."""


class LegalAgentBenchConfigurationError(ValueError):
    """The selected Gym agent or its LAB configuration is invalid."""


def agent_key(agent_server_module: str) -> str:
    parts = agent_server_module.split(".")
    if len(parts) < 2 or parts[-1] != "app":
        raise LegalAgentBenchConfigurationError(f"Agent module must end in '.app': {agent_server_module!r}")
    key = parts[-2]
    if not key.replace("_", "").isalnum():
        raise LegalAgentBenchConfigurationError(f"Invalid agent module key: {key!r}")
    return key


def _results_segment(value: str, *, fallback: str) -> str:
    segment = "".join(
        character if character.isascii() and (character.isalnum() or character in "._-") else "-"
        for character in value
    )
    segment = segment.strip("._-")
    return segment[:120] or fallback


def _results_session_dir(
    results_root: Path,
    *,
    agent_server_module: str,
    model_name: str,
    timestamp: float,
    session_id: str,
) -> Path:
    harness = agent_key(agent_server_module).removesuffix("_agent")
    date = time.strftime("%Y%m%d", time.localtime(timestamp))
    clock = time.strftime("%H%M%S", time.localtime(timestamp))
    model = _results_segment(model_name, fallback="unknown_model")
    return results_root / f"{harness}_jobs" / model / f"{date}-{clock}_{session_id}"


def resolve_agent_setup_script(agent_server_module: str) -> Path:
    key = agent_key(agent_server_module)
    script = PARENT_DIR / "responses_api_agents" / key / "scripts" / f"{key}_deps.sh"
    if not script.is_file():
        raise LegalAgentBenchConfigurationError(
            f"Configurable LAB agent {key!r} requires dependency setup script {script.relative_to(PARENT_DIR)}"
        )
    return script


def _recipe_hash(paths: list[Path], *, values: list[str] | None = None) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode())
        if path.is_file():
            digest.update(path.read_bytes())
        elif path.is_dir():
            for child in sorted(item for item in path.rglob("*") if item.is_file()):
                if "__pycache__" in child.parts or child.suffix in {".pyc", ".pyo"}:
                    continue
                digest.update(child.relative_to(path).as_posix().encode())
                digest.update(child.read_bytes())
    for value in values or []:
        digest.update(value.encode())
    return digest.hexdigest()


def agent_runtime_env(agent_server_module: str, agent_kwargs: dict[str, Any]) -> dict[str, str]:
    key = agent_key(agent_server_module)
    pin = AGENT_CLI_PINS.get(key)
    if pin is None:
        return {}
    field, environment_variable, package = pin
    version = agent_kwargs.get(field)
    if not isinstance(version, str) or not version.strip():
        raise LegalAgentBenchConfigurationError(
            f"Configurable LAB agent {key!r} requires a pinned agent_kwargs.{field}"
        )
    return {environment_variable: f"{package}@{version.strip()}"}


def ensure_agent_runtime(
    agent_server_module: str,
    *,
    agent_kwargs: dict[str, Any],
    image: str,
    docker_network: Optional[str],
    timeout_seconds: int,
) -> Path:
    key = agent_key(agent_server_module)
    script = resolve_agent_setup_script(agent_server_module)
    requirements = PARENT_DIR / "responses_api_agents" / key / "requirements.txt"
    runtime_env = agent_runtime_env(agent_server_module, agent_kwargs)
    docker = shutil.which("docker")
    if not docker:
        raise FileNotFoundError("Docker CLI is required to provision Legal Agent Bench agent dependencies")
    image_info = subprocess.run(
        [docker, "image", "inspect", "--format", "{{.Id}}:{{.Os}}/{{.Architecture}}", image],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()
    recipe = _recipe_hash(
        [
            PORTABLE_PYTHON_SH,
            script,
            requirements,
            PARENT_DIR / "pyproject.toml",
            PARENT_DIR / "nemo_gym",
        ],
        values=[image_info, json.dumps(runtime_env, sort_keys=True)],
    )
    deps_dir = PACKAGE_DIR / ".deps" / key
    sentinel = deps_dir / ".installed"
    if sentinel.is_file() and sentinel.read_text().strip() == recipe:
        return deps_dir

    if deps_dir.exists():
        shutil.rmtree(deps_dir)
    deps_dir.mkdir(parents=True)
    env = {
        "PORTABLE_PYTHON_SH": (
            "/nemo_gym_mount/responses_api_agents/legal_agent_bench_agent/setup_scripts/_portable_python.sh"
        ),
        "DEPS_DIR": "/agent_deps",
        "NEMO_GYM_ROOT": "/nemo_gym_mount",
        "HOME": "/tmp",
        **runtime_env,
    }
    command = [
        docker,
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--volume",
        f"{PARENT_DIR}:/nemo_gym_mount:ro",
        "--volume",
        f"{deps_dir}:/agent_deps",
    ]
    if docker_network:
        command.extend(["--network", docker_network])
    for name, value in env.items():
        command.extend(["--env", f"{name}={value}"])
    command.extend([image, "bash", f"/nemo_gym_mount/{script.relative_to(PARENT_DIR).as_posix()}"])
    subprocess.run(command, check=True, timeout=timeout_seconds)
    sentinel.write_text(recipe)
    return deps_dir


def _empty_response(model_name: str) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": f"resp_{uuid4().hex}",
            "created_at": int(time.time()),
            "model": model_name or "policy_model",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    )


def sandbox_model_url(
    model_url: str,
    *,
    docker_network: Optional[str],
    platform_name: Optional[str] = None,
) -> str:
    """Translate host-loopback model URLs into addresses reachable from a Docker sandbox."""
    parsed = urlsplit(model_url)
    hostname = (parsed.hostname or "").lower()
    platform_name = platform_name or sys.platform
    docker_desktop = platform_name == "darwin" or platform_name.startswith("win")
    uses_bridge = docker_network != "host"
    if hostname not in {"localhost", "127.0.0.1", "0.0.0.0", "::1", "::"} or not (docker_desktop or uses_bridge):
        return model_url

    userinfo = parsed.netloc.rsplit("@", 1)[0] + "@" if "@" in parsed.netloc else ""
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit(parsed._replace(netloc=f"{userinfo}host.docker.internal{port}"))


def _response_output_text(response: NeMoGymResponse) -> str:
    parts: list[str] = []
    for item in response.output:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()


def agent_response_failure(response: NeMoGymResponse, agent_server_module: str) -> Optional[str]:
    """Return a harness-failure reason without treating normal task-quality failures as infrastructure."""
    if response.error is not None:
        return f"Agent returned an error response: {response.error}"
    if response.incomplete_details is not None:
        return f"Agent returned an incomplete response: {response.incomplete_details}"
    if not response.output:
        return "Agent produced an empty trajectory"

    key = agent_key(agent_server_module)
    if key == "hermes_agent":
        message_items = [item for item in response.output if getattr(item, "type", None) == "message"]
        synthetic_messages = bool(message_items) and all(
            getattr(item, "prompt_token_ids", None) == [0] and getattr(item, "generation_token_ids", None) == [0]
            for item in message_items
        )
        has_tool_activity = any(getattr(item, "type", None) != "message" for item in response.output)
        if synthetic_messages and not has_tool_activity:
            detail = _response_output_text(response) or "no assistant message"
            return f"Hermes produced no model trajectory: {detail}"

    usage = response.usage
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else 0
    has_tool_activity = any(getattr(item, "type", None) != "message" for item in response.output)
    if total_tokens == 0 and not has_tool_activity and not _response_output_text(response):
        return f"{key} produced no model activity"
    return None


def _task_name(instance_id: str) -> str:
    alias, separator, task_name = instance_id.partition("::")
    if separator != "::" or alias != DATASET_ALIAS or not task_name:
        raise LegalAgentBenchTaskError(f"instance_id must be '{DATASET_ALIAS}::<task_name>', got {instance_id!r}")
    path = PurePosixPath(task_name)
    if len(path.parts) != 1 or path.parts[0] in {"", ".", ".."}:
        raise LegalAgentBenchTaskError(f"Unsafe Legal Agent Bench task name: {task_name!r}")
    return task_name


def resolve_task_dir(runtime_tasks_dir: str | Path, instance_id: str) -> Path:
    root = resolve_repo_path(runtime_tasks_dir)
    task_dir = (root / _task_name(instance_id)).resolve()
    if task_dir.parent != root or not task_dir.is_dir():
        raise LegalAgentBenchTaskError(f"Legal Agent Bench runtime task not found: {task_dir}")
    required = ("instruction.md", "task.json", "task.toml", "documents", "environment", "tests")
    missing = [name for name in required if not (task_dir / name).exists()]
    if missing:
        raise LegalAgentBenchTaskError(
            f"Legal Agent Bench task {task_dir.name} is incomplete; missing: {', '.join(missing)}"
        )
    return task_dir


def _load_skill_prompt(skills_dir: Path) -> str:
    try:
        validate_harness_skills(skills_dir)
        sections = []
        for name in REQUIRED_SKILLS:
            skill_path = skills_dir / name / "SKILL.md"
            sections.append(f"\n\n## Skill: {name}\n\n{skill_path.read_text(encoding='utf-8')}")
    except (FileNotFoundError, ValueError) as exc:
        raise LegalAgentBenchConfigurationError(f"Invalid LAB skills configuration: {exc}") from exc
    return "".join(sections)


def compose_agent_input(
    task_dir: Path,
    skills_dir: Path,
    params: NeMoGymResponseCreateParamsNonStreaming,
) -> NeMoGymResponseCreateParamsNonStreaming:
    try:
        task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        title = task["title"]
        instructions = task["instructions"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise LegalAgentBenchTaskError(f"Invalid LAB task configuration in {task_dir}: {exc}") from exc
    system_prompt = (
        GENERIC_HARNESS_PREAMBLE + _load_skill_prompt(skills_dir) + "\n\n## Task\n\n" + f"# {title}\n\n{instructions}"
    )
    return params.model_copy(
        update={
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": INITIAL_USER_PROMPT},
            ]
        }
    )


def _task_toml(task_dir: Path) -> dict[str, Any]:
    import tomllib

    try:
        with (task_dir / "task.toml").open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise LegalAgentBenchTaskError(f"Invalid LAB task.toml in {task_dir}: {exc}") from exc


def _verifier_env(task_dir: Path) -> dict[str, str]:
    return {str(key): str(value) for key, value in (_task_toml(task_dir).get("verifier", {}).get("env") or {}).items()}


def _sandbox_resources(task_dir: Path) -> SandboxResources:
    environment = _task_toml(task_dir).get("environment") or {}
    return SandboxResources(
        cpu=float(environment["cpus"]) if environment.get("cpus") is not None else None,
        memory_mib=int(environment["memory_mb"]) if environment.get("memory_mb") is not None else None,
        disk_gib=(
            max(1, int(environment["storage_mb"]) // 1024) if environment.get("storage_mb") is not None else None
        ),
        gpu=int(environment.get("gpus") or 0),
    )


def _environment_hash(environment_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in environment_dir.rglob("*") if item.is_file()):
        digest.update(path.relative_to(environment_dir).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


async def _run_process(args: list[str], *, cwd: Path, timeout: int) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        raise TimeoutError(f"Command timed out after {timeout}s: {args}")
    return process.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


class LegalAgentBenchAgent(SimpleResponsesAPIAgent):
    config: LegalAgentBenchAgentConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _sem: asyncio.Semaphore = PrivateAttr()
    _image_lock: asyncio.Lock = PrivateAttr()
    _runtime_lock: asyncio.Lock = PrivateAttr()
    _session_results_dir: Path = PrivateAttr()
    _deps_dir: Optional[Path] = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        self._sem = asyncio.Semaphore(self.config.concurrency)
        self._image_lock = asyncio.Lock()
        self._runtime_lock = asyncio.Lock()
        results_root = resolve_repo_path(self.config.results_dir)
        timestamp = time.time()
        self._session_results_dir = _results_session_dir(
            results_root,
            agent_server_module=self.config.agent_server_module,
            model_name=self._model_name(),
            timestamp=timestamp,
            session_id=uuid4().hex[:8],
        )
        super().model_post_init(context)

    async def responses(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        raise NotImplementedError("LegalAgentBenchAgent is task-driven through /run, not /v1/responses")

    async def _ensure_runtime(self, image: str) -> Path:
        async with self._runtime_lock:
            if self._deps_dir is None:
                self._deps_dir = await asyncio.to_thread(
                    ensure_agent_runtime,
                    self.config.agent_server_module,
                    agent_kwargs=self.config.agent_kwargs,
                    image=image,
                    docker_network=self.config.docker_network,
                    timeout_seconds=self.config.runtime_build_timeout_seconds,
                )
            return self._deps_dir

    async def _ensure_image(self, task_dir: Path) -> str:
        environment_dir = task_dir / "environment"
        image = f"{self.config.image_repository}:{_environment_hash(environment_dir)[:16]}"
        async with self._image_lock:
            docker = shutil.which("docker")
            if not docker:
                raise FileNotFoundError("Docker CLI is required for Legal Agent Bench")
            inspect, _stdout, _stderr = await _run_process(
                [docker, "image", "inspect", image],
                cwd=environment_dir,
                timeout=60,
            )
            if inspect == 0:
                return image
            code, _stdout, stderr = await _run_process(
                [docker, "build", "--tag", image, "."],
                cwd=environment_dir,
                timeout=self.config.image_build_timeout_seconds,
            )
            if code != 0:
                raise RuntimeError(f"Legal Agent Bench image build failed: {stderr[-2000:]}")
        return image

    def _model_url(self, body: LegalAgentBenchRunRequest) -> str:
        if self.config.sandbox_model_base_url:
            return self.config.sandbox_model_base_url
        try:
            model_config = get_first_server_config_dict(
                self.server_client.global_config_dict,
                self.config.model_server.name,
            )
            base_url = self.server_client._build_server_base_url(model_config)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise LegalAgentBenchConfigurationError(
                f"Unable to resolve policy model server {self.config.model_server.name!r}: {exc}"
            ) from exc
        rollout_id = self.rollout_id_from_run(body)
        prefixed_url = apply_rollout_prefix(base_url, rollout_id) if rollout_id else base_url
        return sandbox_model_url(prefixed_url, docker_network=self.config.docker_network)

    def _model_name(self) -> str:
        global_config = getattr(getattr(self, "server_client", None), "global_config_dict", None)
        configured_name = global_config.get("policy_model_name") if isinstance(global_config, Mapping) else None
        return str(configured_name or self.config.model_server.name)

    def _run_dirs(self, task_name: str) -> dict[str, Path]:
        task_segment = _results_segment(task_name, fallback="unknown_task")
        root = self._session_results_dir / f"{task_segment}_{uuid4().hex[:8]}"
        paths = {
            "root": root,
            "runtime": root / "runtime",
            "agent_source": root / "agent_source",
            "agent": root / "agent",
            "lab_run": root / "agent" / "artifacts" / "lab-run",
            "output": root / "agent" / "artifacts" / "lab-run" / "output",
            "workspace": root / "workspace",
            "verifier": root / "verifier",
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        print(f"LAB rollout artifacts: {root}", flush=True)
        return paths

    def _stage_agent_source(self, paths: dict[str, Path]) -> None:
        key = agent_key(self.config.agent_server_module)
        source = PARENT_DIR / "responses_api_agents" / key
        if not source.is_dir():
            raise LegalAgentBenchConfigurationError(f"Configured Gym agent source not found: {source}")
        destination = paths["agent_source"] / "responses_api_agents" / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                ".pytest_cache",
                ".venv",
                ".deps",
                ".claude_node",
                ".codex_node",
                "configs",
                "data",
                "scripts",
                "tests",
            ),
        )

    def _write_runner_config(
        self,
        paths: dict[str, Path],
        params: NeMoGymResponseCreateParamsNonStreaming,
        model_url: str,
    ) -> None:
        (paths["runtime"] / "agent_runner.py").write_text(_RUNNER_SOURCE)
        runner = {
            "agent_server_module": self.config.agent_server_module,
            "agent_server_class": self.config.agent_server_class,
            "agent_config_class": self.config.agent_config_class,
            "agent_kwargs": self.config.agent_kwargs,
            "model_url": model_url,
            "model_connect_timeout_seconds": self.config.model_connect_timeout_seconds,
            "disable_endpoint_metadata_probe": agent_key(self.config.agent_server_module) == "hermes_agent",
            "responses_create_params": params.model_dump(mode="json", exclude_none=True),
        }
        (paths["runtime"] / "runner.json").write_text(json.dumps(runner, indent=2))

    def _provider(self) -> DockerProvider:
        create: dict[str, Any] = {"network": self.config.docker_network, "pids_limit": 4096}
        if sys.platform == "linux" and self.config.docker_network != "host":
            create["extra_run_args"] = ["--add-host", "host.docker.internal:host-gateway"]
        return DockerProvider(
            create=create,
            exec={"default_timeout_s": self.config.agent_timeout_seconds, "concurrency": 8},
        )

    def _agent_sandbox(
        self,
        *,
        image: str,
        task_dir: Path,
        skills_dir: Path,
        deps_dir: Path,
        paths: dict[str, Path],
    ) -> AsyncSandbox:
        volumes = [
            f"{paths['agent_source']}:/agent_source_mount:ro",
            f"{deps_dir}:/agent_deps_mount:ro",
            f"{task_dir / 'documents'}:/workspace/vdr:ro",
            f"{skills_dir}:/workspace/skills:ro",
            f"{paths['runtime']}:/trajectories_mount",
            f"{paths['agent']}:/logs/agent",
            f"{paths['output']}:/workspace/output",
            f"{paths['workspace']}:/workspace/workspace",
        ]
        return AsyncSandbox(
            self._provider(),
            SandboxSpec(
                image=image,
                ttl_s=self.config.sandbox_ttl_seconds,
                workdir="/workspace/output",
                resources=_sandbox_resources(task_dir),
                provider_options={"volumes": volumes},
            ),
        )

    def _verifier_sandbox(
        self,
        *,
        image: str,
        task_dir: Path,
        paths: dict[str, Path],
    ) -> AsyncSandbox:
        return AsyncSandbox(
            self._provider(),
            SandboxSpec(
                image=image,
                ttl_s=self.config.sandbox_ttl_seconds,
                workdir="/logs/agent/artifacts/lab-run/output",
                resources=_sandbox_resources(task_dir),
                provider_options={
                    "volumes": [
                        f"{paths['lab_run']}:/logs/agent/artifacts/lab-run:ro",
                    ]
                },
            ),
        )

    async def _stage_and_run_verifier(
        self,
        sandbox: AsyncSandbox,
        task_dir: Path,
        paths: dict[str, Path],
    ) -> tuple[dict[str, Any], bool, Optional[str]]:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temporary:
            archive_path = Path(temporary.name)
        try:
            with tarfile.open(archive_path, "w:gz") as archive:
                for child in sorted((task_dir / "tests").iterdir(), key=lambda item: item.name):
                    archive.add(child, arcname=child.name)
            await sandbox.upload(archive_path, "/tmp/legal-agent-bench-tests.tar.gz")
        finally:
            archive_path.unlink(missing_ok=True)

        command = (
            "rm -rf /tests /logs/verifier && mkdir -p /tests /logs/verifier && "
            "tar -xzf /tmp/legal-agent-bench-tests.tar.gz -C /tests && "
            "bash /tests/test.sh"
        )
        result = await sandbox.exec(
            command,
            cwd="/workspace/output",
            env=_verifier_env(task_dir),
            timeout_s=self.config.verifier_timeout_seconds,
            user="root",
        )

        for filename in ("reward.json", "scores.json", "transcript.jsonl", "report.html", "error.json"):
            exists = await sandbox.exec(f"test -f /logs/verifier/{filename}", timeout_s=30, user="root")
            if exists.return_code == 0:
                await sandbox.download(f"/logs/verifier/{filename}", paths["verifier"] / filename)

        timed_out = result.error_type == "timeout"
        reward_path = paths["verifier"] / "reward.json"
        if not reward_path.is_file():
            reason = result.stderr or result.stdout or "LAB verifier did not produce reward.json"
            return {}, timed_out, reason[-2000:]
        return json.loads(reward_path.read_text(encoding="utf-8")), timed_out, None

    @staticmethod
    def _runner_status(paths: dict[str, Path]) -> dict[str, Any]:
        status_path = paths["runtime"] / "runner_status.json"
        if not status_path.is_file():
            return {}
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"ok": False, "phase": "agent_execution", "error": f"Invalid runner status: {exc}"}
        return (
            status
            if isinstance(status, dict)
            else {
                "ok": False,
                "phase": "agent_execution",
                "error": "Invalid runner status: expected an object",
            }
        )

    def _artifacts(
        self,
        paths: dict[str, Path],
        *,
        task_name: str,
        model_name: str,
        agent_elapsed: float,
        response: NeMoGymResponse,
        failure_reason: Optional[str],
    ) -> None:
        config = {
            "agent_id": agent_key(self.config.agent_server_module),
            "agent_config_id": self.config.name,
            "model": model_name,
            "task": task_name,
            "run_id": paths["root"].name,
            "tool_runtime": "gym-agent-in-docker",
            "agent_server_module": self.config.agent_server_module,
            "agent_server_class": self.config.agent_server_class,
            "skills": list(REQUIRED_SKILLS),
        }
        metrics = {
            "model": model_name,
            "task": task_name,
            "run_id": paths["root"].name,
            "wall_clock_seconds": round(agent_elapsed, 3),
            "agent_error": failure_reason,
        }
        (paths["lab_run"] / "config.json").write_text(json.dumps(config, indent=2))
        (paths["lab_run"] / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (paths["agent"] / "trajectory.json").write_text(response.model_dump_json(indent=2))

    def _write_run_summary(self, result: LegalAgentBenchAgentResponse) -> None:
        if not result.artifact_dir:
            return
        summary_path = Path(result.artifact_dir) / "run_summary.json"
        output_dir = Path(result.output_dir) if result.output_dir else None
        output_files = (
            sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file())
            if output_dir and output_dir.is_dir()
            else []
        )
        summary = {
            "instance_id": result.instance_id,
            "reward": result.reward,
            "criteria_pass_rate": result.criteria_pass_rate,
            "mask_sample": result.mask_sample,
            "failure_reason": result.failure_reason,
            "flags": {
                "agent_failed": result.agent_failed,
                "model_connection_failed": result.model_connection_failed,
                "agent_timed_out": result.agent_timed_out,
                "verifier_failed": result.verifier_failed,
                "verifier_timed_out": result.verifier_timed_out,
                "sandbox_failed": result.sandbox_failed,
                "task_failed": result.task_failed,
                "configuration_failed": result.configuration_failed,
                "judge_error_count": result.judge_error_count,
                "verifier_error": result.verifier_error,
            },
            "paths": {
                "artifact_dir": result.artifact_dir,
                "agent_trace": result.agent_trace_path,
                "agent_stdout": result.agent_stdout_path,
                "agent_stderr": result.agent_stderr_path,
                "verifier_report": result.verifier_report_path,
                "output_dir": result.output_dir,
            },
            "output_files": output_files,
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        if result.mask_sample:
            print(
                f"LAB rollout failed: {result.failure_reason or 'unreliable result'}; artifacts={result.artifact_dir}",
                flush=True,
            )
        else:
            print(
                f"LAB rollout complete: reward={result.reward} "
                f"criteria_pass_rate={result.criteria_pass_rate} artifacts={result.artifact_dir}",
                flush=True,
            )

    def _response(
        self,
        *,
        body: LegalAgentBenchRunRequest,
        params: NeMoGymResponseCreateParamsNonStreaming,
        response: NeMoGymResponse,
        reward_data: dict[str, Any],
        paths: Optional[dict[str, Path]],
        agent_failed: bool = False,
        model_connection_failed: bool = False,
        agent_timed_out: bool = False,
        verifier_failed: bool = False,
        verifier_timed_out: bool = False,
        sandbox_failed: bool = False,
        task_failed: bool = False,
        configuration_failed: bool = False,
        failure_reason: Optional[str] = None,
    ) -> LegalAgentBenchAgentResponse:
        verifier_error = int(reward_data.get("verifier_error") or 0)
        judge_errors = int(reward_data.get("judge_error_count") or 0)
        unreliable = bool(
            agent_failed
            or model_connection_failed
            or agent_timed_out
            or verifier_failed
            or verifier_timed_out
            or sandbox_failed
            or task_failed
            or configuration_failed
            or verifier_error
            or judge_errors
        )
        return LegalAgentBenchAgentResponse(
            responses_create_params=params,
            response=response,
            reward=float(reward_data.get("reward") or 0.0),
            instance_id=body.instance_id,
            criteria_pass_rate=float(reward_data.get("criteria_pass_rate") or 0.0),
            judge_error_count=judge_errors,
            verifier_error=verifier_error,
            mask_sample=unreliable,
            agent_failed=agent_failed,
            model_connection_failed=model_connection_failed,
            agent_timed_out=agent_timed_out,
            verifier_failed=verifier_failed,
            verifier_timed_out=verifier_timed_out,
            sandbox_failed=sandbox_failed,
            task_failed=task_failed,
            configuration_failed=configuration_failed,
            failure_reason=failure_reason,
            artifact_dir=str(paths["root"]) if paths else None,
            run_summary_path=str(paths["root"] / "run_summary.json") if paths else None,
            agent_trace_path=str(paths["agent"] / "trajectory.json") if paths else None,
            agent_stdout_path=str(paths["agent"] / "stdout.log") if paths else None,
            agent_stderr_path=str(paths["agent"] / "stderr.log") if paths else None,
            verifier_report_path=(
                str(paths["verifier"] / "report.html")
                if paths and (paths["verifier"] / "report.html").is_file()
                else None
            ),
            output_dir=str(paths["output"]) if paths else None,
        )

    async def run(self, request: Request, body: LegalAgentBenchRunRequest) -> LegalAgentBenchAgentResponse:
        async with self._sem:
            model_name = self.config.model_server.name
            params = body.responses_create_params
            response = _empty_response(model_name)
            paths: Optional[dict[str, Path]] = None
            sandbox: Optional[AsyncSandbox] = None
            agent_failed = agent_timed_out = verifier_failed = verifier_timed_out = sandbox_failed = False
            model_connection_failed = False
            task_failed = configuration_failed = False
            failure_reason: Optional[str] = None
            reward_data: dict[str, Any] = {}
            verifier_sandbox: Optional[AsyncSandbox] = None

            try:
                model_name = self._model_name()
                response = _empty_response(model_name)
                task_dir = resolve_task_dir(self.config.runtime_tasks_dir, body.instance_id)
                skills_dir = resolve_repo_path(self.config.skills_dir)
                params = compose_agent_input(task_dir, skills_dir, params)
                paths = self._run_dirs(task_dir.name)
                image = await self._ensure_image(task_dir)
                deps_dir = await self._ensure_runtime(image)
                self._stage_agent_source(paths)
                self._write_runner_config(paths, params, self._model_url(body))

                sandbox = self._agent_sandbox(
                    image=image,
                    task_dir=task_dir,
                    skills_dir=skills_dir,
                    deps_dir=deps_dir,
                    paths=paths,
                )
                await sandbox.start()
                started = time.time()
                agent_result = await sandbox.exec(
                    "/agent_deps_mount/bin/python /trajectories_mount/agent_runner.py",
                    cwd="/workspace/output",
                    timeout_s=self.config.agent_timeout_seconds,
                )
                agent_elapsed = time.time() - started
                (paths["agent"] / "stdout.log").write_text(agent_result.stdout or "")
                (paths["agent"] / "stderr.log").write_text(agent_result.stderr or "")
                agent_timed_out = agent_result.error_type == "timeout"
                runner_status = self._runner_status(paths)
                if runner_status.get("ok") is False:
                    agent_failed = True
                    model_connection_failed = runner_status.get("phase") == "model_connectivity"
                    failure_reason = str(runner_status.get("error") or "Agent runner failed")[-2000:]
                if agent_result.return_code != 0:
                    agent_failed = True
                    failure_reason = (
                        failure_reason or (agent_result.stderr or agent_result.stdout or "Agent runner failed")[-2000:]
                    )
                if agent_timed_out:
                    agent_failed = True
                    failure_reason = failure_reason or f"Agent timed out after {self.config.agent_timeout_seconds}s"

                response_path = paths["runtime"] / "response.json"
                if response_path.is_file():
                    try:
                        response = NeMoGymResponse.model_validate_json(response_path.read_text())
                    except ValueError as exc:
                        agent_failed = True
                        failure_reason = failure_reason or f"Invalid agent response: {exc}"
                else:
                    agent_failed = True
                    failure_reason = failure_reason or "Agent did not produce response.json"
                if not agent_failed:
                    response_failure = agent_response_failure(response, self.config.agent_server_module)
                    if response_failure:
                        agent_failed = True
                        failure_reason = response_failure
                self._artifacts(
                    paths,
                    task_name=task_dir.name,
                    model_name=model_name,
                    agent_elapsed=agent_elapsed,
                    response=response,
                    failure_reason=failure_reason,
                )
                await sandbox.stop()
                sandbox = None

                if not agent_failed:
                    verifier_sandbox = self._verifier_sandbox(image=image, task_dir=task_dir, paths=paths)
                    await verifier_sandbox.start()
                    reward_data, verifier_timed_out, verifier_failure = await self._stage_and_run_verifier(
                        verifier_sandbox, task_dir, paths
                    )
                    verifier_failed = verifier_failure is not None
                    failure_reason = verifier_failure or failure_reason
            except LegalAgentBenchTaskError as exc:
                task_failed = True
                failure_reason = f"{type(exc).__name__}: {exc}"
            except LegalAgentBenchConfigurationError as exc:
                configuration_failed = True
                failure_reason = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                sandbox_failed = True
                failure_reason = f"{type(exc).__name__}: {exc}"
            finally:
                if sandbox is not None:
                    try:
                        await sandbox.stop()
                    except Exception as exc:
                        sandbox_failed = True
                        failure_reason = failure_reason or f"Sandbox cleanup failed: {exc}"
                if verifier_sandbox is not None:
                    try:
                        await verifier_sandbox.stop()
                    except Exception as exc:
                        sandbox_failed = True
                        failure_reason = failure_reason or f"Verifier sandbox cleanup failed: {exc}"

            result = self._response(
                body=body,
                params=params,
                response=response,
                reward_data=reward_data,
                paths=paths,
                agent_failed=agent_failed,
                model_connection_failed=model_connection_failed,
                agent_timed_out=agent_timed_out,
                verifier_failed=verifier_failed,
                verifier_timed_out=verifier_timed_out,
                sandbox_failed=sandbox_failed,
                task_failed=task_failed,
                configuration_failed=configuration_failed,
                failure_reason=failure_reason,
            )
            self._write_run_summary(result)
            return result


if __name__ == "__main__":
    LegalAgentBenchAgent.run_webserver()
