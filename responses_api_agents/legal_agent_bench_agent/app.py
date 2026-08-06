# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a config-selected Gym agent inside a Legal Agent Bench task sandbox."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import stat
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
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY, NG_TERMINAL_KEY
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxResources,
    SandboxSpec,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_utils import apply_rollout_prefix
from resources_servers.legal_agent_bench.prepare import (
    DEFAULT_RUNTIME_TASKS_DIR,
    DEFAULT_SKILLS_DIR,
    REQUIRED_SKILLS,
    resolve_repo_path,
    validate_harness_skills,
)
from resources_servers.legal_agent_bench.vendor.harvey_labs.lab_harbor.tools import (
    get_all_tool_definitions,
)


PACKAGE_DIR = Path(__file__).resolve().parent
PORTABLE_PYTHON_SH = PACKAGE_DIR / "setup_scripts" / "_portable_python.sh"
DATASET_ALIAS = "legal_agent_bench"
INITIAL_USER_PROMPT = "Please begin working on the task described in the system prompt."
NATIVE_AGENT_MODULE = "responses_api_agents.legal_agent_bench_native_agent.app"
AGENT_FAILURE_CLASS_METADATA_KEY = "nemo_gym_failure_class"
PROPAGATED_AGENT_FAILURE_CLASSES = frozenset({"agent_timed_out", "model_connection_failed"})
LAB_SYSTEM_PROMPT = (
    PARENT_DIR / "resources_servers" / "legal_agent_bench" / "vendor" / "harvey_labs" / "harness" / "system-prompt.md"
).read_text(encoding="utf-8")
AGENT_CLI_PINS = {
    "claude_code_agent": ("claude_code_version", "CLAUDE_SPEC", "@anthropic-ai/claude-code"),
    "codex_agent": ("codex_version", "CODEX_SPEC", "@openai/codex"),
}
PINNED_NPM_VERSION = re.compile(
    r"^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
IMMUTABLE_IMAGE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
CODEX_MODEL_CATALOG_PATH = "/trajectories_mount/codex_model_catalog.json"
CODEX_MODEL_CATALOG = {
    "models": [
        {
            "slug": "gym-policy-model",
            "display_name": "Gym policy model",
            "description": "Model selected by the configured Gym policy server.",
            "supported_reasoning_levels": [],
            "shell_type": "default",
            "visibility": "none",
            "supported_in_api": True,
            "priority": 99,
            "availability_nux": None,
            "upgrade": None,
            "base_instructions": "",
            "supports_reasoning_summaries": True,
            "support_verbosity": False,
            "default_verbosity": None,
            "apply_patch_tool_type": None,
            "truncation_policy": {"mode": "bytes", "limit": 10_000},
            "supports_parallel_tool_calls": False,
            "context_window": 272_000,
            "max_context_window": 272_000,
            "experimental_supported_tools": [],
        }
    ]
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
# The inner runner receives its complete configuration below. Prevent Gym's
# shared aiohttp client from invoking Hydra's CLI config loader in the output
# directory when the native agent makes its first model request.
os.environ.setdefault("NEMO_GYM_CONFIG_DICT", "{}")
os.chdir("/workspace/output")

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient

runner = json.loads(Path("/trajectories_mount/runner.json").read_text())
# ECS Fargate injects this value after resolving the host-side policy URL through
# its SSH reverse tunnel. Other providers use the URL persisted in runner.json.
model_url = os.environ.get("LAB_POLICY_MODEL_URL", runner["model_url"]).rstrip("/")
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

    client = ServerClient.model_construct(
        global_config_dict={
            "policy_model": {
                "responses_api_models": {
                    "policy_model": {},
                }
            }
        }
    )
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
    sandbox_provider: str | dict[str, Any] = Field(default_factory=lambda: {"docker": {}})
    sandbox_image: Optional[str] = None
    runtime_docker_platform: Optional[str] = None
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


class LegalAgentBenchArtifactError(RuntimeError):
    """The sandbox returned an unsafe or malformed artifact tree."""


def _provider_name(provider: Mapping[str, Any]) -> str:
    names = list(provider)
    if len(names) != 1:
        raise LegalAgentBenchConfigurationError(
            f"sandbox_provider must resolve to exactly one provider, got {names!r}"
        )
    return str(names[0])


def _create_archive(destination: Path, entries: list[tuple[Path, str]]) -> None:
    """Archive trusted host inputs under explicit sandbox-relative names."""
    with tarfile.open(destination, "w:gz", dereference=False) as archive:
        for source, arcname in entries:
            if not source.exists():
                raise FileNotFoundError(source)
            archive.add(source, arcname=arcname, recursive=True)


def _validate_archive_member(member: tarfile.TarInfo) -> None:
    if member.name in {".", "./"} and member.isdir():
        return
    path = PurePosixPath(member.name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise LegalAgentBenchArtifactError(f"Unsafe sandbox artifact path: {member.name!r}")
    if member.issym() or member.islnk():
        raise LegalAgentBenchArtifactError(f"Sandbox artifacts may not contain links: {member.name!r}")
    if not (member.isdir() or member.isreg()):
        raise LegalAgentBenchArtifactError(
            f"Sandbox artifacts may contain only directories and regular files: {member.name!r}"
        )


def _extract_untrusted_archive(archive_path: Path, destination: Path) -> None:
    """Materialize sandbox output without following links or restoring ownership."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            _validate_archive_member(member)
        archive.extractall(destination, members=members, filter="data")


def _copy_downloaded_file(source: Path, destination: Path) -> None:
    mode = source.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise LegalAgentBenchArtifactError(f"Expected a regular downloaded file, got {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def ensure_runtime_archive(deps_dir: Path) -> Path:
    """Create one immutable, process-safe transport archive for a cached runtime."""
    archive_path = deps_dir.parent / f"{deps_dir.name}.tar.gz"
    if archive_path.is_file():
        return archive_path
    lock_path = deps_dir.parent / f".{deps_dir.name}.archive.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if archive_path.is_file():
                return archive_path
            with tempfile.NamedTemporaryFile(
                suffix=".tar.gz",
                prefix=f".{deps_dir.name}-",
                dir=deps_dir.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                _create_archive(temporary_path, [(deps_dir, "agent_deps_mount")])
                temporary_path.replace(archive_path)
            finally:
                temporary_path.unlink(missing_ok=True)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return archive_path


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
    key = agent_key(agent_server_module)
    harness = "native" if key == "legal_agent_bench_native_agent" else key.removesuffix("_agent")
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
    normalized_version = version.strip() if isinstance(version, str) else ""
    if not PINNED_NPM_VERSION.fullmatch(normalized_version):
        raise LegalAgentBenchConfigurationError(
            f"Configurable LAB agent {key!r} requires agent_kwargs.{field} to be an exact npm version"
        )
    return {environment_variable: f"{package}@{normalized_version}"}


def ensure_agent_runtime(
    agent_server_module: str,
    *,
    agent_kwargs: dict[str, Any],
    image: str,
    docker_network: Optional[str],
    timeout_seconds: int,
    docker_platform: Optional[str] = None,
) -> Path:
    key = agent_key(agent_server_module)
    script = resolve_agent_setup_script(agent_server_module)
    requirements = PARENT_DIR / "responses_api_agents" / key / "requirements.txt"
    runtime_env = agent_runtime_env(agent_server_module, agent_kwargs)
    docker = shutil.which("docker")
    if not docker:
        raise FileNotFoundError("Docker CLI is required to provision Legal Agent Bench agent dependencies")
    if docker_platform:
        subprocess.run(
            [docker, "pull", "--platform", docker_platform, image],
            check=True,
            timeout=timeout_seconds,
        )
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
            PARENT_DIR / "README.md",
            PARENT_DIR / "nemo_gym",
        ],
        values=[image_info, json.dumps(runtime_env, sort_keys=True)],
    )
    cache_root = PACKAGE_DIR / ".deps" / key
    deps_dir = cache_root / recipe
    sentinel = deps_dir / ".installed"
    if sentinel.is_file() and sentinel.read_text().strip() == recipe:
        return deps_dir

    cache_root.mkdir(parents=True, exist_ok=True)
    lock_root = PACKAGE_DIR / ".deps" / ".locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"{key}-{recipe}.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if sentinel.is_file() and sentinel.read_text().strip() == recipe:
                return deps_dir

            with tempfile.TemporaryDirectory(prefix=f".{recipe}-bundle-", dir=cache_root) as bundle_raw:
                bundle = Path(bundle_raw)
                shutil.copy2(PARENT_DIR / "pyproject.toml", bundle / "pyproject.toml")
                shutil.copy2(PARENT_DIR / "README.md", bundle / "README.md")
                shutil.copytree(PARENT_DIR / "nemo_gym", bundle / "nemo_gym")
                staged_agent = bundle / "responses_api_agents" / key
                (staged_agent / "scripts").mkdir(parents=True)
                shutil.copy2(requirements, staged_agent / "requirements.txt")
                shutil.copy2(script, staged_agent / "scripts" / script.name)
                staged_setup = bundle / "responses_api_agents" / "legal_agent_bench_agent" / "setup_scripts"
                staged_setup.mkdir(parents=True)
                shutil.copy2(PORTABLE_PYTHON_SH, staged_setup / PORTABLE_PYTHON_SH.name)

                build_dir = Path(tempfile.mkdtemp(prefix=f".{recipe}-build-", dir=cache_root))
                try:
                    env = {
                        "PORTABLE_PYTHON_SH": (
                            "/nemo_gym_mount/responses_api_agents/legal_agent_bench_agent/"
                            "setup_scripts/_portable_python.sh"
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
                        f"{bundle}:/nemo_gym_mount:ro",
                        "--volume",
                        f"{build_dir}:/agent_deps",
                    ]
                    if docker_platform:
                        command.extend(["--platform", docker_platform])
                    if docker_network:
                        command.extend(["--network", docker_network])
                    for name, value in env.items():
                        command.extend(["--env", f"{name}={value}"])
                    command.extend(
                        [image, "bash", f"/nemo_gym_mount/responses_api_agents/{key}/scripts/{script.name}"]
                    )
                    subprocess.run(command, check=True, timeout=timeout_seconds)
                    (build_dir / ".installed").write_text(recipe)
                    if deps_dir.exists():
                        shutil.rmtree(build_dir)
                    else:
                        build_dir.replace(deps_dir)
                finally:
                    if build_dir.exists():
                        shutil.rmtree(build_dir)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
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


def host_tunnel_model_url(model_url: str) -> str:
    """Translate wildcard listeners into a concrete host endpoint for an ECS reverse tunnel."""
    parsed = urlsplit(model_url)
    if (parsed.hostname or "").lower() not in {"0.0.0.0", "::"}:
        return model_url
    userinfo = parsed.netloc.rsplit("@", 1)[0] + "@" if "@" in parsed.netloc else ""
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit(parsed._replace(netloc=f"{userinfo}127.0.0.1{port}"))


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
    # Max-turn and context-limit stops are valid incomplete model outcomes. The
    # harnesses represent those with ``incomplete_details`` so LAB can still
    # verify and score whatever artifacts the agent produced.
    if response.incomplete_details is not None:
        return None
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


def agent_response_failure_flags(response: NeMoGymResponse, agent_server_module: str) -> tuple[bool, bool]:
    """Return structured model-connection and timeout flags from a failed native response."""
    if agent_server_module != NATIVE_AGENT_MODULE or response.error is None:
        return False, False
    metadata = response.metadata or {}
    failure_class = metadata.get(AGENT_FAILURE_CLASS_METADATA_KEY)
    if failure_class not in PROPAGATED_AGENT_FAILURE_CLASSES:
        return False, False
    return failure_class == "model_connection_failed", failure_class == "agent_timed_out"


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


def native_tool_definitions() -> list[dict[str, Any]]:
    """Translate LAB's canonical tools to OpenAI Responses function tools."""
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
            "strict": False,
        }
        for tool in get_all_tool_definitions()
    ]


def compose_agent_input(
    task_dir: Path,
    skills_dir: Path,
    params: NeMoGymResponseCreateParamsNonStreaming,
    *,
    native: bool = False,
) -> NeMoGymResponseCreateParamsNonStreaming:
    try:
        task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        title = task["title"]
        instructions = task["instructions"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise LegalAgentBenchTaskError(f"Invalid LAB task configuration in {task_dir}: {exc}") from exc
    preamble = LAB_SYSTEM_PROMPT if native else GENERIC_HARNESS_PREAMBLE
    system_prompt = preamble + _load_skill_prompt(skills_dir) + "\n\n## Task\n\n" + f"# {title}\n\n{instructions}"
    update: dict[str, Any] = {
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": INITIAL_USER_PROMPT},
        ]
    }
    if native:
        update.update(
            {
                "tools": native_tool_definitions(),
                "tool_choice": "auto",
                "parallel_tool_calls": False,
            }
        )
    payload = params.model_dump(mode="json")
    payload.update(update)
    return NeMoGymResponseCreateParamsNonStreaming.model_validate(payload)


def _normalized_reward_data(reward_data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate verifier metrics before they can reach response construction or host artifacts."""

    def metric(name: str) -> float:
        value = reward_data.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LegalAgentBenchArtifactError(f"Invalid verifier {name}: expected a number")
        normalized = float(value)
        if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
            raise LegalAgentBenchArtifactError(f"Invalid verifier {name}: expected a finite value in [0, 1]")
        return normalized

    def count(name: str) -> int:
        value = reward_data.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LegalAgentBenchArtifactError(f"Invalid verifier {name}: expected a non-negative integer")
        return value

    normalized = dict(reward_data)
    normalized.update(
        reward=metric("reward"),
        criteria_pass_rate=metric("criteria_pass_rate"),
        verifier_error=count("verifier_error"),
        judge_error_count=count("judge_error_count"),
    )
    return normalized


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
                    docker_platform=self.config.runtime_docker_platform,
                    timeout_seconds=self.config.runtime_build_timeout_seconds,
                )
            return self._deps_dir

    async def _ensure_image(self, task_dir: Path) -> str:
        if self.config.sandbox_image:
            if _provider_name(self._provider_config()) != "docker" and not IMMUTABLE_IMAGE.fullmatch(
                self.config.sandbox_image
            ):
                raise LegalAgentBenchConfigurationError(
                    "Non-Docker LAB sandbox_image must use a complete immutable @sha256 digest"
                )
            return self.config.sandbox_image
        if _provider_name(self._provider_config()) != "docker":
            raise LegalAgentBenchConfigurationError(
                "Non-Docker LAB sandboxes require sandbox_image to reference an immutable registry image"
            )
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
        provider_name = _provider_name(self._provider_config())
        if provider_name not in {"docker", "ecs_fargate"}:
            raise LegalAgentBenchConfigurationError(
                "Non-Docker LAB sandboxes require sandbox_model_base_url reachable from the provider"
            )
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
        if provider_name == "docker":
            return sandbox_model_url(prefixed_url, docker_network=self.config.docker_network)
        return host_tunnel_model_url(prefixed_url)

    def _agent_provider_options(self, model_url: str) -> dict[str, Any]:
        """Return provider-specific routing without exposing policy credentials to the sandbox."""
        if _provider_name(self._provider_config()) != "ecs_fargate" or self.config.sandbox_model_base_url:
            return {}
        return {
            "outside_endpoints": [
                {
                    "url": model_url,
                    "env_var": "LAB_POLICY_MODEL_URL",
                }
            ]
        }

    def _model_name(self) -> str:
        global_config = getattr(getattr(self, "server_client", None), "global_config_dict", None)
        configured_name = global_config.get("policy_model_name") if isinstance(global_config, Mapping) else None
        return str(configured_name or self.config.model_server.name)

    @staticmethod
    def _paths_for_root(root: Path, *, create: bool = False) -> dict[str, Path]:
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
        if create:
            for path in paths.values():
                path.mkdir(parents=True, exist_ok=True)
        return paths

    def _run_root(self, task_name: str) -> Path:
        task_segment = _results_segment(task_name, fallback="unknown_task")
        return self._session_results_dir / f"{task_segment}_{uuid4().hex[:8]}"

    def _run_dirs(self, task_name: str) -> dict[str, Path]:
        root = self._run_root(task_name)
        paths = self._paths_for_root(root, create=True)
        print(f"LAB rollout artifacts: {root}", flush=True)
        return paths

    def _publish_staged_run(self, staged: dict[str, Path], final_root: Path) -> dict[str, Path]:
        final_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staged["root"], final_root)
        print(f"LAB rollout artifacts: {final_root}", flush=True)
        return self._paths_for_root(final_root)

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
        agent_kwargs = dict(self.config.agent_kwargs)
        if agent_key(self.config.agent_server_module) == "codex_agent":
            extra_config = dict(agent_kwargs.get("extra_config") or {})
            if "model_catalog_json" not in extra_config:
                (paths["runtime"] / "codex_model_catalog.json").write_text(json.dumps(CODEX_MODEL_CATALOG, indent=2))
                extra_config["model_catalog_json"] = CODEX_MODEL_CATALOG_PATH
            agent_kwargs["extra_config"] = extra_config
        runner = {
            "agent_server_module": self.config.agent_server_module,
            "agent_server_class": self.config.agent_server_class,
            "agent_config_class": self.config.agent_config_class,
            "agent_kwargs": agent_kwargs,
            "model_url": model_url,
            "model_connect_timeout_seconds": self.config.model_connect_timeout_seconds,
            "disable_endpoint_metadata_probe": agent_key(self.config.agent_server_module) == "hermes_agent",
            "responses_create_params": params.model_dump(mode="json", exclude_none=True),
        }
        (paths["runtime"] / "runner.json").write_text(json.dumps(runner, indent=2))

    def _provider_config(self) -> dict[str, Any]:
        global_config = getattr(getattr(self, "server_client", None), "global_config_dict", None)
        provider = resolve_provider_config(self.config.sandbox_provider, global_config)
        if _provider_name(provider) != "docker":
            return provider

        docker = dict(provider.get("docker") or {})
        create = dict(docker.get("create") or {})
        create.setdefault("network", self.config.docker_network)
        create.setdefault("pids_limit", 4096)
        if sys.platform == "linux" and self.config.docker_network != "host":
            extra_run_args = list(create.get("extra_run_args") or [])
            host_gateway = ["--add-host", "host.docker.internal:host-gateway"]
            if host_gateway[0] not in extra_run_args:
                extra_run_args.extend(host_gateway)
            create["extra_run_args"] = extra_run_args
        execution = dict(docker.get("exec") or {})
        execution.setdefault("default_timeout_s", self.config.agent_timeout_seconds)
        execution.setdefault("concurrency", 8)
        docker.update({"create": create, "exec": execution})
        return {"docker": docker}

    def _sandbox_metadata(self) -> dict[str, Any]:
        global_config = getattr(getattr(self, "server_client", None), "global_config_dict", None)
        return resolve_provider_metadata(self.config.sandbox_provider, global_config)

    def _agent_sandbox(
        self,
        *,
        image: str,
        task_dir: Path,
        skills_dir: Path,
        deps_dir: Path,
        paths: dict[str, Path],
        model_url: str,
    ) -> AsyncSandbox:
        return AsyncSandbox(
            self._provider_config(),
            SandboxSpec(
                image=image,
                ttl_s=self.config.sandbox_ttl_seconds,
                workdir="/workspace/output",
                resources=_sandbox_resources(task_dir),
                metadata=self._sandbox_metadata(),
                provider_options=self._agent_provider_options(model_url),
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
            self._provider_config(),
            SandboxSpec(
                image=image,
                ttl_s=self.config.sandbox_ttl_seconds,
                workdir="/logs/agent/artifacts/lab-run/output",
                resources=_sandbox_resources(task_dir),
                metadata=self._sandbox_metadata(),
            ),
        )

    async def _stage_agent_sandbox(
        self,
        sandbox: AsyncSandbox,
        *,
        task_dir: Path,
        skills_dir: Path,
        deps_dir: Path,
        paths: dict[str, Path],
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temporary:
            archive_path = Path(temporary.name)
        try:
            _create_archive(
                archive_path,
                [
                    (paths["agent_source"], "agent_source_mount"),
                    (task_dir / "documents", "workspace/vdr"),
                    (skills_dir, "workspace/skills"),
                    (paths["runtime"], "trajectories_mount"),
                ],
            )
            await sandbox.upload(archive_path, "/tmp/legal-agent-bench-agent-input.tar.gz")
            await sandbox.upload(ensure_runtime_archive(deps_dir), "/tmp/legal-agent-bench-runtime.tar.gz")
        finally:
            archive_path.unlink(missing_ok=True)

        command = (
            "mkdir -p /agent_source_mount /agent_deps_mount /trajectories_mount "
            "/workspace/vdr /workspace/skills /workspace/output /workspace/workspace && "
            "tar -xzf /tmp/legal-agent-bench-agent-input.tar.gz -C / && "
            "tar -xzf /tmp/legal-agent-bench-runtime.tar.gz -C / && "
            "chmod -R a+rX,a-w /agent_source_mount /agent_deps_mount /workspace/vdr /workspace/skills && "
            "chown -R nobody:nogroup /trajectories_mount /workspace/output /workspace/workspace"
        )
        result = await sandbox.exec(command, timeout_s=300, user="root")
        if result.return_code != 0:
            raise RuntimeError(result.stderr or result.stdout or "Failed to stage LAB agent sandbox")

    async def _collect_agent_sandbox(self, sandbox: AsyncSandbox, download_dir: Path) -> dict[str, Path]:
        downloads: dict[str, Path] = {}
        for filename in ("response.json", "runner_status.json"):
            destination = download_dir / filename
            try:
                await sandbox.download(f"/trajectories_mount/{filename}", destination)
            except Exception:
                continue
            downloads[filename] = destination

        archive_path = download_dir / "output.tar.gz"
        archive_result = await sandbox.exec(
            "tar -czf /tmp/legal-agent-bench-output.tar.gz -C /workspace/output .",
            timeout_s=300,
            user="root",
        )
        if archive_result.return_code != 0:
            raise RuntimeError(archive_result.stderr or archive_result.stdout or "Failed to collect LAB output")
        await sandbox.download("/tmp/legal-agent-bench-output.tar.gz", archive_path)
        downloads["output.tar.gz"] = archive_path
        return downloads

    @staticmethod
    def _materialize_agent_downloads(
        paths: dict[str, Path],
        downloads: dict[str, Path],
        *,
        stdout: str,
        stderr: str,
    ) -> None:
        (paths["agent"] / "stdout.log").write_text(stdout)
        (paths["agent"] / "stderr.log").write_text(stderr)
        for filename in ("response.json", "runner_status.json"):
            source = downloads.get(filename)
            if source is not None:
                _copy_downloaded_file(source, paths["runtime"] / filename)
        output_archive = downloads.get("output.tar.gz")
        if output_archive is None:
            raise LegalAgentBenchArtifactError("Agent sandbox did not return an output archive")
        _extract_untrusted_archive(output_archive, paths["output"])

    async def _stage_and_run_verifier(
        self,
        sandbox: AsyncSandbox,
        task_dir: Path,
        paths: dict[str, Path],
    ) -> tuple[dict[str, bytes], bool, Optional[str]]:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temporary:
            archive_path = Path(temporary.name)
        try:
            _create_archive(
                archive_path,
                [
                    (paths["lab_run"], "logs/agent/artifacts/lab-run"),
                    (task_dir / "tests", "tests"),
                ],
            )
            await sandbox.upload(archive_path, "/tmp/legal-agent-bench-verifier-input.tar.gz")
        finally:
            archive_path.unlink(missing_ok=True)

        command = (
            "rm -rf /tests /logs/agent /logs/verifier && mkdir -p /logs/verifier && "
            "tar -xzf /tmp/legal-agent-bench-verifier-input.tar.gz -C / && "
            "chmod -R a-w /logs/agent && "
            "bash /tests/test.sh"
        )
        result = await sandbox.exec(
            command,
            cwd="/workspace/output",
            env=_verifier_env(task_dir),
            timeout_s=self.config.verifier_timeout_seconds,
            user="root",
        )

        downloaded: dict[str, bytes] = {}
        with tempfile.TemporaryDirectory(prefix="legal-agent-bench-verifier-") as temporary_dir:
            temporary_path = Path(temporary_dir)
            for filename in ("reward.json", "scores.json", "transcript.jsonl", "report.html", "error.json"):
                destination = temporary_path / filename
                try:
                    await sandbox.download(f"/logs/verifier/{filename}", destination)
                except Exception:
                    continue
                if not stat.S_ISREG(destination.lstat().st_mode):
                    return {}, result.error_type == "timeout", f"Verifier returned unsafe {filename}"
                downloaded[filename] = destination.read_bytes()

        timed_out = result.error_type == "timeout"
        if "reward.json" not in downloaded:
            reason = result.stderr or result.stdout or "LAB verifier did not produce reward.json"
            return {}, timed_out, reason[-2000:]
        return downloaded, timed_out, None

    @staticmethod
    def _materialize_verifier_downloads(paths: dict[str, Path], downloaded: dict[str, bytes]) -> dict[str, Any]:
        try:
            reward_data = json.loads(downloaded["reward.json"].decode("utf-8"))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LegalAgentBenchArtifactError(f"Invalid verifier reward.json: {exc}") from exc
        if not isinstance(reward_data, dict):
            raise LegalAgentBenchArtifactError("Invalid verifier reward.json: expected an object")
        normalized = _normalized_reward_data(reward_data)
        for filename, contents in downloaded.items():
            (paths["verifier"] / filename).write_bytes(contents)
        return normalized

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
        try:
            normalized_reward = _normalized_reward_data(reward_data)
        except LegalAgentBenchArtifactError as exc:
            normalized_reward = _normalized_reward_data({})
            verifier_failed = True
            failure_reason = failure_reason or str(exc)
        verifier_error = normalized_reward["verifier_error"]
        judge_errors = normalized_reward["judge_error_count"]
        verifier_failed = verifier_failed or bool(verifier_error or judge_errors)
        if failure_reason is None and judge_errors:
            suffix = "error" if judge_errors == 1 else "errors"
            failure_reason = f"Verifier reported {judge_errors} judge {suffix}"
        elif failure_reason is None and verifier_error:
            failure_reason = "Verifier reported an internal error"
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
        failure_class: Optional[str] = None
        failure_terminal = False
        if task_failed:
            failure_class = "task_failed"
            failure_terminal = True
        elif configuration_failed:
            failure_class = "configuration_failed"
            failure_terminal = True
        elif model_connection_failed:
            failure_class = "model_connection_failed"
        elif sandbox_failed:
            failure_class = "sandbox_failed"
        elif verifier_failed or verifier_timed_out or verifier_error or judge_errors:
            failure_class = "verifier_failed"
        elif agent_timed_out:
            failure_class = "agent_timed_out"
        elif agent_failed:
            failure_class = "agent_failed"

        routing: dict[str, Any] = {}
        if failure_class is not None:
            routing[NG_FAILURE_CLASS_KEY] = failure_class
        if failure_terminal:
            routing[NG_TERMINAL_KEY] = True
        return LegalAgentBenchAgentResponse(
            responses_create_params=params,
            response=response,
            reward=normalized_reward["reward"],
            instance_id=body.instance_id,
            criteria_pass_rate=normalized_reward["criteria_pass_rate"],
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
            **routing,
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
            staged_paths: Optional[dict[str, Path]] = None
            final_root: Optional[Path] = None
            staging_temp: Optional[tempfile.TemporaryDirectory[str]] = None

            try:
                model_name = self._model_name()
                response = _empty_response(model_name)
                task_dir = resolve_task_dir(self.config.runtime_tasks_dir, body.instance_id)
                skills_dir = resolve_repo_path(self.config.skills_dir)
                if self.config.agent_server_module == NATIVE_AGENT_MODULE:
                    params = compose_agent_input(task_dir, skills_dir, params, native=True)
                else:
                    params = compose_agent_input(task_dir, skills_dir, params)
                final_root = self._run_root(task_dir.name)
                staging_temp = tempfile.TemporaryDirectory(prefix="legal-agent-bench-stage-")
                staged_paths = self._paths_for_root(Path(staging_temp.name), create=True)
                image = await self._ensure_image(task_dir)
                deps_dir = await self._ensure_runtime(image)
                self._stage_agent_source(staged_paths)
                model_url = self._model_url(body)
                self._write_runner_config(staged_paths, params, model_url)

                sandbox = self._agent_sandbox(
                    image=image,
                    task_dir=task_dir,
                    skills_dir=skills_dir,
                    deps_dir=deps_dir,
                    paths=staged_paths,
                    model_url=model_url,
                )
                await sandbox.start()
                await self._stage_agent_sandbox(
                    sandbox,
                    task_dir=task_dir,
                    skills_dir=skills_dir,
                    deps_dir=deps_dir,
                    paths=staged_paths,
                )
                with tempfile.TemporaryDirectory(prefix="legal-agent-bench-agent-") as download_raw:
                    started = time.time()
                    agent_result = await sandbox.exec(
                        "/agent_deps_mount/bin/python /trajectories_mount/agent_runner.py",
                        cwd="/workspace/output",
                        timeout_s=self.config.agent_timeout_seconds,
                        user="nobody",
                    )
                    agent_elapsed = time.time() - started
                    downloads = await self._collect_agent_sandbox(sandbox, Path(download_raw))
                    await sandbox.stop()
                    sandbox = None
                    self._materialize_agent_downloads(
                        staged_paths,
                        downloads,
                        stdout=agent_result.stdout or "",
                        stderr=agent_result.stderr or "",
                    )
                paths = self._publish_staged_run(staged_paths, final_root)
                staging_temp.cleanup()
                staging_temp = None
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
                response_model_connection_failed, response_agent_timed_out = agent_response_failure_flags(
                    response, self.config.agent_server_module
                )
                model_connection_failed = model_connection_failed or response_model_connection_failed
                agent_timed_out = agent_timed_out or response_agent_timed_out
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

                if not agent_failed:
                    verifier_sandbox = self._verifier_sandbox(image=image, task_dir=task_dir, paths=paths)
                    await verifier_sandbox.start()
                    verifier_downloads, verifier_timed_out, verifier_failure = await self._stage_and_run_verifier(
                        verifier_sandbox, task_dir, paths
                    )
                    await verifier_sandbox.stop()
                    verifier_sandbox = None
                    if verifier_failure is None:
                        try:
                            reward_data = self._materialize_verifier_downloads(paths, verifier_downloads)
                        except LegalAgentBenchArtifactError as exc:
                            verifier_failure = str(exc)
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
                if staging_temp is not None:
                    if staged_paths is not None and final_root is not None:
                        try:
                            paths = self._publish_staged_run(staged_paths, final_root)
                        except Exception as exc:
                            sandbox_failed = True
                            failure_reason = failure_reason or f"Artifact publication failed: {exc}"
                    staging_temp.cleanup()

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
