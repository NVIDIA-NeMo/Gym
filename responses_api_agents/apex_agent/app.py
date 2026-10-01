# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym agent that runs Stirrup in a pinned Archipelago task sandbox."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlsplit

from fastapi import Body, Request
from pydantic import ConfigDict, Field

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, resolve_provider_config
from nemo_gym.sandbox.config import resolve_provider_metadata
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.apex_agent.runtime_bootstrap import STIRRUP_PREFLIGHT, stirrup_runtime_bootstrap_script
from responses_api_agents.apex_agent.runtime_setup import (
    ApexImageBuildConfig,
    resolve_image,
    stirrup_cache_path,
)
from responses_api_agents.apex_agent.stirrup_runtime import (
    RESUME_MANIFEST_FILENAME,
    RelayServer,
    ResumeCheckpoint,
    load_resume_checkpoint,
    partial_result_from_checkpoint,
    serve_unix_to_tcp,
)


LOG = logging.getLogger(__name__)
_RUNNER_PATH = Path(__file__).with_name("sandbox_entrypoint.py")
_PREBUILT_RUNNER_PATH = Path(__file__).with_name("prebuilt_world_entrypoint.py")
_EGRESS_MOUNT = "/egress"
_EGRESS_SOCKET_NAME = "policy.sock"
_STIRRUP_RUNTIME_PATH = Path(__file__).with_name("stirrup_runtime.py")
_STIRRUP_SETUP_PATH = Path(__file__).with_name("setup_stirrup.sh")
_STIRRUP_REQUIREMENTS_PATH = Path(__file__).with_name("stirrup-requirements.txt")
_GUEST_ROOT = "/app/apex-gym"
_STIRRUP_ROOT = "/app/stirrup-runtime"
_GUEST_PARTIAL_RESULT_PATH = "/sandbox/partial_result.json"
# Prebuilt-world logs kept for failed rollouts; the failure message only carries their last few kilobytes.
_GUEST_WORLD_LOGS = {
    "environment.log": f"{_GUEST_ROOT}/output/environment.log",
    "world_bundle.txt": "/app/logs/world_bundle.txt",
}
NG_FAILURE_CLASS_KEY = "_ng_failure_class"
NG_FAILURE_TERMINAL_KEY = "_ng_failure_terminal"
_WORLD_ID_RE = re.compile(r"^world_[0-9a-f]{32}$")


class ApexAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    concurrency: int = Field(gt=0)
    timeout: int = Field(gt=0)
    image: str
    image_build: ApexImageBuildConfig
    sandbox_provider: str | Dict[str, Any]
    sandbox_spec: Dict[str, Any]

    edgar_user_agent: Optional[str]
    max_turns: int = Field(gt=0, le=200)
    max_output_tokens: int = Field(gt=0)
    # Policy context window reported to Stirrup, which summarizes at 70% of it. Unset keeps Stirrup using
    # max_output_tokens as the window.
    context_window_tokens: Optional[int] = Field(default=None, gt=0)
    supports_vision: bool
    temperature: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0)

    max_snapshot_bytes: Optional[int] = Field(default=None, gt=0)
    max_world_bytes: Optional[int] = Field(default=None, gt=0)
    artifact_output_dir: Optional[str] = None
    prebuilt_world_manifest: Optional[str] = None
    prebuilt_startup_timeout_seconds: int = Field(default=1800, gt=0)
    # Relay policy traffic through a unix socket bound into the sandbox. "auto"
    # enables it when the apptainer sandbox runs in its own network namespace
    # (extra_start_args contain --net or a --network option).
    policy_egress_relay: Literal["auto", "always", "never"] = "auto"

    # Mid-rollout checkpoint and resume. A per-rollout directory under this
    # host path is bind-mounted into the sandbox at `resume_checkpoint_mount`;
    # the runtime checkpoints there at turn boundaries and a rollout that Gym
    # re-dispatches after a mid-flight kill continues from it. None disables
    # the feature. Requires a sandbox provider with per-sandbox bind mounts
    # (the apptainer provider) and a host path that outlives the node.
    resume_checkpoint_dir: Optional[str] = None
    resume_checkpoint_mount: str = "/checkpoint"
    resume_checkpoint_interval_seconds: float = Field(default=60.0, ge=0.0)
    # A resumed segment is not started when less than this much of the
    # per-task budget is left; the rollout is reported as timed out instead.
    resume_min_remaining_seconds: int = Field(default=300, ge=0)


class ApexAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_id: str
    world_id: str
    task_input_files: Optional[str] = None
    domain: Optional[str] = None
    foundry_services: List[str] = Field(default_factory=list)
    runtime_mode: Literal["world_zip", "prebuilt_world"] = "world_zip"
    task_slug: Optional[str] = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]*$")


class ApexAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


def load_runner_source() -> str:
    return _RUNNER_PATH.read_text(encoding="utf-8")


def load_prebuilt_runner_source() -> str:
    return _PREBUILT_RUNNER_PATH.read_text(encoding="utf-8")


def instruction_from_input(params: NeMoGymResponseCreateParamsNonStreaming) -> str:
    if isinstance(params.input, str):
        return params.input
    parts: list[str] = []
    for item in params.input:
        payload = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        if payload.get("role") != "user":
            continue
        content = payload.get("content", "")
        if isinstance(content, str):
            parts.append(content)
            continue
        for block in content or []:
            block = block.model_dump() if hasattr(block, "model_dump") else block
            if isinstance(block, dict) and block.get("type") in {"input_text", "output_text", "text"}:
                parts.append(str(block.get("text") or ""))
    return "\n\n".join(part for part in parts if part).strip()


def _safe_id(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return cleaned[:128] or "unknown"


class PolicyEgressRelay:
    """Unix socket on the host that forwards sandbox connections to the model server."""

    def __init__(self, socket_dir: Path, server: RelayServer) -> None:
        self.socket_dir = socket_dir
        self._server = server

    @property
    def socket_name(self) -> str:
        return _EGRESS_SOCKET_NAME

    @classmethod
    async def start(cls, model_base_url: str) -> "PolicyEgressRelay":
        parts = urlsplit(model_base_url)
        if parts.scheme != "http" or not parts.hostname:
            raise ValueError(f"the policy egress relay needs an http model_base_url, got {model_base_url!r}")
        socket_dir = Path(tempfile.mkdtemp(prefix="apex-egress-"))
        if len(str(socket_dir / _EGRESS_SOCKET_NAME)) > 100:  # AF_UNIX paths are capped at 107 bytes
            shutil.rmtree(socket_dir, ignore_errors=True)
            socket_dir = Path(tempfile.mkdtemp(prefix="apex-egress-", dir="/tmp"))
        try:
            server = await serve_unix_to_tcp(str(socket_dir / _EGRESS_SOCKET_NAME), parts.hostname, parts.port or 80)
        except Exception:
            shutil.rmtree(socket_dir, ignore_errors=True)
            raise
        return cls(socket_dir, server)

    async def close(self) -> None:
        await self._server.close()
        shutil.rmtree(self.socket_dir, ignore_errors=True)


class ApexAgent(SimpleResponsesAPIAgent):
    """Run one upstream Apex rollout, then hand changed artifacts to Gym verification."""

    config: ApexAgentConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _semaphore: Any = None
    _sandbox_provider: Any = None
    _sandbox_metadata: Any = None
    _setup_lock: Any = None
    _image: Any = None
    _stirrup_archive: Any = None
    _host_stirrup_runtime: Any = None
    _prebuilt_worlds: Any = None

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._semaphore = asyncio.Semaphore(self.config.concurrency)
        global_config = getattr(self.server_client, "global_config_dict", None)
        self._sandbox_provider = resolve_provider_config(self.config.sandbox_provider, global_config)
        self._sandbox_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config)
        self._setup_lock = asyncio.Lock()
        self._image = None
        self._stirrup_archive = None
        self._host_stirrup_runtime = None
        self._prebuilt_worlds = self._load_prebuilt_worlds()

    def _load_prebuilt_worlds(self) -> dict[str, dict[str, Any]]:
        if not self.config.prebuilt_world_manifest:
            return {}
        path = Path(self.config.prebuilt_world_manifest).expanduser()
        if not path.is_absolute():
            path = PARENT_DIR / path
        payload = json.loads(path.resolve().read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid prebuilt-world manifest: {path}")
        raw_cache_root = payload.get("image_cache_root")
        worlds = payload.get("worlds") or {}
        if not isinstance(raw_cache_root, str) or not raw_cache_root.strip() or not isinstance(worlds, dict):
            raise ValueError(f"invalid prebuilt-world manifest: {path}")
        cache_root = Path(raw_cache_root).expanduser()
        if not cache_root.is_absolute():
            raise ValueError(f"prebuilt-world image cache must be absolute: {cache_root}")
        cache_root = cache_root.resolve()
        resolved: dict[str, dict[str, Any]] = {}
        for world_id, entry in worlds.items():
            if not isinstance(world_id, str) or not _WORLD_ID_RE.fullmatch(world_id) or not isinstance(entry, dict):
                raise ValueError(f"invalid prebuilt-world entry for {world_id}")
            raw_image = entry.get("runtime_image")
            if not isinstance(raw_image, str) or not raw_image.strip():
                raise ValueError(f"invalid prebuilt-world image for {world_id}")
            image = Path(raw_image).expanduser().resolve()
            if image.parent != cache_root or image.name != f"{world_id}.sif":
                raise ValueError(f"untrusted prebuilt-world image path for {world_id}: {image}")
            ownership = entry.get("startup_ownership", [])
            if not isinstance(ownership, list):
                raise ValueError(f"invalid startup ownership for {world_id}")
            for setting in ownership:
                if not isinstance(setting, dict) or set(setting) != {"path", "user", "group"}:
                    raise ValueError(f"invalid startup ownership for {world_id}")
                path, user, group = (setting[key] for key in ("path", "user", "group"))
                if not all(isinstance(value, str) for value in (path, user, group)):
                    raise ValueError(f"invalid startup ownership for {world_id}")
                parts = PurePosixPath(path).parts
                if (
                    len(parts) != 7
                    or parts[:4] != ("/", "app", "tools", "mcp_servers")
                    or parts[5] != ".state"
                    or parts[6] in (".", "..")
                    or not re.fullmatch(r"[a-z][a-z0-9_]*", parts[4])
                    or user != f"svc_{parts[4]}"
                    or group != f"appsdata_{parts[4]}"
                ):
                    raise ValueError(f"invalid startup ownership for {world_id}: {path}")
            resolved[str(world_id)] = {"image": str(image), "startup_ownership": ownership}
        return resolved

    def _prebuilt_image(self, body: ApexAgentRunRequest) -> str:
        if not body.task_slug:
            raise ValueError(f"prebuilt-world task {body.task_id} is missing task_slug")
        world = self._prebuilt_worlds.get(body.world_id)
        if not world:
            raise ValueError(f"world {body.world_id} is absent from the trusted prebuilt-world manifest")
        image = world["image"]
        if not Path(image).is_file():
            raise FileNotFoundError(f"prebuilt-world image is missing: {image}")
        return image

    async def responses(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        raise NotImplementedError("ApexAgent is driven through /run, not /v1/responses")

    def _model_base_url(self, body: ApexAgentRunRequest) -> str:
        cfg = get_first_server_config_dict(self.server_client.global_config_dict, self.config.model_server.name)
        root = self.server_client._build_server_base_url(cfg)
        return self.base_url_for_run(root, body).rstrip("/") + "/v1"

    def _policy_model(self) -> str:
        value = self.server_client.global_config_dict.get("policy_model_name")
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError("policy_model_name must be set in Gym's env.yaml or with gym eval run --model")
        return value.strip()

    def _sandbox_has_private_network(self) -> bool:
        provider = self._sandbox_provider if isinstance(self._sandbox_provider, dict) else {}
        create = (provider.get("apptainer") or {}).get("create") or {}
        args = create.get("extra_start_args") or []
        return any(str(arg) == "--net" or str(arg).startswith("--network") for arg in args)

    def _sandbox_uses_fakeroot(self) -> bool:
        provider = self._sandbox_provider if isinstance(self._sandbox_provider, dict) else {}
        create = (provider.get("apptainer") or {}).get("create") or {}
        return "--fakeroot" in (create.get("extra_start_args") or [])

    def _policy_egress_relay_enabled(self) -> bool:
        if self.config.policy_egress_relay == "auto":
            return self._sandbox_has_private_network()
        return self.config.policy_egress_relay == "always"

    @asynccontextmanager
    async def _policy_egress(self, body: ApexAgentRunRequest) -> AsyncIterator[PolicyEgressRelay | None]:
        if not self._policy_egress_relay_enabled():
            yield None
            return
        relay = await PolicyEgressRelay.start(self._model_base_url(body))
        try:
            yield relay
        finally:
            await relay.close()

    def _sandbox_parts(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
        extra = dict(self.config.sandbox_spec)
        provider_options = dict(extra.pop("provider_options", {}) or {})
        metadata = dict(self._sandbox_metadata)
        metadata.update(extra.pop("metadata", {}) or {})
        resources = extra.pop("resources", {})
        if isinstance(resources, dict):
            resources = SandboxResources.from_mapping(resources)
        return extra, provider_options, metadata, resources

    async def _build_stirrup_archive(self, image: str) -> Path:
        """Build the small pinned Stirrup runtime once inside the Archipelago image."""
        archive = stirrup_cache_path(
            agent_dir=Path(__file__).parent,
            setup_path=_STIRRUP_SETUP_PATH,
            requirements_path=_STIRRUP_REQUIREMENTS_PATH,
            image=image,
        )
        if archive.exists():
            return archive

        extra, provider_options, metadata, resources = self._sandbox_parts()
        build_root = "/app/stirrup-build"
        remote_archive = f"{build_root}/stirrup-runtime.tar.gz"
        spec = SandboxSpec(
            image=image,
            workdir="/app",
            env={},
            metadata=metadata,
            provider_options=provider_options,
            resources=resources,
            **extra,
        )
        temporary = archive.with_suffix(".tmp")
        async with AsyncSandbox(self._sandbox_provider, spec) as sandbox:
            await sandbox.start()
            created = await sandbox.exec(f"mkdir -p {shlex.quote(build_root)}", timeout_s=30)
            if created.return_code != 0:
                raise RuntimeError(f"could not create Stirrup build directory: {(created.stderr or '')[-1000:]}")
            await sandbox.upload(_STIRRUP_SETUP_PATH, f"{build_root}/setup_stirrup.sh")
            await sandbox.upload(_STIRRUP_REQUIREMENTS_PATH, f"{build_root}/stirrup-requirements.txt")
            install = await sandbox.exec(
                f"bash {shlex.quote(build_root + '/setup_stirrup.sh')}",
                timeout_s=max(self.config.timeout, 1800),
            )
            if install.return_code != 0:
                details = (install.stderr or install.stdout or "")[-4000:]
                raise RuntimeError(f"pinned Stirrup installation failed: {details}")
            packed = await sandbox.exec(
                f"tar -czf {shlex.quote(remote_archive)} -C {shlex.quote(_STIRRUP_ROOT)} .",
                timeout_s=600,
            )
            if packed.return_code != 0:
                details = (packed.stderr or packed.stdout or "")[-2000:]
                raise RuntimeError(f"Stirrup runtime archive creation failed: {details}")
            await sandbox.download(remote_archive, temporary)
        temporary.replace(archive)
        return archive

    async def _ensure_runtime_setup(self) -> Path:
        """Resolve the bootstrap image and cached portable Stirrup runtime."""
        async with self._setup_lock:
            if self._image is None:
                self._image = await asyncio.to_thread(
                    resolve_image,
                    agent_dir=Path(__file__).parent,
                    parent_dir=PARENT_DIR,
                    image=self.config.image,
                    image_build=self.config.image_build,
                    sandbox_provider=self._sandbox_provider,
                )
            if self._stirrup_archive is None:
                self._stirrup_archive = await self._build_stirrup_archive(self._image)
            return self._stirrup_archive

    async def _prepare_host_stirrup_runtime(self, archive: Path) -> Path:
        """Extract once on the host so fakeroot worlds do not unpack the archive."""
        async with self._setup_lock:
            if self._host_stirrup_runtime is None:
                host_runtime = tempfile.TemporaryDirectory(prefix="apex-stirrup-runtime-")
                try:
                    await asyncio.to_thread(
                        subprocess.run,
                        [
                            "tar",
                            "--no-same-owner",
                            "--no-same-permissions",
                            "-xzf",
                            str(archive),
                            "-C",
                            host_runtime.name,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except Exception as exc:
                    host_runtime.cleanup()
                    if isinstance(exc, subprocess.CalledProcessError):
                        raise RuntimeError(
                            f"could not prepare host Stirrup runtime: {(exc.stderr or '')[-2000:]}"
                        ) from exc
                    raise
                self._host_stirrup_runtime = host_runtime
            return Path(self._host_stirrup_runtime.name)

    async def _download_world(self, cookies: Any, target: Path) -> None:
        response = await self.server_client.get(
            server_name=self.config.resources_server.name,
            url_path="/world",
            cookies=cookies,
        )
        await raise_for_status(response)
        data = await response.read()
        if self.config.max_world_bytes is not None and len(data) > self.config.max_world_bytes:
            raise RuntimeError(f"world archive is {len(data)} bytes; limit is {self.config.max_world_bytes}")
        target.write_bytes(data)

    async def _download_task_files(self, cookies: Any, target: Path) -> None:
        response = await self.server_client.get(
            server_name=self.config.resources_server.name,
            url_path="/task_files",
            cookies=cookies,
        )
        await raise_for_status(response)
        data = await response.read()
        if self.config.max_world_bytes is not None and len(data) > self.config.max_world_bytes:
            raise RuntimeError(f"task attachment archive is {len(data)} bytes; limit is {self.config.max_world_bytes}")
        target.write_bytes(data)

    def _sandbox_spec(
        self,
        body: ApexAgentRunRequest,
        instruction: str,
        *,
        image: str | None = None,
        prebuilt_world: bool = False,
        relay: PolicyEgressRelay | None = None,
        host_stirrup_runtime: Path | None = None,
        resume_dir: Path | None = None,
        resume_checkpoint: ResumeCheckpoint | None = None,
    ) -> SandboxSpec:
        extra, provider_options, metadata, resources = self._sandbox_parts()
        binds = provider_options.get("binds")
        binds = [binds] if isinstance(binds, str) else list(binds or [])
        if relay is not None:
            binds.append(f"{relay.socket_dir}:{_EGRESS_MOUNT}")
        if host_stirrup_runtime is not None:
            binds.append(f"{host_stirrup_runtime}:{_STIRRUP_ROOT}:ro")
        if resume_dir is not None:
            binds.append(f"{resume_dir}:{self.config.resume_checkpoint_mount}")
        if binds:
            provider_options["binds"] = binds
        metadata.update({"nemo_gym_agent": self.config.name, "task_id": _safe_id(body.task_id)})
        policy_model = self._policy_model()
        if "edgar" in body.foundry_services and not self.config.edgar_user_agent:
            raise ValueError(
                "this world requires EDGAR; set apex_edgar_user_agent in env.yaml to a valid SEC contact identity"
            )
        runner_config = {
            "task_id": body.task_id,
            "world_id": body.world_id,
            "instruction": instruction,
            "model_base_url": self._model_base_url(body),
            "policy_model": policy_model,
            "max_turns": self.config.max_turns,
            "supports_vision": self.config.supports_vision,
            "max_output_tokens": (
                body.responses_create_params.max_output_tokens
                if body.responses_create_params.max_output_tokens is not None
                else self.config.max_output_tokens
            ),
            "temperature": (
                body.responses_create_params.temperature
                if body.responses_create_params.temperature is not None
                else self.config.temperature
            ),
            "top_p": (
                body.responses_create_params.top_p
                if body.responses_create_params.top_p is not None
                else self.config.top_p
            ),
            "foundry_services": body.foundry_services,
            "edgar_user_agent": self.config.edgar_user_agent,
            "task_slug": body.task_slug,
            "resume_checkpoint_dir": self.config.resume_checkpoint_mount if resume_dir is not None else None,
            "resume_allowed": resume_checkpoint is not None,
            "resume_checkpoint_interval_seconds": self.config.resume_checkpoint_interval_seconds,
        }
        if prebuilt_world:
            runner_config["startup_timeout_seconds"] = self.config.prebuilt_startup_timeout_seconds
            runner_config["startup_ownership"] = self._prebuilt_worlds[body.world_id]["startup_ownership"]
        if relay is not None:
            runner_config["model_egress_socket"] = f"{_EGRESS_MOUNT}/{relay.socket_name}"
        if self.config.context_window_tokens is not None:
            runner_config["context_window_tokens"] = self.config.context_window_tokens
        files = {
            f"{_GUEST_ROOT}/sandbox_entrypoint.py": (
                load_prebuilt_runner_source() if prebuilt_world else load_runner_source()
            ),
            f"{_GUEST_ROOT}/stirrup_runtime.py": _STIRRUP_RUNTIME_PATH.read_text(encoding="utf-8"),
            f"{_GUEST_ROOT}/runner_config.json": json.dumps(runner_config),
        }
        return SandboxSpec(
            image=image or self._image or self.config.image,
            workdir=_GUEST_ROOT,
            env={
                "HF_HUB_OFFLINE": "1",
                "LOGURU_LEVEL": "WARNING",
                "NO_PROXY": "127.0.0.1,localhost",
            },
            files=files,
            metadata=metadata,
            provider_options=provider_options,
            resources=resources,
            **extra,
        )

    @staticmethod
    def _response_from_result(result: dict[str, Any], model: str) -> NeMoGymResponse:
        answer = str(result.get("final_answer") or "")
        input_tokens = int(result.get("n_input_tokens") or 0)
        output_tokens = int(result.get("n_output_tokens") or 0)
        reasoning_tokens = int(result.get("n_reasoning_tokens") or 0)
        response = NeMoGymResponse(
            id=f"resp_{uuid.uuid4().hex}",
            created_at=time.time(),
            model=model,
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid.uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text=answer, annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=False,
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=reasoning_tokens),
                total_tokens=input_tokens + output_tokens,
            ),
        )
        response.apex_trajectory = result.get("trajectory") or []
        response.apex_agent_mode = result.get("agent_mode")
        response.apex_completion_status = result.get("completion_status")
        response.apex_resume_segments = result.get("resume_segments")
        response.apex_resumed_from_turn = result.get("resumed_from_turn")
        response.apex_resume_checkpoints = result.get("n_resume_checkpoints")
        return response

    def _failure(
        self,
        body: ApexAgentRunRequest,
        error: str,
        return_code: int | None = None,
        failure_class: str = "apex_error",
        failure_terminal: bool = False,
        partial_result: Optional[dict[str, Any]] = None,
    ) -> ApexAgentVerifyResponse:
        LOG.error("Apex rollout failed for task %s: %s", body.task_id, error)
        try:
            model = self._policy_model()
        except RuntimeError:
            model = body.responses_create_params.model or "error"
        partial_result = partial_result or {"final_answer": ""}
        response = self._response_from_result(partial_result, model)
        payload = body.model_dump() | {
            "response": response,
            "reward": 0.0,
            "apex_error": error,
            "container_exit_code": return_code,
            "apex_trajectory": partial_result.get("trajectory") or [],
            "apex_completion_status": partial_result.get("completion_status"),
        }
        payload[NG_FAILURE_CLASS_KEY] = failure_class
        if failure_terminal:
            payload[NG_FAILURE_TERMINAL_KEY] = True
        return ApexAgentVerifyResponse.model_validate(payload)

    @staticmethod
    async def _recover_partial_result(
        sandbox: AsyncSandbox,
        destination: Path,
    ) -> dict[str, Any] | None:
        try:
            await sandbox.download(_GUEST_PARTIAL_RESULT_PATH, destination)
            recovered = json.loads(destination.read_text(encoding="utf-8"))
        except Exception as exc:
            LOG.debug("No recoverable Apex trajectory checkpoint was available: %s", exc)
            return None
        return recovered if isinstance(recovered, dict) else None

    def _artifact_run_dir(self, body: ApexAgentRunRequest, label: str) -> Path | None:
        if not self.config.artifact_output_dir:
            return None
        root = Path(self.config.artifact_output_dir).expanduser()
        if not root.is_absolute():
            root = PARENT_DIR / root
        extra = body.__pydantic_extra__ or {}
        run_name = (
            f"rollout_{extra.get('_ng_rollout_index', 0)}_"
            f"attempt_{extra.get('_ng_attempt_index', 0)}_{label}_{uuid.uuid4().hex[:8]}"
        )
        output_dir = root.resolve() / _safe_id(body.task_id) / run_name
        output_dir.mkdir(parents=True)
        return output_dir

    async def _persist_world_logs(self, sandbox: AsyncSandbox, body: ApexAgentRunRequest) -> Path | None:
        """Keep a failed prebuilt world's full startup logs next to the other rollout artifacts."""
        output_dir = self._artifact_run_dir(body, "failed")
        if output_dir is None:
            return None
        for name, source in _GUEST_WORLD_LOGS.items():
            try:
                await sandbox.download(source, output_dir / name)
            except Exception as exc:
                LOG.debug("Could not keep prebuilt-world log %s: %s", source, exc)
        return output_dir

    def _persist_ungraded_snapshots(
        self,
        body: ApexAgentRunRequest,
        result: dict[str, Any],
        initial_snapshot: bytes,
        final_snapshot: bytes,
    ) -> Path | None:
        """Keep max-turn/incomplete snapshots locally without invoking grading."""
        output_dir = self._artifact_run_dir(body, "ungraded")
        if output_dir is None:
            return None
        (output_dir / "initial_snapshot.zip").write_bytes(initial_snapshot)
        (output_dir / "final_snapshot.zip").write_bytes(final_snapshot)
        (output_dir / "rollout.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return output_dir

    @staticmethod
    def _rollout_key(body: ApexAgentRunRequest) -> tuple[int, int, int] | None:
        """Gym's (task index, rollout index, attempt index) for this request.

        The attempt index counts prior classed failures and is absent on the first
        attempt. A request without Gym's dispatch stamps has no stable identity
        across re-dispatches, so it gets no key and never resumes.
        """
        extra = body.__pydantic_extra__ or {}
        if "_ng_task_index" not in extra or "_ng_rollout_index" not in extra:
            return None
        return (
            int(extra["_ng_task_index"]),
            int(extra["_ng_rollout_index"]),
            int(extra.get("_ng_attempt_index", 0) or 0),
        )

    def _resume_root(self) -> Path | None:
        if not self.config.resume_checkpoint_dir:
            return None
        root = Path(self.config.resume_checkpoint_dir).expanduser()
        if not root.is_absolute():
            root = PARENT_DIR / root
        return root.resolve()

    def _resume_checkpoint_dir(self, body: ApexAgentRunRequest) -> Path | None:
        """One directory per (task, rollout, attempt).

        A rollout killed mid-flight is re-dispatched with the same three indices and
        finds its checkpoint here; a retry after a classed failure carries the next
        attempt index and therefore starts in an empty directory.
        """
        root = self._resume_root()
        key = self._rollout_key(body)
        if root is None or key is None:
            return None
        task_index, rollout_index, attempt_index = key
        return root / _safe_id(body.task_id) / f"t{task_index}_r{rollout_index}_a{attempt_index}"

    async def _prepare_resume(self, directory: Path | None) -> ResumeCheckpoint | None:
        """Return the verified checkpoint in ``directory``, or None for a fresh start.

        Verification hashes the world snapshot, so it runs off the event loop. A
        manifest that is present but fails to verify is retried once: shared
        filesystems return transient errors, and a wrong "no checkpoint" costs the
        whole rollout. Nothing is deleted here; new checkpoint files carry a fresh
        generation and the next manifest supersedes whatever is left.
        """
        if directory is None:
            return None
        checkpoint = await asyncio.to_thread(load_resume_checkpoint, directory)
        if checkpoint is None and (directory / RESUME_MANIFEST_FILENAME).exists():
            await asyncio.sleep(2.0)
            checkpoint = await asyncio.to_thread(load_resume_checkpoint, directory)
        directory.mkdir(parents=True, exist_ok=True)
        return checkpoint

    async def _discard_resume_checkpoint(self, directory: Path | None) -> None:
        """Remove one rollout's checkpoint directory; never anything outside the configured root."""
        root = self._resume_root()
        if directory is None or root is None or directory.parent.parent != root:
            return
        await asyncio.to_thread(shutil.rmtree, directory, True)

    async def run(self, request: Request, body: ApexAgentRunRequest) -> ApexAgentVerifyResponse:
        resume_dir = None if body.runtime_mode == "prebuilt_world" else self._resume_checkpoint_dir(body)
        result = await self._run_rollout(request, body, resume_dir)
        # Any row, graded or a classed failure, ends this attempt's lineage. Only a
        # kill mid-flight returns nothing and leaves the checkpoint for the re-dispatch.
        await self._discard_resume_checkpoint(resume_dir)
        return result

    async def _run_rollout(
        self, request: Request, body: ApexAgentRunRequest, resume_dir: Path | None
    ) -> ApexAgentVerifyResponse:
        instruction = instruction_from_input(body.responses_create_params)
        if not instruction:
            return self._failure(
                body,
                "task input contains no user instruction",
                failure_class="invalid_task_input",
                failure_terminal=True,
            )

        async with self._semaphore:
            result: dict[str, Any] | None = None
            try:
                policy_model = self._policy_model()
                prebuilt_world = body.runtime_mode == "prebuilt_world"
                selected_image = None
                if prebuilt_world:
                    try:
                        selected_image = self._prebuilt_image(body)
                    except (ValueError, FileNotFoundError) as exc:
                        return self._failure(
                            body,
                            str(exc),
                            failure_class="prebuilt_world_config_error",
                            failure_terminal=True,
                        )
                stirrup_archive = await self._ensure_runtime_setup()
                host_stirrup_runtime = (
                    await self._prepare_host_stirrup_runtime(stirrup_archive)
                    if prebuilt_world and self._sandbox_uses_fakeroot()
                    else None
                )
                resume_checkpoint = None if prebuilt_world else await self._prepare_resume(resume_dir)
                timeout_s = self.config.timeout
                if resume_checkpoint is not None:
                    remaining = self.config.timeout - resume_checkpoint.elapsed_seconds
                    if remaining < self.config.resume_min_remaining_seconds:
                        return self._failure(
                            body,
                            f"per-task budget exhausted before resume: {resume_checkpoint.elapsed_seconds:.0f}s of "
                            f"{self.config.timeout}s used across {resume_checkpoint.segments} segment(s)",
                            failure_class="timeout_exceeded",
                            partial_result=partial_result_from_checkpoint(resume_checkpoint),
                        )
                    timeout_s = max(1, int(remaining))
                cookies = request.cookies
                with tempfile.TemporaryDirectory(prefix=f"apex-{_safe_id(body.task_id)}-") as scratch:
                    scratch_path = Path(scratch)
                    world_zip = scratch_path / "world.zip"
                    task_files_zip = scratch_path / "task_files.zip"
                    result_path = scratch_path / "result.json"
                    partial_result_path = scratch_path / "partial_result.json"
                    initial_snapshot_path = scratch_path / "initial.zip"
                    snapshot_path = scratch_path / "final.zip"
                    if not prebuilt_world:
                        seed = await self.server_client.post(
                            server_name=self.config.resources_server.name,
                            url_path="/seed_session",
                            json=body.model_dump(),
                            cookies=request.cookies,
                        )
                        await raise_for_status(seed)
                        cookies = seed.cookies
                        if resume_checkpoint is None:
                            await self._download_world(cookies, world_zip)
                            if body.task_input_files:
                                await self._download_task_files(cookies, task_files_zip)
                    async with self._policy_egress(body) as relay:
                        spec = self._sandbox_spec(
                            body,
                            instruction,
                            image=selected_image,
                            prebuilt_world=prebuilt_world,
                            relay=relay,
                            host_stirrup_runtime=host_stirrup_runtime,
                            resume_dir=None if prebuilt_world else resume_dir,
                            resume_checkpoint=resume_checkpoint,
                        )
                        async with AsyncSandbox(self._sandbox_provider, spec) as sandbox:
                            await sandbox.start()
                            if not prebuilt_world and resume_checkpoint is None:
                                await sandbox.upload(world_zip, f"{_GUEST_ROOT}/world.zip")
                                if body.task_input_files:
                                    await sandbox.upload(task_files_zip, f"{_GUEST_ROOT}/task_files.zip")
                            if host_stirrup_runtime is None:
                                await sandbox.upload(stirrup_archive, f"{_GUEST_ROOT}/stirrup-runtime.tar.gz")
                            extraction = (
                                ""
                                if host_stirrup_runtime is not None
                                else f"mkdir -p {shlex.quote(_STIRRUP_ROOT)} && "
                                f"tar --no-same-owner -xzf "
                                f"{shlex.quote(_GUEST_ROOT + '/stirrup-runtime.tar.gz')} "
                                f"-C {shlex.quote(_STIRRUP_ROOT)} && "
                            )
                            bootstrap = (
                                f"( {stirrup_runtime_bootstrap_script(_STIRRUP_ROOT)} ) && " if prebuilt_world else ""
                            )
                            unpack = await sandbox.exec(
                                f"{extraction}{bootstrap}"
                                f"{shlex.quote(_STIRRUP_ROOT + '/bin/python')} -c "
                                f"{shlex.quote(STIRRUP_PREFLIGHT)}",
                                user="root",
                                timeout_s=600,
                            )
                            if unpack.return_code != 0:
                                detail = (unpack.stderr or unpack.stdout or "")[-4000:]
                                return self._failure(body, f"could not install sandbox Stirrup runtime: {detail}")
                            protection_prefix = (
                                f"chmod 700 {shlex.quote(_STIRRUP_ROOT)} && " if host_stirrup_runtime is None else ""
                            )
                            protect = await sandbox.exec(
                                f"{protection_prefix}"
                                f"chmod -R go-rwx {shlex.quote(_GUEST_ROOT)} && "
                                f"mkdir -p {shlex.quote(_GUEST_ROOT + '/output')} && "
                                f"chmod 700 {shlex.quote(_GUEST_ROOT + '/output')}",
                                user="root",
                            )
                            if protect.return_code != 0:
                                detail = (protect.stderr or protect.stdout or "")[-4000:]
                                return self._failure(body, f"could not protect sandbox inputs: {detail}")
                            if prebuilt_world:
                                hosts = await sandbox.exec(
                                    "printf '%s\\n' '127.0.0.1 localhost' >> /etc/hosts",
                                    user="root",
                                    timeout_s=30,
                                )
                                if hosts.return_code != 0:
                                    detail = (hosts.stderr or hosts.stdout or "")[-4000:]
                                    return self._failure(body, f"could not configure sandbox localhost: {detail}")
                            process = await sandbox.exec(
                                f"{shlex.quote(_STIRRUP_ROOT + '/bin/python')} "
                                f"{shlex.quote(_GUEST_ROOT + '/sandbox_entrypoint.py')}",
                                timeout_s=timeout_s,
                            )
                            if process.return_code != 0:
                                detail = (process.stderr or process.stdout or "")[-4000:]
                                result = await self._recover_partial_result(sandbox, partial_result_path)
                                failure_class = (
                                    "timeout_exceeded" if process.error_type == "timeout" else "sandbox_error"
                                )
                                if (
                                    failure_class == "sandbox_error"
                                    and isinstance(result, dict)
                                    and isinstance(result.get("checkpoint_error"), str)
                                    and result["checkpoint_error"].startswith("ContextOverflowError:")
                                ):
                                    failure_class = "context_overflow"
                                world_logs_dir = (
                                    await self._persist_world_logs(sandbox, body) if prebuilt_world else None
                                )
                                failure = self._failure(
                                    body,
                                    f"sandbox Stirrup rollout exited: {detail}",
                                    process.return_code,
                                    failure_class=failure_class,
                                    partial_result=result,
                                )
                                if world_logs_dir is not None:
                                    failure.artifact_output_dir = str(world_logs_dir)
                                return failure
                            await sandbox.download(f"{_GUEST_ROOT}/output/result.json", result_path)
                            await sandbox.download(f"{_GUEST_ROOT}/output/initial.zip", initial_snapshot_path)
                            await sandbox.download(f"{_GUEST_ROOT}/output/final.zip", snapshot_path)

                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    initial_snapshot = initial_snapshot_path.read_bytes()
                    snapshot = snapshot_path.read_bytes()
                    if self.config.max_snapshot_bytes is not None:
                        for name, data in {"initial": initial_snapshot, "final": snapshot}.items():
                            if len(data) > self.config.max_snapshot_bytes:
                                return self._failure(
                                    body,
                                    f"{name} artifact snapshot is {len(data)} bytes; "
                                    f"limit is {self.config.max_snapshot_bytes}",
                                    failure_class="snapshot_too_large",
                                    failure_terminal=True,
                                    partial_result=result,
                                )
                    if not result.get("completed"):
                        output_dir = self._persist_ungraded_snapshots(body, result, initial_snapshot, snapshot)
                        failure = self._failure(
                            body,
                            f"Stirrup did not submit a completed Finish call (status={result.get('completion_status')})",
                            failure_class="agent_incomplete",
                            failure_terminal=True,
                            partial_result=result,
                        )
                        if output_dir is not None:
                            failure.artifact_output_dir = str(output_dir)
                            failure.initial_snapshot_path = str(output_dir / "initial_snapshot.zip")
                            failure.final_snapshot_path = str(output_dir / "final_snapshot.zip")
                        return failure
                    response = self._response_from_result(result, policy_model)
                    payload = body.model_dump() | {
                        "response": response.model_dump(),
                        "initial_artifact_snapshot_b64": base64.b64encode(initial_snapshot).decode("ascii"),
                        "artifact_snapshot_b64": base64.b64encode(snapshot).decode("ascii"),
                        "artifact_manifest": result.get("artifact_manifest") or [],
                        "apex_trajectory": result.get("trajectory") or [],
                    }
            except Exception as exc:
                return self._failure(body, str(exc), partial_result=result)

        try:
            verify = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=payload,
                cookies=cookies,
            )
            await raise_for_status(verify)
            return ApexAgentVerifyResponse.model_validate(await get_response_json(verify))
        except Exception as exc:
            return self._failure(
                body,
                f"Apex verification failed: {exc}",
                failure_class="verification_error",
                partial_result=result,
            )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    ApexAgent.run_webserver()
