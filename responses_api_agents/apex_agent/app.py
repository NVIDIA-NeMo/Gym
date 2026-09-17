# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym agent that runs Stirrup in a pinned Archipelago task sandbox."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import shlex
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from responses_api_agents.apex_agent.runtime_setup import (
    ApexImageBuildConfig,
    resolve_image,
    stirrup_cache_path,
)
from responses_api_agents.apex_agent.stirrup_runtime import (
    RESUME_MANIFEST_FILENAME,
    ResumeCheckpoint,
    load_resume_checkpoint,
    partial_result_from_checkpoint,
)


LOG = logging.getLogger(__name__)
_RUNNER_PATH = Path(__file__).with_name("sandbox_entrypoint.py")
_STIRRUP_RUNTIME_PATH = Path(__file__).with_name("stirrup_runtime.py")
_STIRRUP_SETUP_PATH = Path(__file__).with_name("setup_stirrup.sh")
_STIRRUP_REQUIREMENTS_PATH = Path(__file__).with_name("stirrup-requirements.txt")
_GUEST_ROOT = "/app/apex-gym"
_STIRRUP_ROOT = "/app/stirrup-runtime"
_GUEST_PARTIAL_RESULT_PATH = "/sandbox/partial_result.json"
NG_FAILURE_CLASS_KEY = "_ng_failure_class"
NG_FAILURE_TERMINAL_KEY = "_ng_failure_terminal"


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
    supports_vision: bool
    temperature: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0)

    max_snapshot_bytes: Optional[int] = Field(default=None, gt=0)
    max_world_bytes: Optional[int] = Field(default=None, gt=0)
    artifact_output_dir: Optional[str] = None

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


class ApexAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


def load_runner_source() -> str:
    return _RUNNER_PATH.read_text(encoding="utf-8")


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

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._semaphore = asyncio.Semaphore(self.config.concurrency)
        global_config = getattr(self.server_client, "global_config_dict", None)
        self._sandbox_provider = resolve_provider_config(self.config.sandbox_provider, global_config)
        self._sandbox_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config)
        self._setup_lock = asyncio.Lock()
        self._image = None
        self._stirrup_archive = None

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

    async def _ensure_runtime_setup(self) -> None:
        """Resolve the Archipelago image and cached Stirrup runtime."""
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
        resume_dir: Path | None = None,
        resume_checkpoint: ResumeCheckpoint | None = None,
    ) -> SandboxSpec:
        extra, provider_options, metadata, resources = self._sandbox_parts()
        if resume_dir is not None:
            binds = provider_options.get("binds")
            binds = [binds] if isinstance(binds, str) else list(binds or [])
            binds.append(f"{resume_dir}:{self.config.resume_checkpoint_mount}")
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
            "resume_checkpoint_dir": self.config.resume_checkpoint_mount if resume_dir is not None else None,
            "resume_allowed": resume_checkpoint is not None,
            "resume_checkpoint_interval_seconds": self.config.resume_checkpoint_interval_seconds,
        }
        return SandboxSpec(
            image=self._image or self.config.image,
            workdir=_GUEST_ROOT,
            env={
                "HF_HUB_OFFLINE": "1",
                "LOGURU_LEVEL": "WARNING",
                "NO_PROXY": "127.0.0.1,localhost",
            },
            files={
                f"{_GUEST_ROOT}/sandbox_entrypoint.py": load_runner_source(),
                f"{_GUEST_ROOT}/stirrup_runtime.py": _STIRRUP_RUNTIME_PATH.read_text(encoding="utf-8"),
                f"{_GUEST_ROOT}/runner_config.json": json.dumps(runner_config),
            },
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

    def _persist_ungraded_snapshots(
        self,
        body: ApexAgentRunRequest,
        result: dict[str, Any],
        initial_snapshot: bytes,
        final_snapshot: bytes,
    ) -> Path | None:
        """Keep max-turn/incomplete snapshots locally without invoking grading."""
        if not self.config.artifact_output_dir:
            return None
        root = Path(self.config.artifact_output_dir).expanduser()
        if not root.is_absolute():
            root = PARENT_DIR / root
        extra = body.__pydantic_extra__ or {}
        run_name = (
            f"rollout_{extra.get('_ng_rollout_index', 0)}_"
            f"attempt_{extra.get('_ng_attempt_index', 0)}_ungraded_{uuid.uuid4().hex[:8]}"
        )
        output_dir = root.resolve() / _safe_id(body.task_id) / run_name
        output_dir.mkdir(parents=True)
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
        resume_dir = self._resume_checkpoint_dir(body)
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
                await self._ensure_runtime_setup()
                resume_checkpoint = await self._prepare_resume(resume_dir)
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
                with tempfile.TemporaryDirectory(prefix=f"apex-{_safe_id(body.task_id)}-") as scratch:
                    scratch_path = Path(scratch)
                    world_zip = scratch_path / "world.zip"
                    task_files_zip = scratch_path / "task_files.zip"
                    result_path = scratch_path / "result.json"
                    partial_result_path = scratch_path / "partial_result.json"
                    initial_snapshot_path = scratch_path / "initial.zip"
                    snapshot_path = scratch_path / "final.zip"
                    seed = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/seed_session",
                        json=body.model_dump(),
                        cookies=request.cookies,
                    )
                    await raise_for_status(seed)
                    cookies = seed.cookies
                    # A resumed segment restores the world from its checkpoint instead.
                    if resume_checkpoint is None:
                        await self._download_world(cookies, world_zip)
                        if body.task_input_files:
                            await self._download_task_files(cookies, task_files_zip)
                    spec = self._sandbox_spec(
                        body, instruction, resume_dir=resume_dir, resume_checkpoint=resume_checkpoint
                    )
                    async with AsyncSandbox(self._sandbox_provider, spec) as sandbox:
                        await sandbox.start()
                        if resume_checkpoint is None:
                            await sandbox.upload(world_zip, f"{_GUEST_ROOT}/world.zip")
                            if body.task_input_files:
                                await sandbox.upload(task_files_zip, f"{_GUEST_ROOT}/task_files.zip")
                        await sandbox.upload(self._stirrup_archive, f"{_GUEST_ROOT}/stirrup-runtime.tar.gz")
                        unpack = await sandbox.exec(
                            f"mkdir -p {shlex.quote(_STIRRUP_ROOT)} && "
                            f"tar -xzf {shlex.quote(_GUEST_ROOT + '/stirrup-runtime.tar.gz')} "
                            f"-C {shlex.quote(_STIRRUP_ROOT)}",
                            user="root",
                            timeout_s=600,
                        )
                        if unpack.return_code != 0:
                            detail = (unpack.stderr or unpack.stdout or "")[-4000:]
                            return self._failure(body, f"could not install sandbox Stirrup runtime: {detail}")
                        protect = await sandbox.exec(
                            f"chmod -R go-rwx {shlex.quote(_STIRRUP_ROOT)} {shlex.quote(_GUEST_ROOT)} && "
                            f"mkdir -p {shlex.quote(_GUEST_ROOT + '/output')} && "
                            f"chmod 700 {shlex.quote(_GUEST_ROOT + '/output')}",
                            user="root",
                        )
                        if protect.return_code != 0:
                            detail = (protect.stderr or protect.stdout or "")[-4000:]
                            return self._failure(body, f"could not protect sandbox inputs: {detail}")
                        process = await sandbox.exec(
                            f"{shlex.quote(_STIRRUP_ROOT + '/bin/python')} "
                            f"{shlex.quote(_GUEST_ROOT + '/sandbox_entrypoint.py')}",
                            timeout_s=timeout_s,
                        )
                        if process.return_code != 0:
                            detail = (process.stderr or process.stdout or "")[-4000:]
                            result = await self._recover_partial_result(sandbox, partial_result_path)
                            return self._failure(
                                body,
                                f"sandbox Stirrup rollout exited: {detail}",
                                process.return_code,
                                failure_class="timeout_exceeded"
                                if process.error_type == "timeout"
                                else "sandbox_error",
                                partial_result=result,
                            )
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
