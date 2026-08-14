# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym agent that runs the upstream Apex harness in a per-task sandbox."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import shlex
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
    HARNESS_REVISION,
    ApexImageBuildConfig,
    harness_cache_path,
    prepare_harness_source_archive,
    resolve_image,
)


LOG = logging.getLogger(__name__)
_RUNNER_PATH = Path(__file__).with_name("sandbox_entrypoint.py")
_HARNESS_SETUP_PATH = Path(__file__).with_name("setup_harness.sh")
_HARNESS_REQUIREMENTS_PATH = Path(__file__).with_name("harness-requirements.txt")
_GUEST_ROOT = "/app/apex-gym"
_HARNESS_ROOT = "/app/apex-harness-runtime"


class ApexAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    concurrency: int = Field(gt=0)
    timeout: int = Field(gt=0)
    image: str
    image_build: ApexImageBuildConfig
    harness_repo: str
    harness_root: Optional[str]
    harness_github_token: Optional[str]
    sandbox_provider: str | Dict[str, Any]
    sandbox_spec: Dict[str, Any]

    edgar_user_agent: Optional[str]
    max_turns: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    max_tool_calls_per_turn: int = Field(gt=0)
    temperature: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0)

    max_snapshot_bytes: Optional[int] = Field(default=None, gt=0)
    max_world_bytes: Optional[int] = Field(default=None, gt=0)


class ApexAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_id: str
    world_id: str
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
    _harness_source_archive: Any = None
    _harness_archive: Any = None

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._semaphore = asyncio.Semaphore(self.config.concurrency)
        global_config = getattr(self.server_client, "global_config_dict", None)
        self._sandbox_provider = resolve_provider_config(self.config.sandbox_provider, global_config)
        self._sandbox_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config)
        self._setup_lock = asyncio.Lock()
        self._image = None
        self._harness_source_archive = None
        self._harness_archive = None

    def setup_webserver(self):
        app = super().setup_webserver()
        app.router.on_startup.append(self._preflight_harness_source)
        return app

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

    async def _preflight_harness_source(self) -> None:
        """Fetch the pinned harness before the agent server accepts rollouts."""
        async with self._setup_lock:
            if self._harness_source_archive is None:
                LOG.info("Checking access to pinned Apex harness commit %s", HARNESS_REVISION)
                self._harness_source_archive = await asyncio.to_thread(
                    prepare_harness_source_archive,
                    agent_dir=Path(__file__).parent,
                    repo=self.config.harness_repo,
                    source_root=self.config.harness_root,
                    github_token=self.config.harness_github_token,
                )

    async def _build_harness_archive(self, image: str, source_archive: Path) -> Path:
        """Build the pinned upstream harness once inside the Archipelago base image."""
        archive = harness_cache_path(
            agent_dir=Path(__file__).parent,
            setup_path=_HARNESS_SETUP_PATH,
            requirements_path=_HARNESS_REQUIREMENTS_PATH,
            image=image,
            source_archive=source_archive,
        )
        if archive.exists():
            return archive

        extra, provider_options, metadata, resources = self._sandbox_parts()
        build_root = "/app/apex-harness-build"
        remote_archive = f"{build_root}/apex-harness.tar.gz"
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
                raise RuntimeError(f"could not create harness build directory: {(created.stderr or '')[-1000:]}")
            await sandbox.upload(_HARNESS_SETUP_PATH, f"{build_root}/setup_harness.sh")
            await sandbox.upload(_HARNESS_REQUIREMENTS_PATH, f"{build_root}/harness-requirements.txt")
            await sandbox.upload(source_archive, f"{build_root}/apex-harness-source.tar.gz")
            install = await sandbox.exec(
                f"bash {shlex.quote(build_root + '/setup_harness.sh')}",
                timeout_s=max(self.config.timeout, 1800),
            )
            if install.return_code != 0:
                details = (install.stderr or install.stdout or "")[-4000:]
                raise RuntimeError(f"pinned Apex harness installation failed: {details}")
            packed = await sandbox.exec(
                f"tar -czf {shlex.quote(remote_archive)} -C {shlex.quote(_HARNESS_ROOT)} .",
                timeout_s=600,
            )
            if packed.return_code != 0:
                details = (packed.stderr or packed.stdout or "")[-2000:]
                raise RuntimeError(f"Apex harness archive creation failed: {details}")
            await sandbox.download(remote_archive, temporary)
        temporary.replace(archive)
        return archive

    async def _ensure_runtime_setup(self) -> None:
        """Validate harness access before any expensive image setup or task seeding."""
        await self._preflight_harness_source()
        async with self._setup_lock:
            if self._image is None:
                self._image = await asyncio.to_thread(
                    resolve_image,
                    agent_dir=Path(__file__).parent,
                    parent_dir=PARENT_DIR,
                    image=self.config.image,
                    image_build=self.config.image_build,
                    sandbox_provider=self.config.sandbox_provider,
                )
            if self._harness_archive is None:
                if self._harness_source_archive is None:
                    raise RuntimeError("Apex harness source archive was not prepared")
                self._harness_archive = await self._build_harness_archive(self._image, self._harness_source_archive)

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

    def _sandbox_spec(self, body: ApexAgentRunRequest, instruction: str) -> SandboxSpec:
        extra, provider_options, metadata, resources = self._sandbox_parts()
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
            "max_output_tokens": (
                body.responses_create_params.max_output_tokens
                if body.responses_create_params.max_output_tokens is not None
                else self.config.max_output_tokens
            ),
            "max_tool_calls_per_turn": self.config.max_tool_calls_per_turn,
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
        }
        return SandboxSpec(
            image=self._image or self.config.image,
            workdir=_GUEST_ROOT,
            env={
                "FOUNDRY_LOCAL_ROOT": f"{_HARNESS_ROOT}/.apex",
                "HF_HUB_OFFLINE": "1",
                "LOGURU_LEVEL": "WARNING",
                "NO_PROXY": "127.0.0.1,localhost",
            },
            files={
                f"{_GUEST_ROOT}/sandbox_entrypoint.py": load_runner_source(),
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
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            ),
        )
        response.apex_trajectory = result.get("trajectory") or []
        response.apex_agent_mode = result.get("agent_mode")
        return response

    def _failure(
        self, body: ApexAgentRunRequest, error: str, return_code: int | None = None
    ) -> ApexAgentVerifyResponse:
        LOG.error("Apex rollout failed for task %s: %s", body.task_id, error)
        try:
            model = self._policy_model()
        except RuntimeError:
            model = body.responses_create_params.model or "error"
        response = self._response_from_result({"final_answer": ""}, model)
        return ApexAgentVerifyResponse(
            **body.model_dump(),
            response=response,
            reward=0.0,
            apex_error=error,
            container_exit_code=return_code,
        )

    async def run(self, request: Request, body: ApexAgentRunRequest) -> ApexAgentVerifyResponse:
        instruction = instruction_from_input(body.responses_create_params)
        if not instruction:
            return self._failure(body, "task input contains no user instruction")

        async with self._semaphore:
            try:
                policy_model = self._policy_model()
                await self._ensure_runtime_setup()
                with tempfile.TemporaryDirectory(prefix=f"apex-{_safe_id(body.task_id)}-") as scratch:
                    scratch_path = Path(scratch)
                    world_zip = scratch_path / "world.zip"
                    result_path = scratch_path / "result.json"
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
                    await self._download_world(cookies, world_zip)
                    spec = self._sandbox_spec(body, instruction)
                    async with AsyncSandbox(self._sandbox_provider, spec) as sandbox:
                        await sandbox.start()
                        await sandbox.upload(world_zip, f"{_GUEST_ROOT}/world.zip")
                        await sandbox.upload(self._harness_archive, f"{_GUEST_ROOT}/apex-harness.tar.gz")
                        unpack = await sandbox.exec(
                            f"mkdir -p {shlex.quote(_HARNESS_ROOT)} && "
                            f"tar -xzf {shlex.quote(_GUEST_ROOT + '/apex-harness.tar.gz')} "
                            f"-C {shlex.quote(_HARNESS_ROOT)}",
                            user="root",
                            timeout_s=600,
                        )
                        if unpack.return_code != 0:
                            detail = (unpack.stderr or unpack.stdout or "")[-4000:]
                            return self._failure(body, f"could not install sandbox harness runtime: {detail}")
                        protect = await sandbox.exec(
                            f"chmod -R go-rwx {shlex.quote(_HARNESS_ROOT)} {shlex.quote(_GUEST_ROOT)} && "
                            f"mkdir -p {shlex.quote(_GUEST_ROOT + '/output')} && "
                            f"chmod 700 {shlex.quote(_GUEST_ROOT + '/output')}",
                            user="root",
                        )
                        if protect.return_code != 0:
                            detail = (protect.stderr or protect.stdout or "")[-4000:]
                            return self._failure(body, f"could not protect sandbox inputs: {detail}")
                        process = await sandbox.exec(
                            f"{shlex.quote(_HARNESS_ROOT + '/bin/python')} "
                            f"{shlex.quote(_GUEST_ROOT + '/sandbox_entrypoint.py')}",
                            timeout_s=self.config.timeout,
                        )
                        if process.return_code != 0:
                            detail = (process.stderr or process.stdout or "")[-4000:]
                            return self._failure(body, f"sandbox harness exited: {detail}", process.return_code)
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
                                )
                    response = self._response_from_result(result, policy_model)
                    payload = body.model_dump() | {
                        "response": response.model_dump(),
                        "initial_artifact_snapshot_b64": base64.b64encode(initial_snapshot).decode("ascii"),
                        "artifact_snapshot_b64": base64.b64encode(snapshot).decode("ascii"),
                        "artifact_manifest": result.get("artifact_manifest") or [],
                        "apex_trajectory": result.get("trajectory") or [],
                    }
            except Exception as exc:
                return self._failure(body, str(exc))

        verify = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=payload,
            cookies=cookies,
        )
        await raise_for_status(verify)
        return ApexAgentVerifyResponse.model_validate(await get_response_json(verify))

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
