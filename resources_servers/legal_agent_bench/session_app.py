# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The LAB resources server creates and cleans up task and verifier sandboxes."""

from __future__ import annotations

import asyncio
import copy
import fcntl
import hashlib
import json
import logging
import math
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from pydantic import ConfigDict, Field, JsonValue, PrivateAttr

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import get_global_config_dict
from nemo_gym.core import EpisodeId
from nemo_gym.orchestration.jobs import installed_gym_commit
from nemo_gym.sandbox import (
    AsyncSandbox,
    DirectSandboxConnection,
    SandboxAccess,
    SandboxExecResult,
    SandboxResources,
    SandboxSpec,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import (
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFences,
    ResourcesSessionLocks,
    SimpleResourcesServer,
    lost_session_verification,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.legal_agent_bench.app import LegalAgentBenchResourcesServerConfig as _BaseLabConfig
from resources_servers.legal_agent_bench.app import _build_verifier_env
from resources_servers.legal_agent_bench.archive_worker import ArchiveLimits
from resources_servers.legal_agent_bench.artifacts import download_archive, extract_outputs
from resources_servers.legal_agent_bench.prepare import (
    _DOCKERFILE,
    BUILD_CONTEXT_FILES,
    LAB_SOURCE_REVISION,
    PACKAGE_DIR,
    TOOL_RUNNER_SOURCE,
    ensure_assets,
    resolve_repo_path,
    task_documents_dir,
)


LOG = logging.getLogger(__name__)
_ENV = {
    "VDR_DIR": "/workspace/vdr",
    "DOCUMENTS_DIR": "/workspace/vdr",
    "OUTPUT_DIR": "/workspace/output",
    "WORKSPACE_DIR": "/workspace",
    "SKILLS_DIR": "/workspace/skills",
    "NODE_PATH": "/usr/local/lib/node_modules",
}


class LabSessionResourcesConfig(_BaseLabConfig):
    """Task sandbox settings shared by every LAB harness pairing."""

    sandbox_provider: str = "sandbox"
    sandbox_config: dict[str, JsonValue] = Field(default_factory=dict)
    sandbox_image: str | None = None
    concurrency: int = Field(default=1, ge=1)
    agent_timeout_seconds: float = Field(default=10800, gt=0)
    verifier_timeout_seconds: float = Field(default=3600, gt=0)
    staging_timeout_seconds: float = Field(default=900, gt=0)
    results_dir: str = "results/legal_agent_bench"
    archive_limits: ArchiveLimits = Field(default_factory=ArchiveLimits)


class LabSessionVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    instance_id: str


class LabSessionVerifyResponse(BaseVerifyResponse):
    """Keep LAB task identity, grading evidence and artifact references on the wire."""

    instance_id: str
    lab_metrics: dict[str, float] = Field(default_factory=dict)
    lab_scores: dict[str, JsonValue] = Field(default_factory=dict)
    source_revision: str = LAB_SOURCE_REVISION
    artifact_dir: str | None = None


@dataclass
class _TaskSession:
    seed: ResourcesSeedSessionRequest
    artifact_dir: Path
    task_dir: Path
    prompt: str
    image: str
    setup_complete: bool = False
    sandbox: AsyncSandbox | None = None
    verifier: AsyncSandbox | None = None
    export: Path | None = None
    verification_request: LabSessionVerifyRequest | None = None
    result: LabSessionVerifyResponse | None = None


def _archive_inputs(destination: Path, entries: list[tuple[Path, str]]) -> None:
    with tarfile.open(destination, "w:gz", dereference=False) as archive:
        for path, name in entries:
            archive.add(path, arcname=name)


def _build_local_image() -> str:
    """Build a content-addressed image from explicit non-secret runtime files."""
    digest = hashlib.sha256(_DOCKERFILE.encode() + TOOL_RUNNER_SOURCE.read_bytes())
    for name, path in sorted(BUILD_CONTEXT_FILES.items()):
        digest.update(name.encode() + path.read_bytes())
    image = "nemo-gym-legal-agent-bench:" + digest.hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / ("lab-image-" + digest.hexdigest() + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        inspected = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=60)
        if inspected.returncode == 0:
            return str(json.loads(inspected.stdout)[0]["Id"])
        with tempfile.TemporaryDirectory(prefix="lab-image-") as temporary:
            context = Path(temporary)
            (context / "Dockerfile").write_text(_DOCKERFILE)
            (context / "harness").mkdir()
            shutil.copyfile(TOOL_RUNNER_SOURCE, context / "harness" / "container_tool_runner.py")
            for name, path in BUILD_CONTEXT_FILES.items():
                shutil.copyfile(path, context / name)
            result = subprocess.run(
                ["docker", "build", "--tag", image, str(context)],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=3600,
            )
            if result.returncode:
                raise RuntimeError(f"LAB image build failed: {result.stderr[-6000:]}")
    inspected = subprocess.run(["docker", "image", "inspect", image], check=True, capture_output=True, timeout=60)
    return str(json.loads(inspected.stdout)[0]["Id"])


class LabSessionResourcesServer(SimpleResourcesServer):
    """Create and clean up LAB sandboxes, keeping grading inputs separate from the agent."""

    config: LabSessionResourcesConfig
    _sessions: dict[str, _TaskSession] = PrivateAttr(default_factory=dict)
    _fences: ResourcesSessionFences[EpisodeId] = PrivateAttr()
    _locks: ResourcesSessionLocks = PrivateAttr(default_factory=ResourcesSessionLocks)
    _slots: asyncio.Semaphore = PrivateAttr()
    grading_may_use_uninstrumented_model = True
    _assets: dict[str, Path] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: object) -> None:
        self._fences = ResourcesSessionFences(self.config.session_close_retry_window_seconds)
        self._slots = asyncio.Semaphore(self.config.concurrency)

    def setup_webserver(self) -> FastAPI:
        self._assets = ensure_assets(
            tasks_dir=self.config.harbor_tasks_cache_dir,
            skills_dir=self.config.harness_skills_dir,
            allow_download=self.config.auto_prepare_assets,
        )
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(application: FastAPI) -> AsyncIterator[None]:
            async with parent_lifespan(application), self._lifespan(application):
                yield

        app.router.lifespan_context = lifespan
        return app

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            for session_id, state in list(self._sessions.items()):
                try:
                    await self.close_resources_session(
                        ResourcesCloseSessionRequest(resources_session_id=session_id, episode_id=state.seed.episode_id)
                    )
                except Exception:
                    LOG.exception("Could not release LAB session %s at shutdown", session_id)

    def _provider(self) -> tuple[dict[str, object], dict[str, str]]:
        global_config = get_global_config_dict()
        provider = copy.deepcopy(resolve_provider_config(self.config.sandbox_provider, global_config))
        if "apptainer" in provider:
            # Apptainer otherwise binds the Gym host's home and cwd automatically.
            # Keep both task and verifier environments limited to explicit staging.
            settings = provider["apptainer"] or {}
            create = settings.setdefault("create", {})
            create["extra_start_args"] = [
                *(create.get("extra_start_args") or []),
                "--containall",
                "--no-home",
                "--no-mount",
                "cwd,hostfs",
            ]
            provider["apptainer"] = settings
        return provider, resolve_provider_metadata(self.config.sandbox_provider, global_config)

    async def _image(self) -> str:
        if self.config.sandbox_image:
            return self.config.sandbox_image
        provider, _ = self._provider()
        if set(provider) != {"docker"}:
            raise ResourcesSeedRejectedError(
                "Set sandbox_image to a prepared LAB image for this provider; automatic image building uses local Docker"
            )
        return await asyncio.to_thread(_build_local_image)

    def _new_sandbox(self, *, image: str, task_name: str) -> tuple[AsyncSandbox, SandboxSpec]:
        provider, metadata = self._provider()
        settings = self.config.sandbox_config
        spec = SandboxSpec(
            image=image,
            workdir="/workspace",
            env=dict(_ENV),
            resources=SandboxResources.from_mapping(
                settings.get("resources") or {"cpu": 1, "memory_mib": 4096, "disk_gib": 10}
            ),
            metadata={**metadata, **(settings.get("metadata") or {}), "lab_task": task_name[:63].rstrip("_-")},
            provider_options=dict(settings.get("provider_options") or {}),
            ttl_s=settings.get("ttl_s")
            or self.config.agent_timeout_seconds
            + self.config.verifier_timeout_seconds
            + self.config.staging_timeout_seconds,
            ready_timeout_s=settings.get("ready_timeout_s") or self.config.staging_timeout_seconds,
        )
        return AsyncSandbox(provider), spec

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        session_id = body.resources_session_id
        async with self._locks.locked(session_id):
            self._fences.raise_if_fenced(session_id)
            previous = self._sessions.get(session_id)
            if previous is not None:
                if previous.seed != body:
                    raise ResourcesSessionConflictError("LAB session is bound to another episode or task")
                return await self._seed_response(request, session_id, previous)
            pointer = body.task_data.get("instance_id")
            if (
                not isinstance(pointer, str)
                or not pointer.startswith("legal_agent_bench::")
                or body.task_id.task_id != pointer
            ):
                raise ResourcesSeedRejectedError("LAB task identity must match instance_id")
            name = pointer.removeprefix("legal_agent_bench::")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ResourcesSeedRejectedError("Invalid LAB task name")
            task_dir = resolve_repo_path(self.config.harbor_tasks_cache_dir) / name
            if not (task_dir / "task.json").is_file():
                raise ResourcesSeedRejectedError(f"LAB task is not prepared: {name}")
            task = json.loads((task_dir / "task.json").read_text())
            if (task.get("metadata") or {}).get("lab_source_revision") != LAB_SOURCE_REVISION:
                raise ResourcesSeedRejectedError("LAB task assets do not match the pinned revision")
            skills = resolve_repo_path(self.config.harness_skills_dir)
            preamble = (PACKAGE_DIR / "harness" / "system-prompt.md").read_text()
            preamble = preamble.split("## Completion", 1)[0]
            manuals = "\n\n".join(
                f"## Skill: {skill.name}\n{(skill / 'SKILL.md').read_text()}"
                for skill in sorted(skills.iterdir())
                if (skill / "SKILL.md").is_file()
            )
            prompt = f"{preamble}\n\n{manuals}\n\n## Task\n\n{task['title']}\n\n{task['instructions']}"
            image = await self._image()
            await self._slots.acquire()
            state = _TaskSession(
                seed=body.model_copy(deep=True),
                task_dir=task_dir,
                prompt=prompt,
                image=image,
                artifact_dir=resolve_repo_path(self.config.results_dir)
                / name
                / hashlib.sha256(session_id.encode()).hexdigest()[:24],
            )
            self._sessions[session_id] = state
            try:
                sandbox, spec = self._new_sandbox(image=image, task_name=name)
                state.sandbox = sandbox
                documents = task_documents_dir(
                    task_dir, task, root=resolve_repo_path(self.config.harbor_tasks_cache_dir)
                )
                with tempfile.TemporaryDirectory(prefix="lab-seed-") as temporary:
                    archive = Path(temporary) / "inputs.tar.gz"
                    await asyncio.to_thread(_archive_inputs, archive, [(documents, "vdr"), (skills, "skills")])

                    async def setup(box: AsyncSandbox) -> None:
                        await box.upload(archive, "/tmp/lab-inputs.tar.gz")
                        result = await box.exec(
                            "mkdir -p /workspace/output /workspace/workspace && tar -xzf /tmp/lab-inputs.tar.gz -C /workspace && rm /tmp/lab-inputs.tar.gz",
                            timeout_s=self.config.staging_timeout_seconds,
                        )
                        self._check_exec(result, "prepare task inputs")

                    await sandbox.start_with_setup(spec, setup)
                state.setup_complete = True
                return await self._seed_response(request, session_id, state)
            except BaseException:
                state.setup_complete = False
                # Keep the handle if release fails, so an explicit close can retry it.
                try:
                    await self._stop_task(state)
                except Exception:
                    LOG.exception("LAB failed-seed cleanup needs a later close: %s", session_id)
                else:
                    self._sessions.pop(session_id, None)
                    self._slots.release()
                raise

    async def _seed_response(
        self, request: Request, session_id: str, state: _TaskSession
    ) -> ResourcesSeedSessionResponse:
        if not state.setup_complete:
            raise ResourcesSessionConflictError(
                "LAB session setup did not complete; close it and use a new session ID"
            )
        if state.sandbox is None:
            raise ResourcesSessionConflictError("LAB task sandbox has already been released")
        descriptor = await state.sandbox.serialize()
        request.session[SESSION_ID_KEY] = session_id
        return ResourcesSeedSessionResponse(
            resources_session_id=session_id,
            observation=state.prompt,
            sandbox_access=SandboxAccess(
                connection=DirectSandboxConnection(
                    provider_config_ref=self.config.sandbox_provider, descriptor=descriptor
                ),
                workdir="/workspace",
                env=dict(_ENV),
            ),
        )

    @staticmethod
    def _check_exec(result: SandboxExecResult, operation: str) -> None:
        if result.error_type is not None or result.return_code != 0:
            raise RuntimeError(
                f"LAB could not {operation}: {result.error_type or result.stderr or result.return_code}"
            )

    async def _stop_task(self, state: _TaskSession) -> None:
        if state.verifier is not None:
            await state.verifier.stop()
            state.verifier = None
        if state.sandbox is not None:
            await state.sandbox.stop()
            state.sandbox = None

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        async with self._locks.locked(body.resources_session_id):
            closed = self._fences.get(body.resources_session_id)
            if closed is not None and closed != body.episode_id:
                raise ResourcesSessionConflictError("LAB close belongs to another episode")
            state = self._sessions.get(body.resources_session_id)
            if state is not None:
                if state.seed.episode_id != body.episode_id:
                    raise ResourcesSessionConflictError("LAB close belongs to another episode")
                await self._stop_task(state)
                self._sessions.pop(body.resources_session_id)
                self._slots.release()
            self._fences.fence(body.resources_session_id, body.episode_id)
            return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    async def verify(self, request: Request, body: LabSessionVerifyRequest) -> LabSessionVerifyResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        if not isinstance(session_id, str):
            return LabSessionVerifyResponse.model_validate(
                body.model_dump() | lost_session_verification("LAB session cookie missing")
            )
        async with self._locks.locked(session_id):
            state = self._sessions.get(session_id)
            if state is None:
                return LabSessionVerifyResponse.model_validate(
                    body.model_dump() | lost_session_verification("LAB session is closed or lost")
                )
            if state.seed.task_data.get("instance_id") != body.instance_id:
                raise ResourcesSessionConflictError("LAB verify belongs to another task")
            if state.verification_request is not None and state.verification_request != body:
                raise ResourcesSessionConflictError("LAB verification is bound to another agent response")
            if state.verification_request is None:
                state.verification_request = body.model_copy(deep=True)
            if state.result is not None:
                return state.result.model_copy(deep=True)
            state.artifact_dir.mkdir(parents=True, exist_ok=True)
            if state.export is None:
                if state.sandbox is None:
                    raise RuntimeError("LAB sandbox was lost before output capture")
                transfer = state.artifact_dir / ".output-transfer.tar.gz"
                result = await state.sandbox.exec(
                    "tar -czf /tmp/lab-output.tar.gz -C /workspace output",
                    timeout_s=self.config.staging_timeout_seconds,
                )
                self._check_exec(result, "capture task output")
                await download_archive(
                    state.sandbox,
                    "/tmp/lab-output.tar.gz",
                    transfer,
                    provider_config=self._provider()[0],
                    limits=self.config.archive_limits,
                )
                state.export = transfer
            # No host-written artifact is published through an agent-controlled mount.
            await self._stop_task(state)
            await extract_outputs(state.export, state.artifact_dir, prefix="output", limits=self.config.archive_limits)
            (state.artifact_dir / "agent-response.json").write_text(body.response.model_dump_json(indent=2))
            (state.artifact_dir / "transcript.jsonl").write_text(
                "".join(item.model_dump_json() + "\n" for item in body.response.output)
            )
            provenance = {
                "source_revision": LAB_SOURCE_REVISION,
                "gym_commit": installed_gym_commit(),
                "image": state.image,
                "instance_id": body.instance_id,
                "response_model": body.response.model,
                "judge_profile": self.config.judge_profile,
                "harness_config": self._harness_provenance(),
            }
            (state.artifact_dir / "config.json").write_text(
                json.dumps({"run_id": session_id, "task": body.instance_id, **provenance})
            )
            (state.artifact_dir / "metadata.json").write_text(json.dumps(provenance, indent=2))
            reward, scores = await self._grade(state)
            state.result = LabSessionVerifyResponse.model_validate(
                body.model_dump()
                | {
                    "reward": reward["reward"],
                    "lab_metrics": reward,
                    "lab_scores": scores,
                    "source_revision": LAB_SOURCE_REVISION,
                    "artifact_dir": str(state.artifact_dir),
                }
            )
            (state.artifact_dir / "result.json").write_text(state.result.model_dump_json(indent=2))
            return state.result.model_copy(deep=True)

    @staticmethod
    def _harness_provenance() -> dict[str, JsonValue]:
        # Runtime pins only: never retain the resolved configuration's credentials.
        result: dict[str, JsonValue] = {}
        for name, instance in get_global_config_dict().items():
            if not isinstance(instance, Mapping) or "responses_api_agents" not in instance:
                continue
            agents = instance["responses_api_agents"]
            if not isinstance(agents, Mapping):
                continue
            for kind, settings in agents.items():
                if not isinstance(settings, Mapping):
                    continue
                result[name] = {
                    "implementation": kind,
                    **{
                        key: settings[key]
                        for key in ("codex_version", "claude_code_version", "max_turns", "model", "reasoning_effort")
                        if key in settings
                    },
                }
                if kind == "hermes_agent":
                    result[name]["runtime_pin"] = "26bb847a88493342ca1b194e0455b479073ae21d"
        return result

    async def _grade(self, state: _TaskSession) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
        if state.verifier is not None:
            await state.verifier.stop()
            state.verifier = None
        box, spec = self._new_sandbox(image=state.image, task_name=state.task_dir.name)
        state.verifier = box
        with tempfile.TemporaryDirectory(prefix="lab-verifier-") as temporary:
            root = Path(temporary)
            payload = root / "verification-inputs.tar.gz"
            await asyncio.to_thread(
                _archive_inputs,
                payload,
                [
                    (state.task_dir / "tests", "tests"),
                    (state.artifact_dir / "output", "lab-run/output"),
                    (state.artifact_dir / "config.json", "lab-run/config.json"),
                ],
            )

            async def setup(sandbox: AsyncSandbox) -> None:
                await sandbox.upload(payload, "/tmp/lab-verifier-inputs.tar.gz")
                result = await sandbox.exec(
                    "tar -xzf /tmp/lab-verifier-inputs.tar.gz -C /workspace && rm /tmp/lab-verifier-inputs.tar.gz && mkdir -p /workspace/verification",
                    timeout_s=self.config.staging_timeout_seconds,
                )
                self._check_exec(result, "stage isolated verification")

            try:
                await box.start_with_setup(spec, setup)
                result = await box.exec(
                    "python3 /workspace/tests/legal_agent_bench_verify.py --task-json /workspace/tests/task.json --run-dir /workspace/lab-run --verifier-dir /workspace/verification --reward-json /workspace/verification/reward.json",
                    env=_build_verifier_env(self.config),
                    timeout_s=self.config.verifier_timeout_seconds,
                )
                # Preserve reports before propagating any nonzero verifier outcome.
                collected = await box.exec(
                    "tar -czf /tmp/lab-verifier-output.tar.gz -C /workspace verification",
                    timeout_s=self.config.staging_timeout_seconds,
                )
                self._check_exec(collected, "collect verifier reports")
                await download_archive(
                    box,
                    "/tmp/lab-verifier-output.tar.gz",
                    state.artifact_dir / ".verifier-transfer.tar.gz",
                    provider_config=self._provider()[0],
                    limits=self.config.archive_limits,
                )
            finally:
                await box.stop()
                state.verifier = None
            await extract_outputs(
                state.artifact_dir / ".verifier-transfer.tar.gz",
                state.artifact_dir,
                prefix="verification",
                limits=self.config.archive_limits,
            )
            self._check_exec(result, "complete grading")
        reward = json.loads((state.artifact_dir / "verification" / "reward.json").read_text())
        if (
            not isinstance(reward, dict)
            or reward.get("verifier_error")
            or reward.get("judge_error_count")
            or type(reward.get("reward")) not in (int, float)
            or not math.isfinite(reward["reward"])
            or not 0 <= reward["reward"] <= 1
        ):
            raise RuntimeError("LAB verifier did not produce a complete valid reward")
        score_name = "scores_dual.json" if self.config.judge_profile == "official_dual" else "scores.json"
        scores = json.loads((state.artifact_dir / "verification" / score_name).read_text())
        return reward, scores


if __name__ == "__main__":
    LabSessionResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = LabSessionResourcesServer.run_webserver()  # noqa: F401
