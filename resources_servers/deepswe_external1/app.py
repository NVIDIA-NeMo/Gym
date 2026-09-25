# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSWE-style tasks with committed-patch transfer into a fresh verifier."""

from __future__ import annotations

import hashlib
import logging
from contextlib import asynccontextmanager
from math import ceil
from pathlib import Path
from time import monotonic
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, Request
from pydantic import Field, model_validator

from nemo_gym.base_resources_server import SimpleResourcesServer
from nemo_gym.failure_kinds import VERIFIER_ERROR
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from resources_servers.deepswe.app import (
    DeepSWEResourcesServer,
    DeepSWEResourcesServerConfig,
    DeepSWESeedSessionRequest,
    DeepSWESeedSessionResponse,
    DeepSWEVerifyRequest,
    DeepSWEVerifyResponse,
    VerifierResult,
    _resolve_repo_path,
    _resolve_task,
)
from resources_servers.deepswe.task_store import task_sandbox_resources
from resources_servers.deepswe_external1.task_store import PreparedTask, PreparedTaskStore


logger = logging.getLogger(__name__)

VERIFIER_PYTHON_SETUP = """\
set -eu
if ! command -v python3 >/dev/null 2>&1; then
    if [ "$ALLOW_PYTHON_INSTALL" != 1 ]; then
        echo "Verifier Python is missing; a network-disabled verifier needs Python preinstalled in its image." >&2
        exit 1
    fi
    if [ "$(id -u)" != 0 ]; then
        echo "Installing verifier Python requires root; use a verifier image with Python preinstalled." >&2
        exit 1
    fi
    echo "Installing Python in the disposable verifier sandbox."
    if command -v apt-get >/dev/null 2>&1; then
        export DEBIAN_FRONTEND=noninteractive
        apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update
        apt-get -o DPkg::Lock::Timeout=60 -o Acquire::Retries=2 -y --no-install-recommends install python3
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache python3
    elif command -v microdnf >/dev/null 2>&1; then
        microdnf -y install python3
    elif command -v dnf >/dev/null 2>&1; then
        dnf -y install python3
    elif command -v yum >/dev/null 2>&1; then
        yum -y install python3
    else
        echo "No supported package manager for verifier Python; use a verifier image with Python preinstalled." >&2
        exit 1
    fi
fi
python3 --version
"""


class DeepsweExternal1ResourcesServerConfig(DeepSWEResourcesServerConfig):
    logs_dir: Path = Path("resources_servers/deepswe_external1/logs")
    clear_verifier_logs: Literal[False] = False
    is_verifying_null_patch: bool = False
    enforce_verifier_no_network: bool = False

    @model_validator(mode="after")
    def one_validation_mode(self) -> DeepsweExternal1ResourcesServerConfig:
        if self.is_verifying_golden_patch and self.is_verifying_null_patch:
            raise ValueError("Golden and null validation modes are mutually exclusive")
        return self


class DeepsweExternal1SeedSessionRequest(DeepSWESeedSessionRequest):
    task_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")


class DeepsweExternal1VerifyRequest(DeepSWEVerifyRequest):
    task_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")


class DeepsweExternal1VerifyResponse(DeepSWEVerifyResponse):
    validation_mode: Literal["agent", "golden", "null"]
    task_fingerprint: str
    agent_sandbox_id: str | None = None
    verifier_sandbox_id: str | None = None
    golden_execution_time_s: float = 0.0
    failure_stage: str | None = None
    cleanup_errors: list[str] = Field(default_factory=list)


class DeepsweExternal1ResourcesServer(DeepSWEResourcesServer):
    """Reuse DeepSWE staging/grading with validated, independently prepared tasks."""

    config: DeepsweExternal1ResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        SimpleResourcesServer.model_post_init(self, context)
        self._task_store = PreparedTaskStore(
            _resolve_repo_path(self.config.tasks_dir), expected_task_count=self.config.expected_task_count
        )
        self._agent_sessions = {}

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(application: FastAPI):
            async with original_lifespan(application) as state:
                try:
                    yield state
                finally:
                    sessions = list(self._agent_sessions.values())
                    self._agent_sessions.clear()
                    for session in sessions:
                        await self._stop_sandbox(session.sandbox, task_id=session.task_id, phase="server-shutdown")

        app.router.lifespan_context = lifespan
        return app

    def _checked_task(self, body: DeepsweExternal1SeedSessionRequest | DeepsweExternal1VerifyRequest) -> PreparedTask:
        task = _resolve_task(body, self._task_store)
        if not isinstance(task, PreparedTask) or body.task_fingerprint != task.definition.fingerprint():
            raise ValueError("Request does not match the prepared task fingerprint")
        task.validate_assets()
        return task

    def _provider_options(self, *, phase: str) -> dict[str, Any]:
        options = super()._provider_options(phase=phase)
        if phase != "agent" and self.config.enforce_verifier_no_network:
            options["network_policy"] = {"defaultAction": "deny", "egress": []}
        return options

    async def _create_sandbox(self, task: PreparedTask, *, phase: str) -> AsyncSandbox:
        global_config = get_global_config_dict()
        definition = task.definition
        agent = phase == "agent"
        phase_limits = definition.agent if agent else definition.verifier
        resources = task_sandbox_resources(task, phase=phase)
        resources["cpu"] *= self.config.task_cpu_multiplier
        resources["memory_mib"] = ceil(resources["memory_mib"] * self.config.task_memory_multiplier)
        resources.update(self.config.sandbox_config.get("resources", {}))
        spec = SandboxSpec(
            image=definition.image if agent else definition.verifier_image,
            workdir=definition.workdir,
            ttl_s=self.config.sandbox_config.get("ttl_s"),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s"),
            env=phase_limits.env | dict(self.config.sandbox_config.get("env", {})),
            files={},
            metadata=resolve_provider_metadata(self.config.sandbox_provider, global_config)
            | dict(self.config.sandbox_config.get("metadata", {}))
            | {"task": definition.task_id[:63].rstrip("._-"), "phase": phase, "nemo_gym_agent": self.config.name},
            resources=SandboxResources.from_mapping(resources),
            provider_options=self._provider_options(phase=phase),
        )
        sandbox = AsyncSandbox(resolve_provider_config(self.config.sandbox_provider, global_config))

        async def setup(started: AsyncSandbox) -> None:
            command = (
                "set -eu; cd /app; git config --global --add safe.directory /app; "
                'test "$(git rev-parse --show-toplevel)" = /app; '
                f"git cat-file -e {definition.base_commit}^{{commit}}; "
            )
            if agent:
                command += (
                    "git config --global user.email agent@nemo-gym.local; "
                    "git config --global user.name 'NeMo Gym Agent'"
                )
            else:
                command += "mkdir -p /tests /logs/artifacts /logs/verifier"
            result = await started.exec(command, timeout_s=60)
            if result.return_code != 0:
                raise RuntimeError(f"{phase} image setup failed: {(result.stderr or '')[-2000:]}")
            if not agent:
                await self._ensure_verifier_python(started)

        await sandbox.start_with_setup(spec, setup)
        return sandbox

    async def _ensure_verifier_python(self, sandbox: AsyncSandbox) -> None:
        # Run only in fresh B, before staging any candidate code or held-out tests.
        # Never relax an explicitly requested no-network policy to install packages.
        allow_install = int(not self.config.enforce_verifier_no_network)
        result = await sandbox.exec(
            f"ALLOW_PYTHON_INSTALL={allow_install}\n" + VERIFIER_PYTHON_SETUP,
            timeout_s=300,
        )
        details = ((result.stdout or "") + (result.stderr or "")).strip()
        if result.return_code != 0:
            raise RuntimeError(f"Verifier Python setup failed (exit {result.return_code}): {details[-4000:]}")
        logger.info("Verifier Python setup: %s", details)

    async def _stop_sandbox(self, sandbox: AsyncSandbox, *, task_id: str, phase: str) -> None:
        await self._release_sandbox(sandbox, task_id=task_id, phase=phase)

    async def _release_sandbox(self, sandbox: AsyncSandbox, *, task_id: str, phase: str) -> bool:
        try:
            await sandbox.stop()
            return True
        except Exception:
            logger.exception("Could not release %s sandbox for %s", phase, task_id)
            return False

    async def seed_session(
        self, request: Request, body: DeepsweExternal1SeedSessionRequest
    ) -> DeepSWESeedSessionResponse:
        self._checked_task(body)
        if self.config.is_verifying_null_patch:
            raise RuntimeError("seed_session is unavailable in null-validation mode")
        return await super().seed_session(request, body)

    async def _execute_golden(self, sandbox: AsyncSandbox, task: PreparedTask, log_dir: Path) -> None:
        result = await sandbox.exec("mkdir -p /solution", timeout_s=60)
        if result.return_code != 0:
            raise RuntimeError("Could not create the golden solution directory")
        for filename in ("solve.sh", "solution.patch"):
            await sandbox.upload(task.asset_path(f"solution/{filename}"), f"/solution/{filename}")
        result = await sandbox.exec(
            "bash /solution/solve.sh", cwd=task.definition.workdir, timeout_s=task.definition.solution_timeout_sec
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "golden.log").write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
        if result.return_code != 0:
            raise RuntimeError(f"Golden solution failed with exit code {result.return_code}")

    async def verify(self, request: Request, body: DeepsweExternal1VerifyRequest) -> DeepsweExternal1VerifyResponse:
        try:
            task = self._checked_task(body)
        except Exception:
            session = self._agent_sessions.pop(str(request.session.get(SESSION_ID_KEY, "")), None)
            if session is not None:
                await self._stop_sandbox(session.sandbox, task_id=session.task_id, phase="invalid-verification")
            raise
        return await self._verify_task(request, body, task)

    async def _verify_task(
        self, request: Request, body: DeepsweExternal1VerifyRequest, task: PreparedTask
    ) -> DeepsweExternal1VerifyResponse:
        mode: Literal["agent", "golden", "null"] = "agent"
        if self.config.is_verifying_golden_patch:
            mode = "golden"
        elif self.config.is_verifying_null_patch:
            mode = "null"
        task_id = task.definition.task_id
        session_id = str(request.session.get(SESSION_ID_KEY, "validation"))
        # Session IDs are untrusted path components; each attempt gets its own generated log directory.
        log_dir = _resolve_repo_path(self.config.logs_dir) / task_id / uuid4().hex
        model_patch = b""
        agent_sandbox: AsyncSandbox | None = None
        verifier_sandbox: AsyncSandbox | None = None
        agent_id = verifier_id = None
        collect_time = start_time = verify_time = golden_time = 0.0
        result = VerifierResult(evaluation_completed=False, reward=0.0)
        cleanup_errors: list[str] = []
        failure_stage: str | None = "agent_setup"
        try:
            if mode == "agent":
                session = self._agent_sessions.pop(session_id, None)
                if session is None:
                    raise RuntimeError("No seeded agent sandbox exists for this session")
                agent_sandbox = session.sandbox
                agent_id = session.sandbox_handle
                if session.task_id != task_id or session.image != task.definition.image:
                    raise RuntimeError("Seeded session task/image does not match verification")
                if body.sandbox_handle is not None and body.sandbox_handle != agent_id:
                    raise RuntimeError("Sandbox handle does not match the seeded session")
            else:
                agent_sandbox = await self._create_sandbox(task, phase="agent")
                agent_id = str((await agent_sandbox.serialize())["sandbox_id"])

            try:
                if mode == "golden":
                    failure_stage = "golden_execution"
                    started = monotonic()
                    await self._execute_golden(agent_sandbox, task, log_dir)
                    golden_time = monotonic() - started
                failure_stage = "patch_collection"
                started = monotonic()
                model_patch = await self._collect_model_patch(agent_sandbox, task)
                collect_time = monotonic() - started
                if mode == "golden" and task.asset_path("solution/solution.patch").stat().st_size and not model_patch:
                    raise RuntimeError("Golden solution produced no committed patch")
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / "model.patch").write_bytes(model_patch)
            finally:
                released = await self._release_sandbox(agent_sandbox, task_id=task_id, phase="agent")
                agent_sandbox = None
                if not released:
                    cleanup_errors.append("agent")

            if cleanup_errors:
                failure_stage = "agent_cleanup"
                raise RuntimeError("Agent sandbox cleanup failed; verifier was not started")

            failure_stage = "verifier_setup"
            started = monotonic()
            verifier_sandbox = await self._create_sandbox(task, phase="verifier")
            verifier_id = str((await verifier_sandbox.serialize())["sandbox_id"])
            if verifier_id == agent_id:
                raise RuntimeError("Provider reused the agent sandbox as the verifier")
            start_time = monotonic() - started
            failure_stage = "native_verifier"
            started = monotonic()
            result = await self._run_verifier(verifier_sandbox, task, model_patch, log_dir)
            verify_time = monotonic() - started
            if result.evaluation_completed:
                failure_stage = None
        except Exception as error:
            logger.exception("Task %s failed during %s", task_id, failure_stage)
            result = VerifierResult(
                evaluation_completed=False, reward=0.0, verifier_error=f"{type(error).__name__}: {error}"
            )
        finally:
            if agent_sandbox is not None:
                if not await self._release_sandbox(agent_sandbox, task_id=task_id, phase="agent"):
                    cleanup_errors.append("agent")
            if verifier_sandbox is not None:
                if not await self._release_sandbox(verifier_sandbox, task_id=task_id, phase="verifier"):
                    cleanup_errors.append("verifier")

        response = DeepsweExternal1VerifyResponse.model_validate(
            body.model_dump()
            | result.model_dump()
            | {
                "task_id": task_id,
                "validation_mode": mode,
                "task_fingerprint": task.definition.fingerprint(),
                "agent_sandbox_id": agent_id,
                "verifier_sandbox_id": verifier_id,
                "golden_execution_time_s": golden_time,
                "model_patch": model_patch.decode("utf-8", errors="replace")
                if self.config.include_model_patch_in_response
                else None,
                "model_patch_sha256": hashlib.sha256(model_patch).hexdigest(),
                "model_patch_bytes": len(model_patch),
                "log_dir": str(log_dir),
                "patch_collection_time_s": collect_time,
                "sandbox_start_time_s": start_time,
                "verification_time_s": verify_time,
                "mask_sample": not result.evaluation_completed,
                "failure_kind": VERIFIER_ERROR if not result.evaluation_completed else None,
                "failure_reason": result.verifier_error,
                "failure_stage": failure_stage,
                "cleanup_errors": cleanup_errors,
            }
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "result.json").write_text(response.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return response


if __name__ == "__main__":
    DeepsweExternal1ResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = DeepsweExternal1ResourcesServer.run_webserver()  # noqa: F401
