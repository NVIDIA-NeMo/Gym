# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay saved responses through an explicitly configured, trusted SOL evaluator.

The external evaluator owns extraction, native execution, workload coverage, and
scoring. This server supplies transport, durable attempts, and Gym integration.
It neither installs an evaluator nor provides a sandbox for candidate code.
"""

import asyncio
import hashlib
import json
import os
import signal
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, Any, ClassVar

from fastapi import FastAPI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    field_validator,
    model_validator,
)

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from resources_servers.sol_execbench.metrics import MEASURED_OUTCOMES, aggregate_sol_results


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", strict=True)]
TaskId = Annotated[str, Field(min_length=1, strict=True)]
TERMINATION_GRACE_S = 5.0


def canonical_json(value: Any) -> bytes:
    """Encode identities without newline normalization or non-JSON numeric values."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    """Create an immutable record; an interrupted write remains an unresolved attempt."""
    with path.open("xb") as stream:
        stream.write(canonical_json(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


class ManifestTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: TaskId


class EvaluationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Annotated[int, Field(strict=True, ge=1, le=1)]
    protocol_sha256: Sha256
    samples_per_task: Annotated[int, Field(strict=True, ge=1)]
    tasks: list[ManifestTask] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_tasks(self) -> "EvaluationManifest":
        ids = [task.task_id for task in self.tasks]
        if len(ids) != len(set(ids)) or any(not task.strip() for task in ids):
            raise ValueError("Manifest task IDs must be nonblank and unique")
        return self


class SolExecBenchResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    num_workers: Annotated[int, Field(strict=True, ge=1, le=1)] = 1
    manifest_path: Path
    manifest_sha256: Sha256
    evaluator_command: list[str] = Field(min_length=1)
    artifact_root: Path
    gpu_uuid: str = Field(pattern=r"^GPU-[0-9a-fA-F-]+$")
    runner_timeout_s: float = Field(default=1800, gt=0, allow_inf_nan=False)
    timeout_zero_sensitivity: bool = False

    @field_validator("artifact_root", "manifest_path")
    @classmethod
    def absolute_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("Evaluator paths must be absolute")
        return value

    @field_validator("evaluator_command")
    @classmethod
    def nonempty_argv(cls, value: list[str]) -> list[str]:
        if any(not item or "\x00" in item for item in value):
            raise ValueError("Evaluator argv entries must be nonempty and contain no NUL")
        return value


class SolExecBenchVerifyRequest(BaseVerifyRequest):
    task_id: TaskId
    verifier_metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluatorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: Sha256
    task_id: TaskId
    protocol_sha256: Sha256
    outcome: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    infrastructure_error: StrictBool
    solved: StrictBool
    sol_score: Annotated[float, Field(strict=True, allow_inf_nan=False)] | None
    native_result: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def consistent_outcome(self) -> "EvaluatorResult":
        if self.solved != (self.outcome == "PASSED"):
            raise ValueError("Only PASSED results may be solved")
        if self.infrastructure_error:
            if self.solved or self.sol_score is not None:
                raise ValueError("Infrastructure failures require unsolved and null score")
        elif self.outcome not in MEASURED_OUTCOMES:
            raise ValueError("Unknown or infrastructure outcomes cannot be measured results")
        elif self.sol_score is None or (not self.solved and self.sol_score != 0):
            raise ValueError("Measured results require a score; unsolved results require zero")
        return self


class SolExecBenchVerifyResponse(BaseVerifyResponse, EvaluatorResult):
    verifier_metadata: dict[str, Any] = Field(default_factory=dict)


class SolExecBenchResourcesServer(SimpleResourcesServer):
    config: SolExecBenchResourcesServerConfig
    ray_enabled: ClassVar[bool] = False
    _manifest: EvaluationManifest = PrivateAttr()
    _semaphore: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(1))
    _inflight: dict[str, asyncio.Task[EvaluatorResult]] = PrivateAttr(default_factory=dict)
    _halt_reason: str | None = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        manifest_bytes = self.config.manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != self.config.manifest_sha256:
            raise ValueError("Manifest SHA256 mismatch")
        self._manifest = EvaluationManifest.model_validate_json(manifest_bytes)
        self.config.artifact_root.mkdir(parents=True, exist_ok=True)
        if self._halt_path.exists():
            self._halt_reason = "A prior infrastructure failure stopped this GPU worker"

    @property
    def _halt_path(self) -> Path:
        return self.config.artifact_root / f"worker-{self.config.gpu_uuid}-halted.json"

    def _halt(self, result: EvaluatorResult) -> None:
        if result.infrastructure_error and result.outcome != "EVALUATION_TIMEOUT":
            self._halt_reason = result.outcome
            with suppress(FileExistsError):
                write_json(self._halt_path, result.model_dump(mode="json"))

    def _failure(self, request: dict[str, Any], outcome: str, detail: str) -> EvaluatorResult:
        return EvaluatorResult(
            **{key: request[key] for key in ("request_id", "task_id", "protocol_sha256")},
            outcome=outcome,
            infrastructure_error=True,
            solved=False,
            sol_score=None,
            native_result={"detail": detail},
        )

    def _validate_result(self, path: Path, request: dict[str, Any]) -> EvaluatorResult:
        raw = json.loads(path.read_bytes())
        canonical_json(raw)
        result = EvaluatorResult.model_validate(raw)
        if any(getattr(result, key) != request[key] for key in ("request_id", "task_id", "protocol_sha256")):
            raise ValueError("Evaluator result identity does not match request")
        return result

    async def verify(self, body: SolExecBenchVerifyRequest) -> SolExecBenchVerifyResponse:
        request = {
            "schema_version": 1,
            "task_id": body.task_id,
            "protocol_sha256": self._manifest.protocol_sha256,
            "gpu_uuid": self.config.gpu_uuid,
            "responses_create_params": body.responses_create_params.model_dump(mode="json"),
            "response": body.response.model_dump(mode="json"),
            "verifier_metadata": body.verifier_metadata,
        }
        request["request_id"] = hashlib.sha256(
            canonical_json(
                {
                    "manifest_sha256": self.config.manifest_sha256,
                    "evaluator_command": self.config.evaluator_command,
                    **request,
                }
            )
        ).hexdigest()
        request_id = request["request_id"]
        if body.task_id not in {task.task_id for task in self._manifest.tasks}:
            result = self._failure(request, "UNKNOWN_TASK", "Task is not in the pinned evaluation manifest")
            self._halt(result)
        else:
            task = self._inflight.get(request_id)
            if task is None:
                task = asyncio.create_task(self._evaluate(request))
                self._inflight[request_id] = task
                task.add_done_callback(lambda finished: self._inflight.pop(request_id, None))
            # A disconnected caller must not cancel another caller's identical evaluation.
            result = await asyncio.shield(task)
        response_data = body.model_dump()
        response_data.update(result.model_dump())
        response_data.update(
            reward=result.sol_score if result.sol_score is not None else 0.0,
            mask_sample=result.infrastructure_error,
            failure_kind=f"sol_execbench:{result.outcome.lower()}" if result.infrastructure_error else None,
            failure_reason=str(result.native_result.get("detail", result.outcome))
            if result.infrastructure_error
            else None,
        )
        return SolExecBenchVerifyResponse(**response_data)

    async def _evaluate(self, request: dict[str, Any]) -> EvaluatorResult:
        async with self._semaphore:
            try:
                return await self._evaluate_locked(request)
            except OSError as exc:
                result = self._failure(request, "PERSISTENCE_FAILURE", str(exc))
                with suppress(OSError):
                    self._halt(result)
                return result

    async def _evaluate_locked(self, request: dict[str, Any]) -> EvaluatorResult:
        attempt = self.config.artifact_root / request["request_id"]
        try:
            attempt.mkdir()
        except FileExistsError:
            try:
                if canonical_json(json.loads((attempt / "request.json").read_bytes())) != canonical_json(request):
                    raise ValueError("Stored request differs from request identity")
                result = self._validate_result(attempt / "accepted.json", request)
            except (OSError, ValueError) as exc:
                result = self._failure(request, "ATTEMPT_UNRESOLVED", str(exc))
            self._halt(result)
            return result
        write_json(attempt / "request.json", request)
        if self._halt_reason:
            result = self._failure(request, "WORKER_STOPPED", self._halt_reason)
        else:
            result = await self._run(request, attempt)
        write_json(attempt / "accepted.json", result.model_dump(mode="json"))
        self._halt(result)
        return result

    async def _run(self, request: dict[str, Any], attempt: Path) -> EvaluatorResult:
        argv = [
            *self.config.evaluator_command,
            "--request",
            str(attempt / "request.json"),
            "--result",
            str(attempt / "result.json"),
        ]
        process: asyncio.subprocess.Process | None = None
        record: dict[str, Any] = {
            "argv": argv,
            "gpu_uuid": self.config.gpu_uuid,
            "started_at": time.time(),
        }
        cancelled = False
        try:
            with (
                (attempt / "stdout.txt").open("xb") as stdout,
                (attempt / "stderr.txt").open("xb") as stderr,
            ):
                process = await asyncio.create_subprocess_exec(
                    *argv,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": self.config.gpu_uuid},
                )
                record["pid"] = process.pid
                await asyncio.wait_for(process.wait(), timeout=self.config.runner_timeout_s)
            if process.returncode != 0:
                result = self._failure(
                    request,
                    "RUNNER_FAILED",
                    f"Evaluator exited with status {process.returncode}",
                )
            else:
                try:
                    result = self._validate_result(attempt / "result.json", request)
                except (OSError, ValueError) as exc:
                    result = self._failure(request, "INVALID_RESULT", str(exc))
        except (TimeoutError, asyncio.CancelledError) as exc:
            cancelled = isinstance(exc, asyncio.CancelledError)
            result = self._failure(
                request,
                "RUNNER_CANCELLED" if cancelled else "RUNNER_TIMEOUT",
                "Evaluator process group stopped by the transport watchdog or server shutdown",
            )
            if process is not None:
                record["termination_grace_s"] = TERMINATION_GRACE_S
                await self._stop_process_group(process)
        except OSError as exc:
            result = self._failure(request, "RUNNER_FAILED", str(exc))
        record.update(
            ended_at=time.time(),
            returncode=process.returncode if process else None,
            outcome=result.outcome,
        )
        write_json(attempt / "process.json", record)
        if cancelled:
            write_json(attempt / "accepted.json", result.model_dump(mode="json"))
            self._halt(result)
            raise asyncio.CancelledError
        return result

    @staticmethod
    async def _stop_process_group(process: asyncio.subprocess.Process) -> None:
        """Allow the trusted runner to clean detached children, then reap its group.

        Detached process groups are owned by the configured runner/runtime. Sending
        TERM first gives its handler a bounded opportunity to clean those groups.
        """
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        with suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=TERMINATION_GRACE_S)
        # The group can outlive its leader, so do this even if wait() already completed.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await process.wait()

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        return aggregate_sol_results(
            body.verify_responses,
            task_ids=[task.task_id for task in self._manifest.tasks],
            samples_per_task=self._manifest.samples_per_task,
            protocol_sha256=self._manifest.protocol_sha256,
            timeout_zero_sensitivity=self.config.timeout_zero_sensitivity,
        )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(application: FastAPI):
            try:
                async with parent_lifespan(application):
                    yield
            finally:
                tasks = list(self._inflight.values())
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        app.router.lifespan_context = lifespan
        return app


if __name__ == "__main__":
    SolExecBenchResourcesServer.run_webserver()
