# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native SOL-ExecBench verification in one disposable OpenSandbox GPU sandbox."""

import asyncio
import hashlib
import json
import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import ClassVar, Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, JsonValue, PrivateAttr, model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.sol_execbench.metrics import aggregate_results
from resources_servers.sol_execbench.problem_store import (
    NATIVE_REVISION,
    Problem,
    ProblemManifest,
    Sha256,
    canonical_json,
    checked_asset,
    load_manifest,
    safe_relative_path,
)


logger = logging.getLogger(__name__)
HERE = Path(__file__).parent
REMOTE = "/sol-eval"
CANDIDATE_FAILURES = {"INCORRECT_SHAPE", "INCORRECT_DTYPE", "INCORRECT_NUMERICAL", "COMPILE_ERROR", "REWARD_HACK"}


class BenchmarkConfig(BaseModel):
    """The pinned native defaults, kept explicit in every attempt's protocol."""

    model_config = ConfigDict(extra="forbid")
    warmup_runs: int = Field(default=10, ge=0)
    iterations: int = Field(default=50, gt=0)
    lock_clocks: bool = False
    benchmark_reference: bool = False
    seed: int = 200


class SolExecBenchResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    num_workers: Literal[1] = 1
    problem_manifest_path: Path
    problem_manifest_sha256: Sha256
    artifact_root: Path
    sandbox_provider: str | dict[str, JsonValue] = "sandbox"
    sandbox_image: str = Field(pattern=r"^.+@sha256:[0-9a-f]{64}$")
    target_hardware: Literal["B200", "LOCAL"] = "B200"
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    compile_timeout_s: int = Field(default=120, gt=0)
    evaluation_timeout_s: int = Field(default=600, gt=0)
    runner_timeout_s: int = Field(default=900, gt=0)
    sandbox_ready_timeout_s: int = Field(default=300, gt=0)
    samples_per_task: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def timeout_order(self) -> "SolExecBenchResourcesServerConfig":
        if self.runner_timeout_s <= self.compile_timeout_s + self.evaluation_timeout_s:
            raise ValueError("runner_timeout_s must exceed both native timeout budgets combined")
        return self


class VerifierMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(min_length=1)
    problem_digest: Sha256


class SolExecBenchVerifyRequest(BaseVerifyRequest):
    verifier_metadata: VerifierMetadata
    rollout_index: int = Field(default=0, alias="_ng_rollout_index", ge=0)


class NativeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outcome: str
    solved: bool = False
    infrastructure_error: bool = False
    detail: str = ""
    # No public, verified SOL anchors are embedded or inferred from reference timing.
    sol_score: float | None = None
    latency_ms: dict[str, float] = Field(default_factory=dict)
    reference_latency_ms: dict[str, float] = Field(default_factory=dict)
    native_traces: list[dict[str, JsonValue]] = Field(default_factory=list)
    hardware: dict[str, JsonValue] = Field(default_factory=dict)


class SolExecBenchVerifyResponse(BaseVerifyResponse, NativeResult):
    verifier_metadata: VerifierMetadata
    request_id: Sha256
    task_id: str
    protocol_sha256: Sha256
    artifact_path: str
    rollout_index: int


def unresolved(outcome: str, detail: str) -> NativeResult:
    """An unmeasured sample has no score and must be masked by Gym."""
    return NativeResult(outcome=outcome, infrastructure_error=True, detail=detail)


def classify_native_result(
    *,
    problem: Problem,
    solution_name: str,
    return_code: int,
    traces: list[dict],
    benchmark_reference: bool,
) -> NativeResult:
    """Classify complete native traces, never process success or RPC elapsed time."""
    expected = {workload["uuid"] for workload in problem.workloads}
    observed = [trace.get("workload", {}).get("uuid") for trace in traces]
    if len(observed) != len(expected) or set(observed) != expected:
        return unresolved("INCOMPLETE_TRACE", "Native traces must cover each trusted workload UUID exactly once")
    statuses = []
    latencies, references = {}, {}
    for trace in traces:
        if trace.get("definition") != problem.definition.get("name") or trace.get("solution") != solution_name:
            return unresolved("INVALID_TRACE", "Native trace definition or solution identity mismatch")
        evaluation = trace.get("evaluation")
        if not isinstance(evaluation, dict):
            return unresolved("INVALID_TRACE", "Native trace is missing its evaluation")
        status = evaluation.get("status")
        statuses.append(status)
        if status not in CANDIDATE_FAILURES | {"PASSED"}:
            # RUNTIME_ERROR also represents missing inputs, clock-lock failures and timing failures.
            return unresolved("NATIVE_UNRESOLVED", f"Native status {status!r}: {evaluation.get('log', '')}")
        if status == "PASSED":
            correctness = evaluation.get("correctness")
            performance = evaluation.get("performance")
            if not isinstance(correctness, dict) or not isinstance(performance, dict):
                return unresolved("INVALID_TRACE", "Passing native traces require correctness and performance")
            if correctness.get("has_nan", False) or correctness.get("has_inf", False):
                return unresolved("INVALID_TRACE", "Passing trace contains nonfinite correctness results")
            for field in ("latency_ms", "reference_latency_ms", "speedup_factor"):
                value = performance.get(field)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    return unresolved("INVALID_TRACE", "Native performance values must be finite and nonnegative")
            if performance["latency_ms"] <= 0 or (
                benchmark_reference
                and (performance["reference_latency_ms"] <= 0 or performance["speedup_factor"] <= 0)
            ):
                return unresolved(
                    "INVALID_TRACE", "Measured native latency and enabled reference metrics must be positive"
                )
            uid = trace["workload"]["uuid"]
            latencies[uid] = performance["latency_ms"]
            if benchmark_reference:
                references[uid] = performance["reference_latency_ms"]
    solved = all(status == "PASSED" for status in statuses)
    if return_code != (0 if solved else 1):
        return unresolved("INVALID_EXIT_STATUS", "Native CLI exit status disagrees with complete workload traces")
    return NativeResult(
        outcome="PASSED" if solved else "CANDIDATE_FAILED",
        solved=solved,
        latency_ms=latencies,
        reference_latency_ms=references,
        native_traces=traces,
    )


def extract_solution(body: SolExecBenchVerifyRequest, problem: Problem) -> dict:
    """Accept one native Solution JSON object with inline sources, optionally JSON fenced."""
    texts = [
        part.text
        for item in body.response.output
        if item.type == "message" and item.role == "assistant"
        for part in item.content
        if part.type == "output_text"
    ]
    text = "\n".join(texts).strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[len("```json\n") : -len("\n```")]
    solution = json.loads(text)
    if not isinstance(solution, dict) or solution.get("definition") != problem.definition.get("name"):
        raise ValueError("Return a native Solution object for the requested definition")
    sources = solution.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Solution requires inline sources")
    reserved = {"definition.json", "workload.jsonl", "solution.json", "config.json", "eval_driver.py", "build_ext.py"}
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            raise ValueError("Each source requires a relative path")
        path = safe_relative_path(source["path"])
        if path in reserved or not isinstance(source.get("content"), str) or not source["content"]:
            raise ValueError("Sources must have inline content and cannot replace native harness files")
    # The installed native Solution schema performs full language/build validation in the sandbox.
    canonical_json(solution)
    return solution


def write_json(path: Path, value: object) -> None:
    """Create a record once; an interrupted write remains visible and is not retried."""
    with path.open("xb") as stream:
        stream.write(canonical_json(value) + b"\n")


class SolExecBenchResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    config: SolExecBenchResourcesServerConfig
    _manifest: ProblemManifest = PrivateAttr()
    _provider_config: dict = PrivateAttr()
    _protocol: dict = PrivateAttr()
    _protocol_sha256: str = PrivateAttr()
    _semaphore: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(1))
    _inflight: dict[str, asyncio.Task] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._manifest = load_manifest(self.config.problem_manifest_path, self.config.problem_manifest_sha256)
        self._provider_config = resolve_provider_config(
            self.config.sandbox_provider, self.server_client.global_config_dict
        )
        if set(self._provider_config) != {"opensandbox"}:
            raise ValueError("The native SOL verifier requires OpenSandbox")
        operations = self._provider_config["opensandbox"].setdefault("operations", {})
        if operations.get("command_retries", 0) != 0:
            raise ValueError("OpenSandbox command retries must be disabled")
        operations["command_retries"] = 0
        self._protocol = {
            "native_revision": NATIVE_REVISION,
            "problem_manifest_sha256": self.config.problem_manifest_sha256,
            "sandbox_image": self.config.sandbox_image,
            "target_hardware": self.config.target_hardware,
            "benchmark": self.config.benchmark.model_dump(),
            "compile_timeout_s": self.config.compile_timeout_s,
            "evaluation_timeout_s": self.config.evaluation_timeout_s,
            "runner_timeout_s": self.config.runner_timeout_s,
            "native_source_hashes": json.loads((HERE / "native_source_hashes.json").read_text()),
            "runner_sha256": hashlib.sha256((HERE / "native_runner.py").read_bytes()).hexdigest(),
            "verifier_source_hashes": {
                name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                for name in ("app.py", "metrics.py", "problem_store.py")
            },
            "reward": "native_all_workloads_correct",
        }
        self._protocol_sha256 = hashlib.sha256(canonical_json(self._protocol)).hexdigest()
        self.config.artifact_root.mkdir(parents=True, exist_ok=True)

    async def verify(self, body: SolExecBenchVerifyRequest) -> SolExecBenchVerifyResponse:
        problem = next((p for p in self._manifest.problems if p.task_id == body.verifier_metadata.task_id), None)
        if body.rollout_index >= self.config.samples_per_task:
            raise ValueError("Rollout index exceeds configured samples_per_task")
        if problem is None or problem.problem_digest != body.verifier_metadata.problem_digest:
            raise ValueError("Verifier metadata must match the server-owned problem manifest")
        request_data = {"protocol_sha256": self._protocol_sha256, "request": body.model_dump(mode="json")}
        request_id = hashlib.sha256(canonical_json(request_data)).hexdigest()
        if request_id not in self._inflight:
            task = asyncio.create_task(self._evaluate(request_id, request_data, body, problem))
            self._inflight[request_id] = task
            task.add_done_callback(lambda finished: self._inflight.pop(request_id, None))
        result = await asyncio.shield(self._inflight[request_id])
        return SolExecBenchVerifyResponse(
            **body.model_dump(),
            **result.model_dump(),
            request_id=request_id,
            task_id=problem.task_id,
            protocol_sha256=self._protocol_sha256,
            artifact_path=str(self.config.artifact_root / request_id),
            reward=float(result.solved),
            mask_sample=result.infrastructure_error,
            failure_kind=f"sol_execbench:{result.outcome.lower()}" if result.infrastructure_error else None,
            failure_reason=result.detail if result.infrastructure_error else None,
        )

    async def _evaluate(
        self, request_id: str, request_data: dict, body: SolExecBenchVerifyRequest, problem: Problem
    ) -> NativeResult:
        async with self._semaphore:
            attempt = self.config.artifact_root / request_id
            try:
                attempt.mkdir()
            except FileExistsError:
                try:
                    if json.loads((attempt / "request.json").read_bytes()) != request_data:
                        raise ValueError("Cached request identity mismatch")
                    return NativeResult.model_validate_json((attempt / "result.json").read_bytes())
                except (OSError, ValueError) as exc:
                    return unresolved("ATTEMPT_UNRESOLVED", str(exc))
            write_json(attempt / "request.json", request_data)
            write_json(attempt / "protocol.json", self._protocol)
            try:
                solution = extract_solution(body, problem)
            except (ValueError, TypeError) as exc:
                result = NativeResult(outcome="INVALID_SOLUTION", detail=str(exc))
            else:
                try:
                    result = await self._run(attempt, problem, solution)
                except Exception as exc:
                    logger.exception("Native SOL sandbox failed")
                    result = unresolved("SANDBOX_FAILURE", str(exc))
            write_json(attempt / "result.json", result.model_dump(mode="json"))
            return result

    async def _run(self, attempt: Path, problem: Problem, solution: dict) -> NativeResult:
        write_json(attempt / "definition.json", problem.definition)
        (attempt / "workload.jsonl").write_bytes(b"\n".join(canonical_json(w) for w in problem.workloads) + b"\n")
        write_json(attempt / "solution.json", solution)
        write_json(attempt / "config.json", self.config.benchmark.model_dump())
        assets = [
            (checked_asset(self.config.problem_manifest_path.parent, asset), asset.path) for asset in problem.assets
        ]
        sandbox = AsyncSandbox(
            self._provider_config,
            SandboxSpec(
                image=self.config.sandbox_image,
                workdir=REMOTE,
                resources={"gpu": 1},
                entrypoint=["/bin/sh", "-c", "exec sleep infinity"],
                ready_timeout_s=self.config.sandbox_ready_timeout_s,
                ttl_s=self.config.runner_timeout_s + self.config.sandbox_ready_timeout_s + 600,
                metadata={"workload": "sol-execbench"},
            ),
        )
        result = unresolved("SANDBOX_FAILURE", "Sandbox did not complete")
        try:
            await sandbox.start()
            setup = await sandbox.exec("mkdir -p /sol-eval/problem /sol-eval/assets", timeout_s=30)
            if setup.return_code != 0 or setup.error_type:
                raise RuntimeError(f"Sandbox setup failed: {setup.stderr}")
            uploads = {
                "definition.json": "problem/definition.json",
                "workload.jsonl": "problem/workload.jsonl",
                "solution.json": "solution.json",
                "config.json": "config.json",
                "protocol.json": "protocol.json",
            }
            for local, remote in uploads.items():
                await sandbox.upload(attempt / local, f"{REMOTE}/{remote}")
            await sandbox.upload(HERE / "native_runner.py", f"{REMOTE}/native_runner.py")
            await sandbox.upload(HERE / "native_source_hashes.json", f"{REMOTE}/native_source_hashes.json")
            for local, relative in assets:
                await sandbox.upload(local, f"{REMOTE}/assets/{relative}")
            execution = await sandbox.exec(
                "/venv/bin/python /sol-eval/native_runner.py", timeout_s=self.config.runner_timeout_s
            )
            (attempt / "runner.stdout").write_text(execution.stdout or "")
            (attempt / "runner.stderr").write_text(execution.stderr or "")
            # Retain every available artifact even when the native process timed out or failed.
            for name in (
                "hardware.json",
                "validation.json",
                "execution.json",
                "trace.jsonl",
                "native.stdout",
                "native.stderr",
            ):
                try:
                    await sandbox.download(f"{REMOTE}/{name}", attempt / name)
                except Exception as exc:
                    (attempt / f"{name}.download-error.txt").write_text(str(exc))
            if execution.error_type or execution.return_code != 0:
                result = unresolved(
                    "RUNNER_FAILURE", f"error_type={execution.error_type}; exit={execution.return_code}"
                )
            else:
                result = self._read_result(attempt, problem, solution)
        finally:
            try:
                await sandbox.stop()
            except Exception as exc:
                logger.exception("Native SOL sandbox cleanup failed")
                result = unresolved("CLEANUP_FAILURE", str(exc))
        return result

    def _read_result(self, attempt: Path, problem: Problem, solution: dict) -> NativeResult:
        try:
            hardware = json.loads((attempt / "hardware.json").read_bytes())
            validation = json.loads((attempt / "validation.json").read_bytes())
            if validation["valid"] is False:
                return NativeResult(outcome="INVALID_SOLUTION", detail=validation["detail"], hardware=hardware)
            execution = json.loads((attempt / "execution.json").read_bytes())
            if execution.get("native_schema_validated") is not True:
                raise ValueError("Missing complete native-schema-validated traces")
            traces = [json.loads(line) for line in (attempt / "trace.jsonl").read_text().splitlines() if line.strip()]
            result = classify_native_result(
                problem=problem,
                solution_name=solution["name"],
                return_code=execution["return_code"],
                traces=traces,
                benchmark_reference=self.config.benchmark.benchmark_reference,
            )
            result.hardware = hardware
            return result
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return unresolved("INVALID_NATIVE_RESULT", str(exc))

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        return aggregate_results(
            body.verify_responses,
            task_ids=[p.task_id for p in self._manifest.problems],
            samples_per_task=self.config.samples_per_task,
            protocol_sha256=self._protocol_sha256,
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


# The fixture uses synthetic native traces and the production result classifier; no GPU is implied.
from resources_servers.sol_execbench.fixture import NativeFixtureRequest, NativeVerifierFixture  # noqa: E402


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=NativeVerifierFixture,
    request_model=NativeFixtureRequest,
    cases_path=HERE / "tests/verifier_cases.jsonl",
)

if __name__ == "__main__":
    SolExecBenchResourcesServer.run_webserver()
