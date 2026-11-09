# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource server backed by SpatialClaw's native benchmark evaluators."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from shlex import quote
from typing import Annotated, Any, ClassVar, Literal, Self

from fastapi import FastAPI, Request
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PositiveFloat, PrivateAttr, model_validator

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import ModelServerRef
from nemo_gym.metrics.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.sandbox import SandboxAccess, SandboxConfig
from nemo_gym.server_runtime import (
    BaseSeedSessionRequest,
    LentSandboxes,
    MCPTool,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionLostError,
    lost_session_verification,
    reject_partial_typed_seed,
)
from nemo_gym.server_runtime.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from resources_servers.spatialclaw.session_runtime import SpatialClawSessionRuntime, context_result
from resources_servers.spatialclaw.task_data import TaskData
from responses_api_agents.spatialclaw_agent.app import (
    SPATIALCLAW_COMMIT,
    SPATIALCLAW_URL,
    _config_path,
    _install_source_path,
    _validate_spatialclaw_checkout,
    ensure_spatialclaw_checkout,
)


LOG = logging.getLogger(__name__)


class SpatialClawResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED
    spatialclaw_url: str = SPATIALCLAW_URL
    spatialclaw_commit: str = SPATIALCLAW_COMMIT
    spatialclaw_root: str | None = None
    source_cache_root: str | None = None
    dataset_config: str
    data_root: str | None = None
    model_server: ModelServerRef = Field(
        default_factory=lambda: ModelServerRef(type="responses_api_models", name="policy_model")
    )
    model_config_path: str | None = None
    model_name: str | None = None
    gpu_server_registry: str | None = None
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    max_output_tokens: int | None = None
    workspace_root: str = "outputs/spatialclaw_resources"
    frame_cache_root: str = "outputs/spatialclaw_frame_cache"
    runtime_setup_timeout: PositiveFloat = 600
    runtime_close_timeout: PositiveFloat = 60
    keep_workspaces: bool = True
    score_step_limit_answer: bool = False
    sandbox_provider: str | None = None
    sandbox_config: SandboxConfig = Field(default_factory=SandboxConfig)
    sandbox_image: str | None = None
    sandbox_workdir: str = "/workspace"
    sandbox_network_access: Literal["inherit", "model_only", "model_and_tools"] = "inherit"

    @model_validator(mode="after")
    def validate_sandbox_workdir(self) -> Self:
        path = PurePosixPath(self.sandbox_workdir)
        if not path.is_absolute() or ".." in path.parts or path == PurePosixPath("/"):
            raise ValueError("sandbox_workdir must be an absolute task directory")
        return self


class SpatialClawSeedRequest(TaskData, BaseSeedSessionRequest):
    pass


class ExecuteRequest(BaseModel):
    code: str = Field(max_length=100000)
    timeout: PositiveFloat | None = None


class KernelRequest(BaseModel):
    operation: Literal[
        "execute",
        "start",
        "shutdown",
        "reset_namespace",
        "get_variables",
        "check_sentinel",
        "clear_sentinel",
        "finalize_answer",
    ]
    code: str | None = None
    answer: str | None = None
    timeout: PositiveFloat | None = None
    name: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")

    @model_validator(mode="after")
    def validate_operation_payload(self) -> Self:
        if self.operation == "finalize_answer" and self.answer is None:
            raise ValueError("finalize_answer requires answer")
        if self.operation == "execute" and self.code is None:
            raise ValueError("execute requires code")
        return self


class SpatialClawRunRequest(TaskData, BaseRunRequest):
    pass


class SpatialClawVerifyRequest(SpatialClawRunRequest, BaseVerifyRequest):
    pass


class SpatialClawVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    sample_id: str
    prediction: str
    extracted_answer: str | None
    native_score: float | None
    scored: bool
    evaluation_completed: bool
    step_limit_reached: bool = False


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float | int]:
    """Flatten native result summaries while dropping row-level detail blobs."""
    flattened: dict[str, float | int] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"detailed_results", "details"}:
                continue
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            flattened.update(_flatten_numeric(child, child_prefix))
    elif isinstance(value, bool):
        flattened[prefix] = int(value)
    elif isinstance(value, (int, float)):
        flattened[prefix] = value
    return flattened


class SpatialClawResourcesServer(SimpleResourcesServer):
    """Use the pinned loader for row scoring and its full evaluator for metrics."""

    config: SpatialClawResourcesServerConfig
    seed_request_model: ClassVar[type[BaseSeedSessionRequest]] = SpatialClawSeedRequest
    _benchmark_instance: Any = PrivateAttr(default=None)
    _sample_by_id: dict[str, Any] = PrivateAttr(default_factory=dict)
    _load_lock: asyncio.Lock | None = PrivateAttr(default=None)
    _runtime_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    _session_runtimes: dict[str, SpatialClawSessionRuntime] = PrivateAttr(default_factory=dict)
    _source_root: Path | None = PrivateAttr(default=None)
    _lent_sandboxes: LentSandboxes | None = PrivateAttr(default=None)
    _sandbox_accesses: dict[str, SandboxAccess] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._load_lock = asyncio.Lock()
        if self.config.sandbox_provider:
            self._lent_sandboxes = LentSandboxes(
                self.config.sandbox_provider,
                self.config.sandbox_config,
                image=self.config.sandbox_image,
                workdir=self.config.sandbox_workdir,
                network_access=self.config.sandbox_network_access,
                model_server=self.config.model_server.name
                if self.config.sandbox_network_access != "inherit"
                else None,
                tool_servers=[self.config.name] if self.config.sandbox_network_access == "model_and_tools" else (),
            )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/get_context", response_model=None)(self.get_context)
        app.post("/execute", response_model=None)(self.execute)
        app.post("/kernel")(self.kernel)
        parent = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent(app) as state:
                    yield state
            finally:
                for session_id in list(self._session_runtimes):
                    try:
                        async with self.session_lock(session_id):
                            await self.release_session_state(session_id)
                    except Exception:
                        LOG.exception("SpatialClaw resource cleanup failed for %s", session_id)
                if self._lent_sandboxes is not None:
                    await self._lent_sandboxes.release_all()

        app.router.lifespan_context = lifespan
        return app

    def mcp_tools(self, harvested: list[MCPTool], catchall: Any) -> list[MCPTool]:
        # Native-kernel control is an adapter API, never a model-visible tool.
        return [tool for tool in harvested if tool.name in {"get_context", "execute"}]

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self._session_runtimes

    async def initialize_session_state(self, session_id: str, body: SpatialClawSeedRequest) -> None:
        benchmark = await self._load_benchmark()
        sample = self._sample_by_id.get(body.sample_id)
        if sample is None:
            raise ResourcesSeedRejectedError(f"Unknown SpatialClaw sample id: {body.sample_id}")
        runtime = SpatialClawSessionRuntime.new(session_id, body.sample_id, self.config.workspace_root)
        self._session_runtimes[session_id] = runtime
        identity = self.session_identity(session_id)
        capture_key = identity[0].capture_key if identity is not None else session_id
        async with self._runtime_lock:
            await asyncio.wait_for(
                runtime.initialize(self, sample, benchmark, capture_key, media=body.media),
                timeout=self.config.runtime_setup_timeout,
            )
        if self._lent_sandboxes is not None:
            if identity is None:
                raise ResourcesSeedRejectedError("Lent SpatialClaw sandboxes require a typed episode seed")
            files = await asyncio.to_thread(self._sandbox_files, runtime)
            access = await self._lent_sandboxes.lend(session_id, identity[1], files=files)
            self._sandbox_accesses[session_id] = access.model_copy(update={"workdir": self._sandbox_workdir(runtime)})

    def _sandbox_workdir(self, runtime: SpatialClawSessionRuntime) -> str:
        return str(PurePosixPath(self.config.sandbox_workdir) / runtime.workspace.name)

    def _sandbox_files(self, runtime: SpatialClawSessionRuntime) -> dict[str, str | bytes]:
        context = runtime.context or {}
        metadata = context.get("metadata") or {}
        paths = [*context.get("images", []), *metadata.get("ref_images", [])]
        paths = list(dict.fromkeys(paths))
        target = PurePosixPath(self._sandbox_workdir(runtime))
        mapped = {path: str(target / "media" / f"{i:04d}-{Path(path).name}") for i, path in enumerate(paths)}
        files: dict[str, str | bytes] = {mapped[path]: Path(path).read_bytes() for path in paths}
        manifest = {
            "instruction": context.get("instruction", ""),
            "images": [mapped[path] for path in context.get("images", [])],
            "reference_images": [mapped[path] for path in metadata.get("ref_images", [])],
        }
        files[str(target / "task.json")] = json.dumps(manifest)
        if runtime.context is not None:
            runtime.context["sandbox_media"] = manifest
        return files

    async def release_session_state(self, session_id: str) -> None:
        if self._lent_sandboxes is not None:
            access = self._sandbox_accesses.get(session_id)
            sandbox = self._lent_sandboxes.get(session_id)
            if access is not None and sandbox is not None:
                # The child is derived from the resource session's SHA-256, not
                # from caller-supplied paths. Local preserves caller workdirs,
                # so remove only our staged child before stopping the sandbox.
                cleaned = await sandbox.exec(
                    "/bin/rm -rf -- " + quote(access.workdir),
                    cwd=self.config.sandbox_workdir,
                    timeout_s=self.config.runtime_close_timeout,
                )
                if cleaned.error_type is not None or cleaned.return_code != 0:
                    raise RuntimeError("Could not remove SpatialClaw's staged sandbox media")
            await self._lent_sandboxes.release(session_id)
        self._sandbox_accesses.pop(session_id, None)
        runtime = self._session_runtimes.get(session_id)
        if runtime is not None:
            await runtime.close(timeout=self.config.runtime_close_timeout, keep_workspace=self.config.keep_workspaces)
            self._session_runtimes.pop(session_id, None)

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest
        | Annotated[SpatialClawSeedRequest, BeforeValidator(reject_partial_typed_seed)],
    ) -> ResourcesSeedSessionResponse:
        response = await super().seed_session(body, request=request)
        if isinstance(response, ResourcesSeedSessionResponse):
            return response.model_copy(
                update={
                    "sandbox_access": self._sandbox_accesses.get(body.resources_session_id),
                    "observation": (
                        "Use the granted get_context tool to inspect this task's native media and Python environment. "
                        "Use execute for Python cells; submit the answer with ReturnAnswer(...)."
                    ),
                }
            )
        return response

    @staticmethod
    def _request_session_id(request: Request) -> str | None:
        return request.session.get(SESSION_ID_KEY)

    def _runtime(self, session_id: str | None) -> SpatialClawSessionRuntime:
        runtime = self._session_runtimes.get(session_id or "")
        if runtime is None:
            raise ResourcesSessionLostError("SpatialClaw task session is no longer available")
        if runtime.verified:
            raise ResourcesSessionLostError("SpatialClaw task session has already been verified")
        return runtime

    async def get_context(self, request: Request) -> dict[str, Any]:
        """Get the task's native instructions and initial images, without answers or grading data."""
        session_id = self._request_session_id(request)
        async with self.session_lock(session_id or ""):
            runtime = self._runtime(session_id)
            if runtime.context is None:
                raise ResourcesSessionLostError("SpatialClaw task initialization is incomplete")
            return context_result(runtime.context)

    async def execute(self, request: Request, body: ExecuteRequest) -> dict[str, Any]:
        """Execute one bounded Python cell in this task's persistent native environment."""
        session_id = self._request_session_id(request)
        async with self.session_lock(session_id or ""):
            runtime = self._runtime(session_id)
            return await runtime.execute(body.code, body.timeout)

    async def kernel(self, request: Request, body: KernelRequest) -> dict[str, Any]:
        """Private native-adapter kernel protocol; not advertised through MCP."""
        session_id = self._request_session_id(request)
        async with self.session_lock(session_id or ""):
            runtime = self._runtime(session_id)
            result = await runtime.rpc(
                body.operation, code=body.code, timeout=body.timeout, name=body.name, answer=body.answer
            )
            return {"result": result}

    def _load_benchmark_sync(self) -> Any:
        if self._benchmark_instance is not None:
            return self._benchmark_instance

        if self.config.spatialclaw_root:
            root = _validate_spatialclaw_checkout(Path(self.config.spatialclaw_root), self.config.spatialclaw_commit)
        else:
            cache_root = (
                Path(self.config.source_cache_root).expanduser().resolve()
                if self.config.source_cache_root
                else Path(__file__).parent / ".sources"
            )
            root = ensure_spatialclaw_checkout(self.config.spatialclaw_url, self.config.spatialclaw_commit, cache_root)
        _install_source_path(root)
        self._source_root = root

        from spatial_agent.config import SpatialAgentConfig, set_config
        from spatial_agent.evals.factory import BenchmarkFactory

        spatial_config = SpatialAgentConfig()
        spatial_config._load_from_envs()
        spatial_config.update_from_dataset_json(_config_path(root, self.config.dataset_config, "dataset"))
        set_config(spatial_config)
        data_root = (
            Path(self.config.data_root).expanduser().resolve() if self.config.data_root else (root / "data").resolve()
        )
        benchmark = BenchmarkFactory.create_benchmark(
            spatial_config.benchmark,
            data_root=str(data_root),
            question_type=spatial_config.question_type,
        )
        if benchmark is None:
            raise RuntimeError(f"SpatialClaw dataset config selected no benchmark: {self.config.dataset_config}")

        sample_by_id: dict[str, Any] = {}
        for sample in benchmark.data:
            sample_id = str(sample.sample_id)
            if sample_id in sample_by_id:
                raise RuntimeError(f"Duplicate SpatialClaw sample id: {sample_id}")
            sample_by_id[sample_id] = sample
        self._benchmark_instance = benchmark
        self._sample_by_id = sample_by_id
        return benchmark

    async def _load_benchmark(self) -> Any:
        if self._benchmark_instance is not None:
            return self._benchmark_instance
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()
        async with self._load_lock:
            return await asyncio.to_thread(self._load_benchmark_sync)

    async def verify(self, body: SpatialClawVerifyRequest, request: Request = None) -> SpatialClawVerifyResponse:
        # Stateless verification remains available for saved historical rollouts.
        session_id = self._request_session_id(request) if request is not None else None
        if session_id is not None:
            async with self.session_lock(session_id):
                runtime = self._session_runtimes.get(session_id)
                if runtime is None:
                    return SpatialClawVerifyResponse.model_validate(
                        body.model_dump()
                        | {
                            **lost_session_verification("SpatialClaw task session was lost before verification"),
                            "prediction": "",
                            "extracted_answer": None,
                            "native_score": None,
                            "scored": False,
                            "evaluation_completed": False,
                        }
                    )
                if runtime.sample_id != body.sample_id:
                    raise ResourcesSeedRejectedError("verify sample_id does not match the seeded task")
                runtime.verified = True
                journal = runtime.workspace / "gym-model-responses.jsonl"
                if journal.is_file() and journal.stat().st_size:
                    # The harness reports only its own provider calls. Resource VLM
                    # queries are captured separately, so its token total is partial.
                    response = body.response.model_copy(
                        update={
                            "usage": None,
                            "metadata": dict(body.response.metadata or {})
                            | {"spatialclaw_resource_usage_unknown": "true"},
                        }
                    )
                    body = body.model_copy(update={"response": response})
                step_limit = (body.response.metadata or {}).get("harness_step_limit_reached") == "true"
                prediction = runtime.accepted_answer or ""
                if step_limit and not self.config.score_step_limit_answer:
                    prediction = ""
                result = await self._score_prediction(body, prediction)
                if step_limit and not self.config.score_step_limit_answer:
                    result = result.model_copy(update={"reward": 0.0, "native_score": 0.0, "scored": True})
                return result
        return await self._score_prediction(body, (body.response.output_text or "").strip())

    async def _score_prediction(self, body: SpatialClawVerifyRequest, prediction: str) -> SpatialClawVerifyResponse:
        benchmark = await self._load_benchmark()
        sample = self._sample_by_id.get(str(body.sample_id))
        if sample is None:
            raise ValueError(f"Unknown SpatialClaw sample id: {body.sample_id}")

        score = benchmark.evaluate_single(sample, prediction)
        native_score = None if score is None else float(score)
        extracted = benchmark.extract_answer(prediction) or None
        return SpatialClawVerifyResponse.model_validate(
            body.model_dump()
            | {
                "reward": 0.0 if native_score is None else native_score,
                "prediction": prediction,
                "extracted_answer": extracted,
                "native_score": native_score,
                "scored": native_score is not None,
                "evaluation_completed": True,
                "step_limit_reached": (body.response.metadata or {}).get("harness_step_limit_reached") == "true",
            }
        )

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        benchmark = self._load_benchmark_sync()
        selected_samples = []
        predictions: dict[Any, str] = {}
        for rollouts in tasks:
            if len(rollouts) > 1 or any(rollout.get("_ng_rollout_index", 0) != 0 for rollout in rollouts):
                raise ValueError("SpatialClaw native metrics require num_repeats=1")
            if not rollouts:
                continue
            rollout = rollouts[0]
            if rollout.get("mask_sample"):
                continue
            sample = self._sample_by_id.get(str(rollout.get("sample_id", "")))
            if sample is None:
                continue
            predictions[sample.sample_id] = str(rollout.get("prediction", ""))

        original_data = benchmark.data
        dropped_groups = 0
        if type(benchmark).__name__ == "VideoMMEv2Bench":
            # Native v2 scores consecutive blocks of four. Gym filters masked rows
            # before this method, so reconstruct groups from the native dataset.
            for offset in range(0, len(original_data), 4):
                group = original_data[offset : offset + 4]
                if len(group) == 4 and all(sample.sample_id in predictions for sample in group):
                    selected_samples.extend(group)
                else:
                    dropped_groups += 1
        else:
            selected_samples = [sample for sample in original_data if sample.sample_id in predictions]
        benchmark.data = selected_samples
        try:
            native_results = benchmark.evaluate(predictions, output_dir=None) if selected_samples else {}
        finally:
            benchmark.data = original_data

        metrics, _, _, _ = compute_pass_majority_metrics(
            tasks,
            score_fn=lambda rollout: (
                {"native_score": float(rollout["native_score"])}
                if not rollout.get("mask_sample") and rollout.get("native_score") is not None
                else {}
            ),
            answer_key="extracted_answer",
        )
        metrics.update({f"spatialclaw/{key}": value for key, value in _flatten_numeric(native_results).items()})
        metrics["spatialclaw/num_evaluated_tasks"] = len(selected_samples)
        measured = [rollout for task in tasks for rollout in task if not rollout.get("mask_sample")]
        metrics["spatialclaw/step_limit_reached"] = (
            sum(bool(rollout.get("step_limit_reached")) for rollout in measured) / len(measured) if measured else 0.0
        )
        if type(benchmark).__name__ == "VideoMMEv2Bench":
            metrics["spatialclaw/num_evaluated_groups"] = len(selected_samples) // 4
            metrics["spatialclaw/num_dropped_groups"] = dropped_groups
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        key = {
            name: value
            for name, value in agent_metrics.items()
            if name
            in {
                "spatialclaw/overall_accuracy",
                "spatialclaw/overall_accuracy_pct",
                "spatialclaw/overall_score",
                "spatialclaw/overall_score_pct",
                "spatialclaw/final_rating/total",
            }
        }
        key.update(
            highest_k_metrics(
                agent_metrics,
                "pass@1[avg-of-{k}]",
                score_names=["native_score"],
            )
        )
        return key


if __name__ == "__main__":
    SpatialClawResourcesServer.run_webserver()
