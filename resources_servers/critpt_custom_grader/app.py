# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stateless HTTP adapter over the sandbox-API grader.

Standard SimpleResourcesServer routes. ``/verify`` and ``/seed_session`` read the whole request body, refuse one larger
than MAX_REQUEST_BYTES, then parse and validate it inside the handler, so a malformed or oversized request answers a
category-only error and never reaches core's body-logging validation handler. These two handlers never log request
bodies, validation details, provider exceptions or task sources; the stock typed routes (``/close_session``,
``/aggregate_metrics``) keep core's behaviour.
"""

import json
import logging
import re
from contextlib import asynccontextmanager
from typing import Any, ClassVar

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest, BenchmarkDatasetConfig, DatasetConfig
from nemo_gym.config.global_config import (
    get_global_config_dict,  # same import as resources_servers/swebench_pro/app.py
)
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, failure_kinds
from nemo_gym.metrics import compute_aggregate_metrics
from nemo_gym.sandbox import (
    SandboxResources,
    SandboxSpec,
    create_provider,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedSessionResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.task_data import normalize_task_fields
from resources_servers.critpt_custom_grader.execution import (
    CLEANUP_STATUSES,
    FAILED_CATEGORIES,
    UNSCORABLE_CATEGORIES,
    ExecutionPolicy,
    Grader,
    GradeResult,
    RuntimeProbeError,
    probe_runtime,
    validate_candidate_source,
)
from resources_servers.critpt_custom_grader.reference_cache import ReferenceCache
from resources_servers.critpt_custom_grader.task_data import MAX_TASK_BYTES, TaskData, bounded_json, tolerance_text


logger = logging.getLogger(__name__)

MAX_RESPONSE_BYTES = 1_048_576
MAX_REQUEST_BYTES = 2 * MAX_TASK_BYTES + 2 * MAX_RESPONSE_BYTES
_DOMAINS = frozenset({"reference", "comparator", "candidate"})
_RESOLVED = frozenset({"deleted", "not_created"})
# Candidate-code extraction: first ```python fence, else first bare ``` fence, else the whole response text.
_PYTHON_FENCE = re.compile(r"```python[ \t]*\r?\n(.*?)\r?\n```", re.DOTALL)
_BARE_FENCE = re.compile(r"```[ \t]*\r?\n(.*?)\r?\n```", re.DOTALL)
# Framework-owned row keys that are never task data: the routing keys nemo_gym/task_materialization.py strips
# (agent_ref, task_source, skills_ref) and the keys rollout collection and reverify attach to a row. Every _ng_* key
# is dropped too.
_FRAMEWORK_ROW_KEYS = frozenset(
    {"agent_ref", "task_source", "skills_ref", "reuse_cached_deliverable", "deliverables_dir", "reference_ids"}
)
_NG_FAILURE_CLASS_RESPONSE_INCOMPLETE = "critpt_custom_grader:response_incomplete"
_NEEDS_REGENERATION_CATEGORIES = frozenset({"response_incomplete"})
# Verify-response fields written only when they carry a value.
_OMITTED_WHEN_UNSET = ("failure_class", "failure_subcategory", "failure_terminal")


def _error(category: str, status: int) -> JSONResponse:
    return JSONResponse(content={"error": {"category": category}}, status_code=status)


class ComparisonDefaults(BaseModel):
    """Operator-owned default numeric tolerance for a genuinely silent leaf (decimal strings, validated at load)."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    default_rtol: str = "5e-12"
    default_atol: str = "0"

    @field_validator("default_rtol", "default_atol", mode="before")
    @classmethod
    def nonnegative_decimal(cls, value: Any) -> str:
        text = tolerance_text(value)
        if text is None:
            raise ValueError("comparison default tolerance must be a finite nonnegative decimal")
        return text


class SandboxSpecConfig(BaseModel):
    """The SandboxSpec every grading sandbox is created from. The workdir must exist in the image.

    ``files`` maps an absolute path in the sandbox to operator-supplied text the sandbox API uploads at start.
    """

    model_config = ConfigDict(extra="forbid")

    image: str | None = None
    workdir: str = "/work"
    ready_timeout_s: float | None = 600.0
    ttl_s: float | None = 10400.0
    resources: dict[str, Any] = Field(default_factory=lambda: {"cpu": 1, "memory_mib": 4096})
    env: dict[str, str] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    provider_options: dict[str, Any] = Field(default_factory=dict)


class CritPtCustomGraderConfig(BaseResourcesServerConfig):
    # The loader validates the server's whole YAML block plus its name. This model declares its own keys and every key
    # Gym core reads from a resources-server block, so any other key is refused rather than ignored: a misspelt
    # sandbox_provider would otherwise fall back to the `sandbox` block. A validation error never echoes its input,
    # which can hold an inline provider mapping with an env-resolved credential.
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    sandbox_provider: str | dict[str, Any] = "sandbox"
    sandbox_spec: SandboxSpecConfig = Field(default_factory=SandboxSpecConfig)
    execution: ExecutionPolicy = Field(default_factory=ExecutionPolicy)
    comparison: ComparisonDefaults = Field(default_factory=ComparisonDefaults)
    reference_cache_dir: str | None = None
    startup_probe: bool = True
    verified: bool = False
    description: str | None = None
    value: str | None = None
    datasets: list[DatasetConfig | BenchmarkDatasetConfig] | None = None
    # Framework-owned keys Gym core reads from this block; the grader ignores them. The agent and model-adapter
    # pairing filters take any shape here (a bare-string ++ override is legal) and Gym validates them where it reads
    # them; verified_url is README/registry metadata and grading_mode an environment-manifest label.
    allowed_agents: Any = None
    allowed_model_types: Any = None
    verified_url: str | None = None
    grading_mode: str | None = None

    @model_validator(mode="after")
    def consistent_sandbox_bounds(self) -> "CritPtCustomGraderConfig":
        """Refuse to start on a policy that disagrees with the spec, or on bounds that cannot hold one job.

        Every domain works under ``sandbox_spec.workdir``; ``execution.workdir`` is a second copy that must agree.
        ``create_timeout_s`` bounds the provider's create and readiness wait plus two transfers: the spec's files and
        the workdir setup. The comparator sandbox lives for almost the whole job and its stop may follow the job
        deadline, so a set ``ttl_s`` (the provider-side backstop) must cover the job deadline plus one cleanup bound;
        a null ``ttl_s`` leaves the lifetime to the provider's default.
        """
        if self.execution.workdir != self.sandbox_spec.workdir:
            raise ValueError(
                f"execution.workdir ({self.execution.workdir!r}) must equal sandbox_spec.workdir "
                f"({self.sandbox_spec.workdir!r})"
            )
        needed = (self.sandbox_spec.ready_timeout_s or 0.0) + 2 * self.execution.transfer_timeout_s
        if self.execution.create_timeout_s < needed:
            raise ValueError(
                f"execution.create_timeout_s ({self.execution.create_timeout_s:g}) must be at least "
                f"sandbox_spec.ready_timeout_s + 2 * execution.transfer_timeout_s ({needed:g})"
            )
        ttl = self.sandbox_spec.ttl_s
        lifetime = self.execution.effective_job_timeout_s() + self.execution.cleanup_timeout_s
        if ttl is not None and ttl < lifetime:
            raise ValueError(
                f"sandbox_spec.ttl_s ({ttl:g}) must be at least the job deadline (execution.job_timeout_s, or its "
                f"derived minimum when null) + execution.cleanup_timeout_s ({lifetime:g})"
            )
        return self


class CritPtRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    responses_create_params: dict[str, Any]

    @field_validator("responses_create_params")
    @classmethod
    def public_params(cls, value: dict) -> dict:
        bounded_json(value, max_bytes=MAX_RESPONSE_BYTES)
        NeMoGymResponseCreateParamsNonStreaming.model_validate(value)
        if value.get("tools"):
            raise ValueError("tools are not supported")
        if value.get("tool_choice") not in (None, "auto", "none"):
            raise ValueError("tool_choice requires tools")
        return value


class CritPtVerifyRequest(CritPtRunRequest, BaseVerifyRequest):
    response: dict[str, Any]

    @field_validator("response")
    @classmethod
    def saved_response(cls, value: dict) -> dict:
        bounded_json(value, max_bytes=MAX_RESPONSE_BYTES)
        NeMoGymResponse.model_validate(value)
        return value


class CritPtVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    responses_create_params: dict[str, Any]
    response: dict[str, Any]
    problem_id: str | None = None
    scored: bool
    category: str
    cleanup_status: str
    sandbox_observations: list[dict[str, Any]] = Field(default_factory=list)
    failure_class: str | None = Field(default=None, alias="_ng_failure_class")
    failure_subcategory: str | None = Field(default=None, alias="_ng_failure_subcategory")
    failure_terminal: bool | None = Field(default=None, alias="_ng_failure_terminal")

    @model_serializer(mode="wrap")
    def _omit_unset_failure_keys(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """A scored row carries no ``_ng_failure_*`` keys and a nonterminal row no ``_ng_failure_terminal``."""
        data = handler(self)
        for name in _OMITTED_WHEN_UNSET:
            if getattr(self, name) is None:
                data.pop(name, None)
                data.pop(type(self).model_fields[name].alias, None)
        return data


def _task(body: CritPtRunRequest) -> TaskData:
    row = body.model_dump(exclude={"response", "capture_rollout_id"})
    row = {key: value for key, value in row.items() if not key.startswith("_ng_") and key not in _FRAMEWORK_ROW_KEYS}
    fields, conflicts = normalize_task_fields(row)
    if conflicts:
        raise ValueError("conflicting task fields")
    return TaskData.model_validate(fields)


def _final_message_incomplete(response: dict) -> bool:
    messages = [
        item
        for item in response.get("output", [])
        if item.get("type") == "message" and item.get("role") == "assistant"
    ]
    return bool(messages) and messages[-1].get("status") not in (None, "completed")


def _needs_regeneration(response: dict) -> bool:
    return bool(
        response.get("status") != "completed"
        or response.get("incomplete_details")
        or response.get("error")
        or _final_message_incomplete(response)
    )


def _source(response: dict) -> str | None:
    if _needs_regeneration(response):
        return None
    messages = [
        item for item in response["output"] if item.get("type") == "message" and item.get("role") == "assistant"
    ]
    if not messages:
        return None
    message = messages[-1]
    if message.get("status") not in (None, "completed"):
        return None
    content = message.get("content", [])
    if any(item.get("type") != "output_text" for item in content):
        return None
    text = "\n".join(item["text"] for item in content).strip()
    if not text:
        return None
    match = _PYTHON_FENCE.search(text) or _BARE_FENCE.search(text)
    return match.group(1).strip() if match else text


def _cleanup(result: GradeResult) -> str:
    cleanup = result.cleanup
    if type(cleanup) is not dict:
        return "unknown"
    if not cleanup:
        return "not_started"
    if set(cleanup) != _DOMAINS or any(
        type(status) is not str or status not in CLEANUP_STATUSES for status in cleanup.values()
    ):
        return "unknown"
    return "resolved" if all(status in _RESOLVED for status in cleanup.values()) else "unresolved"


async def _close_provider(provider: Any) -> None:
    """Close a provider's clients. A failure is logged by class name only: its message may carry provider text."""
    try:
        await provider.aclose()
    except Exception as exc:
        logger.warning("sandbox provider close failed (%s)", type(exc).__name__)


class CritPtCustomGraderServer(SimpleResourcesServer):
    ray_enabled = False  # grading runs in sandboxes through Gym's sandbox API; no Ray in the verify path

    config: CritPtCustomGraderConfig
    _grader: Any = PrivateAttr(default=None)
    _provider: Any = PrivateAttr(default=None)
    _probe_info: dict = PrivateAttr(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self.startup()
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        return app

    def _named_configs(self) -> dict | None:
        return get_global_config_dict() if isinstance(self.config.sandbox_provider, str) else None

    def _sandbox_spec(self) -> SandboxSpec:
        cfg = self.config.sandbox_spec
        metadata = {
            **resolve_provider_metadata(self.config.sandbox_provider, self._named_configs()),
            **cfg.metadata,
            "nemo_gym_server": (self.config.name or "critpt_custom_grader")[:63],
        }
        return SandboxSpec(
            image=cfg.image,
            ttl_s=cfg.ttl_s,
            ready_timeout_s=cfg.ready_timeout_s,
            workdir=cfg.workdir,
            env=dict(cfg.env),
            files=dict(cfg.files),
            metadata=metadata,
            resources=SandboxResources.from_mapping(cfg.resources),
            provider_options=dict(cfg.provider_options),
        )

    async def startup(self) -> None:
        """Resolve the provider once, probe the runtime, build the grader. Any probe failure refuses to serve.

        This server requires ``sandbox_spec.image`` on every provider but ``local``, even one that could start from a
        snapshot, a template or a default image: without it, startup refuses with ``image_missing`` before any sandbox
        starts, whether or not the probe runs. On any failure the provider is closed and the server holds neither
        provider nor grader.
        """
        if self._grader is not None:
            return
        provider = self._provider or create_provider(
            resolve_provider_config(self.config.sandbox_provider, self._named_configs())
        )
        try:
            spec = self._sandbox_spec()
            # No image is shipped. The local provider runs the host interpreter and ignores the image; it is told apart
            # by the name every provider declares. A provider's own fallback (a snapshot, a template, a default image)
            # does not stand in for it: the reference cache names the runtime by its image.
            if not spec.image and getattr(provider, "name", None) != "local":  # unset or empty
                raise RuntimeProbeError(
                    "image_missing",
                    "sandbox_spec.image is unset or empty; it is required on every provider but local (a provider's snapshot, template or "
                    "default image does not replace it): build runtime/Dockerfile and set CRITPT_CUSTOM_GRADER_IMAGE "
                    "or sandbox_spec.image",
                )
            if self.config.startup_probe:
                self._probe_info = await probe_runtime(provider, spec, self.config.execution)
            grader = Grader(
                self.config.execution,
                provider=provider,
                spec=spec,
                default_rtol=self.config.comparison.default_rtol,
                default_atol=self.config.comparison.default_atol,
                reference_cache=ReferenceCache(self.config.reference_cache_dir),
            )
        except BaseException:
            self._provider = None
            await _close_provider(provider)
            raise
        self._provider, self._grader = provider, grader

    async def shutdown(self) -> None:
        """Drain the grader, then close the provider even when the drain fails or is cancelled, then forget both.

        A second call is a no-op, and ``startup`` can run again afterwards.
        """
        grader, provider = self._grader, self._provider
        try:
            if grader is not None:
                await grader.shutdown()
        finally:
            try:
                if provider is not None:
                    await _close_provider(provider)
            finally:
                self._grader = self._provider = None
                self._probe_info = {}

    @staticmethod
    async def _read_json(request: Request) -> tuple[dict | None, JSONResponse | None]:
        body = await request.body()
        if len(body) > MAX_REQUEST_BYTES:
            return None, _error("request_limit", 413)
        try:
            value = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeError, RecursionError):
            return None, _error("request_invalid", 422)
        if type(value) is not dict:
            return None, _error("request_invalid", 422)
        return value, None

    async def seed_session(self, request: Request):  # type: ignore[override]
        value, error = await self._read_json(request)
        if error is not None:
            return error
        if "resources_session_id" in value:
            return ResourcesSeedSessionResponse(resources_session_id=str(value["resources_session_id"]))
        # The request shape (prompt params, tools, tool_choice) first, as /verify does; then the task fields.
        try:
            body = CritPtRunRequest.model_validate(value)
        except Exception:
            return _error("request_invalid", 422)
        try:
            _task(body)
        except Exception:
            return _error("task_invalid", 422)
        return BaseSeedSessionResponse()

    def _response(
        self,
        body: CritPtVerifyRequest,
        task: TaskData | None,
        category: str,
        *,
        scored: bool = False,
        reward: float = 0.0,
        terminal: bool = False,
        cleanup: str = "not_started",
        sandboxes: list[dict[str, Any]] | None = None,
    ) -> CritPtVerifyResponse:
        if scored:
            failure_class = None
        elif terminal:
            failure_class = failure_kinds.VERIFIER_ERROR
        elif category in _NEEDS_REGENERATION_CATEGORIES:
            failure_class = _NG_FAILURE_CLASS_RESPONSE_INCOMPLETE
        else:
            failure_class = failure_kinds.PROVIDER_UNAVAILABLE
        return CritPtVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            problem_id=task.problem_id if task else None,
            reward=reward,
            mask_sample=not scored,
            scored=scored,
            category=category,
            cleanup_status=cleanup,
            sandbox_observations=sandboxes or [],
            failure_class=failure_class,
            failure_kind=failure_class,
            failure_reason=None if scored else category,
            failure_subcategory=None if scored else category,
            failure_terminal=True if terminal else None,
        )

    def _project(self, body: CritPtVerifyRequest, task: TaskData, result: GradeResult) -> CritPtVerifyResponse:
        if not isinstance(result, GradeResult):
            return self._response(body, task, "execution_unknown", cleanup="unknown")
        cleanup = _cleanup(result)
        observations = list(result.sandboxes.values()) if type(result.sandboxes) is dict else []
        counts_valid = (
            type(result.case_count) is int
            and type(result.cases_attempted) is int
            and type(result.cases_equal) is int
            and result.case_count == len(task.test_cases)
            and 0 <= result.cases_equal <= result.cases_attempted <= result.case_count
        )
        clean = result.failure_class is None and result.terminal is None and counts_valid
        if (
            clean
            and result.outcome == "passed"
            and result.category == "passed"
            and result.reward == 1.0
            and result.cases_equal == result.case_count
        ):
            return self._response(
                body, task, "passed", scored=True, reward=1.0, cleanup=cleanup, sandboxes=observations
            )
        if clean and result.outcome == "failed" and result.reward == 0.0 and result.category in FAILED_CATEGORIES:
            return self._response(body, task, result.category, scored=True, cleanup=cleanup, sandboxes=observations)
        if result.outcome == "unscorable" and result.reward is None and result.category in UNSCORABLE_CATEGORIES:
            failure_class, terminal = UNSCORABLE_CATEGORIES[result.category]
            if result.failure_class == failure_class and result.terminal is terminal:
                return self._response(
                    body, task, result.category, terminal=terminal, cleanup=cleanup, sandboxes=observations
                )
        return self._response(body, task, "execution_unknown", cleanup="unknown", sandboxes=observations)

    async def verify(self, request: Request):  # type: ignore[override]
        value, error = await self._read_json(request)
        if error is not None:
            return error
        try:
            body = CritPtVerifyRequest.model_validate(value)
        except Exception:
            return _error("request_invalid", 422)
        try:
            task = _task(body)
        except Exception:
            return self._response(body, None, "task_invalid", terminal=True)
        source = _source(body.response)
        if source is None:
            if _needs_regeneration(body.response):
                return self._response(body, task, "response_incomplete")
            return self._response(body, task, "candidate_source_limit", scored=True)
        try:
            contract = validate_candidate_source(source, self.config.execution)
        except UnicodeError:
            contract = "candidate_source_limit"
        if contract:
            return self._response(body, task, "candidate_source_limit", scored=True)
        if self._grader is None:
            return _error("server_not_ready", 503)
        try:
            result = await self._grader.grade(task, source)
        except Exception as exc:
            # The class name only, as for a provider close failure: the message may carry provider or task text.
            logger.warning("grader call failed (%s)", type(exc).__name__)
            return self._response(body, task, "execution_unknown", cleanup="unknown")
        try:
            return self._project(body, task, result)
        except Exception as exc:
            # A grader result the projection cannot read is a grader fault: a category-only row, never a 500. The log
            # names the exception class only; its message could quote the result.
            logger.warning("verify projection failed (%s)", type(exc).__name__)
            return self._response(body, task, "execution_unknown", cleanup="unknown")

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        rows = body.verify_responses
        scored = [
            row
            for row in rows
            if row.get("scored") is True
            and not row.get("_ng_failure_class")
            and not row.get("mask_sample")
            and type(row.get("reward")) in (int, float)
            and row["reward"] in (0, 1)
        ]
        public_rows = [
            {
                "_ng_task_index": row.get("_ng_task_index", index),
                "_ng_rollout_index": row.get("_ng_rollout_index", 0),
                "reward": row["reward"],
            }
            for index, row in enumerate(scored)
        ]
        if any(
            type(row[key]) is not int or not 0 <= row[key] < 2**63
            for row in public_rows
            for key in ("_ng_task_index", "_ng_rollout_index")
        ):
            raise ValueError("invalid aggregation identity")
        metrics = compute_aggregate_metrics(public_rows)
        metrics.agent_metrics.update(
            {
                "supplied_attempted": len(rows),
                "supplied_scored": len(scored),
                "supplied_failed": sum(row["reward"] == 0 for row in scored),
                "supplied_unscorable": len(rows) - len(scored),
                "accuracy_scorable": (sum(row["reward"] == 1 for row in scored) / len(scored)) if scored else None,
                "accuracy_all": (sum(row["reward"] == 1 for row in scored) / len(rows)) if rows else None,
                "accounting_scope": "supplied_rows",
                "all_attempts_known": False,
            }
        )
        metrics.key_metrics = {
            key: value
            for key, value in metrics.agent_metrics.items()
            if key.startswith(("supplied_", "accuracy_")) or key == "mean/reward"
        }
        return metrics


if __name__ == "__main__":
    CritPtCustomGraderServer.run_webserver()
