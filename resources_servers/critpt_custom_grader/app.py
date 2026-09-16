# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stateless HTTP adapter. Task execution and ownership remain in the remote grader.

The saved response is an opaque, bounded recovery artifact, not a diagnostic. No
request bodies, validation details, provider exceptions or task sources are logged.
"""

import asyncio
import re
from contextlib import asynccontextmanager
from typing import Any, ClassVar, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator
from starlette.exceptions import HTTPException

from nemo_gym import failure_kinds
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, BenchmarkDatasetConfig, DatasetConfig
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.task_data import normalize_task_fields
from resources_servers.critpt_custom_grader.execution import (
    CANDIDATE_FAULT_CATEGORIES,
    CLEANUP_STATUSES,
    UNSCORABLE_CATEGORIES,
    ExecutionPolicy,
    Grader,
    GradeResult,
    validate_candidate_source,
)
from resources_servers.critpt_custom_grader.server_support import (
    ServerRequestPolicy,
    ServerRequestPrivacy,
    private_error_response,
)
from resources_servers.critpt_custom_grader.task_data import (
    MAX_TASK_BYTES,
    TaskData,
    bounded_json,
    tolerance_text,
)


MAX_RESPONSE_BYTES = 1_048_576
MAX_REQUEST_BYTES = 2 * MAX_TASK_BYTES + 2 * MAX_RESPONSE_BYTES
_DOMAINS = frozenset({"reference", "comparator", "candidate"})
_RESOLVED = frozenset({"deleted", "absent", "not_created", "create_failed_clean"})
# Candidate-code extraction matches the official CritPt harness: first ```python fence, else first bare
# ``` fence, else the whole response text.
_PYTHON_FENCE = re.compile(r"```python[ \t]*\r?\n(.*?)\r?\n```", re.DOTALL)
_BARE_FENCE = re.compile(r"```[ \t]*\r?\n(.*?)\r?\n```", re.DOTALL)


# The `_ng_failure_class` values this server writes on a verify response.
#   verifier_unavailable: no verdict for a good saved response. Reverify-recoverable.
#   reference_failed: a terminal task or reference fault. Never recovered.
#   needs_regeneration: no usable source. NOT reverify-recoverable. A new candidate is needed.
_NG_FAILURE_CLASS_VERIFIER_UNAVAILABLE = "verifier_unavailable"
_NG_FAILURE_CLASS_REFERENCE_FAILED = "reference_failed"
_NG_FAILURE_CLASS_NEEDS_REGENERATION = "needs_regeneration"
# Categories the grader emits with `needs_regeneration`, not `verifier_unavailable`.
_NEEDS_REGENERATION_CATEGORIES = frozenset({"response_incomplete"})


def _error(category: str, status: int) -> JSONResponse:
    return private_error_response(category, status)


class _PrivateRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def private_handler(request: Request):
            try:
                return await handler(request)
            except RequestValidationError:
                # SimpleServer registers a body-printing handler AFTER setup_webserver.
                # Catch inside the route so that handler never receives private input.
                return _error("request_invalid", 422)
            except HTTPException:
                return _error("request_invalid", 400)
            except Exception:
                return _error("verifier_unavailable", 503)

        return private_handler


class _PrivateIngress:
    """Bound bytes before JSON/model validation and suppress outer exception details."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = False

        async def safe_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            buffer = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                if len(buffer) + len(chunk) > MAX_REQUEST_BYTES:
                    return await _error("request_limit", 413)(scope, receive, send)
                buffer.extend(chunk)
                if not message.get("more_body", False):
                    break
            body = bytes(buffer)
            del buffer
            delivered = False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            await self.app(scope, replay, safe_send)
        except asyncio.CancelledError:
            raise
        except Exception:
            if not started:
                await _error("verifier_unavailable", 503)(scope, receive, send)


class ComparisonDefaults(BaseModel):
    """Operator-owned default numeric tolerance for a genuinely silent leaf.

    ``default_rtol``/``default_atol`` apply only to a leaf with no statement promise, no task-level
    tolerance and no per-leaf tolerance, all of which keep precedence. These defaults do not reproduce
    the delivered verdicts, which used authored per-leaf tolerances absent from the delivered task data.
    Values are decimal strings validated at load. A parse failure or a negative value refuses startup.
    """

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


class CritPtCustomGraderConfig(BaseResourcesServerConfig):
    model_config = ConfigDict(extra="forbid", frozen=True)

    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    execution: ExecutionPolicy
    comparison: ComparisonDefaults = Field(default_factory=ComparisonDefaults)
    request_policy: ServerRequestPolicy
    request_privacy: ServerRequestPrivacy
    num_workers: Literal[1] = 1
    expose_tools_over_mcp: Literal[False] = False
    verified: bool = False
    description: str | None = None
    datasets: list[DatasetConfig | BenchmarkDatasetConfig] | None = None
    max_queued_jobs: int = Field(default=0, ge=0, le=32, strict=True)
    # safe_request_policy requires the deadline to exceed queue + job + cleanup.
    queue_timeout_s: float = Field(default=30.0, gt=0, le=7 * 86400, allow_inf_nan=False)

    @field_validator("request_privacy")
    @classmethod
    def private_requests_required(cls, value: ServerRequestPrivacy) -> ServerRequestPrivacy:
        if not value.private_requests:
            raise ValueError("request_privacy requires private_requests")
        return value

    @model_validator(mode="after")
    def safe_request_policy(self) -> "CritPtCustomGraderConfig":
        policy = self.request_policy
        minimum = self.queue_timeout_s + self.execution.effective_job_timeout_s()
        minimum += 6 * self.execution.cleanup_timeout_s
        if not policy.no_resubmission or policy.deadline_seconds is None or policy.deadline_seconds <= minimum:
            raise ValueError("request_policy requires no_resubmission and a deadline covering queue, job and cleanup")
        return self


class CritPtRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    responses_create_params: dict[str, Any]

    @field_validator("responses_create_params")
    @classmethod
    def public_params(cls, value: dict) -> dict:
        bounded_json(value, max_bytes=MAX_RESPONSE_BYTES)
        NeMoGymResponseCreateParamsNonStreaming.model_validate(value)
        if value.get("tools") or value.get("tool_choice") not in (None, "none"):
            raise ValueError("tools are not supported")
        return value


class CritPtVerifyRequest(CritPtRunRequest, BaseVerifyRequest):
    # Preserve the original JSON rather than a lossy model dump with replay defaults.
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
    cleanup_status: Literal["not_started", "resolved", "unresolved", "unknown"]
    failure_class: str | None = Field(default=None, alias="_ng_failure_class")
    failure_subcategory: str | None = Field(default=None, alias="_ng_failure_subcategory")
    failure_terminal: Literal[True] | None = Field(default=None, alias="_ng_failure_terminal")


def _task(body: CritPtRunRequest) -> TaskData:
    row = body.model_dump(exclude={"response", "capture_rollout_id"})
    fields, conflicts = normalize_task_fields(row)
    if conflicts:
        raise ValueError("conflicting task fields")
    # Normalize delivered numbers to tags BEFORE the restricted JSON boundary.
    return TaskData.model_validate(fields)


def _final_message_incomplete(response: dict) -> bool:
    """True when the last assistant message did not itself complete, even inside a completed envelope."""
    messages = [
        item
        for item in response.get("output", [])
        if item.get("type") == "message" and item.get("role") == "assistant"
    ]
    return bool(messages) and messages[-1].get("status") not in (None, "completed")


def _needs_regeneration(response: dict) -> bool:
    """True when the response did not complete, or carries an incomplete or error envelope, or ends on an
    incomplete assistant message.

    A completed response with none of these is a finished answer, so an empty source is a candidate fault."""
    return bool(
        response.get("status") != "completed"
        or response.get("incomplete_details")
        or response.get("error")
        or _final_message_incomplete(response)
    )


def _source(response: dict) -> str | None:
    """The candidate code carried by a completed response, or ``None`` when it has no usable source.

    Extraction matches the official CritPt harness: first ```python fence, else first bare ``` fence,
    else the whole response text stripped. Extra fences are tolerated."""
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
    if not cleanup and result.category in {"candidate_source_limit", "busy"}:
        return "not_started"
    if set(cleanup) != _DOMAINS:
        return "unknown"
    if any(type(status) is not str or status not in CLEANUP_STATUSES for status in cleanup.values()):
        return "unknown"
    return "resolved" if all(status in _RESOLVED for status in cleanup.values()) else "unresolved"


def _report_resolved(report: object) -> bool:
    """True when every reconcile-report entry reached a resolved cleanup state. An empty report is resolved."""
    if type(report) is not list:
        return False
    return all(type(item) is dict and item.get("status") in _RESOLVED for item in report)


class CritPtCustomGraderServer(SimpleResourcesServer):
    config: CritPtCustomGraderConfig
    _grader: Any = PrivateAttr(default=None)
    _owns_grader: bool = PrivateAttr(default=False)
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    _jobs: set = PrivateAttr(default_factory=set)
    _pending: int = PrivateAttr(default=0)
    _blocked: bool = PrivateAttr(default=False)
    _closing: bool = PrivateAttr(default=False)
    _shutdown_report: list[dict[str, str]] = PrivateAttr(default_factory=list)

    def setup_webserver(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app):
            # Reconcile the journal before the first request. A crash can leave an unresolved record on disk,
            # which would otherwise make a restart refuse every request.
            self._ensure_grader()
            await self.reconcile()
            try:
                yield
            finally:
                await self.shutdown()
                app.state.cleanup_report = self._shutdown_report
                # A fully resolved report clears the gate so an earlier transient latch does not fail a clean
                # shutdown. An unresolved report keeps the gate closed and raises.
                if _report_resolved(self._shutdown_report):
                    self._blocked = False
                if self._blocked:
                    raise RuntimeError("grader cleanup unresolved")

        # Do not install judge_failsafe or exception-recording trace wrappers here.
        app = FastAPI(lifespan=lifespan)
        app.router.route_class = _PrivateRoute
        app.add_middleware(_PrivateIngress)
        app.post("/seed_session", response_model=None)(self.seed_session)
        app.post("/verify", response_model_exclude_none=True)(self.verify)
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        app.get("/reverify_mode")(self.get_reverify_mode)
        return app

    def setup_exception_middleware(self, app: FastAPI) -> None:
        # _PrivateIngress is already installed. Core's formatter exposes exception text.
        pass

    def setup_telemetry(self) -> None:
        pass

    def instrument_app_for_telemetry(self, app: FastAPI) -> None:
        pass

    async def seed_session(self, body: CritPtRunRequest) -> BaseSeedSessionResponse | JSONResponse:
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
    ) -> CritPtVerifyResponse:
        if scored:
            failure_class = None
        elif terminal:
            failure_class = _NG_FAILURE_CLASS_REFERENCE_FAILED
        elif category in _NEEDS_REGENERATION_CATEGORIES:
            # No usable source. Reverify rejects it the same way every pass, so mark for regeneration.
            failure_class = _NG_FAILURE_CLASS_NEEDS_REGENERATION
        else:
            failure_class = _NG_FAILURE_CLASS_VERIFIER_UNAVAILABLE
        return CritPtVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            problem_id=task.problem_id if task else None,
            reward=reward,
            scored=scored,
            category=category,
            cleanup_status=cleanup,
            failure_class=failure_class,
            failure_subcategory=None if scored else category,
            failure_terminal=True if terminal else None,
        )

    def _project(self, body: CritPtVerifyRequest, task: TaskData, result: GradeResult) -> CritPtVerifyResponse:
        if not isinstance(result, GradeResult):
            self._blocked = True
            return self._response(body, task, "execution_unknown", cleanup="unknown")
        cleanup = _cleanup(result)
        if cleanup in {"unknown", "unresolved"}:
            self._blocked = True
            return self._response(body, task, "ownership_unresolved", cleanup=cleanup)
        counts_valid = (
            type(result.case_count) is int
            and type(result.cases_attempted) is int
            and type(result.cases_equal) is int
            and result.case_count == len(task.test_cases)
            and 0 <= result.cases_equal <= result.cases_attempted <= result.case_count
        )
        clean_verdict = result.failure_class is None and result.terminal is None and counts_valid
        all_ran = set(result.cleanup) == _DOMAINS and all(
            status in {"deleted", "absent"} for status in result.cleanup.values()
        )
        if (
            clean_verdict
            and all_ran
            and result.outcome == "passed"
            and result.category == "passed"
            and result.reward == 1.0
            and result.cases_equal == result.case_count
        ):
            return self._response(body, task, "passed", scored=True, reward=1.0, cleanup=cleanup)
        if (
            clean_verdict
            and result.outcome == "failed"
            and result.reward == 0.0
            and (
                (
                    all_ran
                    and result.cases_attempted == result.case_count
                    and result.cases_equal < result.case_count
                    and result.category in {"candidate_mismatch", "candidate_invalid_output"}
                )
                or result.category == "candidate_source_limit"
                # A whole-run candidate fault (source error or timeout) need not attempt every case, so no
                # per-case count is required here.
                or (all_ran and result.cases_equal == 0 and result.category in CANDIDATE_FAULT_CATEGORIES)
            )
        ):
            return self._response(body, task, result.category, scored=True, cleanup=cleanup)
        if (
            counts_valid
            and result.cases_attempted == 0
            and result.cases_equal == 0
            and result.outcome == "unscorable"
            and result.reward is None
            and result.failure_class == failure_kinds.VERIFIER_ERROR
            and result.terminal is True
            and result.category in {"task_invalid", "reference_mismatch"}
        ):
            return self._response(body, task, "reference_invalid", terminal=True, cleanup=cleanup)
        # Bounded diagnostics, not fault attribution.
        if (
            counts_valid
            and result.outcome == "unscorable"
            and result.reward is None
            and result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE
            and result.terminal is False
            and type(result.category) is str
            and UNSCORABLE_CATEGORIES.get(result.category) == (failure_kinds.PROVIDER_UNAVAILABLE, False)
        ):
            return self._response(body, task, result.category, cleanup=cleanup)
        return self._response(body, task, "execution_unknown", cleanup=cleanup)

    def _ensure_grader(self):
        """Build the process-owned grader once, or reuse an injected one. Bound to the disk journal directory."""
        if self._grader is None:
            self._grader = Grader(
                self.config.execution,
                default_rtol=self.config.comparison.default_rtol,
                default_atol=self.config.comparison.default_atol,
            )
            self._owns_grader = True
        return self._grader

    def _acquire_timeout(self, deadline: float | None) -> float:
        """The lock wait for one acquire. ``deadline`` shares one queue budget across the reconcile and the
        execute of a single latched-gate request, so their combined wait never exceeds one queue timeout."""
        if deadline is None:
            return self.config.queue_timeout_s
        return max(0.0, deadline - asyncio.get_running_loop().time())

    async def _execute(
        self, body: CritPtVerifyRequest, task: TaskData, source: str, deadline: float | None = None
    ) -> CritPtVerifyResponse:
        acquired = False
        try:
            try:
                await asyncio.wait_for(self._lock.acquire(), timeout=self._acquire_timeout(deadline))
                acquired = True
            except asyncio.TimeoutError:
                return self._response(body, task, "queue_timeout")
            if self._closing:
                return self._response(body, task, "ownership_unresolved", cleanup="unknown")
            if self._blocked:
                # The gate latched while this job waited on the lock. Reconcile once under the held lock.
                await self._reconcile_locked()
                if self._blocked:
                    return self._response(body, task, "ownership_unresolved", cleanup="unknown")
            result = await self._ensure_grader().grade(task, source)
            return self._project(body, task, result)
        except asyncio.CancelledError:
            # Cancellation is not evidence that ownership is unresolved: grade() runs its own cleanup. Latch the
            # gate only when a reconcile still reports an owned resource unresolved.
            if acquired and not await self._settled_after_cancel():
                self._blocked = True
            raise
        except Exception:
            if acquired:
                self._blocked = True
            return self._response(body, task, "execution_unknown", cleanup="unknown")
        finally:
            if acquired:
                self._lock.release()

    async def _settled_after_cancel(self) -> bool:
        """Reconcile whatever a cancelled job's cleanup could not finish. True when every owned domain resolves.

        Runs under the job lock the cancelled job still holds, so no other job races it."""
        if self._grader is None:
            return True
        try:
            report = await self._grader.reconcile()
        except asyncio.CancelledError:
            raise
        except Exception:
            return False
        return _report_resolved(report)

    async def _reconcile_locked(self) -> list[dict[str, str]]:
        """Retry cleanup of whatever a cancelled or failed job left unresolved. The caller must hold the job lock.

        Clear the gate only when every owned resource resolves."""
        if self._grader is None:
            return []
        try:
            report = await self._grader.reconcile()
        except asyncio.CancelledError:
            raise
        except Exception:
            return [{"status": "unknown"}]
        if _report_resolved(report):
            self._blocked = False
        return report

    async def reconcile(self, deadline: float | None = None) -> list[dict[str, str]]:
        """Acquire the job lock, then reconcile once. Return a busy report when the lock does not free in time.

        ``deadline`` bounds this acquire to the queue budget a latched-gate request already shares with its
        follow-on execute, so the request never waits the full queue timeout twice."""
        if self._grader is None or self._closing:
            return []
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=self._acquire_timeout(deadline))
        except asyncio.TimeoutError:
            return [{"status": "busy"}]
        try:
            return await self._reconcile_locked()
        finally:
            self._lock.release()

    def _job_done(self, job: asyncio.Task) -> None:
        self._jobs.discard(job)
        self._pending -= 1
        if not job.cancelled() and job.exception() is not None:
            self._blocked = True

    async def verify(self, body: CritPtVerifyRequest) -> CritPtVerifyResponse:
        try:
            task = _task(body)
        except Exception:
            return self._response(body, None, "task_invalid", terminal=True)
        source = _source(body.response)
        if source is None:
            if _needs_regeneration(body.response):
                # Reverify would reach the same verdict, so regenerate rather than retry the verifier.
                return self._response(body, task, "response_incomplete")
            # A completed response with no usable source is an empty submission, scored 0 as an invalid candidate
            # before any sandbox is created, not a regeneration request.
            return self._response(body, task, "candidate_source_limit", scored=True)
        try:
            contract = validate_candidate_source(source, self.config.execution)
        except UnicodeError:
            contract = "candidate_source_limit"
        if contract:
            return self._response(body, task, "candidate_source_limit", scored=True)
        if self._closing:
            return self._response(body, task, "ownership_unresolved", cleanup="unknown")
        deadline: float | None = None
        if self._blocked:
            # A transient cleanup fault latched the gate. Reconcile once before this request refuses. One queue
            # budget covers this reconcile and the execute below together, so the request never waits it twice.
            deadline = asyncio.get_running_loop().time() + self.config.queue_timeout_s
            await self.reconcile(deadline)
            if self._blocked:
                return self._response(body, task, "ownership_unresolved", cleanup="unknown")
        if self._pending >= 1 + self.config.max_queued_jobs:
            return self._response(body, task, "busy")
        self._pending += 1
        job = asyncio.create_task(self._execute(body, task, source, deadline))
        self._jobs.add(job)
        job.add_done_callback(self._job_done)
        try:
            return await asyncio.shield(job)
        except asyncio.CancelledError:
            # The tracked worker, not the HTTP caller, owns the gate through cleanup.
            job.cancel()
            raise

    async def shutdown(self) -> None:
        self._closing = True
        jobs = set(self._jobs)
        for job in jobs:
            job.cancel()
        if jobs:
            _, pending = await asyncio.wait(jobs, timeout=6 * self.config.execution.cleanup_timeout_s)
            if pending:
                self._blocked = True
                self._shutdown_report = [{"status": "busy"}]
                return
        if self._grader is None or not self._owns_grader:
            return
        try:
            report = await asyncio.wait_for(
                self._grader.shutdown(), timeout=6 * self.config.execution.cleanup_timeout_s
            )
            safe = []
            for item in report:
                status = item.get("status")
                safe.append({"status": status if status in CLEANUP_STATUSES else "unknown"})
            self._shutdown_report = safe
            if safe:
                self._blocked = self._blocked or any(item["status"] not in _RESOLVED for item in safe)
        except Exception:
            self._blocked = True
            self._shutdown_report = [{"status": "unknown"}]

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        rows = body.verify_responses
        scored = [
            row
            for row in rows
            if row.get("scored") is True
            and not row.get("_ng_failure_class")
            and type(row.get("reward")) in (int, float)
            and row["reward"] in (0, 1)
        ]
        # Feed only scoring fields into Gym's existing profiler, never arbitrary extras.
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
                "accounting_scope": "supplied_rows",
                "all_attempts_known": False,
            }
        )
        metrics.key_metrics = {
            key: value
            for key, value in metrics.agent_metrics.items()
            if key.startswith("supplied_") or key == "mean/reward"
        }
        return metrics


if __name__ == "__main__":
    CritPtCustomGraderServer.run_webserver()
