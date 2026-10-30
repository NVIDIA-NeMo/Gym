# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grading controller. Three single-use sandboxes per job through Gym's sandbox API.
Comparator is sole verdict authority.

Job order: bounds check, reference run, comparator preflight of every reference outcome against stored expectations,
then a candidate run judged by the same comparator, then stop every sandbox. The candidate sandbox receives only the
candidate source, the normalized inputs, the public runner package and the operator's ``sandbox_spec.files``. A result
file is read only behind a process exit the provider independently reports, is shape-checked without decoding any
value, and must carry the paired status and the nonce this host wrote. None of these checks authenticates a result.
Before a candidate runs, a reference or comparator fault is nonterminal uncertainty, not fault attribution. After a
completed preflight, a candidate source error, suite timeout, exception or encoding-error observation is a
candidate-attributable fault scored 0 with the category preserved. When a reference cache is configured, the
reference-and-preflight outcome is cached per task, so a task's reference runs once per process. Importing this module
executes nothing remote and no task content.
"""

import asyncio
import hashlib
import json
import logging
import re
import secrets
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym import failure_kinds
from nemo_gym.sandbox import SandboxProvider, SandboxSpec

from .reference_cache import CachedReference, ReferenceCache, provider_class_name
from .runner import (
    CLEANUP_BUDGET_S,
    COMPARE_MODE,
    COMPARE_STATUSES,
    EXIT_COMPLETED,
    EXIT_RUNNER_ERROR,
    EXIT_SOURCE_ERROR,
    EXIT_TIMEOUT,
    MAX_CODE_LENGTH,
    MAX_NONCE_LENGTH,
    PROTOCOL_VERSION,
    RESULT_BYTES_LIMIT,
    RUN_MODE,
    RUN_STATUSES,
    RUNNER_ERROR_CODES,
    RUNTIME_PACKAGE,
    SOURCE_ERROR_CODES,
    VERDICT_STATUSES,
)
from .sandbox_domain import (
    CLEANUP_STATUSES,
    SAFE_PATH,
    SIZE_TEXT,
    TRANSFER_GRACE_S,
    CandidateWorkerAbort,
    SandboxDomain,
    Unscorable,
    reported_exit,
)
from .task_data import MAX_CASES, MAX_TEXT_BYTES, WIRE_FORMAT, TaskData, bounded_json


logger = logging.getLogger(__name__)
PACKAGE_DIR = Path(__file__).resolve().parent
RUNTIME_FILES = ("task_data.py", "codec.py", "symbolic.py", "runner.py")
COMPARATOR_FILES = RUNTIME_FILES + ("comparator.py",)
DOMAINS = ("reference", "comparator", "candidate")
_SAFE_PATH = SAFE_PATH
_Unscorable = Unscorable  # the gate functions below and the tests use these private aliases
_reported_exit = reported_exit
# Bounded transfers one job awaits: each run domain uploads runtime files, source and job and downloads one result
# (plus one size probe); the comparator uploads its files once, then uploads a job and downloads a result twice.
_JOB_TRANSFERS = 2 * (len(RUNTIME_FILES) + 4) + (len(COMPARATOR_FILES) + 6)
_ENVELOPE_KEYS = frozenset({"version", "mode", "nonce", "status", "code", "timed_out_case"})
_WIRE_KEYS = frozenset({"format", "value"})
_VERDICT_KEYS = frozenset({"version", "status", "equal", "side", "code", "path"})
_VERDICT_VERSION = 1
_MAX_VERDICT_LABEL = 256
_MAX_VERDICT_PATH = 1024
_MAX_UID = 2**31 - 1
_STATUS_EXIT_CODES = {
    "completed": EXIT_COMPLETED,
    "source_error": EXIT_SOURCE_ERROR,
    "runner_error": EXIT_RUNNER_ERROR,
    "timeout": EXIT_TIMEOUT,
}

PASSED = "passed"
CANDIDATE_FAULT_CATEGORIES = {
    "candidate_source_error": "candidate process reported a source error after a completed preflight",
    "candidate_timeout": "candidate process exceeded the suite timeout after a completed preflight",
    "candidate_exception": "candidate produced an exception observation after a completed preflight",
    "candidate_encoding_error": "candidate produced an unencodable observation after a completed preflight",
    "candidate_worker_abort": "candidate worker aborted or broke the runner protocol after a completed preflight",
}
FAILED_CATEGORIES = {
    "candidate_mismatch": "comparator found a mismatch for an observed value",
    "candidate_source_limit": "candidate source violated the host size or encoding contract",
    "candidate_invalid_output": "comparator validated an observed value as invalid",
    **CANDIDATE_FAULT_CATEGORIES,
}
UNSCORABLE_CATEGORIES = {
    # category: (failure class, terminal)
    "task_invalid": (failure_kinds.VERIFIER_ERROR, True),
    "reference_mismatch": (failure_kinds.VERIFIER_ERROR, True),
    "exec_user_unsupported": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "reference_error": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "reference_timeout": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "comparator_uncertain": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "provider_create": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "provider_exec": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "transfer_limit": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "result_invalid": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "candidate_no_result": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "reference_no_result": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "comparator_no_result": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "job_deadline": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "server_draining": (failure_kinds.PROVIDER_UNAVAILABLE, False),
}
# The only refusals the reference cache stores: facts about the task itself. Any other unscorable category, terminal
# or not, describes the infrastructure or the configuration and is never cached.
CACHED_REFUSALS = frozenset({"task_invalid", "reference_mismatch"})


class ExecutionPolicy(BaseModel):
    """Operator-owned runtime policy. Tasks cannot select any field. Unknown fields are rejected.

    Provider connection, image and resources live in the server's ``sandbox_provider``/``sandbox_spec`` config, not
    here. ``exec_user`` names the non-root identity every runner command runs as (``None`` keeps the provider's
    default identity, which the ``local`` provider requires). The runner itself still refuses euid 0.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    interpreter: str = "/usr/local/bin/python3"
    workdir: str = "/work"
    exec_user: str | int | None = None
    memory_limit_mib: int = Field(default=1536, ge=64, le=65536)
    cpu_time_limit_s: int = Field(default=1800, ge=1, le=86400)
    max_processes: int = Field(default=64, ge=1, le=4096)
    create_timeout_s: float = Field(default=900.0, gt=0, le=7200)
    transfer_timeout_s: float = Field(default=60.0, gt=0, le=3600)
    exec_timeout_margin_s: float = Field(default=20.0, gt=0, le=3600)
    suite_timeout_s: int = Field(default=1800, ge=1, le=86400, description="Whole-suite deadline per run.")
    compare_per_case_s: int = Field(default=20, ge=1, le=3600, description="Wall budget for one comparison case.")
    compare_timeout_s: int = Field(default=600, ge=1, le=86400, description="Cap for one whole compare batch.")
    job_timeout_s: float | None = Field(default=None, gt=0, le=7 * 86400)
    cleanup_timeout_s: float = Field(default=60.0, gt=0, le=3600)
    max_candidate_source_bytes: int = Field(default=65536, ge=1, le=1_048_576)
    max_concurrent_jobs: int = Field(default=4, ge=1, le=256)

    @model_validator(mode="after")
    def bounded_policy(self) -> "ExecutionPolicy":
        if self.exec_user in ("root", 0):
            raise ValueError("exec_user must not be root")
        if isinstance(self.exec_user, str) and not re.fullmatch(r"[a-z_][a-z0-9_-]*", self.exec_user):
            raise ValueError("exec_user must be a plain account name or a uid")
        if isinstance(self.exec_user, int) and not 1 <= self.exec_user <= _MAX_UID:
            raise ValueError(f"exec_user must be a uid from 1 to {_MAX_UID}")
        for name in ("interpreter", "workdir"):
            if not _SAFE_PATH.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be an absolute path without quoting characters")
        if self.cpu_time_limit_s > self.suite_timeout_s + self.exec_timeout_margin_s:
            raise ValueError("cpu_time_limit_s must not exceed the suite deadline plus margin")
        if self.exec_timeout_margin_s < CLEANUP_BUDGET_S:
            raise ValueError(
                f"exec_timeout_margin_s must be at least the runner cleanup budget ({CLEANUP_BUDGET_S:.0f} s)"
            )
        minimum = self.minimum_job_timeout_s()
        if self.job_timeout_s is not None and self.job_timeout_s < minimum:
            raise ValueError(f"job_timeout_s must be at least {minimum:.0f}")
        return self

    def compare_deadline_s(self, cases: int) -> int:
        """One compare batch: a per-case budget per case plus 30 s of slack, never above ``compare_timeout_s``."""
        return min(self.compare_timeout_s, cases * self.compare_per_case_s + 30)

    def minimum_job_timeout_s(self) -> float:
        """Sum of every bound awaited inside one job, so a legal but slow job never trips ``job_deadline``."""
        margin = 2 * self.exec_timeout_margin_s
        return (
            2 * (self.suite_timeout_s + margin)
            + 2 * (self.compare_timeout_s + margin)
            + 3 * self.create_timeout_s
            + _JOB_TRANSFERS * (self.transfer_timeout_s + TRANSFER_GRACE_S)
            + 3 * self.cleanup_timeout_s
        )

    def effective_job_timeout_s(self) -> float:
        return self.job_timeout_s if self.job_timeout_s is not None else self.minimum_job_timeout_s()


@dataclass(frozen=True)
class GradeResult:
    """Public verdict metadata only: no outputs, expectations, source or provider text."""

    outcome: Literal["passed", "failed", "unscorable"]
    reward: float | None
    category: str
    failure_class: str | None
    terminal: bool | None
    job_id: str
    case_count: int
    cases_attempted: int
    cases_equal: int
    first_failed_case: int | None
    cleanup: dict[str, str]
    sandboxes: dict[str, dict[str, Any]] = field(default_factory=dict)


def _bounded_text(value: Any, limit: int) -> bool:
    return type(value) is str and len(value) <= limit


def _check_outcome(item: Any) -> bool:
    """One run outcome in one of the runner's three shapes. A value is a single accepted wire envelope, undecoded."""
    if type(item) is not dict or type(item.get("kind")) is not str:
        return False
    kind = item["kind"]
    if kind == "value":
        wire = item.get("value")
        if set(item) != {"kind", "value"} or type(wire) is not dict or set(wire) != _WIRE_KEYS:
            return False
        if wire["format"] != WIRE_FORMAT or type(wire["value"]) is not list:
            return False
        try:
            bounded_json(wire)  # the byte, node and depth limits every accepted task wire value already meets
        except (ValueError, UnicodeError, RecursionError, TypeError):
            return False
        return True
    if kind == "exception":
        names = set(item) - {"kind"}
        return names <= {"type", "message"} and all(
            item[name] is None or _bounded_text(item[name], MAX_TEXT_BYTES) for name in names
        )
    if kind == "encoding_error":
        return set(item) == {"kind", "code"} and _bounded_text(item["code"], MAX_CODE_LENGTH)
    return False


def _check_verdict(item: Any) -> bool:
    """One comparator verdict exactly as ``comparator._result`` shapes it. ``equal`` must agree with ``status``."""
    if type(item) is not dict or set(item) != _VERDICT_KEYS:
        return False
    status = item["status"]
    if type(item["version"]) is not int or item["version"] != _VERDICT_VERSION:
        return False
    if type(status) is not str or status not in VERDICT_STATUSES:
        return False
    if item["equal"] is not (True if status == "equal" else False if status == "mismatch" else None):
        return False
    if not all(item[name] is None or _bounded_text(item[name], _MAX_VERDICT_LABEL) for name in ("side", "code")):
        return False
    return _bounded_text(item["path"], _MAX_VERDICT_PATH)


def _validate_verdict_batch(results: list[dict], role: str, attempted: int = 0) -> None:
    """Check every role/status pair before disposition, without interpreting values or diagnostic codes.

    These sides mirror ``comparator.compare_request``'s ``_result`` calls. Expectation defects use ``expected``.
    Uncertainty can concern the observed role, the comparator or the request, but never an unrelated role.
    """
    sides = {
        "equal": {role},
        f"invalid_{role}": {role},
        "invalid_expected": {"expected"},
        "uncertain": {role, "comparator", "request"},
    }
    if role == "candidate":
        sides["mismatch"] = {role}
    if any(verdict["side"] not in sides.get(verdict["status"], ()) for verdict in results):
        raise _Unscorable("result_invalid", attempted)


def _parse_result(data: bytes, mode: str) -> dict:
    """Shape-validate an untrusted result file without decoding any value. Contents stay untrusted observations.

    Bounds apply in cost order: total bytes, then the fixed envelope and item count, then each item on its own. A
    ``value`` outcome carries one wire value held to the limits an accepted task wire value must meet, so no
    single item can exceed that budget. A malformed file is ``result_invalid`` and attributes nothing.
    """
    if type(data) is not bytes or len(data) > RESULT_BYTES_LIMIT:
        raise _Unscorable("result_invalid")
    try:
        result = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError, TypeError):
        raise _Unscorable("result_invalid") from None
    key = "outcomes" if mode == RUN_MODE else "results"
    if type(result) is not dict or set(result) != _ENVELOPE_KEYS | {key}:
        raise _Unscorable("result_invalid")
    status, code, timed_out, items = result["status"], result["code"], result["timed_out_case"], result[key]
    statuses = RUN_STATUSES if mode == RUN_MODE else COMPARE_STATUSES
    codes = (
        SOURCE_ERROR_CODES if status == "source_error" else RUNNER_ERROR_CODES if status == "runner_error" else None
    )
    code_ok = code is None if codes is None else _bounded_text(code, MAX_CODE_LENGTH) and code in codes
    timed_out_ok = timed_out is None or (status == "timeout" and type(timed_out) is int and 0 <= timed_out < MAX_CASES)
    if (
        type(result["version"]) is not int
        or result["version"] != PROTOCOL_VERSION
        or result["mode"] != mode
        or not _bounded_text(result["nonce"], MAX_NONCE_LENGTH)
        or type(status) is not str
        or status not in statuses
        or not code_ok
        or not timed_out_ok
        or type(items) is not list
        or len(items) > MAX_CASES
    ):
        raise _Unscorable("result_invalid")
    check = _check_outcome if mode == RUN_MODE else _check_verdict
    if not all(check(item) for item in items):
        raise _Unscorable("result_invalid")
    return result


def _correlate_result(result: dict, exit_code: int, nonce: str) -> None:
    """Necessary cross-checks between a shape-valid result and what this host knows independently of the file.

    The provider-reported exit must be exactly the paired one, and the payload must echo this invocation's nonce.
    The nonce is correlation only, never authentication, since any same-UID code in the domain can read the job
    file. A mismatch is ``result_invalid`` and attributes nothing.
    """
    paired = _STATUS_EXIT_CODES.get(result["status"])
    if type(exit_code) is not int or exit_code != paired or result["nonce"] != nonce:
        raise _Unscorable("result_invalid")


def _runtime_files(names: tuple[str, ...]) -> dict[str, bytes]:
    return {f"{RUNTIME_PACKAGE}/{name}": (PACKAGE_DIR / name).read_bytes() for name in names}


def _limits(policy: ExecutionPolicy, deadline_s: int, per_case_s: int | None = None) -> dict[str, int]:
    limits = {
        "suite_deadline_s": deadline_s,
        "memory_mib": policy.memory_limit_mib,
        "cpu_time_s": min(policy.cpu_time_limit_s, deadline_s + int(policy.exec_timeout_margin_s)),
        "processes": policy.max_processes,
    }
    if per_case_s is not None:
        limits["per_case_s"] = per_case_s
    return limits


def _dump(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def build_run_job(
    task: TaskData, entrypoint: str, nonce: str, policy: ExecutionPolicy, *, workdir: str | None = None
) -> dict:
    """Normalized invocation inputs only: no expectations, tolerances or other cases' material."""
    return {
        "version": PROTOCOL_VERSION,
        "mode": RUN_MODE,
        "nonce": nonce,
        "entrypoint": entrypoint,
        "source_path": f"{workdir or policy.workdir}/source.py",
        # Always present as a list, possibly empty: the runner requires the key and applies the per-position
        # symbolic conversions before the call. None resolves to [], meaning no conversion for any position.
        "input_conversions": task.input_conversions or [],
        "cases": [
            {
                "args": [value.model_dump(mode="json") for value in case.args],
                "kwargs": {key: value.model_dump(mode="json") for key, value in case.kwargs.items()},
            }
            for case in task.test_cases
        ],
        "limits": _limits(policy, policy.suite_timeout_s),
    }


def build_compare_job(
    task: TaskData,
    role: str,
    outcomes: list[dict],
    nonce: str,
    policy: ExecutionPolicy,
    default_rtol: str | None = None,
    default_atol: str | None = None,
    per_case_s: int | None = None,
) -> dict:
    """Comparator input: trusted role, stored expectations and projected policy per case."""
    return {
        "version": PROTOCOL_VERSION,
        "mode": COMPARE_MODE,
        "nonce": nonce,
        "observed_role": role,
        "cases": [
            {
                "observed": outcomes[index],
                "expected": case.expected.model_dump(mode="json"),
                "policy": task.comparison_policy(index, default_rtol, default_atol),
            }
            for index, case in enumerate(task.test_cases[: len(outcomes)])
        ],
        "limits": _limits(policy, policy.compare_deadline_s(len(outcomes)), per_case_s),
    }


def validate_candidate_source(source: Any, policy: ExecutionPolicy) -> str | None:
    """Host-side contract check only. Never parses or executes. Returns a failed category or None."""
    if type(source) is not str or not source.strip() or "\x00" in source:
        return "candidate_source_limit"
    try:
        if len(source.encode("utf-8", errors="strict")) > policy.max_candidate_source_bytes:
            return "candidate_source_limit"
    except UnicodeEncodeError:
        return "candidate_source_limit"
    return None


def _runtime_digest(spec: SandboxSpec) -> str:
    """The runtime files, plus the operator's ``sandbox_spec.files`` (names and bytes) and ``sandbox_spec.env``.

    Every sandbox receives the operator's files and environment too, so they can change a reference outcome.
    """
    digest = hashlib.sha256()
    for name in COMPARATOR_FILES:
        digest.update(name.encode())
        digest.update((PACKAGE_DIR / name).read_bytes())
    operator = {
        "files": sorted([name, hashlib.sha256(text.encode("utf-8")).hexdigest()] for name, text in spec.files.items()),
        "env": sorted([name, value] for name, value in spec.env.items()),
    }
    digest.update(json.dumps(operator, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


async def _run_domain(domain: SandboxDomain, mode: str, deadline_s: int, nonce: str) -> dict:
    """Run the runner in one domain and apply the host gates: shape, status/exit pairing and nonce."""
    exit_code, data = await domain.run(mode, deadline_s)
    parsed = _parse_result(data, mode)
    _correlate_result(parsed, exit_code, nonce)
    return parsed


class Grader:
    """Grades jobs through three single-use sandboxes per job, ``max_concurrent_jobs`` at a time.

    One provider instance is shared by every domain; the server owns and closes it. Cleanup is per domain in a
    ``finally``, in parallel, under ``cleanup_timeout_s``; the spec's ``ttl_s`` and the metadata labels are the
    provider-side backstop. ``shutdown`` drains: new jobs, and running jobs at their next sandbox create, answer
    ``server_draining``, and live sandboxes are stopped.
    """

    def __init__(
        self,
        policy: ExecutionPolicy,
        *,
        provider: SandboxProvider,
        spec: SandboxSpec,
        default_rtol: str | None = None,
        default_atol: str | None = None,
        reference_cache: ReferenceCache | None = None,
    ):
        self.policy = policy
        self._provider = provider
        self._spec = spec
        self._default_rtol = default_rtol
        self._default_atol = default_atol
        self._cache = reference_cache
        self._runtime_digest = _runtime_digest(spec)
        self._provider_class = provider_class_name(provider)  # names the runtime in the cache key with the image
        self._semaphore = asyncio.Semaphore(policy.max_concurrent_jobs)
        self._draining = False
        self._live: set[SandboxDomain] = set()

    async def grade(self, task: TaskData, candidate_source: str, *, job_id: str | None = None) -> GradeResult:
        if not isinstance(task, TaskData):
            raise TypeError("task must be a validated TaskData")
        job_id = job_id if job_id is not None and re.fullmatch(r"[a-z0-9]{8,32}", job_id) else secrets.token_hex(8)
        counts = {"attempted": 0, "equal": 0, "first_failed": None}
        cleanup: dict[str, str] = {}
        sandboxes: dict[str, dict[str, Any]] = {}

        def unscorable(category: str) -> GradeResult:
            failure_class, terminal = UNSCORABLE_CATEGORIES[category]
            return GradeResult(
                "unscorable",
                None,
                category,
                failure_class,
                terminal,
                job_id,
                len(task.test_cases),
                counts["attempted"],
                counts["equal"],
                counts["first_failed"],
                cleanup,
                sandboxes,
            )

        contract = validate_candidate_source(candidate_source, self.policy)
        if contract is not None:
            return GradeResult(
                "failed", 0.0, contract, None, None, job_id, len(task.test_cases), 0, 0, None, cleanup, sandboxes
            )
        if self._draining:
            return unscorable("server_draining")
        async with self._semaphore:
            if self._draining:
                return unscorable("server_draining")
            domains = {
                name: SandboxDomain(name, self.policy, self._provider, self._spec, job_id=job_id) for name in DOMAINS
            }
            self._live.update(domains.values())
            try:
                try:
                    verdict = await asyncio.wait_for(
                        self._run_job(task, candidate_source, domains, counts, job_id),
                        timeout=self.policy.effective_job_timeout_s(),
                    )
                except asyncio.TimeoutError:
                    verdict = unscorable("job_deadline")
                except Unscorable as exc:
                    counts["attempted"] = max(counts["attempted"], exc.attempted)
                    verdict = unscorable(exc.category)
            finally:
                try:
                    await self._close_domains(domains.values(), cleanup)
                finally:
                    self._live.difference_update(domains.values())
                for name, domain in domains.items():
                    sandboxes[name] = domain.observation(_sandbox_outcome(domain))
        return GradeResult(**{**verdict.__dict__, "cleanup": cleanup, "sandboxes": sandboxes})

    async def _run_job(
        self, task: TaskData, source: str, domains: dict[str, SandboxDomain], counts: dict, job_id: str
    ) -> GradeResult:
        policy, cases = self.policy, len(task.test_cases)
        reference, comparator, candidate = domains["reference"], domains["comparator"], domains["candidate"]
        comparator_ready = False
        key, cached = None, None
        if self._cache is not None:
            key = ReferenceCache.key(
                task,
                policy,
                self._runtime_digest,
                self._spec.image,
                self._default_rtol,
                self._default_atol,
                provider_class=self._provider_class,
            )
            cached = self._cache.get(key)
            if cached is not None and cached.refusal is not None and cached.refusal not in CACHED_REFUSALS:
                cached = None  # a refusal this build does not hold terminal is a miss, never a verdict
        if cached is None:
            # 1. Reference run in its own domain.
            reference_nonce = secrets.token_hex(16)
            await self._create(reference)
            await reference.upload(_runtime_files(RUNTIME_FILES))
            await reference.upload({"source.py": task.reference_source.encode("utf-8")})
            await reference.upload(
                {
                    "job.json": _dump(
                        build_run_job(
                            task,
                            task.effective_reference_entrypoint,
                            reference_nonce,
                            policy,
                            workdir=reference.workdir,
                        )
                    )
                }
            )
            try:
                reference_result = await _run_domain(reference, RUN_MODE, policy.suite_timeout_s, reference_nonce)
                await reference.close()
                if reference_result["status"] == "timeout":
                    raise Unscorable("reference_timeout")
                if reference_result["status"] != "completed":
                    raise Unscorable("reference_error")
                if len(reference_result["outcomes"]) != cases:
                    raise Unscorable("result_invalid")
                for outcome, case in zip(reference_result["outcomes"], task.test_cases):
                    if outcome["kind"] == "value":
                        continue
                    if outcome["kind"] == "exception" and case.expected.kind == "exception":
                        continue
                    raise Unscorable("reference_error")
                # 2. Preflight: the comparator checks every reference value against stored expectations.
                await self._create(comparator)
                await comparator.upload(_runtime_files(COMPARATOR_FILES))
                comparator_ready = True
                preflight_nonce = secrets.token_hex(16)
                await comparator.upload(
                    {
                        "job.json": _dump(
                            build_compare_job(
                                task,
                                "reference",
                                reference_result["outcomes"],
                                preflight_nonce,
                                policy,
                                self._default_rtol,
                                self._default_atol,
                                per_case_s=policy.compare_per_case_s,
                            )
                        )
                    }
                )
                preflight = await _run_domain(
                    comparator, COMPARE_MODE, policy.compare_deadline_s(cases), preflight_nonce
                )
                if preflight["status"] != "completed":
                    raise Unscorable("comparator_uncertain")
                if len(preflight["results"]) != cases:
                    raise Unscorable("result_invalid")
                _validate_verdict_batch(preflight["results"], "reference")
                if any(verdict["status"] == "uncertain" for verdict in preflight["results"]):
                    raise Unscorable("comparator_uncertain")
                for verdict in preflight["results"]:
                    status = verdict["status"]
                    if status == "equal":
                        continue
                    if status == "invalid_reference":
                        raise Unscorable("reference_mismatch")
                    if status == "invalid_expected":
                        raise Unscorable("task_invalid")
                    raise Unscorable("result_invalid")
            except Unscorable as exc:
                if key is not None and exc.category in CACHED_REFUSALS:
                    self._remember(key, CachedReference(refusal=exc.category))  # a task-level refusal
                raise
            if key is not None:
                self._remember(key, CachedReference(refusal=None))
        elif cached.refusal is not None:
            raise Unscorable(cached.refusal)

        # 3. Candidate run, only after a complete successful preflight (live or cached).
        def failed(category: str, first: int | None) -> GradeResult:
            counts["first_failed"] = first
            return GradeResult(
                "failed", 0.0, category, None, None, job_id, cases, counts["attempted"], counts["equal"], first, {}
            )

        await self._create(candidate)
        await candidate.upload(_runtime_files(RUNTIME_FILES))
        await candidate.upload({"source.py": source.encode("utf-8")})
        candidate_nonce = secrets.token_hex(16)
        await candidate.upload(
            {
                "job.json": _dump(
                    build_run_job(task, task.entrypoint, candidate_nonce, policy, workdir=candidate.workdir)
                )
            }
        )
        try:
            candidate_result = await _run_domain(candidate, RUN_MODE, policy.suite_timeout_s, candidate_nonce)
        except CandidateWorkerAbort:
            return failed("candidate_worker_abort", None)
        await candidate.close()
        outcomes = candidate_result["outcomes"]
        counts["attempted"] = min(len(outcomes), cases)
        status = candidate_result["status"]
        if status == "source_error":
            return failed("candidate_source_error", None)
        if status == "timeout":
            return failed("candidate_timeout", None)
        if status != "completed":
            raise Unscorable("provider_exec", counts["attempted"])
        if len(outcomes) != cases:
            raise Unscorable("result_invalid", counts["attempted"])
        for index, outcome in enumerate(outcomes):
            if outcome["kind"] == "exception" and task.test_cases[index].expected.kind != "exception":
                return failed("candidate_exception", index)
            if outcome["kind"] == "encoding_error":
                return failed("candidate_encoding_error", index)

        # 4. Candidate value verdicts from the comparator domain. Expectations never left it.
        if not comparator_ready:
            await self._create(comparator)
            await comparator.upload(_runtime_files(COMPARATOR_FILES))
        comparison_nonce = secrets.token_hex(16)
        await comparator.upload(
            {
                "job.json": _dump(
                    build_compare_job(
                        task,
                        "candidate",
                        outcomes,
                        comparison_nonce,
                        policy,
                        self._default_rtol,
                        self._default_atol,
                        per_case_s=policy.compare_per_case_s,
                    )
                )
            }
        )
        judged = await _run_domain(comparator, COMPARE_MODE, policy.compare_deadline_s(cases), comparison_nonce)
        await comparator.close()
        if judged["status"] == "timeout":
            # The whole batch ran out of time after a clean preflight: candidate-attributable at the case the
            # runner timed out on (a per-case timeout inside one comparison is `uncertain`, see comparator._exhausted).
            timed_out_case = judged["timed_out_case"]
            return failed("candidate_invalid_output", timed_out_case if type(timed_out_case) is int else None)
        if judged["status"] != "completed":
            raise Unscorable("comparator_uncertain", cases)
        if len(judged["results"]) != cases:
            raise Unscorable("result_invalid", cases)
        _validate_verdict_batch(judged["results"], "candidate", cases)
        statuses = {verdict["status"] for verdict in judged["results"]}
        decided_failures = statuses & {"mismatch", "invalid_candidate"}
        # Precedence: an expectation defect, then the first decided failure (a mismatch or an invalid output), which
        # names the category and the failed case, then uncertainty on any case.
        if "invalid_expected" in statuses or ("uncertain" in statuses and not decided_failures):
            raise Unscorable("comparator_uncertain", cases)
        for index, verdict in enumerate(judged["results"]):
            status = verdict["status"]
            if status == "equal":
                counts["equal"] += 1
                continue
            if status == "uncertain":
                continue  # outranked by the decided failure on another case, and never counted equal
            if status == "mismatch":
                return failed("candidate_mismatch", index)
            if status == "invalid_candidate":
                return failed("candidate_invalid_output", index)
            if status == "invalid_reference":
                raise Unscorable("result_invalid", cases)
            raise Unscorable("comparator_uncertain", cases)
        return GradeResult("passed", 1.0, PASSED, None, None, job_id, cases, cases, counts["equal"], None, {})

    async def _create(self, domain: SandboxDomain) -> None:
        """Create a domain unless the grader is draining: ``shutdown`` stops only sandboxes that exist when it runs."""
        if self._draining:
            raise Unscorable("server_draining")
        await domain.create()

    def _remember(self, key: str, entry: CachedReference) -> None:
        """Best-effort cache write: a failing store is logged by class name and never replaces the job's outcome."""
        try:
            self._cache.put(key, entry)
        except OSError as exc:
            logger.debug("reference cache write failed (%s)", type(exc).__name__)

    async def _close_domains(self, domains: Iterable[SandboxDomain], statuses: dict[str, str]) -> None:
        """Stop every domain once, in parallel, each under its own bound. A cancellation is re-raised after all ran."""
        domains = list(domains)
        results = await asyncio.gather(*(domain.close() for domain in domains), return_exceptions=True)
        pending: BaseException | None = None
        for domain, outcome in zip(domains, results):
            statuses[domain.name] = domain.cleanup
            if isinstance(outcome, asyncio.CancelledError):
                pending = outcome
        if pending is not None:
            raise pending

    async def shutdown(self) -> None:
        """Drain: refuse new jobs, stop every live sandbox this process tracks; running jobs create no more."""
        self._draining = True
        live = list(self._live)
        if live:
            await asyncio.gather(*(domain.close() for domain in live), return_exceptions=True)


def _sandbox_outcome(domain: SandboxDomain) -> str:
    if domain.cleanup == "not_created":
        return "unknown"
    if domain.last_error_type == "timeout":
        return "timeout"
    if domain.last_error_type is not None:
        return "sandbox_error"
    if domain.last_exit is None:
        return "unknown"
    return "completed" if domain.last_exit in _STATUS_EXIT_CODES.values() else "failed"


# Two directories below the probe domain's workdir: the upload itself must create them, as a grading domain's runtime
# package upload must.
_PROBE_MARKER = "probe/nested/marker"
_PROBE_CONTENT = b"x"


class RuntimeProbeError(RuntimeError):
    """The configured provider, image or exec user cannot run the grader. Raised once at startup, never per row."""

    def __init__(self, category: str, detail: str = ""):
        super().__init__(category if not detail else f"{category}: {detail}")
        self.category = category


async def probe_runtime(provider: SandboxProvider, spec: SandboxSpec, policy: ExecutionPolicy) -> dict[str, str]:
    """Start one sandbox from the operator's spec and check the four facts every job relies on.

    (1) commands run as a non-root identity under ``exec_user``; (2) the provider accepts that identity;
    (3) the interpreter carries the pinned runtime packages in its own site-packages; (4) an upload to a nested path
    creates its parent directories, the domain's own size probe finds the file and prints plain digits, and the
    domain's bounded download returns the uploaded byte. Every exec is bounded on the host as a domain's are. The
    sandbox is stopped whatever the outcome. Categories are fixed vocabulary; provider text never leaves this function:
    a swallowed exception is logged by class name only and never chained.
    """
    domain = SandboxDomain("probe", policy, provider, spec, job_id="probe" + secrets.token_hex(4))
    try:
        try:
            await domain.create()
        except Unscorable as exc:
            raise RuntimeProbeError(exc.category) from None
        box = domain._box()

        async def run_bounded(command: str) -> Any:
            return await asyncio.wait_for(
                box.exec(command, cwd=domain.workdir, timeout_s=policy.transfer_timeout_s, user=policy.exec_user),
                timeout=policy.transfer_timeout_s + TRANSFER_GRACE_S,
            )

        try:
            uid = await run_bounded("id -u")
        except Exception as exc:
            logger.warning("runtime probe: id -u failed (%s)", type(exc).__name__)
            # Either class is how a provider declines a user= it cannot honour; with none requested it is no refusal.
            user_refused = policy.exec_user is not None and isinstance(exc, (ValueError, NotImplementedError))
            if user_refused:
                raise RuntimeProbeError("exec_user_unsupported", str(policy.exec_user)) from None
            raise RuntimeProbeError("provider_exec") from None
        uid_text = (uid.stdout or "").strip()
        if uid.error_type is not None or uid.return_code != 0 or not (uid_text.isascii() and uid_text.isdigit()):
            raise RuntimeProbeError("exec_user_unsupported", "id -u did not report a uid")
        if int(uid_text) == 0:
            raise RuntimeProbeError(
                "exec_user_is_root", "the runner refuses uid 0; set execution.exec_user to a non-root account"
            )
        try:
            imports = await run_bounded(f'"{policy.interpreter}" -I -B -c "import pydantic, sympy, numpy, scipy"')
        except Exception as exc:
            logger.warning("runtime probe: import check failed (%s)", type(exc).__name__)
            raise RuntimeProbeError("provider_exec") from None
        if imports.error_type is not None or imports.return_code != 0:
            raise RuntimeProbeError("interpreter_missing_dependency", policy.interpreter)
        marker = f"{domain.workdir}/{_PROBE_MARKER}"
        try:
            await domain.upload({_PROBE_MARKER: _PROBE_CONTENT})
        except Exception as exc:
            logger.warning("runtime probe: upload failed (%s)", type(exc).__name__)
            raise RuntimeProbeError("upload_unsupported", "uploads must create parent directories") from None
        try:
            size = await run_bounded(f'test -f "{marker}" && stat -c %s "{marker}"')  # the domain's own size probe
        except Exception as exc:
            logger.warning("runtime probe: size probe failed (%s)", type(exc).__name__)
            raise RuntimeProbeError("upload_unsupported", "the size probe did not run") from None
        if (
            size.error_type is not None
            or size.return_code != 0
            or not SIZE_TEXT.fullmatch((size.stdout or "").strip())
        ):
            raise RuntimeProbeError("upload_unsupported", "the size probe must find the upload and print plain digits")
        try:
            readback = await domain._download_bounded(marker)  # the bounded read every result file goes through
        except Exception as exc:
            logger.warning("runtime probe: download failed (%s)", type(exc).__name__)
            raise RuntimeProbeError("upload_unsupported", "the bounded download failed") from None
        if readback != _PROBE_CONTENT:
            raise RuntimeProbeError("upload_unsupported", "the bounded download returned other bytes")
        return {
            "uid": uid_text,
            "interpreter": policy.interpreter,
            "sandbox_id": domain.sandbox_id or "",
            "upload": "ok",
        }
    finally:
        await domain.close()


__all__ = [
    "CACHED_REFUSALS",
    "CANDIDATE_FAULT_CATEGORIES",
    "CLEANUP_STATUSES",
    "DOMAINS",
    "ExecutionPolicy",
    "FAILED_CATEGORIES",
    "GradeResult",
    "Grader",
    "PASSED",
    "RuntimeProbeError",
    "UNSCORABLE_CATEGORIES",
    "build_compare_job",
    "build_run_job",
    "probe_runtime",
    "validate_candidate_source",
]
