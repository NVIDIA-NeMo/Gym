# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import shlex
import tempfile
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import BaseRunServerInstanceConfig
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxCreateError,
    SandboxProvider,
    SandboxResources,
    SandboxSpec,
)


LOG = logging.getLogger(__name__)


class GraderDependencyError(RuntimeError):
    """A required grading dependency is missing, so grading must fail loud instead of skewing results."""


@dataclass
class SweTask:
    """A single SWE task to provision and/or verify."""

    instance_id: str
    image: str | None = None
    base_commit: str | None = None
    repo_workdir: str = "/testbed"
    test_command: str = ""
    test_framework: str = ""
    model_patch: str = ""
    test_patch: str = ""
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    benchmark: str = "swe-bench-ext"
    split: str = "test"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalArtifacts:
    """Raw evaluation output retrieved from the sandbox, before grading."""

    test_output: str = ""
    return_code: int = 0
    patch_applied: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class SweEvalReport:
    """Graded result of a single task. A non-None ``error_kind`` marks an infra failure that masks the sample."""

    instance_id: str
    resolved: bool = False
    patch_applied: bool = False
    patch_exists: bool = False
    error_kind: str | None = None
    tests_status: dict[str, Any] = field(default_factory=dict)


class SweTaskHarness(ABC):
    """Per-family provisioning + (server-private) grading recipe."""

    #: registry key, e.g. ``"swe-bench-ext"``.
    name: str = ""
    #: ``"flat-host-grade"`` (parse host-side) or ``"nested-harness"`` (in-container grader).
    grade_strategy: str = "flat-host-grade"

    @abstractmethod
    def build_spec(self, task: SweTask) -> SandboxSpec:
        """Build the sandbox spec for a task."""

    def supports_provider(self, provider_name: str) -> bool:
        return True

    def with_flat_eval(self) -> "SweTaskHarness":
        """Return a variant that grades host-side, a no-op for harnesses that already do."""
        return self

    async def materialize(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        """Upload the model patch and test patch into the started sandbox."""
        if task.model_patch:
            await env.write_text("/root/patch.diff", _ensure_trailing_newline(task.model_patch))
        if task.test_patch:
            await env.write_text("/root/test_patch.diff", _ensure_trailing_newline(task.test_patch))

    async def reset_repo(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        # never git clean, it deletes the image's prebuilt artifacts and breaks the tests
        if task.base_commit:
            await env.execute(f"git reset --hard {shlex.quote(task.base_commit)}", cwd=task.repo_workdir)

    @abstractmethod
    async def run_eval(self, env: "AsyncSweEnvironment", task: SweTask) -> EvalArtifacts:
        """Apply the patches and run the evaluation, returning raw artifacts."""

    @abstractmethod
    def grade(self, task: SweTask, artifacts: EvalArtifacts) -> SweEvalReport:
        """Parse raw artifacts host-side into a graded report."""


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def compute_resolved(
    *,
    fail_to_pass: Iterable[str],
    pass_to_pass: Iterable[str],
    passed: Iterable[str],
    eval_type: str = "pass_and_fail",
    status_map: dict[str, str] | None = None,
) -> bool:
    """Apply the SWE-bench resolution rule, mirroring swebench's per-repo pass_and_fail / fail_only grading."""
    required = list(fail_to_pass) + list(pass_to_pass)
    if not required:
        return False
    if eval_type == "fail_only":
        sm = status_map or {}
        # swebench's check_fail_only: a required test fails only when present AND explicitly FAILED.
        return all(not (test in sm and sm[test] == "FAILED") for test in required)
    if status_map is not None:
        # swebench check_pass_and_fail: absent or FAILED/ERROR fails, SKIPPED/XPASS are neutral
        return all(not (test not in status_map or status_map[test] in ("FAILED", "ERROR")) for test in required)
    passed_set = set(passed)
    return all(test in passed_set for test in required)


def reward_from_report(report: SweEvalReport) -> float:
    """Map a graded report to a reward, always a float. A set ``error_kind`` yields a masked 0.0."""
    if report.error_kind is not None:
        return 0.0
    return 1.0 if report.resolved else 0.0


class AsyncSweEnvironment:
    """Thin async wrapper around a started ``AsyncSandbox``, sandbox I/O only."""

    def __init__(self, sandbox: AsyncSandbox) -> None:
        self._sandbox = sandbox
        self._closed = False

    @classmethod
    async def start(
        cls,
        provider: Mapping[str, Any] | SandboxProvider,
        spec: SandboxSpec,
    ) -> "AsyncSweEnvironment":
        sandbox = AsyncSandbox(provider, spec)
        await sandbox.start()
        return cls(sandbox)

    @property
    def sandbox(self) -> AsyncSandbox:
        return self._sandbox

    @property
    def sandbox_id(self) -> str | None:
        handle = getattr(self._sandbox, "_handle", None)
        return handle.sandbox_id if handle is not None else None

    @property
    def provider_name(self) -> str | None:
        handle = getattr(self._sandbox, "_handle", None)
        return handle.provider_name if handle is not None else None

    async def execute(
        self,
        command: str,
        *,
        cwd: str | None = None,
        user: str | int | None = "root",
        timeout_s: int | float | None = None,
        is_eval: bool = False,
    ) -> dict[str, Any]:
        """Run a command and return a normalized result dict. ``is_eval`` is caller bookkeeping only."""
        result = await self._sandbox.exec(command, cwd=cwd, env=None, timeout_s=timeout_s, user=user)
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        output = "\n".join(part for part in (stdout, stderr) if part)
        return {
            "output": output,
            "returncode": result.return_code,
            "stdout": stdout,
            "stderr": stderr,
            "error_type": result.error_type,
        }

    async def upload(self, local_path: Path | str, remote_path: str) -> None:
        await self._sandbox.upload(local_path, remote_path)

    async def download(self, remote_path: str, local_path: Path | str) -> None:
        await self._sandbox.download(remote_path, local_path)

    async def write_text(self, remote_path: str, content: str) -> None:
        """Write a string to a file inside the sandbox via a temporary upload."""
        tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
        try:
            tmp.write(content)
            tmp.flush()
            tmp.close()
            await self._sandbox.upload(tmp.name, remote_path)
        finally:
            os.unlink(tmp.name)

    async def cleanup(self) -> None:
        """Stop the sandbox. Idempotent: subsequent calls are no-ops."""
        if self._closed:
            return
        self._closed = True
        await self._sandbox.stop()

    async def __aenter__(self) -> "AsyncSweEnvironment":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.cleanup()


@asynccontextmanager
async def acquire_sandbox(
    provider: Mapping[str, Any] | SandboxProvider,
    spec: SandboxSpec,
    *,
    instance_id: str = "",
) -> AsyncIterator[AsyncSweEnvironment]:
    """Start a fresh sandbox, yield it, and always stop it on exit. ``instance_id`` is for logging only."""
    env: AsyncSweEnvironment | None = None
    try:
        env = await AsyncSweEnvironment.start(provider, spec)
        yield env
    finally:
        if env is not None:
            try:
                await env.cleanup()
            except Exception:
                pass


# SWE-bench eval-log sentinels, kept here so grading never imports swebench.
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
APPLY_PATCH_PASS = ">>>>> Applied Patch"
RESET_FAILED = ">>>>> Reset Failed"
TESTS_ERROR = ">>>>> Tests Errored"
TESTS_TIMEOUT = ">>>>> Tests Timed Out"
START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"

# setup failed before tests could be trusted, forces empty status map and patch_applied=False
_BAD_CODES = (APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT)

# Pytest-style per-test status tokens at the start of a line. XFAIL counts as a pass.
_PASS_TOKENS = ("PASSED", "XFAIL")
_FAIL_TOKENS = ("FAILED", "ERROR")
_STATUS_TOKENS = _PASS_TOKENS + _FAIL_TOKENS + ("SKIPPED",)

EVAL_SCRIPT_PATH = "/root/eval.sh"
EVAL_LOG_PATH = "/root/eval_output.log"


def parse_eval_log(log: str) -> tuple[dict[str, str], bool]:
    """Parse an eval-script log host-side into ``(status_map, patch_applied)``."""
    if any(code in log for code in _BAD_CODES):
        return {}, False
    if START_TEST_OUTPUT not in log or END_TEST_OUTPUT not in log:
        return {}, False

    between = log.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]
    status_map = _parse_pytest_status_lines(between)
    if not status_map:
        # Some runners emit per-test lines outside the markers (e.g. to stderr).
        status_map = _parse_pytest_status_lines(log)
    return status_map, True


def _parse_pytest_status_lines(text: str) -> dict[str, str]:
    """Parse ``"<STATUS> <node_id>"`` pytest-style lines into a status map, last status wins."""
    status_map: dict[str, str] = {}
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        token = next((t for t in _STATUS_TOKENS if line.startswith(t)), None)
        if token is None:
            continue
        if token == "FAILED":
            # FAILED lines may read "FAILED <id> - <reason>", strip the separator.
            line = line.replace(" - ", " ")
        fields = line.split()
        if len(fields) <= 1:
            continue
        node_id = fields[1]
        status_map[node_id] = fields[0]
    return status_map


def passed_tests(status_map: dict[str, str]) -> list[str]:
    return [node for node, status in status_map.items() if status in _PASS_TOKENS]


async def flat_run_eval(env: "AsyncSweEnvironment", task: SweTask) -> EvalArtifacts:
    """Run ``task.metadata["eval_script"]`` in the sandbox and capture its log."""
    eval_script = task.metadata.get("eval_script", "")
    if not eval_script:
        return EvalArtifacts(
            test_output="",
            return_code=1,
            patch_applied=False,
            raw={"error_type": "eval_error", "flat": True},
        )

    await env.write_text(EVAL_SCRIPT_PATH, eval_script if eval_script.endswith("\n") else eval_script + "\n")
    result = await env.execute(
        f"bash {EVAL_SCRIPT_PATH} 2>&1 | tee {EVAL_LOG_PATH}; exit ${{PIPESTATUS[0]}}",
        cwd=task.repo_workdir,
        is_eval=True,
        # provider-independent default, exec defaults differ (docker 3600s vs apptainer 180s)
        timeout_s=task.metadata.get("tests_timeout", 1800),
    )
    log_text = result["output"]
    if not log_text.strip() and result.get("error_type") not in {"sandbox", "timeout"}:
        cat = await env.execute(f"cat {EVAL_LOG_PATH}", cwd=task.repo_workdir)
        if cat["returncode"] == 0:
            log_text = cat["output"]

    return EvalArtifacts(
        test_output=log_text,
        return_code=result["returncode"],
        patch_applied=bool(task.model_patch),
        raw={"error_type": result.get("error_type"), "flat": True},
    )


def flat_grade(task: SweTask, artifacts: EvalArtifacts) -> SweEvalReport:
    """Grade a flat eval log host-side, masking only sandbox/timeout infra failures via ``error_kind``."""
    # a missing eval spec is not masked, it grades unmasked resolved=False
    if artifacts.raw.get("error_type") in {"sandbox", "timeout"}:
        return SweEvalReport(
            instance_id=task.instance_id,
            patch_exists=bool(task.model_patch),
            patch_applied=artifacts.patch_applied,
            error_kind=artifacts.raw["error_type"],
        )

    status_map, log_patch_applied = parse_eval_log(artifacts.test_output)
    passed = passed_tests(status_map)
    # full status_map keeps SKIPPED/XPASS neutral, mirroring get_eval_tests_report
    resolved = log_patch_applied and compute_resolved(
        fail_to_pass=task.fail_to_pass,
        pass_to_pass=task.pass_to_pass,
        passed=passed,
        status_map=status_map,
    )
    return SweEvalReport(
        instance_id=task.instance_id,
        resolved=resolved,
        patch_applied=log_patch_applied,
        patch_exists=bool(task.model_patch),
        tests_status={"passed": passed, "all": status_map},
    )


def flat_eval_enabled(harness_flag: bool, task: SweTask) -> bool:
    """Return whether flat mode applies: the harness selects it or the task opts in via metadata."""
    return bool(harness_flag) or bool(task.metadata.get("flat_eval", False))


# Per-test status tokens swebench's repo parsers emit that count as a pass.
_SWEBENCH_PASS_STATUSES = frozenset({"PASSED", "XFAIL"})

_VALID_NAMES = frozenset({"swe-bench", "swe-bench-multilingual"})


class SweBenchHarness(SweTaskHarness):
    """SWE-bench (and multilingual) harness, one instance per family, host-side (flat) graded."""

    grade_strategy = "flat-host-grade"

    def __init__(self, name: str = "swe-bench") -> None:
        if name not in _VALID_NAMES:
            raise ValueError(f"Unknown swe-bench family: {name!r} (expected one of {sorted(_VALID_NAMES)})")
        self.name = name

    def build_spec(self, task: SweTask) -> SandboxSpec:
        return SandboxSpec(
            image=task.image,
            workdir=task.repo_workdir,
            ttl_s=task.metadata.get("ttl_s", 1800),
            ready_timeout_s=task.metadata.get("ready_timeout_s", 600),
            # no GIT_CONFIG_GLOBAL=/dev/null, older instance images' git cannot parse it
            env={"GIT_PAGER": "cat"},
            metadata={
                "instance_id": task.instance_id[:63],
                "benchmark": task.benchmark,
                "harness": self.name,
                **({"slurm_job_id": str(task.metadata["slurm_job_id"])} if task.metadata.get("slurm_job_id") else {}),
            },
            resources=SandboxResources.from_mapping(task.metadata.get("resources", {})),
            provider_options=dict(task.metadata.get("provider_options", {})),
        )

    async def materialize(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        """Write the bare ``/root/patch.diff`` the eval script applies."""
        if task.model_patch:
            await env.write_text("/root/patch.diff", _ensure_trailing_newline(task.model_patch))

    def _flat_eval_script(self, task: SweTask) -> str:
        """Build the official SWE-bench eval script, or ``""`` when it cannot be constructed."""
        instance = task.metadata.get("instance_dict")
        if not instance:
            return ""
        try:
            from swebench.harness.test_spec.test_spec import make_test_spec

            spec = make_test_spec(instance, namespace="swebench")
        except Exception:
            return ""
        # swebench GIT_APPLY_CMDS ladder, never --3way
        apply_model = (
            "cd /testbed && "
            "(git apply --verbose /root/patch.diff || "
            "git apply --verbose --reject /root/patch.diff || "
            "patch --batch --fuzz=5 -p1 -i /root/patch.diff || "
            "echo 'NEMO_GYM_PATCH_APPLY_FAILED')\n"
        )
        # -rA forces a result line per test, some eval commands omit it and parse to zero passes
        pytest_addopts = 'export PYTEST_ADDOPTS="-rA ${PYTEST_ADDOPTS:-}"\n'
        return pytest_addopts + apply_model + spec.eval_script

    async def run_eval(self, env: "AsyncSweEnvironment", task: SweTask) -> EvalArtifacts:
        """Run the instance's eval script in-sandbox and collect its log."""
        if not task.metadata.get("eval_script"):
            task = dataclasses.replace(task, metadata={**task.metadata, "eval_script": self._flat_eval_script(task)})
        return await flat_run_eval(env, task)

    def grade(self, task: SweTask, artifacts: EvalArtifacts) -> SweEvalReport:
        """Grade with swebench's per-repo parser, falling back to the generic parser when no spec exists."""
        report = self._swebench_flat_grade(task, artifacts)
        return report if report is not None else flat_grade(task, artifacts)

    def _swebench_flat_grade(self, task: SweTask, artifacts: EvalArtifacts) -> "SweEvalReport | None":
        """Grade with swebench's official per-repo log parser, or return None to fall back."""
        # the generic parser is pytest-only and mis-scores repos with other runners
        error_type = artifacts.raw.get("error_type")
        if error_type in {"sandbox", "timeout"}:
            return SweEvalReport(
                instance_id=task.instance_id,
                patch_exists=bool(task.model_patch),
                patch_applied=artifacts.patch_applied,
                error_kind=error_type,
            )
        instance = task.metadata.get("instance_dict")
        if not instance:
            return None
        try:
            from swebench.harness.constants import FAIL_ONLY_REPOS
            from swebench.harness.grading import get_logs_eval
            from swebench.harness.test_spec.test_spec import make_test_spec
        except Exception as exc:
            # fail loud rather than degrade to the pytest-only parser and skew resolve rates
            raise GraderDependencyError(
                "swebench is required to grade SWE-bench instances faithfully (per-repo log "
                "parsers) but could not be imported; install the pinned 'swebench==4.1.0'."
            ) from exc
        log_fp = None
        try:
            spec = make_test_spec(instance, namespace="swebench")
            with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as handle:
                handle.write(artifacts.test_output or "")
                log_fp = handle.name
            status_map, markers_found = get_logs_eval(spec, log_fp)
        except Exception:
            return None
        finally:
            if log_fp is not None and os.path.exists(log_fp):
                os.unlink(log_fp)
        passed = [node for node, status in status_map.items() if status in _SWEBENCH_PASS_STATUSES]
        # the JS multilingual repos use the fail-only rule, per swebench get_eval_report
        eval_type = "fail_only" if spec.repo in FAIL_ONLY_REPOS else "pass_and_fail"
        resolved = bool(markers_found) and compute_resolved(
            fail_to_pass=task.fail_to_pass,
            pass_to_pass=task.pass_to_pass,
            passed=passed,
            eval_type=eval_type,
            status_map=status_map,
        )
        return SweEvalReport(
            instance_id=task.instance_id,
            resolved=resolved,
            patch_applied=bool(markers_found),
            patch_exists=bool(task.model_patch),
            tests_status={"passed": passed, "all": status_map},
        )


class R2EGymHarness(SweBenchHarness):
    """R2E-Gym harness. Runs the instance eval script and grades strict PASSED-only, matching swe_agents."""

    def __init__(self) -> None:
        self.name = "r2e-gym"

    def _flat_eval_script(self, task: SweTask) -> str:
        instance = task.metadata.get("instance_dict") or {}
        eval_script = (
            instance.get("eval_script")
            or task.metadata.get("eval_script")
            or (
                "if [ -f /run_tests.sh ]; then bash /run_tests.sh; "
                "elif [ -f /testbed/run_tests.sh ]; then bash /testbed/run_tests.sh; "
                "elif [ -f /root/run_tests.sh ]; then bash /root/run_tests.sh; "
                "else echo 'R2E eval script not found'; exit 127; fi"
            )
        )
        apply_model = (
            "cd /testbed && "
            "(git apply --verbose /root/patch.diff || "
            "git apply --verbose --reject /root/patch.diff || "
            "patch --batch --fuzz=5 -p1 -i /root/patch.diff || "
            "echo 'NEMO_GYM_PATCH_APPLY_FAILED')\n"
        )
        return 'export PYTEST_ADDOPTS="-rA ${PYTEST_ADDOPTS:-}"\n' + apply_model + str(eval_script)

    async def run_eval(self, env: "AsyncSweEnvironment", task: SweTask) -> EvalArtifacts:
        task = dataclasses.replace(task, metadata={**task.metadata, "eval_script": self._flat_eval_script(task)})
        return await flat_run_eval(env, task)

    def grade(self, task: SweTask, artifacts: EvalArtifacts) -> SweEvalReport:
        error_type = artifacts.raw.get("error_type")
        if error_type in {"sandbox", "timeout"}:
            return SweEvalReport(
                instance_id=task.instance_id,
                patch_exists=bool(task.model_patch),
                patch_applied=artifacts.patch_applied,
                error_kind=error_type,
            )
        statuses: dict[str, str] = {}
        for raw_line in (artifacts.test_output or "").splitlines():
            fields = raw_line.strip().replace(" - ", " ").split()
            if len(fields) > 1 and fields[0] in ("PASSED", "FAILED", "ERROR", "XFAIL", "SKIPPED"):
                statuses[fields[1]] = fields[0]
        required = list(task.fail_to_pass) + list(task.pass_to_pass)
        resolved = bool(required) and all(statuses.get(test) == "PASSED" for test in required)
        return SweEvalReport(
            instance_id=task.instance_id,
            resolved=resolved,
            patch_exists=bool(task.model_patch),
            patch_applied="NEMO_GYM_PATCH_APPLY_FAILED" not in (artifacts.test_output or ""),
            tests_status={"passed": sorted(k for k, v in statuses.items() if v == "PASSED"), "all": statuses},
        )


_HARNESSES = {
    h.name: h for h in (SweBenchHarness("swe-bench"), SweBenchHarness("swe-bench-multilingual"), R2EGymHarness())
}

# HuggingFace dataset names don't match harness names, so map by substring (most specific first)
_HF_NAME_ALIASES = [
    ("SWE-bench_Multilingual", "swe-bench-multilingual"),
    ("R2E-Gym", "r2e-gym"),
    ("SWE-bench", "swe-bench"),
]


def get_harness(name: str) -> SweTaskHarness:
    if name in _HARNESSES:
        return _HARNESSES[name]
    for needle, key in _HF_NAME_ALIASES:
        if needle in name:
            return _HARNESSES[key]
    raise KeyError(f"Unknown SWE harness {name!r}. Registered: {', '.join(sorted(_HARNESSES))}")


class ProviderCapabilityError(RuntimeError):
    """Raised when a task's harness does not support the configured provider."""


def _provider_name(provider: Mapping[str, Any] | SandboxProvider) -> str:
    if isinstance(provider, Mapping):
        return next(iter(provider), "?")
    return getattr(provider, "name", "?")


async def verify_task(
    provider: Mapping[str, Any] | SandboxProvider,
    task: SweTask,
    *,
    run_golden: bool = False,
    eval_timeout_s: float | None = None,
) -> SweEvalReport:
    """Grade a task's patch in a fresh sandbox and return a report."""
    # infra failures mask via error_kind, other eval failures grade unmasked resolved=False
    harness = get_harness(task.benchmark)
    if task.metadata.get("flat_eval"):
        harness = harness.with_flat_eval()

    if run_golden:
        task = dataclasses.replace(task, model_patch=task.metadata.get("golden_patch", ""))

    if not (task.model_patch or "").strip():
        return SweEvalReport(instance_id=task.instance_id, patch_exists=False, resolved=False)

    provider_name = _provider_name(provider)
    if not harness.supports_provider(provider_name):
        raise ProviderCapabilityError(
            f"Harness {harness.name!r} does not support provider {provider_name!r} "
            f"(grade_strategy={harness.grade_strategy})"
        )

    spec = harness.build_spec(task)
    timeout = eval_timeout_s if eval_timeout_s is not None else float(task.metadata.get("eval_timeout_s", 1800))
    # propagate the eval budget, provider exec defaults differ (apptainer 180s vs docker 3600s)
    if "tests_timeout" not in task.metadata:
        task = dataclasses.replace(task, metadata={**task.metadata, "tests_timeout": timeout})
    # TTL lets the backend self-expire a sandbox orphaned by a hard crash
    if spec.ttl_s is None:
        spec = dataclasses.replace(spec, ttl_s=timeout + 600)

    try:
        async with acquire_sandbox(provider, spec, instance_id=task.instance_id) as env:

            async def _sequence() -> SweEvalReport:
                await harness.reset_repo(env, task)
                await harness.materialize(env, task)
                artifacts = await harness.run_eval(env, task)
                return harness.grade(task, artifacts)

            return await asyncio.wait_for(_sequence(), timeout=timeout)
    except GraderDependencyError:
        # Propagate so a misconfigured grader fails loudly instead of silently skewing the resolve rate.
        raise
    except (asyncio.TimeoutError, TimeoutError):
        # Genuine wall-clock eval timeout is masked, mirroring main's eval_timed_out behavior.
        return SweEvalReport(
            instance_id=task.instance_id,
            patch_exists=bool(task.model_patch),
            error_kind="eval_timeout",
            tests_status={"timeout_s": timeout},
        )
    except SandboxCreateError as exc:
        # mask provisioning failures so infra hiccups do not leak into the training signal
        return SweEvalReport(
            instance_id=task.instance_id,
            patch_exists=bool(task.model_patch),
            error_kind="sandbox",
            tests_status={"sandbox_create_error": repr(exc)},
        )
    except Exception as exc:
        # NOT masked: main keeps a non-timeout eval crash in the gradient at reward 0.
        return SweEvalReport(
            instance_id=task.instance_id,
            patch_exists=bool(task.model_patch),
            resolved=False,
            error_kind=None,
            tests_status={"exception": repr(exc)},
        )


def report_to_reward(report: SweEvalReport) -> float:
    return reward_from_report(report)


class AnySweVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, Any] = Field(default_factory=dict)


class AnySweConfig(BaseRunServerInstanceConfig):
    max_pass_to_pass: int = 20
    eval_timeout: int = 1800
    # single-key provider mapping for the fresh grading sandbox
    sandbox_provider: dict[str, Any] = Field(default_factory=dict)


class AnySweResourcesServer(SimpleResourcesServer):
    config: AnySweConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def verify(self, body: AnySweVerifyRequest) -> BaseVerifyResponse:
        vm = body.verifier_metadata or {}
        md = getattr(body.response, "metadata", None) or {}
        model_patch = md.get("model_patch")
        reward = 0.0
        if model_patch:
            rcp_md = getattr(body.responses_create_params, "metadata", None) or {}
            task = SweTask(
                instance_id=vm.get("instance_id") or "",
                image=rcp_md.get("docker_image"),
                model_patch=model_patch,
                test_patch=vm.get("test_patch") or "",
                fail_to_pass=list(vm.get("fail_to_pass") or []),
                pass_to_pass=list(vm.get("pass_to_pass") or [])[: self.config.max_pass_to_pass],
                benchmark=vm.get("benchmark") or "swe-bench",
                metadata={"instance_dict": vm.get("instance_dict") or {}},
            )
            report = await verify_task(
                self.config.sandbox_provider, task, eval_timeout_s=float(self.config.eval_timeout)
            )
            reward = 1.0 if report.resolved else 0.0
            LOG.info("verify instance=%s reward=%s", vm.get("instance_id"), reward)
        else:
            LOG.warning("no model_patch in response metadata for %s, reward=0", vm.get("instance_id"))
        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    AnySweResourcesServer.run_webserver()
