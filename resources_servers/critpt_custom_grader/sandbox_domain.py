# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One owned sandbox per grading domain, driven through Gym's provider-neutral ``AsyncSandbox``.

A domain is single-use: ``create`` starts a sandbox from the operator's ``SandboxSpec`` with a per-domain working
directory ``<spec.workdir>/<job>-<domain>`` made in the setup hook, ``upload`` writes runtime files under it,
``run`` executes the fixed runner invocation as ``exec_user`` and returns the provider-reported exit together with
the bounded result bytes, and ``close`` stops it. A result carrying ``error_type``, a raised provider error or a
missing exit is "no process exit" (``provider_exec``), never a verdict; an exit the protocol pairs with no result
is ``<domain>_no_result``; a file above ``RESULT_BYTES_LIMIT`` is ``transfer_limit``. A sandbox whose stop did not
complete, the stop after a failed setup included, keeps its facade, so the next ``close`` retries the stop; the spec's
``ttl_s`` plus the metadata labels are the backstop, and there is no journal. A provider exception turned into a
category is logged by class name only: its message may carry provider text.
"""

import asyncio
import dataclasses
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from nemo_gym.sandbox import AsyncSandbox, SandboxProvider, SandboxSpec, cpu_cap_env

from .runner import RESULT_BYTES_LIMIT, RESULT_EXIT_CODES, RUNTIME_PACKAGE, WORKER_ABORT_EXIT_CODES


logger = logging.getLogger(__name__)

LABEL_JOB, LABEL_DOMAIN = "ng-grader-job", "ng-grader-domain"
CLEANUP_STATUSES = frozenset({"not_created", "created", "deleted", "delete_failed"})
# Every file transfer is awaited for ``transfer_timeout_s`` plus this grace for the wrapper itself.
TRANSFER_GRACE_S = 5.0
SAFE_PATH = re.compile(r"/[A-Za-z0-9_./-]{1,255}\Z")
# ``stat -c %s`` prints the size as a 64-bit unsigned decimal: anything else is not a size.
SIZE_TEXT = re.compile(r"[0-9]{1,20}\Z")
# A provider-sourced sandbox id is reported truncated to this many characters.
_MAX_SANDBOX_ID_CHARS = 256
SCRUBBED_ENV = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "MPLBACKEND": "Agg",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class Unscorable(Exception):
    def __init__(self, category: str, attempted: int = 0):
        super().__init__(category)
        self.category = category
        self.attempted = attempted


class CandidateWorkerAbort(Exception):
    """A runner-authenticated candidate worker abort. The candidate step catches it and scores 0."""


def reported_exit(result: Any) -> int | None:
    """The process exit the provider reported, or None when ``error_type`` is set or the code is not an int."""
    if getattr(result, "error_type", None) is not None:
        return None
    exit_code = getattr(result, "return_code", None)
    return exit_code if type(exit_code) is int else None


class SandboxDomain:
    def __init__(self, name: str, policy: Any, provider: SandboxProvider, spec: SandboxSpec, *, job_id: str) -> None:
        base = (spec.workdir or "/work").rstrip("/") or "/"
        if not SAFE_PATH.fullmatch(base):
            raise ValueError("sandbox_spec.workdir must be an absolute path without quoting characters")
        self.name = name
        self.policy = policy
        self._provider = provider
        self.workdir = f"{base}/{job_id}-{name}"
        # Every other field, files included and any field Gym adds later, passes through untouched.
        self._spec = dataclasses.replace(
            spec,
            workdir=base,
            env={**cpu_cap_env(spec.resources.cpu), **spec.env},
            metadata={**spec.metadata, LABEL_JOB: job_id, LABEL_DOMAIN: name},
            provider_options=dict(spec.provider_options),
        )
        self._sandbox: AsyncSandbox | None = None
        self._created = False  # a facade kept after a failed create is there for close() alone
        self.sandbox_id: str | None = None
        self.cleanup = "not_created"
        self.wall_time_s = 0.0
        self.last_exit: int | None = None
        self.last_error_type: str | None = None
        self._close_lock = asyncio.Lock()

    async def create(self) -> None:
        sandbox = AsyncSandbox(self._provider, owns_provider=False)
        user_refused = False

        async def setup(box: AsyncSandbox) -> None:
            nonlocal user_refused
            try:
                result = await box.exec(
                    f'mkdir -p "{self.workdir}"',
                    cwd=self._spec.workdir,
                    timeout_s=self.policy.transfer_timeout_s,
                    user=self.policy.exec_user,
                )
            except (ValueError, NotImplementedError):  # how a provider declines a user= it cannot honour
                user_refused = self.policy.exec_user is not None  # with no user requested there is nothing to refuse
                raise
            if result.error_type is not None or result.return_code != 0:
                raise RuntimeError("workdir setup failed")

        try:
            await asyncio.wait_for(sandbox.start_with_setup(self._spec, setup), timeout=self.policy.create_timeout_s)
        except BaseException as exc:
            # start_with_setup stops a sandbox whose setup failed and providers destroy half-created sandboxes. A
            # handle whose stop failed or was cancelled is still live: the domain keeps its facade, so close() retries
            # the stop, and reports it as delete_failed until a stop succeeds, never "not_created". The spec's ttl
            # stays the backstop.
            handle = getattr(sandbox, "_handle", None)
            if handle is not None and not getattr(sandbox, "_stopped", False):
                self._sandbox = sandbox
                self.sandbox_id = getattr(handle, "sandbox_id", None)
                self.cleanup = "delete_failed"
            if not isinstance(exc, Exception):
                raise  # a cancellation propagates as it came
            logger.warning("sandbox domain %s: create failed (%s)", self.name, type(exc).__name__)
            raise Unscorable("exec_user_unsupported" if user_refused else "provider_create") from None
        self._sandbox = sandbox
        self._created = True
        handle = getattr(sandbox, "_handle", None)
        self.sandbox_id = getattr(handle, "sandbox_id", None)
        self.cleanup = "created"

    def _box(self) -> AsyncSandbox:
        if self._sandbox is None or not self._created:
            raise RuntimeError("domain not created")
        return self._sandbox

    async def upload(self, files: dict[str, bytes]) -> None:
        for relative, data in files.items():
            with tempfile.TemporaryDirectory(prefix="ng-grader-upload-") as directory:
                local = Path(directory) / "payload"
                local.write_bytes(data)
                await self._transfer(self._box().upload(local, f"{self.workdir}/{relative}"))

    async def _transfer(self, awaitable: Any) -> Any:
        try:
            return await asyncio.wait_for(awaitable, timeout=self.policy.transfer_timeout_s + TRANSFER_GRACE_S)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("sandbox domain %s: transfer failed (%s)", self.name, type(exc).__name__)
            raise Unscorable("provider_exec") from None

    async def run(self, mode: str, deadline_s: int) -> tuple[int, bytes]:
        """Run the runner and return (provider-reported exit, bounded result bytes). Gates live in execution.py."""
        self.last_exit = None  # the diagnostics describe this run, never a previous one
        self.last_error_type = None
        runner = f"{self.workdir}/{RUNTIME_PACKAGE}/runner.py"
        command = (
            f'"{self.policy.interpreter}" -I -B "{runner}" {mode} "{self.workdir}/job.json" '
            f'"{self.workdir}/result.json" >/dev/null 2>&1'
        )
        # The provider must wait for the runner to reap the worker after the wall deadline (CLEANUP_BUDGET_S), or a
        # clean timeout run is cut short and mislabelled provider_exec; bounded_policy keeps the margin above it.
        timeout_s = deadline_s + int(self.policy.exec_timeout_margin_s)
        env = {**SCRUBBED_ENV, "HOME": self.workdir, "TMPDIR": self.workdir}
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self._box().exec(command, cwd=self.workdir, env=env, timeout_s=timeout_s, user=self.policy.exec_user),
                timeout=timeout_s + self.policy.exec_timeout_margin_s,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:  # the wait above expired, or the provider timed out itself
            self.last_error_type = "timeout"
            logger.warning("sandbox domain %s: run failed (%s)", self.name, type(exc).__name__)
            raise Unscorable("provider_exec") from None
        except Exception as exc:
            logger.warning("sandbox domain %s: run failed (%s)", self.name, type(exc).__name__)
            raise Unscorable("provider_exec") from None
        finally:
            self.wall_time_s += time.monotonic() - started
        self.last_error_type = getattr(result, "error_type", None)
        exit_code = reported_exit(result)
        self.last_exit = exit_code
        if exit_code is None:
            raise Unscorable("provider_exec")  # a sentinel or nothing: the provider reported no process exit
        if exit_code not in RESULT_EXIT_CODES:
            if self.name == "candidate" and exit_code in WORKER_ABORT_EXIT_CODES:
                raise CandidateWorkerAbort()
            raise Unscorable(f"{self.name}_no_result")
        return exit_code, await self._download_bounded(f"{self.workdir}/result.json")

    async def _download_bounded(self, remote: str) -> bytes:
        try:
            probe = await asyncio.wait_for(
                self._box().exec(
                    f'test -f "{remote}" && stat -c %s "{remote}"',
                    cwd=self.workdir,
                    timeout_s=self.policy.transfer_timeout_s,
                    user=self.policy.exec_user,
                ),
                timeout=self.policy.transfer_timeout_s + TRANSFER_GRACE_S,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("sandbox domain %s: size probe failed (%s)", self.name, type(exc).__name__)
            raise Unscorable("provider_exec") from None
        if probe.error_type is not None:
            raise Unscorable("provider_exec")
        if probe.return_code != 0:
            # Absent behind an accepted exit proves nothing about the domain that should have written it.
            raise Unscorable(f"{self.name}_no_result")
        size_text = (probe.stdout or "").strip()
        if not SIZE_TEXT.fullmatch(size_text):
            raise Unscorable(f"{self.name}_no_result")  # nothing trustworthy to download
        if int(size_text) > RESULT_BYTES_LIMIT:
            raise Unscorable("transfer_limit")
        with tempfile.TemporaryDirectory(prefix="ng-grader-result-") as directory:
            local = Path(directory) / "result.json"
            try:
                await asyncio.wait_for(
                    self._box().download(remote, local), timeout=self.policy.transfer_timeout_s + TRANSFER_GRACE_S
                )
                data = local.read_bytes()  # a download that returned without producing the file is no result either
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("sandbox domain %s: download failed (%s)", self.name, type(exc).__name__)
                raise Unscorable(f"{self.name}_no_result") from None
        if len(data) > RESULT_BYTES_LIMIT:
            raise Unscorable("transfer_limit")
        return data

    async def close(self) -> str:
        async with self._close_lock:  # concurrent closers reach the provider once
            if self._sandbox is None or self.cleanup == "deleted":
                return self.cleanup
            try:
                await asyncio.wait_for(self._sandbox.stop(), timeout=self.policy.cleanup_timeout_s)
            except asyncio.CancelledError:
                self.cleanup = "delete_failed"
                raise
            except Exception as exc:
                # The class name only: the message may carry provider text.
                logger.warning("sandbox domain %s: stop failed (%s)", self.name, type(exc).__name__)
                self.cleanup = "delete_failed"
                return self.cleanup
            self.cleanup = "deleted"
            return self.cleanup

    def observation(self, outcome: str) -> dict[str, Any]:
        """Category-only sandbox facts for the verify response (rollout_observability.SandboxObservation fields)."""
        return {
            "kind": "sandbox",
            "role": "verifier",
            "provider": getattr(self._provider, "name", type(self._provider).__name__),
            "sandbox_id": self.sandbox_id[:_MAX_SANDBOX_ID_CHARS] if isinstance(self.sandbox_id, str) else None,
            "outcome": outcome,
            "exit_code": self.last_exit,
            "wall_time_s": round(self.wall_time_s, 3),
            "error_type": self.last_error_type,
        }


__all__ = [
    "CLEANUP_STATUSES",
    "LABEL_DOMAIN",
    "LABEL_JOB",
    "SCRUBBED_ENV",
    "CandidateWorkerAbort",
    "SandboxDomain",
    "Unscorable",
    "reported_exit",
]
