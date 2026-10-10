# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Borrow a task sandbox and supervise one whole NOOA activation."""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from shlex import quote
from uuid import uuid4

from nemo_gym.rollout_observability import AgentObservationBundle, ObservationGap
from nemo_gym.sandbox import AsyncSandbox
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig
from responses_api_agents.nooa_agent.runner import NOOARunFailure, NOOARunRequest, NOOARunResult
from responses_api_agents.nooa_agent.sandbox_entrypoint import SandboxInput, SandboxResult


LOGGER = logging.getLogger(__name__)


class SandboxNOOARunner:
    """Keep provider access outside the worker and require a stop receipt before close."""

    def __init__(
        self,
        *,
        sandbox: AsyncSandbox,
        workdir: str,
        python: str,
        invocation: NOOAInvocationConfig,
        model_base_url: str,
        model_server_name: str,
        max_policy_calls: int | None,
        context_window: int | None = None,
    ) -> None:
        self.sandbox = sandbox
        self.workdir = workdir
        self.python = python
        self.invocation = invocation
        self.model_base_url = model_base_url
        self.model_server_name = model_server_name
        self.max_policy_calls = max_policy_calls
        self.context_window = context_window
        self.directory = f"/tmp/nemo-gym-nooa/{uuid4().hex}"
        self.launched = False
        self.stopped = False
        self.artifact: SandboxResult | None = None
        self.observations: AgentObservationBundle | None = None
        self.diagnostics = ""

    async def prepare(self) -> None:
        """Create session files outside the graded repository."""
        result = await self.sandbox.exec(f"mkdir -p {quote(self.directory)}", cwd="/", timeout_s=30)
        if result.return_code != 0:
            raise RuntimeError("Could not prepare NOOA session directory")

    async def _read(self, name: str) -> str:
        with tempfile.TemporaryDirectory(prefix="nooa-result-") as directory:
            path = Path(directory) / name
            await self.sandbox.download(f"{self.directory}/{name}", path)
            return path.read_text()

    async def run(self, request: NOOARunRequest) -> NOOARunResult:
        payload = SandboxInput(
            invocation=self.invocation,
            request=request,
            model_base_url=self.model_base_url,
            model_server_name=self.model_server_name,
            max_policy_calls=self.max_policy_calls,
            context_window=self.context_window,
        )
        with tempfile.TemporaryDirectory(prefix="nooa-input-") as directory:
            path = Path(directory) / "input.json"
            path.write_text(payload.model_dump_json())
            await self.sandbox.upload(path, f"{self.directory}/input.json")
        d, python = quote(self.directory), quote(self.python)
        # The atomic claim also fences an exec whose response is lost or delayed until after close.
        command = (
            f"(trap '' TERM; ln -s launch {d}/launch.claim 2>/dev/null || exit 0; "
            f"{python} -I -m responses_api_agents.nooa_agent.sandbox_supervisor "
            f"{d}; echo $? > {d}/runner.exit) "
            f">{d}/stdout.log 2>{d}/stderr.log </dev/null &"
        )
        self.launched = True
        # Episode cancellation comes from the NOOA environment; no second episode clock.
        try:
            launch = await self.sandbox.exec(
                command, cwd=self.workdir, timeout_s=30, preserve_background_services=True
            )
            if launch.return_code != 0:
                raise RuntimeError("Could not launch NOOA supervisor")
            while True:
                # One small exec avoids repeated failed provider downloads while the
                # worker runs. A cleanup receipt also covers launch/setup failures.
                status = await self.sandbox.exec(
                    f"test -f {d}/completion.json || test -f {d}/cleanup.json || test -f {d}/runner.exit",
                    cwd="/",
                    timeout_s=30,
                )
                if status.return_code == 0:
                    break
                await asyncio.sleep(2)
            await self.collect()
            if self.artifact is None:
                raise RuntimeError(f"NOOA sandbox result is missing or malformed: {self.diagnostics}")
            try:
                completion = json.loads(await self._read("completion.json"))
            except Exception:
                raise RuntimeError("NOOA task completion is unconfirmed") from None
            # Task completion precedes worker exit: the worker holds services
            # for verification until close asks the shared supervisor to reap.
            if not isinstance(completion, dict) or completion.get("task_completed") is not True:
                raise RuntimeError(f"NOOA task did not complete: {completion}")
            request.model_cookies.update(self.artifact.model_cookies)
            request.resource_cookies.update(self.artifact.resource_cookies)
            if self.artifact.error is not None:
                detail = self.artifact.error.message
                error = ConnectionError(detail) if self.artifact.error.kind == "transient" else RuntimeError(detail)
                if self.artifact.response is not None:
                    raise NOOARunFailure(error, self.artifact.run_result()) from error
                raise error
            return self.artifact.run_result()
        except BaseException:
            # Failures and cancellation never retain task services. Keep the
            # original terminal error if cleanup fails; close can retry the handle.
            try:
                await self.stop()
            except Exception:
                LOGGER.exception("NOOA cleanup failed after interrupted execution")
            raise

    async def stop(self) -> None:
        """Fence a pending launch or wait for the shared supervisor to reap descendants."""
        if not self.launched or self.stopped:
            return
        d = quote(self.directory)
        script = (
            f"touch {d}/runner.stop || exit 1; "
            f"ln -s stop {d}/launch.claim 2>/dev/null || true; "
            f'if [ "$(readlink {d}/launch.claim)" = stop ]; then '
            f"printf '%s' '{{\"cleanup_confirmed\":true}}' > {d}/cleanup.tmp && "
            f"mv {d}/cleanup.tmp {d}/cleanup.json; exit $?; fi; "
            f'if [ -s {d}/runner.pid ]; then kill -TERM "$(cat {d}/runner.pid)" 2>/dev/null || true; fi; '
            f"for _ in $(seq 1 20); do [ -f {d}/cleanup.json ] && exit 0; sleep 1; done; exit 1"
        )
        # A completed supervisor already wrote its receipt. Avoid signaling a reused PID.
        try:
            receipt = json.loads(await self._read("cleanup.json"))
        except Exception:
            receipt = {}
        if not isinstance(receipt, dict) or receipt.get("cleanup_confirmed") is not True:
            await self.sandbox.exec(script, cwd="/", timeout_s=25)
            receipt = json.loads(await self._read("cleanup.json"))
        if not isinstance(receipt, dict) or receipt.get("cleanup_confirmed") is not True:
            raise RuntimeError("NOOA descendant cleanup is unconfirmed; close must retry")
        self.stopped = True

    async def collect(self) -> None:
        """Retain a graceful checkpoint, or explicitly mark evidence lost to a hard stop."""
        if not self.launched or self.artifact is not None:
            return
        try:
            self.artifact = SandboxResult.model_validate_json(await self._read("result.json"))
            self.observations = self.artifact.observations
        except Exception:
            try:
                self.diagnostics = (await self._read("stderr.log"))[-2000:]
            except Exception:
                self.diagnostics = "Runner diagnostics unavailable"
            self.observations = AgentObservationBundle(
                source="nooa",
                gaps=[
                    ObservationGap(
                        code="sandbox_result_unavailable",
                        detail="NOOA result missing or malformed; in-memory hooks may be lost. External model capture remains.",
                    )
                ],
            )

    async def close(self) -> None:
        """Stop before disconnecting; a failed cleanup leaves this handle available for retry."""
        await self.stop()
        await self.collect()
        d, retired = quote(self.directory), quote(self.directory + ".closed")
        result = await self.sandbox.exec(
            f"if [ -d {d} ]; then mv {d} {retired} || exit 1; fi; rm -rf {retired}", cwd="/", timeout_s=30
        )
        if result.return_code != 0:
            raise RuntimeError("Could not remove NOOA session files")
        await self.sandbox.disconnect()
