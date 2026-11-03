# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Borrow a task sandbox and supervise one whole NOOA activation."""

import asyncio
import logging
from dataclasses import replace
from shlex import quote
from uuid import uuid4

from pydantic import ValidationError

from nemo_gym.agents import SandboxCommand, SandboxSession, resolve_agent_timeout
from nemo_gym.sandbox import AsyncSandbox, read_text, upload_text
from nemo_gym.trajectory import AgentObservationBundle, ObservationGap
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig
from responses_api_agents.nooa_agent.result import is_transient_infrastructure_error, set_response_lifecycle
from responses_api_agents.nooa_agent.runner import NOOARunFailure, NOOARunRequest, NOOARunResult
from responses_api_agents.nooa_agent.sandbox_entrypoint import SandboxCheckpoint, SandboxInput, SandboxResult
from responses_api_agents.nooa_agent.termination import NOOATerminationReason


LOGGER = logging.getLogger(__name__)


class SandboxNOOARunner:
    """Translate NOOA artifacts while the shared session owns process lifecycle."""

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
        timeout: float = 21600,
        close_timeout: float = 30,
    ) -> None:
        self.sandbox = sandbox
        self.python = python
        self.invocation = invocation
        self.model_base_url = model_base_url
        self.model_server_name = model_server_name
        self.max_policy_calls = max_policy_calls
        self.timeout = timeout
        self.close_timeout = close_timeout
        # Construct before prepare so failed setup can release the borrowed client.
        self.session = SandboxSession[None](
            sandbox=sandbox,
            session_dir=f"/tmp/nemo-gym-nooa/{uuid4().hex}",
            workdir=workdir,
            harness="NOOA",
            owns_sandbox=False,
        )
        self.artifact: SandboxResult | None = None
        self.checkpoint: SandboxCheckpoint | None = None
        self.observations: AgentObservationBundle | None = None
        self.diagnostics = ""

    @property
    def directory(self) -> str:
        """Location of activation files outside the graded repository."""
        return self.session.session_dir

    async def prepare(self) -> None:
        """Create session files outside the graded repository."""
        result = await self.sandbox.exec(f"mkdir -p {quote(self.directory)}", cwd="/", timeout_s=30)
        if result.return_code != 0 or result.error_type:
            raise RuntimeError("Could not prepare NOOA session directory")

    async def _stage_activation(self, request: NOOARunRequest) -> SandboxCommand:
        payload = SandboxInput(
            invocation=self.invocation,
            request=request,
            model_base_url=self.model_base_url,
            model_server_name=self.model_server_name,
            max_policy_calls=self.max_policy_calls,
        )
        await upload_text(self.sandbox, path=f"{self.directory}/input.json", text=payload.model_dump_json())
        return SandboxCommand(
            argv=[
                self.python,
                "-I",
                "-m",
                "responses_api_agents.nooa_agent.sandbox_entrypoint",
                f"{self.directory}/input.json",
                f"{self.directory}/result.json",
            ],
        )

    async def run(self, request: NOOARunRequest) -> NOOARunResult:
        """Execute once, preserving evidence without allowing it to override failure."""
        try:
            capture_failure: Exception | None = None
            try:
                await self.session.execute(
                    stage_activation=lambda: self._stage_activation(request),
                    collect=self._collect_artifacts,
                    timeout=resolve_agent_timeout(request.responses_create_params.metadata, ceiling=self.timeout),
                    close_timeout=self.close_timeout,
                )
            except Exception as error:
                # Some providers report absent artifacts as generic transfer errors.
                # Inspect the confirmed worker outcome before propagating capture errors;
                # independent execution/transport failures still retain precedence.
                capture_error = self.session.capture_error
                if capture_error is None or (error is not capture_error and error.__cause__ is not capture_error):
                    raise
                capture_failure = error
            if self.session.closing:
                raise asyncio.CancelledError("NOOA sandbox session closed during execution")
            cleanup = self.session.cleanup
            if cleanup is None:
                raise RuntimeError("NOOA worker termination is unconfirmed")
            if (
                cleanup["timed_out"]
                or cleanup["error"]
                or cleanup["return_code"] != 0
                or self.artifact is None
                or capture_failure is not None
            ):
                await self._collect_diagnostics()
            if cleanup["timed_out"]:
                # Exhausting the worker budget is not a retryable dependency outage.
                raise RuntimeError("NOOA sandbox worker exceeded its execution deadline")
            if cleanup["error"]:
                raise RuntimeError(f"NOOA sandbox supervisor failed: {cleanup['error']}")
            if cleanup["return_code"] != 0:
                raise RuntimeError(f"NOOA sandbox worker did not exit successfully (exit {cleanup['return_code']})")
            if capture_failure is not None:
                raise capture_failure
            if self.artifact is None:
                raise RuntimeError(f"NOOA sandbox result is missing or malformed: {self.diagnostics}")
            request.model_cookies.update(self.artifact.model_cookies)
            request.resource_cookies.update(self.artifact.resource_cookies)
            if self.artifact.error is not None:
                detail = self.artifact.error.message
                error = ConnectionError(detail) if self.artifact.error.kind == "transient" else RuntimeError(detail)
                raise error
            return self.artifact.run_result()
        except (Exception, asyncio.CancelledError) as error:
            self._record_capture_error(error)
            evidence = self.artifact or self.checkpoint
            if evidence is not None:
                request.model_cookies.update(evidence.model_cookies)
                request.resource_cookies.update(evidence.resource_cookies)
                if evidence.response is not None:
                    result = evidence.run_result()
                    reason = (
                        NOOATerminationReason.CANCELLED
                        if isinstance(error, asyncio.CancelledError)
                        else NOOATerminationReason.INFRASTRUCTURE_ERROR
                    )
                    detail = str(error) or "NOOA execution was cancelled"
                    result.termination_reason = reason
                    result.termination_error = detail
                    result.episode = replace(
                        result.episode,
                        response=set_response_lifecycle(result.episode.response, reason, detail),
                        observations=self.observations or result.episode.observations,
                    )
                    self.observations = result.episode.observations
                    if isinstance(error, asyncio.CancelledError):
                        error.nooa_result = result
                    else:
                        raise NOOARunFailure(error, result) from error
            raise

    async def _collect_diagnostics(self) -> None:
        """Retain a bounded log tail only after the important evidence is captured."""
        try:
            async with asyncio.timeout(min(1.0, self.close_timeout)):
                tail = (await self.session.read_output_log())[-2000:]
        except TimeoutError:
            return
        self.diagnostics = "".join(char for char in tail if char.isprintable() or char in "\n\t")
        if self.diagnostics:
            observations = (self.observations or AgentObservationBundle(source="nooa")).model_copy(deep=True)
            observations.gaps.append(ObservationGap(code="sandbox_worker_diagnostics", detail=self.diagnostics))
            self.observations = observations

    async def _read_artifact(self, name: str) -> str:
        # The shared session bounds the entire capture, including this one retry.
        try:
            return await read_text(self.sandbox, path=f"{self.directory}/{name}")
        except Exception as error:
            if not is_transient_infrastructure_error(error):
                raise
            LOGGER.warning("Retrying NOOA %s capture after %s", name, type(error).__name__)
        return await read_text(self.sandbox, path=f"{self.directory}/{name}")

    async def _collect_artifacts(self) -> None:
        """Prefer terminal evidence; recovered checkpoints can never establish completion."""
        capture_error: Exception | None = None
        try:
            artifact = SandboxResult.model_validate_json(await self._read_artifact("result.json"))
        except (FileNotFoundError, ValidationError) as error:
            unavailable = f"missing or malformed ({type(error).__name__})"
        except Exception as error:
            capture_error = error
            unavailable = f"unavailable ({type(error).__name__})"
        else:
            if artifact.response is not None or artifact.error is not None:
                self.artifact = artifact
                self.checkpoint = None
                self.observations = artifact.observations
                return
            unavailable = "missing or malformed (neither a response nor an error)"

        gap = ObservationGap(
            code="sandbox_result_unavailable",
            detail=f"NOOA terminal result is {unavailable}; recovered evidence cannot establish completion.",
        )
        try:
            checkpoint = SandboxCheckpoint.model_validate_json(await self._read_artifact("checkpoint.json"))
        except (FileNotFoundError, ValidationError):
            checkpoint = self.checkpoint
        except Exception as error:
            capture_error = capture_error or error
            checkpoint = self.checkpoint
        if checkpoint is None:
            self.observations = AgentObservationBundle(source="nooa", gaps=[gap])
        else:
            detail = (
                "NOOA worker exited without a valid final result; checkpoint evidence is partial and not gradable."
            )
            observations = checkpoint.observations.model_copy(deep=True)
            observations.gaps = list(
                {
                    item.model_dump_json(): item
                    for item in [*observations.gaps, gap, ObservationGap(code="partial_checkpoint", detail=detail)]
                }.values()
            )
            self.checkpoint = checkpoint.model_copy(
                update={
                    "response": set_response_lifecycle(
                        checkpoint.response, NOOATerminationReason.INFRASTRUCTURE_ERROR, detail
                    )
                    if checkpoint.response is not None
                    else None,
                    "observations": observations,
                    "termination_reason": NOOATerminationReason.INFRASTRUCTURE_ERROR,
                    "termination_error": detail,
                }
            )
            self.observations = observations
        if capture_error is not None:
            raise capture_error

    def _record_capture_error(self, failure: BaseException | None = None) -> None:
        error: BaseException | None = self.session.capture_error or self.session.termination_error
        if error is None and self.session.launch_started and self.session.cleanup is None:
            # Unknown cleanup blocks capture but remains retryable in the shared session.
            error = failure
        if error is None:
            return
        observations = (self.observations or AgentObservationBundle(source="nooa")).model_copy(deep=True)
        gap = ObservationGap(code="sandbox_capture_failed", detail=f"NOOA artifact capture unavailable: {error}")
        observations.gaps = list({item.model_dump_json(): item for item in [*observations.gaps, gap]}.values())
        self.observations = observations

    async def close(self) -> None:
        """Join shared finalization and retain captured evidence after releasing the client."""
        try:
            await self.session.close(timeout=self.close_timeout)
        except (Exception, asyncio.CancelledError) as error:
            self._record_capture_error(error)
            raise
        else:
            self._record_capture_error()
