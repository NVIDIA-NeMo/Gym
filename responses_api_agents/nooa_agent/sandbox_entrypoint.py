# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON boundary for running the existing NOOA harness in a task sandbox.

A missing or malformed result plus a nonzero exit is an infrastructure failure,
including invalid input, transport setup, finalization, or publication failures.
The controller must not grade a recovered checkpoint or infer success from it;
process status and stderr distinguish these failures from an abrupt worker kill.

The worker has no independent wall-clock deadline. Its caller must supervise it:
SandboxNOOARunner (introduced in PR #3851) supplies the shared supervisor's
execution timeout, which also bounds model requests and their connection retries.
Standalone callers must enforce the same process deadline.

Artifacts are private (0600). The supervisor launches the worker without changing
its UID, and the provider's artifact reader must use that UID or privileged file
access. Cross-UID, unprivileged artifact collection is not supported.
"""

import asyncio
import os
import signal
import sys
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

from aiohttp import ClientResponse
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import rollout_context
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle, ObservationGap, TrajectoryRecord
from nemo_gym.server_utils import (
    GlobalAIOHTTPAsyncClientConfig,
    request,
    set_global_aiohttp_client,
)
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig
from responses_api_agents.nooa_agent.result import finalize_run_result, is_transient_infrastructure_error
from responses_api_agents.nooa_agent.runner import InProcessNOOARunner, NOOARunFailure, NOOARunRequest, NOOARunResult
from responses_api_agents.nooa_agent.termination import NOOATerminationReason


class SandboxInput(BaseModel):
    """The complete allowlist of state sent into the agent process."""

    model_config = ConfigDict(extra="forbid")
    invocation: NOOAInvocationConfig
    request: NOOARunRequest
    model_base_url: AnyHttpUrl
    model_server_name: str
    max_policy_calls: int | None = Field(default=None, gt=0)


class RunnerError(BaseModel):
    kind: Literal["transient", "fatal", "cancelled"]
    message: str


class SandboxResult(BaseModel):
    """Agent-reported evaluation evidence, never an authoritative reward."""

    model_config = ConfigDict(extra="forbid")
    response: NeMoGymResponse | None = None
    observations: AgentObservationBundle
    trajectory: TrajectoryRecord | None = None
    model_cookies: dict[str, str]
    resource_cookies: dict[str, str]
    termination_reason: NOOATerminationReason | None = None
    termination_error: str | None = None
    error: RunnerError | None = None

    def run_result(self) -> NOOARunResult:
        """Restore the existing result contract without serializing Python return values."""
        if self.response is None:
            raise RuntimeError("NOOA sandbox returned no response")
        return NOOARunResult(
            episode=AgentEpisode(response=self.response, observations=self.observations),
            return_value=None,
            model_cookies=self.model_cookies,
            resource_cookies=self.resource_cookies,
            termination_reason=self.termination_reason,
            termination_error=self.termination_error,
            trajectory=self.trajectory,
        )


class SandboxCheckpoint(SandboxResult):
    """Partial evidence only; its presence never establishes a terminal outcome."""

    partial: Literal[True]


def _write_json(path: Path, value: BaseModel) -> None:
    # Serialize before opening the temporary file. A failed or interrupted write
    # leaves the previous complete snapshot intact; unique paths avoid temp races.
    serialized = value.model_dump_json()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_checkpoint(path: Path, result: NOOARunResult) -> None:
    """Atomically persist serializable evidence without arbitrary return values."""
    gap = ObservationGap(
        code="partial_checkpoint", detail="Partial NOOA checkpoint: evidence only, not a terminal or gradable result."
    )
    checkpoint = SandboxCheckpoint(
        partial=True,
        response=result.episode.response.model_copy(update={"status": "incomplete", "error": None}),
        observations=result.episode.observations.model_copy(update={"gaps": [*result.episode.observations.gaps, gap]}),
        trajectory=(
            result.trajectory.model_copy(update={"gaps": [*result.trajectory.gaps, gap]})
            if result.trajectory is not None
            else None
        ),
        model_cookies=result.model_cookies,
        resource_cookies=result.resource_cookies,
    )
    _write_json(path, checkpoint)


class _ModelClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def post(
        self,
        *,
        server_name: str,
        url_path: str,
        json: NeMoGymResponseCreateParamsNonStreaming,
        cookies: dict[str, str],
        headers: dict[str, str] | None = None,
    ) -> ClientResponse:
        # The supplied path already includes rollout and token-capture routing.
        # The caller's process deadline bounds the shared client's connection retries.
        return await request(
            "POST",
            self.base_url + url_path,
            _internal=True,
            json=json.model_dump(exclude_unset=True),
            cookies=cookies,
            headers=headers,
        )


async def execute(
    payload: SandboxInput, *, on_progress: Callable[[NOOARunResult], None] | None = None
) -> SandboxResult:
    """Run once, retaining evidence on both exceptions and graceful cancellation."""
    result = None
    error = None
    try:
        runner = InProcessNOOARunner(
            invocation=payload.invocation,
            server_client=_ModelClient(str(payload.model_base_url)),
            model_server_name=payload.model_server_name,
            max_policy_calls=payload.max_policy_calls,
            on_progress=on_progress,
        )
        with rollout_context(payload.request.rollout_id):
            result = await runner.run(payload.request)
    except (Exception, asyncio.CancelledError) as exc:
        result = exc.result if isinstance(exc, NOOARunFailure) else getattr(exc, "nooa_result", None)
        kind = (
            "cancelled"
            if isinstance(exc, asyncio.CancelledError)
            else ("transient" if is_transient_infrastructure_error(exc) else "fatal")
        )
        error = RunnerError(kind=kind, message=f"{type(exc).__name__}: {exc}"[:2000])
        if result is not None:
            result.termination_reason = (
                NOOATerminationReason.CANCELLED if kind == "cancelled" else NOOATerminationReason.INFRASTRUCTURE_ERROR
            )
            result.termination_error = error.message
    response, observations = finalize_run_result(result) if result else (None, AgentObservationBundle(source="nooa"))
    return SandboxResult(
        response=response,
        observations=observations,
        trajectory=result.trajectory if result else None,
        model_cookies=payload.request.model_cookies,
        resource_cookies=payload.request.resource_cookies,
        termination_reason=result.termination_reason if result else None,
        termination_error=result.termination_error if result else None,
        error=error,
    )


async def _main(input_path: Path, output_path: Path) -> None:
    payload = SandboxInput.model_validate_json(input_path.read_text())
    task = asyncio.create_task(
        execute(payload, on_progress=partial(write_checkpoint, output_path.with_name("checkpoint.json")))
    )
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    client = set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
    try:
        result = await task
    finally:
        await client.close()
    _write_json(output_path, result)


if __name__ == "__main__":
    asyncio.run(_main(Path(sys.argv[1]), Path(sys.argv[2])))
