# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON boundary for running the existing NOOA harness in a task sandbox."""

import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import Literal

from aiohttp import ClientResponse
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import rollout_context
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle, TrajectoryRecord
from nemo_gym.server_utils import (
    GlobalAIOHTTPAsyncClientConfig,
    request,
    set_global_aiohttp_client,
)
from responses_api_agents.nooa_agent.invocation import NOOAInvocationConfig
from responses_api_agents.nooa_agent.result import finalize_run_result, is_transient_infrastructure_error
from responses_api_agents.nooa_agent.runner import InProcessNOOARunner, NOOARunFailure, NOOARunRequest, NOOARunResult


class SandboxInput(BaseModel):
    """The complete allowlist of state sent into the agent process."""

    model_config = ConfigDict(extra="forbid")
    invocation: NOOAInvocationConfig
    request: NOOARunRequest
    model_base_url: AnyHttpUrl
    model_server_name: str
    max_policy_calls: int | None = Field(default=None, gt=0)
    context_window: int | None = Field(default=None, gt=0)


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
    termination_reason: str | None = None
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
        return await request(
            "POST",
            self.base_url + url_path,
            _internal=True,
            json=json.model_dump(exclude_unset=True),
            cookies=cookies,
            headers=headers,
        )


async def execute(payload: SandboxInput) -> SandboxResult:
    """Run once, retaining evidence on both exceptions and graceful cancellation."""
    result = None
    error = None
    try:
        runner = InProcessNOOARunner(
            invocation=payload.invocation,
            server_client=_ModelClient(str(payload.model_base_url)),
            model_server_name=payload.model_server_name,
            max_policy_calls=payload.max_policy_calls,
            context_window=payload.context_window,
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
            result.termination_reason = "cancelled" if kind == "cancelled" else "infrastructure_error"
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


async def _main(input_path: Path, output_path: Path, *, stop_path: Path, completion_path: Path) -> None:
    # The shared supervisor checks this fence before spawning. Check again after
    # interpreter startup, then keep watching it so a lost TERM cannot lose stop.
    if stop_path.exists():
        return
    payload = SandboxInput.model_validate_json(input_path.read_text())
    client = set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
    task = asyncio.create_task(execute(payload))
    stopping = asyncio.Event()

    def stop() -> None:
        if not stopping.is_set():
            stopping.set()
            if not task.done():
                task.cancel()

    async def watch_stop() -> None:
        while not stop_path.exists():
            await asyncio.sleep(0.05)
        stop()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop)
    watcher = asyncio.create_task(watch_stop())
    try:
        try:
            result = await task
        finally:
            await client.close()
        temporary = output_path.with_suffix(".tmp")
        temporary.write_text(result.model_dump_json())
        temporary.replace(output_path)
        temporary = completion_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"task_completed": True}))
        temporary.replace(completion_path)
        # Keep the unchanged supervisor's child alive until verification ends.
        # Failure/cancellation exits immediately so its descendants are drained.
        if result.error is None and result.response is not None:
            await stopping.wait()
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        loop.remove_signal_handler(signal.SIGTERM)


if __name__ == "__main__":
    asyncio.run(
        _main(
            Path(sys.argv[1]),
            Path(sys.argv[2]),
            stop_path=Path(sys.argv[3]),
            completion_path=Path(sys.argv[4]),
        )
    )
