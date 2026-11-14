# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native OpenClaw activation; the Resources Server owns verification and the task container."""

from __future__ import annotations

import asyncio
import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

import anyio
from fastapi import Body, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionConflictError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config import get_first_server_config_dict, get_global_config_dict
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.trajectory import AgentObservationBundle, ObservationGap
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.openclaw_agent.app import openclaw_session_conversation
from responses_api_agents.openclaw_agent.observability import build_openclaw_observations
from responses_api_agents.wildclawbench_agent.execution import (
    ExecutionConfig,
    NativeProcess,
    episode_output,
    image_identity,
    read_lease,
)


class WildClawConfig(BaseResponsesAPIAgentConfig, ExecutionConfig):
    """The native harness has one trusted-host execution mode."""


class WildClawContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    execution_key: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    owner_host: str | None = None


@dataclass
class WildClawSession(AgentSessionState):
    output: Path
    process: NativeProcess
    lease: dict
    task: asyncio.Task | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    observations: AgentObservationBundle | None = None


class WildClawAgent(SimpleResponsesAPIAgent):
    config: WildClawConfig
    agent_context_model: ClassVar[type[BaseModel]] = WildClawContext
    _semaphore: asyncio.Semaphore = PrivateAttr()
    _image_identity: str | None = PrivateAttr(default=None)

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        policy = get_first_server_config_dict(get_global_config_dict(), self.config.model_server.name)
        if "uses_reasoning_parser" in policy and (
            policy.get("entrypoint") != "wildclawbench.py"
            or policy.get("propagate_context_overflow_errors") is not True
        ):
            raise ValueError("WildClaw vLLM requires wildclawbench.py and propagate_context_overflow_errors=true")
        self._semaphore = asyncio.Semaphore(1)

    def multiple_workers_unsupported_reason(self) -> str:
        return "fixed warmup ports and episode ownership are local to one process"

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> WildClawSession:
        if body.sandbox_access is not None:
            raise AgentSeedRejectedError(
                "WildClawBench does not support sandbox access; it uses its native resources owner"
            )
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise AgentSeedRejectedError("WildClawBench does not support required external tool grants")
        context = self.parse_agent_context(body)
        assert isinstance(context, WildClawContext)
        if not context.execution_key or context.owner_host != socket.gethostname():
            raise AgentSeedRejectedError("Native resources and agent must share one host and workspace")
        execution = ExecutionConfig.model_validate(self.config.model_dump())
        try:
            lease = read_lease(execution, context.execution_key)
        except (OSError, ValueError, KeyError):
            raise AgentSeedRejectedError("Native resources lease is missing or incompatible") from None
        if lease["episode_id"] != body.episode_id.model_dump(mode="json") or lease[
            "task_id"
        ] != body.task_id.model_dump(mode="json"):
            raise AgentSessionConflictError("Native lease belongs to another episode or task")
        if self._image_identity is None:
            self._image_identity = image_identity(execution)
        output = episode_output(execution, context.execution_key)
        binding = dict(agent_session_id=body.agent_session_id, episode_id=lease["episode_id"])
        try:
            with (output / "agent_binding.json").open("x") as stream:
                json.dump(binding, stream)
        except FileExistsError:
            if json.loads((output / "agent_binding.json").read_text()) != binding:
                raise AgentSessionConflictError("Native lease already belongs to another agent session") from None
        worker = dict(
            source_root=execution.source_root,
            output=str(output),
            metadata=lease["metadata"],
            input_prompt=lease["prompt"],
            model_name=execution.model_name,
            policy_base_url=self.resolve_model_base_url(execution.model_server.name, body.episode_id.capture_key),
            context_window=execution.context_window,
            max_tokens=execution.max_tokens,
            thinking=execution.thinking,
            evaluation_protocol=dict(
                model_name=execution.model_name,
                context_window=execution.context_window,
                max_tokens=execution.max_tokens,
                thinking=execution.thinking,
                judge_model=execution.judge_model,
                sandbox_image=self._image_identity,
                model_input=["text", "image"],
                model_reasoning=True,
                gym_protocol="agent_sessions_v1",
            ),
        )
        (output / "worker.json").write_text(json.dumps(worker, indent=2))
        return WildClawSession(
            request=body.model_copy(deep=True),
            output=output,
            process=NativeProcess(execution, output, self._image_identity),
            lease=lease,
        )

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise AgentSessionConflictError("WildClawBench activation requires a seeded agent session")
        state = self._require_agent_session(session_id)
        assert isinstance(state, WildClawSession)
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise AgentSessionConflictError("Activation route differs from the seeded episode")
        if state.task is None:
            dumped = body.model_dump(exclude_none=True, exclude_defaults=True)
            inputs = dumped.pop("input", None)
            # Model settings are config-owned. Never silently discard a caller's sampling/tools/system overrides.
            if dumped:
                raise HTTPException(
                    422, "WildClawBench supports only the prepared input; model settings belong in config"
                )
            if (
                not isinstance(inputs, list)
                or len(inputs) != 1
                or inputs[0].get("role") != "user"
                or inputs[0].get("content") != state.lease["prompt"]
            ):
                raise HTTPException(422, "Use the unmodified prepared single user prompt")
            state.activation_request = body.model_copy(deep=True)
            state.task = asyncio.create_task(self._activate(state))
        elif state.activation_request != body:
            raise AgentSessionConflictError("WildClawBench supports one activation; retry the same request")
        return (await asyncio.shield(state.task)).model_copy(deep=True)

    async def _activate(self, state: WildClawSession) -> NeMoGymResponse:
        async with self._semaphore:
            result = await state.process.invoke("agent", int(state.lease["metadata"]["timeout_seconds"]) + 900)
        events = result.get("events", [])
        conversation = openclaw_session_conversation(events)
        state.observations = build_openclaw_observations(
            state.request.episode_id.capture_key,
            conversation,
            events,
            transcript_available=bool(events),
            prefer_openclaw_session_id=False,
            capture_correlated=True,
            model_ref=self.config.model_server,
        )
        return NeMoGymResponse.model_validate(
            dict(
                id="resp_" + uuid4().hex,
                created_at=int(time.time()),
                model=self.config.model_name,
                object="response",
                status="incomplete" if result.get("errors") else "completed",
                output=[
                    item for item in conversation if getattr(item, "role", None) not in {"user", "system", "developer"}
                ],
                parallel_tool_calls=False,
                tools=[],
                tool_choice="auto",
            )
        )

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        assert isinstance(state, WildClawSession)
        if state.task is not None and not state.task.done():
            state.task.cancel()
        with anyio.CancelScope(shield=True):
            if state.task is not None:
                try:
                    await asyncio.shield(state.task)
                except (asyncio.CancelledError, Exception):
                    pass
            await state.process.stop()
        # A killed worker without its native finally receipt is inconclusive until the Resources owner confirms
        # container cleanup. Keep state on failure so a later close can consume that owner's cleanup receipt.
        if (state.output / "agent_started.json").exists():
            receipt_path = state.output / "resources_closed.json"
            if not receipt_path.exists():
                receipt_path = state.output / "harness_stopped.json"
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("cleanup_confirmed") is not True:
                raise RuntimeError("Native harness termination is unconfirmed")
        observations = state.observations or AgentObservationBundle(
            source="wildclawbench_agent",
            gaps=[ObservationGap(code="observation_capture_failed")],
        )
        (state.output / "agent_closed.json").write_text(
            json.dumps(
                dict(
                    cleanup_confirmed=True,
                    episode_id=state.request.episode_id.model_dump(mode="json"),
                )
            )
        )
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id, agent_observations=observations, cleanup_confirmed=True
        )


if __name__ == "__main__":
    WildClawAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = WildClawAgent.run_webserver()
