# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox execution mode for the LangChain DeepAgents adapter."""

import asyncio
import time
from pathlib import PurePosixPath
from typing import Any, Literal
from uuid import uuid4

from fastapi import Body, HTTPException, Request
from pydantic import Field

from nemo_gym.agent_outcome import AgentOutcome
from nemo_gym.agent_utils.sandbox_session import SandboxSession
from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.failure_kinds import MODEL_REQUEST_ERROR
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseUsage,
    accumulate_response_usage,
)
from nemo_gym.rollout_observability import AgentObservationBundle, ObservationGap, TrajectoryRecord
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox.agent_tools import sandbox_server_url
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.providers import create_provider
from responses_api_agents.langchain_deepagents_agent.runtime_messages import from_items
from responses_api_agents.langchain_deepagents_agent.sandbox import DeepAgentsSandboxSession


class SandboxedDeepAgentConfig(BaseResponsesAPIAgentConfig):
    """Pinned native SDK configuration; model limits apply to each internal call."""

    model_server: ModelServerRef
    deepagents_version: Literal["0.6.12"] = "0.6.12"
    model: str = "default"
    max_input_tokens: int = Field(default=262144, gt=0)
    max_output_tokens: int = Field(default=32768, gt=0)
    temperature: float = 1.0
    top_p: float = 0.95
    system_prompt: str | None = None
    recursion_limit: int = Field(default=1000, gt=0)
    concurrency: int = Field(default=8, gt=0)
    sandbox_provider: str | None = None
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    sandbox_install_timeout_seconds: float = Field(default=900, gt=0, allow_inf_nan=False)
    sandbox_runner_timeout_seconds: float = Field(default=12600, gt=0, allow_inf_nan=False)
    shell_timeout_seconds: int = Field(default=120, gt=0)
    session_close_timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)


class SandboxedDeepAgent(SimpleResponsesAPIAgent):
    """Run a native DeepAgents graph in a borrowed or explicitly configured sandbox."""

    config: SandboxedDeepAgentConfig
    checkpoint_mode = "restart_only"
    sem: Any = Field(default=None, exclude=True)

    def model_post_init(self, context: Any) -> None:
        self.sem = asyncio.Semaphore(self.config.concurrency)

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> DeepAgentsSandboxSession:
        required = [grant.name for grant in self.effective_tool_accesses(body) if grant.required]
        if required:
            raise AgentSeedRejectedError(
                f"Sandboxed DeepAgents does not support required tool grants: {', '.join(required)}"
            )
        access = body.sandbox_access
        owns = access is None
        provider_ref = self.config.sandbox_provider if owns else access.connection.provider_config_ref
        workdir = self.config.sandbox_config.get("workdir") if owns else access.workdir
        if provider_ref is None:
            raise AgentSeedRejectedError("Sandboxed DeepAgents requires sandbox_access or sandbox_provider")
        if workdir is not None and not PurePosixPath(workdir).is_absolute():
            raise AgentSeedRejectedError("Sandboxed DeepAgents requires an absolute workdir")
        if access is not None and access.user is not None:
            raise AgentSeedRejectedError("Sandboxed DeepAgents does not yet support a task user override")
        provider = create_provider(resolve_provider_config(provider_ref, get_global_config_dict()))
        try:
            if provider.name != "local":
                sandbox_server_url(self.config.model_server.name, require_reachable=True)
            if owns:
                sandbox = AsyncSandbox(provider)
                spec = dict(self.config.sandbox_config)
                spec.setdefault(
                    "ttl_s",
                    self.config.sandbox_install_timeout_seconds + self.config.sandbox_runner_timeout_seconds + 600,
                )
                await sandbox.start(SandboxSpec(**spec))
            else:
                sandbox = await AsyncSandbox.connect(access.connection.descriptor, provider=provider)
        except BaseException:
            await provider.aclose()
            raise
        state = DeepAgentsSandboxSession(
            request=body,
            sandbox_id=sandbox.sandbox_id,
            session=SandboxSession(
                sandbox=sandbox,
                workdir=workdir,
                session_dir=f"/tmp/nemo-gym-deepagents-sessions/{uuid4().hex}",
                harness="DeepAgents",
                owns_sandbox=owns,
            ),
        )
        try:
            if workdir is None:
                result = await sandbox.exec("pwd", timeout_s=30)
                workdir = (result.stdout or "").strip()
                if result.return_code != 0 or result.error_type or not PurePosixPath(workdir).is_absolute():
                    raise RuntimeError("Could not determine DeepAgents sandbox working directory")
                state.session.workdir = workdir
            await state.install_runtime(timeout=self.config.sandbox_install_timeout_seconds)
        except BaseException as error:
            try:
                await state.close(timeout=self.config.session_close_timeout_seconds)
            except BaseException:
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    def _artifacts(self, state: DeepAgentsSandboxSession) -> tuple[AgentObservationBundle, TrajectoryRecord | None]:
        raw = state.session.artifacts
        if raw is None:
            return AgentObservationBundle(
                source="langchain_deepagents", gaps=[ObservationGap(code="observation_capture_failed")]
            ), None
        bundle = AgentObservationBundle.model_validate(
            {
                "source": "langchain_deepagents",
                "records": raw["invocations"] + raw["tools"] + raw.get("compactions", []),
                "gaps": raw["gaps"],
            }
        )
        if raw["usages"] and any(usage is None for usage in raw["usages"]):
            bundle.gaps.append(
                ObservationGap(code="model_usage_unavailable", detail="at least one native model call omitted usage")
            )
        trajectory = TrajectoryRecord.model_validate(
            {
                "task_id": state.request.task_id.task_id,
                "rollout_id": state.request.episode_id.capture_key,
                # The observation bundle is authoritative for invocation/tool records.
                # Keeping duplicate producer invocations would shadow capture-enriched retry references.
                "turns": raw["turns"],
                "gaps": raw["gaps"],
            }
        )
        return bundle, trajectory

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, DeepAgentsSandboxSession):
            raise TypeError("Expected DeepAgents sandbox session")
        await state.close(timeout=self.config.session_close_timeout_seconds)
        bundle, trajectory = self._artifacts(state)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id, agent_observations=bundle, trajectory=trajectory
        )

    def classify_outcome(self, native_result: dict[str, Any]) -> AgentOutcome:
        if native_result.get("error_status"):
            return AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=native_result["error_status"])
        return super().classify_outcome(native_result)

    async def _activate(
        self, body: NeMoGymResponseCreateParamsNonStreaming, state: DeepAgentsSandboxSession
    ) -> NeMoGymResponse:
        params = body.model_dump(mode="json")
        if body.tools:
            raise HTTPException(422, "Sandboxed DeepAgents uses its native tools; request tools are unsupported")
        if body.tool_choice not in (None, "auto"):
            raise HTTPException(422, "Sandboxed DeepAgents supports only automatic native tool selection")
        for field in ("max_tool_calls", "previous_response_id", "conversation", "text"):
            if params.get(field) is not None:
                raise HTTPException(422, f"Sandboxed DeepAgents does not support {field}")
        try:
            from_items(params["input"])
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(422, str(error)) from error
        defaults = {
            "model": body.model or self.config.model,
            "temperature": self.config.temperature if body.temperature is None else body.temperature,
            "top_p": self.config.top_p if body.top_p is None else body.top_p,
            "max_output_tokens": body.max_output_tokens or self.config.max_output_tokens,
            "parallel_tool_calls": body.parallel_tool_calls,
        }
        if body.reasoning is not None:
            defaults["reasoning"] = body.reasoning
        prompt = "\n\n".join(p for p in (self.config.system_prompt, body.instructions) if p) or None
        payload = {
            "input": params["input"],
            "system_prompt": prompt,
            "model_defaults": defaults,
            "model_url": self.resolve_model_base_url(
                self.config.model_server.name, state.request.episode_id.capture_key
            ).rstrip("/")
            + "/responses",
            "model_ref": self.config.model_server.model_dump(mode="json"),
            "assistant_message_header": "x-deepagents-model-run-id",
            "workdir": state.session.workdir,
            "task_id": state.request.task_id.task_id,
            "rollout_id": state.request.episode_id.capture_key,
            "sandbox_id": state.sandbox_id,
            "stop_request_path": state.session.stop_request_path,
            "max_input_tokens": self.config.max_input_tokens,
            "recursion_limit": self.config.recursion_limit,
            "shell_timeout_seconds": self.config.shell_timeout_seconds,
            "task_env": state.request.sandbox_access.env if state.request.sandbox_access else {},
        }
        try:
            raw = await state.execute(
                payload,
                timeout=min(
                    self.config.sandbox_runner_timeout_seconds,
                    state.request.agent_timeout_sec or self.config.sandbox_runner_timeout_seconds,
                ),
                close_timeout=self.config.session_close_timeout_seconds,
            )
        except TimeoutError:
            raw = state.session.artifacts
            if raw is None or raw.get("stop_reason") != "wall_time":
                raise
        usage = None
        if raw["usages"] and all(u is not None for u in raw["usages"]):
            for u in raw["usages"]:
                usage = accumulate_response_usage(usage, NeMoGymResponseUsage.model_validate(u))
        elif not raw["usages"] and raw.get("error_status") in {400, 401, 403, 404, 413, 422}:
            # A deterministic rejection before any completed model call processed no response tokens.
            usage = NeMoGymResponseUsage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                input_tokens_details={"cached_tokens": None},
                output_tokens_details={"reasoning_tokens": None},
            )
        output = []
        for item in raw.get("output_items", []):
            if item.get("type") == "message":
                item = {**item, "id": item.get("id", f"msg_{uuid4().hex}"), "status": "completed"}
            output.append(item)
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=time.time(),
            model=defaults["model"],
            object="response",
            output=output,
            parallel_tool_calls=True,
            tools=[],
            tool_choice="auto",
            status=raw["status"],
            usage=usage,
            error={"code": "server_error", "message": raw["error"]} if raw["status"] == "failed" else None,
            metadata={"stop_reason": raw["stop_reason"]} if raw.get("stop_reason") else None,
        )
        return self.report_outcome(response, raw)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(409, "Sandboxed DeepAgents requires an agent session")
        state = self._require_agent_session(session_id)
        if not isinstance(state, DeepAgentsSandboxSession):
            raise TypeError("Expected DeepAgents sandbox session")
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise HTTPException(409, "Agent-session episode_id does not match rollout route")
        if state.task is None:
            state.activation_request = body.model_copy(deep=True)

            async def activate() -> NeMoGymResponse:
                async with self.sem:
                    return await self._activate(body, state)

            state.task = asyncio.create_task(activate())
        elif body != state.activation_request:
            raise HTTPException(409, "Retry an identical activation request")
        return (await asyncio.shield(state.task)).model_copy(deep=True)

    async def run(self, request: Request, body: BaseRunRequest = Body()) -> None:
        raise HTTPException(409, "Sandboxed DeepAgents requires an Environment Server")


if __name__ == "__main__":
    SandboxedDeepAgent.run_webserver()
