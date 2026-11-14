# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym session adapter for original-harness and OpenCode AgentVista activation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    maybe_rollout_id_from_run_body,
)
from nemo_gym.server_runtime import BaseRunRequest
from nemo_gym.trajectory import AgentObservationBundle, ObservationGap
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.agentvista_agent.runtime.paths import output_path


class AgentVistaConfig(BaseResponsesAPIAgentConfig):
    resources_server: None = None
    source_root: str
    dataset: str
    image_folder: str
    task_selector: str = "all"
    model_server: ModelServerRef
    model_name: str
    output_root: str
    worker_python: str | None = None
    code_sandbox_python: str | None = None
    harness: Literal["opencode", "default"] = "opencode"
    default_max_turns: int = Field(default=30, ge=1)
    default_max_images: int = Field(default=100, ge=1)
    command: str = "opencode"
    opencode_version: str = "1.17.11"
    concurrency: int = Field(default=1, ge=1)
    max_steps: int = Field(default=150, ge=1)
    context_tokens: int = 65536
    output_tokens: int = 8192
    temperature: float = 0.0
    top_p: float = 1.0
    mcp_timeout_ms: int = Field(default=180000, ge=1000)
    agent_timeout: int = Field(default=5400, ge=1)
    episode_timeout: int = Field(default=7200, ge=1)


class AgentVistaRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, str]


class AgentVistaContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verifier_metadata: dict[str, str] = Field(default_factory=dict)


@dataclass
class AgentVistaSession(AgentSessionState):
    context: AgentVistaContext
    activation: asyncio.Task | None = None


class AgentVistaAgent(SimpleResponsesAPIAgent):
    """Execute episodes in separate processes so runtime globals cannot cross slots."""

    config: AgentVistaConfig
    _semaphore: asyncio.Semaphore = PrivateAttr()
    _inflight: dict[str, asyncio.Task] = PrivateAttr(default_factory=dict)
    _fingerprints: dict[str, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._semaphore = asyncio.Semaphore(self.config.concurrency)

    agent_context_model = AgentVistaContext

    def multiple_workers_unsupported_reason(self) -> str:
        return "AgentVista worker admission and in-flight episode bindings are process-local; use one server worker"

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AgentVistaSession:
        if body.sandbox_access is not None:
            raise AgentSeedRejectedError(
                "AgentVista does not support sandbox access; its code tool owns a namespace sandbox"
            )
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise AgentSeedRejectedError("AgentVista does not support required external tool grants")
        context = self.parse_agent_context(body)
        if context is None:
            raise AgentSeedRejectedError("AgentVista requires verifier_metadata in agent_context")
        required = {"task_id", "source_sha256", "category_id", "harness_version"}
        if not required.issubset(context.verifier_metadata):
            raise AgentSeedRejectedError("AgentVista task context is incomplete; prepare the dataset again")
        return AgentVistaSession(request=body, context=context)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        assert isinstance(state, AgentVistaSession)
        if state.activation is not None and not state.activation.done():
            state.activation.cancel()
            try:
                await state.activation
            except asyncio.CancelledError:
                pass
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            cleanup_confirmed=True,
            agent_observations=AgentObservationBundle(
                source="agentvista",
                gaps=[
                    ObservationGap(
                        code="observation_capture_failed",
                        detail="Native events are retained in the result; typed tool/model joins are unavailable",
                    )
                ],
            ),
        )

    async def responses(self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(409, "Seed an AgentVista agent session before activation")
        state = self._require_agent_session(session_id)
        assert isinstance(state, AgentVistaSession)
        legacy = AgentVistaRunRequest(
            responses_create_params=body,
            verifier_metadata=state.context.verifier_metadata,
            _ng_rollout_id=state.request.episode_id.capture_key,
        )
        if state.activation is not None and not state.activation.done():
            raise HTTPException(409, "AgentVista session already has an active activation")
        state.activation = asyncio.create_task(self._execute(legacy))
        result = await state.activation
        # The worker has no judge credentials and never assigns a measured reward.
        # The Resources Server receives the evidence, applies the original judge, and aggregates scores.
        from responses_api_agents.agentvista_agent.runtime.results import policy_response

        return policy_response(legacy, result, self.config.model_name)

    async def _execute(self, body: AgentVistaRunRequest) -> dict:
        # Capture IDs distinguish repeats and make HTTP retries idempotent.
        rollout_id = maybe_rollout_id_from_run_body(body)
        if not rollout_id:
            raise HTTPException(400, "AgentVista Bench requires _ng_rollout_id; enable Gym observability")
        identity = hashlib.sha256(rollout_id.encode()).hexdigest()
        from responses_api_agents.agentvista_agent.runtime import HARNESS_VERSION

        requested_version = body.verifier_metadata.get("harness_version")
        if requested_version and requested_version != HARNESS_VERSION:
            raise HTTPException(
                409,
                "Prepared dataset targets a different harness version; prepare it again",
            )
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "request": body.model_dump(),
                    "config": self.config.model_dump(),
                    "harness_version": HARNESS_VERSION,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        output = output_path(self.config.output_root) / identity
        if identity in self._fingerprints and self._fingerprints[identity] != fingerprint:
            raise HTTPException(409, "Rollout ID reused with different request")
        if identity not in self._inflight:
            self._fingerprints[identity] = fingerprint
            self._inflight[identity] = asyncio.create_task(self._episode(body, output, fingerprint, rollout_id))
        # Duplicate requests share completion; cancelling the owning request cleans up the episode.
        task = self._inflight[identity]
        try:
            return await task
        finally:
            if task.done():
                self._inflight.pop(identity, None)
                self._fingerprints.pop(identity, None)

    async def _episode(self, body, output: Path, fingerprint: str, rollout_id: str) -> dict:
        async with self._semaphore:
            output.mkdir(parents=True, exist_ok=True)
            # A persisted binding prevents retries from silently changing the task.
            binding = output / "binding.json"
            try:
                with binding.open("x") as stream:
                    json.dump({"fingerprint": fingerprint, "rollout_id": rollout_id}, stream)
            except FileExistsError:
                if json.loads(binding.read_text())["fingerprint"] != fingerprint:
                    raise HTTPException(409, "Rollout ID reused with different request")
                result_path = output / "result.json"
                if not result_path.exists():
                    raise HTTPException(
                        409,
                        "Incomplete episode requires inspection; use a new rollout ID to retry",
                    )
                return json.loads(result_path.read_text())
            config = self.config.model_dump()
            # Row sampling controls from the public CLI override the configured defaults.
            sampling = body.responses_create_params
            for name, target in (
                ("temperature", "temperature"),
                ("top_p", "top_p"),
                ("max_output_tokens", "output_tokens"),
            ):
                value = getattr(sampling, name, None)
                if value is not None:
                    config[target] = value
            config.update(
                defer_verification=True,
                output=str(output),
                code_sandbox_python=self.config.code_sandbox_python or sys.executable,
                task_id=body.verifier_metadata["task_id"],
                source_sha256=body.verifier_metadata["source_sha256"],
                model=self.config.model_name,
                endpoint=self.resolve_model_base_url(self.config.model_server.name, rollout_id),
            )
            config_path = output / "worker.json"
            config_path.write_text(json.dumps(config, indent=2))
            env = {key: value for key, value in os.environ.items() if not key.startswith("VERIFIER_")}
            gym_root = str(Path(__file__).resolve().parents[2])
            env["PYTHONPATH"] = gym_root + os.pathsep + str(Path(self.config.source_root).resolve())
            env["AGENTVISTA_ROOT"] = str(Path(self.config.source_root).resolve())
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            for key, folder in {
                "HOME": "home",
                "XDG_CACHE_HOME": "cache",
                "XDG_CONFIG_HOME": "config",
                "XDG_DATA_HOME": "data",
                "XDG_STATE_HOME": "state",
                "HF_HOME": "huggingface",
                "MPLCONFIGDIR": "matplotlib",
                "IPYTHONDIR": "ipython",
                "JUPYTER_DATA_DIR": "jupyter-data",
                "JUPYTER_RUNTIME_DIR": "jupyter-runtime",
                "TMPDIR": "tmp",
            }.items():
                directory = output / "storage" / folder
                directory.mkdir(parents=True, exist_ok=True)
                env[key] = str(directory)
            process = None
            with (output / "worker.log").open("w") as log:
                try:
                    process = await asyncio.create_subprocess_exec(
                        self.config.worker_python or sys.executable,
                        "-m",
                        "responses_api_agents.agentvista_agent.runtime.default_worker"
                        if self.config.harness == "default"
                        else "responses_api_agents.agentvista_agent.runtime.worker",
                        str(config_path),
                        env=env,
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                    )
                    await asyncio.wait_for(process.wait(), timeout=self.config.episode_timeout)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    if process and process.returncode is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            await asyncio.wait_for(process.wait(), timeout=45)
                        except asyncio.TimeoutError:
                            os.killpg(process.pid, signal.SIGKILL)
                            await process.wait()
                    raise
            result_path = output / "result.json"
            if not result_path.is_file():
                raise HTTPException(500, f"Worker exited without result; inspect {output}")
            return json.loads(result_path.read_text())


if __name__ == "__main__":
    AgentVistaAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = AgentVistaAgent.run_webserver()
