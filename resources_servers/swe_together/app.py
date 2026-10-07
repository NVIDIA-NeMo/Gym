# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SWE-Together Resources: task ownership, interactive user, and fixed evaluation."""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.agent_utils.ordered_operations import OrderedOperationLedger
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesVerifyRequest,
    SimpleResourcesServer,
)
from nemo_gym.config_types import AgentServerRef, ModelServerRef
from nemo_gym.interactive_agent_types import (
    InteractiveResourcesSeedResponse,
    InteractiveVerificationInput,
    ResourcesStepRequest,
    ResourcesStepResponse,
)
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.python_runtime import ensure_python
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.swe_together.artifacts import RepositorySnapshots
from resources_servers.swe_together.context import INCREMENTAL_NOTICE, project_turn
from resources_servers.swe_together.evaluation import interaction_metrics, run_judge
from resources_servers.swe_together.guidance import session_analysis
from resources_servers.swe_together.model_client import AuxiliaryModel
from resources_servers.swe_together.patch_normalize import apply_candidates
from resources_servers.swe_together.repo_config import discover_repo_config_files
from resources_servers.swe_together.simulator import UserAgent
from resources_servers.swe_together.task import SOURCE_REVISION, TaskData, load_task
from resources_servers.swe_together.verifier_fixture import VERIFIER_FIXTURE as VERIFIER_FIXTURE


class SWETConfig(BaseResourcesServerConfig):
    asset_root: str
    artifact_root: str = "artifacts/swe_together"
    simulator_model: ModelServerRef
    interaction_model: ModelServerRef
    judge_agent: AgentServerRef
    sandbox_provider: str = "sandbox"
    sandbox_config: dict = Field(default_factory=dict)
    judge_sandbox_config: dict = Field(default_factory=dict)
    network_qualification: dict = Field(default_factory=dict)
    simulator_temperature: float = 0.5
    user_context_chars: int = 3000
    max_resumes: int = Field(default=15, ge=0, le=15)
    scoring_profile: Literal["reference", "judge_all"] = "reference"
    protocol_profile: Literal["reference", "smoke"] = "reference"
    close_timeout: float = 90
    trial_budget_seconds: float = 5400
    python_runtime_url: str | None = None
    python_runtime_sha256: str | None = None


class SWETVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    judge_score: float | None = None
    task_id: str


@dataclass
class Session:
    request: ResourcesSeedSessionRequest
    task: TaskData
    task_dir: Path
    directory: Path
    cookies: dict[str, str]
    sandbox: AsyncSandbox | None = None
    judge_sandbox: AsyncSandbox | None = None
    creations: dict[str, asyncio.Task] = field(default_factory=dict)
    seed: asyncio.Task | None = None
    closing: bool = False
    simulator: UserAgent | None = None
    simulator_model: AuxiliaryModel | None = None
    snapshots: RepositorySnapshots | None = None
    prompt: str = ""
    raw_history: list[str] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    noops: int = 0
    stopped: bool = False
    started_at: float = field(default_factory=monotonic)
    steps: OrderedOperationLedger = field(default_factory=OrderedOperationLedger)
    verification: asyncio.Task | None = None
    verification_body: object = None


class SWETResourcesServer(SimpleResourcesServer):
    config: SWETConfig
    _sessions: dict[str, Session] = PrivateAttr(default_factory=dict)
    _locks: dict[str, asyncio.Lock] = PrivateAttr(default_factory=dict)
    _closed: dict = PrivateAttr(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/step")(self.step)
        return app

    async def _new_sandbox(self, state: Session, *, judge: bool = False) -> AsyncSandbox:
        sandbox_config = self.config.sandbox_config | (self.config.judge_sandbox_config if judge else {})
        options = sandbox_config.get("provider_options", {})
        if self.config.protocol_profile == "reference" and not judge:
            evidence = self.config.network_qualification
            required = ["external_enforcement", "model_pinned_route", "task_package_denial", "bypass_preflight_passed"]
            if (
                not all(evidence.get(key) is True for key in required)
                or not evidence.get("policy_digest")
                or not evidence.get("evidence_artifact")
            ):
                raise ValueError("Reference profile requires verified network policy and preflight evidence")
            policy = options.get("network_policy", {})
            if policy.get("defaultAction", policy.get("default_action")) != "deny":
                raise ValueError("Reference profile requires externally enforced default-deny network policy")
        provider = resolve_provider_config(self.config.sandbox_provider, self.server_client.global_config_dict)
        sandbox = AsyncSandbox(provider)
        if judge:
            state.judge_sandbox = sandbox
        else:
            state.sandbox = sandbox
        metadata = load_task(state.task_dir, state.task)
        memory = 8192 if metadata["environment"].get("memory") == "8G" else 4096
        image = state.task.image.split("@")[0] + "@" + state.task.image_digest
        role = "judge" if judge else "candidate"
        creation = asyncio.create_task(
            sandbox.start(
                SandboxSpec(
                    image=image,
                    workdir=state.task.workdir,
                    ttl_s=sandbox_config.get("ttl_s", 7200),
                    ready_timeout_s=sandbox_config.get("ready_timeout_s", 300),
                    resources=SandboxResources(cpu=4, memory_mib=memory, disk_gib=10),
                    provider_options=options,
                    metadata={
                        "benchmark": "swe-together",
                        "task_id": state.task.task_id,
                        "role": role,
                        "episode_id": state.request.episode_id.capture_key,
                        "resources_session_id": state.request.resources_session_id,
                    },
                )
            )
        )
        state.creations[role] = creation
        # Provider creates can complete after transport cancellation. Retain the
        # task and shield it so close can wait for the actual owned handle.
        await asyncio.shield(creation)
        return sandbox

    async def _stop_owned_sandbox(self, state: Session, role: str) -> None:
        sandbox = state.judge_sandbox if role == "judge" else state.sandbox
        if sandbox is None:
            return
        receipt = {
            "episode_id": state.request.episode_id.capture_key,
            "resources_session_id": state.request.resources_session_id,
            "role": role,
            "cleanup_confirmed": False,
        }
        state.directory.mkdir(parents=True, exist_ok=True)
        try:
            async with asyncio.timeout(self.config.close_timeout):
                if creation := state.creations.get(role):
                    await asyncio.shield(creation)
                receipt["descriptor"] = await sandbox.serialize()
                await sandbox.stop()
            receipt["cleanup_confirmed"] = True
        except BaseException as error:
            receipt["error"] = str(error) or type(error).__name__
            raise
        finally:
            (state.directory / f"{role}-sandbox-close.json").write_text(json.dumps(receipt, indent=2))

    async def _prepare(self, state: Session) -> InteractiveResourcesSeedResponse:
        metadata = load_task(state.task_dir, state.task)
        sandbox = await self._new_sandbox(state)
        state.directory.mkdir(parents=True, exist_ok=True)
        python = await ensure_python(
            sandbox, runtime_url=self.config.python_runtime_url, runtime_sha256=self.config.python_runtime_sha256
        )
        probe = await sandbox.exec(
            'id; printf "HOME=%s\\nPATH=%s\\n" "$HOME" "$PATH"; pwd; command -v git', timeout_s=30
        )
        if probe.return_code:
            raise RuntimeError("Task sandbox is missing Python or Git")
        (state.directory / "preflight.txt").write_text((probe.stdout or "") + (probe.stderr or ""))
        state.snapshots = RepositorySnapshots(state.directory / "patches", python_executable=python)
        await state.snapshots.capture(sandbox)
        # Pinned upstream images already remove unavailable future history. Some
        # deliberately add task branches afterward; preserve that prepared state.
        guidance = await discover_repo_config_files(sandbox)
        state.prompt = (state.task_dir / "instruction.md").read_text()
        if guidance:
            state.prompt += "\n\n" + guidance
        state.prompt += INCREMENTAL_NOTICE
        analysis, messages = session_analysis(state.task_dir)
        state.simulator_model = AuxiliaryModel(
            self.server_client,
            self.config.simulator_model.name,
            cookies=state.cookies,
            temperature=self.config.simulator_temperature,
        )
        state.simulator = UserAgent(
            llm=state.simulator_model, original_user_messages=messages, session_analysis=analysis, max_messages=None
        )
        result = InteractiveResourcesSeedResponse(
            resources_session_id=state.request.resources_session_id,
            runtime_policy=(
                {"format": "harbor.agent-kwargs.v1", "settings": metadata["agent_kwargs"]}
                if metadata["agent_kwargs"]
                else None
            ),
            sandbox_access=SandboxAccess(
                connection=DirectSandboxConnection(
                    provider_config_ref=self.config.sandbox_provider, descriptor=await sandbox.serialize()
                ),
                workdir=state.task.workdir,
            ),
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": state.prompt}]
            ),
        )
        (state.directory / "provenance.json").write_text(
            json.dumps(
                {
                    "source_revision": SOURCE_REVISION,
                    "python_executable": python,
                    "python_runtime_sha256": self.config.python_runtime_sha256,
                    "history_policy": metadata["record"]["history_policy"],
                    "agent_kwargs": metadata["agent_kwargs"],
                    "task": state.task.model_dump(),
                    "protocol_profile": self.config.protocol_profile,
                    "network_qualification": self.config.network_qualification,
                    "scoring_profile": self.config.scoring_profile,
                    "simulator_model": self.config.simulator_model.name,
                    "judge_agent": self.config.judge_agent.name,
                },
                indent=2,
            )
        )
        return result

    async def seed_session(
        self, request: Request, body: ResourcesSeedSessionRequest
    ) -> InteractiveResourcesSeedResponse:
        session_id = body.resources_session_id
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            if session_id in self._closed:
                raise HTTPException(409, "Resources session is closed")
            state = self._sessions.get(session_id)
            if state and (state.request != body or state.closing):
                raise HTTPException(409, "Resources session binding mismatch or closing")
            if state is None:
                task = TaskData.model_validate(body.task_data)
                if task.task_id != body.task_id.task_id:
                    raise ValueError("Task identity mismatch")
                directory = Path(self.config.artifact_root) / body.episode_id.capture_key
                state = Session(
                    body, task, Path(self.config.asset_root) / task.task_id, directory, dict(request.cookies)
                )
                self._sessions[session_id] = state
                state.seed = asyncio.create_task(self._prepare(state))
            request.session[SESSION_ID_KEY] = session_id
            seed = state.seed
        return await asyncio.shield(seed)

    def _get(self, request: Request, session_id: str, episode) -> Session:
        state = self._sessions.get(session_id)
        if not state or state.closing or state.request.episode_id != episode:
            raise HTTPException(409, "Unknown or closed Resources session")
        if request.session.get(SESSION_ID_KEY) != session_id:
            raise HTTPException(409, "Resources cookie mismatch")
        return state

    async def step(self, request: Request, body: ResourcesStepRequest) -> ResourcesStepResponse:
        state = self._get(request, body.resources_session_id, body.episode_id)
        return await state.steps.execute(
            index=body.activation.activation_id, request=body, operation=lambda: self._step(state, body)
        )

    async def _step(self, state: Session, body: ResourcesStepRequest) -> ResourcesStepResponse:
        if state.stopped:
            raise HTTPException(409, "Episode has already stopped")
        activation = body.activation
        turn = activation.activation_id
        if turn >= self.config.max_resumes:
            state.stopped = True
            return ResourcesStepResponse(activation_id=turn, continue_episode=False, stop_reason="max_resumes")
        diff = await state.snapshots.capture(state.sandbox, turn)
        if (
            activation.stop_reason in {"session_budget_exhausted", "budget_exhausted", "model_budget_exhausted"}
            or monotonic() - state.started_at >= self.config.trial_budget_seconds
        ):
            state.stopped = True
            return ResourcesStepResponse(
                activation_id=turn, continue_episode=False, stop_reason="session_budget_exhausted"
            )
        if activation.observation.raw_log is not None:
            state.raw_history.append(activation.observation.raw_log)
        if not activation.turn_complete and activation.stop_reason in {
            "timeout",
            "wall_time_limit",
            "execution_timeout",
            "activation_timeout",
        }:
            return ResourcesStepResponse(
                activation_id=turn,
                continue_episode=True,
                synthetic=True,
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                    input=[
                        {
                            "role": "user",
                            "content": "Your previous run was interrupted. Please continue with the task from where you left off.",
                        }
                    ]
                ),
                metadata={"cap_rescue": True},
            )
        activity, report = project_turn(
            activation.observation, raw_history=state.raw_history, context_chars=self.config.user_context_chars
        )
        try:
            decision = await state.simulator.process(
                task_description=state.prompt,
                recent_trajectory=activity,
                latest_observation=report,
                latest_analysis=None,
                step_count=turn + 1,
                is_completion_attempt=activation.turn_complete,
                elapsed_sec=activation.observation.elapsed_seconds,
                turn_duration_sec=activation.observation.duration_seconds,
                code_changes_diff=diff,
            )
        except BaseException as error:
            evidence = {
                "turn": turn + 1,
                "error": str(error) or type(error).__name__,
                "simulator_messages": state.simulator.last_messages_sent,
                "source_observation": activation.observation.model_dump(mode="json"),
            }
            (state.directory / f"turn-{turn}-simulator.json").write_text(json.dumps(evidence, indent=2))
            (state.directory / "simulator-model-calls.json").write_text(
                json.dumps(state.simulator_model.calls, indent=2)
            )
            raise
        evidence = {
            "turn": turn + 1,
            "action": decision.action,
            "content": decision.content,
            "raw_response": decision.raw_response,
            "has_message": decision.has_message,
            "simulator_messages": state.simulator.last_messages_sent,
            "source_observation": activation.observation.model_dump(mode="json"),
        }
        (state.directory / f"turn-{turn}-simulator.json").write_text(json.dumps(evidence, indent=2))
        if decision.has_message:
            state.noops = 0
            state.simulator.advance_original_index()
            state.messages.append(
                {
                    "trial_idx": len(state.messages) + 1,
                    "turn": turn + 1,
                    "action": decision.action,
                    "text": decision.content,
                }
            )
            message = decision.content
        else:
            state.noops += 1
            message = "continue"
        if state.noops >= 4:
            state.stopped = True
            return ResourcesStepResponse(
                activation_id=turn,
                continue_episode=False,
                stop_reason="consecutive_noops",
                metadata={"decision": decision.action},
            )
        return ResourcesStepResponse(
            activation_id=turn,
            continue_episode=True,
            synthetic=not decision.has_message,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": message}]
            ),
            metadata={"decision": decision.action, "original_cursor": state.simulator._cursor},
        )

    async def verify(
        self, request: Request, body: ResourcesVerifyRequest[InteractiveVerificationInput]
    ) -> SWETVerifyResponse:
        data = body.verification_input
        state = self._get(request, data.resources_session_id, body.episode_id)
        if state.request.task_id != body.task_id or not data.agent_close.cleanup_confirmed:
            raise HTTPException(409, "Candidate cleanup is not confirmed")
        if not data.activations:
            raise ValueError("No candidate activation to verify")
        async with self._locks[data.resources_session_id]:
            if state.verification_body is not None and state.verification_body != body:
                raise HTTPException(409, "Different verification already submitted")
            if state.verification is None:
                state.verification_body = body.model_copy(deep=True)
                state.verification = asyncio.create_task(self._verify(state, body))
            verification = state.verification
        return await asyncio.shield(verification)

    async def _verify(
        self, state: Session, body: ResourcesVerifyRequest[InteractiveVerificationInput]
    ) -> SWETVerifyResponse:
        data = body.verification_input
        result = {
            "responses_create_params": data.responses_create_params,
            "response": data.activations[-1].response,
            "reward": 0.0,
            "judge_score": None,
            "task_id": state.task.task_id,
            "scoring_profile": self.config.scoring_profile,
            "protocol_profile": self.config.protocol_profile,
        }
        try:
            await state.snapshots.capture(state.sandbox, len(data.activations))
            patch = state.snapshots.final_patch
            if self.config.scoring_profile == "reference" and (
                len(patch.strip()) < 100 or not apply_candidates(patch)
            ):
                raise ValueError("Reference scorer skips empty or shorter-than-100-character patches")
            sandbox = await self._new_sandbox(state, judge=True)
            metadata = load_task(state.task_dir, state.task)
            verdict = await run_judge(
                client=self.server_client,
                agent=self.config.judge_agent,
                sandbox=sandbox,
                provider=self.config.sandbox_provider,
                task_dir=state.task_dir,
                patch=patch,
                workdir=state.task.workdir,
                episode=body.episode_id,
                task_id=body.task_id,
                timeout=metadata["judge_timeout"],
                cookies=state.cookies,
                artifact_dir=state.directory,
            )
            result.update(
                verdict=verdict, judge_score=verdict["judge_score"], reward=float(verdict["judge_score"] >= 0.85)
            )
        except Exception as error:
            result.update(mask_sample=True, failure_kind="swe_together:judge_failed", failure_reason=str(error))
        finally:
            if state.judge_sandbox:
                await self._stop_owned_sandbox(state, "judge")
                state.judge_sandbox = None
        auxiliary = AuxiliaryModel(
            self.server_client, self.config.interaction_model.name, cookies=state.cookies, temperature=0
        )
        result.update(
            await interaction_metrics(
                auxiliary, task_dir=state.task_dir, messages=state.messages, artifact_dir=state.directory
            )
        )
        (state.directory / "auxiliary-model-calls.json").write_text(
            json.dumps({"simulator": state.simulator_model.calls, "interaction": auxiliary.calls}, indent=2)
        )
        verdict = SWETVerifyResponse.model_validate(result)
        (state.directory / "verification.json").write_text(verdict.model_dump_json(indent=2))
        return verdict

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        async with self._locks.setdefault(body.resources_session_id, asyncio.Lock()):
            if body.resources_session_id in self._closed:
                if self._closed[body.resources_session_id] != body.episode_id:
                    raise HTTPException(409, "Closed episode identity mismatch")
                return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)
            state = self._sessions.get(body.resources_session_id)
            if state:
                if state.request.episode_id != body.episode_id:
                    raise HTTPException(409, "Episode identity mismatch")
                state.closing = True
                await state.steps.close(timeout=self.config.close_timeout)
                for task in [state.seed, state.verification]:
                    if task is not None and not task.done():
                        task.cancel()
                        try:
                            await asyncio.wait_for(asyncio.shield(task), self.config.close_timeout)
                        except asyncio.CancelledError:
                            if not task.cancelled():
                                raise
                for role in ["judge", "candidate"]:
                    await self._stop_owned_sandbox(state, role)
                self._sessions.pop(body.resources_session_id)
            self._closed[body.resources_session_id] = body.episode_id
            return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    async def shutdown(self) -> None:
        for state in list(self._sessions.values()):
            await self.close_resources_session(
                None,
                ResourcesCloseSessionRequest(
                    resources_session_id=state.request.resources_session_id, episode_id=state.request.episode_id
                ),
            )


if __name__ == "__main__":
    SWETResourcesServer.run_webserver()
