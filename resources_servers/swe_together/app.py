# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SWE-Together Resources: task ownership, interactive user, and fixed evaluation."""

import asyncio
import json
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.agent_utils.ordered_operations import OrderedOperationLedger
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesVerifyRequest,
    SimpleResourcesServer,
)
from nemo_gym.config_types import AgentServerRef, ModelServerRef
from nemo_gym.failure_kinds import JUDGE_FAILED
from nemo_gym.interactive_agent_types import (
    InteractionBudget,
    InteractiveResourcesSeedRequest,
    InteractiveResourcesSeedResponse,
    InteractiveVerificationInput,
    NextUserTurnRequest,
    NextUserTurnResponse,
)
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from resources_servers.swe_together.artifacts import RepositorySnapshots
from resources_servers.swe_together.context import INCREMENTAL_NOTICE, project_turn
from resources_servers.swe_together.evaluation import PASS_THRESHOLD, interaction_metrics, run_judge
from resources_servers.swe_together.guidance import session_analysis
from resources_servers.swe_together.metrics import aggregate
from resources_servers.swe_together.model_client import AuxiliaryModel
from resources_servers.swe_together.patch_normalize import apply_candidates
from resources_servers.swe_together.repo_config import discover_repo_config_files
from resources_servers.swe_together.runtime import PYTHON, RuntimeProvisioner
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
    runtime_manifest_path: str
    runtime_helper_image: str = "python:3.13-slim"
    planned_trials: int = Field(default=218, gt=0)
    repeats: int = Field(default=2, gt=0)
    simulator_temperature: float = 0.5
    user_context_chars: int = 3000
    max_resumes: int = Field(default=15, ge=0, le=15)
    scoring_profile: Literal["reference", "judge_all"] = "reference"
    protocol_profile: Literal["reference", "smoke"] = "reference"
    close_timeout: float = 90


class SWETVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    judge_score: float | None = None
    task_id: str


@dataclass
class Session:
    request: InteractiveResourcesSeedRequest
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
    budget_bound: bool = False
    task_metadata: dict | None = None
    runtime: RuntimeProvisioner | None = None
    interaction_budget: InteractionBudget | None = None
    steps: OrderedOperationLedger = field(default_factory=OrderedOperationLedger)
    verification: asyncio.Task | None = None
    verification_body: object = None


class SWETResourcesServer(SimpleResourcesServer):
    config: SWETConfig
    _sessions: dict[str, Session] = PrivateAttr(default_factory=dict)
    _locks: dict[str, asyncio.Lock] = PrivateAttr(default_factory=dict)
    _closed: dict = PrivateAttr(default_factory=dict)
    _runtime: RuntimeProvisioner | None = PrivateAttr(default=None)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as maybe_state:
                    yield maybe_state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        app.post("/next-user-turn")(self.next_user_turn)
        return app

    async def _new_sandbox(self, state: Session, *, judge: bool = False) -> AsyncSandbox:
        sandbox_config = self.config.sandbox_config | (self.config.judge_sandbox_config if judge else {})
        options = dict(sandbox_config.get("provider_options", {}))
        policy = options.get("network_policy", {})
        if policy.get("defaultAction", policy.get("default_action")) != "deny":
            raise ValueError("Task sandboxes require externally enforced default-deny network policy")
        role = "judge" if judge else "candidate"
        if state.runtime is None:
            raise RuntimeError("Trusted runtime must be prepared before task sandbox creation")
        mount = state.runtime.mount_options(state.request.resources_session_id, role)
        options["volumes"] = list(options.get("volumes", [])) + mount["volumes"]
        provider = resolve_provider_config(self.config.sandbox_provider, self.server_client.global_config_dict)
        sandbox = AsyncSandbox(provider)
        if judge:
            state.judge_sandbox = sandbox
        else:
            state.sandbox = sandbox
        metadata = state.task_metadata
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
                    metadata=state.runtime.metadata()
                    | {
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
                    try:
                        await asyncio.shield(creation)
                    except Exception as error:
                        # A provider can allocate a handle before readiness fails.
                        # Failed creation still requires a stop attempt.
                        receipt["creation_error"] = str(error)
                try:
                    receipt["descriptor"] = await sandbox.serialize()
                except RuntimeError:
                    pass
                await sandbox.stop()
            receipt["cleanup_confirmed"] = True
        except BaseException as error:
            receipt["error"] = str(error) or type(error).__name__
            raise
        finally:
            (state.directory / f"{role}-sandbox-close.json").write_text(json.dumps(receipt, indent=2))

    async def _prepare(self, state: Session) -> InteractiveResourcesSeedResponse:
        if state.task_metadata is None:
            state.task_metadata = load_task(state.task_dir, state.task)
        metadata = state.task_metadata
        state.directory.mkdir(parents=True, exist_ok=True)
        if self._runtime is None:
            provider = resolve_provider_config(self.config.sandbox_provider, self.server_client.global_config_dict)
            self._runtime = RuntimeProvisioner(
                provider_config=provider,
                manifest_path=Path(self.config.runtime_manifest_path),
                helper_image=self.config.runtime_helper_image,
            )
        state.runtime = self._runtime
        await state.runtime.prepare(state.request.resources_session_id)
        sandbox = await self._new_sandbox(state)
        python = PYTHON
        python_probe = await sandbox.exec(python + " --version", timeout_s=30)
        if python_probe.return_code or python_probe.error_type:
            raise RuntimeError("Full Python runtime was not mounted before task setup")
        probe = await sandbox.exec(
            'id; printf "HOME=%s\\nPATH=%s\\n" "$HOME" "$PATH"; pwd; command -v git', timeout_s=30
        )
        if probe.return_code:
            raise RuntimeError("Task sandbox is missing Python or Git")
        (state.directory / "preflight.txt").write_text((probe.stdout or "") + (probe.stderr or ""))
        state.snapshots = RepositorySnapshots(
            state.directory / "patches", python_executable=python, workdir=state.task.workdir
        )
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
            supports_interaction_budget=True,
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
                    "runtime_sha256": {
                        name: item["sha256"] for name, item in state.runtime.manifest["artifacts"].items()
                    },
                    "history_policy": metadata["record"]["history_policy"],
                    "agent_kwargs": metadata["agent_kwargs"],
                    "task": state.task.model_dump(),
                    "protocol_profile": self.config.protocol_profile,
                    "scoring_profile": self.config.scoring_profile,
                    "simulator_model": self.config.simulator_model.name,
                    "judge_agent": self.config.judge_agent.name,
                },
                indent=2,
            )
        )
        return result

    async def seed_session(
        self, request: Request, body: InteractiveResourcesSeedRequest
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
                state.seed = asyncio.create_task(self._prepare_with_deadline(state))
            request.session[SESSION_ID_KEY] = session_id
            seed = state.seed
        return await asyncio.shield(seed)

    async def _prepare_with_deadline(self, state: Session) -> InteractiveResourcesSeedResponse:
        deadline = state.request.setup_deadline_unix_seconds
        if deadline is not None and deadline <= time():
            raise TimeoutError("Resources setup deadline expired")
        async with asyncio.timeout_at(
            asyncio.get_running_loop().time() + max(0, deadline - time()) if deadline else None
        ):
            return await self._prepare(state)

    def _get(self, request: Request, session_id: str, episode) -> Session:
        state = self._sessions.get(session_id)
        if not state or state.closing or state.request.episode_id != episode:
            raise HTTPException(409, "Unknown or closed Resources session")
        if request.session.get(SESSION_ID_KEY) != session_id:
            raise HTTPException(409, "Resources cookie mismatch")
        return state

    async def next_user_turn(self, request: Request, body: NextUserTurnRequest) -> NextUserTurnResponse:
        state = self._get(request, body.resources_session_id, body.episode_id)
        if state.budget_bound and body.interaction_budget != state.interaction_budget:
            raise HTTPException(409, "Interaction budget changed during the Resources session")
        return await state.steps.execute(
            index=body.activation.activation_id, request=body, operation=lambda: self._next_user_turn(state, body)
        )

    async def _next_user_turn(self, state: Session, body: NextUserTurnRequest) -> NextUserTurnResponse:
        if state.stopped:
            raise HTTPException(409, "Episode has already stopped")
        activation = body.activation
        turn = activation.activation_id
        if not state.budget_bound:
            state.interaction_budget = body.interaction_budget
            if state.interaction_budget:
                (state.directory / "interaction-budget.json").write_text(
                    state.interaction_budget.model_dump_json(indent=2)
                )
            state.budget_bound = True
        elif body.interaction_budget != state.interaction_budget:
            raise HTTPException(409, "Interaction budget changed during the Resources session")
        if turn >= self.config.max_resumes:
            state.stopped = True
            return NextUserTurnResponse(activation_id=turn, kind="stop", stop_reason="max_resumes")
        try:
            async with asyncio.timeout(max(0, self._remaining_interaction_seconds(state))):
                if self._remaining_interaction_seconds(state) <= 0:
                    raise TimeoutError("Interaction deadline expired before snapshot")
                diff = await state.snapshots.capture(state.sandbox, turn)
        except TimeoutError:
            state.stopped = True
            return NextUserTurnResponse(activation_id=turn, kind="stop", stop_reason="session_budget_exhausted")
        if (
            activation.stop_reason in {"session_budget_exhausted", "model_budget_exhausted"}
            or self._remaining_interaction_seconds(state) <= 0
        ):
            state.stopped = True
            return NextUserTurnResponse(activation_id=turn, kind="stop", stop_reason="session_budget_exhausted")
        if activation.observation.raw_log is not None:
            state.raw_history.append(activation.observation.raw_log)
        if not activation.turn_complete and activation.stop_reason == "activation_timeout":
            return NextUserTurnResponse(
                activation_id=turn,
                kind="continue",
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
        # Keep a proposed turn isolated until it can be accepted before the
        # deadline. The auxiliary client stays shared so canceled calls retain
        # their diagnostics, while late simulator history/counters are discarded.
        simulator = deepcopy(state.simulator, {id(state.simulator._llm): state.simulator._llm})
        timer = asyncio.timeout(max(0.0, self._remaining_interaction_seconds(state)))
        try:
            async with timer:
                decision = await simulator.process(
                    task_description=state.prompt,
                    recent_trajectory=activity,
                    latest_observation=report,
                    latest_analysis=None,
                    step_count=turn + 1,
                    # The source wrapper consults the simulator after an exit-zero
                    # permission rejection, then resumes the same conversation.
                    is_completion_attempt=activation.turn_complete or activation.stop_reason == "permission_denied",
                    elapsed_sec=activation.observation.elapsed_seconds,
                    turn_duration_sec=activation.observation.duration_seconds,
                    code_changes_diff=diff,
                )
        except BaseException as error:
            if isinstance(error, TimeoutError) and timer.expired():
                return self._expired_simulator_turn(state, body, simulator)
            evidence = {
                "turn": turn + 1,
                "error": str(error) or type(error).__name__,
                "simulator_messages": simulator.last_messages_sent,
                "source_observation": activation.observation.model_dump(mode="json"),
            }
            (state.directory / f"turn-{turn}-simulator.json").write_text(json.dumps(evidence, indent=2))
            (state.directory / "simulator-model-calls.json").write_text(
                json.dumps(state.simulator_model.calls, indent=2)
            )
            raise
        accepted_at = time()
        if timer.expired() or self._remaining_interaction_seconds(state) <= 0:
            return self._expired_simulator_turn(state, body, simulator)
        state.simulator = simulator
        evidence = {
            "turn": turn + 1,
            "accepted_at_unix_seconds": accepted_at,
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
            return NextUserTurnResponse(
                activation_id=turn,
                kind="stop",
                stop_reason="consecutive_noops",
                metadata={"decision": decision.action},
            )
        return NextUserTurnResponse(
            activation_id=turn,
            kind="continue",
            synthetic=not decision.has_message,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": message}]
            ),
            metadata={"decision": decision.action, "original_cursor": state.simulator._cursor},
        )

    def _remaining_interaction_seconds(self, state: Session) -> float:
        return state.interaction_budget.deadline_unix_seconds - time() if state.interaction_budget else float("inf")

    def _expired_simulator_turn(
        self, state: Session, body: NextUserTurnRequest, simulator: UserAgent
    ) -> NextUserTurnResponse:
        state.stopped = True
        evidence = {
            "turn": body.activation.activation_id + 1,
            "discarded_at_unix_seconds": time(),
            "stop_reason": "session_budget_exhausted",
            "discarded": True,
            "simulator_messages": simulator.last_messages_sent,
            "source_observation": body.activation.observation.model_dump(mode="json"),
        }
        (state.directory / f"turn-{body.activation.activation_id}-simulator.json").write_text(
            json.dumps(evidence, indent=2)
        )
        (state.directory / "simulator-model-calls.json").write_text(json.dumps(state.simulator_model.calls, indent=2))
        return NextUserTurnResponse(
            activation_id=body.activation.activation_id,
            kind="stop",
            stop_reason="session_budget_exhausted",
            metadata={"simulator_turn_discarded": True},
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
                state.verification = asyncio.create_task(self._verify_with_deadline(state, body))
            verification = state.verification
        return await asyncio.shield(verification)

    async def _verify_with_deadline(self, state: Session, body) -> SWETVerifyResponse:
        deadline = body.verification_input.grading_deadline_unix_seconds
        if deadline is not None and deadline <= time():
            raise TimeoutError("Resources grading deadline expired")
        async with asyncio.timeout_at(
            asyncio.get_running_loop().time() + max(0, deadline - time()) if deadline else None
        ):
            return await self._verify(state, body)

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
                len(patch.strip()) < 100 or not apply_candidates(patch, workdir=state.task.workdir)
            ):
                raise ValueError("Reference scorer skips empty or shorter-than-100-character patches")
            sandbox = await self._new_sandbox(state, judge=True)
            metadata = state.task_metadata
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
                deadline=data.grading_deadline_unix_seconds,
            )
            result.update(
                verdict=verdict,
                judge_score=verdict["judge_score"],
                reward=float(verdict["judge_score"] >= PASS_THRESHOLD),
                judge_close=verdict["judge_close"],
            )
        except Exception as error:
            result.update(
                mask_sample=True,
                failure_kind=JUDGE_FAILED,
                failure_reason=str(error),
            )
        finally:
            if state.judge_sandbox:
                await self._stop_owned_sandbox(state, "judge")
                state.judge_sandbox = None
                state.creations.pop("judge", None)
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
                errors = []
                try:
                    await state.steps.close(timeout=self.config.close_timeout)
                except Exception as error:
                    errors.append(error)
                for task in [state.seed, state.verification]:
                    if task is not None and not task.done():
                        task.cancel()
                        try:
                            await asyncio.wait_for(asyncio.shield(task), self.config.close_timeout)
                        except asyncio.CancelledError:
                            if not task.cancelled():
                                raise
                        except Exception as error:
                            errors.append(error)
                for role in ["judge", "candidate"]:
                    try:
                        await self._stop_owned_sandbox(state, role)
                        if role == "judge":
                            state.judge_sandbox = None
                        else:
                            state.sandbox = None
                        state.creations.pop(role, None)
                    except Exception as error:
                        errors.append(error)
                if errors:
                    raise ExceptionGroup("Task sandbox cleanup remains unconfirmed", errors)
                if state.runtime:
                    await state.runtime.release(body.resources_session_id)
                self._sessions.pop(body.resources_session_id)
            self._closed[body.resources_session_id] = body.episode_id
            return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    def compute_metrics(self, tasks: list[list[dict]]) -> dict:
        records = [record for task in tasks for record in task]
        return super().compute_metrics(tasks) | aggregate(
            records, planned=self.config.planned_trials, repeats=self.config.repeats
        )

    def get_key_metrics(self, agent_metrics: dict) -> dict:
        names = ("MeanJudge", "pass@1", "stable_solve_rate", "pass2", "graded", "masked", "missing")
        return {name: agent_metrics.get(name) for name in names}

    async def shutdown(self) -> None:
        errors = []
        for state in list(self._sessions.values()):
            try:
                await self.close_resources_session(
                    None,
                    ResourcesCloseSessionRequest(
                        resources_session_id=state.request.resources_session_id, episode_id=state.request.episode_id
                    ),
                )
            except Exception as error:
                errors.append(error)
        if self._runtime and not self._sessions:
            try:
                async with asyncio.timeout(self.config.close_timeout):
                    await self._runtime.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Resources shutdown remains unconfirmed", errors)


if __name__ == "__main__":
    SWETResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    from nemo_gym.server_utils import create_nemo_gym_fastapi_app

    app = create_nemo_gym_fastapi_app(SWETResourcesServer)
