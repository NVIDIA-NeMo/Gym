# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SWE-Together owns task setup, interaction, frozen-state grading and teardown."""

import asyncio
import json
import logging
import math
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic, time
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI
from pydantic import Field, PrivateAttr

from environment_servers.swe_together.artifacts import RepositorySnapshots
from environment_servers.swe_together.context import INCREMENTAL_NOTICE, project_turn
from environment_servers.swe_together.egress import DependencyAccess, install_trust_bundle
from environment_servers.swe_together.evaluation import PASS_THRESHOLD, interaction_metrics, run_judge
from environment_servers.swe_together.guidance import session_analysis
from environment_servers.swe_together.metrics import aggregate
from environment_servers.swe_together.model_client import AuxiliaryModel, error_evidence
from environment_servers.swe_together.repo_config import discover_repo_config_files
from environment_servers.swe_together.runtime import PYTHON, RuntimeProvisioner
from environment_servers.swe_together.schemas import (
    CandidateTurn,
    SWETFailure,
    SWETRequest,
    SWETResponse,
    SWETResult,
    UserTurn,
    episode_response,
)
from environment_servers.swe_together.simulator import UserAgent
from environment_servers.swe_together.submission import replicate_submission
from environment_servers.swe_together.task import SOURCE_REVISION, TaskData, load_task
from nemo_gym.agents import (
    AGENT_ACTIVATION_HEADER,
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentContinuationRequirements,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import (
    TOKEN_CAPTURE_PATH_SEGMENT,
    TOKEN_ID_CAPTURE_BLOCK,
    AgentServerRef,
    AggregateMetrics,
    AggregateMetricsRequest,
    ModelServerRef,
    get_first_server_config_dict,
)
from nemo_gym.core import JUDGE_FAILED, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.metrics import compute_aggregate_metrics
from nemo_gym.sandbox import (
    AsyncSandbox,
    DirectSandboxConnection,
    SandboxAccess,
    SandboxResources,
    SandboxSpec,
    resolve_provider_config,
)
from nemo_gym.server_runtime import (
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    CleanupContext,
    EpisodeFailure,
    HandledEpisodeError,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


LOGGER = logging.getLogger(__name__)


class SWETConfig(BaseEnvironmentServerConfig):
    """Models, pinned assets and budgets for one benchmark implementation."""

    agent_server: AgentServerRef
    judge_agent: AgentServerRef
    simulator_model: ModelServerRef
    interaction_model: ModelServerRef
    asset_root: str
    artifact_root: str = "artifacts/swe_together"
    sandbox_provider: str = "sandbox"
    sandbox_config: dict = Field(default_factory=dict)
    judge_sandbox_config: dict = Field(default_factory=dict)
    runtime_manifest_path: str
    dependency_proxy_url: str | None = None
    runtime_helper_image: str = "python:3.13-slim"
    planned_trials: int = Field(default=218, gt=0)
    repeats: int = Field(default=2, gt=0)
    simulator_temperature: float = 0.5
    user_context_chars: int = Field(default=3000, gt=0)
    max_resumes: int = Field(default=15, ge=0, le=15)
    scoring_profile: Literal["reference", "judge_all"] = "reference"
    protocol_profile: Literal["reference", "smoke"] = "reference"
    setup_timeout_seconds: float = Field(default=1800, gt=0, allow_inf_nan=False)
    activation_timeout_seconds: float = Field(default=1800, gt=0, allow_inf_nan=False)
    interaction_timeout_seconds: float = Field(default=4800, gt=0, allow_inf_nan=False)
    grading_timeout_seconds: float = Field(default=1800, gt=0, allow_inf_nan=False)
    cleanup_timeout_seconds: float = Field(default=300, gt=0, allow_inf_nan=False)


def memory_mib(value: str) -> int:
    """Parse upstream binary memory declarations without silently substituting 4 GiB."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT])(?:i?B)?\s*", str(value), re.I)
    if not match:
        raise ValueError(f"Unsupported task memory declaration: {value!r}")
    amount, unit = match.groups()
    result = math.ceil(float(amount) * 1024 ** ("KMGT".index(unit.upper()) - 1))
    if result <= 0:
        raise ValueError("Task memory must be positive")
    return result


@dataclass
class Episode:
    request: SWETRequest
    task: TaskData
    task_dir: Path
    directory: Path
    session_id: str = field(default_factory=lambda: "swet-" + uuid4().hex)
    sandbox: AsyncSandbox | None = None
    judge_sandbox: AsyncSandbox | None = None
    creations: dict[str, asyncio.Task] = field(default_factory=dict)
    simulator: UserAgent | None = None
    simulator_model: AuxiliaryModel | None = None
    snapshots: RepositorySnapshots | None = None
    prompt: str = ""
    raw_history: list[str] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    noops: int = 0
    task_metadata: dict | None = None
    runtime: RuntimeProvisioner | None = None
    interaction_started: float | None = None
    interaction_deadline: float | None = None
    cleanup_receipts: dict = field(default_factory=dict)
    dependency_token: str = field(default_factory=lambda: uuid4().hex + uuid4().hex)
    dependency_access: DependencyAccess | None = None


class SWETEnvironmentServer(BaseEnvironmentServer[SWETRequest, SWETResponse]):
    config: SWETConfig
    request_model = SWETRequest
    response_model = SWETResponse
    _runtime: RuntimeProvisioner | None = PrivateAttr(default=None)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        return app

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        result = compute_aggregate_metrics(body.verify_responses)
        benchmark = aggregate(body.verify_responses, planned=self.config.planned_trials, repeats=self.config.repeats)
        result.agent_metrics.update(benchmark)
        result.key_metrics = {
            name: benchmark[name]
            for name in ("MeanJudge", "pass@1", "stable_solve_rate", "pass2", "graded", "masked", "missing")
        }
        return result

    async def run(self, request: SWETRequest, cleanup: CleanupContext) -> SWETResponse:
        task = request.task.task_input.task_data
        if task.task_id != request.task.task_id.task_id:
            raise ValueError("Task identity mismatch")
        state = Episode(
            request,
            task,
            Path(self.config.asset_root) / task.task_id,
            Path(self.config.artifact_root) / request.episode_id.capture_key,
        )
        state.directory.mkdir(parents=True, exist_ok=True)
        cookies = {}
        agent_id = "candidate-" + uuid4().hex
        agent_close = None
        failure: SWETFailure | None = None
        activations = []
        steps = []
        stage = "seed"
        activation_attempted = False

        async def release_runtime() -> None:
            if state.runtime and state.sandbox is None and state.judge_sandbox is None:
                await state.runtime.release(state.session_id)

        async def close_sandbox(role: str) -> None:
            await self._stop_owned_sandbox(state, role)
            if role == "candidate":
                state.sandbox = None
            else:
                state.judge_sandbox = None

        async def close_agent() -> None:
            nonlocal agent_close
            response = await self.server_client.post(
                self.config.agent_server.name,
                "/v1/agent_sessions/close",
                json=AgentCloseSessionRequest(agent_session_id=agent_id, episode_id=request.episode_id),
                cookies=cookies,
                _control=True,
            )
            await raise_for_status(response)
            agent_close = AgentCloseSessionResponse.model_validate(await get_response_json(response))
            state.cleanup_receipts["candidate_agent"] = agent_close.model_dump(mode="json")
            (state.directory / "candidate-agent-close.json").write_text(agent_close.model_dump_json(indent=2))
            if agent_close.agent_session_id != agent_id or not agent_close.cleanup_confirmed:
                raise RuntimeError("Candidate execution stop was not confirmed")

        async def persist_cleanup_evidence() -> None:
            (state.directory / "cleanup-receipts.json").write_text(json.dumps(state.cleanup_receipts, indent=2))
            if failure is not None:
                failure.agent_close = agent_close
                failure.agent_observations = agent_close.agent_observations if agent_close else None
                failure.trajectory = request.trajectory_for_collection(agent_close.trajectory) if agent_close else None
                failure.cleanup_receipts = state.cleanup_receipts
                (state.directory / "failure.json").write_text(failure.model_dump_json(indent=2))

        async def release_dependency_access() -> None:
            if state.dependency_access is not None:
                await state.dependency_access.revoke(state.dependency_token)
                state.cleanup_receipts["dependency_access"] = {"revoked": True}

        # Run after all participant callbacks, including on failure/cancellation.
        cleanup.register_cleanup("episode evidence", persist_cleanup_evidence)
        dependency_cleanup = cleanup.register_cleanup("dependency access", release_dependency_access)
        runtime_cleanup = cleanup.register_cleanup("runtime bundle", release_runtime)
        candidate_cleanup = cleanup.register_cleanup("candidate sandbox", lambda: close_sandbox("candidate"))
        judge_cleanup = cleanup.register_cleanup("judge sandbox", lambda: close_sandbox("judge"))
        agent_cleanup = None
        try:
            async with asyncio.timeout(self.config.setup_timeout_seconds):
                await self._prepare(state)
                stage = "agent"
                agent_cleanup = cleanup.register_cleanup("candidate agent", close_agent)
                required = AgentContinuationRequirements()
                response = await self.server_client.post(
                    self.config.agent_server.name,
                    "/v1/agent_sessions",
                    json=AgentSeedSessionRequest(
                        agent_session_id=agent_id,
                        episode_id=request.episode_id,
                        task_id=request.task.task_id,
                        sandbox_access=SandboxAccess(
                            connection=DirectSandboxConnection(
                                provider_config_ref=self.config.sandbox_provider,
                                descriptor=await state.sandbox.serialize(),
                            ),
                            workdir=task.workdir,
                        ),
                        continuation=required,
                    ),
                    cookies=cookies,
                )
                await raise_for_status(response)
                cookies.update({k: v.value for k, v in response.cookies.items()})
                seeded = AgentSeedSessionResponse.model_validate(await get_response_json(response))
                if seeded.agent_session_id != agent_id:
                    raise ValueError("Candidate returned an unexpected session ID")
                if seeded.capabilities is None or not set(required.observations) <= set(
                    seeded.capabilities.observations
                ):
                    raise ValueError("Candidate did not confirm the required native continuation observations")
                if not cookies:
                    raise ValueError("Candidate seed did not establish a session cookie")
            state.interaction_started = monotonic()
            state.interaction_deadline = state.interaction_started + self.config.interaction_timeout_seconds
            initial = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": state.prompt}])
            next_input = initial
            for activation_id in range(1, self.config.max_resumes + 2):
                if self._remaining_interaction_seconds(state) <= 0:
                    break
                stage = "agent"
                turn_deadline = min(monotonic() + self.config.activation_timeout_seconds, state.interaction_deadline)
                activation_attempted = True
                started = monotonic()
                collection_deadline = min(
                    turn_deadline + self.config.cleanup_timeout_seconds, state.interaction_deadline
                )
                try:
                    async with asyncio.timeout_at(collection_deadline) as timer:
                        response = await self.server_client.post(
                            self.config.agent_server.name,
                            self._activation_path(request),
                            json=next_input.model_copy(
                                update={
                                    "metadata": {"agent_timeout_sec": str(max(0.001, turn_deadline - monotonic()))},
                                }
                            ),
                            headers={AGENT_ACTIVATION_HEADER: str(activation_id)},
                            cookies=cookies,
                        )
                        await raise_for_status(response)
                        cookies.update({k: v.value for k, v in response.cookies.items()})
                        activation = CandidateTurn(
                            activation_id=activation_id,
                            response=NeMoGymResponse.model_validate(await get_response_json(response)),
                            duration_seconds=monotonic() - started,
                        )
                except TimeoutError:
                    if not timer.expired() or self._remaining_interaction_seconds(state) > 0:
                        raise
                    # Canceling an HTTP waiter does not stop a retained activation.
                    # Close fences delayed retries, terminates native work, and
                    # returns evidence before any filesystem can enter grading.
                    stage = "cleanup"
                    await agent_cleanup.close()
                    activations.append(
                        CandidateTurn.from_deadline_close(agent_close, activation_id, monotonic() - started)
                    )
                    steps.append(
                        UserTurn(activation_id=activation_id, kind="stop", stop_reason="session_budget_exhausted")
                    )
                    break
                if activation.activation_id != activation_id:
                    raise ValueError("Candidate returned an unexpected activation ID")
                activations.append(activation)
                if activation.response.status == "failed" or activation.response.error:
                    raise RuntimeError("Candidate execution failed: " + activation.response.model_dump_json())
                stage = "agent"
                step = await self._next_user_turn(state, activation)
                steps.append(step)
                if step.kind == "stop":
                    break
                next_input = step.responses_create_params
            stage = "cleanup"
            await agent_cleanup.close()
            stage = "verification"
            verification = await self._verify(state, initial, activations)
            await judge_cleanup.close()
            await candidate_cleanup.close()
            await runtime_cleanup.close()
            await dependency_cleanup.close()
            response = SWETResponse(
                episode_id=request.episode_id,
                task_id=request.task.task_id,
                result=SWETResult.model_validate(
                    verification
                    | {
                        "ng_activations": activations,
                        "ng_steps": steps,
                        "ng_agent_close": agent_close,
                        "ng_trajectory": request.trajectory_for_collection(agent_close.trajectory),
                        "ng_agent_observations": agent_close.agent_observations,
                        "cleanup_receipts": state.cleanup_receipts,
                        "artifact_directory": str(state.directory),
                    }
                ),
            )
            response.result.cleanup_receipts = state.cleanup_receipts
            return response
        except Exception as error:
            LOGGER.exception("SWE-Together episode failed at %s", stage)
            dependency = error_evidence(error)
            failure = SWETFailure(
                stage=stage,
                terminal=activation_attempted or isinstance(error, (ValueError, FileNotFoundError)),
                failure_reason=(f"{type(error).__name__}: {error}")[:2000],
                partial_response=activations[-1].response if activations else None,
                activations=activations,
                agent_close=agent_close,
                agent_observations=agent_close.agent_observations if agent_close else None,
                trajectory=request.trajectory_for_collection(agent_close.trajectory) if agent_close else None,
                cleanup_receipts=state.cleanup_receipts,
                artifact_directory=str(state.directory),
                dependency_error=dependency,
            )
            (state.directory / "failure.json").write_text(failure.model_dump_json(indent=2))
            failure.cleanup_receipts = state.cleanup_receipts
            raise HandledEpisodeError(failure) from error

    def failure_response(self, request: SWETRequest, failure: EpisodeFailure) -> SWETResponse:
        if isinstance(failure, SWETFailure):
            # Final cleanup updates this same object before the base class serializes it.
            return SWETResponse(episode_id=request.episode_id, task_id=request.task.task_id, failure=failure)
        return super().failure_response(request, failure)

    def _activation_path(self, request: SWETRequest) -> str:
        block = self.server_client.global_config_dict.get(TOKEN_ID_CAPTURE_BLOCK) or {}
        config = get_first_server_config_dict(self.server_client.global_config_dict, self.config.agent_server.name)
        capture = bool(block.get("enabled")) and (block.get("all_agents") or config.get("token_id_capture"))
        segment = f"/{TOKEN_CAPTURE_PATH_SEGMENT}" if capture else ""
        return f"/ng-rollout/{request.episode_id.capture_key}{segment}/v1/responses"

    async def _new_sandbox(self, state: Episode, *, judge: bool = False) -> AsyncSandbox:
        sandbox_config = self.config.sandbox_config | (self.config.judge_sandbox_config if judge else {})
        options = dict(sandbox_config.get("provider_options", {}))
        policy = options.get("network_policy", {})
        if policy.get("defaultAction", policy.get("default_action")) != "deny":
            raise ValueError("Task sandboxes require externally enforced default-deny network policy")
        role = "judge" if judge else "candidate"
        if state.runtime is None:
            raise RuntimeError("Trusted runtime must be prepared before task sandbox creation")
        if not self.config.dependency_proxy_url:
            raise ValueError("Configure dependency_proxy_url for reference dependency access")
        if state.dependency_access is None:
            state.dependency_access = DependencyAccess(self.config.dependency_proxy_url)
        certificate = await state.dependency_access.certificate()
        options["network_policy"], proxy_env, proxy_files = state.dependency_access.sandbox_settings(
            state.dependency_token, policy, certificate
        )
        mount = state.runtime.mount_options(state.session_id, role)
        options["volumes"] = list(options.get("volumes", [])) + mount["volumes"]
        provider = resolve_provider_config(self.config.sandbox_provider, self.server_client.global_config_dict)
        sandbox = AsyncSandbox(provider)
        if judge:
            state.judge_sandbox = sandbox
        else:
            state.sandbox = sandbox
        metadata = state.task_metadata
        memory = memory_mib(metadata["environment"].get("memory", "4G"))
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
                    env=proxy_env,
                    files=proxy_files,
                    provider_options=options,
                    metadata=state.runtime.metadata()
                    | dict(sandbox_config.get("metadata", {}))
                    | {
                        "benchmark": "swe-together",
                        "task_id": state.task.task_id,
                        "role": role,
                        "episode_id": state.request.episode_id.capture_key,
                        "environment_session_id": state.session_id,
                    },
                )
            )
        )
        state.creations[role] = creation
        # Provider creates can complete after transport cancellation. Retain the
        # task and shield it so close can wait for the actual owned handle.
        await asyncio.shield(creation)
        await install_trust_bundle(sandbox)
        if judge:
            installed = await sandbox.exec(
                "command -v python3 >/dev/null || ln -s /opt/gym-runtime/python/bin/python3 /usr/local/bin/python3",
                timeout_s=30,
            )
            if installed.return_code or installed.error_type:
                raise RuntimeError("Cannot prepare grading Python runtime")
        return sandbox

    async def _stop_owned_sandbox(self, state: Episode, role: str) -> None:
        sandbox = state.judge_sandbox if role == "judge" else state.sandbox
        if sandbox is None:
            return
        receipt = {
            "episode_id": state.request.episode_id.capture_key,
            "environment_session_id": state.session_id,
            "role": role,
            "cleanup_confirmed": False,
        }
        state.directory.mkdir(parents=True, exist_ok=True)
        try:
            async with asyncio.timeout(self.config.cleanup_timeout_seconds):
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
            state.cleanup_receipts[role + "_sandbox"] = receipt
            (state.directory / f"{role}-sandbox-close.json").write_text(json.dumps(receipt, indent=2))

    async def _prepare(self, state: Episode) -> None:
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
        await state.runtime.prepare(state.session_id)
        sandbox = await self._new_sandbox(state)
        await state.dependency_access.register(
            state.dependency_token,
            sandbox,
            workdir=state.task.workdir,
            task_dir=state.task_dir,
            artifact_dir=state.directory,
            ttl=min(
                86400,
                self.config.setup_timeout_seconds
                + self.config.interaction_timeout_seconds
                + self.config.grading_timeout_seconds
                + 4 * self.config.cleanup_timeout_seconds,
            ),
        )
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
            cookies={},
            temperature=self.config.simulator_temperature,
        )
        state.simulator = UserAgent(
            llm=state.simulator_model, original_user_messages=messages, session_analysis=analysis, max_messages=None
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

    async def _next_user_turn(self, state: Episode, activation: CandidateTurn) -> UserTurn:
        turn = activation.activation_id - 1
        if turn >= self.config.max_resumes:
            return UserTurn(activation_id=activation.activation_id, kind="stop", stop_reason="max_resumes")
        remaining = self._remaining_interaction_seconds(state)
        if remaining <= 0:
            return UserTurn(
                activation_id=activation.activation_id, kind="stop", stop_reason="session_budget_exhausted"
            )
        snapshot_deadline = asyncio.timeout(remaining)
        try:
            async with snapshot_deadline:
                diff = await state.snapshots.capture(state.sandbox, turn)
        except TimeoutError:
            if not snapshot_deadline.expired():
                raise
            return UserTurn(
                activation_id=activation.activation_id, kind="stop", stop_reason="session_budget_exhausted"
            )
        if self._remaining_interaction_seconds(state) <= 0:
            return UserTurn(
                activation_id=activation.activation_id, kind="stop", stop_reason="session_budget_exhausted"
            )
        activity, report = project_turn(activation.response, context_chars=self.config.user_context_chars)
        # Keep a proposed turn isolated until it can be accepted before the
        # deadline. The auxiliary client stays shared so canceled calls retain
        # their diagnostics, while late simulator history/counters are discarded.
        simulator = state.simulator.fork_turn()
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
                    elapsed_sec=monotonic() - state.interaction_started,
                    turn_duration_sec=activation.duration_seconds,
                    code_changes_diff=diff,
                )
        except BaseException as error:
            if isinstance(error, TimeoutError) and timer.expired():
                return self._expired_simulator_turn(state, activation, simulator)
            evidence = {
                "turn": turn + 1,
                "error": str(error) or type(error).__name__,
                "error_detail": error_evidence(error),
                "simulator_messages": simulator.last_messages_sent,
                "source_observation": activation.response.model_dump(mode="json"),
            }
            (state.directory / f"turn-{turn}-simulator.json").write_text(json.dumps(evidence, indent=2))
            (state.directory / "simulator-model-calls.json").write_text(
                json.dumps(state.simulator_model.calls, indent=2)
            )
            raise
        accepted_at = time()
        if timer.expired() or self._remaining_interaction_seconds(state) <= 0:
            return self._expired_simulator_turn(state, activation, simulator)
        state.simulator = simulator
        evidence = {
            "turn": turn + 1,
            "accepted_at_unix_seconds": accepted_at,
            "action": decision.action,
            "content": decision.content,
            "raw_response": decision.raw_response,
            "has_message": decision.has_message,
            "simulator_messages": state.simulator.last_messages_sent,
            "source_observation": activation.response.model_dump(mode="json"),
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
            return UserTurn(
                activation_id=activation.activation_id,
                kind="stop",
                stop_reason="consecutive_noops",
                metadata={"decision": decision.action},
            )
        return UserTurn(
            activation_id=activation.activation_id,
            kind="continue",
            synthetic=not decision.has_message,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": message}]
            ),
            metadata={"decision": decision.action, "original_cursor": state.simulator.original_cursor},
        )

    def _remaining_interaction_seconds(self, state: Episode) -> float:
        return state.interaction_deadline - monotonic() if state.interaction_deadline is not None else float("inf")

    def _expired_simulator_turn(self, state: Episode, activation: CandidateTurn, simulator: UserAgent) -> UserTurn:
        evidence = {
            "turn": activation.activation_id,
            "discarded_at_unix_seconds": time(),
            "stop_reason": "session_budget_exhausted",
            "discarded": True,
            "simulator_messages": simulator.last_messages_sent,
            "source_observation": activation.response.model_dump(mode="json"),
        }
        (state.directory / f"turn-{activation.activation_id - 1}-simulator.json").write_text(
            json.dumps(evidence, indent=2)
        )
        (state.directory / "simulator-model-calls.json").write_text(json.dumps(state.simulator_model.calls, indent=2))
        return UserTurn(
            activation_id=activation.activation_id,
            kind="stop",
            stop_reason="session_budget_exhausted",
            metadata={"simulator_turn_discarded": True},
        )

    async def _verify(
        self,
        state: Episode,
        initial: NeMoGymResponseCreateParamsNonStreaming,
        activations: list[CandidateTurn],
    ) -> dict:
        result = {
            "responses_create_params": initial,
            "response": episode_response(activations),
            "reward": 0.0,
            "judge_score": None,
            "task_id": state.task.task_id,
            "scoring_profile": self.config.scoring_profile,
            "protocol_profile": self.config.protocol_profile,
        }
        deadline = monotonic() + self.config.grading_timeout_seconds
        grading = asyncio.timeout(self.config.grading_timeout_seconds)
        try:
            async with grading:
                await state.snapshots.capture(state.sandbox, len(activations))
                patch = state.snapshots.final_patch
                if self.config.scoring_profile == "reference" and len(patch.strip()) < 100:
                    # A patch too short to judge is a measured zero, not a missing measurement.
                    result.update(
                        judge_score=0.0,
                        verdict={"judge_score": 0.0, "reason": "short_patch"},
                        evaluation_completed=True,
                    )
                else:
                    sandbox = await self._new_sandbox(state, judge=True)
                    result["submission"] = await replicate_submission(
                        state.sandbox,
                        sandbox,
                        roots=[state.task.workdir, *state.snapshots.baseline],
                        artifact_dir=state.directory / "submission",
                        storage=state.runtime,
                        session_id=state.session_id,
                    )
                    verdict = await run_judge(
                        client=self.server_client,
                        agent=self.config.judge_agent,
                        sandbox=sandbox,
                        provider=self.config.sandbox_provider,
                        task_dir=state.task_dir,
                        patch=patch,
                        workdir=state.task.workdir,
                        episode=state.request.episode_id,
                        task_id=state.request.task.task_id,
                        timeout=state.task_metadata["judge_timeout"],
                        cookies={},
                        artifact_dir=state.directory,
                        deadline=deadline,
                        cleanup_timeout=self.config.cleanup_timeout_seconds,
                        evidence=result,
                    )
                    result.update(
                        verdict=verdict,
                        judge_score=verdict["judge_score"],
                        reward=float(verdict["judge_score"] >= PASS_THRESHOLD),
                        evaluation_completed=True,
                    )
        except Exception as error:
            LOGGER.exception("SWE-Together judge measurement failed")
            result.update(
                mask_sample=True,
                evaluation_completed=False,
                failure_kind=JUDGE_FAILED,
                failure_reason=f"{type(error).__name__}: {error}",
                # Only this server's own grading deadline is a grading_timeout; a provider TimeoutError is not.
                judge_failure_cause=getattr(
                    error,
                    "cause",
                    "grading_timeout"
                    if grading.expired()
                    else "judge_transport_error"
                    if result.get("judge_transport_error")
                    else "judge_error",
                ),
                _ng_failure_class=JUDGE_FAILED,
                _ng_failure_judge_error=str(error),
            )
        auxiliary = AuxiliaryModel(self.server_client, self.config.interaction_model.name, cookies={}, temperature=0)
        try:
            async with asyncio.timeout(max(0, deadline - monotonic())):
                result.update(
                    await interaction_metrics(
                        auxiliary, task_dir=state.task_dir, messages=state.messages, artifact_dir=state.directory
                    )
                )
        except TimeoutError:
            result["interaction_metrics_error"] = "Grading deadline exhausted"
        (state.directory / "auxiliary-model-calls.json").write_text(
            json.dumps({"simulator": state.simulator_model.calls, "interaction": auxiliary.calls}, indent=2)
        )
        (state.directory / "simulator-state.json").write_text(
            json.dumps(
                {
                    "accepted_messages": state.messages,
                    "stats": state.simulator.get_stats(),
                    "history": state.simulator.history,
                },
                indent=2,
            )
        )
        (state.directory / "verification.json").write_text(SWETResult.model_validate(result).model_dump_json(indent=2))
        return result

    async def shutdown(self) -> None:
        if self._runtime:
            async with asyncio.timeout(self.config.cleanup_timeout_seconds):
                await self._runtime.close()


if __name__ == "__main__":
    SWETEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    from nemo_gym.server_runtime.server_utils import create_nemo_gym_fastapi_app

    app = create_nemo_gym_fastapi_app(SWETEnvironmentServer)
