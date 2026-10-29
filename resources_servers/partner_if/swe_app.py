# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scale-SWE lifecycle with visible-agent evidence and the shared IF grader.

An Environment Server seeds each rollout: this server starts the row's Scale-SWE image, runs the row's setup and
hands the sandbox to the agent (`SandboxAccess`). The row's own `pre_commands` configure the agent's harness, and its
grading contract carries the judge's copies of the actor's system prompt and tool-description additions. Scale-SWE's
native verdict is a separate diagnostic in `task_outcome`; it never supplies IF points.
"""

import asyncio
import shlex
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import ClassVar

from fastapi import FastAPI, HTTPException, Request
from pydantic import Field, JsonValue, PrivateAttr, ValidationError

from nemo_gym.base_resources_server import (
    ClosedResourcesSessionRecords,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ReverifyMode,
)
from nemo_gym.episode_types import EpisodeId
from nemo_gym.failure_kinds import SESSION_LOST
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, DirectSandboxConnection, SandboxAccess
from nemo_gym.server_utils import SESSION_ID_KEY
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.partner_if.app import (
    NoAnswer,
    PartnerIFGradingConfig,
    PartnerIFGradingMixin,
    VerifyResponse,
    check_finished,
)
from resources_servers.partner_if.evidence import EvidenceBundle
from resources_servers.partner_if.grading_contract import NO_ANSWER, GradingContract
from resources_servers.partner_if.metrics import PartnerIFMetricsMixin
from resources_servers.partner_if.native_evidence import visible_events
from resources_servers.partner_if.receipt import masked_fields
from resources_servers.partner_if.schema import VerifierMetadata
from resources_servers.partner_if.verifier_fixture import create_fixture_server
from resources_servers.scale_swe.app import (
    ScaleSWEInstanceRequest,
    ScaleSWEResourcesServer,
    ScaleSWEResourcesServerConfig,
    ScaleSWEVerifyRequest,
)
from resources_servers.scale_swe.verification import clean_commands
from resources_servers.swebench.anti_cheat import apply_anti_cheat_setup
from resources_servers.swebench.patch_capture import prepare_git_for_commits


class SWEIFConfig(PartnerIFGradingConfig, ScaleSWEResourcesServerConfig):
    # Verification needs the actor's live sandbox, so reverify cannot run it again.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(
            input=[], max_output_tokens=32768, temperature=0
        )
    )
    judge_max_concurrency: int = Field(default=2, gt=0)


class SWEIFVerifyRequest(ScaleSWEVerifyRequest):
    verifier_metadata: VerifierMetadata
    task_name: str | None = None


class SWEIFVerifyResponse(VerifyResponse):
    # Scale-SWE's own verdict.
    task_outcome: dict[str, JsonValue] | None = None
    task_name: str | None = None


def actor_contract(contract: GradingContract) -> GradingContract:
    """The judge grades the trajectory against the system prompt the actor ran under, so the contract carries it."""
    if contract.evidence_mode != "live_agent" or contract.actor_system_prompt is None:
        raise HTTPException(422, "SWE evidence requires a live-agent grading contract with the actor's system prompt")
    return contract


# A tool output longer than this keeps only its start and end. A long agent run's evidence must fit the judge's
# input, and the SWE rubrics grade the agent's messages and tool calls, which stay whole.
TOOL_OUTPUT_LIMIT = 500


def shorten_tool_output(text: str, limit: int = TOOL_OUTPUT_LIMIT) -> str:
    """The start and end of a tool output longer than `limit` characters, with a marker for what is left out."""
    if len(text) <= limit:
        return text
    keep = limit // 2
    return f"{text[:keep]}\n[… {len(text) - 2 * keep:,} characters omitted …]\n{text[-keep:]}"


def build_swe_evidence(
    *,
    context: list[dict[str, JsonValue]],
    system_prompt: str,
    output: list[dict[str, JsonValue]],
) -> EvidenceBundle:
    """Keep every public message and tool call in order, and the start and end of long tool outputs; never private
    reasoning. The answer is the agent's final message, which the trajectory also holds."""
    bundle = EvidenceBundle(context=context, instructions=system_prompt)
    try:
        events = visible_events(output)
    except ValueError as error:
        bundle.trajectory_error = str(error)
        return bundle
    if not events:
        raise NoAnswer(NO_ANSWER, "The trajectory has no visible events")
    bundle.trajectory = [
        event.model_copy(update={"text": shorten_tool_output(event.text)}) if event.kind == "tool_result" else event
        for event in events
    ]
    bundle.answer = next((event.text for event in reversed(bundle.trajectory) if event.kind == "assistant"), "")
    return bundle


async def _run(sandbox: AsyncSandbox, command: str, step: str) -> str:
    result = await sandbox.exec(command)
    if result.error_type is not None or result.return_code != 0:
        raise RuntimeError(
            f"{step} failed in the task sandbox (exit {result.return_code}, {result.error_type}): "
            f"{(result.stderr or '')[-2000:]}"
        )
    return result.stdout or ""


class SWEIFResourcesServer(PartnerIFGradingMixin, PartnerIFMetricsMixin, ScaleSWEResourcesServer):
    # Actor sandboxes live in worker memory, so the server runs one worker (see model_post_init) and no Ray.
    ray_enabled = False

    config: SWEIFConfig
    _seeds: dict[str, tuple[ResourcesSeedSessionRequest, SandboxAccess]] = PrivateAttr(default_factory=dict)
    # Seed, verify, close and retire hold the session's lock. A sandbox start awaits, and a close or a repeated seed
    # that ran meanwhile would leak its sandbox or start a second one.
    _session_locks: dict[str, asyncio.Lock] = PrivateAttr(default_factory=dict)
    _closed: ClosedResourcesSessionRecords[EpisodeId | None] = PrivateAttr()

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("Actor sandboxes live in the server's memory, so it requires num_workers=1")
        if self.config.is_verifying_golden_patch:
            raise ValueError("Golden patches are not agent instruction-following trajectories")
        self._closed = ClosedResourcesSessionRecords(self.config.session_close_retry_window_seconds)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def stop_sandboxes_at_shutdown(application: FastAPI):
            async with lifespan(application) as state:
                try:
                    yield state
                finally:
                    sandboxes = list(self._session_id_to_sandbox.values())
                    self._session_id_to_sandbox.clear()
                    await asyncio.gather(*(self._stop_sandbox(sandbox) for sandbox in sandboxes))

        app.router.lifespan_context = stop_sandboxes_at_shutdown
        return app

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        return self._session_locks.setdefault(session_id, asyncio.Lock())

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        session_id = body.resources_session_id
        try:
            instance = ScaleSWEInstanceRequest.model_validate(body.task_data)
            contract = VerifierMetadata.model_validate(body.task_data.get("verifier_metadata")).partner_if
        except ValidationError as error:
            # The errors name the fields without their values, which can hold the task's private content.
            issues = error.errors(include_url=False, include_context=False, include_input=False)
            raise HTTPException(422, f"Invalid SWE task data: {issues}") from error
        # A contract the verifier would refuse fails here, before the episode spends a sandbox and an agent run.
        actor_contract(contract)
        async with self._session_lock(session_id):
            if session_id in self._closed:
                raise HTTPException(409, "Session is closed")
            seeded = self._seeds.get(session_id)
            if seeded is None:
                seeded = body, await self._start_instance(session_id, instance)
                self._seeds[session_id] = seeded
            elif seeded[0] != body:
                raise HTTPException(409, "Session ID belongs to another episode/task")
        request.session[SESSION_ID_KEY] = session_id
        return ResourcesSeedSessionResponse(resources_session_id=session_id, sandbox_access=seeded[1])

    async def _start_instance(self, session_id: str, instance: ScaleSWEInstanceRequest) -> SandboxAccess:
        """Scale-SWE's seed, except that a failed step stops the sandbox and fails the seed. The row's
        `pre_commands` also set up the agent's harness, so a sandbox where they failed does not hold the task its
        contract grades."""
        sandbox = await self._create_sandbox(instance)
        workdir = shlex.quote(instance.workdir)
        try:
            if pre := clean_commands(instance.pre_commands):
                await _run(sandbox, f"cd {workdir} && {pre}", "The task's pre_commands")
            await self._ensure_git_repo(sandbox, instance.workdir)
            if self.config.apply_anti_cheating:
                await apply_anti_cheat_setup(sandbox, instance.workdir, instance.instance_id, "scale_swe")
            # The anti-cheat scrub leaves no committer identity, so the agent's `git commit` would fail.
            await prepare_git_for_commits(sandbox, instance.workdir, "scale_swe")
            base_commit = (await _run(sandbox, f"git -C {workdir} rev-parse HEAD", "Reading the base commit")).strip()
            pristine_untracked = await self._pristine_untracked_files(sandbox, instance.workdir)
            access = SandboxAccess(
                connection=DirectSandboxConnection(
                    provider_config_ref=self.config.sandbox_provider, descriptor=await sandbox.serialize()
                ),
                workdir=instance.workdir,
            )
        except BaseException:
            await self._stop_sandbox(sandbox)
            raise
        self._session_id_to_base_commit[session_id] = base_commit
        self._session_id_to_pristine_untracked[session_id] = pristine_untracked
        self._session_id_to_sandbox[session_id] = sandbox
        return access

    def _forget(self, session_id: str) -> None:
        self._seeds.pop(session_id, None)
        self._session_id_to_sandbox.pop(session_id, None)
        self._session_id_to_base_commit.pop(session_id, None)
        self._session_id_to_pristine_untracked.pop(session_id, None)

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        session_id = body.resources_session_id
        async with self._session_lock(session_id):
            closed_episode = self._closed.get(session_id)
            if closed_episode is not None and closed_episode != body.episode_id:
                raise HTTPException(409, "Session belongs to another episode")
            seeded = self._seeds.get(session_id)
            if seeded is not None and seeded[0].episode_id != body.episode_id:
                raise HTTPException(409, "Session belongs to another episode")
            if (sandbox := self._session_id_to_sandbox.get(session_id)) is not None:
                # A sandbox that fails to stop raises and stays, so a retried close stops it again.
                await sandbox.stop()
            self._forget(session_id)
            self._closed.remember(session_id, body.episode_id)
        # Later requests for a closed session are refused, so it needs no lock; any waiting on this one see the close.
        self._session_locks.pop(session_id, None)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def retire_session_state(self, session_id: str) -> None:
        async with self._session_lock(session_id):
            seeded = self._seeds.get(session_id)
            sandbox = self._session_id_to_sandbox.get(session_id)
            self._forget(session_id)
            self._closed.remember(session_id, seeded[0].episode_id if seeded is not None else None)
            await self._stop_sandbox(sandbox)
        self._session_locks.pop(session_id, None)

    async def verify(self, request: Request, body: SWEIFVerifyRequest) -> SWEIFVerifyResponse:
        contract = actor_contract(body.verifier_metadata.partner_if)
        session_id = request.session[SESSION_ID_KEY]
        async with self._session_lock(session_id):
            if session_id in self._closed:
                # Its sandbox is gone, so neither its patch nor the native tests can run: the sample is masked.
                return SWEIFVerifyResponse(
                    **body.model_dump(), **masked_fields(contract, SESSION_LOST, "The session closed before verify")
                )
            # Scale-SWE's verify extracts the actor's patch, stops its sandbox and runs the native tests.
            outcome = await super().verify(request, body)
        graded = await self.grade_if(body)
        return graded.model_copy(
            update={"task_outcome": outcome.model_dump(mode="json", exclude={"response", "responses_create_params"})}
        )

    async def grade_if(self, body: SWEIFVerifyRequest) -> SWEIFVerifyResponse:
        """Grade instruction following from the actor's visible trajectory, without Scale-SWE's native verdict."""
        contract = actor_contract(body.verifier_metadata.partner_if)

        def evidence() -> EvidenceBundle:
            check_finished(body.response)
            bundle = build_swe_evidence(
                context=body.responses_create_params.model_dump(mode="json", exclude_none=True)["input"],
                system_prompt=contract.actor_system_prompt,
                output=[item.model_dump(mode="json", exclude_none=True) for item in body.response.output],
            )
            bundle.tools = [
                {"name": name, "description_addition": text}
                for name, text in contract.tool_description_additions.items()
            ]
            return bundle

        return SWEIFVerifyResponse.model_validate(body.model_dump() | await self.grade_response(contract, evidence))


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=partial(
        create_fixture_server,
        SWEIFResourcesServer,
        SWEIFConfig,
        entrypoint="swe_app.py",
        sandbox_provider="sandbox",
        sandbox_config={},
    ),
    request_model=SWEIFVerifyRequest,
    cases_path=Path(__file__).parent / "tests/swe_verifier_cases.jsonl",
    invoke=SWEIFResourcesServer.grade_if,
)


if __name__ == "__main__":
    SWEIFResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SWEIFResourcesServer.run_webserver()
